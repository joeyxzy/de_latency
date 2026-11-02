"""EngineCore input processing and main-loop timing."""
import os
import threading
import time

from ...runtime.transport import TraceSender
from ..registry import register_hook, apply_method_patch
from .context import ProcessState


def emit_stage(stage, start, end, request_id=None, extra=None):
    if start is None or end is None or end < start:
        return
    payload = {"request_id": request_id, "request_name": request_id, "stage": stage,
               "start_ns": int(start), "end_ns": int(end), "duration_ns": int(end) - int(start),
               "timestamp_ns": int(end), "pid": os.getpid(), "tid": threading.get_native_id()}
    payload.update(extra or {})
    TraceSender.emit("req_generate_stage", payload)


@register_hook("vllm.v1.engine.core")
def patch_v1_engine_core(module):
    core = module.EngineCore

    def preprocess(original):
        def wrapper(self, request):
            start = time.time_ns()
            rid = getattr(request, "request_id", None)
            try:
                return original(self, request)
            finally:
                end = time.time_ns()
                emit_stage("engine_core.preprocess_add_request", start, end, rid)
                if rid:
                    with ProcessState.lock:
                        ProcessState.preprocess_done[rid] = end
        return wrapper

    def add(original):
        def wrapper(self, request, request_wave=0):
            start = time.time_ns()
            try:
                return original(self, request, request_wave)
            finally:
                emit_stage("engine_core.add_request", start, time.time_ns(), getattr(request, "request_id", None))
        return wrapper

    def handle(original):
        def wrapper(self, request_type, request):
            start, rid = None, None
            if request_type == module.EngineCoreRequestType.ADD:
                start = time.time_ns()
                rid = getattr(request[0], "request_id", None)
                with ProcessState.lock:
                    done = ProcessState.preprocess_done.pop(rid, None)
                if done is not None and start >= done:
                    emit_stage("engine_core.input_queue_wait_after_preprocess", done, start, rid)
            try:
                return original(self, request_type, request)
            finally:
                if start is not None:
                    emit_stage("engine_core.handle_client_add", start, time.time_ns(), rid)
        return wrapper

    def loop_phase(name):
        def factory(original):
            def wrapper(self, *args, **kwargs):
                start = time.time_ns()
                try:
                    return original(self, *args, **kwargs)
                finally:
                    end = time.time_ns()
                    TraceSender.emit("enginecore_mainloop_span", {"phase": name, "start_ns": start,
                        "end_ns": end, "duration_ns": end - start, "timestamp_ns": end,
                        "pid": os.getpid(), "tid": threading.get_native_id()})
            return wrapper
        return factory

    apply_method_patch(core, "preprocess_add_request", preprocess)
    apply_method_patch(core, "add_request", add)
    apply_method_patch(module.EngineCoreProc, "_handle_client_request", handle)
    for name in ("_process_input_queue", "_process_engine_step"):
        apply_method_patch(module.EngineCoreProc, name, loop_phase(name))
