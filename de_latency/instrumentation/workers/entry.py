"""Worker readiness and monotonic execution windows for eBPF attribution."""
import os
import threading
import time

from ...runtime.control import is_enabled
from ...runtime.transport import TraceSender
from ..execution_context import push, pop
from ..registry import register_hook, apply_method_patch
from .context import ProcessState, build


def emit_ready(stage):
    if not is_enabled():
        return
    pid = os.getpid()
    with ProcessState.lock:
        if pid in ProcessState.ready_pids:
            return
        ProcessState.ready_pids.add(pid)
    TraceSender.emit("worker_process_ready", {"pid": pid, "tid": threading.get_native_id(),
                                             "stage": stage, "timestamp_ns": time.time_ns()})


@register_hook("vllm.v1.worker.gpu_worker")
def patch_worker_base(module):
    cls = module.Worker

    def execute(original):
        def wrapper(self, scheduler_output, *args, **kwargs):
            emit_ready("execute_model")
            start = time.time_ns()
            context = build(self, scheduler_output)
            context["runner"] = getattr(self, "model_runner", None)
            payload = {key: context[key] for key in ("pid", "tid", "req_ids", "ranks", "dispatch_key")}
            if context["req_ids"]:
                TraceSender.emit("worker_preprocess_start", {**payload, "event": "preprocess_start", "timestamp_ns": start})
            mono = time.monotonic_ns()
            push(context)
            try:
                return original(self, scheduler_output, *args, **kwargs)
            finally:
                pop()
                end = time.monotonic_ns()
                if context["req_ids"]:
                    TraceSender.emit("worker_model_execute_span", {**payload, "start_ns": mono,
                        "end_ns": end, "duration_ns": end - mono})
        return wrapper

    def ready(stage):
        def factory(original):
            def wrapper(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                emit_ready(stage)
                return result
            return wrapper
        return factory

    apply_method_patch(cls, "execute_model", execute)
    for name in ("__init__", "init_device", "load_model"):
        apply_method_patch(cls, name, ready(name))
