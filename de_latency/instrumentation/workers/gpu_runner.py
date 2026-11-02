"""Shared GPU runner wrappers; the two 0.20 runners differ only in context recovery."""
import inspect
import os
import threading
import time

from ...runtime.transport import TraceSender
from ..execution_context import current, push, pop
from ..markers import mark
from ..registry import register_hook, apply_method_patch
from .context import build, refresh, runner_output, fields

PHASES = {name: name for name in ("Preprocess", "Forward", "Postprocess", "Bookkeep", "Draft", "EPLB")}
PHASES.update({"gpu_model_runner: " + key: value for key, value in (
    ("preprocess", "Preprocess"), ("forward", "Forward"), ("postprocess", "Postprocess"),
    ("bookkeep", "Bookkeep"), ("draft", "Draft"), ("eplb", "EPLB"), ("ModelRunnerOutput", "ModelRunnerOutput"))})


def gpu_mark(role, key):
    try:
        import torch
        mark(torch.cuda.current_stream(), role, key)
    except Exception:
        pass  # Markers are optional when libtracer/CUDA is unavailable.


class PhaseScope:
    def __init__(self, scope, phase, raw):
        self.scope, self.phase, self.raw = scope, phase, raw

    def __enter__(self):
        self.start = time.time_ns()
        return self.scope.__enter__()

    def __exit__(self, exc_type, exc, tb):
        try:
            return self.scope.__exit__(exc_type, exc, tb)
        finally:
            context = current()
            if context is not None:
                end = time.time_ns()
                refresh(context)
                TraceSender.emit("gpu_phase_span", {**fields(context), "phase": self.phase,
                    "raw_phase": self.raw, "method_name": context.get("method_name"),
                    "start_ns": self.start, "end_ns": end, "timestamp_ns": end,
                    "scheduled_tokens_by_req": dict(context.get("scheduled_tokens_by_req", {}))})


def install_runner(module, version):
    cls = module.GPUModelRunner
    class_path = module.__name__ + ".GPUModelRunner"
    if version == 1 and not getattr(module, "_de_latency_phase_scope_patch_installed", False):
        original_scope = module.record_function_or_nullcontext

        def scope(name):
            wrapped = original_scope(name)
            phase = PHASES.get(name)
            return wrapped if phase is None else PhaseScope(wrapped, phase, name)

        module.record_function_or_nullcontext = scope
        module._de_latency_phase_scope_patch_installed = True

    def execute(original):
        signature = inspect.signature(original)

        def wrapper(self, *args, **kwargs):
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                output = bound.arguments.get("scheduler_output")
            except TypeError:
                output = None
            start = time.time_ns()
            context = build(self, output, original.__name__)
            self._de_latency_runner_context = dict(context)
            push(context)
            gpu_mark("exec_start", context.get("dispatch_key"))
            try:
                return original(self, *args, **kwargs)
            finally:
                end = time.time_ns()
                gpu_mark("exec_end", context.get("dispatch_key"))
                context = refresh(pop() or context)
                TraceSender.emit("gpu_execute_model", {**fields(context),
                    "hooked_method": original.__name__, "class_path": class_path,
                    "input_batch_req_ids": list(context.get("input_batch_req_ids", [])),
                    "start_ns": start, "end_ns": end,
                    "num_computed_by_req": dict(context.get("num_computed_by_req", {})),
                    "scheduled_tokens_by_req": dict(context.get("scheduled_tokens_by_req", {}))})
        return wrapper

    def sample(original):
        def wrapper(self, *args, **kwargs):
            start, mono = time.time_ns(), time.monotonic_ns()
            saved = getattr(self, "_de_latency_runner_context", {})
            if version == 2:
                context = {**saved, "runner": self, "pid": os.getpid(), "tid": threading.get_native_id(),
                           "input_type": "unknown", "method_name": original.__name__}
            else:
                output = runner_output(self)
                context = build(self, output, original.__name__)
                if output is None and saved.get("dispatch_key"):
                    context["dispatch_key"] = saved["dispatch_key"]
            push(context)
            gpu_mark("sample_start", context.get("dispatch_key"))
            try:
                return original(self, *args, **kwargs)
            finally:
                gpu_mark("sample_end", context.get("dispatch_key"))
                context = pop() or context
                if version == 1:
                    refresh(context)
                end, end_mono = time.time_ns(), time.monotonic_ns()
                if context.get("req_ids"):
                    payload = {**fields(context), "scheduled_tokens_by_req": dict(context.get("scheduled_tokens_by_req", {}))}
                    TraceSender.emit("gpu_phase_span", {**payload, "phase": "GPUModelRunner sample_tokens",
                        "method_name": original.__name__, "start_ns": start, "end_ns": end})
                    TraceSender.emit("worker_model_execute_span", {**payload, "hooked_method": original.__name__,
                        "class_path": class_path, "start_ns": mono, "end_ns": end_mono,
                        "duration_ns": end_mono - mono})
        return wrapper

    apply_method_patch(cls, "execute_model", execute)
    apply_method_patch(cls, "sample_tokens", sample)


@register_hook("vllm.v1.worker.gpu_model_runner")
def patch_gpu_model_runner(module):
    install_runner(module, version=1)


@register_hook("vllm.v1.worker.gpu.model_runner")
def patch_v2_gpu_model_runner(module):
    install_runner(module, version=2)
