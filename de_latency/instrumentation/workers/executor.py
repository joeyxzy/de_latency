"""Executor dispatch submission and token-sampling spans."""
import inspect
import os
import threading
import time

from ...runtime.transport import TraceSender
from ..registry import register_hook, apply_method_patch
from ..scheduler_metadata import request_ids, dispatch_key, step_maps, safe_int
from .context import ProcessState


def execute(original):
    if getattr(original, "_de_latency_scheduler_out_rpc_patch", False):
        return original
    signature = inspect.signature(original)

    def wrapper(self, *args, **kwargs):
        try:
            output = signature.bind(self, *args, **kwargs).arguments.get("scheduler_output")
        except TypeError:
            output = kwargs.get("scheduler_output")
        if output is not None:
            ids = request_ids(output)
            if ids:
                computed, scheduled = step_maps(output)
                TraceSender.emit("req_scheduler_out_rpc", {"req_ids": ids, "dispatch_key": dispatch_key(output),
                    "event": "execute_start", "batch_size": len(ids),
                    "total_scheduled_tokens": safe_int(output.total_num_scheduled_tokens, 0),
                    "num_computed_by_req": computed, "scheduled_tokens_by_req": scheduled,
                    "timestamp_ns": time.time_ns(), "pid": os.getpid(), "tid": threading.get_native_id()})
        return original(self, *args, **kwargs)
    wrapper._de_latency_scheduler_out_rpc_patch = True
    return wrapper


def sample(original):
    if getattr(original, "_de_latency_sample_tokens_patch", False):
        return original
    def wrapper(self, *args, **kwargs):
        key = ProcessState.last_dispatch_key
        if key:
            TraceSender.emit("scheduler_sample_tokens", {"timestamp_ns": time.time_ns(), "dispatch_key": key})
        return original(self, *args, **kwargs)
    wrapper._de_latency_sample_tokens_patch = True
    return wrapper


def patch_classes(module, names, sample_tokens=True):
    for name in names:
        cls = getattr(module, name, None)
        if cls is not None:
            apply_method_patch(cls, "execute_model", execute)
            if sample_tokens:
                apply_method_patch(cls, "sample_tokens", sample)


@register_hook("vllm.v1.executor.abstract")
def patch_abstract(module):
    patch_classes(module, ("UniProcExecutor", "Executor"), sample_tokens=False)


@register_hook("vllm.v1.executor.multiproc_executor")
def patch_multiproc(module):
    patch_classes(module, ("MultiprocExecutor",))


@register_hook("vllm.v1.executor.uniproc_executor")
def patch_uniproc(module):
    patch_classes(module, ("UniProcExecutor", "ExecutorWithExternalLauncher"))


@register_hook("vllm.v1.executor.ray_executor")
def patch_ray(module):
    patch_classes(module, ("RayDistributedExecutor",))
