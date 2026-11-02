"""asyncio/uvloop callback timing, independent of vLLM method replacement."""
import asyncio
from functools import wraps
import logging
import os
import time

from ...runtime.control import is_enabled
from ...runtime.transport import send_event
from ..task_events import emit_coroutine_exec_slice, emit_coroutine_sched_latency
from .state import tasks

logger = logging.getLogger(__name__)
debug = os.getenv("TRACE_PATCH_DEBUG", "0") == "1"


def scheduled(callback, ready, task_id, request_id, request_name):
    def wrapped(*args):
        tasks.announce_role("asyncio_eventloop")
        info = tasks.lookup(task_id) if task_id is not None else None
        if not info and request_id:
            info = {"request_id": request_id, "request_name": request_name or request_id, "task_name": None}
        start = time.time_ns()
        if info and start >= ready:
            if info.get("task_kind") in ("output_handler", "output_socket") and task_id is not None:
                with tasks.lock:
                    current = tasks.contexts.get(task_id)
                    if current is not None:
                        current.setdefault("pending_sched_events", []).append(
                            {"ready_ts_ns": ready, "run_ts_ns": start, "queue_ns": start - ready})
            else:
                emit_coroutine_sched_latency(ready, start, start - ready, info.get("request_id"),
                    info.get("request_name"), task_id, info.get("task_name"), info.get("task_kind") or "request")
        try:
            return callback(*args)
        finally:
            end = time.time_ns()
            if info and end > start and info.get("task_kind") not in ("output_handler", "output_socket", "generate_task"):
                emit_coroutine_exec_slice(start, end, info.get("request_id"), info.get("request_name"),
                    task_id, info.get("task_name"), info.get("task_kind") or "request")
    wrapped._de_latency_sched_wrapped = True
    return wrapped


def patch_loop(cls):
    if getattr(cls, "_de_latency_sched_patch_installed", False):
        return False

    def factory(original):
        @wraps(original)
        def wrapper(self, callback, *args, context=None):
            if is_enabled():
                task_id, rid, name = tasks.resolve(callback, context)
                if (task_id is not None or rid) and not getattr(callback, "_de_latency_sched_wrapped", False):
                    callback = scheduled(callback, time.time_ns(), task_id, rid, name)
            return original(self, callback, *args, context=context)
        return wrapper

    for name in ("call_soon", "call_soon_threadsafe"):
        original = getattr(cls, name, None)
        if original is not None:
            setattr(cls, name, factory(original))
    cls._de_latency_sched_patch_installed = True
    return True


def install():
    if getattr(asyncio, "_de_latency_sched_patch_installed", False):
        return
    installed = patch_loop(asyncio.BaseEventLoop)
    try:
        import uvloop
        installed = patch_loop(uvloop.Loop) or installed
    except (ImportError, TypeError) as exc:
        if debug:
            logger.info("uvloop callback timing unavailable: %s", exc)
    if installed:
        asyncio._de_latency_sched_patch_installed = True
        send_event("monkey_patch", "coroutine_sched_patch_status", {"pid": os.getpid(),
            "debug_enabled": debug, "status": "installed", "timestamp_ns": time.time_ns()})
