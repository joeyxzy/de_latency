"""Task identity/context registry shared by independent API hook installers."""
import asyncio
import inspect
import os
import threading
import time

from ...runtime.control import is_enabled
from ...runtime.request_context import get_request_ctx_from_context
from ...runtime.transport import send_event


class TaskRegistry:
    def __init__(self):
        self.lock = threading.RLock()
        self.contexts = {}
        self.roles = set()

    def register(self, task, request_id=None, request_name=None, task_kind="request", extra=None):
        if task is None:
            return None
        info = {"request_id": request_id, "request_name": request_name or request_id,
                "task_name": task.get_name(), "task_kind": task_kind}
        info.update(extra or {})
        with self.lock:
            self.contexts[id(task)] = info
        if not getattr(task, "_de_latency_ctx_cleanup_installed", False):
            task.add_done_callback(self.unregister)
            task._de_latency_ctx_cleanup_installed = True
        return task

    def unregister(self, task):
        if task is not None:
            with self.lock:
                self.contexts.pop(id(task), None)

    def lookup(self, task_id):
        with self.lock:
            return self.contexts.get(task_id)

    def resolve(self, callback, context):
        task = getattr(callback, "__self__", None)
        if not isinstance(task, asyncio.Task) and not (task is not None and task.__class__.__name__.endswith("Task")):
            task = None
        task_id = id(task) if task is not None else None
        rid, name = get_request_ctx_from_context(context)
        if not rid:
            current_id, current_name = get_request_ctx_from_context(None)
            rid, name = rid or current_id, name or current_name
        info = self.lookup(task_id) if task_id else None
        if info:
            rid, name = rid or info.get("request_id"), name or info.get("request_name")
        return task_id, rid, name

    def announce_role(self, role, extra=None):
        if not is_enabled():
            return
        pid, tid = os.getpid(), threading.get_native_id()
        with self.lock:
            if (pid, tid, role) in self.roles:
                return
            self.roles.add((pid, tid, role))
        send_event("monkey_patch", "thread_role", {"role": role, "pid": pid, "tid": tid,
            "thread_ident": threading.get_ident(), "mono_timestamp_ns": time.monotonic_ns(),
            "timestamp_ns": time.time_ns(), **(extra or {})})


tasks = TaskRegistry()


def argument(func, name, args, kwargs):
    if name in kwargs:
        return kwargs[name]
    try:
        signature = inspect.signature(func)
        parameters = list(signature.parameters)
        candidates = [(None, *args), args] if parameters and parameters[0] in ("self", "cls") else [args]
        for values in candidates:
            try:
                bound = signature.bind_partial(*values, **kwargs)
                bound.apply_defaults()
                if name in bound.arguments:
                    return bound.arguments[name]
            except TypeError:
                continue
    except (TypeError, ValueError):
        pass
    return None


def mark(func):
    func._de_latency_req_stage_wrapped = True
    return func


def patched(func):
    return getattr(func, "_de_latency_req_stage_wrapped", False)
