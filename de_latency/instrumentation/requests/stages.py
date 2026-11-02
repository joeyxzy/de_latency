"""Small synchronous/asynchronous request-stage wrappers with shared ID lookup."""
from functools import wraps
import time

from ...runtime.request_context import remember_request_name
from ..task_events import emit_stage_duration
from .state import argument, mark, patched


def instrument(cls, method, stage, id_argument, request_object=False, asynchronous=False, client_class=False):
    original = getattr(cls, method, None)
    if original is None or patched(original):
        return

    def identity(args, kwargs):
        value = argument(original, id_argument, args, kwargs)
        return getattr(value, "request_id", None) if request_object else value

    def emit(self, start, rid):
        extra = {"client_class": self.__class__.__name__} if client_class else None
        emit_stage_duration(stage, start, time.time_ns(), rid, remember_request_name(rid, None), extra)

    if asynchronous:
        @wraps(original)
        async def wrapper(self, *args, **kwargs):
            rid, start = identity(args, kwargs), time.time_ns()
            try:
                return await original(self, *args, **kwargs)
            finally:
                emit(self, start, rid)
    else:
        @wraps(original)
        def wrapper(self, *args, **kwargs):
            rid, start = identity(args, kwargs), time.time_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                emit(self, start, rid)
    setattr(cls, method, mark(wrapper))


def install(llm, processor, output_processor, clients):
    instrument(processor, "process_inputs", "input_processor.process_inputs", "request_id")
    instrument(output_processor, "add_request", "output_processor.add_request", "request", request_object=True)
    instrument(llm, "add_request", "async_llm.add_request", "request_id", asynchronous=True)
    instrument(llm, "_add_request", "async_llm._add_request", "request", request_object=True, asynchronous=True)
    for name in ("AsyncMPClient", "DPAsyncMPClient", "DPLBAsyncMPClient", "InprocClient"):
        cls = getattr(clients, name, None)
        if cls is not None:
            instrument(cls, "add_request_async", "engine_core.add_request_async", "request",
                       request_object=True, asynchronous=True, client_class=True)
