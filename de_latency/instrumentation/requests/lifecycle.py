"""Tracing boundaries inside the vLLM 0.20 generate-consume lifecycle."""
import asyncio
from functools import wraps
import logging
import os
import threading
import time

from ...runtime.control import is_enabled
from ...runtime.request_context import bind_request_context, reset_request_context, remember_request_name, forget_request
from ...runtime.transport import send_event
from ..task_events import emit_stage_duration, emit_coroutine_exec_slice, emit_coroutine_sched_latency
from .state import argument, mark, tasks

logger = logging.getLogger("vllm.v1.engine.async_llm")


def install(cls):
    from vllm.v1.engine.async_llm import STREAM_FINISHED, InputStreamError
    from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
    original = cls.generate

    @wraps(original)
    async def generate(self, *args, **kwargs):
        if not is_enabled():
            async for output in original(self, *args, **kwargs):
                yield output
            return
        wrapper_start = time.time_ns()
        values = {key: argument(original, key, args, kwargs) for key in (
            "prompt", "sampling_params", "request_id", "lora_request", "trace_headers", "priority",
            "data_parallel_rank", "tokenization_kwargs", "prompt_text", "reasoning_ended", "reasoning_parser_kwargs")}
        rid = values["request_id"]
        name = remember_request_name(rid, argument(original, "request_name", args, kwargs) or rid)
        pid, tid = os.getpid(), threading.get_native_id()
        tasks.announce_role("asyncio_eventloop", {"component": "async_llm.generate"})
        task = asyncio.current_task()
        task_id, task_name = (id(task), task.get_name()) if task else (None, None)
        coroutine_id = task_id or id((rid, time.time_ns()))
        begin, timestamp = time.monotonic(), time.time_ns()
        tasks.register(task, rid, name, "generate_task")
        emit_stage_duration("async_llm.generate_wrapper_setup", wrapper_start, timestamp, rid, name)
        send_event("monkey_patch", "coroutine_start", {"request_id": rid, "request_name": name,
            "pid": pid, "tid": tid, "coroutine_id": coroutine_id, "timestamp_ns": timestamp})
        queue = None
        try:
            tokens = bind_request_context(rid, name)
            try:
                queue = await self.add_request(rid, values["prompt"], values["sampling_params"],
                    **{key: value for key, value in values.items() if key not in ("request_id", "prompt", "sampling_params")})
                internal = getattr(queue, "request_id", rid)
                if internal and str(internal) != str(rid):
                    remember_request_name(internal, name)
                    emit_stage_duration("async_llm.generate_wrapper_setup", wrapper_start, timestamp,
                                        internal, name, {"external_request_id": rid})
            finally:
                reset_request_context(tokens)
            finished = False
            while not finished:
                current = asyncio.current_task()
                current_id, current_name = (id(current), current.get_name()) if current else (task_id, task_name)
                consume_id = getattr(queue, "request_id", rid)
                consume_name = remember_request_name(consume_id, name)
                start = time.time_ns()
                ready = getattr(queue, "_de_latency_ready_since_ns", None)
                output = queue.get_nowait()
                if output is None:
                    output = await queue.get()
                    start = time.time_ns()
                    ready = getattr(queue, "_de_latency_ready_since_ns", None)
                if ready is not None and start >= ready:
                    emit_coroutine_sched_latency(ready, start, start - ready, consume_id,
                        consume_name, current_id, current_name, "generate_consume")
                queue._de_latency_ready_since_ns = None
                finished = getattr(output, "finished", False)
                emit_coroutine_exec_slice(start, time.time_ns(), consume_id, consume_name,
                                          current_id, current_name, "generate_consume")
                if output is not STREAM_FINISHED:
                    yield output
        except (asyncio.CancelledError, GeneratorExit):
            if queue is not None:
                await self.abort(queue.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", rid)
            raise
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", rid)
            raise
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", rid)
            raise
        except InputStreamError as exc:
            if queue is not None:
                await self.abort(queue.request_id, internal=True)
            raise exc.cause from exc
        except Exception as exc:
            if queue is not None:
                await self.abort(queue.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", rid)
            raise EngineGenerateError() from exc
        finally:
            send_event("monkey_patch", "coroutine_end", {"request_id": rid, "request_name": name,
                "pid": pid, "tid": threading.get_native_id(), "coroutine_id": coroutine_id,
                "duration_ms": (time.monotonic() - begin) * 1000, "timestamp_ns": time.time_ns()})
            if queue is not None:
                queue.close()
            forget_request(rid)
            if queue is not None:
                forget_request(getattr(queue, "request_id", None))
    cls.generate = mark(generate)
