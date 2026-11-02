"""0.20 output-loop installation, collector readiness and per-slice execution."""
import asyncio
from functools import wraps
import time
import weakref

from ...runtime.request_context import get_request_ctx_from_context, remember_request_name
from ..output_loop import run_output_handler
from ..task_events import emit_stage_duration, emit_output_handler_exec_slice, emit_output_socket_exec_slice
from .rounds import begin_round, end_round, begin_slice, register_output_task
from .state import argument, mark, patched


def install(async_llm, processor, collector, clients):
    cls = async_llm.AsyncLLM
    original = cls._run_output_handler
    if not patched(original):
        @wraps(original)
        def run(self, *args, **kwargs):
            rid, name = get_request_ctx_from_context(None)
            start = time.time_ns()
            try:
                if self.output_handler is not None:
                    return None
                self._logger_ref = [self.logger_manager]
                self.output_handler = asyncio.create_task(run_output_handler(
                    self.engine_core, self.output_processor, self.log_stats, self._logger_ref,
                    self.renderer, async_llm.envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE,
                    async_llm.IterationStats, async_llm.logger, begin_round, end_round))
                register_output_task(self.output_handler, "output_handler", "output_handler")
            finally:
                emit_stage_duration("async_llm._run_output_handler", start, time.time_ns(), rid, name)
        cls._run_output_handler = mark(run)

    original_process = processor.process_outputs
    if not patched(original_process):
        @wraps(original_process)
        def process(self, *args, **kwargs):
            ids, sequence = begin_slice(argument(original_process, "engine_core_outputs", args, kwargs), "output_handler")
            start = time.time_ns()
            try:
                return original_process(self, *args, **kwargs)
            finally:
                end = time.time_ns()
                if ids and end > start:
                    task = asyncio.current_task()
                    for rid in ids:
                        remember_request_name(rid, None)
                    emit_output_handler_exec_slice(start, end, ids, id(task) if task else None,
                                                    task.get_name() if task else None, sequence)
        processor.process_outputs = mark(process)

    original_put = collector.put
    if not patched(original_put):
        @wraps(original_put)
        def put(self, *args, **kwargs):
            empty, ready = getattr(self, "output", None) is None, time.time_ns()
            try:
                return original_put(self, *args, **kwargs)
            finally:
                if empty and getattr(self, "output", None) is not None:
                    self._de_latency_ready_since_ns = ready
        collector.put = mark(put)
    for name in ("AsyncMPClient", "DPAsyncMPClient", "DPLBAsyncMPClient"):
        client = getattr(clients, name, None)
        original = getattr(client, "_ensure_output_queue_task", None)
        if original is not None and not patched(original):
            client._ensure_output_queue_task = mark(socket_wrapper(original, clients))


def socket_wrapper(original, clients):
    from vllm.v1.engine.exceptions import EngineDeadError

    @wraps(original)
    def ensure(self, *args, **kwargs):
        resources = self.resources
        if resources.output_queue_task is not None:
            return original(self, *args, **kwargs)
        decoder, utility, queue = self.decoder, self.utility_results, self.outputs_queue
        handler = getattr(self.__class__, "process_engine_outputs", None)
        notification = getattr(self.__class__, "eep_process_engine_core_notification", None)
        notification_id = getattr(clients, "EEP_NOTIFICATION_CALL_ID", None)
        ref = weakref.ref(self) if handler or notification else None
        socket = resources.output_socket
        assert socket is not None

        async def receive():
            try:
                while True:
                    frames = await socket.recv_multipart(copy=False)
                    start = time.time_ns()
                    resources.validate_alive(frames)
                    outputs = decoder.decode(frames)
                    if outputs.utility_output:
                        if notification_id is not None and outputs.utility_output.call_id == notification_id and notification:
                            owner = ref()
                            if not owner:
                                return
                            if outputs.utility_output.result is not None:
                                asyncio.create_task(notification(owner, outputs.utility_output.result.result))
                        else:
                            clients._process_utility_output(outputs.utility_output, utility)
                        continue
                    ids, sequence = begin_slice(outputs.outputs, "output_socket")
                    if handler:
                        owner = ref()
                        if not owner:
                            return
                        await handler(owner, outputs)
                    if outputs.outputs or outputs.scheduler_stats:
                        queue.put_nowait(outputs)
                    end = time.time_ns()
                    if ids and end > start:
                        task = asyncio.current_task()
                        for rid in ids:
                            remember_request_name(rid, None)
                        emit_output_socket_exec_slice(start, end, ids, id(task) if task else None,
                                                      task.get_name() if task else None, sequence)
            except Exception as exc:
                queue.put_nowait(exc)
            except asyncio.CancelledError:
                queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(receive(), name="EngineCoreOutputQueueTask")
        register_output_task(resources.output_queue_task, "output_socket", "process_outputs_socket")
    return ensure
