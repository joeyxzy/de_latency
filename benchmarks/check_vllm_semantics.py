"""Compare traced request lifecycle with the installed vLLM 0.20 implementation.

Runs without loading model weights. Execute with sitecustomize disabled.
"""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm.v1.engine.async_llm import AsyncLLM, InputStreamError, RequestOutput, STREAM_FINISHED
from de_latency.instrumentation.bootstrap import check_version

original_generate = AsyncLLM.generate
original_output_handler = AsyncLLM._run_output_handler
check_version()
from de_latency.instrumentation import request_hooks  # installs the traced methods


async def generate_case(method, scenario):
    calls = []
    output = object.__new__(RequestOutput)
    output.finished = scenario != "cancel_during"
    error = RuntimeError("input failed")

    class Queue:
        request_id = "internal"

        def get_nowait(self):
            if scenario == "input_error":
                raise InputStreamError(error)
            if scenario == "engine_error":
                raise RuntimeError("engine failed")
            return STREAM_FINISHED if scenario == "sentinel" else output

        def close(self):
            calls.append("close")

    async def add_request(*args, **kwargs):
        calls.append(("add", args, kwargs))
        if scenario == "cancel_before":
            raise asyncio.CancelledError()
        if scenario == "invalid":
            raise ValueError("bad input")
        return Queue()

    async def abort(rid, internal=False):
        calls.append(("abort", rid, internal))

    engine = SimpleNamespace(add_request=add_request, abort=abort, log_requests=False)
    generator = method(engine, "prompt", None, "request")
    received = []
    failure = None
    try:
        async for value in generator:
            received.append(value.finished)
            if scenario == "cancel_during":
                await generator.aclose()
                break
    except BaseException as exc:
        failure = (type(exc).__name__, str(exc), type(exc.__cause__).__name__)
    return calls, received, failure


async def output_case(method, fail=False):
    calls = []
    batches = [[], [1, 2, 3]]

    async def get_output():
        if not batches:
            raise asyncio.CancelledError()
        return SimpleNamespace(outputs=batches.pop(0), timestamp=42,
                               scheduler_stats="stats", engine_index=0)

    def process_outputs(chunk, timestamp, stats):
        calls.append(("process", chunk, timestamp, stats is not None))
        if fail:
            raise ValueError("processing failed")
        # Exercise logger replacement used by elastic EP.
        engine._logger_ref[0] = replacement_logger
        return SimpleNamespace(request_outputs=[], reqs_to_abort=[str(chunk[-1])])

    async def abort(ids):
        calls.append(("abort", ids))

    replacement_logger = SimpleNamespace(record=lambda **kw: calls.append(
        ("record", kw["engine_idx"], kw["iteration_stats"] is not None, kw["mm_cache_stats"])))
    engine = SimpleNamespace(
        output_handler=None, engine_core=SimpleNamespace(get_output_async=get_output, abort_requests_async=abort),
        output_processor=SimpleNamespace(process_outputs=process_outputs,
            update_scheduler_stats=lambda value: calls.append(("stats", value)),
            propagate_error=lambda exc: calls.append(("error", type(exc).__name__, str(exc)))),
        log_stats=True, logger_manager=replacement_logger,
        renderer=SimpleNamespace(stat_mm_cache=lambda: "cache"),
    )
    method(engine)
    try:
        await engine.output_handler
    except asyncio.CancelledError:
        calls.append("cancelled")
    return calls


async def main():
    for scenario in ("complete", "sentinel", "cancel_before", "cancel_during", "invalid", "input_error", "engine_error"):
        expected = await generate_case(original_generate, scenario)
        actual = await generate_case(AsyncLLM.generate, scenario)
        assert actual == expected, (scenario, expected, actual)
        print(f"PASS generate: {scenario}")
    # Override only the test chunk size to force a multi-slice round.
    from vllm.v1.engine import async_llm
    from de_latency.instrumentation.output_loop import run_output_handler

    async_llm.envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE = 2
    def traced(engine):
        engine._logger_ref = [engine.logger_manager]
        engine.output_handler = asyncio.create_task(run_output_handler(
            engine.engine_core, engine.output_processor, engine.log_stats,
            engine._logger_ref, engine.renderer, 2, async_llm.IterationStats,
            Mock(), lambda slices: None, lambda: None))

    for fail in (False, True):
        expected = await output_case(original_output_handler, fail)
        actual = await output_case(traced, fail)
        assert actual == expected, (expected, actual)
        print(f"PASS output loop: error={fail}")


if __name__ == "__main__":
    asyncio.run(main())
