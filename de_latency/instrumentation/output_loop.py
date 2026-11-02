"""vLLM 0.20 output loop with explicit tracing boundaries.

Keep business operations in the order used by AsyncLLM._run_output_handler:
process chunk, yield between chunks, abort, update scheduler stats, log.
The coroutine retains components rather than the AsyncLLM instance.
"""
import asyncio


async def run_output_handler(engine_core, output_processor, log_stats,
                             logger_ref, renderer, chunk_size, iteration_stats_cls,
                             logger, begin_round, end_round):
    try:
        while True:
            outputs = await engine_core.get_output_async()
            count = len(outputs.outputs)
            stats = iteration_stats_cls() if log_stats and count else None
            slices = tuple(outputs.outputs[start:start + chunk_size]
                           for start in range(0, count, chunk_size))
            begin_round(slices)
            try:
                for index, chunk in enumerate(slices):
                    processed = output_processor.process_outputs(chunk, outputs.timestamp, stats)
                    assert not processed.request_outputs
                    if index + 1 < len(slices):
                        await asyncio.sleep(0)
                    if processed.reqs_to_abort:
                        await engine_core.abort_requests_async(processed.reqs_to_abort)
            finally:
                end_round()
            output_processor.update_scheduler_stats(outputs.scheduler_stats)
            if logger_ref[0]:
                logger_ref[0].record(
                    engine_idx=outputs.engine_index,
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=stats,
                    mm_cache_stats=renderer.stat_mm_cache(),
                )
    except Exception as exc:
        logger.exception("AsyncLLM output_handler failed.")
        output_processor.propagate_error(exc)
