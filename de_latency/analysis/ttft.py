"""Exclusive TTFT wall-time partition for schema-v2 captures."""
from .clocks import to_ns
from .intervals import clip_interval, merge_intervals, sum_clipped_interval_items
from .wall_time import ns_to_ms, normalize_component_map_ms

COMPONENTS = ("preprocess", "enginecore_queue", "cpu_sched_wait", "gpu_dispatch_wait",
              "prefill_exec", "os_sched_wait", "postprocess_transport", "other_gap")


def partition_ttft(start_ns, end_ns, preprocess_end_ns, engine_queue,
                   cpu_wait, gpu_wait, executions, os_wait, transport_start_ns):
    """Clip all observations to first-token time and count each ns once.

    OS intervals are a subcategory of execution, not additional elapsed time.
    Overlapping worker/rank observations are merged before attribution.
    Execution wins over host dispatch waits when PP steps overlap.
    """
    if end_ns <= start_ns:
        raise ValueError("TTFT window must have positive duration")
    spans = {
        "preprocess": [(start_ns, preprocess_end_ns)],
        "enginecore_queue": engine_queue,
        "cpu_sched_wait": cpu_wait,
        "gpu_dispatch_wait": gpu_wait,
        "prefill_exec": executions,
        "postprocess_transport": [(transport_start_ns, end_ns)],
    }
    spans = {key: merge_intervals([span for a, b in pairs
                                  if (span := clip_interval(a, b, start_ns, end_ns))])
             for key, pairs in spans.items()}
    # OS data is valid only where an observed worker execution overlaps it.
    spans["os_sched_wait"] = merge_intervals([
        overlap for a, b in os_wait for left, right in spans["prefill_exec"]
        if (overlap := clip_interval(a, b, left, right))
    ])
    boundaries = sorted({start_ns, end_ns} | {
        boundary for pairs in spans.values() for pair in pairs for boundary in pair})
    priority = ("os_sched_wait", "prefill_exec", "preprocess", "enginecore_queue",
                "cpu_sched_wait", "gpu_dispatch_wait", "postprocess_transport")
    totals = dict.fromkeys(COMPONENTS, 0)
    for left, right in zip(boundaries, boundaries[1:]):
        component = next((key for key in priority
                          if any(a < right and b > left for a, b in spans[key])), "other_gap")
        totals[component] += right - left
    return totals


def build_ttft_entry(start_ns, end_ns, dispatches, stage_intervals, enqueue_ns,
                     os_wait, *, legacy_os_delay_ns=0, legacy=False,
                     start_rule="coroutine_start", phase_reentry=False):
    """Build one request's TTFT result without renderer or process state.

    The legacy arithmetic is retained only for numerical replay of v1 logs;
    v2 uses the exclusive wall-time partition above.
    """
    dispatches = sorted(dispatches, key=lambda item: (item["start_ns"], item["end_ns"]))
    first_start = to_ns(dispatches[0]["start_ns"])
    last_end = max(item["end_ns"] for item in dispatches)
    engine_queue = [item for item in stage_intervals
                    if item.get("stage") == "engine_core.input_queue_wait_after_preprocess"]
    queue_ns, queue_pairs = sum_clipped_interval_items(engine_queue, start_ns, end_ns)
    ordered_queue = sorted(queue_pairs)
    boundary = ordered_queue[0][0] if ordered_queue else (
        enqueue_ns if enqueue_ns is not None else first_start)
    boundary = max(start_ns, min(boundary, end_ns))
    cpu_start = enqueue_ns if enqueue_ns is not None else (
        ordered_queue[-1][1] if ordered_queue else boundary)
    cpu_start = max(cpu_start, boundary)
    cpu_wait, gpu_wait = [], []

    def append_wait(left, right, rpc):
        if rpc is not None and rpc > left:
            cpu_wait.append((left, rpc))
            if right > rpc:
                gpu_wait.append((rpc, right))
        elif right > left:
            gpu_wait.append((left, right))

    append_wait(cpu_start, first_start, to_ns(dispatches[0].get("rpc_start_ns")))
    for previous, following in zip(dispatches, dispatches[1:]):
        if following["start_ns"] > previous["end_ns"]:
            append_wait(previous["end_ns"], following["start_ns"], to_ns(following.get("rpc_start_ns")))
    exec_ns, executions = sum_clipped_interval_items(dispatches, first_start, last_end, merge_overlaps=True)
    total = end_ns - start_ns
    if legacy:
        components = dict(zip(COMPONENTS, (
            max(0, boundary - start_ns), queue_ns,
            sum(b - a for a, b in cpu_wait), sum(b - a for a, b in gpu_wait),
            exec_ns, legacy_os_delay_ns, max(0, end_ns - last_end), 0,
        )))
        components["other_gap"] = max(0, total - sum(components.values()))
    else:
        components = partition_ttft(start_ns, end_ns, boundary, queue_pairs, cpu_wait,
                                    gpu_wait, executions, os_wait, last_end)
    return {"window_ns": total, "window_ms": ns_to_ms(total),
            "start_ns": start_ns, "end_ns": end_ns,
            "initial_prefill_end_ns": last_end, "initial_prefill_end_ms": ns_to_ms(last_end),
            "ttft_start_rule": start_rule, "prefill_dispatch_count": len(dispatches),
            "ttft_end_rule": "first_generate_consume_end", "phase_reentry_after_decode": phase_reentry,
            "components_ns": components, "components_ms": normalize_component_map_ms(components)}
