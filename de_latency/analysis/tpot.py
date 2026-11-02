"""TPOT window and queue/execution partition, independent of pipeline state."""
from .clocks import to_ns
from .intervals import clip_interval, sum_clipped_interval_items, uncovered_interval_ns
from .wall_time import ns_to_ms, normalize_component_map_ms


def build_tpot_entry(ttft_entry, dispatches, generate_consume_intervals, queue_intervals,
                     phase_reentry_after_decode):
    tpot_entry = None
    tpot_window_start_ns = to_ns(ttft_entry.get("initial_prefill_end_ns"))
    last_generate_consume_end_ns = None
    if generate_consume_intervals:
        last_generate_consume_end_ns = max(
            to_ns(it.get("end_ns"))
            for it in generate_consume_intervals
            if to_ns(it.get("end_ns")) is not None
        )

    post_start_dispatches = [d for d in dispatches if d["end_ns"] > tpot_window_start_ns]
    post_start_prefill_dispatches = [d for d in post_start_dispatches if d.get("phase") == "prefill"]
    post_start_decode_dispatches = [d for d in post_start_dispatches if d.get("phase") == "decode"]
    decode_steps = len(post_start_decode_dispatches)
    last_decode_end_ns = max((d["end_ns"] for d in post_start_decode_dispatches), default=None)
    tpot_window_end_ns = last_generate_consume_end_ns
    if (
        tpot_window_start_ns is not None
        and decode_steps > 0
        and last_decode_end_ns is not None
        and tpot_window_end_ns is not None
        and last_decode_end_ns > tpot_window_start_ns
        and tpot_window_end_ns >= last_decode_end_ns
    ):
        tpot_exec_end_ns = last_decode_end_ns
        tpot_exec_dispatches = []
        for d in post_start_dispatches:
            span = clip_interval(d.get("start_ns"), d.get("end_ns"), tpot_window_start_ns, tpot_exec_end_ns)
            if not span:
                continue
            d_copy = dict(d)
            d_copy["start_ns"], d_copy["end_ns"] = span
            tpot_exec_dispatches.append(d_copy)
        tpot_exec_ns, tpot_exec_pairs = sum_clipped_interval_items(
            tpot_exec_dispatches,
            tpot_window_start_ns,
            tpot_exec_end_ns,
            merge_overlaps=True,
        )
        tpot_recompute_prefill_exec_ns, _ = sum_clipped_interval_items(
            [d for d in tpot_exec_dispatches if d.get("phase") == "prefill"],
            tpot_window_start_ns,
            tpot_exec_end_ns,
            merge_overlaps=True,
        )
        tpot_decode_exec_ns, _ = sum_clipped_interval_items(
            [d for d in tpot_exec_dispatches if d.get("phase") == "decode"],
            tpot_window_start_ns,
            tpot_exec_end_ns,
            merge_overlaps=True,
        )

        cpu_queue_ns = 0
        tpot_queue_pairs = []
        for q in queue_intervals:
            span = clip_interval(q.get("start_ns"), q.get("end_ns"), tpot_window_start_ns, tpot_exec_end_ns)
            if not span:
                continue
            start_ns, end_ns = span
            tpot_queue_pairs.append(span)
            cpu_queue_ns += end_ns - start_ns

        tpot_gpu_queue_ns = 0
        tpot_gpu_queue_pairs = []
        tpot_dispatches_sorted = sorted(tpot_exec_dispatches, key=lambda x: (x["start_ns"], x["end_ns"]))
        if tpot_dispatches_sorted:
            first_tpot_rpc = to_ns(tpot_dispatches_sorted[0].get("rpc_start_ns"))
            if first_tpot_rpc is not None and first_tpot_rpc > tpot_window_start_ns and first_tpot_rpc < tpot_dispatches_sorted[0]["start_ns"]:
                tpot_gpu_queue_ns += tpot_dispatches_sorted[0]["start_ns"] - first_tpot_rpc
                tpot_gpu_queue_pairs.append((first_tpot_rpc, tpot_dispatches_sorted[0]["start_ns"]))
        for prev_d, next_d in zip(tpot_dispatches_sorted, tpot_dispatches_sorted[1:]):
            prev_end = prev_d["end_ns"]
            next_start = next_d["start_ns"]
            next_rpc = to_ns(next_d.get("rpc_start_ns"))
            if prev_end is None or next_start is None or next_start <= prev_end:
                continue
            if next_rpc is not None and next_rpc > prev_end and next_rpc < next_start:
                tpot_gpu_queue_ns += next_start - next_rpc
                tpot_gpu_queue_pairs.append((next_rpc, next_start))

        tpot_total_ns = tpot_window_end_ns - tpot_window_start_ns
        tail_transport_ns = max(0, tpot_window_end_ns - last_decode_end_ns)
        tpot_other_gap_ns = uncovered_interval_ns(
            tpot_window_start_ns,
            tpot_window_end_ns,
            tpot_queue_pairs + tpot_gpu_queue_pairs + tpot_exec_pairs + [(last_decode_end_ns, tpot_window_end_ns)],
        )
        tpot_components_ns = {
            "cpu_queue": cpu_queue_ns,
            "gpu_queue": tpot_gpu_queue_ns,
            "decode_exec": tpot_exec_ns,
            "tail_transport": tail_transport_ns,
            "other_gap": tpot_other_gap_ns,
        }
        tpot_components_ms_per_token = {
            key: round(val / decode_steps / 1e6, 6)
            for key, val in tpot_components_ns.items()
        }
        tpot_entry = {
            "window_ns": tpot_total_ns,
            "window_ms": ns_to_ms(tpot_total_ns),
            "start_ns": tpot_window_start_ns,
            "end_ns": tpot_window_end_ns,
            "decode_dispatch_count": decode_steps,
            "tpot_start_rule": "initial_prefill_end",
            "tpot_end_rule": "last_generate_consume_end",
            "exec_total_ns": tpot_exec_ns,
            "exec_total_ms": ns_to_ms(tpot_exec_ns),
            "decode_exec_ns": tpot_exec_ns,
            "decode_exec_ms": ns_to_ms(tpot_exec_ns),
            "components_ns_total": tpot_components_ns,
            "components_ms_total": normalize_component_map_ms(tpot_components_ns),
            "components_ms_per_token": tpot_components_ms_per_token,
            "avg_ms_per_token": round(tpot_total_ns / decode_steps / 1e6, 6),
            "denominator_kind": "decode_dispatch_count",
            "phase_reentry_after_decode": phase_reentry_after_decode,
        }

    return tpot_entry
