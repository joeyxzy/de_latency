"""Compute-side TTFT/TPOT summaries."""
from .dispatch import DispatchAnalysis
from .ingestion import LogData
from .os_scheduling import OsAttribution
from .request_index import RequestIndex
from .request_stages import RequestStages
from .request_timing import RequestTiming
from collections import defaultdict
from de_latency.analysis.clocks import to_ns
from de_latency.analysis.ttft import build_ttft_entry
from dataclasses import dataclass
from .tpot import build_tpot_entry

@dataclass
class ComputeBreakdown:
    compute_breakdown: dict


def build_compute_breakdown(dispatch_analysis: DispatchAnalysis, log_data: LogData, os_result: OsAttribution, request_index: RequestIndex, request_stages: RequestStages, request_timing: RequestTiming) -> ComputeBreakdown:
    compute_breakdown = {
        "version": 4,
        "tpot_denominator": "decode_dispatch_count",
        "wall_time_policy": "TTFT prefill_exec and TPOT exec phases use TP-aware wall-time attribution; pp_gap is measured from adjacent PP Step Execution windows within the same dispatch/batch after merging TP ranks in each PP stage.",
        "scheduling_wait_policy": "preempt_gap is classified only from explicit vLLM per-request preempt events; waiting behind another running batch remains normal_gap.",
        "requests": {},
        "summary": {},
        "skipped_requests": {
            "ttft_missing": [],
            "tpot_missing": [],
        },
        "phase_reentry_after_decode_requests": [],
    }
    if log_data.diagnostics.schema_versions != {1}:
        compute_breakdown["version"] = 5
        compute_breakdown["ttft_partition_policy"] = (
            "exclusive wall-time intervals clipped to first generate consume; "
            "OS wait is a union within execution, not additional elapsed time"
        )
    ttft_component_totals_ns = defaultdict(int)
    ttft_req_count = 0
    tpot_component_totals_ns = defaultdict(int)
    tpot_req_count = 0
    tpot_total_decode_steps = 0
    tpot_per_req_values_ns = []

    for rid in sorted(request_index.all_req_ids):
        dispatches = sorted(dispatch_analysis.req_dispatch_intervals_map.get(rid, []), key=lambda x: (x["start_ns"], x["end_ns"]))
        prefill_dispatches = [d for d in dispatches if d.get("phase") == "prefill"]
        decode_dispatches = [d for d in dispatches if d.get("phase") == "decode"]
        first_decode_index = next((idx for idx, d in enumerate(dispatches) if d.get("phase") == "decode"), None)
        if first_decode_index is None:
            initial_prefill_dispatches = prefill_dispatches
            phase_reentry_after_decode = False
        else:
            initial_prefill_dispatches = [d for d in dispatches[:first_decode_index] if d.get("phase") == "prefill"]
            phase_reentry_after_decode = any(d.get("phase") == "prefill" for d in dispatches[first_decode_index + 1:])
        if phase_reentry_after_decode:
            compute_breakdown["phase_reentry_after_decode_requests"].append(rid)

        dispatch_phase_by_start = {}
        for d in dispatches:
            dispatch_phase_by_start.setdefault(d["start_ns"], d.get("phase", "unknown"))

        queue_intervals = []
        for q in sorted(dispatch_analysis.queue_intervals_map.get(rid, []), key=lambda x: (x["start_ns"], x["end_ns"])):
            q_copy = dict(q)
            q_copy["phase"] = dispatch_phase_by_start.get(q_copy.get("end_ns"))
            queue_intervals.append(q_copy)

        ttft_entry = None
        generate_setup_intervals = [
            s for s in request_stages.req_stage_intervals.get(rid, [])
            if s.get("stage") == "async_llm.generate_wrapper_setup"
        ]
        generate_setup_start_ns = None
        if generate_setup_intervals:
            generate_setup_start_ns = min(to_ns(s.get("start_ns")) for s in generate_setup_intervals if to_ns(s.get("start_ns")) is not None)
        ttft_start_ns = generate_setup_start_ns or to_ns(request_timing.req_generate_start_map.get(rid))
        generate_consume_intervals = sorted(
            [
                it for it in request_timing.req_generate_exec_intervals.get(rid, [])
                if it.get("task_kind") == "generate_consume"
            ],
            key=lambda x: (x["start_ns"], x["end_ns"]),
        )
        first_generate_consume_end_ns = None
        if generate_consume_intervals:
            first_generate_consume_end_ns = to_ns(generate_consume_intervals[0].get("end_ns"))

        if ttft_start_ns is not None and initial_prefill_dispatches and first_generate_consume_end_ns is not None:
            ttft_end_ns = first_generate_consume_end_ns
            if ttft_end_ns > ttft_start_ns:
                offset = os_result.offset_ns
                os_wait_pairs = [
                    (iv["start_mono_ns"] + offset, iv["end_mono_ns"] + offset)
                    for dispatch in initial_prefill_dispatches
                    for iv in os_result.overlaps.get(dispatch.get("dispatch_key"), [])
                ] if offset is not None else []
                ttft_entry = build_ttft_entry(
                    ttft_start_ns, ttft_end_ns, initial_prefill_dispatches,
                    request_stages.req_stage_intervals.get(rid, []), to_ns(dispatch_analysis.req_enqueue_map.get(rid)), os_wait_pairs,
                    legacy_os_delay_ns=sum(os_result.dispatch_delay_ns.get(d.get("dispatch_key"), 0)
                                           for d in initial_prefill_dispatches),
                    legacy=log_data.diagnostics.schema_versions == {1},
                    start_rule="generate_wrapper_setup_start" if generate_setup_start_ns is not None else "coroutine_start",
                    phase_reentry=phase_reentry_after_decode,
                )
                ttft_req_count += 1
                for key, val in ttft_entry["components_ns"].items():
                    ttft_component_totals_ns[key] += val
            else:
                compute_breakdown["skipped_requests"]["ttft_missing"].append(rid)
        else:
            compute_breakdown["skipped_requests"]["ttft_missing"].append(rid)

        tpot_entry = build_tpot_entry(ttft_entry, dispatches, generate_consume_intervals,
                                      queue_intervals, phase_reentry_after_decode) if ttft_entry and decode_dispatches else None
        if tpot_entry:
            tpot_req_count += 1
            decode_steps = tpot_entry["decode_dispatch_count"]
            tpot_total_decode_steps += decode_steps
            tpot_per_req_values_ns.append(tpot_entry["window_ns"] / decode_steps)
            for key, val in tpot_entry["components_ns_total"].items():
                tpot_component_totals_ns[key] += val
        else:
            compute_breakdown["skipped_requests"]["tpot_missing"].append(rid)

        compute_breakdown["requests"][rid] = {
            "request_name": log_data.req_name_map.get(rid, rid),
            "ttft": ttft_entry,
            "tpot": tpot_entry,
        }

    ttft_avg_components_ms = {}
    if ttft_req_count > 0:
        ttft_avg_components_ms = {
            key: round(ttft_component_totals_ns.get(key, 0) / ttft_req_count / 1e6, 6)
            for key in ["preprocess", "enginecore_queue", "cpu_sched_wait", "gpu_dispatch_wait", "prefill_exec", "os_sched_wait", "postprocess_transport", "other_gap"]
        }
    tpot_avg_components_ms = {}
    if tpot_total_decode_steps > 0:
        tpot_avg_components_ms = {
            key: round(tpot_component_totals_ns.get(key, 0) / tpot_total_decode_steps / 1e6, 6)
            for key in ["cpu_queue", "gpu_queue", "decode_exec", "tail_transport", "other_gap"]
        }

    compute_breakdown["summary"] = {
        "ttft_request_count": ttft_req_count,
        "ttft_avg_components_ms": ttft_avg_components_ms,
        "ttft_avg_ms": round(sum(ttft_avg_components_ms.values()), 6) if ttft_avg_components_ms else None,
        "tpot_request_count": tpot_req_count,
        "tpot_total_decode_steps": tpot_total_decode_steps,
        "tpot_avg_components_ms_per_token": tpot_avg_components_ms,
        "tpot_avg_ms_per_token": round(sum(tpot_avg_components_ms.values()), 6) if tpot_avg_components_ms else None,
        "tpot_mean_request_ms_per_token": round(sum(tpot_per_req_values_ns) / len(tpot_per_req_values_ns) / 1e6, 6) if tpot_per_req_values_ns else None,
        "phase_reentry_after_decode_request_count": len(compute_breakdown["phase_reentry_after_decode_requests"]),
    }

    print("=== Compute-Side Breakdown Summary ===")
    if ttft_avg_components_ms:
        print(
            f"TTFT Avg ({ttft_req_count} reqs, compute-side): "
            f"{compute_breakdown['summary']['ttft_avg_ms']:.3f} ms"
        )
        for key in ["preprocess", "enginecore_queue", "cpu_sched_wait", "gpu_dispatch_wait", "prefill_exec", "os_sched_wait", "postprocess_transport", "other_gap"]:
            print(f"  - {key}: {ttft_avg_components_ms.get(key, 0.0):.3f} ms")
    else:
        print("TTFT Avg (compute-side): 无可用请求")

    if tpot_avg_components_ms:
        print(
            f"TPOT Avg ({tpot_req_count} reqs, {tpot_total_decode_steps} decode steps, compute-side): "
            f"{compute_breakdown['summary']['tpot_avg_ms_per_token']:.3f} ms/token "
            f"(denominator=decode_dispatch_count)"
        )
        for key in ["cpu_queue", "gpu_queue", "decode_exec", "tail_transport", "other_gap"]:
            print(f"  - {key}: {tpot_avg_components_ms.get(key, 0.0):.3f} ms/token")
    else:
        print("TPOT Avg (compute-side): 无可用 decode 请求")

    if compute_breakdown["phase_reentry_after_decode_requests"]:
        print(
            f"[warn] observed {len(compute_breakdown['phase_reentry_after_decode_requests'])} requests with prefill re-entry after decode; "
            "TTFT keeps only the initial prefill segment, and the later recompute-prefill cost is attributed to TPOT."
        )


    return ComputeBreakdown(
        compute_breakdown=compute_breakdown,
    )
