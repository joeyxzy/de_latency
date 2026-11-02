"""Request stage intervals and totals."""
from .dispatch import DispatchAnalysis
from .ingestion import LogData
from .request_timing import RequestTiming
from collections import defaultdict
from de_latency.analysis.clocks import to_ns, to_int
from dataclasses import dataclass

@dataclass
class RequestStages:
    req_stage_intervals: dict
    req_stage_ns: dict


def analyze_request_stages(dispatch_analysis: DispatchAnalysis, log_data: LogData, request_timing: RequestTiming) -> RequestStages:
    print(f"Processing Request Stage Events ({len(log_data.req_stage_events)})...")
    req_stage_ns = defaultdict(lambda: defaultdict(int))
    req_stage_counts = defaultdict(lambda: defaultdict(int))
    req_stage_intervals = defaultdict(list)

    for ev in log_data.req_stage_events:
        payload = ev.get("payload", {})
        rid = payload.get("request_id") or payload.get("req_id")
        if not rid:
            continue
        rname = payload.get("request_name")
        if rname:
            log_data.req_name_map[rid] = rname
        stage = payload.get("stage")
        if not stage:
            continue
        start_ns = to_ns(payload.get("start_ns"))
        end_ns = to_ns(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        if dur_ns is None:
            if start_ns is None or end_ns is None or end_ns < start_ns:
                continue
            dur_ns = end_ns - start_ns
        if dur_ns < 0:
            continue
        req_stage_ns[rid][stage] += dur_ns
        req_stage_counts[rid][stage] += 1
        if start_ns is not None and end_ns is not None and end_ns > start_ns:
            req_stage_intervals[rid].append({
                "stage": stage,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
            })

    if req_stage_ns:
        stage_order = [
            "async_llm.generate_wrapper_setup",
            "async_llm._run_output_handler",
            "processor.process_inputs",
            "output_processor.add_request",
            "engine_core.add_request_async",
            "engine_core.preprocess_add_request",
            "engine_core.input_queue_wait_after_preprocess",
            "engine_core.handle_client_add",
            "engine_core.add_request",
            "async_llm._add_request",
            "async_llm.add_request",
        ]
        all_stage_reqs = set(req_stage_ns.keys()) | set(request_timing.req_generate_start_map.keys()) | set(dispatch_analysis.req_enqueue_map.keys())
        print("=== Generate -> Scheduler Enqueue Breakdown ===")
        for rid in sorted(
            all_stage_reqs,
            key=lambda x: (
                (to_ns(dispatch_analysis.req_enqueue_map.get(x)) or 0) - (to_ns(request_timing.req_generate_start_map.get(x)) or 0)
                if (to_ns(dispatch_analysis.req_enqueue_map.get(x)) and to_ns(request_timing.req_generate_start_map.get(x)))
                else sum(req_stage_ns.get(x, {}).values())
            ),
            reverse=True,
        ):
            stage_map = req_stage_ns.get(rid, {})
            if not stage_map:
                continue
            known_ns = sum(stage_map.values())
            gen_start_ns = to_ns(request_timing.req_generate_start_map.get(rid))
            enqueue_ns = to_ns(dispatch_analysis.req_enqueue_map.get(rid))
            pre_enqueue_total_ns = None
            if gen_start_ns is not None and enqueue_ns is not None and enqueue_ns >= gen_start_ns:
                pre_enqueue_total_ns = enqueue_ns - gen_start_ns
            uncovered_ns = None
            if pre_enqueue_total_ns is not None:
                uncovered_ns = max(0, pre_enqueue_total_ns - known_ns)

            label = log_data.req_name_map.get(rid, rid)
            total_str = f"{pre_enqueue_total_ns / 1e6:.3f} ms" if pre_enqueue_total_ns is not None else "N/A"
            uncovered_str = f"{uncovered_ns / 1e6:.3f} ms" if uncovered_ns is not None else "N/A"
            print(f"  - {rid} ({label}): total(generate->enqueue)={total_str}, known_stages={known_ns / 1e6:.3f} ms, uncovered={uncovered_str}")

            printed = set()
            for stage in stage_order:
                if stage in stage_map:
                    print(
                        f"      * {stage}: {stage_map[stage] / 1e6:.3f} ms "
                        f"(count={req_stage_counts[rid].get(stage, 0)})"
                    )
                    printed.add(stage)
            for stage, dur_ns in sorted(stage_map.items(), key=lambda x: x[1], reverse=True):
                if stage in printed:
                    continue
                print(
                    f"      * {stage}: {dur_ns / 1e6:.3f} ms "
                    f"(count={req_stage_counts[rid].get(stage, 0)})"
                )
    else:
        print("  未统计到 generate->enqueue 阶段事件")


    return RequestStages(
        req_stage_intervals=req_stage_intervals,
        req_stage_ns=req_stage_ns,
    )
