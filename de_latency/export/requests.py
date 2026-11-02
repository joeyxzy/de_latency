"""Request interference trace tracks; does not calculate compute attribution."""
import json
from collections import defaultdict
from de_latency.analysis.intervals import clip_interval, merge_intervals, sum_interval_pairs
from de_latency.export.lanes import assign_non_overlapping_lanes
from de_latency.export.request_layout import build_request_lane_tids, build_raw_dispatch_display_intervals

def append_request_trace(compute_analysis, dispatch_analysis, log_data, os_result, request_index, request_stages, request_timing, worker_analysis, trace_events, writer):
    REQUEST_PID = 11000
    trace_events.append({"name": "process_name", "ph": "M", "pid": REQUEST_PID, "args": {"name": "Request Interference"}})
    create_perfetto_event = writer.event
    def req_sort_key(rid):
        span = request_timing.request_spans.get(rid)
        if span and span.get("start_ns"):
            return span["start_ns"]
        q = dispatch_analysis.queue_intervals_map.get(rid)
        if q:
            return q[0]["start_ns"]
        tq = request_timing.req_generate_task_sched_intervals.get(rid)
        if tq:
            return tq[0]["start_ns"]
        osq = request_timing.req_output_socket_sched_intervals.get(rid)
        if osq:
            return osq[0]["start_ns"]
        o = request_index.req_os_intervals_wall.get(rid)
        if o:
            return o[0]["start_ns"]
        s = request_stages.req_stage_intervals.get(rid)
        if s:
            return min(it["start_ns"] for it in s)
        oh = request_timing.req_output_handler_sched_intervals.get(rid)
        if oh:
            return oh[0]["start_ns"]
        gx = request_timing.req_generate_exec_intervals.get(rid)
        if gx:
            return gx[0]["start_ns"]
        sx = request_timing.req_output_socket_exec_intervals.get(rid)
        if sx:
            return sx[0]["start_ns"]
        ox = request_timing.req_output_handler_exec_intervals.get(rid)
        if ox:
            return ox[0]["start_ns"]
        return float("inf")

    sorted_req_ids = sorted(request_index.all_req_ids, key=req_sort_key)

    def _request_pp_gap_intervals(rid):
        req_entry = compute_analysis.compute_breakdown["requests"].get(rid, {})
        intervals = []
        for section_name in ("ttft", "tpot"):
            section = req_entry.get(section_name) or {}
            breakdown = section.get("pp_gap_breakdown") or {}
            for dispatch in breakdown.get("dispatches", []):
                for gap in dispatch.get("pp_gap_intervals", []):
                    item = dict(gap)
                    item["section"] = section_name
                    item["dispatch_key"] = dispatch.get("dispatch_key")
                    item["phase"] = dispatch.get("phase")
                    intervals.append(item)
        return sorted(intervals, key=lambda item: (item.get("start_ns") or 0, item.get("end_ns") or 0))

    def _request_pp_gap_wall_ns(rid):
        pairs = []
        for item in _request_pp_gap_intervals(rid):
            span = clip_interval(item.get("start_ns"), item.get("end_ns"))
            if span:
                pairs.append(span)
        return sum_interval_pairs(merge_intervals(pairs))

    for req_index, rid in enumerate(sorted_req_ids, start=1):
        lane_tids = build_request_lane_tids(req_index)
        dispatch_intervals = list(dispatch_analysis.req_dispatch_intervals_map.get(rid, []))
        dispatch_display_intervals = build_raw_dispatch_display_intervals(dispatch_intervals)
        dispatch_lane_indices, dispatch_lane_count = assign_non_overlapping_lanes(dispatch_display_intervals)
        pp_gap_intervals = _request_pp_gap_intervals(rid)
        pp_gap_lane_indices, pp_gap_lane_count = assign_non_overlapping_lanes(pp_gap_intervals)
        lane_names = {
            "label": f"{rid[:12]}",
            "lifecycle": f"{rid[:12]} | lifecycle",
            "vllm_queue": f"{rid[:12]} | cpu sched queue",
            "coro_queue": f"{rid[:12]} | coroutine queue",
            "output_socket_queue": f"{rid[:12]} | output socket queue",
            "output_queue": f"{rid[:12]} | output queue",
            "dispatch": f"{rid[:12]} | dispatch",
            "generate_exec": f"{rid[:12]} | generate exec",
            "output_socket_exec": f"{rid[:12]} | output socket exec",
            "output_exec": f"{rid[:12]} | output exec",
            "os": f"{rid[:12]} | os",
            "stage": f"{rid[:12]} | stage",
            "task": f"{rid[:12]} | task",
            "pp_gap": f"{rid[:12]} | pp gap",
            "gpu_queue": f"{rid[:12]} | gpu dispatch queue",
        }
        if dispatch_lane_count > 1:
            lane_names["dispatch"] = f"{rid[:12]} | dispatch 1"
            for lane_idx in range(1, dispatch_lane_count):
                key = f"dispatch_{lane_idx}"
                lane_tids[key] = req_index * 100 + 50 + lane_idx
                lane_names[key] = f"{rid[:12]} | dispatch {lane_idx + 1}"
        if pp_gap_lane_count > 1:
            lane_names["pp_gap"] = f"{rid[:12]} | pp gap 1"
            for lane_idx in range(1, pp_gap_lane_count):
                key = f"pp_gap_{lane_idx}"
                lane_tids[key] = req_index * 100 + 70 + lane_idx
                lane_names[key] = f"{rid[:12]} | pp gap {lane_idx + 1}"
        for lane_key, lane_tid in lane_tids.items():
            trace_events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": REQUEST_PID,
                "tid": lane_tid,
                "args": {"name": lane_names[lane_key]},
            })

        coro_sched_queue_ms = request_timing.req_coro_sched_ns.get(rid, 0) / 1e6
        generate_task_sched_queue_ms = request_timing.req_generate_task_sched_ns.get(rid, 0) / 1e6
        generate_exec_ms = request_timing.req_generate_exec_ns.get(rid, 0) / 1e6
        output_socket_sched_queue_ms = request_timing.req_output_socket_sched_ns.get(rid, 0) / 1e6
        output_socket_exec_ms = request_timing.req_output_socket_exec_ns.get(rid, 0) / 1e6
        output_handler_sched_queue_ms = request_timing.req_output_handler_sched_ns.get(rid, 0) / 1e6
        output_handler_exec_ms = request_timing.req_output_handler_exec_ns.get(rid, 0) / 1e6
        vllm_queue_ms = dispatch_analysis.req_vllm_queue_ns.get(rid, 0) / 1e6
        pp_gap_ms = _request_pp_gap_wall_ns(rid) / 1e6
        vllm_queue_from_enqueue_ms = dispatch_analysis.req_vllm_queue_from_enqueue_ns.get(rid, 0) / 1e6
        vllm_queue_from_step_ready_ms = dispatch_analysis.req_vllm_queue_from_step_ready_ns.get(rid, 0) / 1e6
        gpu_queue_ms = dispatch_analysis.req_gpu_queue_ns.get(rid, 0) / 1e6
        os_delay_ms = os_result.request_delay_ns.get(rid, 0) / 1e6
        engine_input_queue_wait_ms = request_stages.req_stage_ns.get(rid, {}).get(
            "engine_core.input_queue_wait_after_preprocess", 0) / 1e6
        prefill_exec_ms = dispatch_analysis.req_dispatch_phase_ns.get(rid, {}).get("prefill", 0) / 1e6
        decode_exec_ms = dispatch_analysis.req_dispatch_phase_ns.get(rid, {}).get("decode", 0) / 1e6
        unknown_exec_ms = dispatch_analysis.req_dispatch_phase_ns.get(rid, {}).get("unknown", 0) / 1e6

        span = request_timing.request_spans.get(rid)
        if span and span.get("end_ns", 0) > span.get("start_ns", 0):
            trace_events.append(create_perfetto_event(
                name=f"Req {rid[:8]} Lifecycle",
                cat="request_lifecycle",
                ph="X",
                ts=span["start_ns"],
                dur=span["end_ns"] - span["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids["lifecycle"],
                args={
                    "request_id": rid,
                    "request_name": log_data.req_name_map.get(rid, rid),
                    "coro_sched_queue_ms": round(coro_sched_queue_ms, 3),
                    "generate_task_sched_queue_ms": round(generate_task_sched_queue_ms, 3),
                    "generate_exec_ms": round(generate_exec_ms, 3),
                    "output_socket_sched_queue_ms": round(output_socket_sched_queue_ms, 3),
                    "output_socket_exec_ms": round(output_socket_exec_ms, 3),
                    "output_handler_sched_queue_ms": round(output_handler_sched_queue_ms, 3),
                    "output_handler_exec_ms": round(output_handler_exec_ms, 3),
                    "cpu_sched_queue_ms": round(vllm_queue_ms, 3),
                    "pp_gap_ms": round(pp_gap_ms, 3),
                    "cpu_sched_queue_from_enqueue_ms": round(vllm_queue_from_enqueue_ms, 3),
                    "cpu_sched_queue_from_step_ready_ms": round(vllm_queue_from_step_ready_ms, 3),
                    "gpu_dispatch_queue_ms": round(gpu_queue_ms, 3),
                    "engine_input_queue_wait_ms": round(engine_input_queue_wait_ms, 3),
                    "os_sched_delay_ms": round(os_delay_ms, 3),
                    "prefill_exec_ms": round(prefill_exec_ms, 3),
                    "decode_exec_ms": round(decode_exec_ms, 3),
                    "unknown_exec_ms": round(unknown_exec_ms, 3),
                },
                cname="good",
            ))

        for dispatch_idx, d in enumerate(dispatch_display_intervals):
            if d["end_ns"] <= d["start_ns"]:
                continue
            dispatch_lane_idx = dispatch_lane_indices[dispatch_idx] if dispatch_idx < len(dispatch_lane_indices) else 0
            dispatch_lane_key = "dispatch" if dispatch_lane_idx == 0 else f"dispatch_{dispatch_lane_idx}"
            phase = d.get("phase", "unknown")
            if phase == "prefill":
                d_name = "Prefill Dispatch"
                d_cat = "request_dispatch_prefill"
                d_color = "olive"
            elif phase == "decode":
                d_name = "Decode Dispatch"
                d_cat = "request_dispatch_decode"
                d_color = "rail_response"
            else:
                d_name = "Unknown Dispatch"
                d_cat = "request_dispatch_unknown"
                d_color = "background"
            trace_events.append(create_perfetto_event(
                name=d_name,
                cat=d_cat,
                ph="X",
                ts=d["start_ns"],
                dur=d["end_ns"] - d["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids.get(dispatch_lane_key, lane_tids["dispatch"]),
                args={
                    "request_id": rid,
                    "phase": phase,
                    "phase_reason": d.get("phase_reason"),
                    "decode_step": d.get("decode_step"),
                    "dispatch_key": d.get("dispatch_key"),
                    "dispatch_keys": d.get("dispatch_keys"),
                    "dispatch_key_count": d.get("dispatch_key_count"),
                    "step_count": d.get("step_count"),
                    "batch_req_ids": d.get("batch_req_ids", []),
                    "batch_req_count": len(d.get("batch_req_ids") or []),
                    "duration_ms": round((d["end_ns"] - d["start_ns"]) / 1e6, 3),
                    "start_source": d.get("start_source"),
                    "end_source": d.get("end_source"),
                    "gpu_marker_start_ns": d.get("gpu_marker_start_ns"),
                    "gpu_marker_end_ns": d.get("gpu_marker_end_ns"),
                    "gpu_queue_ns": d.get("gpu_queue_ns", 0),
                    "rpc_start_ns": d.get("rpc_start_ns"),
                    "scheduler_ready_lag_ms": (
                        round((d["scheduler_ready_ns"] - d["worker_end_ns"]) / 1e6, 3)
                        if d.get("scheduler_ready_ns") is not None and d.get("worker_end_ns") is not None
                        else None
                    ),
                    "rpc_to_worker_queue_ms": (
                        round(d["max_rpc_to_worker_queue_ns"] / 1e6, 3)
                        if d.get("max_rpc_to_worker_queue_ns") is not None
                        else round((d["worker_start_ns"] - d["rpc_start_ns"]) / 1e6, 3)
                        if d.get("worker_start_ns") is not None and d.get("rpc_start_ns") is not None
                        else None
                    ),
                    "worker_span_count": d.get("worker_span_count"),
                    "worker_span_count_total": d.get("worker_span_count_total"),
                    "worker_boundary_policy": d.get("worker_boundary_policy"),
                    "scheduled_tokens_by_req": json.dumps(
                        worker_analysis.dispatch_scheduled_tokens_cache.get(d.get("dispatch_key"), {}),
                        separators=(",", ":")
                    ) if d.get("dispatch_key") else None,
                    "prefill_tokens": worker_analysis.dispatch_prefill_tokens_cache.get(d.get("dispatch_key")),
                    "decode_tokens": worker_analysis.dispatch_decode_tokens_cache.get(d.get("dispatch_key")),
                },
                cname=d_color,
            ))

        for gap_idx, gap in enumerate(pp_gap_intervals):
            if gap["end_ns"] > gap["start_ns"]:
                pp_gap_lane_idx = pp_gap_lane_indices[gap_idx] if gap_idx < len(pp_gap_lane_indices) else 0
                pp_gap_lane_key = "pp_gap" if pp_gap_lane_idx == 0 else f"pp_gap_{pp_gap_lane_idx}"
                trace_events.append(create_perfetto_event(
                    name="PP Gap",
                    cat="request_pp_gap",
                    ph="X",
                    ts=gap["start_ns"],
                    dur=gap["end_ns"] - gap["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids.get(pp_gap_lane_key, lane_tids["pp_gap"]),
                    args={
                        "request_id": rid,
                        "dispatch_key": gap.get("dispatch_key"),
                        "phase": gap.get("phase"),
                        "section": gap.get("section"),
                        "edge": gap.get("edge"),
                        "dp_rank": gap.get("dp_rank"),
                        "src_pp": gap.get("src_pp"),
                        "dst_pp": gap.get("dst_pp"),
                        "duration_ms": gap.get("duration_ms"),
                    },
                    cname="rail_load",
                ))

        for q0 in request_timing.req_coro_sched_intervals.get(rid, []):
            if q0["end_ns"] > q0["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Coroutine Scheduler Queue Wait",
                    cat="request_coro_sched_queue",
                    ph="X",
                    ts=q0["start_ns"],
                    dur=q0["end_ns"] - q0["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["coro_queue"],
                    args={"request_id": rid, "request_name": log_data.req_name_map.get(rid, rid)},
                    cname="yellow",
                ))

        for tq in request_timing.req_generate_task_sched_intervals.get(rid, []):
            if tq["end_ns"] > tq["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Generate Task Runnable Queue",
                    cat="request_generate_task_queue",
                    ph="X",
                    ts=tq["start_ns"],
                    dur=tq["end_ns"] - tq["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["task"],
                    args={"request_id": rid, "request_name": log_data.req_name_map.get(rid, rid)},
                    cname="yellow",
                ))

        for gx in request_timing.req_generate_exec_intervals.get(rid, []):
            if gx["end_ns"] > gx["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Generate Coroutine Exec",
                    cat="request_generate_exec",
                    ph="X",
                    ts=gx["start_ns"],
                    dur=gx["end_ns"] - gx["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["generate_exec"],
                    args={
                        "request_id": rid,
                        "request_name": log_data.req_name_map.get(rid, rid),
                        "task_name": gx.get("task_name"),
                    },
                    cname="rail_idle",
                ))

        for sq in request_timing.req_output_socket_sched_intervals.get(rid, []):
            if sq["end_ns"] > sq["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Socket Scheduler Queue Wait",
                    cat="request_output_socket_sched_queue",
                    ph="X",
                    ts=sq["start_ns"],
                    dur=sq["end_ns"] - sq["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_socket_queue"],
                    args={
                        "request_id": rid,
                        "request_name": log_data.req_name_map.get(rid, rid),
                        "shared_req_count": sq.get("shared_req_count"),
                        "round_seq": sq.get("round_seq"),
                    },
                    cname="yellow",
                ))

        for sx in request_timing.req_output_socket_exec_intervals.get(rid, []):
            if sx["end_ns"] > sx["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Socket Exec (Attributed)",
                    cat="request_output_socket_exec",
                    ph="X",
                    ts=sx["start_ns"],
                    dur=sx["end_ns"] - sx["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_socket_exec"],
                    args={
                        "request_id": rid,
                        "request_name": log_data.req_name_map.get(rid, rid),
                        "shared_req_count": sx.get("shared_req_count"),
                        "round_seq": sx.get("round_seq"),
                    },
                    cname="cyan",
                ))

        for qh in request_timing.req_output_handler_sched_intervals.get(rid, []):
            if qh["end_ns"] > qh["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Handler Scheduler Queue Wait",
                    cat="request_output_handler_sched_queue",
                    ph="X",
                    ts=qh["start_ns"],
                    dur=qh["end_ns"] - qh["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_queue"],
                    args={
                        "request_id": rid,
                        "request_name": log_data.req_name_map.get(rid, rid),
                        "shared_req_count": qh.get("shared_req_count"),
                        "round_seq": qh.get("round_seq"),
                    },
                    cname="yellow",
                ))

        for ox in request_timing.req_output_handler_exec_intervals.get(rid, []):
            if ox["end_ns"] > ox["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Handler Exec (Attributed)",
                    cat="request_output_handler_exec",
                    ph="X",
                    ts=ox["start_ns"],
                    dur=ox["end_ns"] - ox["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_exec"],
                    args={
                        "request_id": rid,
                        "request_name": log_data.req_name_map.get(rid, rid),
                        "shared_req_count": ox.get("shared_req_count"),
                        "round_seq": ox.get("round_seq"),
                    },
                    cname="cyan",
                ))

        for gq in dispatch_analysis.gpu_queue_intervals_map.get(rid, []):
            if gq["end_ns"] > gq["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name=f"GPU Dispatch Wait ({gq.get('phase', 'unknown')})",
                    cat="request_gpu_queue",
                    ph="X",
                    ts=gq["start_ns"],
                    dur=gq["end_ns"] - gq["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["gpu_queue"],
                    args={
                        "request_id": rid,
                        "dispatch_key": gq.get("dispatch_key"),
                        "phase": gq.get("phase"),
                        "duration_ms": round(gq.get("duration_ns", 0) / 1e6, 3),
                        "start_source": gq.get("start_source", "scheduler_out_rpc"),
                        "end_source": gq.get("end_source", "gpu_or_worker_start"),
                    },
                    cname="lightorange",
                ))

        for q in dispatch_analysis.queue_intervals_map.get(rid, []):
            if q["end_ns"] > q["start_ns"]:
                scheduled_from = q.get("scheduled_from")
                gap_kind = q.get("gap_kind", "normal_wait")
                if gap_kind == "preempt_wait":
                    q_name = "CPU Preempt Wait (explicit preempt)"
                    q_cat = "request_queue_preempt"
                    q_color = "terrible"
                elif scheduled_from == "enqueue":
                    q_name = "CPU Sched Wait (enqueue→out_rpc)"
                    q_cat = "request_queue_scheduled_enqueue"
                    q_color = "yellow"
                elif scheduled_from == "step_ready":
                    q_name = "CPU Sched Wait (step_ready→out_rpc)"
                    q_cat = "request_queue_scheduled_step_ready"
                    q_color = "yellow"
                else:
                    q_name = "CPU Sched Wait"
                    q_cat = "request_queue"
                    q_color = "yellow"
                trace_events.append(create_perfetto_event(
                    name=q_name,
                    cat=q_cat,
                    ph="X",
                    ts=q["start_ns"],
                    dur=q["end_ns"] - q["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["vllm_queue"],
                    args={
                        "request_id": rid,
                        "reason": q["reason"],
                        "scheduled_from": scheduled_from,
                        "gap_kind": gap_kind,
                        "preempt_ts_ns": q.get("preempt_ts_ns", []),
                        "prev_dispatch_key": q.get("prev_dispatch_key"),
                        "next_dispatch_key": q.get("next_dispatch_key"),
                    },
                    cname=q_color,
                ))

        for o in request_index.req_os_intervals_wall.get(rid, []):
            if o["end_ns"] <= o["start_ns"]:
                continue
            trace_events.append(create_perfetto_event(
                name="OS Sched Delay",
                cat="request_os",
                ph="X",
                ts=o["start_ns"],
                dur=o["end_ns"] - o["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids["os"],
                args={
                    "request_id": rid,
                    "worker_tid": o.get("worker_tid"),
                    "pp_rank": o.get("pp_rank"),
                    "tp_rank": o.get("tp_rank"),
                },
                cname="terrible",
            ))

        for s in request_stages.req_stage_intervals.get(rid, []):
            if s["stage"] != "engine_core.input_queue_wait_after_preprocess":
                continue
            trace_events.append(create_perfetto_event(
                name="EngineCore Input Queue Wait",
                cat="request_engine_core_input_queue",
                ph="X",
                ts=s["start_ns"],
                dur=s["end_ns"] - s["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids["stage"],
                args={
                    "request_id": rid,
                    "stage": s["stage"],
                    "duration_ms": round((s["end_ns"] - s["start_ns"]) / 1e6, 3),
                },
                cname="yellow",
            ))

    _worker_os_events = defaultdict(lambda: {"pp_rank": None, "tp_rank": None, "req_ids": set()})

    for rid in sorted(request_index.all_req_ids, key=req_sort_key):
        for o in request_index.req_os_intervals_wall.get(rid, []):
            if o["end_ns"] <= o["start_ns"]:
                continue
            worker_pid = o.get("worker_pid")
            worker_tid = o.get("worker_tid")
            if worker_pid is not None and worker_tid is not None:
                dedup_key = (o["start_ns"], o["end_ns"], worker_pid, worker_tid)
                _worker_os_events[dedup_key]["pp_rank"] = o.get("pp_rank")
                _worker_os_events[dedup_key]["tp_rank"] = o.get("tp_rank")
                _worker_os_events[dedup_key]["req_ids"].add(rid)

    # 画出去重后的 worker-lane OS delay
    for (start_ns, end_ns, wpid, wtid), meta in _worker_os_events.items():
        pp = meta["pp_rank"]
        tp = meta["tp_rank"]
        req_list = sorted(meta["req_ids"])[:10]
        label = f"OS Sched (pp{pp} tp{tp})"
        trace_events.append(create_perfetto_event(
            name=label,
            cat="worker_os_sched",
            ph="X",
            ts=start_ns,
            dur=end_ns - start_ns,
            pid=wpid,
            tid=wtid,
            args={
                "request_ids": ",".join(req_list),
                "request_count": len(meta["req_ids"]),
                "pp_rank": pp,
                "tp_rank": tp,
            },
            cname="terrible",
        ))

    print("=== Request Interference Summary ===")
    for rid in sorted(
        request_index.all_req_ids,
        key=lambda x: (
            request_timing.req_coro_sched_ns.get(x, 0)
            + request_timing.req_generate_task_sched_ns.get(x, 0)
            + request_timing.req_generate_exec_ns.get(x, 0)
            + request_timing.req_output_socket_sched_ns.get(x, 0)
            + request_timing.req_output_socket_exec_ns.get(x, 0)
            + request_timing.req_output_handler_sched_ns.get(x, 0)
            + request_timing.req_output_handler_exec_ns.get(x, 0)
            + dispatch_analysis.req_vllm_queue_ns.get(x, 0)
            + dispatch_analysis.req_gpu_queue_ns.get(x, 0)
            + _request_pp_gap_wall_ns(x)
            + os_result.request_delay_ns.get(x, 0)
        ),
        reverse=True,
    ):
        lifecycle_ms = None
        if rid in request_timing.request_spans:
            lifecycle_ms = (request_timing.request_spans[rid]["end_ns"] - request_timing.request_spans[rid]["start_ns"]) / 1e6
        pp_gap_ms = _request_pp_gap_wall_ns(rid) / 1e6
        print(
            f"  - {rid}: "
            f"lifecycle={f'{lifecycle_ms:.3f} ms' if lifecycle_ms is not None else 'N/A'}, "
            f"coro_sched_queue={request_timing.req_coro_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"generate_task_queue={request_timing.req_generate_task_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"generate_exec={request_timing.req_generate_exec_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_socket_sched_queue={request_timing.req_output_socket_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_socket_exec={request_timing.req_output_socket_exec_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_handler_sched_queue={request_timing.req_output_handler_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_handler_exec={request_timing.req_output_handler_exec_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"cpu_sched_queue={dispatch_analysis.req_vllm_queue_ns.get(rid, 0) / 1e6:.3f} ms "
            f"(scheduled-enqueue={dispatch_analysis.req_vllm_queue_from_enqueue_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"scheduled-step_ready={dispatch_analysis.req_vllm_queue_from_step_ready_ns.get(rid, 0) / 1e6:.3f} ms), "
            f"gpu_queue={dispatch_analysis.req_gpu_queue_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"pp_gap={pp_gap_ms:.3f} ms, "
            f"os_delay={os_result.request_delay_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"prefill_exec={dispatch_analysis.req_dispatch_phase_ns.get(rid, {}).get('prefill', 0) / 1e6:.3f} ms, "
            f"decode_exec={dispatch_analysis.req_dispatch_phase_ns.get(rid, {}).get('decode', 0) / 1e6:.3f} ms"
        )


    return sorted_req_ids
