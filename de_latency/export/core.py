"""EngineCore and scheduler trace tracks."""
from de_latency.analysis.clocks import to_ns

def append_core_trace(log_data, trace_events, writer):
    create_perfetto_event = writer.event
    print(f"Processing EngineCore Main Loop Events ({len(log_data.enginecore_loop_events)})...")
    if log_data.enginecore_loop_events:
        ENGINECORE_PID = 12000
        trace_events.append({"name": "process_name", "ph": "M", "pid": ENGINECORE_PID, "args": {"name": "EngineCore Main Loop"}})
        tid_alias_map = {}
        next_tid_alias = 1
        for item in sorted(log_data.enginecore_loop_events, key=lambda x: x.get("ts") or 0):
            payload = item.get("payload", {})
            phase = payload.get("phase", "unknown")
            start_ns = to_ns(payload.get("start_ns"))
            end_ns = to_ns(payload.get("end_ns"))
            pid = payload.get("pid")
            tid = payload.get("tid")
            if start_ns is None or end_ns is None or end_ns <= start_ns:
                continue

            key = (pid, tid)
            if key not in tid_alias_map:
                tid_alias_map[key] = next_tid_alias
                trace_events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": ENGINECORE_PID,
                    "tid": next_tid_alias,
                    "args": {"name": f"pid={pid} tid={tid}"},
                })
                next_tid_alias += 1
            tid_alias = tid_alias_map[key]

            display_name = "EngineCore: Input Queue Phase" if phase == "_process_input_queue" else "EngineCore: Engine Step Phase"
            trace_events.append(create_perfetto_event(
                name=display_name,
                cat="enginecore_loop",
                ph="X",
                ts=start_ns,
                dur=end_ns - start_ns,
                pid=ENGINECORE_PID,
                tid=tid_alias,
                args={
                    "phase": phase,
                    "source_pid": pid,
                    "source_tid": tid,
                    "duration_ms": round((end_ns - start_ns) / 1e6, 3),
                },
                cname="rail_idle" if phase == "_process_input_queue" else "rail_response",
            ))

    # --- 4.6b Scheduler Latency 可视化（dispatch_key 配对） ---
    SCHEDULER_PID = 13000

    schedule_by_dk = {}
    for ev in log_data.scheduler_schedule_events:
        p = ev.get("payload", {})
        sn = to_ns(p.get("start_ns"))
        en = to_ns(p.get("end_ns"))
        dk = p.get("dispatch_key")
        if dk and sn is not None and en is not None and en > sn:
            schedule_by_dk[dk] = {
                "start_ns": sn,
                "end_ns": en,
                "batch_size": p.get("scheduled_batch_size"),
                "num_requests": p.get("num_requests"),
                "num_running": p.get("num_running"),
                "num_waiting": p.get("num_waiting"),
            }

    exec_by_dk = {}
    for item in log_data.scheduler_events:
        if item["type"] != "req_scheduler_out_rpc":
            continue
        p = item["payload"]
        dk = p.get("dispatch_key")
        ts = to_ns(p.get("timestamp_ns"))
        if dk and ts is not None:
            exec_by_dk[dk] = ts

    sample_by_dk = {}
    for ev in log_data.scheduler_sample_tokens_events:
        p = ev.get("payload", {})
        dk = p.get("dispatch_key")
        ts = to_ns(p.get("timestamp_ns"))
        if dk and ts is not None:
            sample_by_dk[dk] = ts

    step_ready_by_dk = {}
    for item in log_data.scheduler_events:
        if item["type"] != "req_step_ready":
            continue
        p = item["payload"]
        dk = p.get("dispatch_key")
        before_ns = to_ns(p.get("scheduler_before_ns"))
        after_ns = to_ns(p.get("timestamp_ns"))
        if dk and before_ns is not None and after_ns is not None and after_ns > before_ns:
            step_ready_by_dk[dk] = {
                "before_ns": before_ns,
                "after_ns": after_ns,
            }

    if schedule_by_dk:
        trace_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": SCHEDULER_PID,
            "args": {"name": "Scheduler Latency"},
        })
        trace_events.append({
            "name": "thread_name", "ph": "M", "pid": SCHEDULER_PID, "tid": 1,
            "args": {"name": "Schedule"},
        })
        trace_events.append({
            "name": "thread_name", "ph": "M", "pid": SCHEDULER_PID, "tid": 2,
            "args": {"name": "Execute Model Out"},
        })
        trace_events.append({
            "name": "thread_name", "ph": "M", "pid": SCHEDULER_PID, "tid": 3,
            "args": {"name": "Sample Tokens Out"},
        })
        trace_events.append({
            "name": "thread_name", "ph": "M", "pid": SCHEDULER_PID, "tid": 4,
            "args": {"name": "Update Output"},
        })

        sched_flow_id_base = 2_000_000_000
        sched_flow_counter = 0

        for dk, sched in sorted(schedule_by_dk.items(),
                                key=lambda x: x[1]["start_ns"]):
            # --- Schedule bar (tid=1) ---
            trace_events.append(create_perfetto_event(
                name="Schedule",
                cat="scheduler_schedule",
                ph="X", ts=sched["start_ns"],
                dur=sched["end_ns"] - sched["start_ns"],
                pid=SCHEDULER_PID, tid=1,
                args={
                    "dispatch_key": dk,
                    "batch_size": sched.get("batch_size"),
                    "num_requests": sched.get("num_requests"),
                    "num_running": sched.get("num_running"),
                    "num_waiting": sched.get("num_waiting"),
                    "duration_ms": round((sched["end_ns"] - sched["start_ns"]) / 1e6, 3),
                },
                cname="rail_response",
            ))

            # --- Execute Model Out instant point (tid=2) ---
            exec_ts = exec_by_dk.get(dk)
            if exec_ts is not None:
                trace_events.append({
                    "name": "Execute Model",
                    "cat": "scheduler_exec_out",
                    "ph": "i", "ts": (exec_ts - log_data.TRACE_TS_ORIGIN_NS) // 1000,
                    "pid": SCHEDULER_PID, "tid": 2,
                    "s": "t",
                    "args": {"dispatch_key": dk},
                })

            # --- Sample Tokens Out instant point (tid=3) ---
            sample_ts = sample_by_dk.get(dk)
            if sample_ts is not None:
                trace_events.append({
                    "name": "Sample Tokens",
                    "cat": "scheduler_sample_out",
                    "ph": "i", "ts": (sample_ts - log_data.TRACE_TS_ORIGIN_NS) // 1000,
                    "pid": SCHEDULER_PID, "tid": 3,
                    "s": "t",
                    "args": {"dispatch_key": dk},
                })

            # --- Update Output bar (tid=4) ---
            sr = step_ready_by_dk.get(dk)
            if not sr:
                continue
            trace_events.append(create_perfetto_event(
                name="Update Output",
                cat="scheduler_update_output",
                ph="X", ts=sr["before_ns"],
                dur=sr["after_ns"] - sr["before_ns"],
                pid=SCHEDULER_PID, tid=4,
                args={
                    "dispatch_key": dk,
                    "duration_ms": round((sr["after_ns"] - sr["before_ns"]) / 1e6, 3),
                },
                cname="olive",
            ))

            # --- Flow arrows: schedule → exec → sample → update ---
            flow_points = [(1, sched["end_ns"])]
            if exec_ts is not None:
                flow_points.append((2, exec_ts))
            if sample_ts is not None:
                flow_points.append((3, sample_ts))
            flow_points.append((4, sr["before_ns"]))
            flow_points.sort(key=lambda x: x[1])

            for i in range(len(flow_points) - 1):
                sched_flow_counter += 1
                flow_id = sched_flow_id_base + sched_flow_counter
                src_tid, src_ts_ns = flow_points[i]
                dst_tid, dst_ts_ns = flow_points[i + 1]
                trace_events.append({
                    "name": "sched_flow",
                    "cat": "flow",
                    "ph": "s", "pid": SCHEDULER_PID, "tid": src_tid,
                    "ts": (src_ts_ns - log_data.TRACE_TS_ORIGIN_NS) // 1000,
                    "id": flow_id,
                })
                flow_end = {
                    "name": "sched_flow",
                    "cat": "flow",
                    "ph": "f", "pid": SCHEDULER_PID, "tid": dst_tid,
                    "ts": (dst_ts_ns - log_data.TRACE_TS_ORIGIN_NS) // 1000,
                    "id": flow_id,
                }
                if dst_tid != 4:
                    flow_end["bp"] = "e"
                trace_events.append(flow_end)
