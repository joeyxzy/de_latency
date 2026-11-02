"""Request lifecycle, coroutine and output-task interval attribution."""
from .ingestion import LogData
from collections import defaultdict, deque
from de_latency.analysis.clocks import to_ns, to_int
from de_latency.analysis.requests import normalize_request_id, normalize_request_ids, normalize_req_event_type
from dataclasses import dataclass

@dataclass
class RequestTiming:
    phase_hint_queue_by_req: dict
    req_coro_sched_intervals: dict
    req_coro_sched_ns: dict
    req_generate_exec_intervals: dict
    req_generate_exec_ns: dict
    req_generate_start_map: dict
    req_generate_task_sched_intervals: dict
    req_generate_task_sched_ns: dict
    req_output_handler_exec_intervals: dict
    req_output_handler_exec_ns: dict
    req_output_handler_sched_intervals: dict
    req_output_handler_sched_ns: dict
    req_output_socket_exec_intervals: dict
    req_output_socket_exec_ns: dict
    req_output_socket_sched_intervals: dict
    req_output_socket_sched_ns: dict
    req_preempt_ts_by_req: dict
    request_spans: dict


def analyze_request_timing(log_data: LogData) -> RequestTiming:
    print(f"Processing Coroutine Lifecycle Events ({len(log_data.coroutine_events)})...")
    log_data.coroutine_events.sort(key=lambda x: x.get('ts') or 0)

    active_coroutines = {}
    request_spans = {}
    req_generate_start_map = {}
    req_coro_sched_ns = defaultdict(int)
    req_coro_sched_intervals = defaultdict(list)
    req_generate_task_sched_ns = defaultdict(int)
    req_generate_task_sched_intervals = defaultdict(list)
    req_generate_exec_ns = defaultdict(int)
    req_generate_exec_intervals = defaultdict(list)
    req_output_socket_sched_ns = defaultdict(int)
    req_output_socket_sched_intervals = defaultdict(list)
    req_output_socket_exec_ns = defaultdict(int)
    req_output_socket_exec_intervals = defaultdict(list)
    req_output_handler_sched_ns = defaultdict(int)
    req_output_handler_sched_intervals = defaultdict(list)
    req_output_handler_exec_ns = defaultdict(int)
    req_output_handler_exec_intervals = defaultdict(list)

    for ev in log_data.coroutine_events:
        etype = ev['type']
        payload = ev.get('payload', {})
        rid = payload.get('request_id') or payload.get('req_id')
        cid = normalize_request_id(payload.get('coroutine_id'))
        ts_ns = payload.get('timestamp_ns', ev.get('ts'))

        if etype == 'coroutine_sched_latency':
            ready_ts_ns = to_int(payload.get('ready_ts_ns'))
            run_ts_ns = to_int(payload.get('run_ts_ns'))
            queue_ns = to_int(payload.get('queue_ns'))
            task_kind = payload.get("task_kind")
            if task_kind == "generate_task":
                if rid and queue_ns is not None and queue_ns >= 0:
                    req_generate_task_sched_ns[rid] += queue_ns
                if rid and ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                    req_generate_task_sched_intervals[rid].append({
                        "start_ns": ready_ts_ns,
                        "end_ns": run_ts_ns,
                    })
            else:
                if rid and queue_ns is not None and queue_ns >= 0:
                    req_coro_sched_ns[rid] += queue_ns
                if rid and ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                    req_coro_sched_intervals[rid].append({
                        "start_ns": ready_ts_ns,
                        "end_ns": run_ts_ns,
                    })
            continue

        if etype == 'coroutine_start' and rid and cid and ts_ns:
            prev_start = req_generate_start_map.get(rid)
            if prev_start is None or ts_ns < prev_start:
                req_generate_start_map[rid] = ts_ns
            active_coroutines[cid] = {
                "request_id": rid,
                "start_ns": ts_ns,
                "pid": payload.get('pid', 0),
                "tid": payload.get('tid', 0),
            }
            continue

        if etype == 'coroutine_end' and rid and ts_ns:
            start_info = active_coroutines.pop(cid, None) if cid else None
            if start_info:
                start_ns = start_info['start_ns']
                pid = start_info['pid']
                tid = start_info['tid']
            else:
                duration_ms = payload.get('duration_ms')
                duration_ns = int(duration_ms * 1e6) if duration_ms else 0
                start_ns = ts_ns - duration_ns if duration_ns > 0 else ts_ns
                pid = payload.get('pid', 0)
                tid = payload.get('tid', 0)

            if ts_ns > start_ns:
                if rid not in request_spans:
                    request_spans[rid] = {
                        "start_ns": start_ns,
                        "end_ns": ts_ns,
                        "pid": pid,
                        "tid": tid,
                    }
                else:
                    request_spans[rid]["start_ns"] = min(request_spans[rid]["start_ns"], start_ns)
                    request_spans[rid]["end_ns"] = max(request_spans[rid]["end_ns"], ts_ns)

    if req_coro_sched_ns:
        print(f"共统计到 {len(req_coro_sched_ns)} 个请求的协程调度排队时间：")
        for rid, lat_ns in sorted(req_coro_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Coroutine Scheduler Queue)")
    else:
        print("  未统计到协程调度排队时间")

    if req_generate_task_sched_ns:
        print(f"共统计到 {len(req_generate_task_sched_ns)} 个请求的 generate task 运行队列时间：")
        for rid, lat_ns in sorted(req_generate_task_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Generate Task Runnable Queue)")
    else:
        print("  未统计到 generate task 运行队列时间")

    if log_data.coroutine_exec_events:
        print(f"Processing Generate Coroutine Exec Events ({len(log_data.coroutine_exec_events)})...")
    for ev in log_data.coroutine_exec_events:
        payload = ev.get("payload", {})
        rid = payload.get("request_id") or payload.get("req_id")
        if not rid:
            continue
        start_ns = to_int(payload.get("start_ns"))
        end_ns = to_int(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        if dur_ns is None and start_ns is not None and end_ns is not None:
            dur_ns = end_ns - start_ns
        if dur_ns is None or dur_ns < 0:
            continue
        req_generate_exec_ns[rid] += dur_ns
        if start_ns and end_ns and end_ns > start_ns:
            req_generate_exec_intervals[rid].append({
                "start_ns": start_ns,
                "end_ns": end_ns,
                "task_name": payload.get("task_name"),
                "task_kind": payload.get("task_kind"),
            })

    if req_generate_exec_ns:
        print(f"共统计到 {len(req_generate_exec_ns)} 个请求的 generate 协程执行时间：")
        for rid, lat_ns in sorted(req_generate_exec_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Generate Coroutine Exec)")
    elif log_data.coroutine_exec_events:
        print("  generate 协程执行事件存在，但未找到可归因的 request_id")

    if log_data.output_socket_sched_events:
        print(f"Processing Output Socket Scheduler Events ({len(log_data.output_socket_sched_events)})...")
    for ev in log_data.output_socket_sched_events:
        payload = ev.get("payload", {})
        ready_ts_ns = to_int(payload.get("ready_ts_ns"))
        run_ts_ns = to_int(payload.get("run_ts_ns"))
        queue_ns = to_int(payload.get("queue_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if not req_ids or queue_ns is None or queue_ns < 0:
            continue
        for rid in req_ids:
            req_output_socket_sched_ns[rid] += queue_ns
            if ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                req_output_socket_sched_intervals[rid].append({
                    "start_ns": ready_ts_ns,
                    "end_ns": run_ts_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_socket_sched_ns:
        print(f"共统计到 {len(req_output_socket_sched_ns)} 个请求的 output_socket 调度排队时间：")
        for rid, lat_ns in sorted(req_output_socket_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Socket Scheduler Queue)")
    elif log_data.output_socket_sched_events:
        print("  output_socket 调度事件存在，但未找到可归因的 req_ids")

    if log_data.output_socket_exec_events:
        print(f"Processing Output Socket Exec Events ({len(log_data.output_socket_exec_events)})...")
    for ev in log_data.output_socket_exec_events:
        payload = ev.get("payload", {})
        start_ns = to_int(payload.get("start_ns"))
        end_ns = to_int(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if dur_ns is None and start_ns is not None and end_ns is not None:
            dur_ns = end_ns - start_ns
        if not req_ids or dur_ns is None or dur_ns < 0:
            continue
        for rid in req_ids:
            req_output_socket_exec_ns[rid] += dur_ns
            if start_ns and end_ns and end_ns > start_ns:
                req_output_socket_exec_intervals[rid].append({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_socket_exec_ns:
        print(f"共统计到 {len(req_output_socket_exec_ns)} 个请求的 output_socket 执行时间：")
        for rid, lat_ns in sorted(req_output_socket_exec_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Socket Exec)")
    elif log_data.output_socket_exec_events:
        print("  output_socket 执行事件存在，但未找到可归因的 req_ids")

    if log_data.output_handler_sched_events:
        print(f"Processing Output Handler Scheduler Events ({len(log_data.output_handler_sched_events)})...")
    for ev in log_data.output_handler_sched_events:
        payload = ev.get("payload", {})
        ready_ts_ns = to_int(payload.get("ready_ts_ns"))
        run_ts_ns = to_int(payload.get("run_ts_ns"))
        queue_ns = to_int(payload.get("queue_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if not req_ids or queue_ns is None or queue_ns < 0:
            continue
        for rid in req_ids:
            req_output_handler_sched_ns[rid] += queue_ns
            if ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                req_output_handler_sched_intervals[rid].append({
                    "start_ns": ready_ts_ns,
                    "end_ns": run_ts_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_handler_sched_ns:
        print(f"共统计到 {len(req_output_handler_sched_ns)} 个请求的 output_handler 调度排队时间：")
        for rid, lat_ns in sorted(req_output_handler_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Handler Scheduler Queue)")
    elif log_data.output_handler_sched_events:
        print("  output_handler 调度事件存在，但未找到可归因的 req_ids")

    if log_data.output_handler_exec_events:
        print(f"Processing Output Handler Exec Events ({len(log_data.output_handler_exec_events)})...")
    for ev in log_data.output_handler_exec_events:
        payload = ev.get("payload", {})
        start_ns = to_int(payload.get("start_ns"))
        end_ns = to_int(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if dur_ns is None and start_ns is not None and end_ns is not None:
            dur_ns = end_ns - start_ns
        if not req_ids or dur_ns is None or dur_ns < 0:
            continue
        for rid in req_ids:
            req_output_handler_exec_ns[rid] += dur_ns
            if start_ns and end_ns and end_ns > start_ns:
                req_output_handler_exec_intervals[rid].append({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_handler_exec_ns:
        print(f"共统计到 {len(req_output_handler_exec_ns)} 个请求的 output_handler 执行时间：")
        for rid, lat_ns in sorted(req_output_handler_exec_ns.items(), key=lambda x: x[1], reverse=True):
            label = log_data.req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Handler Exec)")
    elif log_data.output_handler_exec_events:
        print("  output_handler 执行事件存在，但未找到可归因的 req_ids")

    # 为 request dispatch phase 做回退兜底：
    # 从 req_metrics_events 中按时间提取每个请求的 "scheduled/dequeue" phase 顺序。
    phase_hint_queue_by_req = defaultdict(deque)
    req_preempt_ts_by_req = defaultdict(list)
    req_metric_events_sorted = sorted(
        log_data.req_metric_events,
        key=lambda x: to_ns(x.get("batch_ts_ns")) or 0,
    )
    for item in req_metric_events_sorted:
        payload = item.get("payload", item)
        updates = payload.get("req_events", [])
        for update in updates:
            rid = update.get("req_id")
            if not rid:
                continue
            for ev in update.get("events", []):
                ev_type = normalize_req_event_type(ev.get("type"))
                ev_ts_ns = to_ns(ev.get("ts") or ev.get("timestamp_ns"))
                if ev_type == "preempt" and ev_ts_ns is not None:
                    req_preempt_ts_by_req[rid].append(ev_ts_ns)
                if ev_type != "dequeue":
                    continue
                phase = ev.get("phase")
                if phase in ("prefill", "decode"):
                    phase_hint_queue_by_req[rid].append(phase)
                break


    return RequestTiming(
        phase_hint_queue_by_req=phase_hint_queue_by_req,
        req_coro_sched_intervals=req_coro_sched_intervals,
        req_coro_sched_ns=req_coro_sched_ns,
        req_generate_exec_intervals=req_generate_exec_intervals,
        req_generate_exec_ns=req_generate_exec_ns,
        req_generate_start_map=req_generate_start_map,
        req_generate_task_sched_intervals=req_generate_task_sched_intervals,
        req_generate_task_sched_ns=req_generate_task_sched_ns,
        req_output_handler_exec_intervals=req_output_handler_exec_intervals,
        req_output_handler_exec_ns=req_output_handler_exec_ns,
        req_output_handler_sched_intervals=req_output_handler_sched_intervals,
        req_output_handler_sched_ns=req_output_handler_sched_ns,
        req_output_socket_exec_intervals=req_output_socket_exec_intervals,
        req_output_socket_exec_ns=req_output_socket_exec_ns,
        req_output_socket_sched_intervals=req_output_socket_sched_intervals,
        req_output_socket_sched_ns=req_output_socket_sched_ns,
        req_preempt_ts_by_req=req_preempt_ts_by_req,
        request_spans=request_spans,
    )
