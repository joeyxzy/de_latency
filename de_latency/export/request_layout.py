"""Request lanes, dispatch display intervals and alignment debug tracks."""
from de_latency.analysis.clocks import to_ns

def emit_thread_name_once(trace_events, emitted_keys, pid, tid, name):
    key = (pid, tid, name)
    if key in emitted_keys:
        return
    emitted_keys.add(key)
    trace_events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "args": {"name": name},
    })

def build_request_lane_tids(req_index):
    # Keep queue/exec producers on dedicated lanes to avoid Perfetto hiding
    # overlapping slices on the same request track.
    base = req_index * 100
    return {
        "label": base + 1,
        "lifecycle": base + 2,
        "vllm_queue": base + 3,
        "coro_queue": base + 4,
        "output_socket_queue": base + 5,
        "output_queue": base + 6,
        "dispatch": base + 7,
        "generate_exec": base + 8,
        "output_socket_exec": base + 9,
        "output_exec": base + 10,
        "os": base + 11,
        "stage": base + 12,
        "task": base + 13,
        "pp_gap": base + 14,
        "gpu_queue": base + 15,
    }

def build_raw_dispatch_display_intervals(intervals):
    """Normalize per-dispatch request intervals for request-lane display."""
    display = []
    for interval in intervals:
        start_ns = to_ns(interval.get("start_ns"))
        end_ns = to_ns(interval.get("end_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        item = dict(interval)
        item["start_ns"] = start_ns
        item["end_ns"] = end_ns
        item["duration_ns"] = end_ns - start_ns
        item["phase"] = item.get("phase") or "unknown"
        item["step_count"] = 1
        dispatch_key = item.get("dispatch_key")
        item["dispatch_keys"] = [dispatch_key] if dispatch_key else []
        item["dispatch_key_count"] = 1 if dispatch_key else 0
        display.append(item)
    return sorted(display, key=lambda item: (item["start_ns"], item["end_ns"], item["phase"]))

def build_alignment_debug_trace(worker_spans, ebpf_events):
    """
    生成一个只包含 Worker 执行区间(Green) 和 eBPF 调度事件(Red) 的 Trace。
    用于肉眼 Debug 时间对齐问题。
    """
    
    trace_events = []
    
    # 1. 添加 Worker Spans (绿色条)
    for item in worker_spans:
        # 兼容处理：有些 payload 在 item 里，有些 item 本身就是 payload
        p = item.get('payload', item)
        
        pid = p.get('pid', 0)
        tid = p.get('tid', 0)
        start = p.get('start_ns')
        end = p.get('end_ns')
        
        if start and end and tid:
            trace_events.append({
                "name": "Worker Span (Python)",
                "cat": "debug_worker",
                "ph": "X",
                "ts": start / 1000.0,       # ns -> us
                "dur": (end - start) / 1000.0,
                "pid": pid,
                "tid": tid,
                "cname": "good",            # 绿色
                "args": {"req_ids": p.get("req_ids")}
            })

    # 2. 添加 eBPF Events (红色条)
    # 为了防止 eBPF 事件太短看不清，或者为了方便对比，
    # 我们将 eBPF 事件放在同一个 PID/TID 轨道上，或者放在一个专门的 Debug 轨道
    for item in ebpf_events:
        p = item.get('payload', item)
        
        tid = p.get('tid', 0)
        start = p.get('start_ns')
        end = p.get('end_ns')
        dur = p.get('dur_us', 0)
        
        # eBPF 数据通常没有 PID，为了让它能在 Perfetto 里跟 Worker 显示在一起，
        # 我们尝试"借用" Worker 的 PID。这里我们做一个特殊的处理：
        # 如果我们不知道 PID，就设为 TID (Perfetto 会把它们分到不同组，但至少能看见)
        # 或者设为 0。为了对比，最好让它们在同一轨道。
        
        if start and end and tid:
            trace_events.append({
                "name": "OS Latency (eBPF)",
                "cat": "debug_ebpf",
                "ph": "X",
                "ts": start / 1000.0,       # ns -> us
                "dur": (end - start) / 1000.0,
                "pid": 0,                   # 故意设为0 或者一个特定值，方便在最顶层看到
                # "pid": p.get('pid', tid), # 如果你想让它和 Worker 挤在一起，解开这行
                "tid": tid,                 # 关键：TID 必须一致
                "cname": "terrible",        # 红色
                "args": {"reason": p.get("reason")}
            })

    # 3. 输出
    # 按时间排序
    trace_events.sort(key=lambda x: x['ts'])
    
    return trace_events

