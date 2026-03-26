import json
import sys
from collections import defaultdict, deque
import os
import math
from decimal import Decimal, InvalidOperation

TRACE_TS_ORIGIN_NS = 0

def load_log_lines(filepath):
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                # 兼容某些情况下 json 前面有杂质的情况
                json_start = line.find('{')
                if json_start != -1:
                    events.append(json.loads(line[json_start:]))
            except json.JSONDecodeError:
                continue
    return events

def create_perfetto_event(name, cat, ph, ts, dur, pid, tid, args=None, cname=None):
    # Perfetto 要求时间单位是微秒 (us)
    event = {
        "name": name, "cat": cat, "ph": ph, "ts": (ts - TRACE_TS_ORIGIN_NS) / 1000.0, 
        "pid": pid, "tid": tid, "args": args or {}
    }
    if dur is not None: 
        event["dur"] = dur / 1000.0
    if cname:
        event["cname"] = cname
    return event

def create_flow_event(ph, ts, pid, tid, corr_id):
    return {
        "name": "link", "cat": "flow", "ph": ph, "ts": (ts - TRACE_TS_ORIGIN_NS) / 1000.0, 
        "pid": pid, "tid": tid, "id": corr_id
    }

def infer_trace_origin_ns(
    worker_events,
    scheduler_events,
    coroutine_events,
    req_stage_events,
    enginecore_loop_events,
    execute_model_span,
):
    """
    推断 wall 时间轴的最小 ns 作为 trace 原点。
    目的是把超大绝对时间戳平移到接近 0，避免浮点在 1e15 us 量级产生 0.25us 量化误差。
    """
    min_ns = None

    def _update(candidate):
        nonlocal min_ns
        ns = to_ns(candidate)
        if ns is None:
            return
        if min_ns is None or ns < min_ns:
            min_ns = ns

    for ev in worker_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in scheduler_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns"))

    for ev in coroutine_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))

    for ev in req_stage_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in enginecore_loop_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for p in execute_model_span:
        _update(p.get("timestamp_ns"))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    return min_ns or 0

def nudge_equal_boundaries(
    trace_events,
    target_pid=None,
    epsilon_us=0.001,
    cat_prefixes=None,
    snap_tolerance_us=0.0,
):
    """
    某些可视化器在同一轨道上遇到“前一个块的 end == 后一个块的 start”时，
    可能出现后一个块不稳定渲染。这里做最小量微调：
    - 仅处理 ph='X' 且 dur>0 的区间事件；
    - 同一 (pid, tid) 轨道上，若 start 与上一个 end 完全相等，则将 start + epsilon，
      同时 dur - epsilon（保持 end 基本不变，且不会改成负数）。
    """
    prefixes = tuple(cat_prefixes or ())
    slots = []
    for idx, ev in enumerate(trace_events):
        if ev.get("ph") != "X":
            continue
        if target_pid is not None and ev.get("pid") != target_pid:
            continue
        if prefixes:
            cat = ev.get("cat", "")
            if not any(str(cat).startswith(pre) for pre in prefixes):
                continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if ts is None or dur is None:
            continue
        try:
            ts = float(ts)
            dur = float(dur)
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        slots.append((ev.get("pid"), ev.get("tid"), ts, ts + dur, idx))

    slots.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    last_end_by_track = {}
    nudged = 0

    for pid, tid, start, _end, idx in slots:
        track = (pid, tid)
        prev_end = last_end_by_track.get(track)
        if prev_end is not None:
            gap = start - prev_end
        else:
            gap = None

        # 两类情况都做修正：
        # 1) 完全相等：start == prev_end
        # 2) 近似贴边/微小重叠：|start - prev_end| <= snap_tolerance_us
        need_nudge = (
            prev_end is not None
            and (
                start == prev_end
                or (
                    snap_tolerance_us > 0
                    and gap is not None
                    and -snap_tolerance_us <= gap <= snap_tolerance_us
                )
            )
        )

        if need_nudge:
            ev = trace_events[idx]
            old_dur = float(ev["dur"])
            # 只做最小修正，保持事件仍为正时长
            target_start = prev_end + epsilon_us
            delta = target_start - start
            if delta > 0 and old_dur > delta:
                ev["ts"] = start + delta
                ev["dur"] = old_dur - delta
                nudged += 1
                start = float(ev["ts"])
                _end = start + float(ev["dur"])
        last_end_by_track[track] = _end

    return nudged

def to_ns(raw_ts):
    """
    统一时间戳到 ns。
    - 若原值是大整数（>1e14），认为已是 ns。
    - 否则按秒（float）转 ns（兼容 EngineCoreEvent.timestamp）。
    """
    if raw_ts is None:
        return None

    # 优先走整数通道，避免 1e18 量级 ns 经 float 造成 256ns 级精度丢失。
    if isinstance(raw_ts, bool):
        return None
    if isinstance(raw_ts, int):
        val_int = raw_ts
        if val_int <= 0:
            return None
        if val_int > 1e14:
            return val_int
        return int(val_int * 1e9)

    if isinstance(raw_ts, str):
        s = raw_ts.strip()
        if not s:
            return None
        try:
            if s.isdigit() or (s[0] in "+-" and s[1:].isdigit()):
                val_int = int(s)
                if val_int <= 0:
                    return None
                if val_int > 1e14:
                    return val_int
                return int(val_int * 1e9)

            val_dec = Decimal(s)
            if not val_dec.is_finite() or val_dec <= 0:
                return None
            if val_dec > Decimal("1e14"):
                return int(val_dec)
            return int(val_dec * Decimal("1e9"))
        except (ValueError, InvalidOperation):
            return None

    if isinstance(raw_ts, float):
        if not math.isfinite(raw_ts) or raw_ts <= 0:
            return None
        if raw_ts > 1e14:
            return int(raw_ts)
        return int(raw_ts * 1e9)

    try:
        val_int = int(raw_ts)
        if val_int <= 0:
            return None
        if val_int > 1e14:
            return val_int
        return int(val_int * 1e9)
    except (TypeError, ValueError):
        return None

def to_int(raw_val):
    if raw_val is None:
        return None
    try:
        return int(raw_val)
    except (TypeError, ValueError):
        return None

def normalize_req_event_type(raw_type):
    if raw_type is None:
        return None
    s = str(raw_type).strip()
    up = s.upper()

    if s == "2" or "DEQUEUE" in up or "SCHEDULED" in up or "DISPATCH" in up:
        return "dequeue"
    if s == "3" or "PREEMPT" in up:
        return "preempt"
    if s == "1" or "QUEUE" in up or "ENQUEUE" in up:
        return "enqueue"
    return None

def median_int(values):
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) // 2

#DEBUG: 调试OS调度层的时间对齐问题（将ebpf和execute_model_span的区间在perfetto上画出来）
def export_alignment_debug_trace(worker_spans, ebpf_events, output_filename="debug_align.json"):
    """
    生成一个只包含 Worker 执行区间(Green) 和 eBPF 调度事件(Red) 的 Trace。
    用于肉眼 Debug 时间对齐问题。
    """
    print(f"\n🔍 Generating Debug Trace: {output_filename} ...")
    
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
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({"traceEvents": trace_events}, f)
    
    print(f"✅ Debug Trace Saved: {output_filename}")

def process_logs(input_file, output_file):
    print(f"🔄 Loading logs from {input_file}...")
    raw_events = load_log_lines(input_file)
    trace_events = []

    # --- 1. 数据清洗与分类 ---
    # 我们将所有事件分为三类：Worker生命周期、Scheduler、CUPTI
    worker_events = []
    scheduler_events = []
    cupti_events = []
    req_metric_events = []
    coroutine_events = []
    req_stage_events = []
    enginecore_loop_events = []
    execute_model_span = []
    ebpf_sched_latency_events = []
    req_name_map = {}
    
    # 辅助字典：存储 Kernel 信息
    all_kernels_map = {}
    eager_kernels = []

    for entry in raw_events:
        # 处理可能的 TraceSender 格式差异
        if 'meta' in entry:
            # 如果是 ZeroMQ 发送的原始格式
            meta = entry.get('meta', {})
            payload = meta.get('payload', {})
            src = meta.get('source')
            etype = meta.get('event_type')
            ts = meta.get('timestamp_ns')
        else:
            # 如果是打平后的格式
            payload = entry.get('payload', entry)
            src = entry.get('source', 'monkey_patch')
            etype = entry.get('event_type')
            ts = entry.get('timestamp_ns', payload.get('timestamp_ns'))

        if not payload: continue
        rid = payload.get("request_id") or payload.get("req_id")
        rname = payload.get("request_name")
        if rid and rname:
            req_name_map[rid] = rname
        # 兜底：确保 payload 里有 timestamp_ns
        if 'timestamp_ns' not in payload and ts:
            payload['timestamp_ns'] = ts
        
        # 修正：你的代码中 gpu_forward 的 payload 包含 start_ns/end_ns，但 event_type 叫 'gpu_forward'
        # 我们统一整理到 worker_events
        if src == 'monkey_patch':
            # 新增事件类型支持
            if etype in ['worker_preprocess_start', 'gpu_forward_start', 'gpu_forward_end', 
                         'gpu_sample', 'gpu_bookkeeping', 'gpu_execute_model']:
                worker_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('start_ns', ts))
                })
            elif etype in ['req_enqueue_scheduler', 'req_scheduler_out_rpc', 'req_step_ready']:
                scheduler_events.append({"type": etype, "payload": payload})
            elif etype=="req_metrics_events":
                req_metric_events.append({
                    "payload": payload,
                    "batch_ts_ns": ts
                })
            elif etype in ['coroutine_start', 'coroutine_end', 'coroutine_sched_latency']:
                coroutine_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', ts)
                })
            elif etype == "req_generate_stage":
                req_stage_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', ts),
                })
            elif etype == "enginecore_mainloop_span":
                enginecore_loop_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype=="worker_model_execute_span":
                execute_model_span.append(payload)

        elif src == 'CUPTI':
            corr_id = payload.get('correlationId')
            if etype in ['runtime', 'driver']:
                cupti_events.append(payload)
            elif etype in ['kernel', 'memset', 'memcpy']:
                if corr_id not in all_kernels_map:
                    all_kernels_map[corr_id] = []
                all_kernels_map[corr_id].append(payload)
                if payload.get('start_ns', 0) > 0:
                    eager_kernels.append(payload)

        elif src == 'ebpf':
            if etype == 'sched_latency':
                ebpf_sched_latency_events.append(payload)

    global TRACE_TS_ORIGIN_NS
    TRACE_TS_ORIGIN_NS = infer_trace_origin_ns(
        worker_events=worker_events,
        scheduler_events=scheduler_events,
        coroutine_events=coroutine_events,
        req_stage_events=req_stage_events,
        enginecore_loop_events=enginecore_loop_events,
        execute_model_span=execute_model_span,
    )
    if TRACE_TS_ORIGIN_NS:
        print(f"Trace time origin (wall ns): {TRACE_TS_ORIGIN_NS}")

    # --- 2. Worker 状态机 (支持细粒度阶段) ---
    print(f"Processing Worker Events ({len(worker_events)})...")
    
    worker_events.sort(key=lambda x: x['ts'])

    # 状态存储: Key=(pid, tid), Value={
    #   't_pre_start': ..., 
    #   't_fwd_start': ..., 
    #   't_fwd_end': ..., 
    #   'req_ids': [...],
    #   'batch_size': ..., 'input_type': ...
    # }
    worker_states = {}
    generated_dispatch_slices = []  # 仍可用于 CUPTI 关联（用整个 execute_model 区间）

    for ev in worker_events:
        etype = ev['type']
        payload = ev['payload']
        ts = ev['ts']
        pid = payload.get('pid')
        tid = payload.get('tid')
        key = (pid, tid)

        # ----------------------------------------------------------------
        # [Step 0] 整体 execute_model 区间（容器）
        # ----------------------------------------------------------------
        if etype == 'gpu_execute_model':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                req_ids = payload.get('req_ids', [])
                batch_size = payload.get('batch_size', 0)
                input_type = payload.get('input_type', 'unknown')
                
                # 记录完整区间（用于 CUPTI 关联）
                generated_dispatch_slices.append({
                    'pid': pid, 'tid': tid,
                    'start': t_start, 'end': t_end,
                    'req_ids': req_ids
                })
                
                # 画一个透明容器（浅灰色背景）
                trace_events.append(create_perfetto_event(
                    name="Step Execution", cat="worker", ph="X",
                    ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(req_ids), "batch_size": batch_size},
                    cname="background"
                ))
            continue  # 不进入状态机流转

        # ----------------------------------------------------------------
        # [Step 1] Preprocess Start
        # ----------------------------------------------------------------
        if etype == 'worker_preprocess_start':
            worker_states[key] = {
                't_pre_start': ts,
                'req_ids': payload.get('req_ids', []),
                'batch_size': payload.get('batch_size'),
                'input_type': payload.get('input_type', 'unknown')
            }

        # ----------------------------------------------------------------
        # [Step 2] Forward Start
        # ----------------------------------------------------------------
        elif etype == 'gpu_forward_start':
            state = worker_states.get(key)
            if state and 't_pre_start' in state:
                t1 = state['t_pre_start']
                dur = max(0, ts - t1 - 1000)  # 留白
                trace_events.append(create_perfetto_event(
                    name="Worker Preprocess",
                    cat="python", ph="X", ts=t1, dur=dur,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(state['req_ids']), "desc": "Prepare Inputs"},
                    cname="good"  # 黄绿色
                ))
                state['t_fwd_start'] = ts

        # ----------------------------------------------------------------
        # [Step 3] Forward End
        # ----------------------------------------------------------------
        elif etype == 'gpu_forward_end':
            state = worker_states.get(key)
            if state and 't_fwd_start' in state:
                t2 = state['t_fwd_start']
                dur = max(0, ts - t2)
                trace_events.append(create_perfetto_event(
                    name="Model Forward",
                    cat="python", ph="X", ts=t2, dur=dur,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(state['req_ids']), "batch_size": state.get('batch_size', 0)},
                    cname="olive"  # 橄榄绿
                ))
                state['t_fwd_end'] = ts

        # ----------------------------------------------------------------
        # [Step 4] Sample (区间事件)
        # ----------------------------------------------------------------
        elif etype == 'gpu_sample':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                state = worker_states.get(key)
                req_ids = state['req_ids'] if state else payload.get('req_ids', [])
                trace_events.append(create_perfetto_event(
                    name="Sampling",
                    cat="python", ph="X", ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(req_ids)},
                    cname="mauve"  # 紫红色
                ))

        # ----------------------------------------------------------------
        # [Step 5] Bookkeeping / Sync (区间事件)
        # ----------------------------------------------------------------
        elif etype == 'gpu_bookkeeping':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                state = worker_states.get(key)
                req_ids = state['req_ids'] if state else payload.get('req_ids', [])
                trace_events.append(create_perfetto_event(
                    name="Bookkeeping/Sync",
                    cat="python", ph="X", ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(req_ids)},
                    cname="cyan"  # 青色
                ))

    # --- 3. 请求生命周期 (AsyncLLM.generate 包装) ---
    print(f"Processing Coroutine Lifecycle Events ({len(coroutine_events)})...")
    coroutine_events.sort(key=lambda x: x.get('ts') or 0)

    active_coroutines = {}
    request_spans = {}
    req_generate_start_map = {}
    req_coro_sched_ns = defaultdict(int)
    req_coro_sched_intervals = defaultdict(list)

    for ev in coroutine_events:
        etype = ev['type']
        payload = ev.get('payload', {})
        rid = payload.get('request_id')
        cid = payload.get('coroutine_id')
        ts_ns = payload.get('timestamp_ns', ev.get('ts'))

        if etype == 'coroutine_sched_latency':
            ready_ts_ns = to_int(payload.get('ready_ts_ns'))
            run_ts_ns = to_int(payload.get('run_ts_ns'))
            queue_ns = to_int(payload.get('queue_ns'))
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
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Coroutine Scheduler Queue)")
    else:
        print("  未统计到协程调度排队时间")

    # 为 request dispatch phase 做回退兜底：
    # 从 req_metrics_events 中按时间提取每个请求的 "scheduled/dequeue" phase 顺序。
    phase_hint_queue_by_req = defaultdict(deque)
    req_metric_events_sorted = sorted(
        req_metric_events,
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
                if ev_type != "dequeue":
                    continue
                phase = ev.get("phase")
                if phase in ("prefill", "decode"):
                    phase_hint_queue_by_req[rid].append(phase)
                break

    # --- 4. Scheduler 处理 (保持原逻辑) ---
    print("Processing Scheduler Events...")
    QUEUE_PID = 1000
    QUEUE_TID = 0
    trace_events.append({"name": "process_name", "ph": "M", "pid": QUEUE_PID, "args": {"name": "Scheduler Queue"}})
    
    req_enqueue_map = {}
    req_first_dispatch_done = set()
    queue_intervals_map = defaultdict(list)  # 可视化用（墙上时间）
    req_vllm_queue_ns = defaultdict(int)  # 聚合统计（总排队）
    req_vllm_queue_from_enqueue_ns = defaultdict(int)  # 初次调度等待
    req_vllm_queue_from_step_ready_ns = defaultdict(int)  # 后续step等待
    req_dispatch_intervals_map = defaultdict(list)  # 请求每次被调度执行的区间（out_rpc -> step_ready）
    req_dispatch_phase_ns = defaultdict(lambda: defaultdict(int))
    req_dispatch_phase_count = defaultdict(lambda: defaultdict(int))
    
    # 3.1 入队
    for item in scheduler_events:
        if item['type'] == 'req_enqueue_scheduler':
            p = item['payload']
            rid = p.get('req_id')
            ts = p.get('timestamp_ns')
            if rid and ts:
                if rid not in req_enqueue_map:
                    req_enqueue_map[rid] = ts
                flow_id = hash(rid) & 0x7FFFFFFF
                
                trace_events.append(create_perfetto_event(
                    f"Enqueue: {rid[:8]}", "scheduler", "i", ts, 0, QUEUE_PID, QUEUE_TID, {"full_id": rid}
                ))
                trace_events.append(create_flow_event("s", ts, QUEUE_PID, QUEUE_TID, flow_id))

    # 3.2 调度出队 (RPC)
    for item in scheduler_events:
        if item['type'] == 'req_scheduler_out_rpc':
            p = item['payload']
            ts = p.get('timestamp_ns')
            for rid in p.get('req_ids', []):
                if rid in req_enqueue_map:
                    flow_id = hash(rid) & 0x7FFFFFFF
                    trace_events.append(create_perfetto_event(
                        f"Execute: {rid[:8]}", "scheduler", "i", ts, 0, QUEUE_PID, QUEUE_TID, {"full_id": rid}
                    ))
                    # 't' step, 'f' finish. 这里用 t 表示流转到 Worker
                    trace_events.append(create_flow_event("t", ts, QUEUE_PID, QUEUE_TID, flow_id))
    
    # 3.3 Ready
    for item in scheduler_events:
        if item['type'] == 'req_step_ready':
            p = item['payload']
            ts = p.get('timestamp_ns')
            for rid in p.get('req_ids', []):
                if rid in req_enqueue_map:
                    flow_id = hash(rid) & 0x7FFFFFFF
                    trace_events.append(create_perfetto_event(
                        f"Ready: {rid[:8]}", "scheduler", "i", ts, 0, QUEUE_PID, QUEUE_TID, {"full_id": rid}
                    ))
                    # 回到了 Scheduler 视野
                    trace_events.append(create_flow_event("t", ts, QUEUE_PID, QUEUE_TID, flow_id))

    # 3.4 排队时间计算（顺序配对）：
    # 初次：req_scheduler_out_rpc - req_enqueue_scheduler
    # 后续：req_scheduler_out_rpc - 上一次 req_step_ready
    scheduler_events_sorted = sorted(
        scheduler_events,
        key=lambda x: to_ns(x.get("payload", {}).get("timestamp_ns")) or 0,
    )
    pending_ready_by_req = defaultdict(deque)

    for item in scheduler_events_sorted:
        p = item.get("payload", {})
        ev_type = item.get("type")
        ts_ns = to_ns(p.get("timestamp_ns"))
        if ts_ns is None:
            continue

        if ev_type == "req_step_ready":
            for rid in p.get("req_ids", []):
                if rid:
                    pending_ready_by_req[rid].append(ts_ns)
            continue

        if ev_type != "req_scheduler_out_rpc":
            continue

        out_ts = ts_ns
        for rid in p.get("req_ids", []):
            if not rid:
                continue

            start_ts = None
            scheduled_from = None

            enqueue_ts = to_ns(req_enqueue_map.get(rid))

            # 初次调度优先使用 enqueue->out_rpc
            if rid not in req_first_dispatch_done and enqueue_ts is not None and enqueue_ts <= out_ts:
                start_ts = enqueue_ts
                scheduled_from = "enqueue"
                req_first_dispatch_done.add(rid)
            # 后续step使用 step_ready->out_rpc
            elif pending_ready_by_req[rid] and pending_ready_by_req[rid][0] <= out_ts:
                start_ts = pending_ready_by_req[rid].popleft()
                scheduled_from = "step_ready"

            if start_ts is None or out_ts <= start_ts:
                continue

            queue_ns = out_ts - start_ts
            req_vllm_queue_ns[rid] += queue_ns
            if scheduled_from == "enqueue":
                req_vllm_queue_from_enqueue_ns[rid] += queue_ns
            else:
                req_vllm_queue_from_step_ready_ns[rid] += queue_ns

            queue_intervals_map[rid].append({
                "start_ns": start_ts,
                "end_ns": out_ts,
                "reason": "scheduled_enqueue_wait" if scheduled_from == "enqueue" else "scheduled_step_ready_wait",
                "scheduled_from": scheduled_from,
            })

    if req_vllm_queue_ns:
        print(f"共统计到 {len(req_vllm_queue_ns)} 个请求的 vLLM 内部排队时间（Scheduler口径）：")
        for rid, lat_ns in sorted(req_vllm_queue_ns.items(), key=lambda x: x[1], reverse=True):
            enqueue_ns = req_vllm_queue_from_enqueue_ns.get(rid, 0)
            step_ready_ns = req_vllm_queue_from_step_ready_ns.get(rid, 0)
            print(
                f"  - Request {rid}: {lat_ns / 1e6:.3f} ms "
                f"(scheduled-enqueue={enqueue_ns / 1e6:.3f} ms, "
                f"scheduled-step_ready={step_ready_ns / 1e6:.3f} ms)"
            )
    else:
        print("  未统计到 vLLM 内部排队时间（Scheduler口径）")

    # 3.5 每次调度执行区间：
    # 用 req_scheduler_out_rpc 作为步开始，用 req_step_ready 作为步结束；
    # phase 优先来自 req_step_ready.phase_by_req，缺失时回退到 req_metrics_events 的 phase 队列。
    pending_dispatch_start = defaultdict(deque)
    for item in scheduler_events_sorted:
        p = item.get("payload", {})
        ev_type = item.get("type")
        ts_ns = to_ns(p.get("timestamp_ns"))
        if ts_ns is None:
            continue

        if ev_type == "req_scheduler_out_rpc":
            for rid in p.get("req_ids", []):
                if rid:
                    pending_dispatch_start[rid].append(ts_ns)
            continue

        if ev_type != "req_step_ready":
            continue

        phase_by_req = p.get("phase_by_req")
        if not isinstance(phase_by_req, dict):
            phase_by_req = {}

        for rid in p.get("req_ids", []):
            if not rid:
                continue
            if not pending_dispatch_start[rid]:
                continue
            start_ns = pending_dispatch_start[rid].popleft()
            if ts_ns <= start_ns:
                continue

            phase = phase_by_req.get(rid)
            if phase not in ("prefill", "decode"):
                if phase_hint_queue_by_req[rid]:
                    phase = phase_hint_queue_by_req[rid].popleft()
                else:
                    phase = "unknown"
            else:
                # 若 hint 队列头和当前 phase 一致，则消费掉，保持两路数据大致对齐。
                if phase_hint_queue_by_req[rid] and phase_hint_queue_by_req[rid][0] == phase:
                    phase_hint_queue_by_req[rid].popleft()

            dur_ns = ts_ns - start_ns
            req_dispatch_intervals_map[rid].append({
                "start_ns": start_ns,
                "end_ns": ts_ns,
                "phase": phase,
                "duration_ns": dur_ns,
            })
            req_dispatch_phase_ns[rid][phase] += dur_ns
            req_dispatch_phase_count[rid][phase] += 1

    unmatched_dispatch = sum(len(q) for q in pending_dispatch_start.values())
    if unmatched_dispatch > 0:
        print(f"  [warn] 存在 {unmatched_dispatch} 个未闭合 dispatch 起点（可能是日志截断或进程中止）")

    if req_dispatch_intervals_map:
        print(f"共统计到 {len(req_dispatch_intervals_map)} 个请求的 dispatch 区间（prefill/decode）:")
        for rid in sorted(
            req_dispatch_intervals_map.keys(),
            key=lambda x: req_dispatch_phase_ns[x].get("prefill", 0) + req_dispatch_phase_ns[x].get("decode", 0),
            reverse=True
        ):
            prefill_ms = req_dispatch_phase_ns[rid].get("prefill", 0) / 1e6
            decode_ms = req_dispatch_phase_ns[rid].get("decode", 0) / 1e6
            unknown_ms = req_dispatch_phase_ns[rid].get("unknown", 0) / 1e6
            print(
                f"  - Request {rid}: prefill_exec={prefill_ms:.3f} ms "
                f"(count={req_dispatch_phase_count[rid].get('prefill', 0)}), "
                f"decode_exec={decode_ms:.3f} ms "
                f"(count={req_dispatch_phase_count[rid].get('decode', 0)}), "
                f"unknown_exec={unknown_ms:.3f} ms"
            )
    else:
        print("  未统计到可配对的 dispatch 区间（prefill/decode）")

    # --- 4.5 generate->enqueue 阶段耗时拆分（统计 + 部分阶段可视化数据准备） ---
    print(f"Processing Request Stage Events ({len(req_stage_events)})...")
    req_stage_ns = defaultdict(lambda: defaultdict(int))
    req_stage_counts = defaultdict(lambda: defaultdict(int))
    req_stage_intervals = defaultdict(list)

    for ev in req_stage_events:
        payload = ev.get("payload", {})
        rid = payload.get("request_id") or payload.get("req_id")
        if not rid:
            continue
        rname = payload.get("request_name")
        if rname:
            req_name_map[rid] = rname
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
        all_stage_reqs = set(req_stage_ns.keys()) | set(req_generate_start_map.keys()) | set(req_enqueue_map.keys())
        print("=== Generate -> Scheduler Enqueue Breakdown ===")
        for rid in sorted(
            all_stage_reqs,
            key=lambda x: (
                (to_ns(req_enqueue_map.get(x)) or 0) - (to_ns(req_generate_start_map.get(x)) or 0)
                if (to_ns(req_enqueue_map.get(x)) and to_ns(req_generate_start_map.get(x)))
                else sum(req_stage_ns.get(x, {}).values())
            ),
            reverse=True,
        ):
            stage_map = req_stage_ns.get(rid, {})
            if not stage_map:
                continue
            known_ns = sum(stage_map.values())
            gen_start_ns = to_ns(req_generate_start_map.get(rid))
            enqueue_ns = to_ns(req_enqueue_map.get(rid))
            pre_enqueue_total_ns = None
            if gen_start_ns is not None and enqueue_ns is not None and enqueue_ns >= gen_start_ns:
                pre_enqueue_total_ns = enqueue_ns - gen_start_ns
            uncovered_ns = None
            if pre_enqueue_total_ns is not None:
                uncovered_ns = max(0, pre_enqueue_total_ns - known_ns)

            label = req_name_map.get(rid, rid)
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

    # --- 4.6 EngineCore 主循环可视化（单独轨道） ---
    print(f"Processing EngineCore Main Loop Events ({len(enginecore_loop_events)})...")
    if enginecore_loop_events:
        ENGINECORE_PID = 12000
        trace_events.append({"name": "process_name", "ph": "M", "pid": ENGINECORE_PID, "args": {"name": "EngineCore Main Loop"}})
        tid_alias_map = {}
        next_tid_alias = 1
        for item in sorted(enginecore_loop_events, key=lambda x: x.get("ts") or 0):
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


    # --- 4. CUPTI 关联 (使用生成的 Dispatch Slices) ---
    print(f"Linking Runtime events ({len(cupti_events)}) to Dispatch Slices...")
    # 按时间排序 Runtime 事件
    cupti_events.sort(key=lambda x: x['start_ns'])
    
    for rt in cupti_events:
        rt_start = rt['start_ns']
        rt_end = rt['end_ns']
        rt_tid = rt['tid']
        rt_pid = rt.get('pid')  # 直接从 payload 获取 pid
        
        # 如果 payload 里没存 pid，尝试从生成的 slice 里找（备选方案）
        if not rt_pid and generated_dispatch_slices:
            rt_pid = generated_dispatch_slices[0]['pid']

        corr_id = rt.get('correlationId')
        func_name = rt.get('name', 'cuda_runtime')
        
        # 查找关联的 Kernel (无论它是在什么时候 launch 的)
        kernels = all_kernels_map.get(corr_id, [])
        k_names = [k.get('name', 'unknown') for k in kernels]
        
        args = {
            "correlationId": corr_id,
            "kernels": "\n".join(k_names[:5]) + ("..." if len(k_names)>5 else ""),
            "is_graph": "graph" in func_name.lower()
        }
        
        # 只要有 pid 和 tid，就直接在 Worker 轨道绘制 Runtime 条
        # 它会自动出现在 Python "Model Dispatch" 条的下方，因为它们共享 PID/TID
        trace_events.append(create_perfetto_event(
            name=func_name, cat="cuda_runtime", ph="X", ts=rt_start, dur=rt_end-rt_start,
            pid=rt_pid, tid=rt_tid, args=args
        ))
        
        # 画 Flow 线 (Runtime -> Kernel)
        # 哪怕是 CUDAGraph，只要有 correlationId，就尝试连线
        has_eager_kernel = any(k.get('start_ns', 0) > 0 for k in kernels)
        if has_eager_kernel:
            trace_events.append(create_flow_event("s", rt_start, rt_pid, rt_tid, corr_id))

    # --- 5. 画 GPU Kernel ---
    print(f"Processing Eager Kernels ({len(eager_kernels)})...")
    for k in eager_kernels:
        start = k['start_ns']
        end = k['end_ns']
        corr_id = k.get('correlationId')
        stream = k.get('streamId', 0)
        device = k.get('deviceId', 0)
        
        # 虚拟 PID 用于区分 GPU
        gpu_pid = 9000 + device
        
        trace_events.append(create_perfetto_event(
            name=k.get('name', 'kernel'), cat="gpu_kernel", ph="X", ts=start, dur=end-start,
            pid=gpu_pid, tid=stream, args=k
        ))
        
        # 命名轨道
        trace_events.append({"name": "process_name", "ph": "M", "pid": gpu_pid, "args": {"name": f"GPU {device}"}})
        trace_events.append({"name": "thread_name", "ph": "M", "pid": gpu_pid, "tid": stream, "args": {"name": f"Stream {stream}"}})

        # Flow 终点
        if corr_id:
            trace_events.append(create_flow_event("f", start, gpu_pid, stream, corr_id))

    print(f"Processing Request Metric Events ({len(req_metric_events)})...")
    mono_to_wall_offsets = []

    for item in req_metric_events:
        payload = item.get('payload', item)
        batch_ts_ns = to_ns(item.get('batch_ts_ns'))
        updates = payload.get('req_events', [])

        # 仅用于 mono->wall 偏移估计（供 OS overlap 画到 wall 时间轴）
        batch_mono_ns = []
        for update in updates:
            for ev in update.get('events', []):
                ev_ns = to_ns(ev.get('ts'))
                if ev_ns:
                    batch_mono_ns.append(ev_ns)

        if batch_ts_ns and batch_mono_ns:
            mono_to_wall_offsets.append(batch_ts_ns - max(batch_mono_ns))

    global_mono_to_wall_offset = median_int(mono_to_wall_offsets)
    if global_mono_to_wall_offset is not None:
        print(f"Estimated mono->wall offset: {global_mono_to_wall_offset} ns")


    print("\n=== 正在计算 OS 调度/抢占延迟 (eBPF) ===")
    # 1. 预处理 eBPF 数据：按 TID 分组并按时间排序，方便快速查找
    # ebpf_map: { tid: [ {start, end, dur}, ... sorted by start ] }
    export_alignment_debug_trace(execute_model_span, ebpf_sched_latency_events, output_filename="debug_os_latency_align.json")
    ebpf_map = defaultdict(list)
    for ev in ebpf_sched_latency_events:
        # 提取 payload 里的字段，兼容性处理
        payload = ev.get('payload', ev) 
        tid = payload.get('tid')
        start = payload.get('start_ns')
        end = payload.get('end_ns')
        
        if tid and start and end:
            ebpf_map[tid].append({
                'start': start,
                'end': end,
                'dur': end - start
            })

    # 对每个 TID 的事件按开始时间排序
    for tid in ebpf_map:
        ebpf_map[tid].sort(key=lambda x: x['start'])

    # 2. 结果容器：Key=req_id, Value=total_os_delay_ns
    req_latency_map = defaultdict(int)
    req_os_overlap_intervals = defaultdict(list)  # mono 时间轴，后续可转 wall 画图

    # 3. 遍历 Worker Span 计算交集
    for span in execute_model_span:
        payload = span.get('payload', span)
        
        tid = payload.get('tid')
        span_start = payload.get('start_ns')
        span_end = payload.get('end_ns')
        req_ids = payload.get('req_ids', [])

        if not (tid and span_start and span_end and req_ids):
            continue

        # 如果这个 TID 根本没有 eBPF 记录，直接跳过
        if tid not in ebpf_map:
            continue

        # 计算当前 Span 内的总 OS 延迟
        current_span_os_delay = 0
        
        # 遍历该线程的所有 eBPF 事件 (已排序)
        for os_ev in ebpf_map[tid]:
            # 优化：如果 OS 事件开始时间 已经晚于 Span 结束时间，后面的都不用看了
            if os_ev['start'] >= span_end:
                break
            
            # 优化：如果 OS 事件结束时间 早于 Span 开始时间，说明还没到，继续
            if os_ev['end'] <= span_start:
                continue

            # --- 核心：计算区间重叠 ---
            # 重叠起点 = max(Span起点, OS事件起点)
            overlap_start = max(span_start, os_ev['start'])
            # 重叠终点 = min(Span终点, OS事件终点)
            overlap_end = min(span_end, os_ev['end'])

            if overlap_end > overlap_start:
                overlap_ns = overlap_end - overlap_start
                current_span_os_delay += overlap_ns
                for rid in req_ids:
                    req_os_overlap_intervals[rid].append({
                        "start_mono_ns": overlap_start,
                        "end_mono_ns": overlap_end,
                        "worker_tid": tid,
                    })

        # 4. 归因：将计算出的延迟累加到该 Batch 的所有 Req 上
        # 逻辑：如果是 Batch 推理，这段 OS 卡顿影响了 Batch 里的每一个人
        if current_span_os_delay > 0:
            for rid in req_ids:
                req_latency_map[rid] += current_span_os_delay

    # 5. 打印结果
    print(f"共统计到 {len(req_latency_map)} 个请求受到 OS 调度影响：")
    
    # 按延迟从高到低排序
    sorted_reqs = sorted(req_latency_map.items(), key=lambda x: x[1], reverse=True)
    
    for rid, lat_ns in sorted_reqs:
        lat_ms = lat_ns / 1e6
        print(f"  - Request {rid}: {lat_ms:.3f} ms (OS Scheduling/Preemption Delay)")
            
    if not sorted_reqs:
        print("  未统计到调度延迟")

    # --- 6. 按请求聚合可视化：生命周期 + vLLM排队 + OS调度 ---
    print("\n=== Building Request Lifecycle/Queue/OS Timeline ===")
    REQUEST_PID = 11000
    trace_events.append({"name": "process_name", "ph": "M", "pid": REQUEST_PID, "args": {"name": "Request Interference"}})

    all_req_ids = (
        set(request_spans.keys())
        | set(req_coro_sched_ns.keys())
        | set(req_vllm_queue_ns.keys())
        | set(req_latency_map.keys())
        | set(req_stage_ns.keys())
        | set(req_dispatch_intervals_map.keys())
    )
    req_os_intervals_wall = defaultdict(list)
    if global_mono_to_wall_offset is not None:
        for rid, intervals in req_os_overlap_intervals.items():
            for it in intervals:
                s_wall = it["start_mono_ns"] + global_mono_to_wall_offset
                e_wall = it["end_mono_ns"] + global_mono_to_wall_offset
                if e_wall > s_wall:
                    req_os_intervals_wall[rid].append({
                        "start_ns": s_wall,
                        "end_ns": e_wall,
                        "worker_tid": it["worker_tid"],
                    })
    elif req_os_overlap_intervals:
        print("  [warn] 无法估计 mono->wall 偏移，OS overlap 仅做统计，不绘制到请求生命周期轨道")

    def req_sort_key(rid):
        span = request_spans.get(rid)
        if span and span.get("start_ns"):
            return span["start_ns"]
        q = queue_intervals_map.get(rid)
        if q:
            return q[0]["start_ns"]
        o = req_os_intervals_wall.get(rid)
        if o:
            return o[0]["start_ns"]
        s = req_stage_intervals.get(rid)
        if s:
            return min(it["start_ns"] for it in s)
        return float("inf")

    for req_tid, rid in enumerate(sorted(all_req_ids, key=req_sort_key), start=1):
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": REQUEST_PID,
            "tid": req_tid,
            "args": {"name": rid[:12]},
        })

        coro_sched_queue_ms = req_coro_sched_ns.get(rid, 0) / 1e6
        vllm_queue_ms = req_vllm_queue_ns.get(rid, 0) / 1e6
        vllm_queue_from_enqueue_ms = req_vllm_queue_from_enqueue_ns.get(rid, 0) / 1e6
        vllm_queue_from_step_ready_ms = req_vllm_queue_from_step_ready_ns.get(rid, 0) / 1e6
        os_delay_ms = req_latency_map.get(rid, 0) / 1e6
        engine_input_queue_wait_ms = req_stage_ns.get(rid, {}).get(
            "engine_core.input_queue_wait_after_preprocess", 0) / 1e6
        prefill_exec_ms = req_dispatch_phase_ns.get(rid, {}).get("prefill", 0) / 1e6
        decode_exec_ms = req_dispatch_phase_ns.get(rid, {}).get("decode", 0) / 1e6
        unknown_exec_ms = req_dispatch_phase_ns.get(rid, {}).get("unknown", 0) / 1e6

        span = request_spans.get(rid)
        if span and span.get("end_ns", 0) > span.get("start_ns", 0):
            trace_events.append(create_perfetto_event(
                name=f"Req {rid[:8]} Lifecycle",
                cat="request_lifecycle",
                ph="X",
                ts=span["start_ns"],
                dur=span["end_ns"] - span["start_ns"],
                pid=REQUEST_PID,
                tid=req_tid,
                args={
                    "request_id": rid,
                    "request_name": req_name_map.get(rid, rid),
                    "coro_sched_queue_ms": round(coro_sched_queue_ms, 3),
                    "vllm_queue_ms": round(vllm_queue_ms, 3),
                    "vllm_queue_from_enqueue_ms": round(vllm_queue_from_enqueue_ms, 3),
                    "vllm_queue_from_step_ready_ms": round(vllm_queue_from_step_ready_ms, 3),
                    "engine_input_queue_wait_ms": round(engine_input_queue_wait_ms, 3),
                    "os_sched_delay_ms": round(os_delay_ms, 3),
                    "prefill_exec_ms": round(prefill_exec_ms, 3),
                    "decode_exec_ms": round(decode_exec_ms, 3),
                    "unknown_exec_ms": round(unknown_exec_ms, 3),
                },
                cname="good",
            ))

        for d in req_dispatch_intervals_map.get(rid, []):
            if d["end_ns"] <= d["start_ns"]:
                continue
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
                tid=req_tid,
                args={
                    "request_id": rid,
                    "phase": phase,
                    "duration_ms": round((d["end_ns"] - d["start_ns"]) / 1e6, 3),
                },
                cname=d_color,
            ))

        for q0 in req_coro_sched_intervals.get(rid, []):
            if q0["end_ns"] > q0["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Coroutine Scheduler Queue Wait",
                    cat="request_coro_sched_queue",
                    ph="X",
                    ts=q0["start_ns"],
                    dur=q0["end_ns"] - q0["start_ns"],
                    pid=REQUEST_PID,
                    tid=req_tid,
                    args={"request_id": rid, "request_name": req_name_map.get(rid, rid)},
                    cname="yellow",
                ))

        for q in queue_intervals_map.get(rid, []):
            if q["end_ns"] > q["start_ns"]:
                scheduled_from = q.get("scheduled_from")
                if scheduled_from == "enqueue":
                    q_name = "vLLM Queue Wait (scheduled-enqueue)"
                    q_cat = "request_queue_scheduled_enqueue"
                elif scheduled_from == "step_ready":
                    q_name = "vLLM Scheduling Wait (step_ready->out_rpc)"
                    q_cat = "request_queue_scheduled_step_ready"
                else:
                    q_name = "vLLM Queue Wait"
                    q_cat = "request_queue"
                trace_events.append(create_perfetto_event(
                    name=q_name,
                    cat=q_cat,
                    ph="X",
                    ts=q["start_ns"],
                    dur=q["end_ns"] - q["start_ns"],
                    pid=REQUEST_PID,
                    tid=req_tid,
                    args={
                        "request_id": rid,
                        "reason": q["reason"],
                        "scheduled_from": scheduled_from,
                    },
                    cname="yellow",
                ))

        for o in req_os_intervals_wall.get(rid, []):
            if o["end_ns"] > o["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="OS Sched Delay (Attributed)",
                    cat="request_os",
                    ph="X",
                    ts=o["start_ns"],
                    dur=o["end_ns"] - o["start_ns"],
                    pid=REQUEST_PID,
                    tid=req_tid,
                    args={"request_id": rid, "worker_tid": o["worker_tid"]},
                    cname="terrible",
                ))

        for s in req_stage_intervals.get(rid, []):
            if s["stage"] != "engine_core.input_queue_wait_after_preprocess":
                continue
            trace_events.append(create_perfetto_event(
                name="EngineCore Input Queue Wait",
                cat="request_engine_core_input_queue",
                ph="X",
                ts=s["start_ns"],
                dur=s["end_ns"] - s["start_ns"],
                pid=REQUEST_PID,
                tid=req_tid,
                args={
                    "request_id": rid,
                    "stage": s["stage"],
                    "duration_ms": round((s["end_ns"] - s["start_ns"]) / 1e6, 3),
                },
                cname="yellow",
            ))

    print("=== Request Interference Summary ===")
    for rid in sorted(all_req_ids, key=lambda x: (req_coro_sched_ns.get(x, 0) + req_vllm_queue_ns.get(x, 0) + req_latency_map.get(x, 0)), reverse=True):
        lifecycle_ms = None
        if rid in request_spans:
            lifecycle_ms = (request_spans[rid]["end_ns"] - request_spans[rid]["start_ns"]) / 1e6
        print(
            f"  - {rid}: "
            f"lifecycle={f'{lifecycle_ms:.3f} ms' if lifecycle_ms is not None else 'N/A'}, "
            f"coro_sched_queue={req_coro_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"vllm_queue={req_vllm_queue_ns.get(rid, 0) / 1e6:.3f} ms "
            f"(scheduled-enqueue={req_vllm_queue_from_enqueue_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"scheduled-step_ready={req_vllm_queue_from_step_ready_ns.get(rid, 0) / 1e6:.3f} ms), "
            f"os_delay={req_latency_map.get(rid, 0) / 1e6:.3f} ms, "
            f"prefill_exec={req_dispatch_phase_ns.get(rid, {}).get('prefill', 0) / 1e6:.3f} ms, "
            f"decode_exec={req_dispatch_phase_ns.get(rid, {}).get('decode', 0) / 1e6:.3f} ms"
        )

    # 避免同轨道 end==start 的零缝边界导致 Perfetto 局部漏渲染。
    # 只修正 queue/dispatch 这两类相邻阶段，避免影响 lifecycle/OS 等重叠语义事件。
    nudged_count = nudge_equal_boundaries(
        trace_events,
        target_pid=REQUEST_PID,
        epsilon_us=1.0,
        cat_prefixes=("request_queue", "request_dispatch"),
        snap_tolerance_us=1.0,
    )
    if nudged_count > 0:
        print(f"Applied boundary nudge for request tracks: {nudged_count} events (epsilon=1.0us)")

    # --- 输出 ---
    print("✨ Sorting all trace events...")
    trace_events.sort(key=lambda x: x.get('ts', 0))
    
    output_json = {"traceEvents": trace_events}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f)
    print(f"✅ Done. Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_to_trace.py <input.log> <output.json>")
        sys.exit(1)
    process_logs(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "trace.json")
