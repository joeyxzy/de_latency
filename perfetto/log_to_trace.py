import json
import sys
from collections import defaultdict
import os

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
        "name": name, "cat": cat, "ph": ph, "ts": ts / 1000.0, 
        "pid": pid, "tid": tid, "args": args or {}
    }
    if dur is not None: 
        event["dur"] = dur / 1000.0
    if cname:
        event["cname"] = cname
    return event

def create_flow_event(ph, ts, pid, tid, corr_id):
    return {
        "name": "link", "cat": "flow", "ph": ph, "ts": ts / 1000.0, 
        "pid": pid, "tid": tid, "id": corr_id
    }

def to_ns(raw_ts):
    """
    统一时间戳到 ns。
    - 若原值是大整数（>1e14），认为已是 ns。
    - 否则按秒（float）转 ns（兼容 EngineCoreEvent.timestamp）。
    """
    if raw_ts is None:
        return None
    try:
        val = float(raw_ts)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 1e14:
        return int(val)
    return int(val * 1e9)

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
    execute_model_span = []
    ebpf_sched_latency_events = []
    
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
            elif etype in ['coroutine_start', 'coroutine_end']:
                coroutine_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', ts)
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

    for ev in coroutine_events:
        etype = ev['type']
        payload = ev.get('payload', {})
        rid = payload.get('request_id')
        cid = payload.get('coroutine_id')
        ts_ns = payload.get('timestamp_ns', ev.get('ts'))

        if etype == 'coroutine_start' and rid and cid and ts_ns:
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

    # --- 4. Scheduler 处理 (保持原逻辑) ---
    print("Processing Scheduler Events...")
    QUEUE_PID = 1000
    QUEUE_TID = 0
    trace_events.append({"name": "process_name", "ph": "M", "pid": QUEUE_PID, "args": {"name": "Scheduler Queue"}})
    
    req_enqueue_map = {}
    
    # 3.1 入队
    for item in scheduler_events:
        if item['type'] == 'req_enqueue_scheduler':
            p = item['payload']
            rid = p.get('req_id')
            ts = p.get('timestamp_ns')
            if rid and ts:
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
    req_lifecycle = defaultdict(list)  # Key=req_id, Value=list[dict]
    queue_intervals_map = defaultdict(list)  # 可视化用（墙上时间）
    req_vllm_queue_ns = defaultdict(int)  # 聚合统计
    mono_to_wall_offsets = []

    for item in req_metric_events:
        payload = item.get('payload', item)
        batch_ts_ns = to_ns(item.get('batch_ts_ns'))
        updates = payload.get('req_events', [])

        # 用同一批 req_events 估计 mono->wall 偏移量
        batch_mono_ns = []
        for update in updates:
            for ev in update.get('events', []):
                ev_ns = to_ns(ev.get('ts'))
                if ev_ns:
                    batch_mono_ns.append(ev_ns)

        batch_offset = None
        if batch_ts_ns and batch_mono_ns:
            batch_offset = batch_ts_ns - max(batch_mono_ns)
            mono_to_wall_offsets.append(batch_offset)

        for update in updates:
            rid = update.get('req_id')
            if not rid:
                continue
            for ev in update.get('events', []):
                etype = normalize_req_event_type(ev.get('type'))
                mono_ns = to_ns(ev.get('ts'))
                if not etype or not mono_ns:
                    continue
                wall_ns = mono_ns + batch_offset if batch_offset is not None else None
                req_lifecycle[rid].append({
                    "type": etype,
                    "mono_ns": mono_ns,
                    "wall_ns": wall_ns,
                })

    for rid in req_lifecycle:
        req_lifecycle[rid].sort(
            key=lambda x: (
                x.get("mono_ns") if x.get("mono_ns") is not None else float("inf"),
                x.get("wall_ns") if x.get("wall_ns") is not None else float("inf")
            )
        )

    print("Calculating request queue times...")
    for rid, events in req_lifecycle.items():
        enqueue_ev = None
        preempt_ev = None
        for ev in events:
            etype = ev['type']
            if etype == "enqueue":
                enqueue_ev = ev
            elif etype == "preempt":
                preempt_ev = ev
            elif etype == "dequeue":
                start_ev = enqueue_ev if enqueue_ev else preempt_ev
                if not start_ev:
                    continue

                queue_ns = None
                if start_ev.get("mono_ns") and ev.get("mono_ns"):
                    queue_ns = ev["mono_ns"] - start_ev["mono_ns"]
                elif start_ev.get("wall_ns") and ev.get("wall_ns"):
                    queue_ns = ev["wall_ns"] - start_ev["wall_ns"]

                if queue_ns and queue_ns > 0:
                    req_vllm_queue_ns[rid] += queue_ns

                if start_ev.get("wall_ns") and ev.get("wall_ns") and ev["wall_ns"] > start_ev["wall_ns"]:
                    queue_intervals_map[rid].append({
                        "start_ns": start_ev["wall_ns"],
                        "end_ns": ev["wall_ns"],
                        "reason": "enqueue_wait" if enqueue_ev else "preempt_wait",
                    })

                enqueue_ev = None
                preempt_ev = None

    if req_vllm_queue_ns:
        print(f"共统计到 {len(req_vllm_queue_ns)} 个请求的 vLLM 内部排队时间：")
        for rid, lat_ns in sorted(req_vllm_queue_ns.items(), key=lambda x: x[1], reverse=True):
            print(f"  - Request {rid}: {lat_ns / 1e6:.3f} ms (vLLM Internal Queue)")
    else:
        print("  未统计到 vLLM 内部排队时间")

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

    all_req_ids = set(request_spans.keys()) | set(req_vllm_queue_ns.keys()) | set(req_latency_map.keys())
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
        return float("inf")

    for req_tid, rid in enumerate(sorted(all_req_ids, key=req_sort_key), start=1):
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": REQUEST_PID,
            "tid": req_tid,
            "args": {"name": rid[:12]},
        })

        vllm_queue_ms = req_vllm_queue_ns.get(rid, 0) / 1e6
        os_delay_ms = req_latency_map.get(rid, 0) / 1e6

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
                    "vllm_queue_ms": round(vllm_queue_ms, 3),
                    "os_sched_delay_ms": round(os_delay_ms, 3),
                },
                cname="good",
            ))

        for q in queue_intervals_map.get(rid, []):
            if q["end_ns"] > q["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="vLLM Queue Wait",
                    cat="request_queue",
                    ph="X",
                    ts=q["start_ns"],
                    dur=q["end_ns"] - q["start_ns"],
                    pid=REQUEST_PID,
                    tid=req_tid,
                    args={"request_id": rid, "reason": q["reason"]},
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

    print("=== Request Interference Summary ===")
    for rid in sorted(all_req_ids, key=lambda x: (req_vllm_queue_ns.get(x, 0) + req_latency_map.get(x, 0)), reverse=True):
        lifecycle_ms = None
        if rid in request_spans:
            lifecycle_ms = (request_spans[rid]["end_ns"] - request_spans[rid]["start_ns"]) / 1e6
        print(
            f"  - {rid}: "
            f"lifecycle={f'{lifecycle_ms:.3f} ms' if lifecycle_ms is not None else 'N/A'}, "
            f"vllm_queue={req_vllm_queue_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"os_delay={req_latency_map.get(rid, 0) / 1e6:.3f} ms"
        )

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
