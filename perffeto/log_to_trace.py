import json
import sys
import os

def load_log_lines(filepath):
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                json_start = line.find('{')
                if json_start != -1:
                    events.append(json.loads(line[json_start:]))
            except json.JSONDecodeError:
                continue
    return events

def create_perfetto_event(name, cat, ph, ts, dur, pid, tid, args=None):
    event = {
        "name": name, "cat": cat, "ph": ph, "ts": ts / 1000, 
        "pid": pid, "tid": tid, "args": args or {}
    }
    if dur is not None: event["dur"] = dur / 1000
    return event

def create_flow_event(ph, ts, pid, tid, corr_id):
    return {
        "name": "launch_link", "cat": "flow", "ph": ph, "ts": ts / 1000, 
        "pid": pid, "tid": tid, "id": corr_id
    }

def process_logs(input_file, output_file):
    print(f"🔄 Loading logs from {input_file}...")
    raw_events = load_log_lines(input_file)
    trace_events = []

    gpu_forward_events = []
    runtime_events = []
    
    # 存储所有 Kernel 信息，用于填充 Runtime 详情
    all_kernels_map = {}
    
    # 仅存储有真实时间戳的 Kernel，用于画图
    eager_kernels = []

    req_enqueue_scheduler = []
    req_scheduler_out_rpc = []
    req_step_ready = []

    worker_preprocess_events = [] 

    coroutine_starts = {}

    # --- Pass 1: 数据分类 ---
    for entry in raw_events:
        meta = entry.get('meta', {})
        payload = meta.get('payload', {})
        if not payload: continue

        src = meta.get('source')
        etype = meta.get('event_type')
        
        if src == 'monkey_patch':
            if etype == 'gpu_forward':
                gpu_forward_events.append(payload)
            elif etype == 'coroutine_start':
                coroutine_starts[payload.get('coroutine_id')] = payload
            elif etype == 'coroutine_end':
                # 虚拟 TID 逻辑
                cid = payload.get('coroutine_id')
                end_ts = payload.get('timestamp_ns')
                duration_ns = payload.get('duration_ms', 0) * 1_000_000
                start_ts = end_ts - duration_ns
                if cid in coroutine_starts:
                    start_ts = coroutine_starts[cid].get('timestamp_ns', start_ts)

                req_id = payload.get('request_id', 'unknown')
                real_pid = payload.get('pid')
                virtual_tid = (hash(req_id) & 0x7FFFFFFF) if req_id != 'unknown' else cid

                trace_events.append(create_perfetto_event(
                    name=f"Req: {req_id[:8]}", cat="python_coro", ph="X",
                    ts=start_ts, dur=duration_ns, pid=real_pid, tid=virtual_tid,
                    args={"full_req_id": req_id}
                ))
                trace_events.append({
                    "name": "thread_name", "ph": "M", "pid": real_pid, "tid": virtual_tid,
                    "args": {"name": f"Req-{req_id[:6]}"}
                })
            elif etype == 'req_scheduler_out_rpc':
                req_scheduler_out_rpc.append(payload)
            elif etype == 'req_enqueue_scheduler':
                req_enqueue_scheduler.append(payload)
            elif etype == "req_step_ready":
                req_step_ready.append(payload)
            elif etype == "worker_preprocess_start":
                worker_preprocess_events.append(payload)

        elif src == 'CUPTI':
            corr_id = payload.get('correlationId')
            
            if etype in ['runtime', 'driver']:
                runtime_events.append(payload)
            
            elif etype in ['kernel', 'memset', 'memcpy']:
                # 1. 注册到全局索引 (为了让 Runtime 能查到它触发了啥)
                if corr_id not in all_kernels_map:
                    all_kernels_map[corr_id] = []
                all_kernels_map[corr_id].append(payload)

                start_ns = payload.get('start_ns', 0)
                
                # 2. 只有 Eager Kernel (Start > 0) 才会被画出来
                if start_ns > 0:
                    eager_kernels.append(payload)
                # Graph Kernel (Start == 0) 直接丢弃，不放入 eager_kernels 列表

    # --- Pass 2: Python Forward & Runtime ---
    gpu_forward_events.sort(key=lambda x: x['start_ns'])
    runtime_events.sort(key=lambda x: x['start_ns'])

    print("Processing Scheduler Events...")
    print(f"📊 Stats: Enqueues={len(req_enqueue_scheduler)}, out_rpc={len(req_scheduler_out_rpc)}, step_ready={len(req_step_ready)}")
    req_enqueue_map = {} 

    # 3.1 处理入队事件 (画在专门的 Queue 轨道上)
    QUEUE_PID = 1000  # 给调度队列分配一个固定的虚拟 PID
    QUEUE_TID = 0     # 主队列
    
    trace_events.append({
        "name": "process_name", "ph": "M", "pid": QUEUE_PID, 
        "args": {"name": "Scheduler Queue"}
    })

    #1.为请求入队创建队列
    for item in req_enqueue_scheduler:
        req_id = item.get('req_id')
        ts = item.get('timestamp_ns')
        if not req_id or not ts: continue
        
        # 记录下来，供后续连线使用
        req_enqueue_map[req_id] = {"ts": ts, "pid": QUEUE_PID, "tid": QUEUE_TID}
        print(f"Enqueue recorded: {req_id} at {ts}")
        # 在轨道上是Instant Event
        trace_events.append(create_perfetto_event(
            name=f"Enqueue: {req_id[:8]}", # 简写一下 ID 避免太长
            cat="scheduler", ph="i", ts=ts, dur=0, # Instant
            pid=QUEUE_PID, tid=QUEUE_TID,
            args={"full_req_id": req_id}
        ))
        
        # [关键] 开启 Flow (Flow Start 's')
        # 我们用 req_id 字符串本身作为 flow id (我们 hash 一下变成数字)
        flow_id = hash(req_id) & 0x7FFFFFFF
        
        trace_events.append(create_flow_event(
            ph="s", ts=ts, pid=QUEUE_PID, tid=QUEUE_TID, corr_id=flow_id
        ))

    for item in req_scheduler_out_rpc:
        req_ids = item.get('req_ids', [])
        ts = item.get('timestamp_ns')
        
        # 如果日志里没带 pid/tid，就得去 gpu_forward 里找对应时间段的。
        # 简单起见，假设 req_scheduler_out_rpc 的 payload 里也加上 pid/tid。
        
        for req_id in req_ids:
            if req_id in req_enqueue_map:
                flow_id = hash(req_id) & 0x7FFFFFFF
                
                # 画一个 Flow Step ('t') 指向这里
                # 注意：'t' 表示 flow 经过这里但还在继续（因为一个请求可能被调度多次）
                # 如果是最后一次，应该用 'f'。但我们不知道是不是最后一次，用 't' 安全。
                
                # 为了让箭头能显示出来，我们需要在这个时间点有一个 slice。
                # 幸好，execute_model 就在这个时间点附近。
                # 我们可以创建一个极短的 Slice 或者 Instant Event 来接收箭头
                trace_events.append(create_perfetto_event(
                    name=f"Execute: {req_id[:8]}",
                    cat="scheduler", ph="i", ts=ts, dur=0,
                    pid=QUEUE_PID, tid=QUEUE_TID,
                    args={"full_req_id": req_id}
                ))
                trace_events.append(create_flow_event(
                    ph="t", ts=ts, pid=QUEUE_PID, tid=QUEUE_TID, corr_id=flow_id
                ))
    
    for item in req_step_ready:
        req_ids = item.get('req_ids', [])
        ts = item.get('timestamp_ns')
        if not req_ids or not ts: continue
        
        for req_id in req_ids:
            if req_id in req_enqueue_map:
                flow_id = hash(req_id) & 0x7FFFFFFF
                
                trace_events.append(create_perfetto_event(
                    name=f"Step Ready: {req_id[:8]}",
                    cat="scheduler", ph="i", ts=ts, dur=0,
                    pid=QUEUE_PID, tid=QUEUE_TID,
                    args={"full_req_id": req_id}
                ))
                trace_events.append(create_flow_event(
                    ph="t", ts=ts, pid=QUEUE_PID, tid=QUEUE_TID, corr_id=flow_id
                ))

    print("Processing Worker Preprocess Events...")
    print(f"📊 Stats: Worker Preprocess Events={len(worker_preprocess_events)},len of gpu_forward_events={len(gpu_forward_events)}")
    from collections import deque
    preprocess_queues = {}
    
    # 必须排序，保证 FIFO
    worker_preprocess_events.sort(key=lambda x: x.get('timestamp_ns', 0))
    
    for wp in worker_preprocess_events:
        pid = wp.get('pid')
        tid = wp.get('tid')
        key = (pid, tid)
        if key not in preprocess_queues:
            preprocess_queues[key] = deque()
        preprocess_queues[key].append(wp)
    print("all keys in preprocess_queues:",list(preprocess_queues.keys()))
    for py_ev in gpu_forward_events:
        py_start = py_ev['start_ns']
        py_end = py_ev['end_ns']
        py_tid = py_ev['tid']
        py_pid = py_ev['pid'] # 确保取到了 PID
        req_ids = py_ev.get('req_ids', [])
        #print("py_pid=",py_pid," py_tid=",py_tid," req_ids=",req_ids)

        # ========================================================
        # [NEW] 匹配并生成 Worker Preprocess 切片
        # ========================================================
        key = (py_pid, py_tid)
        if key in preprocess_queues and preprocess_queues[key]:
            # 取出队列头部的预处理事件
            # 逻辑：预处理一定发生在 Forward 之前
            wp = preprocess_queues[key][0]
            wp_ts = wp.get('timestamp_ns', 0)
            
            # 只有当预处理时间 早于 Forward 开始时间，才是有效匹配
            if wp_ts < py_start:
                # 消耗掉这个事件
                preprocess_queues[key].popleft()
                
                # 计算时长
                #视觉上1us的误差，因为对于perffeto两个slcie的首尾时间完全一样的时候会出现渲染bug
                prep_dur = (py_start - wp_ts) - 1000 
                
                #print(f"Matched Worker Preprocess for PID={py_pid}, TID={py_tid}, ReqIDs={req_ids}, Start={wp_ts}, Duration={prep_dur}")
                # 生成切片 (黄色)
                trace_events.append(create_perfetto_event(
                    name="Worker Preprocess",
                    cat="worker", ph="X", 
                    ts=wp_ts, dur=prep_dur,
                    pid=py_pid, tid=py_tid,
                    args={
                        "related_reqs": "\n".join(req_ids),
                        "duration_ms": prep_dur / 1e6
                    }
                ))

    print("🔗 Linking events...")
    print(f"📊 Stats: GPU Forwards={len(gpu_forward_events)}, Runtimes={len(runtime_events)}")
    for py_ev in gpu_forward_events:
        py_start = py_ev['start_ns']
        py_end = py_ev['end_ns']
        py_tid = py_ev['tid']
        req_ids = py_ev.get('req_ids', [])
        
        # Python 条目
        trace_events.append(create_perfetto_event(
            name=f"execute_model ({py_ev.get('method', 'unknown')})",
            cat="python", ph="X", ts=py_start, dur=py_end - py_start,
            pid=py_ev['pid'], tid=py_ev['tid'],
            args={
                "batch_size": py_ev.get('batch_size'),
                "req_ids": "\n".join(req_ids),
                "input_type": py_ev.get('input_type')
            }
        ))

        # 查找 Runtime
        for rt_ev in runtime_events:
            rt_start = rt_ev['start_ns']
            rt_tid = rt_ev['tid']
            
            if rt_tid == py_tid and rt_start >= py_start and rt_start <= py_end:
                func_name = rt_ev.get('name', rt_ev.get('type', 'runtime'))
                corr_id = rt_ev.get('correlationId')
                
                # 准备详情 Args
                rt_args = {
                    "correlationId": corr_id,
                    "triggered_by_reqs": "\n".join(req_ids)
                }

                # 填充 Kernel 信息 (即使不画图，信息也要在)
                triggered_kernels = all_kernels_map.get(corr_id, [])
                is_graph_launch = False
                
                if triggered_kernels:
                    k_names = [k.get('name', 'unknown') for k in triggered_kernels]
                    rt_args["kernels_launched"] = f"Count: {len(k_names)}\n" + "\n".join(k_names[:20])
                    if len(k_names) > 20: rt_args["kernels_launched"] += "\n..."
                    
                    # 判断是否有真实时间戳 (如果有 start=0 的，说明是 Graph)
                    if any(k.get('start_ns', 0) == 0 for k in triggered_kernels):
                        is_graph_launch = True

                # Runtime 条目
                trace_events.append(create_perfetto_event(
                    name=func_name, cat="cuda", ph="X", ts=rt_start, 
                    dur=rt_ev['end_ns'] - rt_start, pid=rt_ev['pid'], tid=rt_ev['tid'],
                    args=rt_args
                ))

                # 连线逻辑：只有不是 Graph Launch (即 Eager 模式) 才画线
                # 因为只有 Eager 模式下，GPU 轨道上才有对应的 Kernel 条目作为终点
                if triggered_kernels and not is_graph_launch:
                    trace_events.append(create_flow_event(
                        ph="s", ts=rt_start, pid=rt_ev['pid'], tid=rt_ev['tid'], corr_id=corr_id
                    ))

    # --- Pass 3: Eager Kernel (只画有时间的) ---
    print("🎨 Processing Eager Kernels only...")
    for k_payload in eager_kernels:
        start = k_payload['start_ns']
        end = k_payload['end_ns']
        corr_id = k_payload.get('correlationId')
        stream_id = k_payload.get('streamId', 0)
        
        trace_events.append(create_perfetto_event(
            name=k_payload.get('name', 'kernel'),
            cat="gpu_kernel", ph="X", ts=start, dur=end - start,
            pid=0, tid=stream_id, args=k_payload
        ))

        # Eager 模式连线终点
        if corr_id:
            trace_events.append(create_flow_event(
                ph="f", ts=start, pid=0, tid=stream_id, corr_id=corr_id
            ))
            
    print("✨ Sorting all events by timestamp...")
    trace_events.sort(key=lambda x: x.get('ts', 0))
    # 输出
    output_json = {"traceEvents": trace_events}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f)
    print(f"✅ Done. {len(trace_events)} events saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_to_trace.py <input.log> <output.json>")
        sys.exit(1)
    process_logs(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "trace.json")