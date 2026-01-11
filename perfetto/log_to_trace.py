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

def process_logs(input_file, output_file):
    print(f"🔄 Loading logs from {input_file}...")
    raw_events = load_log_lines(input_file)
    trace_events = []

    # --- 1. 数据清洗与分类 ---
    # 我们将所有事件分为三类：Worker生命周期、Scheduler、CUPTI
    worker_events = []
    scheduler_events = []
    cupti_events = []
    
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
            if etype in ['worker_preprocess_start', 'gpu_forward_start', 'gpu_forward']:
                worker_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload['timestamp_ns'] # 这是事件发生的时间
                })
            elif etype in ['req_enqueue_scheduler', 'req_scheduler_out_rpc', 'req_step_ready']:
                scheduler_events.append({"type": etype, "payload": payload})

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

    # --- 2. Worker 状态机 (核心修改逻辑) ---
    print(f"Processing Worker Events ({len(worker_events)})...")
    
    # 必须按时间排序，这是状态机工作的前提
    worker_events.sort(key=lambda x: x['ts'])

    # 状态存储: Key=(pid, tid), Value={ 't1': start_ts, 'req_ids': [] }
    worker_states = {}
    
    # 存储生成的 Dispatch 切片，用于后续关联 CUPTI
    # Item: {'pid':, 'tid':, 'start':, 'end':, 'req_ids':}
    generated_dispatch_slices = []

    for ev in worker_events:
        etype = ev['type']
        payload = ev['payload']
        ts = ev['ts']
        pid = payload.get('pid')
        tid = payload.get('tid')
        key = (pid, tid)

        # ----------------------------------------------------------------
        # [Step 1] Worker Preprocess Start
        # 对应插桩: Worker.execute_model 入口
        # ----------------------------------------------------------------
        if etype == 'worker_preprocess_start':
            worker_states[key] = {
                't1': ts,
                'req_ids': payload.get('req_ids', [])
            }

        # ----------------------------------------------------------------
        # [Step 2] Forward Start (切分点)
        # 对应插桩: GPUModelRunner._model_forward 入口
        # 动作: 结算 Preprocess 切片 (T1 -> T2)，更新状态为 T2
        # ----------------------------------------------------------------
        elif etype == 'gpu_forward_start':
            state = worker_states.get(key)
            if state and 't1' in state:
                t1 = state['t1']
                # 结算 Preprocess (Duration = Now - T1)
                # 减 1000ns 是为了视觉上稍微留白，防止 Perfetto 渲染层叠
                dur = (ts - t1) - 1000 
                if dur < 0: dur = 0

                trace_events.append(create_perfetto_event(
                    name="Worker Preprocess",
                    cat="python", ph="X", ts=t1, dur=dur,
                    pid=pid, tid=tid,
                    args={
                        "req_ids": "\n".join(state['req_ids']),
                        "desc": "Prepare Inputs & Metadata"
                    },
                    cname="good" # 黄色/绿色系
                ))
                
                # 状态流转：记录 T2，进入 Dispatch 阶段
                state['t2'] = ts
            else:
                # 只有中间没有开始，可能是日志丢失或Worker启动前的残留，忽略
                pass

        # ----------------------------------------------------------------
        # [Step 3] Forward End (整体结束)
        # 对应插桩: GPUModelRunner.execute_model 结束
        # 动作: 结算 Dispatch 切片 (T2 -> T3)
        # 注意: 这里使用 payload['end_ns'] 作为 T3，因为它比事件ts更精准
        # ----------------------------------------------------------------
        elif etype == 'gpu_forward':
            state = worker_states.get(key)
            if state and 't2' in state:
                t2 = state['t2']
                # 优先使用 payload 里的 end_ns，如果没有则用事件时间
                t3 = payload.get('end_ns', ts)
                
                dur = t3 - t2
                if dur < 0: dur = 0 # 异常保护

                # 元数据提取
                batch_size = payload.get('batch_size', 0)
                input_type = payload.get('input_type', 'unknown')
                req_ids = payload.get('req_ids') or state.get('req_ids', [])

                # 生成 Dispatch 切片 (Fwd + Post)
                trace_events.append(create_perfetto_event(
                    name="Model Dispatch (Fwd+Post)",
                    cat="python", ph="X", ts=t2, dur=dur,
                    pid=pid, tid=tid,
                    args={
                        "batch_size": batch_size,
                        "input_type": input_type,
                        "req_ids": "\n".join(req_ids)
                    },
                    cname="terrible" # 红色/紫色系，醒目
                ))

                # 记录下来给 CUPTI 匹配用
                generated_dispatch_slices.append({
                    'pid': pid, 'tid': tid,
                    'start': t2, 'end': t3,
                    'req_ids': req_ids
                })
            
            # 一个 Step 完成，清空状态
            if key in worker_states:
                del worker_states[key]

    # --- 3. Scheduler 处理 (保持原逻辑) ---
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