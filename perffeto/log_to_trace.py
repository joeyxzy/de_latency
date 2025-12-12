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

    print("🔗 Linking events...")

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