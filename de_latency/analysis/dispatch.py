"""Scheduler/worker/marker dispatch joins and queue-gap attribution."""
from .ingestion import LogData
from .request_timing import RequestTiming
from .worker_intervals import WorkerAnalysis
from collections import defaultdict, deque
from de_latency.analysis.clocks import to_ns, to_int
from de_latency.analysis.intervals import merge_intervals, sum_interval_pairs
from dataclasses import dataclass
from .dispatch_windows import index_dispatch_windows
from .dispatch_queues import build_queue_intervals

@dataclass
class DispatchAnalysis:
    gpu_queue_intervals_map: dict
    queue_intervals_map: dict
    req_dispatch_intervals_map: dict
    req_dispatch_phase_ns: dict
    req_enqueue_map: dict
    req_gpu_queue_ns: dict
    req_vllm_queue_from_enqueue_ns: dict
    req_vllm_queue_from_step_ready_ns: dict
    req_vllm_queue_ns: dict


def analyze_dispatches(log_data: LogData, request_timing: RequestTiming, worker_analysis: WorkerAnalysis) -> DispatchAnalysis:
    print("Processing Scheduler Events...")
    req_enqueue_map = {}
    req_dispatch_intervals_map = defaultdict(list)  # 请求每次被调度执行的区间（out_rpc -> step_ready）
    req_dispatch_phase_ns = defaultdict(lambda: defaultdict(int))
    req_dispatch_phase_count = defaultdict(lambda: defaultdict(int))

    # 3.1 只保留事件到请求级区间的计算，不再生成 category=scheduler 的顶层点/流。
    for item in log_data.scheduler_events:
        if item['type'] != 'req_enqueue_scheduler':
            continue
        p = item['payload']
        rid = p.get('req_id')
        ts = p.get('timestamp_ns')
        if rid and ts and rid not in req_enqueue_map:
            req_enqueue_map[rid] = ts

    scheduler_events_sorted = sorted(
        log_data.scheduler_events,
        key=lambda x: to_ns(x.get("payload", {}).get("timestamp_ns")) or 0,
    )

    # 3.5 每次调度执行区间：
    # vLLM 0.20 的 async PP 会提前提交多个未来 step，req_scheduler_out_rpc
    # 更接近 RPC/队列提交点，不再可靠代表 worker 实际执行开始。request 级
    # dispatch 可视化以 scheduler_output/dispatch_key 为 join 单位：request 归属来自
    # scheduler out/ready 的 req_ids，worker 执行窗口来自同 dispatch_key 下所有
    # GPUModelRunner.execute_model 的最早开始和最晚结束。这里刻意不按 worker
    # payload.req_ids 再过滤 request，因为 PP/异步路径里 worker 上下文字段可能不完整
    # 或展示被截断；dispatch_key 才是同一 scheduler_output 的稳定连接键。
    windows = index_dispatch_windows(worker_analysis.generated_dispatch_slices, log_data.cupti_events)
    worker_exec_window_by_dispatch, gpu_batch_by_dispatch = windows.worker, windows.gpu

    # phase 优先来自 req_step_ready.phase_by_req，缺失时回退到 req_metrics_events 的 phase 队列。
    pending_dispatch_start = defaultdict(deque)

    def _pop_dispatch_start_for_ready(rid, dispatch_key):
        queue = pending_dispatch_start[rid]
        if not queue:
            return None
        if dispatch_key:
            for idx, item in enumerate(queue):
                if item.get("dispatch_key") == dispatch_key:
                    del queue[idx]
                    return item
            return None
        return queue.popleft()

    for item in scheduler_events_sorted:
        p = item.get("payload", {})
        ev_type = item.get("type")
        ts_ns = to_ns(p.get("timestamp_ns"))
        if ts_ns is None:
            continue

        if ev_type == "req_scheduler_out_rpc":
            dispatch_key = p.get("dispatch_key")
            batch_req_ids = list(p.get("req_ids", []))
            decode_step_by_req = p.get("decode_step_by_req")
            if not isinstance(decode_step_by_req, dict):
                decode_step_by_req = {}
            for rid in p.get("req_ids", []):
                if rid:
                    pending_dispatch_start[rid].append({
                        "start_ns": ts_ns,
                        "rpc_start_ns": ts_ns,
                        "dispatch_key": dispatch_key,
                        "batch_req_ids": batch_req_ids,
                        "decode_step": to_int(decode_step_by_req.get(rid)),
                    })
            continue

        if ev_type != "req_step_ready":
            continue

        ready_dispatch_key = p.get("dispatch_key")
        phase_by_req = p.get("phase_by_req")
        if not isinstance(phase_by_req, dict):
            phase_by_req = {}
        phase_reason_by_req = p.get("phase_reason_by_req")
        if not isinstance(phase_reason_by_req, dict):
            phase_reason_by_req = {}
        decode_step_by_req = p.get("decode_step_by_req")
        if not isinstance(decode_step_by_req, dict):
            decode_step_by_req = {}

        for rid in p.get("req_ids", []):
            if not rid:
                continue
            start_item = _pop_dispatch_start_for_ready(rid, ready_dispatch_key)
            if start_item is None:
                continue
            dispatch_key = ready_dispatch_key or start_item.get("dispatch_key")
            worker_window = worker_exec_window_by_dispatch.get(dispatch_key)
            rpc_start_ns = to_ns(start_item.get("rpc_start_ns") or start_item.get("start_ns"))
            start_ns = to_ns(start_item.get("start_ns"))
            end_ns = ts_ns
            start_source = "scheduler_out_rpc"
            end_source = "scheduler_step_ready"
            worker_start_ns = None
            worker_end_ns = None
            worker_span_count = 0
            worker_span_count_total = 0
            worker_boundary_policy = None

            if worker_window:
                worker_start_ns = to_ns(worker_window.get("start_ns"))
                worker_end_ns = to_ns(worker_window.get("end_ns"))
                worker_span_count = worker_window.get("span_count", 0)
                worker_span_count_total = worker_window.get("total_span_count", worker_span_count)
                worker_boundary_policy = worker_window.get("boundary_policy")

            gpu_window = gpu_batch_by_dispatch.get(dispatch_key)
            if gpu_window:
                start_ns = gpu_window["gpu_start_ns"]
                end_ns   = gpu_window["gpu_end_ns"]
                start_source = "gpu_marker"
                end_source   = "gpu_marker"
            elif worker_window:
                if worker_start_ns is not None:
                    start_ns = worker_start_ns
                    start_source = "worker_execute_model_min"
                if worker_end_ns is not None:
                    end_ns = worker_end_ns
                    end_source = "worker_execute_model_max"

            if start_ns is None:
                continue
            if end_ns <= start_ns:
                continue

            span = request_timing.request_spans.get(rid)
            if span and end_ns > span.get("end_ns", 0):
                continue

            phase = phase_by_req.get(rid)
            phase_reason = phase_reason_by_req.get(rid)
            if phase not in ("prefill", "decode"):
                if request_timing.phase_hint_queue_by_req[rid]:
                    phase = request_timing.phase_hint_queue_by_req[rid].popleft()
                    phase_reason = "req_metrics_events_fallback"
                else:
                    phase = "unknown"
                    phase_reason = "missing_phase"
            else:
                # 若 hint 队列头和当前 phase 一致，则消费掉，保持两路数据大致对齐。
                if request_timing.phase_hint_queue_by_req[rid] and request_timing.phase_hint_queue_by_req[rid][0] == phase:
                    request_timing.phase_hint_queue_by_req[rid].popleft()

            decode_step = None
            if phase == "decode":
                decode_step = to_int(decode_step_by_req.get(rid))
                if decode_step is None:
                    decode_step = to_int(start_item.get("decode_step"))

            dur_ns = end_ns - start_ns
            gpu_queue_start_ns = rpc_start_ns if rpc_start_ns is not None else None
            gpu_queue_end_ns = start_ns
            gpu_queue_ns = max(0, start_ns - rpc_start_ns) if rpc_start_ns is not None else 0
            req_dispatch_intervals_map[rid].append({
                "start_ns": start_ns,
                "end_ns": end_ns,
                "phase": phase,
                "phase_reason": phase_reason,
                "decode_step": decode_step,
                "duration_ns": dur_ns,
                "dispatch_key": dispatch_key,
                "batch_req_ids": start_item.get("batch_req_ids", []),
                "rpc_start_ns": rpc_start_ns,
                "scheduler_ready_ns": ts_ns,
                "start_source": start_source,
                "end_source": end_source,
                "gpu_marker_start_ns": gpu_window.get("gpu_start_ns") if gpu_window else None,
                "gpu_marker_end_ns": gpu_window.get("gpu_end_ns") if gpu_window else None,
                "worker_start_ns": worker_start_ns,
                "worker_end_ns": worker_end_ns,
                "worker_span_count": worker_span_count,
                "worker_span_count_total": worker_span_count_total,
                "worker_boundary_policy": worker_boundary_policy,
                "gpu_queue_start_ns": gpu_queue_start_ns,
                "gpu_queue_end_ns": gpu_queue_end_ns,
                "gpu_queue_ns": gpu_queue_ns,
            })
            req_dispatch_phase_ns[rid][phase] += dur_ns
            req_dispatch_phase_count[rid][phase] += 1

    unmatched_dispatch = sum(len(q) for q in pending_dispatch_start.values())
    if unmatched_dispatch > 0:
        print(f"  [warn] 存在 {unmatched_dispatch} 个未闭合 dispatch 起点（可能是日志截断或进程中止）")

    # Newer vLLM paths can surface the same scheduler step through multiple
    # executor wrappers.  Keep the visual/event detail, but compute per-request
    # phase totals from de-duplicated wall-clock intervals so parallel/duplicate
    # decode windows are not charged multiple times to one request.
    deduped_dispatch_intervals_map = defaultdict(list)
    for rid, intervals in req_dispatch_intervals_map.items():
        seen = set()
        for interval in sorted(intervals, key=lambda x: (x.get("start_ns") or 0, x.get("end_ns") or 0)):
            start_ns = to_ns(interval.get("start_ns"))
            end_ns = to_ns(interval.get("end_ns"))
            if start_ns is None or end_ns is None or end_ns <= start_ns:
                continue
            phase = interval.get("phase") or "unknown"
            dispatch_key = interval.get("dispatch_key")
            dedupe_key = (phase, dispatch_key, start_ns, end_ns)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            item = dict(interval)
            item["start_ns"] = start_ns
            item["end_ns"] = end_ns
            item["phase"] = phase
            item["duration_ns"] = end_ns - start_ns
            deduped_dispatch_intervals_map[rid].append(item)

    req_dispatch_intervals_map = deduped_dispatch_intervals_map
    for rid, intervals in req_dispatch_intervals_map.items():
        next_decode_step = 1
        for interval in sorted(intervals, key=lambda x: (x["start_ns"], x["end_ns"])):
            if interval.get("phase") != "decode":
                continue
            decode_step = to_int(interval.get("decode_step"))
            if decode_step is None or decode_step <= 0:
                interval["decode_step"] = next_decode_step
                next_decode_step += 1
            else:
                interval["decode_step"] = decode_step
                next_decode_step = max(next_decode_step, decode_step + 1)
    req_dispatch_phase_ns = defaultdict(lambda: defaultdict(int))
    req_dispatch_phase_count = defaultdict(lambda: defaultdict(int))
    for rid, intervals in req_dispatch_intervals_map.items():
        spans_by_phase = defaultdict(list)
        dispatch_keys_by_phase = defaultdict(set)
        for interval in intervals:
            phase = interval.get("phase") or "unknown"
            spans_by_phase[phase].append((interval["start_ns"], interval["end_ns"]))
            dispatch_keys_by_phase[phase].add(
                interval.get("dispatch_key") or (interval["start_ns"], interval["end_ns"])
            )
        for phase, spans in spans_by_phase.items():
            req_dispatch_phase_ns[rid][phase] = sum_interval_pairs(merge_intervals(spans))
            req_dispatch_phase_count[rid][phase] = len(dispatch_keys_by_phase[phase])

    queues = build_queue_intervals(req_dispatch_intervals_map, req_enqueue_map, request_timing.req_preempt_ts_by_req)
    gpu_queue_intervals_map = queues.gpu_queue_intervals_map
    queue_intervals_map = queues.queue_intervals_map
    req_gpu_queue_ns = queues.req_gpu_queue_ns
    req_vllm_queue_ns = queues.req_vllm_queue_ns
    req_vllm_queue_from_enqueue_ns = queues.req_vllm_queue_from_enqueue_ns
    req_vllm_queue_from_step_ready_ns = queues.req_vllm_queue_from_step_ready_ns

    if req_gpu_queue_ns:
        print(f"共统计到 {len(req_gpu_queue_ns)} 个请求的 GPU 调度排队时间（out_rpc → GPU start）：")
        for rid, lat_ns in sorted(req_gpu_queue_ns.items(), key=lambda x: x[1], reverse=True):
            print(f"  - Request {rid}: {lat_ns / 1e6:.3f} ms")
    else:
        print("  未统计到 GPU 调度排队时间")

    if req_vllm_queue_ns:
        print(f"共统计到 {len(req_vllm_queue_ns)} 个请求的 CPU 调度排队时间（Dispatch gap 口径）：")
        for rid, lat_ns in sorted(req_vllm_queue_ns.items(), key=lambda x: x[1], reverse=True):
            enqueue_ns = req_vllm_queue_from_enqueue_ns.get(rid, 0)
            step_ready_ns = req_vllm_queue_from_step_ready_ns.get(rid, 0)
            print(
                f"  - Request {rid}: {lat_ns / 1e6:.3f} ms "
                f"(scheduled-enqueue={enqueue_ns / 1e6:.3f} ms, "
                f"scheduled-step_ready={step_ready_ns / 1e6:.3f} ms)"
            )
    else:
        print("  未统计到 vLLM 内部排队时间（Dispatch gap 口径）")

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


    return DispatchAnalysis(
        gpu_queue_intervals_map=gpu_queue_intervals_map,
        queue_intervals_map=queue_intervals_map,
        req_dispatch_intervals_map=req_dispatch_intervals_map,
        req_dispatch_phase_ns=req_dispatch_phase_ns,
        req_enqueue_map=req_enqueue_map,
        req_gpu_queue_ns=req_gpu_queue_ns,
        req_vllm_queue_from_enqueue_ns=req_vllm_queue_from_enqueue_ns,
        req_vllm_queue_from_step_ready_ns=req_vllm_queue_from_step_ready_ns,
        req_vllm_queue_ns=req_vllm_queue_ns,
    )
