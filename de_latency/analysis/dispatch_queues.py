"""Queue gaps from the canonical, de-duplicated request dispatch sequence."""
from collections import defaultdict
from dataclasses import dataclass
from .clocks import to_ns


@dataclass
class QueueAnalysis:
    gpu_queue_intervals_map: dict
    queue_intervals_map: dict
    req_gpu_queue_ns: dict
    req_vllm_queue_ns: dict
    req_vllm_queue_from_enqueue_ns: dict
    req_vllm_queue_from_step_ready_ns: dict


def build_queue_intervals(req_dispatch_intervals_map: dict, req_enqueue_map: dict,
                          preempt_ts_by_req: dict) -> QueueAnalysis:
    gpu_queue_intervals_map = defaultdict(list)
    queue_intervals_map = defaultdict(list)
    req_gpu_queue_ns = defaultdict(int)
    req_vllm_queue_ns = defaultdict(int)
    req_vllm_queue_from_enqueue_ns = defaultdict(int)
    req_vllm_queue_from_step_ready_ns = defaultdict(int)
    # Derive vLLM queue intervals from the canonical dispatch sequence.
    # This keeps queue slices in the gaps between dispatch slices and prevents
    # visually impossible overlap such as step_ready->out_rpc appearing inside
    # a Decode Dispatch lane.
    overlapped_dispatch_pairs = 0
    gpu_queue_intervals_map = defaultdict(list)
    for rid, dispatches_for_req in req_dispatch_intervals_map.items():
        dispatches_sorted = sorted(
            dispatches_for_req,
            key=lambda x: (x.get("start_ns") or 0, x.get("end_ns") or 0),
        )
        if not dispatches_sorted:
            continue

        enqueue_ts = to_ns(req_enqueue_map.get(rid))
        first_start = to_ns(dispatches_sorted[0].get("start_ns"))
        first_rpc = to_ns(dispatches_sorted[0].get("rpc_start_ns"))
        if enqueue_ts is not None and first_start is not None and first_start > enqueue_ts:
            gap_end = first_start
            rpc_end = first_rpc if first_rpc is not None and first_rpc > enqueue_ts and first_rpc < first_start else None
            cpu_end = rpc_end if rpc_end else gap_end
            cpu_gap_ns = max(0, cpu_end - enqueue_ts)
            gpu_gap_ns = max(0, gap_end - cpu_end)
            if cpu_gap_ns > 0:
                req_vllm_queue_ns[rid] += cpu_gap_ns
                req_vllm_queue_from_enqueue_ns[rid] += cpu_gap_ns
                queue_intervals_map[rid].append({
                    "start_ns": enqueue_ts,
                    "end_ns": cpu_end,
                    "reason": "scheduled_enqueue_wait",
                    "scheduled_from": "enqueue",
                    "gap_kind": "normal_wait",
                    "next_dispatch_key": dispatches_sorted[0].get("dispatch_key"),
                })
            if gpu_gap_ns > 0:
                req_gpu_queue_ns[rid] += gpu_gap_ns
                gpu_queue_intervals_map[rid].append({
                    "start_ns": cpu_end,
                    "end_ns": gap_end,
                    "duration_ns": gpu_gap_ns,
                    "dispatch_key": dispatches_sorted[0].get("dispatch_key"),
                    "phase": dispatches_sorted[0].get("phase", "unknown"),
                    "start_source": "scheduler_out_rpc",
                    "end_source": dispatches_sorted[0].get("start_source", "gpu_or_worker_start"),
                })

        for prev_dispatch, next_dispatch in zip(dispatches_sorted, dispatches_sorted[1:]):
            prev_end = to_ns(prev_dispatch.get("end_ns"))
            next_start = to_ns(next_dispatch.get("start_ns"))
            next_rpc = to_ns(next_dispatch.get("rpc_start_ns"))
            if prev_end is None or next_start is None:
                continue
            if next_start < prev_end:
                overlapped_dispatch_pairs += 1
                continue
            if next_start == prev_end:
                continue
            gap_end = next_start
            rpc_split = next_rpc if next_rpc is not None and next_rpc > prev_end and next_rpc < gap_end else None
            cpu_end = rpc_split if rpc_split else gap_end
            cpu_gap_ns = max(0, cpu_end - prev_end)
            gpu_gap_ns = max(0, gap_end - cpu_end)

            if cpu_gap_ns > 0:
                req_vllm_queue_ns[rid] += cpu_gap_ns
                req_vllm_queue_from_step_ready_ns[rid] += cpu_gap_ns
                preempt_hits = [
                    ts for ts in preempt_ts_by_req.get(rid, [])
                    if prev_end <= ts <= cpu_end
                ]
                gap_kind = "preempt_wait" if preempt_hits else "normal_wait"
                queue_intervals_map[rid].append({
                    "start_ns": prev_end,
                    "end_ns": cpu_end,
                    "reason": "scheduled_preempt_wait" if preempt_hits else "scheduled_step_ready_wait",
                    "scheduled_from": "step_ready",
                    "gap_kind": gap_kind,
                    "preempt_ts_ns": preempt_hits,
                    "prev_dispatch_key": prev_dispatch.get("dispatch_key"),
                    "next_dispatch_key": next_dispatch.get("dispatch_key"),
                })

            if gpu_gap_ns > 0:
                req_gpu_queue_ns[rid] += gpu_gap_ns
                gpu_queue_intervals_map[rid].append({
                    "start_ns": cpu_end,
                    "end_ns": gap_end,
                    "duration_ns": gpu_gap_ns,
                    "dispatch_key": next_dispatch.get("dispatch_key"),
                    "phase": next_dispatch.get("phase", "unknown"),
                    "start_source": "scheduler_out_rpc",
                    "end_source": next_dispatch.get("start_source", "gpu_or_worker_start"),
                })

    if overlapped_dispatch_pairs:
        print(
            f"  [warn] 存在 {overlapped_dispatch_pairs} 对同请求 dispatch 时间重叠，"
            "已跳过对应 queue gap；建议检查 dispatch_key 或日志时序"
        )


    return QueueAnalysis(
        gpu_queue_intervals_map=gpu_queue_intervals_map,
        queue_intervals_map=queue_intervals_map,
        req_gpu_queue_ns=req_gpu_queue_ns,
        req_vllm_queue_ns=req_vllm_queue_ns,
        req_vllm_queue_from_enqueue_ns=req_vllm_queue_from_enqueue_ns,
        req_vllm_queue_from_step_ready_ns=req_vllm_queue_from_step_ready_ns,
    )
