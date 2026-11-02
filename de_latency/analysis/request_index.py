"""Request population and OS overlap wall-clock projection."""
from .dispatch import DispatchAnalysis
from .os_scheduling import OsAttribution
from .request_stages import RequestStages
from .request_timing import RequestTiming
from .worker_intervals import WorkerAnalysis
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class RequestIndex:
    all_req_ids: set
    req_os_intervals_wall: dict


def index_requests(dispatch_analysis: DispatchAnalysis, os_result: OsAttribution, request_stages: RequestStages, request_timing: RequestTiming, worker_analysis: WorkerAnalysis) -> RequestIndex:
    # tid -> pid lookup (for OS delay rendering on worker lanes)
    tid_to_pid = {}
    for span in worker_analysis.generated_dispatch_slices:
        tid = span.get('tid')
        pid = span.get('pid')
        if tid is not None and pid is not None:
            tid_to_pid[tid] = pid

    all_req_ids = (
        set(request_timing.request_spans.keys())
        | set(request_timing.req_coro_sched_ns.keys())
        | set(request_timing.req_generate_task_sched_ns.keys())
        | set(request_timing.req_generate_exec_ns.keys())
        | set(request_timing.req_output_socket_sched_ns.keys())
        | set(request_timing.req_output_socket_exec_ns.keys())
        | set(request_timing.req_output_handler_sched_ns.keys())
        | set(request_timing.req_output_handler_exec_ns.keys())
        | set(dispatch_analysis.req_vllm_queue_ns.keys())
        | set(dispatch_analysis.req_gpu_queue_ns.keys())
        | set(os_result.request_delay_ns.keys())
        | set(request_stages.req_stage_ns.keys())
        | set(dispatch_analysis.req_dispatch_intervals_map.keys())
        | set(worker_analysis.req_step_execution_intervals.keys())
    )
    req_os_intervals_wall = defaultdict(list)
    if os_result.offset_ns is not None:
        for rid, intervals in os_result.overlaps.items():
            req_span = request_timing.request_spans.get(rid)
            for it in intervals:
                s_wall = (it.get("start_mono_ns", 0) or 0) + os_result.offset_ns
                e_wall = (it.get("end_mono_ns", 0) or 0) + os_result.offset_ns
                if e_wall <= s_wall:
                    continue
                # clip to request lifecycle
                if req_span:
                    lc_start = req_span.get("start_ns")
                    lc_end = req_span.get("end_ns")
                    if lc_start is not None:
                        s_wall = max(s_wall, lc_start)
                    if lc_end is not None:
                        e_wall = min(e_wall, lc_end)
                if e_wall <= s_wall:
                    continue
                wtid = it.get("worker_tid")
                req_os_intervals_wall[rid].append({
                    "start_ns": s_wall,
                    "end_ns": e_wall,
                    "worker_tid": wtid,
                    "worker_tids": [wtid] if wtid is not None else [],
                    "dp_ranks": list(it.get("dp_ranks") or []),
                    "reasons": [it.get("reason")] if it.get("reason") else [],
                    "worker_pid": tid_to_pid.get(wtid),
                    "pp_rank": it.get("pp_rank"),
                    "tp_rank": it.get("tp_rank"),
                })
    if os_result.offset_ns is None and os_result.overlaps:
        print("  [warn] 无法估计 mono->wall 偏移，OS overlap 仅做统计，不绘制到请求生命周期轨道")


    return RequestIndex(
        all_req_ids=all_req_ids,
        req_os_intervals_wall=req_os_intervals_wall,
    )
