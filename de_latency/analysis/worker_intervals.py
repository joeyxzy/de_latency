"""Worker execution/phase intervals and PP request attribution."""
from .ingestion import LogData
from collections import defaultdict
from de_latency.analysis.clocks import to_ns, to_int
from de_latency.analysis.requests import normalize_request_ids
from de_latency.analysis.normalization import count_prefill_decode_tokens
from dataclasses import dataclass

@dataclass
class WorkerTimelineSpan:
    """Normalized source span for export; timestamps stay integer nanoseconds."""
    kind: str
    start_ns: int
    end_ns: int
    pid: int
    tid: int
    payload: dict


@dataclass
class WorkerAnalysis:
    timeline: list[WorkerTimelineSpan]
    req_pp_comm_intervals: dict
    pp_comm_attr_stats: dict
    dispatch_decode_tokens_cache: dict
    dispatch_prefill_tokens_cache: dict
    dispatch_scheduled_tokens_cache: dict
    generated_dispatch_slices: list
    req_step_execution_intervals: dict
    req_worker_phase_intervals: dict


def analyze_workers(log_data: LogData) -> WorkerAnalysis:
    timeline = []
    print(f"Processing Worker Events ({len(log_data.worker_events)})...")
    
    log_data.worker_events.sort(key=lambda x: x['ts'])
    has_precise_phase_spans = any(ev.get('type') == 'gpu_phase_span' for ev in log_data.worker_events)
    req_worker_phase_intervals = defaultdict(lambda: defaultdict(list))
    req_step_execution_intervals = defaultdict(list)
    req_pp_comm_intervals = defaultdict(list)
    pp_comm_attr_stats = defaultdict(int)

    # 状态存储: Key=(pid, tid), Value={
    #   't_pre_start': ..., 
    #   't_fwd_start': ..., 
    #   't_fwd_end': ..., 
    #   'req_ids': [...],
    #   'batch_size': ..., 'input_type': ...
    # }
    worker_states = {}
    generated_dispatch_slices = []  # 仍可用于 CUPTI 关联（用整个 execute_model 区间）
    dispatch_scheduled_tokens_cache = {}  # dispatch_key -> scheduled_tokens_by_req
    dispatch_prefill_tokens_cache = {}    # dispatch_key -> prefill_tokens
    dispatch_decode_tokens_cache = {}     # dispatch_key -> decode_tokens

    for ev in log_data.worker_events:
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
                dispatch_key = payload.get("dispatch_key")
                ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
                
                # 记录完整区间（用于 CUPTI 关联）
                generated_dispatch_slices.append({
                    'pid': pid, 'tid': tid,
                    'start': t_start, 'end': t_end,
                    'req_ids': req_ids,
                    'dispatch_key': dispatch_key,
                    'ranks': ranks,
                })
                for rid in normalize_request_ids(req_ids):
                    req_step_execution_intervals[rid].append({
                        "pid": pid,
                        "tid": tid,
                        "start_ns": t_start,
                        "end_ns": t_end,
                        "dispatch_key": dispatch_key,
                        "ranks": ranks,
                        "batch_size": batch_size,
                        "input_type": input_type,
                        "scheduled_tokens_by_req": payload.get("scheduled_tokens_by_req", {}),
                        "num_computed_by_req": payload.get("num_computed_by_req", {}),
                    })
                if dispatch_key:
                    sched_tokens_map = payload.get("scheduled_tokens_by_req")
                    if isinstance(sched_tokens_map, dict):
                        dispatch_scheduled_tokens_cache[dispatch_key] = dict(sched_tokens_map)
                        prefill_tokens, decode_tokens = count_prefill_decode_tokens(sched_tokens_map)
                        dispatch_prefill_tokens_cache[dispatch_key] = prefill_tokens
                        dispatch_decode_tokens_cache[dispatch_key] = decode_tokens
                
                timeline.append(WorkerTimelineSpan("execution", t_start, t_end, pid, tid, dict(payload)))
            continue  # 不进入状态机流转

        if etype == 'gpu_phase_span':
            phase_name = payload.get('phase')
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if phase_name and t_start is not None and t_end is not None:
                req_ids = payload.get('req_ids', [])
                batch_size = payload.get('batch_size')
                input_type = payload.get('input_type')
                dispatch_key = payload.get("dispatch_key")
                ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
                for rid in normalize_request_ids(req_ids):
                    req_worker_phase_intervals[rid][phase_name].append({
                        "start_ns": t_start,
                        "end_ns": t_end,
                        "dispatch_key": dispatch_key,
                        "ranks": ranks,
                    })
                timeline.append(WorkerTimelineSpan("phase", t_start, t_end, pid, tid, dict(payload)))
                state = worker_states.setdefault(key, {})
                if req_ids:
                    state['req_ids'] = req_ids
                if batch_size is not None:
                    state['batch_size'] = batch_size
                if input_type:
                    state['input_type'] = input_type
            continue

        # ----------------------------------------------------------------
        # [Step 1] Preprocess Start
        # ----------------------------------------------------------------
        if not has_precise_phase_spans and etype == 'worker_preprocess_start':
            worker_states[key] = {
                't_pre_start': ts,
                'req_ids': payload.get('req_ids', []),
                'batch_size': payload.get('batch_size'),
                'input_type': payload.get('input_type', 'unknown')
            }

        # ----------------------------------------------------------------
        # [Step 2] Forward Start
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_forward_start':
            state = worker_states.get(key)
            if state and 't_pre_start' in state:
                t1 = state['t_pre_start']
                dur = max(0, ts - t1 - 1000)  # 留白
                for rid in normalize_request_ids(state.get('req_ids', [])):
                    req_worker_phase_intervals[rid]["Preprocess"].append({
                        "start_ns": t1,
                        "end_ns": t1 + dur,
                    })
                timeline.append(WorkerTimelineSpan("preprocess", t1, t1 + dur, pid, tid, dict(state)))
                state['t_fwd_start'] = ts

        # ----------------------------------------------------------------
        # [Step 3] Forward End
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_forward_end':
            state = worker_states.get(key)
            if state and 't_fwd_start' in state:
                t2 = state['t_fwd_start']
                dur = max(0, ts - t2)
                for rid in normalize_request_ids(state.get('req_ids', [])):
                    req_worker_phase_intervals[rid]["Forward"].append({
                        "start_ns": t2,
                        "end_ns": t2 + dur,
                    })
                timeline.append(WorkerTimelineSpan("forward", t2, t2 + dur, pid, tid, dict(state)))
                state['t_fwd_end'] = ts

        # ----------------------------------------------------------------
        # [Step 4] Sample (区间事件)
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_sample':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                state = worker_states.get(key)
                req_ids = state['req_ids'] if state else payload.get('req_ids', [])
                for rid in normalize_request_ids(req_ids):
                    req_worker_phase_intervals[rid]["Sample"].append({
                        "start_ns": t_start,
                        "end_ns": t_end,
                    })
                timeline.append(WorkerTimelineSpan("sample", t_start, t_end, pid, tid, {"req_ids": list(req_ids)}))

        # ----------------------------------------------------------------
        # [Step 5] Bookkeeping / Sync (区间事件)
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_bookkeeping':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                state = worker_states.get(key)
                req_ids = state['req_ids'] if state else payload.get('req_ids', [])
                for rid in normalize_request_ids(req_ids):
                    req_worker_phase_intervals[rid]["Bookkeep"].append({
                        "start_ns": t_start,
                        "end_ns": t_end,
                    })
                timeline.append(WorkerTimelineSpan("bookkeep", t_start, t_end, pid, tid, {"req_ids": list(req_ids)}))

    dispatch_slices_by_pid = defaultdict(list)
    for span in generated_dispatch_slices:
        span_start = to_ns(span.get("start"))
        span_end = to_ns(span.get("end"))
        if span_start is None or span_end is None or span_end <= span_start:
            continue
        dispatch_slices_by_pid[span.get("pid")].append(span)
    for spans in dispatch_slices_by_pid.values():
        spans.sort(key=lambda x: (to_ns(x.get("start")) or 0, to_ns(x.get("end")) or 0))

    def _rank_value_known(value):
        if value is None:
            return False
        if isinstance(value, str) and value.strip() in ("", "unknown", "-1"):
            return False
        return value != -1

    def _rank_compatible(left, right):
        if not _rank_value_known(left) or not _rank_value_known(right):
            return True
        return str(left) == str(right)

    def _infer_pp_comm_request_context(start_ns, end_ns, pid, ranks, dispatch_key):
        candidates = dispatch_slices_by_pid.get(pid, [])
        if not candidates:
            candidates = generated_dispatch_slices

        best_overlap = 0
        best_spans = []
        for span in candidates:
            span_start = to_ns(span.get("start"))
            span_end = to_ns(span.get("end"))
            if span_start is None or span_end is None or span_end <= span_start:
                continue
            if dispatch_key and span.get("dispatch_key") and span.get("dispatch_key") != dispatch_key:
                continue
            span_ranks = span.get("ranks") if isinstance(span.get("ranks"), dict) else {}
            if not (
                _rank_compatible(ranks.get("pp"), span_ranks.get("pp"))
                and _rank_compatible(ranks.get("tp"), span_ranks.get("tp"))
                and _rank_compatible(ranks.get("dp"), span_ranks.get("dp"))
            ):
                continue
            overlap_start = max(start_ns, span_start)
            overlap_end = min(end_ns, span_end)
            overlap_ns = max(0, overlap_end - overlap_start)
            if overlap_ns <= 0:
                continue
            if overlap_ns > best_overlap:
                best_overlap = overlap_ns
                best_spans = [span]
            elif overlap_ns == best_overlap:
                best_spans.append(span)

        inferred_req_ids = []
        inferred_dispatch_keys = []
        for span in best_spans:
            inferred_req_ids.extend(normalize_request_ids(span.get("req_ids")))
            if span.get("dispatch_key"):
                inferred_dispatch_keys.append(span.get("dispatch_key"))

        inferred_req_ids = sorted(set(inferred_req_ids))
        inferred_dispatch_keys = sorted(set(inferred_dispatch_keys))
        inferred_dispatch_key = inferred_dispatch_keys[0] if len(inferred_dispatch_keys) == 1 else dispatch_key
        return inferred_req_ids, inferred_dispatch_key, best_overlap

    if log_data.pp_comm_events:
        print(f"Processing PP Communication Events ({len(log_data.pp_comm_events)})...")
    for ev in sorted(log_data.pp_comm_events, key=lambda x: x.get("ts") or 0):
        payload = ev.get("payload", {})
        start_ns = to_ns(payload.get("start_ns"))
        end_ns = to_ns(payload.get("end_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        pid = payload.get("pid")
        tid = payload.get("tid")
        op = payload.get("op", "pp_comm")
        dispatch_key = payload.get("dispatch_key")
        ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
        req_ids = normalize_request_ids(payload.get("req_ids"))
        pp_rank = ranks.get("pp")
        peer_rank = payload.get("peer_rank")
        attribution_source = "payload_req_ids"
        attribution_overlap_ns = 0
        if not req_ids or not dispatch_key:
            inferred_req_ids, inferred_dispatch_key, attribution_overlap_ns = _infer_pp_comm_request_context(
                start_ns,
                end_ns,
                pid,
                ranks,
                dispatch_key,
            )
            if not req_ids and inferred_req_ids:
                req_ids = inferred_req_ids
                attribution_source = "gpu_execute_model_overlap"
            if not dispatch_key and inferred_dispatch_key:
                dispatch_key = inferred_dispatch_key
                if attribution_source == "payload_req_ids":
                    attribution_source = "gpu_execute_model_overlap_dispatch_key"
        if req_ids:
            pp_comm_attr_stats["attributed"] += 1
            if attribution_source != "payload_req_ids":
                pp_comm_attr_stats["overlap_attributed"] += 1
        else:
            pp_comm_attr_stats["unattributed"] += 1
        interval = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "op": op,
            "async_phase": payload.get("async_phase"),
            "comm_id": payload.get("comm_id"),
            "handle_index": payload.get("handle_index"),
            "launch_start_ns": payload.get("launch_start_ns"),
            "launch_end_ns": payload.get("launch_end_ns"),
            "async_elapsed_ns": payload.get("async_elapsed_ns"),
            "dispatch_key": dispatch_key,
            "ranks": ranks,
            "peer_rank": payload.get("peer_rank"),
            "src": payload.get("src"),
            "dst": payload.get("dst"),
            "tensor_bytes": payload.get("tensor_bytes"),
            "tensor_count": payload.get("tensor_count"),
            "attribution_source": attribution_source,
            "attribution_overlap_ns": attribution_overlap_ns,
        }
        for rid in req_ids:
            req_pp_comm_intervals[rid].append(dict(interval))
        timeline.append(WorkerTimelineSpan("communication", start_ns, end_ns, pid, tid,
            {**payload, "req_ids": req_ids, "dispatch_key": dispatch_key,
             "attribution_source": attribution_source, "attribution_overlap_ns": attribution_overlap_ns}))

    if generated_dispatch_slices and log_data.cupti_nonzero_max_end_by_pid:
        execute_total_by_pid = defaultdict(int)
        execute_after_last_cupti_by_pid = defaultdict(int)
        for span in generated_dispatch_slices:
            pid = to_int(span.get("pid"))
            start_ns = to_ns(span.get("start"))
            if pid is None or start_ns is None:
                continue
            execute_total_by_pid[pid] += 1
            if start_ns > log_data.cupti_nonzero_max_end_by_pid.get(pid, 0):
                execute_after_last_cupti_by_pid[pid] += 1

        affected = {
            pid: execute_after_last_cupti_by_pid[pid]
            for pid in execute_after_last_cupti_by_pid
            if execute_after_last_cupti_by_pid[pid] > 0
        }
        if affected:
            print(
                "  [warn] Some Step Execution spans start after the last non-zero CUPTI event "
                "for the same worker pid."
            )
            for pid in sorted(
                affected,
                key=lambda x: affected[x],
                reverse=True,
            ):
                print(
                    f"    pid={pid}: step_exec_after_last_cupti="
                    f"{affected[pid]}/{execute_total_by_pid.get(pid, 0)}, "
                    f"last_nonzero_cupti_end_ns={log_data.cupti_nonzero_max_end_by_pid.get(pid)}"
                )


    return WorkerAnalysis(
        timeline=timeline,
        req_pp_comm_intervals=req_pp_comm_intervals,
        pp_comm_attr_stats=dict(pp_comm_attr_stats),
        dispatch_decode_tokens_cache=dispatch_decode_tokens_cache,
        dispatch_prefill_tokens_cache=dispatch_prefill_tokens_cache,
        dispatch_scheduled_tokens_cache=dispatch_scheduled_tokens_cache,
        generated_dispatch_slices=generated_dispatch_slices,
        req_step_execution_intervals=req_step_execution_intervals,
        req_worker_phase_intervals=req_worker_phase_intervals,
    )
