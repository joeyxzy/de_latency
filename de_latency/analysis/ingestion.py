"""Log normalization, classification, request aliases and clock samples."""
from collections import defaultdict
from de_latency.protocol import LogDiagnostics, iter_log, analysis_record
from de_latency.analysis.clocks import to_ns, to_int, median_int
from de_latency.analysis.requests import normalize_request_id, normalize_payload_request_fields, canonicalize_payload_request_fields
from de_latency.analysis.correlation import cupti_corr_key
from de_latency.analysis.normalization import infer_trace_origin_ns
from dataclasses import dataclass

@dataclass
class LogData:
    TRACE_TS_ORIGIN_NS: int
    all_kernels_map: dict
    coroutine_events: list
    coroutine_exec_events: list
    cupti_events: list
    cupti_nonzero_max_end_by_pid: dict
    diagnostics: LogDiagnostics
    eager_kernels: list
    ebpf_sched_latency_events: list
    enginecore_loop_events: list
    eventloop_thread_tids: set
    execute_model_span: list
    output_handler_exec_events: list
    output_handler_sched_events: list
    output_socket_exec_events: list
    output_socket_sched_events: list
    pp_comm_events: list
    req_metric_events: list
    req_name_map: dict
    req_stage_events: list
    scheduler_events: list
    scheduler_sample_tokens_events: list
    scheduler_schedule_events: list
    thread_mono_to_wall_offset_by_tid: dict
    thread_roles_by_tid: dict
    worker_events: list
    worker_span_mono_to_wall_offsets: list


def load_log(input_file) -> LogData:
    print(f"🔄 Loading logs from {input_file}...")
    diagnostics = LogDiagnostics()
    raw_events = (analysis_record(event) for event in iter_log(input_file, diagnostics))

    # --- 1. 数据清洗与分类 ---
    # 我们将所有事件分为三类：Worker生命周期、Scheduler、CUPTI
    worker_events = []
    scheduler_events = []
    scheduler_schedule_events = []
    scheduler_sample_tokens_events = []
    cupti_events = []
    req_metric_events = []
    coroutine_events = []
    coroutine_exec_events = []
    output_socket_sched_events = []
    output_socket_exec_events = []
    output_handler_sched_events = []
    output_handler_exec_events = []
    req_stage_events = []
    enginecore_loop_events = []
    pp_comm_events = []
    thread_role_events = []
    execute_model_span = []
    worker_span_mono_to_wall_offsets = []
    ebpf_sched_latency_events = []
    req_name_map = {}
    req_id_aliases = {}
    
    # 辅助字典：存储 Kernel 信息
    all_kernels_map = {}
    eager_kernels = []
    zero_kernel_records = []
    runtime_name_by_corr = {}
    cupti_nonzero_max_end_by_pid = defaultdict(int)

    for entry in raw_events:
        # All historical envelopes have already been normalized by protocol.py.
        payload = entry['payload']
        src, etype, ts = entry['source'], entry['event_type'], entry['timestamp_ns']

        if not payload:
            continue
        payload = normalize_payload_request_fields(payload)
        if not isinstance(payload, dict):
            continue
        rid = payload.get("request_id") or payload.get("req_id")
        rname = payload.get("request_name")
        external_rid = payload.get("external_request_id")
        if rid and external_rid and external_rid != rid:
            req_id_aliases[rid] = external_rid
            req_name_map[external_rid] = external_rid
        elif rid and rname and rname != rid:
            req_id_aliases[rid] = rname
            req_name_map.setdefault(rname, rname)
        if rid and rname:
            # Client-side events can reveal the stable external request id before
            # EngineCore later logs the internal suffixed id as request_name. Keep
            # the external name once discovered so lifecycle and worker phases join.
            if rid not in req_name_map or req_name_map[rid] == rid or rname != rid:
                req_name_map[rid] = rname
        # 兜底：确保 payload 里有 timestamp_ns
        if 'timestamp_ns' not in payload and ts:
            payload['timestamp_ns'] = ts
        
        # 修正：你的代码中 gpu_forward 的 payload 包含 start_ns/end_ns，但 event_type 叫 'gpu_forward'
        # 我们统一整理到 worker_events
        if src == 'monkey_patch':
            # 新增事件类型支持
            if etype in ['worker_preprocess_start', 'gpu_forward_start', 'gpu_forward_end', 
                         'gpu_sample', 'gpu_bookkeeping', 'gpu_execute_model',
                         'gpu_phase_span']:
                worker_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('start_ns', payload.get('timestamp_ns', ts))
                })
            elif etype in ['req_enqueue_scheduler', 'req_scheduler_out_rpc', 'req_step_ready']:
                scheduler_events.append({"type": etype, "payload": payload})
            elif etype == "scheduler_schedule":
                scheduler_schedule_events.append({"type": etype, "payload": payload})
            elif etype == "scheduler_sample_tokens":
                scheduler_sample_tokens_events.append({"type": etype, "payload": payload})
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
            elif etype == "coroutine_exec_slice":
                coroutine_exec_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "output_socket_sched_latency":
                output_socket_sched_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('run_ts_ns', ts)),
                })
            elif etype == "output_socket_exec_slice":
                output_socket_exec_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "output_handler_sched_latency":
                output_handler_sched_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('run_ts_ns', ts)),
                })
            elif etype == "output_handler_exec_slice":
                output_handler_exec_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
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
            elif etype == "pp_comm_span":
                pp_comm_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "thread_role":
                thread_role_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get("timestamp_ns", ts),
                })
            elif etype=="worker_model_execute_span":
                execute_model_span.append({
                    "type": etype,
                    "payload": payload,
                    "ts": ts,
                })
                end_mono_ns = to_ns(payload.get("end_ns"))
                emit_wall_ns = to_ns(ts)
                if (
                    end_mono_ns is not None
                    and emit_wall_ns is not None
                    and emit_wall_ns > end_mono_ns
                ):
                    # 该事件在记录完 monotonic end_ns 后立刻用 wall clock 发出，
                    # 可作为稳定的 mono->wall 对齐样本。
                    worker_span_mono_to_wall_offsets.append(emit_wall_ns - end_mono_ns)

        elif src == 'cupti':
            corr_id = payload.get('correlationId')
            pid = to_int(payload.get("pid"))
            start_ns = to_ns(payload.get("start_ns"))
            end_ns = to_ns(payload.get("end_ns"))
            if (
                pid is not None
                and start_ns is not None
                and end_ns is not None
                and start_ns > 0
                and end_ns > 0
                and end_ns > cupti_nonzero_max_end_by_pid[pid]
            ):
                cupti_nonzero_max_end_by_pid[pid] = end_ns
            if etype in ['runtime', 'driver', 'marker']:
                cupti_events.append(payload)
                if etype == 'runtime':
                    runtime_name_by_corr[
                        cupti_corr_key(payload, corr_id=corr_id, pid_override=pid)
                    ] = payload.get("name", "")
            elif etype in ['kernel', 'memset', 'memcpy']:
                kernel_key = cupti_corr_key(payload, corr_id=corr_id)
                if kernel_key not in all_kernels_map:
                    all_kernels_map[kernel_key] = []
                all_kernels_map[kernel_key].append(payload)
                if start_ns is not None and start_ns > 0:
                    eager_kernels.append(payload)
                elif etype == 'kernel':
                    zero_kernel_records.append(payload)

        elif src == 'ebpf':
            if etype == 'sched_latency':
                ebpf_sched_latency_events.append(payload)

    if len(diagnostics.sessions) > 1:
        raise ValueError("multiple sessions in one log; analyze each session separately")

    def _resolve_request_alias(rid):
        rid = normalize_request_id(rid)
        if not rid:
            return rid
        seen = set()
        while rid in req_id_aliases and rid not in seen:
            seen.add(rid)
            nxt = normalize_request_id(req_id_aliases.get(rid))
            if not nxt:
                break
            rid = nxt
        return rid

    def _canonicalize_event_payloads(events):
        for ev in events:
            payload = ev.get("payload") if isinstance(ev, dict) else None
            canonicalize_payload_request_fields(payload, _resolve_request_alias)

    for _events in (
        worker_events,
        scheduler_events,
        req_metric_events,
        coroutine_events,
        coroutine_exec_events,
        output_socket_sched_events,
        output_socket_exec_events,
        output_handler_sched_events,
        output_handler_exec_events,
        req_stage_events,
        enginecore_loop_events,
        pp_comm_events,
        thread_role_events,
        execute_model_span,
    ):
        _canonicalize_event_payloads(_events)

    canonical_req_name_map = {}
    for rid, name in req_name_map.items():
        canonical_rid = _resolve_request_alias(rid)
        canonical_name = _resolve_request_alias(name)
        if canonical_rid:
            canonical_req_name_map[canonical_rid] = canonical_name or canonical_rid
    req_name_map = canonical_req_name_map
    if req_id_aliases:
        print(f"Canonicalized {len(req_id_aliases)} internal request ids to external request ids.")

    if zero_kernel_records:
        zero_kernel_count_by_pid = defaultdict(int)
        zero_kernel_graph_count_by_pid = defaultdict(int)
        for payload in zero_kernel_records:
            pid = to_int(payload.get("pid"))
            if pid is None:
                continue
            zero_kernel_count_by_pid[pid] += 1
            runtime_name = runtime_name_by_corr.get(
                cupti_corr_key(payload, pid_override=pid),
                "",
            )
            if "graph" in str(runtime_name).lower():
                zero_kernel_graph_count_by_pid[pid] += 1

        zero_total = sum(zero_kernel_count_by_pid.values())
        zero_graph_total = sum(zero_kernel_graph_count_by_pid.values())
        graph_ratio = (zero_graph_total / zero_total) if zero_total else 0.0
        print(
            "  [warn] Detected "
            f"{zero_total} kernel activity records with start_ns/end_ns <= 0; "
            f"{zero_graph_total} ({graph_ratio:.1%}) are linked to CUDA Graph launches."
        )
        for pid in sorted(
            zero_kernel_count_by_pid,
            key=lambda x: zero_kernel_count_by_pid[x],
            reverse=True,
        ):
            zero_cnt = zero_kernel_count_by_pid[pid]
            graph_cnt = zero_kernel_graph_count_by_pid.get(pid, 0)
            ratio = (graph_cnt / zero_cnt) if zero_cnt else 0.0
            print(
                f"    pid={pid}: zero_ts_kernels={zero_cnt}, "
                f"graph_linked={graph_cnt} ({ratio:.1%})"
            )
        print(
            "    These records cannot be placed on the wall-clock timeline. "
            "Blank later Step Execution spans usually mean CUDA Graph replay, not a simple global offset."
        )

    TRACE_TS_ORIGIN_NS = infer_trace_origin_ns(
        worker_events=worker_events,
        scheduler_events=scheduler_events,
        coroutine_events=coroutine_events,
        coroutine_exec_events=coroutine_exec_events,
        output_socket_sched_events=output_socket_sched_events,
        output_socket_exec_events=output_socket_exec_events,
        output_handler_sched_events=output_handler_sched_events,
        output_handler_exec_events=output_handler_exec_events,
        req_stage_events=req_stage_events,
        enginecore_loop_events=enginecore_loop_events,
        pp_comm_events=pp_comm_events,
        execute_model_span=execute_model_span,
        cupti_events=cupti_events,
    )
    # 将新事件类型补入 origin 推断（这里直接扩展 min_ns）
    for ev in scheduler_schedule_events:
        p = ev.get("payload", {})
        for k in ("start_ns", "end_ns", "timestamp_ns"):
            v = to_ns(p.get(k))
            if v is not None and v > 0:
                TRACE_TS_ORIGIN_NS = v if TRACE_TS_ORIGIN_NS is None or v < TRACE_TS_ORIGIN_NS else TRACE_TS_ORIGIN_NS
    for ev in scheduler_sample_tokens_events:
        v = to_ns(ev.get("payload", {}).get("timestamp_ns"))
        if v is not None and v > 0:
            TRACE_TS_ORIGIN_NS = v if TRACE_TS_ORIGIN_NS is None or v < TRACE_TS_ORIGIN_NS else TRACE_TS_ORIGIN_NS
    if TRACE_TS_ORIGIN_NS:
        print(f"Trace time origin (wall ns): {TRACE_TS_ORIGIN_NS}")

    thread_roles_by_tid = {}
    thread_mono_to_wall_offsets = defaultdict(list)
    eventloop_thread_tids = set()
    for item in thread_role_events:
        payload = item.get("payload", {})
        tid = to_int(payload.get("tid"))
        pid = to_int(payload.get("pid"))
        role = payload.get("role")
        wall_ts_ns = to_ns(payload.get("timestamp_ns", item.get("ts")))
        mono_ts_ns = to_ns(payload.get("mono_timestamp_ns"))
        if tid is None or not role:
            continue

        info = thread_roles_by_tid.setdefault(
            tid,
            {
                "pid": pid,
                "roles": set(),
            },
        )
        if info.get("pid") is None and pid is not None:
            info["pid"] = pid
        info["roles"].add(role)
        if role == "asyncio_eventloop":
            eventloop_thread_tids.add(tid)
        if wall_ts_ns is not None and mono_ts_ns is not None and wall_ts_ns > mono_ts_ns:
            thread_mono_to_wall_offsets[tid].append(wall_ts_ns - mono_ts_ns)

    thread_mono_to_wall_offset_by_tid = {
        tid: median_int(offsets)
        for tid, offsets in thread_mono_to_wall_offsets.items()
        if offsets
    }


    return LogData(
        TRACE_TS_ORIGIN_NS=TRACE_TS_ORIGIN_NS,
        all_kernels_map=all_kernels_map,
        coroutine_events=coroutine_events,
        coroutine_exec_events=coroutine_exec_events,
        cupti_events=cupti_events,
        cupti_nonzero_max_end_by_pid=cupti_nonzero_max_end_by_pid,
        diagnostics=diagnostics,
        eager_kernels=eager_kernels,
        ebpf_sched_latency_events=ebpf_sched_latency_events,
        enginecore_loop_events=enginecore_loop_events,
        eventloop_thread_tids=eventloop_thread_tids,
        execute_model_span=execute_model_span,
        output_handler_exec_events=output_handler_exec_events,
        output_handler_sched_events=output_handler_sched_events,
        output_socket_exec_events=output_socket_exec_events,
        output_socket_sched_events=output_socket_sched_events,
        pp_comm_events=pp_comm_events,
        req_metric_events=req_metric_events,
        req_name_map=req_name_map,
        req_stage_events=req_stage_events,
        scheduler_events=scheduler_events,
        scheduler_sample_tokens_events=scheduler_sample_tokens_events,
        scheduler_schedule_events=scheduler_schedule_events,
        thread_mono_to_wall_offset_by_tid=thread_mono_to_wall_offset_by_tid,
        thread_roles_by_tid=thread_roles_by_tid,
        worker_events=worker_events,
        worker_span_mono_to_wall_offsets=worker_span_mono_to_wall_offsets,
    )
