"""Worker and PP communication tracks from attributed source spans."""
import json
from collections import defaultdict
from de_latency.analysis.normalization import count_prefill_decode_tokens
from .styles import describe_gpu_phase

def append_execution(trace_events, writer, payload, t_start, t_end, pid, tid, req_ids, batch_size, dispatch_key, ranks):
    create_perfetto_event = writer.event
    # 画一个透明容器（浅灰色背景）
    class_path = payload.get("class_path", "") or ""
    if "gpu.model_runner" in class_path:
        event_name = "GPUModelRunner (V2).execute_model"
    elif "gpu_model_runner" in class_path:
        event_name = "GPUModelRunner (V1).execute_model"
    else:
        event_name = "Step Execution"
    sched_tokens = payload.get("scheduled_tokens_by_req")
    disp_args = {
        "req_ids": "\n".join(req_ids),
        "batch_size": batch_size,
        "class_path": class_path,
        "dispatch_key": dispatch_key,
        "dp_rank": ranks.get("dp"),
        "pp_rank": ranks.get("pp"),
        "tp_rank": ranks.get("tp"),
    }
    if isinstance(sched_tokens, dict):
        disp_args["scheduled_tokens_by_req"] = json.dumps(sched_tokens, separators=(",", ":"))
        prefill_tokens, decode_tokens = count_prefill_decode_tokens(sched_tokens)
        disp_args["prefill_tokens"] = prefill_tokens
        disp_args["decode_tokens"] = decode_tokens
    num_computed = payload.get("num_computed_by_req")
    if isinstance(num_computed, dict):
        disp_args["num_computed_by_req"] = json.dumps(num_computed, separators=(",", ":"))
    trace_events.append(create_perfetto_event(
        name=event_name, cat="worker", ph="X",
        ts=t_start, dur=t_end - t_start,
        pid=pid, tid=tid,
        args=disp_args,
        cname="background"
    ))


def append_phase(trace_events, writer, payload, phase_name, t_start, t_end, pid, tid, req_ids, batch_size, input_type, dispatch_key, ranks):
    create_perfetto_event = writer.event
    display_name, cat, cname = describe_gpu_phase(phase_name)
    args = {
        "phase": phase_name,
        "req_ids": "\n".join(req_ids),
        "dispatch_key": dispatch_key,
        "dp_rank": ranks.get("dp"),
        "pp_rank": ranks.get("pp"),
        "tp_rank": ranks.get("tp"),
    }
    if batch_size is not None:
        args["batch_size"] = batch_size
    if input_type:
        args["input_type"] = input_type
    sched_tokens = payload.get("scheduled_tokens_by_req")
    if isinstance(sched_tokens, dict):
        args["scheduled_tokens_by_req"] = json.dumps(sched_tokens, separators=(",", ":"))
        prefill_tokens, decode_tokens = count_prefill_decode_tokens(sched_tokens)
        args["prefill_tokens"] = prefill_tokens
        args["decode_tokens"] = decode_tokens
    trace_events.append(create_perfetto_event(
        name=display_name,
        cat=cat, ph="X", ts=t_start, dur=t_end - t_start,
        pid=pid, tid=tid,
        args=args,
        cname=cname
    ))


def append_preprocess(trace_events, writer, state, t1, dur, pid, tid):
    create_perfetto_event = writer.event
    trace_events.append(create_perfetto_event(
                        name="Worker Preprocess",
                        cat="python", ph="X", ts=t1, dur=dur,
                        pid=pid, tid=tid,
                        args={"req_ids": "\n".join(state['req_ids']), "desc": "Prepare Inputs"},
                        cname="good"  # 黄绿色
                    ))

def append_forward(trace_events, writer, state, t2, dur, pid, tid):
    create_perfetto_event = writer.event
    trace_events.append(create_perfetto_event(
                        name="Model Forward",
                        cat="python", ph="X", ts=t2, dur=dur,
                        pid=pid, tid=tid,
                        args={"req_ids": "\n".join(state['req_ids']), "batch_size": state.get('batch_size', 0)},
                        cname="olive"  # 橄榄绿
                    ))

def append_sample(trace_events, writer, req_ids, t_start, t_end, pid, tid):
    create_perfetto_event = writer.event
    trace_events.append(create_perfetto_event(
                        name="Sampling",
                        cat="python", ph="X", ts=t_start, dur=t_end - t_start,
                        pid=pid, tid=tid,
                        args={"req_ids": "\n".join(req_ids)},
                        cname="mauve"  # 紫红色
                    ))

def append_bookkeep(trace_events, writer, req_ids, t_start, t_end, pid, tid):
    create_perfetto_event = writer.event
    trace_events.append(create_perfetto_event(
                        name="Bookkeeping/Sync",
                        cat="python", ph="X", ts=t_start, dur=t_end - t_start,
                        pid=pid, tid=tid,
                        args={"req_ids": "\n".join(req_ids)},
                        cname="cyan"  # 青色
                    ))

def append_communication(trace_events, writer, payload, start_ns, end_ns, pid, tid, op, dispatch_key, ranks, req_ids, attribution_source, attribution_overlap_ns):
    create_perfetto_event = writer.event
    trace_events.append(create_perfetto_event(
        name=f"PP Comm: {op}",
        cat="pp_comm",
        ph="X",
        ts=start_ns,
        dur=end_ns - start_ns,
        pid=pid,
        tid=tid,
        args={
            "dispatch_key": dispatch_key,
            "req_ids": "\n".join(req_ids),
            "attribution_source": attribution_source,
            "attribution_overlap_ns": attribution_overlap_ns,
            "dp_rank": ranks.get("dp"),
            "pp_rank": ranks.get("pp"),
            "tp_rank": ranks.get("tp"),
            "peer_rank": payload.get("peer_rank"),
            "src": payload.get("src"),
            "dst": payload.get("dst"),
            "comm_id": payload.get("comm_id"),
            "async_phase": payload.get("async_phase"),
            "handle_index": payload.get("handle_index"),
            "handle_count": payload.get("handle_count"),
            "async_elapsed_ns": payload.get("async_elapsed_ns"),
            "tensor_count": payload.get("tensor_count"),
            "tensor_bytes": payload.get("tensor_bytes"),
        },
        cname="rail_load",
    ))



def append_worker_trace(trace_events, writer, workers):
    flow_points = defaultdict(dict)
    for span in workers.timeline:
        p = span.payload
        ids = p.get("req_ids", [])
        ranks = p.get("ranks") if isinstance(p.get("ranks"), dict) else {}
        common = (span.start_ns, span.end_ns, span.pid, span.tid)
        if span.kind == "execution":
            append_execution(trace_events, writer, p, *common, ids, p.get("batch_size", 0), p.get("dispatch_key"), ranks)
        elif span.kind == "phase":
            append_phase(trace_events, writer, p, p.get("phase"), *common, ids,
                         p.get("batch_size"), p.get("input_type"), p.get("dispatch_key"), ranks)
        elif span.kind == "preprocess":
            append_preprocess(trace_events, writer, p, span.start_ns, span.end_ns-span.start_ns, span.pid, span.tid)
        elif span.kind == "forward":
            append_forward(trace_events, writer, p, span.start_ns, span.end_ns-span.start_ns, span.pid, span.tid)
        elif span.kind == "sample":
            append_sample(trace_events, writer, ids, *common)
        elif span.kind == "bookkeep":
            append_bookkeep(trace_events, writer, ids, *common)
        elif span.kind == "communication":
            op, key, pp, peer = p.get("op", "pp_comm"), p.get("dispatch_key"), ranks.get("pp"), p.get("peer_rank")
            append_communication(trace_events, writer, p, *common, op, key, ranks, ids,
                                 p["attribution_source"], p["attribution_overlap_ns"])
            if op in ("send_tensor_dict", "send_tensor", "isend_tensor_dict_launch"):
                flow_points[key, pp, peer]["send"] = span.start_ns, span.pid, span.tid
            elif op in ("recv_tensor_dict", "recv_tensor", "irecv_tensor_dict_launch"):
                flow_points[key, peer, pp]["recv"] = span.start_ns, span.pid, span.tid
    for index, points in enumerate(flow_points.values(), start=1):
        if "send" in points and "recv" in points:
            trace_events.append(writer.flow("s", *points["send"], 700000 + index))
            trace_events.append(writer.flow("f", *points["recv"], 700000 + index))
