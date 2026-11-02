"""Task timing events, independent of asyncio patch installation."""
from functools import partial
import os
import threading
import time

from de_latency.runtime.transport import send_event
from de_latency.runtime.request_context import get_request_ctx_from_context, remember_request_name

ENABLE_REQ_STAGE_TIMING = os.getenv("TRACE_REQ_STAGE_TIMING", "1") == "1"


def _emit(kind, timestamp_ns, payload):
    payload.update(pid=os.getpid(), tid=threading.get_native_id(),
                   timestamp_ns=timestamp_ns if timestamp_ns is not None else time.time_ns())
    send_event(source="monkey_patch", event_type=kind, payload=payload)


def _batch_fields(req_ids, task_id, task_name, round_seq):
    return {"req_ids": list(req_ids), "batch_size": len(req_ids),
            "task_id": task_id, "task_name": task_name, "round_seq": round_seq}


def emit_stage_duration(stage, start_ns, end_ns, request_id=None, request_name=None, extra=None):
    if not ENABLE_REQ_STAGE_TIMING:
        return
    if request_id is None or request_name is None:
        ctx_id, ctx_name = get_request_ctx_from_context(None)
        request_id, request_name = request_id or ctx_id, request_name or ctx_name
    payload = {"request_id": request_id,
               "request_name": remember_request_name(request_id, request_name),
               "stage": stage, "start_ns": start_ns, "end_ns": end_ns,
               "duration_ns": max(0, int(end_ns) - int(start_ns))
               if start_ns is not None and end_ns is not None else None}
    # Preserve the existing explicit-extra override semantics.
    payload.update(pid=os.getpid(), tid=threading.get_native_id(),
                   timestamp_ns=end_ns if end_ns is not None else time.time_ns())
    if isinstance(extra, dict):
        payload.update(extra)
    send_event(source="monkey_patch", event_type="req_generate_stage", payload=payload)


def _request_fields(request_id, request_name, task_id, task_name, task_kind):
    return {"request_id": request_id,
            "request_name": remember_request_name(request_id, request_name),
            "task_id": task_id, "task_name": task_name, "task_kind": task_kind}


def emit_coroutine_sched_latency(ready_ts_ns, run_ts_ns, queue_ns, request_id=None,
                                 request_name=None, task_id=None, task_name=None, task_kind="request"):
    if queue_ns is None or queue_ns < 0:
        return
    payload = _request_fields(request_id, request_name, task_id, task_name, task_kind)
    payload.update(ready_ts_ns=ready_ts_ns, run_ts_ns=run_ts_ns, queue_ns=queue_ns)
    _emit("coroutine_sched_latency", run_ts_ns, payload)


def emit_coroutine_exec_slice(start_ns, end_ns, request_id=None, request_name=None,
                              task_id=None, task_name=None, task_kind="request"):
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return
    payload = _request_fields(request_id, request_name, task_id, task_name, task_kind)
    payload.update(start_ns=start_ns, end_ns=end_ns, duration_ns=end_ns - start_ns)
    _emit("coroutine_exec_slice", end_ns, payload)


def emit_batch_wait(component, ready_ts_ns, run_ts_ns, queue_ns, req_ids,
                    task_id=None, task_name=None, round_seq=None):
    if queue_ns is None or queue_ns < 0 or not req_ids:
        return
    payload = _batch_fields(req_ids, task_id, task_name, round_seq)
    payload.update(ready_ts_ns=ready_ts_ns, run_ts_ns=run_ts_ns, queue_ns=queue_ns)
    _emit(f"{component}_sched_latency", run_ts_ns, payload)


def emit_batch_exec(component, start_ns, end_ns, req_ids,
                    task_id=None, task_name=None, round_seq=None):
    if start_ns is None or end_ns is None or end_ns <= start_ns or not req_ids:
        return
    payload = _batch_fields(req_ids, task_id, task_name, round_seq)
    payload.update(start_ns=start_ns, end_ns=end_ns, duration_ns=end_ns - start_ns)
    _emit(f"{component}_exec_slice", end_ns, payload)


emit_output_handler_sched_latency = partial(emit_batch_wait, "output_handler")
emit_output_socket_sched_latency = partial(emit_batch_wait, "output_socket")
emit_output_handler_exec_slice = partial(emit_batch_exec, "output_handler")
emit_output_socket_exec_slice = partial(emit_batch_exec, "output_socket")
