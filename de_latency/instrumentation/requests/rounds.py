"""Attribute output-task scheduling waits to the current and remaining slices."""
import asyncio

from ...runtime.request_context import remember_request_name
from ..task_events import emit_output_handler_sched_latency, emit_output_socket_sched_latency
from .state import tasks


def dedupe(ids):
    seen = set()
    return [rid for rid in ids if rid is not None and not (str(rid) in seen or seen.add(str(rid)))]


def request_ids(outputs):
    return dedupe([output.get("request_id", output.get("req_id")) if isinstance(output, dict)
                   else getattr(output, "request_id", None) for output in (outputs or [])])


def begin_round(slices, task=None):
    task = task or asyncio.current_task()
    if task is None:
        return None
    ids = [request_ids(chunk) for chunk in slices]
    suffix, carry = [[] for _ in ids], []
    for index in range(len(ids) - 1, -1, -1):
        carry = dedupe(ids[index] + carry)
        suffix[index] = carry
    with tasks.lock:
        info = tasks.contexts.get(id(task))
        if not info or info.get("task_kind") != "output_handler":
            return None
        info.update(round_slice_req_ids=ids, round_suffix_req_ids=suffix, round_next_slice_idx=0,
                    current_process_req_ids=[], current_process_size=0)
        if any(ids):
            info["process_seq"] = int(info.get("process_seq") or 0) + 1
        info["active_round_seq"] = info.get("process_seq") if any(ids) else None
        return info["active_round_seq"]


def end_round(task=None):
    task = task or asyncio.current_task()
    if task is not None:
        with tasks.lock:
            info = tasks.contexts.get(id(task))
            if info and info.get("task_kind") == "output_handler":
                info.update(round_slice_req_ids=[], round_suffix_req_ids=[], round_next_slice_idx=0,
                            active_round_seq=None, current_process_req_ids=[], current_process_size=0)


def flush(task, kind, ids):
    with tasks.lock:
        info = tasks.contexts.get(id(task))
        if not info or info.get("task_kind") != kind:
            return 0
        pending = list(info.get("pending_sched_events") or [])
        info["pending_sched_events"] = []
        name = info.get("task_name")
        sequence = info.get("active_round_seq") or info.get("process_seq") if kind == "output_handler" else info.get("process_seq")
    if not ids:
        return 0
    emit = emit_output_handler_sched_latency if kind == "output_handler" else emit_output_socket_sched_latency
    for rid in ids:
        remember_request_name(rid, None)
    for item in pending:
        emit(item.get("ready_ts_ns"), item.get("run_ts_ns"), item.get("queue_ns"), ids, id(task), name, sequence)
    return len(pending)


def begin_slice(outputs, kind):
    task = asyncio.current_task()
    if task is None:
        return [], None
    ids = request_ids(outputs)
    scheduled_ids = ids
    with tasks.lock:
        info = tasks.contexts.get(id(task))
        if not info or info.get("task_kind") != kind:
            return (ids, None) if kind == "output_handler" else ([], None)
        sequence = info.get("active_round_seq") if kind == "output_handler" else None
        slices, suffixes = info.get("round_slice_req_ids") or [], info.get("round_suffix_req_ids") or []
        index = int(info.get("round_next_slice_idx") or 0)
        if kind == "output_handler" and index < len(slices):
            expected = slices[index]
            if not (expected and ids and list(map(str, expected)) != list(map(str, ids))):
                ids = expected or ids
            suffix = suffixes[index] if index < len(suffixes) else []
            scheduled_ids = suffix if suffix and all(str(rid) in set(map(str, suffix)) for rid in ids) else ids
            info["round_next_slice_idx"] = index + 1
        elif ids:
            info["process_seq"] = int(info.get("process_seq") or 0) + 1
            sequence = info["process_seq"]
            if kind == "output_handler":
                info["active_round_seq"] = sequence
        info.update(current_process_req_ids=list(ids), current_process_size=len(ids))
    flush(task, kind, scheduled_ids)
    return ids, sequence


def register_output_task(task, kind, name):
    return tasks.register(task, request_name=name, task_kind=kind, extra={
        "pending_sched_events": [], "current_process_req_ids": [], "current_process_size": 0,
        "process_seq": 0, "active_round_seq": None, "round_slice_req_ids": [],
        "round_suffix_req_ids": [], "round_next_slice_idx": 0})
