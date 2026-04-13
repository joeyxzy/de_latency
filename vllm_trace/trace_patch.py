import asyncio
import time
import json
from functools import wraps
from datetime import datetime
import inspect
import logging
import os
import threading
import weakref
import zmq
from encode import make_metadata, metadata_to_bytes 
from request_context import (
    bind_request_context,
    reset_request_context,
    remember_request_name,
    get_request_ctx_from_context,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
patch_logger = logging.getLogger("vllm_patch")
patch_logger.info("--- VLLM Patch Logger Initialized ---")

coroutine_timers = {}
task_request_ctx = {}
ctx_lock = threading.Lock()
DEBUG_SCHED_PATCH = os.getenv("TRACE_PATCH_DEBUG", "0") == "1"
ENABLE_REQ_STAGE_TIMING = os.getenv("TRACE_REQ_STAGE_TIMING", "1") == "1"
debug_counters = {
    "call_soon_hits": 0,
    "call_soon_threadsafe_hits": 0,
    "emit_hits": 0,
    "skip_no_ctx": 0,
}
SOCK_ADDR = os.getenv("TRACER_ZMQ_ADDR", "ipc:///tmp/tracer.sock")
thread_role_announced = set()

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect(SOCK_ADDR)


def _get_native_tid():
    try:
        return threading.get_native_id()
    except AttributeError:
        return threading.get_ident()

def send_event(source, event_type, payload=None, extra=None):
    meta = make_metadata(source, event_type, extra=extra)
    try:
        # 如果 payload 是结构化数据且小，直接将其合并到 meta（单帧）
        if payload is not None and isinstance(payload, dict) and len(json.dumps(payload)) < 1500:
            meta['payload'] = payload
            sock.send_json(meta, flags=zmq.DONTWAIT)
        else:
            # multipart: meta + binary payload
            meta_b = metadata_to_bytes(meta)
            if payload is None:
                payload_b = b""
            elif isinstance(payload, (bytes, bytearray, memoryview)):
                payload_b = bytes(payload)
            else:
                payload_b = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            sock.send_multipart([meta_b, payload_b], flags=zmq.DONTWAIT)
    except zmq.error.Again:
        # 发送队列满时直接丢弃，绝不影响业务主流程
        pass
    except Exception:
        # 防御性兜底：trace 不得打断推理
        pass


def emit_thread_role(role, extra=None):
    pid = os.getpid()
    tid = _get_native_tid()
    key = (pid, tid, role)
    with ctx_lock:
        if key in thread_role_announced:
            return
        thread_role_announced.add(key)

    payload = {
        "role": role,
        "pid": pid,
        "tid": tid,
        "thread_ident": threading.get_ident(),
        "mono_timestamp_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC),
        "timestamp_ns": time.time_ns(),
    }
    if extra and isinstance(extra, dict):
        payload.update(extra)
    send_event(
        source="monkey_patch",
        event_type="thread_role",
        payload=payload,
    )

def get_arg_value(func, arg_name, args, kwargs):
    """从 args/kwargs 自动解析参数"""
    if arg_name in kwargs:
        return kwargs[arg_name]
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # 对类实例方法/类方法，wrapper 里通常已经显式吃掉了 self/cls，
        # 这里需要在 bind 时补一个占位，避免后续位置参数整体左移。
        candidate_arg_sets = []
        if params and params[0].name in {"self", "cls"}:
            candidate_arg_sets.append((None, *args))
        candidate_arg_sets.append(tuple(args))

        for candidate_args in candidate_arg_sets:
            try:
                bound = sig.bind_partial(*candidate_args, **kwargs)
                bound.apply_defaults()
                if arg_name in bound.arguments:
                    return bound.arguments.get(arg_name, None)
            except TypeError:
                continue
        return None
    except:
        return None


def emit_stage_duration(stage, start_ns, end_ns, request_id=None, request_name=None, extra=None):
    """Emit a per-request stage duration event for generate->enqueue breakdown."""
    if not ENABLE_REQ_STAGE_TIMING:
        return
    rid = request_id
    rname = request_name
    if rid is None or rname is None:
        ctx_rid, ctx_rname = get_request_ctx_from_context(None)
        rid = rid or ctx_rid
        rname = rname or ctx_rname
    rname = remember_request_name(rid, rname)
    dur_ns = None
    if start_ns is not None and end_ns is not None:
        dur_ns = max(0, int(end_ns) - int(start_ns))
    payload = {
        "request_id": rid,
        "request_name": rname,
        "stage": stage,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": dur_ns,
        "pid": os.getpid(),
        "tid": _get_native_tid(),
        "timestamp_ns": end_ns if end_ns is not None else time.time_ns(),
    }
    if extra and isinstance(extra, dict):
        payload.update(extra)
    send_event(
        source="monkey_patch",
        event_type="req_generate_stage",
        payload=payload,
    )


def emit_output_handler_sched_latency(
    ready_ts_ns,
    run_ts_ns,
    queue_ns,
    req_ids,
    task_id=None,
    task_name=None,
    round_seq=None,
):
    if queue_ns is None or queue_ns < 0 or not req_ids:
        return
    send_event(
        source="monkey_patch",
        event_type="output_handler_sched_latency",
        payload={
            "ready_ts_ns": ready_ts_ns,
            "run_ts_ns": run_ts_ns,
            "queue_ns": queue_ns,
            "req_ids": list(req_ids),
            "batch_size": len(req_ids),
            "task_id": task_id,
            "task_name": task_name,
            "round_seq": round_seq,
            "pid": os.getpid(),
            "tid": _get_native_tid(),
            "timestamp_ns": run_ts_ns if run_ts_ns is not None else time.time_ns(),
        },
    )


def emit_coroutine_sched_latency(
    ready_ts_ns,
    run_ts_ns,
    queue_ns,
    request_id=None,
    request_name=None,
    task_id=None,
    task_name=None,
    task_kind="request",
):
    if queue_ns is None or queue_ns < 0:
        return
    rid = request_id
    rname = remember_request_name(rid, request_name)
    send_event(
        source="monkey_patch",
        event_type="coroutine_sched_latency",
        payload={
            "request_id": rid,
            "request_name": rname,
            "task_id": task_id,
            "task_name": task_name,
            "task_kind": task_kind,
            "ready_ts_ns": ready_ts_ns,
            "run_ts_ns": run_ts_ns,
            "queue_ns": queue_ns,
            "pid": os.getpid(),
            "tid": _get_native_tid(),
            "timestamp_ns": run_ts_ns if run_ts_ns is not None else time.time_ns(),
        },
    )


def emit_coroutine_exec_slice(
    start_ns,
    end_ns,
    request_id=None,
    request_name=None,
    task_id=None,
    task_name=None,
    task_kind="request",
):
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return
    rid = request_id
    rname = remember_request_name(rid, request_name)
    send_event(
        source="monkey_patch",
        event_type="coroutine_exec_slice",
        payload={
            "request_id": rid,
            "request_name": rname,
            "task_id": task_id,
            "task_name": task_name,
            "task_kind": task_kind,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "pid": os.getpid(),
            "tid": _get_native_tid(),
            "timestamp_ns": end_ns,
        },
    )


def emit_output_handler_exec_slice(
    start_ns,
    end_ns,
    req_ids,
    task_id=None,
    task_name=None,
    round_seq=None,
):
    if start_ns is None or end_ns is None or end_ns <= start_ns or not req_ids:
        return
    send_event(
        source="monkey_patch",
        event_type="output_handler_exec_slice",
        payload={
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "req_ids": list(req_ids),
            "batch_size": len(req_ids),
            "task_id": task_id,
            "task_name": task_name,
            "round_seq": round_seq,
            "pid": os.getpid(),
            "tid": _get_native_tid(),
            "timestamp_ns": end_ns,
        },
    )


def emit_output_socket_sched_latency(
    ready_ts_ns,
    run_ts_ns,
    queue_ns,
    req_ids,
    task_id=None,
    task_name=None,
    round_seq=None,
):
    if queue_ns is None or queue_ns < 0 or not req_ids:
        return
    send_event(
        source="monkey_patch",
        event_type="output_socket_sched_latency",
        payload={
            "ready_ts_ns": ready_ts_ns,
            "run_ts_ns": run_ts_ns,
            "queue_ns": queue_ns,
            "req_ids": list(req_ids),
            "batch_size": len(req_ids),
            "task_id": task_id,
            "task_name": task_name,
            "round_seq": round_seq,
            "pid": os.getpid(),
            "tid": _get_native_tid(),
            "timestamp_ns": run_ts_ns if run_ts_ns is not None else time.time_ns(),
        },
    )


def emit_output_socket_exec_slice(
    start_ns,
    end_ns,
    req_ids,
    task_id=None,
    task_name=None,
    round_seq=None,
):
    if start_ns is None or end_ns is None or end_ns <= start_ns or not req_ids:
        return
    send_event(
        source="monkey_patch",
        event_type="output_socket_exec_slice",
        payload={
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "req_ids": list(req_ids),
            "batch_size": len(req_ids),
            "task_id": task_id,
            "task_name": task_name,
            "round_seq": round_seq,
            "pid": os.getpid(),
            "tid": _get_native_tid(),
            "timestamp_ns": end_ns,
        },
    )


def _debug_log(msg):
    if DEBUG_SCHED_PATCH:
        patch_logger.info(f"[SCHED_DEBUG] {msg}")


def _extract_task_from_callback(callback):
    if callback is None:
        return None
    task_obj = getattr(callback, "__self__", None)
    if isinstance(task_obj, asyncio.Task):
        return task_obj
    if task_obj is not None and task_obj.__class__.__name__.endswith("Task"):
        return task_obj
    return None


def _register_current_task(request_id, request_name, task_kind="request"):
    return _register_task_context(
        asyncio.current_task(),
        request_id=request_id,
        request_name=request_name,
        task_kind=task_kind,
    )


def _cleanup_task_context(task):
    _unregister_task(task)


def _register_task_context(task, request_id=None, request_name=None, task_kind="request", extra=None):
    if task is None:
        return None
    info = {
        "request_id": request_id,
        "request_name": request_name or request_id,
        "task_name": task.get_name() if hasattr(task, "get_name") else None,
        "task_kind": task_kind,
    }
    if extra and isinstance(extra, dict):
        info.update(extra)
    with ctx_lock:
        task_request_ctx[id(task)] = info
    if not getattr(task, "_de_latency_ctx_cleanup_installed", False):
        try:
            task.add_done_callback(_cleanup_task_context)
            setattr(task, "_de_latency_ctx_cleanup_installed", True)
        except Exception:
            pass
    return task


def _unregister_task(task):
    if task is None:
        return
    with ctx_lock:
        task_request_ctx.pop(id(task), None)


def _resolve_schedule_context(task_obj, context_obj):
    task_id = id(task_obj) if task_obj is not None else None
    req_id, req_name = get_request_ctx_from_context(context_obj)
    if not req_id:
        # 回退到当前上下文，覆盖 context=None 的场景
        cur_req_id, cur_req_name = get_request_ctx_from_context(None)
        req_id = req_id or cur_req_id
        req_name = req_name or cur_req_name
    if task_id and (not req_id or not req_name):
        with ctx_lock:
            info = task_request_ctx.get(task_id)
        if info:
            req_id = req_id or info.get("request_id")
            req_name = req_name or info.get("request_name")
    return task_id, req_id, req_name


def _extract_req_ids_from_engine_core_outputs(engine_core_outputs):
    if engine_core_outputs is None:
        return []
    try:
        if len(engine_core_outputs) == 0:
            return []
    except TypeError:
        pass
    req_ids = []
    seen = set()
    for output in engine_core_outputs:
        rid = getattr(output, "request_id", None)
        if rid is None and isinstance(output, dict):
            rid = output.get("request_id") or output.get("req_id")
        if rid is None:
            continue
        rid_key = str(rid)
        if rid_key in seen:
            continue
        seen.add(rid_key)
        req_ids.append(rid)
    return req_ids


def _extract_req_ids_from_engine_core_output_batch(engine_core_outputs):
    if engine_core_outputs is None:
        return []
    outputs = getattr(engine_core_outputs, "outputs", None)
    if outputs is None:
        return []
    return _extract_req_ids_from_engine_core_outputs(outputs)


def _flush_pending_output_handler_sched_events(task=None, req_ids=None):
    task = task or asyncio.current_task()
    if task is None:
        return 0
    task_id = id(task)
    with ctx_lock:
        info = task_request_ctx.get(task_id)
        if not info or info.get("task_kind") != "output_handler":
            return 0
        target_req_ids = list(req_ids if req_ids is not None else info.get("current_process_req_ids") or [])
        pending = list(info.get("pending_sched_events") or [])
        if not pending:
            return 0
        info["pending_sched_events"] = []
        if req_ids is not None:
            info["current_process_req_ids"] = list(target_req_ids)
            info["current_process_size"] = len(target_req_ids)
        task_name = info.get("task_name")
        round_seq = info.get("process_seq")
    if not target_req_ids:
        return 0
    for rid in target_req_ids:
        remember_request_name(rid, None)
    for item in pending:
        emit_output_handler_sched_latency(
            ready_ts_ns=item.get("ready_ts_ns"),
            run_ts_ns=item.get("run_ts_ns"),
            queue_ns=item.get("queue_ns"),
            req_ids=target_req_ids,
            task_id=task_id,
            task_name=task_name,
            round_seq=round_seq,
        )
    return len(pending)


def _flush_pending_output_socket_sched_events(task=None, req_ids=None):
    task = task or asyncio.current_task()
    if task is None:
        return 0
    task_id = id(task)
    with ctx_lock:
        info = task_request_ctx.get(task_id)
        if not info or info.get("task_kind") != "output_socket":
            return 0
        target_req_ids = list(req_ids if req_ids is not None else info.get("current_process_req_ids") or [])
        pending = list(info.get("pending_sched_events") or [])
        if not pending:
            return 0
        info["pending_sched_events"] = []
        if req_ids is not None:
            info["current_process_req_ids"] = list(target_req_ids)
            info["current_process_size"] = len(target_req_ids)
        task_name = info.get("task_name")
        round_seq = info.get("process_seq")
    if not target_req_ids:
        return 0
    for rid in target_req_ids:
        remember_request_name(rid, None)
    for item in pending:
        emit_output_socket_sched_latency(
            ready_ts_ns=item.get("ready_ts_ns"),
            run_ts_ns=item.get("run_ts_ns"),
            queue_ns=item.get("queue_ns"),
            req_ids=target_req_ids,
            task_id=task_id,
            task_name=task_name,
            round_seq=round_seq,
        )
    return len(pending)


def _begin_output_handler_process_outputs(engine_core_outputs):
    task = asyncio.current_task()
    if task is None:
        return [], None
    task_id = id(task)
    req_ids = _extract_req_ids_from_engine_core_outputs(engine_core_outputs)
    process_seq = None
    with ctx_lock:
        info = task_request_ctx.get(task_id)
        if not info or info.get("task_kind") != "output_handler":
            return [], None
        if req_ids:
            info["current_process_req_ids"] = list(req_ids)
            info["current_process_size"] = len(req_ids)
            info["process_seq"] = int(info.get("process_seq") or 0) + 1
            process_seq = info["process_seq"]
        else:
            info["current_process_req_ids"] = []
            info["current_process_size"] = 0
    _flush_pending_output_handler_sched_events(task=task, req_ids=req_ids)
    return req_ids, process_seq


def _begin_output_socket_process_outputs(engine_core_outputs):
    task = asyncio.current_task()
    if task is None:
        return [], None
    task_id = id(task)
    req_ids = _extract_req_ids_from_engine_core_output_batch(engine_core_outputs)
    process_seq = None
    with ctx_lock:
        info = task_request_ctx.get(task_id)
        if not info or info.get("task_kind") != "output_socket":
            return [], None
        if req_ids:
            info["current_process_req_ids"] = list(req_ids)
            info["current_process_size"] = len(req_ids)
            info["process_seq"] = int(info.get("process_seq") or 0) + 1
            process_seq = info["process_seq"]
        else:
            info["current_process_req_ids"] = []
            info["current_process_size"] = 0
    _flush_pending_output_socket_sched_events(task=task, req_ids=req_ids)
    return req_ids, process_seq


def _make_scheduled_callback(original_callback, ready_ts_ns, task_id, request_id, request_name):
    def wrapped_callback(*cb_args):
        emit_thread_role("asyncio_eventloop")
        ctx_info = None
        if task_id is not None:
            with ctx_lock:
                ctx_info = task_request_ctx.get(task_id)
        if not ctx_info and request_id:
            ctx_info = {
                "request_id": request_id,
                "request_name": request_name or request_id,
                "task_name": None,
            }

        run_ts_ns = time.time_ns()
        if ctx_info:
            queue_ns = run_ts_ns - ready_ts_ns
            if queue_ns >= 0:
                debug_counters["emit_hits"] += 1
                if ctx_info.get("task_kind") in ("output_handler", "output_socket") and task_id is not None:
                    with ctx_lock:
                        info = task_request_ctx.get(task_id)
                        if info is not None:
                            pending = info.setdefault("pending_sched_events", [])
                            pending.append({
                                "ready_ts_ns": ready_ts_ns,
                                "run_ts_ns": run_ts_ns,
                                "queue_ns": queue_ns,
                            })
                else:
                    emit_coroutine_sched_latency(
                        ready_ts_ns=ready_ts_ns,
                        run_ts_ns=run_ts_ns,
                        queue_ns=queue_ns,
                        request_id=ctx_info.get("request_id"),
                        request_name=ctx_info.get("request_name"),
                        task_id=task_id,
                        task_name=ctx_info.get("task_name"),
                        task_kind=ctx_info.get("task_kind") or "request",
                    )
                if DEBUG_SCHED_PATCH and debug_counters["emit_hits"] <= 20:
                    _debug_log(
                        f"emit coroutine_sched_latency hit#{debug_counters['emit_hits']} "
                        f"rid={ctx_info.get('request_id')} q_ns={queue_ns}"
                    )
        else:
            debug_counters["skip_no_ctx"] += 1
            if DEBUG_SCHED_PATCH and debug_counters["skip_no_ctx"] <= 20:
                cb_type = type(original_callback).__name__ if original_callback is not None else "None"
                _debug_log(
                    f"skip callback(no_ctx) hit#{debug_counters['skip_no_ctx']} cb_type={cb_type} task_id={task_id}"
                )
        try:
            return original_callback(*cb_args)
        finally:
            end_ts_ns = time.time_ns()
            if ctx_info and end_ts_ns > run_ts_ns:
                task_kind = ctx_info.get("task_kind")
                if task_kind not in ("output_handler", "output_socket", "generate_task"):
                    emit_coroutine_exec_slice(
                        start_ns=run_ts_ns,
                        end_ns=end_ts_ns,
                        request_id=ctx_info.get("request_id"),
                        request_name=ctx_info.get("request_name"),
                        task_id=task_id,
                        task_name=ctx_info.get("task_name"),
                        task_kind=ctx_info.get("task_kind") or "request",
                    )

    setattr(wrapped_callback, "_de_latency_sched_wrapped", True)
    return wrapped_callback


def _call_with_optional_context(func, loop_obj, callback, args, context):
    try:
        return func(loop_obj, callback, *args, context=context)
    except TypeError:
        # 兼容不支持 context= 的 loop 实现
        return func(loop_obj, callback, *args)


def _patch_loop_class(loop_cls):
    if getattr(loop_cls, "_de_latency_sched_patch_installed", False):
        return False
    if not hasattr(loop_cls, "call_soon"):
        return False

    original_call_soon = loop_cls.call_soon
    original_call_soon_threadsafe = getattr(loop_cls, "call_soon_threadsafe", None)

    @wraps(original_call_soon)
    def patched_call_soon(self, callback, *args, context=None):
        debug_counters["call_soon_hits"] += 1
        task_obj = _extract_task_from_callback(callback)
        task_id, req_id, req_name = _resolve_schedule_context(task_obj, context)
        should_track = bool(task_id is not None or req_id)
        patched_cb = callback
        if should_track and not getattr(callback, "_de_latency_sched_wrapped", False):
            patched_cb = _make_scheduled_callback(
                callback,
                ready_ts_ns=time.time_ns(),
                task_id=task_id,
                request_id=req_id,
                request_name=req_name,
            )
        if DEBUG_SCHED_PATCH and debug_counters["call_soon_hits"] <= 20:
            cb_type = type(callback).__name__ if callback is not None else "None"
            _debug_log(
                f"{loop_cls.__name__}.call_soon hit#{debug_counters['call_soon_hits']} "
                f"cb_type={cb_type} track={should_track} rid={req_id}"
            )
        return _call_with_optional_context(original_call_soon, self, patched_cb, args, context)

    loop_cls.call_soon = patched_call_soon

    if original_call_soon_threadsafe is not None:
        @wraps(original_call_soon_threadsafe)
        def patched_call_soon_threadsafe(self, callback, *args, context=None):
            debug_counters["call_soon_threadsafe_hits"] += 1
            task_obj = _extract_task_from_callback(callback)
            task_id, req_id, req_name = _resolve_schedule_context(task_obj, context)
            should_track = bool(task_id is not None or req_id)
            patched_cb = callback
            if should_track and not getattr(callback, "_de_latency_sched_wrapped", False):
                patched_cb = _make_scheduled_callback(
                    callback,
                    ready_ts_ns=time.time_ns(),
                    task_id=task_id,
                    request_id=req_id,
                    request_name=req_name,
                )
            if DEBUG_SCHED_PATCH and debug_counters["call_soon_threadsafe_hits"] <= 20:
                cb_type = type(callback).__name__ if callback is not None else "None"
                _debug_log(
                    f"{loop_cls.__name__}.call_soon_threadsafe hit#{debug_counters['call_soon_threadsafe_hits']} "
                    f"cb_type={cb_type} track={should_track} rid={req_id}"
                )
            return _call_with_optional_context(original_call_soon_threadsafe, self, patched_cb, args, context)

        loop_cls.call_soon_threadsafe = patched_call_soon_threadsafe

    setattr(loop_cls, "_de_latency_sched_patch_installed", True)
    return True


def patch_asyncio_scheduler():
    if getattr(asyncio, "_de_latency_sched_patch_installed", False):
        return

    patched_any = _patch_loop_class(asyncio.BaseEventLoop)

    try:
        import uvloop  # type: ignore
        patched_any = _patch_loop_class(uvloop.Loop) or patched_any
    except Exception as e:
        if DEBUG_SCHED_PATCH:
            _debug_log(f"uvloop patch skipped: {e}")

    if not patched_any:
        patch_logger.warning("[VLLM_PATCH] Asyncio scheduler patch was not installed on any loop class.")
        return

    asyncio._de_latency_sched_patch_installed = True
    patch_logger.info("[VLLM_PATCH] Asyncio scheduler patch applied successfully.")
    send_event(
        source="monkey_patch",
        event_type="coroutine_sched_patch_status",
        payload={
            "pid": os.getpid(),
            "debug_enabled": DEBUG_SCHED_PATCH,
            "status": "installed",
            "timestamp_ns": time.time_ns(),
        },
    )
    _debug_log(f"patch installed in pid={os.getpid()}")


def patch_vllm():
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.engine.processor import Processor
        from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
        from vllm.v1.engine import core_client as core_client_module
        _process_utility_output = core_client_module._process_utility_output
        from vllm.entrypoints.utils import _validate_truncation_size
        from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
        patch_logger.info("[VLLM_PATCH] Successfully imported AsyncLLM.")
    except ImportError:
        patch_logger.error("[VLLM_PATCH] Could not import AsyncLLM", exc_info=True)
        return

    if getattr(AsyncLLM, "_de_latency_req_stage_patch_installed", False):
        patch_logger.info("[VLLM_PATCH] Request stage patch already installed.")
        return

    def _mark_patched(func):
        setattr(func, "_de_latency_req_stage_wrapped", True)
        return func

    def _is_patched(func):
        return getattr(func, "_de_latency_req_stage_wrapped", False)

    # 1) Patch AsyncLLM._run_output_handler: generate() 早期初始化阶段
    if hasattr(AsyncLLM, "_run_output_handler"):
        original_run_output_handler = AsyncLLM._run_output_handler
        if not _is_patched(original_run_output_handler):
            @wraps(original_run_output_handler)
            def patched_run_output_handler(self, *args, **kwargs):
                rid, rname = get_request_ctx_from_context(None)
                start_ns = time.time_ns()
                prev_output_handler = getattr(self, "output_handler", None)
                try:
                    return original_run_output_handler(self, *args, **kwargs)
                finally:
                    output_handler_task = getattr(self, "output_handler", None)
                    if output_handler_task is not None and output_handler_task is not prev_output_handler:
                        _register_task_context(
                            output_handler_task,
                            request_id=None,
                            request_name="output_handler",
                            task_kind="output_handler",
                            extra={
                                "pending_sched_events": [],
                                "current_process_req_ids": [],
                                "current_process_size": 0,
                                "process_seq": 0,
                            },
                        )
                    emit_stage_duration(
                        stage="async_llm._run_output_handler",
                        start_ns=start_ns,
                        end_ns=time.time_ns(),
                        request_id=rid,
                        request_name=rname,
                    )

            AsyncLLM._run_output_handler = _mark_patched(patched_run_output_handler)
            patch_logger.info("[VLLM_PATCH] Patched AsyncLLM._run_output_handler.")

    # 1.5) Patch OutputProcessor.process_outputs:
    # output_handler 在每次 resume 后，这里会消化该轮/该 chunk 的输出，
    # 用它来把 pending 的 output_handler 调度排队时间归因到当前轮 req_ids。
    if hasattr(OutputProcessor, "process_outputs"):
        original_process_outputs = OutputProcessor.process_outputs
        if not _is_patched(original_process_outputs):
            @wraps(original_process_outputs)
            def patched_process_outputs(self, *args, **kwargs):
                engine_core_outputs = get_arg_value(original_process_outputs, "engine_core_outputs", args, kwargs)
                req_ids, process_seq = _begin_output_handler_process_outputs(engine_core_outputs)
                start_ns = time.time_ns()
                try:
                    return original_process_outputs(self, *args, **kwargs)
                finally:
                    end_ns = time.time_ns()
                    if req_ids and end_ns > start_ns:
                        task = asyncio.current_task()
                        task_id = id(task) if task is not None else None
                        task_name = task.get_name() if (task is not None and hasattr(task, "get_name")) else None
                        for rid in req_ids:
                            remember_request_name(rid, None)
                        emit_output_handler_exec_slice(
                            start_ns=start_ns,
                            end_ns=end_ns,
                            req_ids=req_ids,
                            task_id=task_id,
                            task_name=task_name,
                            round_seq=process_seq,
                        )

            OutputProcessor.process_outputs = _mark_patched(patched_process_outputs)
            patch_logger.info("[VLLM_PATCH] Patched OutputProcessor.process_outputs.")

    # 2) Patch Processor.process_inputs: 输入处理/分词阶段
    if hasattr(Processor, "process_inputs"):
        original_process_inputs = Processor.process_inputs
        if not _is_patched(original_process_inputs):
            @wraps(original_process_inputs)
            def patched_process_inputs(self, *args, **kwargs):
                request_id = get_arg_value(original_process_inputs, "request_id", args, kwargs)
                request_name = remember_request_name(request_id, None)
                start_ns = time.time_ns()
                try:
                    return original_process_inputs(self, *args, **kwargs)
                finally:
                    emit_stage_duration(
                        stage="processor.process_inputs",
                        start_ns=start_ns,
                        end_ns=time.time_ns(),
                        request_id=request_id,
                        request_name=request_name,
                    )

            Processor.process_inputs = _mark_patched(patched_process_inputs)
            patch_logger.info("[VLLM_PATCH] Patched Processor.process_inputs.")

    # 3) Patch OutputProcessor.add_request: 本进程入输出处理队列阶段
    if hasattr(OutputProcessor, "add_request"):
        original_output_add_request = OutputProcessor.add_request
        if not _is_patched(original_output_add_request):
            @wraps(original_output_add_request)
            def patched_output_add_request(self, *args, **kwargs):
                request = get_arg_value(original_output_add_request, "request", args, kwargs)
                request_id = getattr(request, "request_id", None)
                request_name = remember_request_name(request_id, None)
                start_ns = time.time_ns()
                try:
                    return original_output_add_request(self, *args, **kwargs)
                finally:
                    emit_stage_duration(
                        stage="output_processor.add_request",
                        start_ns=start_ns,
                        end_ns=time.time_ns(),
                        request_id=request_id,
                        request_name=request_name,
                    )

            OutputProcessor.add_request = _mark_patched(patched_output_add_request)
            patch_logger.info("[VLLM_PATCH] Patched OutputProcessor.add_request.")

    # 3.5) Patch RequestOutputCollector.put:
    # 记录每个请求输出第一次变为可消费的时间点，用于量内部 generate 消费前的队列停留时间。
    if hasattr(RequestOutputCollector, "put"):
        original_queue_put = RequestOutputCollector.put
        if not _is_patched(original_queue_put):
            @wraps(original_queue_put)
            def patched_queue_put(self, *args, **kwargs):
                was_empty = getattr(self, "output", None) is None
                ready_ts_ns = time.time_ns()
                try:
                    return original_queue_put(self, *args, **kwargs)
                finally:
                    if was_empty and getattr(self, "output", None) is not None:
                        try:
                            setattr(self, "_de_latency_ready_since_ns", ready_ts_ns)
                        except Exception:
                            pass

            RequestOutputCollector.put = _mark_patched(patched_queue_put)
            patch_logger.info("[VLLM_PATCH] Patched RequestOutputCollector.put.")

    # 4) Patch EngineCore client add_request_async: RPC/IPC 发送阶段
    patched_core_classes = []
    for cls_name in ["AsyncMPClient", "DPAsyncMPClient", "DPLBAsyncMPClient", "InprocClient"]:
        cls = getattr(core_client_module, cls_name, None)
        if cls is None or not hasattr(cls, "add_request_async"):
            continue
        original_core_add = getattr(cls, "add_request_async")
        if _is_patched(original_core_add):
            continue

        @wraps(original_core_add)
        async def patched_core_add_request_async(self, *args, __orig=original_core_add, **kwargs):
            request = get_arg_value(__orig, "request", args, kwargs)
            request_id = getattr(request, "request_id", None)
            request_name = remember_request_name(request_id, None)
            start_ns = time.time_ns()
            try:
                return await __orig(self, *args, **kwargs)
            finally:
                emit_stage_duration(
                    stage="engine_core.add_request_async",
                    start_ns=start_ns,
                    end_ns=time.time_ns(),
                    request_id=request_id,
                    request_name=request_name,
                    extra={"client_class": self.__class__.__name__},
                )

        setattr(cls, "add_request_async", _mark_patched(patched_core_add_request_async))
        patched_core_classes.append(cls_name)
    if patched_core_classes:
        patch_logger.info(f"[VLLM_PATCH] Patched EngineCore add_request_async on {patched_core_classes}.")

    # 4.5) Patch AsyncMPClient-like output socket consumer task.
    output_socket_patched = []
    for cls_name in ["AsyncMPClient", "DPAsyncMPClient", "DPLBAsyncMPClient"]:
        cls = getattr(core_client_module, cls_name, None)
        if cls is None or not hasattr(cls, "_ensure_output_queue_task"):
            continue
        original_ensure_output_queue_task = getattr(cls, "_ensure_output_queue_task")
        if _is_patched(original_ensure_output_queue_task):
            continue

        @wraps(original_ensure_output_queue_task)
        def patched_ensure_output_queue_task(self, *args, __orig=original_ensure_output_queue_task, **kwargs):
            resources = self.resources
            if getattr(resources, "output_queue_task", None) is not None:
                return __orig(self, *args, **kwargs)

            decoder = self.decoder
            utility_results = self.utility_results
            outputs_queue = self.outputs_queue
            output_handler = getattr(self.__class__, "process_engine_outputs", None)
            _self_ref = weakref.ref(self) if output_handler else None
            output_socket = resources.output_socket
            assert output_socket is not None

            async def process_outputs_socket():
                try:
                    while True:
                        frames = await output_socket.recv_multipart(copy=False)
                        exec_start_ns = time.time_ns()
                        resources.validate_alive(frames)
                        outputs = decoder.decode(frames)
                        if outputs.utility_output:
                            _process_utility_output(outputs.utility_output, utility_results)
                            continue

                        req_ids, process_seq = _begin_output_socket_process_outputs(outputs)
                        if output_handler is not None:
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            await output_handler(_self, outputs)

                        if outputs.outputs or outputs.scheduler_stats:
                            outputs_queue.put_nowait(outputs)

                        exec_end_ns = time.time_ns()
                        if req_ids and exec_end_ns > exec_start_ns:
                            for rid in req_ids:
                                remember_request_name(rid, None)
                            task = asyncio.current_task()
                            task_id = id(task) if task is not None else None
                            task_name = task.get_name() if (task is not None and hasattr(task, "get_name")) else None
                            emit_output_socket_exec_slice(
                                start_ns=exec_start_ns,
                                end_ns=exec_end_ns,
                                req_ids=req_ids,
                                task_id=task_id,
                                task_name=task_name,
                                round_seq=process_seq,
                            )
                except Exception as e:
                    outputs_queue.put_nowait(e)
                except asyncio.CancelledError:
                    outputs_queue.put_nowait(EngineDeadError())

            resources.output_queue_task = asyncio.create_task(
                process_outputs_socket(), name="EngineCoreOutputQueueTask")
            _register_task_context(
                resources.output_queue_task,
                request_id=None,
                request_name="process_outputs_socket",
                task_kind="output_socket",
                extra={
                    "pending_sched_events": [],
                    "current_process_req_ids": [],
                    "current_process_size": 0,
                    "process_seq": 0,
                },
            )

        setattr(cls, "_ensure_output_queue_task", _mark_patched(patched_ensure_output_queue_task))
        output_socket_patched.append(cls_name)
    if output_socket_patched:
        patch_logger.info(f"[VLLM_PATCH] Patched process_outputs_socket on {output_socket_patched}.")

    # 5) Patch AsyncLLM.add_request: 总请求注册阶段
    if hasattr(AsyncLLM, "add_request"):
        original_add_request = AsyncLLM.add_request
        if not _is_patched(original_add_request):
            @wraps(original_add_request)
            async def patched_add_request(self, *args, **kwargs):
                request_id = get_arg_value(original_add_request, "request_id", args, kwargs)
                request_name = remember_request_name(request_id, None)
                start_ns = time.time_ns()
                try:
                    return await original_add_request(self, *args, **kwargs)
                finally:
                    emit_stage_duration(
                        stage="async_llm.add_request",
                        start_ns=start_ns,
                        end_ns=time.time_ns(),
                        request_id=request_id,
                        request_name=request_name,
                    )

            AsyncLLM.add_request = _mark_patched(patched_add_request)
            patch_logger.info("[VLLM_PATCH] Patched AsyncLLM.add_request.")

    # 6) Patch AsyncLLM._add_request: 单子请求发送总耗时阶段
    if hasattr(AsyncLLM, "_add_request"):
        original__add_request = AsyncLLM._add_request
        if not _is_patched(original__add_request):
            @wraps(original__add_request)
            async def patched__add_request(self, *args, **kwargs):
                request = get_arg_value(original__add_request, "request", args, kwargs)
                request_id = getattr(request, "request_id", None)
                request_name = remember_request_name(request_id, None)
                start_ns = time.time_ns()
                try:
                    return await original__add_request(self, *args, **kwargs)
                finally:
                    emit_stage_duration(
                        stage="async_llm._add_request",
                        start_ns=start_ns,
                        end_ns=time.time_ns(),
                        request_id=request_id,
                        request_name=request_name,
                    )

            AsyncLLM._add_request = _mark_patched(patched__add_request)
            patch_logger.info("[VLLM_PATCH] Patched AsyncLLM._add_request.")

    # 7) Patch AsyncLLM.generate:
    # 不再把外层 StreamingResponse task 的调度当作 generate。
    # 这里直接在内部 q.get* -> yield out 的消费循环里记录 queue/exec。
    original_generate = AsyncLLM.generate
    patch_logger.info("[VLLM_PATCH] Patching AsyncLLM.generate ...")

    @wraps(original_generate)
    async def patched_generate(self, *args, **kwargs):
        wrapper_start_ns = time.time_ns()
        prompt = get_arg_value(original_generate, "prompt", args, kwargs)
        sampling_params = get_arg_value(original_generate, "sampling_params", args, kwargs)
        request_id = get_arg_value(original_generate, "request_id", args, kwargs)
        lora_request = get_arg_value(original_generate, "lora_request", args, kwargs)
        trace_headers = get_arg_value(original_generate, "trace_headers", args, kwargs)
        priority = get_arg_value(original_generate, "priority", args, kwargs)
        data_parallel_rank = get_arg_value(original_generate, "data_parallel_rank", args, kwargs)
        request_name = get_arg_value(original_generate, "request_name", args, kwargs)
        if request_name is None:
            request_name = request_id
        request_name = remember_request_name(request_id, request_name)
        process_id = os.getpid()
        thread_id = _get_native_tid()
        emit_thread_role("asyncio_eventloop", extra={"component": "async_llm.generate"})
        current_task = asyncio.current_task()
        current_task_id = id(current_task) if current_task is not None else None
        current_task_name = current_task.get_name() if (current_task is not None and hasattr(current_task, "get_name")) else None
        coro_id = current_task_id or id((request_id, time.time_ns()))
        start_t = time.monotonic()
        ts_ns = time.time_ns()
        coroutine_timers[coro_id] = (request_id, start_t)
        _register_current_task(
            request_id=request_id,
            request_name=request_name,
            task_kind="generate_task",
        )
        emit_stage_duration(
            stage="async_llm.generate_wrapper_setup",
            start_ns=wrapper_start_ns,
            end_ns=ts_ns,
            request_id=request_id,
            request_name=request_name,
        )

        send_event(
            source="monkey_patch",
            event_type="coroutine_start",
            payload={
                "request_id": request_id,
                "request_name": request_name,
                "pid": process_id,
                "tid": thread_id,
                "coroutine_id": coro_id,
                "timestamp_ns": ts_ns,
            },
        )

        try:
            if (self.vllm_config.cache_config.kv_sharing_fast_prefill
                    and sampling_params.prompt_logprobs):
                raise ValueError(
                    "--kv-sharing-fast-prefill produces incorrect logprobs for "
                    "prompt tokens, please disable it when the requests need "
                    "prompt logprobs")

            context_tokens = bind_request_context(request_id, request_name)
            try:
                self._run_output_handler()

                tokenization_kwargs = {}
                truncate_prompt_tokens = sampling_params.truncate_prompt_tokens
                _validate_truncation_size(
                    self.model_config.max_model_len,
                    truncate_prompt_tokens,
                    tokenization_kwargs,
                )

                q = await self.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=priority,
                    tokenization_kwargs=tokenization_kwargs,
                    data_parallel_rank=data_parallel_rank,
                )
            finally:
                reset_request_context(context_tokens)

            finished = False
            while not finished:
                task = asyncio.current_task()
                task_id = id(task) if task is not None else current_task_id
                task_name = task.get_name() if (task is not None and hasattr(task, "get_name")) else current_task_name

                exec_start_ns = time.time_ns()
                ready_since_ns = getattr(q, "_de_latency_ready_since_ns", None)
                out = q.get_nowait()
                if out is None:
                    out = await q.get()
                    exec_start_ns = time.time_ns()
                    ready_since_ns = getattr(q, "_de_latency_ready_since_ns", None)
                if ready_since_ns is not None and exec_start_ns >= ready_since_ns:
                    emit_coroutine_sched_latency(
                        ready_ts_ns=ready_since_ns,
                        run_ts_ns=exec_start_ns,
                        queue_ns=exec_start_ns - ready_since_ns,
                        request_id=request_id,
                        request_name=request_name,
                        task_id=task_id,
                        task_name=task_name,
                        task_kind="generate_consume",
                    )
                try:
                    setattr(q, "_de_latency_ready_since_ns", None)
                except Exception:
                    pass

                finished = out.finished
                exec_end_ns = time.time_ns()
                emit_coroutine_exec_slice(
                    start_ns=exec_start_ns,
                    end_ns=exec_end_ns,
                    request_id=request_id,
                    request_name=request_name,
                    task_id=task_id,
                    task_name=task_name,
                    task_kind="generate_consume",
                )
                yield out
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            if self.log_requests:
                logging.getLogger("vllm.v1.engine.async_llm").info(
                    "Request %s aborted.", request_id)
            raise
        except EngineDeadError:
            if self.log_requests:
                logging.getLogger("vllm.v1.engine.async_llm").info(
                    "Request %s failed (engine dead).", request_id)
            raise
        except ValueError:
            if self.log_requests:
                logging.getLogger("vllm.v1.engine.async_llm").info(
                    "Request %s failed (bad request).", request_id)
            raise
        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logging.getLogger("vllm.v1.engine.async_llm").info(
                    "Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            end_t = time.monotonic()
            _, begin = coroutine_timers.pop(coro_id, (None, None))
            duration_ms = (end_t - begin) * 1000 if begin else None
            ts_ns_end = time.time_ns()

            send_event(
                source="monkey_patch",
                event_type="coroutine_end",
                payload={
                    "request_id": request_id,
                    "request_name": request_name,
                    "pid": process_id, #直接复用
                    "tid": _get_native_tid(),
                    "coroutine_id": coro_id,
                    "duration_ms": duration_ms,
                    "timestamp_ns": ts_ns_end,
                },
            )

    AsyncLLM.generate = patched_generate
    AsyncLLM._de_latency_req_stage_patch_installed = True
    patch_logger.info("[VLLM_PATCH] Request stage patch applied successfully!")


patch_asyncio_scheduler()
patch_vllm()
