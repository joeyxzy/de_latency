import asyncio
import time
import json
from functools import wraps
from datetime import datetime
import inspect
import logging
import os
import threading
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
debug_counters = {
    "call_soon_hits": 0,
    "call_soon_threadsafe_hits": 0,
    "emit_hits": 0,
    "skip_no_ctx": 0,
}
SOCK_ADDR = "ipc:///tmp/tracer.sock"

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect(SOCK_ADDR)

def send_event(source, event_type, payload=None, extra=None):
    meta = make_metadata(source, event_type, extra=extra)
    # 如果 payload 是结构化数据且小，直接将其合并到 meta（单帧）
    if payload is not None and isinstance(payload, dict) and len(json.dumps(payload)) < 1500:
        meta['payload'] = payload
        sock.send_json(meta, flags=zmq.DONTWAIT)
    else:
        # multipart: meta + binary payload (payload is bytes)
        meta_b = metadata_to_bytes(meta)
        payload_b = payload if payload else b""
        sock.send_multipart([meta_b, payload_b], flags=zmq.DONTWAIT)

def get_arg_value(func, arg_name, args, kwargs):
    """从 args/kwargs 自动解析参数"""
    if arg_name in kwargs:
        return kwargs[arg_name]
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments.get(arg_name, None)
    except:
        return None


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


def _register_current_task(request_id, request_name):
    task = asyncio.current_task()
    if task is None:
        return None
    info = {
        "request_id": request_id,
        "request_name": request_name or request_id,
        "task_name": task.get_name() if hasattr(task, "get_name") else None,
    }
    with ctx_lock:
        task_request_ctx[id(task)] = info
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


def _make_scheduled_callback(original_callback, ready_ts_ns, task_id, request_id, request_name):
    def wrapped_callback(*cb_args):
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
                send_event(
                    source="monkey_patch",
                    event_type="coroutine_sched_latency",
                    payload={
                        "request_id": ctx_info.get("request_id"),
                        "request_name": ctx_info.get("request_name"),
                        "task_id": task_id,
                        "task_name": ctx_info.get("task_name"),
                        "ready_ts_ns": ready_ts_ns,
                        "run_ts_ns": run_ts_ns,
                        "queue_ns": queue_ns,
                    },
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
        return original_callback(*cb_args)

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
        patch_logger.info("[VLLM_PATCH] Successfully imported AsyncLLM.")
    except ImportError:
        patch_logger.error("[VLLM_PATCH] Could not import AsyncLLM", exc_info=True)
        return

    original_generate = AsyncLLM.generate
    patch_logger.info("[VLLM_PATCH] Patching AsyncLLM.generate ...")

    @wraps(original_generate)
    async def patched_generate(self, *args, **kwargs):
        request_id = None
        request_name = None
        if len(args) >= 3:
            request_id = args[2]
        elif "request_id" in kwargs:
            request_id = kwargs["request_id"]
        if "request_name" in kwargs:
            request_name = kwargs["request_name"]
        if request_name is None:
            request_name = request_id
        request_name = remember_request_name(request_id, request_name)
        process_id = os.getpid()
        thread_id = threading.get_ident()

        # 调用真实 generate，获取 async generator
        generator = original_generate(self, *args, **kwargs)
        coro_id = id(generator)
        start_t = time.monotonic()
        ts_ns = time.time_ns()
        coroutine_timers[coro_id] = (request_id, start_t)

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

        context_tokens = bind_request_context(request_id, request_name)
        current_task = _register_current_task(request_id, request_name)
        try:
            async for item in generator:
                yield item
        finally:
            _unregister_task(current_task)
            reset_request_context(context_tokens)
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
                    "tid": threading.get_ident(), #没有直接复用
                    "coroutine_id": coro_id,
                    "duration_ms": duration_ms,
                    "timestamp_ns": ts_ns_end,
                },
            )

    AsyncLLM.generate = patched_generate
    patch_logger.info("[VLLM_PATCH] Patch applied successfully!")


patch_asyncio_scheduler()
patch_vllm()
