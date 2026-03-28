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
ENABLE_REQ_STAGE_TIMING = os.getenv("TRACE_REQ_STAGE_TIMING", "1") == "1"
debug_counters = {
    "call_soon_hits": 0,
    "call_soon_threadsafe_hits": 0,
    "emit_hits": 0,
    "skip_no_ctx": 0,
}
SOCK_ADDR = os.getenv("TRACER_ZMQ_ADDR", "ipc:///tmp/tracer.sock")

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect(SOCK_ADDR)

def send_event(source, event_type, payload=None, extra=None):
    meta = make_metadata(source, event_type, extra=extra)
    try:
        # 如果 payload 是结构化数据且小，直接将其合并到 meta（单帧）
        if payload is not None and isinstance(payload, dict) and len(json.dumps(payload)) < 1500:
            meta['payload'] = payload
            sock.send_json(meta, flags=zmq.DONTWAIT)
        else:
            # multipart: meta + binary payload (payload is bytes)
            meta_b = metadata_to_bytes(meta)
            payload_b = payload if payload else b""
            sock.send_multipart([meta_b, payload_b], flags=zmq.DONTWAIT)
    except zmq.error.Again:
        # 发送队列满时直接丢弃，绝不影响业务主流程
        pass
    except Exception:
        # 防御性兜底：trace 不得打断推理
        pass

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
        "tid": threading.get_ident(),
        "timestamp_ns": end_ns if end_ns is not None else time.time_ns(),
    }
    if extra and isinstance(extra, dict):
        payload.update(extra)
    send_event(
        source="monkey_patch",
        event_type="req_generate_stage",
        payload=payload,
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
        from vllm.v1.engine.processor import Processor
        from vllm.v1.engine.output_processor import OutputProcessor
        from vllm.v1.engine import core_client as core_client_module
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
                try:
                    return original_run_output_handler(self, *args, **kwargs)
                finally:
                    emit_stage_duration(
                        stage="async_llm._run_output_handler",
                        start_ns=start_ns,
                        end_ns=time.time_ns(),
                        request_id=rid,
                        request_name=rname,
                    )

            AsyncLLM._run_output_handler = _mark_patched(patched_run_output_handler)
            patch_logger.info("[VLLM_PATCH] Patched AsyncLLM._run_output_handler.")

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

    # 7) Patch AsyncLLM.generate: 保留现有生命周期事件，并补充 wrapper 开销阶段
    original_generate = AsyncLLM.generate
    patch_logger.info("[VLLM_PATCH] Patching AsyncLLM.generate ...")

    @wraps(original_generate)
    async def patched_generate(self, *args, **kwargs):
        wrapper_start_ns = time.time_ns()
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
    AsyncLLM._de_latency_req_stage_patch_installed = True
    patch_logger.info("[VLLM_PATCH] Request stage patch applied successfully!")


patch_asyncio_scheduler()
patch_vllm()
