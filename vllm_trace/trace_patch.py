import asyncio
import time
import json
from functools import wraps
from datetime import datetime
import inspect
import logging
import os
import threading

# ----------------- 日志配置-----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
patch_logger = logging.getLogger("vllm_patch")
patch_logger.info("--- VLLM Patch Logger Initialized ---")
# ----------------------------------------------------

# 字典存储状态
coroutine_timers = {}

def get_call_argument(func, arg_name, args, kwargs):
    if arg_name in kwargs:
        return kwargs[arg_name]
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs).arguments
        if arg_name in bound_args:
            return bound_args[arg_name]
    except (ValueError, TypeError, IndexError):
        pass
    return f"unknown_{arg_name}"

def patch_vllm():
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        patch_logger.info("[VLLM_PATCH] Successfully imported AsyncLLMEngine.")
    except ImportError:
        patch_logger.error("[VLLM_PATCH] Could not import AsyncLLMEngine.", exc_info=True)
        return

    patch_logger.info("[VLLM_PATCH] Applying patch to AsyncLLMEngine.generate...")
    original_generate = AsyncLLMEngine.generate

    @wraps(original_generate)
    async def patched_generate(self, *args, **kwargs):
        process_id = os.getpid()
        thread_id = threading.get_ident()
        #thread_id = os.gettid()
        request_id = get_call_argument(original_generate, 'request_id', args, kwargs)
        
        generator = original_generate(self, *args, **kwargs)
        coro_id = id(generator)
        start_time = time.monotonic()
        
        coroutine_timers[coro_id] = (request_id, start_time)
        
        log_event = {
            "timestamp": datetime.now().isoformat(),
            "event": "coroutine_start",
            "pid": process_id,
            "tid": thread_id,
            "request_id": request_id,
            "coroutine_id": coro_id,
            "type": "coroutine_lifetime"
        }
        patch_logger.info(json.dumps(log_event))

        try:
            async for output in generator:
                yield output
        finally:
            end_time = time.monotonic()
            if coro_id in coroutine_timers:
                end_process_id = os.getpid()
                end_thread_id = threading.get_ident() #python进程内部的线程号
                #end_thread_id = os.gettid()
                _request_id, _start_time = coroutine_timers.pop(coro_id)
                duration_ms = (end_time - _start_time) * 1000
                
                log_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "coroutine_end",
                    "pid": end_process_id,
                    "tid": end_thread_id,
                    "request_id": _request_id,
                    "coroutine_id": coro_id,
                    "duration_ms": duration_ms,
                    "type": "coroutine_lifetime"
                }
                patch_logger.info(json.dumps(log_event))

    AsyncLLMEngine.generate = patched_generate
    patch_logger.info("[VLLM_PATCH] Patch applied successfully with robust argument retrieval!")

# 立即执行
patch_vllm()