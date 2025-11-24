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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
patch_logger = logging.getLogger("vllm_patch")
patch_logger.info("--- VLLM Patch Logger Initialized ---")

coroutine_timers = {}
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
        if len(args) >= 3:
            request_id = args[2]
        elif "request_id" in kwargs:
            request_id = kwargs["request_id"]
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
                "pid": process_id,
                "tid": thread_id,
                "coroutine_id": coro_id,
                "timestamp_ns": ts_ns,
            },
        )

        try:
            async for item in generator:
                yield item
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
                    "pid": process_id, #直接复用
                    "tid": threading.get_ident(), #没有直接复用
                    "coroutine_id": coro_id,
                    "duration_ms": duration_ms,
                    "timestamp_ns": ts_ns_end,
                },
            )

    AsyncLLM.generate = patched_generate
    patch_logger.info("[VLLM_PATCH] Patch applied successfully!")


patch_vllm()
