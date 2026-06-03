import sys
import importlib
import logging
import os
import threading
import inspect
import time
import json
import hashlib
import itertools
from typing import Dict, List, Any
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
from monkeypatch_runtime import (
    is_enabled as monkeypatch_enabled,
    start_control_server as start_monkeypatch_control_server,
)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRACE] %(message)s")
logger = logging.getLogger("VLLM_HOOK")
_CONTROL_SCRIPT_NAMES = {"collector.py", "cupti_ctl.py", "trace_ctl.py"}
_SKIP_TRACE_BOOTSTRAP = (
    os.getenv("DE_LATENCY_DISABLE_SITECUSTOMIZE", "0") == "1" or
    os.path.basename(sys.argv[0]) in _CONTROL_SCRIPT_NAMES
)
try:
    import zmq
except ModuleNotFoundError:
    if not _SKIP_TRACE_BOOTSTRAP:
        raise
    zmq = None
if not _SKIP_TRACE_BOOTSTRAP:
    try:
        start_monkeypatch_control_server()
    except OSError as exc:
        logger.warning("monkeypatch control bootstrap failed: %s", exc)

# =============================================================================
# Part 1: 基础设施层 (Infrastructure)
# 负责：ZMQ通信、Hook注册表、通用模块拦截器
# =============================================================================

class TraceSender:
    """
    负责将采集到的数据发送给 Log 收集进程 (C++ Collector 或其他消费者)。
    使用 thread_local 确保多线程下的 ZMQ Socket 安全。
    """
    _thread_local = threading.local()
    # ZMQ 地址，需与接收端保持一致
    ZMQ_ADDR = os.getenv("TRACER_ZMQ_ADDR", "ipc:///tmp/tracer.sock")

    @classmethod
    def get_socket(cls):
        if zmq is None:
            return None
        if not hasattr(cls._thread_local, "sock"):
            try:
                ctx = zmq.Context()
                sock = ctx.socket(zmq.PUSH)
                # LINGER=0 确保关闭时不会挂起等待发送
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(cls.ZMQ_ADDR)
                cls._thread_local.sock = sock
            except Exception as e:
                # 仅报错一次，避免刷屏
                if not hasattr(cls._thread_local, "error_logged"):
                    logger.error(f"ZMQ Init Failed: {e}")
                    cls._thread_local.error_logged = True
                cls._thread_local.sock = None
        return cls._thread_local.sock

    @classmethod
    def emit(cls, event_type, payload, source="monkey_patch"):
        """
        发送 Trace 事件。
        flags=zmq.DONTWAIT: 关键设置，如果队列满则丢弃，绝对不阻塞 vLLM 推理线程。
        """
        if not monkeypatch_enabled():
            return
        sock = cls.get_socket()
        if sock:
            try:
                meta = {
                    "source": source,
                    "event_type": event_type,
                    "timestamp_ns": time.time_ns(),
                    "payload": payload
                }
                # 发送两帧：Header(JSON) + Empty Body
                meta_bytes = json.dumps(meta).encode("utf-8")
                sock.send_multipart([meta_bytes, b""], flags=zmq.DONTWAIT)
            except zmq.error.Again:
                # 队列满，丢弃日志，不阻塞业务
                pass
            except Exception:
                pass


_GPU_EXEC_CONTEXT = threading.local()
GPU_PHASE_NAME_ALIASES = {
    "Preprocess": "Preprocess",
    "Forward": "Forward",
    "Postprocess": "Postprocess",
    "Bookkeep": "Bookkeep",
    "Draft": "Draft",
    "EPLB": "EPLB",
    # vLLM 0.20 renamed record_function scopes to namespaced lowercase names.
    "gpu_model_runner: preprocess": "Preprocess",
    "gpu_model_runner: forward": "Forward",
    "gpu_model_runner: postprocess": "Postprocess",
    "gpu_model_runner: bookkeep": "Bookkeep",
    "gpu_model_runner: draft": "Draft",
    "gpu_model_runner: eplb": "EPLB",
    "gpu_model_runner: ModelRunnerOutput": "ModelRunnerOutput",
}
GPU_PHASE_SCOPE_NAMES = set(GPU_PHASE_NAME_ALIASES)


def _canonical_gpu_phase_name(name):
    return GPU_PHASE_NAME_ALIASES.get(name)


def _get_native_tid():
    try:
        return threading.get_native_id()
    except AttributeError:
        return threading.get_ident()


def _normalize_req_ids(req_ids):
    if req_ids is None:
        return []
    if isinstance(req_ids, (list, tuple, set)):
        values = req_ids
    else:
        values = [req_ids]
    normalized = []
    for item in values:
        if item is None:
            continue
        rid = getattr(item, "request_id", None) or getattr(item, "req_id", None) or item
        rid = str(rid).strip()
        if rid:
            normalized.append(rid)
    return normalized


def _extract_req_ids_from_input_batch(input_batch):
    if input_batch is None or not hasattr(input_batch, "req_ids"):
        return []
    try:
        return _normalize_req_ids(list(input_batch.req_ids))
    except Exception:
        return []


def _extract_req_ids_from_scheduler_output(scheduler_output):
    if scheduler_output is None:
        return []

    req_ids = []
    try:
        if hasattr(scheduler_output, "scheduled_new_reqs"):
            for req in scheduler_output.scheduled_new_reqs:
                req_ids.extend(_normalize_req_ids(req))

        if hasattr(scheduler_output, "scheduled_cached_reqs"):
            cached = scheduler_output.scheduled_cached_reqs
            if cached is not None and hasattr(cached, "req_ids"):
                req_ids.extend(_normalize_req_ids(cached.req_ids))
    except Exception:
        return []
    return req_ids


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _extract_scheduler_step_maps(scheduler_output):
    """Build per-request scheduler metadata that stays stable across processes."""
    num_computed_by_req = {}
    scheduled_tokens_by_req = {}

    try:
        if hasattr(scheduler_output, "scheduled_new_reqs"):
            for req in scheduler_output.scheduled_new_reqs:
                rid = _normalize_req_ids(req)
                if not rid:
                    continue
                rid = rid[0]
                num_computed_by_req[rid] = _safe_int(
                    getattr(req, "num_computed_tokens", None)
                )
                token_ids = getattr(req, "prompt_token_ids", None)
                if token_ids is not None:
                    scheduled_tokens_by_req.setdefault(rid, _safe_int(len(token_ids), 0))
    except Exception:
        pass

    try:
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None and hasattr(cached, "req_ids"):
            req_ids = _normalize_req_ids(cached.req_ids)
            computed_values = list(getattr(cached, "num_computed_tokens", []) or [])
            for idx, rid in enumerate(req_ids):
                if idx < len(computed_values):
                    num_computed_by_req[rid] = _safe_int(computed_values[idx])
    except Exception:
        pass

    try:
        val = getattr(scheduler_output, "num_scheduled_tokens", None)
        if isinstance(val, dict):
            for key, item in val.items():
                rid = _normalize_req_ids(key)
                if rid:
                    scheduled_tokens_by_req[rid[0]] = _safe_int(item, 0)
        elif isinstance(val, (list, tuple)):
            req_ids = _extract_req_ids_from_scheduler_output(scheduler_output)
            for rid, item in zip(req_ids, val):
                scheduled_tokens_by_req[rid] = _safe_int(item, 0)
    except Exception:
        pass

    return num_computed_by_req, scheduled_tokens_by_req


def _make_dispatch_key(scheduler_output, phase_by_req=None):
    req_ids = _extract_req_ids_from_scheduler_output(scheduler_output)
    num_computed_by_req, scheduled_tokens_by_req = _extract_scheduler_step_maps(
        scheduler_output
    )
    try:
        total_scheduled = _safe_int(
            getattr(scheduler_output, "total_num_scheduled_tokens", None)
        )
    except Exception:
        total_scheduled = None

    parts = {
        "req_ids": list(req_ids),
        "num_computed_by_req": num_computed_by_req,
        "scheduled_tokens_by_req": scheduled_tokens_by_req,
        "total_scheduled_tokens": total_scheduled,
    }
    try:
        raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        raw = repr(parts)
    return "sched:" + hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _extract_scheduler_output_from_runner_state(runner):
    state = getattr(runner, "execute_model_state", None)
    if state is None:
        return None
    scheduler_output = getattr(state, "scheduler_output", None)
    if scheduler_output is not None:
        return scheduler_output
    try:
        return state[0]
    except Exception:
        return None


def _get_parallel_rank_info(obj=None):
    p_cfg = getattr(obj, "parallel_config", None)
    if p_cfg is None:
        vllm_config = getattr(obj, "vllm_config", None)
        p_cfg = getattr(vllm_config, "parallel_config", None)

    ranks = {
        "rank": -1,
        "dp": -1,
        "tp": -1,
        "pp": -1,
        "dp_size": -1,
        "tp_size": -1,
        "pp_size": -1,
        "is_pp_first": None,
        "is_pp_last": None,
    }

    if p_cfg is not None:
        ranks["rank"] = _safe_int(getattr(p_cfg, "rank", None), -1)
        ranks["dp"] = _safe_int(getattr(p_cfg, "data_parallel_rank", None), -1)
        ranks["dp_size"] = _safe_int(getattr(p_cfg, "data_parallel_size", None), -1)
        ranks["tp_size"] = _safe_int(getattr(p_cfg, "tensor_parallel_size", None), -1)
        ranks["pp_size"] = _safe_int(getattr(p_cfg, "pipeline_parallel_size", None), -1)
        if ranks["rank"] >= 0 and ranks["tp_size"] > 0:
            ranks["tp"] = ranks["rank"] % ranks["tp_size"]
        if ranks["rank"] >= 0 and ranks["tp_size"] > 0 and ranks["pp_size"] > 0:
            ranks["pp"] = (ranks["rank"] // ranks["tp_size"]) % ranks["pp_size"]

    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        pp_group = get_pp_group()
        ranks["pp"] = _safe_int(getattr(pp_group, "rank_in_group", None), ranks["pp"])
        ranks["pp_size"] = _safe_int(getattr(pp_group, "world_size", None), ranks["pp_size"])
        ranks["is_pp_first"] = bool(getattr(pp_group, "is_first_rank", False))
        ranks["is_pp_last"] = bool(getattr(pp_group, "is_last_rank", False))
        tp_group = get_tp_group()
        ranks["tp"] = _safe_int(getattr(tp_group, "rank_in_group", None), ranks["tp"])
        ranks["tp_size"] = _safe_int(getattr(tp_group, "world_size", None), ranks["tp_size"])
    except Exception:
        pass

    return ranks


def _estimate_tensor_payload(value, max_items=32):
    summary = {
        "tensor_count": 0,
        "bytes": 0,
        "items": [],
    }

    def _visit(item, path):
        if summary["tensor_count"] >= max_items:
            return
        if hasattr(item, "numel") and hasattr(item, "element_size"):
            try:
                numel = int(item.numel())
                elem_size = int(item.element_size())
            except Exception:
                numel = 0
                elem_size = 0
            nbytes = max(0, numel * elem_size)
            summary["tensor_count"] += 1
            summary["bytes"] += nbytes
            try:
                shape = list(item.shape)
            except Exception:
                shape = None
            summary["items"].append({
                "path": path,
                "shape": shape,
                "dtype": str(getattr(item, "dtype", "")),
                "device": str(getattr(item, "device", "")),
                "bytes": nbytes,
            })
            return
        if isinstance(item, dict):
            for key, child in list(item.items())[:max_items]:
                _visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(item, (list, tuple)):
            for idx, child in enumerate(list(item)[:max_items]):
                _visit(child, f"{path}[{idx}]")

    _visit(value, "")
    return summary


def _is_pp_group(group):
    try:
        from vllm.distributed.parallel_state import get_pp_group
        pp_group = get_pp_group()
        if group is pp_group:
            return True
        unique_name = str(getattr(group, "unique_name", "") or "").lower()
        if unique_name.startswith("pp") or ":pp" in unique_name or unique_name.endswith("pp"):
            return True
        return (
            list(getattr(group, "ranks", [])) == list(getattr(pp_group, "ranks", []))
            and _safe_int(getattr(group, "world_size", None), 1) > 1
        )
    except Exception:
        return False


def _pp_comm_context_payload(group, op, start_ns, end_ns, peer_rank=None, src=None, dst=None, tensor_payload=None):
    ctx = _current_gpu_exec_context() or {}
    ranks = dict(ctx.get("ranks") or _get_parallel_rank_info(None))
    payload_summary = _estimate_tensor_payload(tensor_payload) if tensor_payload is not None else {}
    return {
        "op": op,
        "pid": os.getpid(),
        "tid": _get_native_tid(),
        "dispatch_key": ctx.get("dispatch_key"),
        "req_ids": list(ctx.get("req_ids") or []),
        "ranks": ranks,
        "group_rank": _safe_int(getattr(group, "rank_in_group", None), -1),
        "group_world_size": _safe_int(getattr(group, "world_size", None), -1),
        "group_ranks": list(getattr(group, "ranks", []) or []),
        "peer_rank": peer_rank,
        "src": src,
        "dst": dst,
        "tensor_count": payload_summary.get("tensor_count", 0),
        "tensor_bytes": payload_summary.get("bytes", 0),
        "tensor_items": payload_summary.get("items", []),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": max(0, end_ns - start_ns),
        "timestamp_ns": end_ns,
    }


_ASYNC_PP_COMM_SEQ = itertools.count(1)


class _TracedCommHandle:
    """Proxy a torch distributed async handle and emit the blocking wait span."""

    def __init__(self, handle, launch_payload, wait_op, handle_index=None):
        self._handle = handle
        self._launch_payload = dict(launch_payload or {})
        self._wait_op = wait_op
        self._handle_index = handle_index

    def is_completed(self):
        return self._handle.is_completed()

    def wait(self, *args, **kwargs):
        wait_start_ns = time.time_ns()
        try:
            return self._handle.wait(*args, **kwargs)
        finally:
            wait_end_ns = time.time_ns()
            payload = dict(self._launch_payload)
            payload.update({
                "op": self._wait_op,
                "tid": _get_native_tid(),
                "start_ns": wait_start_ns,
                "end_ns": wait_end_ns,
                "duration_ns": max(0, wait_end_ns - wait_start_ns),
                "timestamp_ns": wait_end_ns,
                "async_phase": "wait",
                "handle_index": self._handle_index,
                "async_elapsed_ns": max(
                    0, wait_end_ns - _safe_int(payload.get("launch_start_ns"), wait_start_ns)
                ),
            })
            TraceSender.emit(event_type="pp_comm_span", payload=payload)

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _summarize_scheduler_output(scheduler_output):
    req_ids = _extract_req_ids_from_scheduler_output(scheduler_output)
    batch_size = 0
    input_type = "unknown"

    try:
        if hasattr(scheduler_output, "total_num_scheduled_tokens"):
            batch_size = int(scheduler_output.total_num_scheduled_tokens)
            input_type = "scheduler_output_v1"
        elif hasattr(scheduler_output, "num_scheduled_tokens"):
            val = scheduler_output.num_scheduled_tokens
            if isinstance(val, dict):
                batch_size = sum(int(x) for x in val.values())
            elif isinstance(val, (list, tuple)):
                batch_size = sum(int(x) for x in val)
            else:
                batch_size = int(val)
            input_type = "scheduler_output_v2"
    except Exception:
        batch_size = 0
        input_type = "unknown"

    return {
        "req_ids": req_ids,
        "batch_size": batch_size,
        "input_type": input_type,
    }


def _gpu_exec_stack():
    stack = getattr(_GPU_EXEC_CONTEXT, "stack", None)
    if stack is None:
        stack = []
        _GPU_EXEC_CONTEXT.stack = stack
    return stack


def _push_gpu_exec_context(ctx):
    _gpu_exec_stack().append(ctx)


def _pop_gpu_exec_context():
    stack = _gpu_exec_stack()
    if not stack:
        return None
    return stack.pop()


def _current_gpu_exec_context():
    stack = getattr(_GPU_EXEC_CONTEXT, "stack", None)
    if not stack:
        return None
    return stack[-1]


def _refresh_gpu_exec_context(ctx):
    runner = ctx.get("runner")
    req_ids = _extract_req_ids_from_input_batch(getattr(runner, "input_batch", None))
    if req_ids:
        ctx["input_batch_req_ids"] = req_ids
        # In PP/async scheduling the input batch is persistent state and may
        # include requests outside the current SchedulerOutput. Keep the
        # scheduler-derived request set as the authoritative dispatch
        # membership, and only fall back to input_batch when the scheduler
        # output was unavailable.
        if not ctx.get("req_ids"):
            ctx["req_ids"] = req_ids
    return ctx


def _build_gpu_exec_context(runner, scheduler_output=None, method_name=None):
    summary = _summarize_scheduler_output(scheduler_output)
    ranks = _get_parallel_rank_info(runner)

    if scheduler_output is not None:
        dispatch_key = _make_dispatch_key(scheduler_output)
    else:
        dispatch_key = None
        existing = _current_gpu_exec_context()
        if existing is not None:
            dispatch_key = existing.get("dispatch_key")
            if not summary.get("req_ids"):
                summary["req_ids"] = existing.get("req_ids", [])
        if not dispatch_key:
            dispatch_key = _make_dispatch_key(scheduler_output)

    ctx = {
        "runner": runner,
        "pid": os.getpid(),
        "tid": _get_native_tid(),
        "req_ids": list(summary.get("req_ids", [])),
        "batch_size": summary.get("batch_size", 0),
        "input_type": summary.get("input_type", "unknown"),
        "ranks": ranks,
        "dispatch_key": dispatch_key,
        "method_name": method_name,
    }
    return _refresh_gpu_exec_context(ctx)


def _emit_legacy_gpu_phase_events(phase_name, payload):
    if phase_name == "Forward":
        TraceSender.emit(
            event_type="gpu_forward_start",
            payload={
                "pid": payload["pid"],
                "tid": payload["tid"],
                "timestamp_ns": payload["start_ns"],
            },
        )
        TraceSender.emit(
            event_type="gpu_forward_end",
            payload={
                "pid": payload["pid"],
                "tid": payload["tid"],
                "timestamp_ns": payload["end_ns"],
            },
        )
    elif phase_name == "Sample":
        TraceSender.emit(
            event_type="gpu_sample",
            payload={
                "pid": payload["pid"],
                "tid": payload["tid"],
                "req_ids": payload.get("req_ids", []),
                "start_ns": payload["start_ns"],
                "end_ns": payload["end_ns"],
            },
        )
    elif phase_name == "Bookkeep":
        TraceSender.emit(
            event_type="gpu_bookkeeping",
            payload={
                "pid": payload["pid"],
                "tid": payload["tid"],
                "req_ids": payload.get("req_ids", []),
                "start_ns": payload["start_ns"],
                "end_ns": payload["end_ns"],
            },
        )

# --- Hook 注册表 ---
HOOK_REGISTRY = {}
_WORKER_READY_PIDS = set()
_WORKER_READY_LOCK = threading.Lock()
_CORE_PREPROCESS_DONE_NS = {}
_CORE_PREPROCESS_LOCK = threading.Lock()


def _remember_core_preprocess_done(req_id: str, ts_ns: int):
    if not req_id:
        return
    with _CORE_PREPROCESS_LOCK:
        _CORE_PREPROCESS_DONE_NS[req_id] = ts_ns


def _pop_core_preprocess_done(req_id: str):
    if not req_id:
        return None
    with _CORE_PREPROCESS_LOCK:
        return _CORE_PREPROCESS_DONE_NS.pop(req_id, None)


def emit_req_stage(stage: str, start_ns: int, end_ns: int, request_id=None, request_name=None, extra=None):
    if not monkeypatch_enabled():
        return
    if start_ns is None or end_ns is None:
        return
    if end_ns < start_ns:
        return
    rid = request_id
    rname = request_name or rid
    payload = {
        "request_id": rid,
        "request_name": rname,
        "stage": stage,
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "duration_ns": int(end_ns) - int(start_ns),
        "pid": os.getpid(),
        "tid": getattr(threading, "get_native_id", threading.get_ident)(),
        "timestamp_ns": int(end_ns),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    TraceSender.emit(event_type="req_generate_stage", payload=payload)


def emit_worker_process_ready(stage):
    if not monkeypatch_enabled():
        return
    pid = os.getpid()
    with _WORKER_READY_LOCK:
        if pid in _WORKER_READY_PIDS:
            return
        _WORKER_READY_PIDS.add(pid)

    try:
        tid = threading.get_native_id()
    except AttributeError:
        tid = threading.get_ident()

    TraceSender.emit(
        event_type="worker_process_ready",
        payload={
            "pid": pid,
            "tid": tid,
            "stage": stage,
            "timestamp_ns": time.time_ns(),
        },
    )


def register_hook(module_name):
    """装饰器：将函数注册为指定模块的 Patch 逻辑"""
    def decorator(func):
        HOOK_REGISTRY[module_name] = func
        return func
    return decorator

def apply_method_patch(cls, method_name, wrapper_factory):
    """
    辅助工具：将 wrapper 应用到类的方法上。
    wrapper_factory: 接收 (original_func) 返回 (patched_func) 的闭包工厂。
    """
    if not hasattr(cls, method_name):
        # 很多时候 vLLM 版本差异导致函数名不同，这里记录 warn 即可，不要 crash
        logger.debug(f"Method '{method_name}' not found in {cls.__name__}, skipping.")
        return False

    original_func = getattr(cls, method_name)
    patched_func = wrapper_factory(original_func)
    setattr(cls, method_name, patched_func)
    logger.info(f"Successfully patched: {cls.__name__}.{method_name}")
    return True


def _emit_scheduler_out_rpc(scheduler_output, event="execute_start"):
    req_ids = _extract_req_ids_from_scheduler_output(scheduler_output)
    dispatch_key = _make_dispatch_key(scheduler_output)
    num_computed_by_req, scheduled_tokens_by_req = _extract_scheduler_step_maps(
        scheduler_output
    )
    total_scheduled = _safe_int(
        getattr(scheduler_output, "total_num_scheduled_tokens", None),
        0,
    )
    if not req_ids:
        return
    TraceSender.emit(
        event_type="req_scheduler_out_rpc",
        payload={
            "req_ids": req_ids,
            "dispatch_key": dispatch_key,
            "event": event,
            "batch_size": len(req_ids),
            "total_scheduled_tokens": total_scheduled,
            "num_computed_by_req": num_computed_by_req,
            "scheduled_tokens_by_req": scheduled_tokens_by_req,
            "timestamp_ns": time.time_ns(),
        },
    )


def _make_scheduler_out_rpc_wrapper(original_func):
    if getattr(original_func, "_de_latency_scheduler_out_rpc_patch", False):
        return original_func

    def wrapper(self, scheduler_output, *args, **kwargs):
        _emit_scheduler_out_rpc(scheduler_output, event="execute_start")
        return original_func(self, scheduler_output, *args, **kwargs)

    setattr(wrapper, "_de_latency_scheduler_out_rpc_patch", True)
    return wrapper


@register_hook("vllm.v1.engine.core")
def patch_v1_engine_core(module):
    engine_core_cls = getattr(module, "EngineCore", None)
    engine_core_proc_cls = getattr(module, "EngineCoreProc", None)
    request_type_enum = getattr(module, "EngineCoreRequestType", None)
    if engine_core_cls is None:
        logger.warning(f"EngineCore class not found in {module.__name__}")
        return

    def preprocess_add_request_wrapper(original_func):
        def wrapper(self, request):
            start_ns = time.time_ns()
            request_id = getattr(request, "request_id", None)
            try:
                return original_func(self, request)
            finally:
                end_ns = time.time_ns()
                emit_req_stage(
                    stage="engine_core.preprocess_add_request",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    request_id=request_id,
                )
                if request_id:
                    _remember_core_preprocess_done(request_id, end_ns)
        return wrapper

    def add_request_wrapper(original_func):
        def wrapper(self, request, request_wave=0):
            request_id = getattr(request, "request_id", None)
            start_ns = time.time_ns()
            try:
                return original_func(self, request, request_wave)
            finally:
                emit_req_stage(
                    stage="engine_core.add_request",
                    start_ns=start_ns,
                    end_ns=time.time_ns(),
                    request_id=request_id,
                )
        return wrapper

    apply_method_patch(engine_core_cls, "preprocess_add_request", preprocess_add_request_wrapper)
    apply_method_patch(engine_core_cls, "add_request", add_request_wrapper)

    if engine_core_proc_cls is None or request_type_enum is None:
        return

    def handle_client_request_wrapper(original_func):
        def wrapper(self, request_type, request):
            is_add = request_type == request_type_enum.ADD
            request_id = None
            handle_start_ns = None
            if is_add:
                handle_start_ns = time.time_ns()
                try:
                    req, _ = request
                    request_id = getattr(req, "request_id", None)
                except Exception:
                    request_id = None
                if request_id:
                    preprocess_done_ns = _pop_core_preprocess_done(request_id)
                    if preprocess_done_ns is not None and handle_start_ns >= preprocess_done_ns:
                        emit_req_stage(
                            stage="engine_core.input_queue_wait_after_preprocess",
                            start_ns=preprocess_done_ns,
                            end_ns=handle_start_ns,
                            request_id=request_id,
                        )
            try:
                return original_func(self, request_type, request)
            finally:
                if is_add and handle_start_ns is not None:
                    emit_req_stage(
                        stage="engine_core.handle_client_add",
                        start_ns=handle_start_ns,
                        end_ns=time.time_ns(),
                        request_id=request_id,
                    )
        return wrapper

    apply_method_patch(engine_core_proc_cls, "_handle_client_request", handle_client_request_wrapper)

    def process_input_queue_wrapper(original_func):
        def wrapper(self, *args, **kwargs):
            start_ns = time.time_ns()
            try:
                return original_func(self, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="enginecore_mainloop_span",
                    payload={
                        "phase": "_process_input_queue",
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": end_ns - start_ns,
                        "pid": os.getpid(),
                        "tid": getattr(threading, "get_native_id", threading.get_ident)(),
                        "timestamp_ns": end_ns,
                    },
                )
        return wrapper

    def process_engine_step_wrapper(original_func):
        def wrapper(self, *args, **kwargs):
            start_ns = time.time_ns()
            try:
                return original_func(self, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="enginecore_mainloop_span",
                    payload={
                        "phase": "_process_engine_step",
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": end_ns - start_ns,
                        "pid": os.getpid(),
                        "tid": getattr(threading, "get_native_id", threading.get_ident)(),
                        "timestamp_ns": end_ns,
                    },
                )
        return wrapper

    apply_method_patch(engine_core_proc_cls, "_process_input_queue", process_input_queue_wrapper)
    apply_method_patch(engine_core_proc_cls, "_process_engine_step", process_engine_step_wrapper)

# =============================================================================
# [升级版] 鲁棒的通用拦截器 (Proxy Pattern)
# 解决嵌套导入时 Hook 失效的问题
# =============================================================================

class PatchingLoader:
    """代理加载器：包装原始 Loader，在加载完成后执行 Hook"""
    def __init__(self, original_loader, patch_func):
        self.original_loader = original_loader
        self.patch_func = patch_func

    # 代理所有未定义的方法给原始 Loader (如 get_source, is_package 等)
    def __getattr__(self, name):
        return getattr(self.original_loader, name)

    def create_module(self, spec):
        if hasattr(self.original_loader, 'create_module'):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        # 1. 执行原始加载逻辑 (此时 sys.meta_path 里依然有 Patcher，嵌套导入会被捕获)
        if hasattr(self.original_loader, 'exec_module'):
            self.original_loader.exec_module(module)
        
        # 2. 模块加载完毕，执行插桩
        try:
            self.patch_func(module)
        except Exception as e:
            logger.error(f"Patch failed for {module.__name__}: {e}", exc_info=True)

class VllmUniversalPatcher(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # 1. 只处理注册过的模块
        if fullname not in HOOK_REGISTRY:
            return None
        
        # 2. 遍历 sys.meta_path 寻找真正的加载器 (跳过自己)
        for finder in sys.meta_path:
            if finder is self:
                continue
            
            try:
                # 让后续的 Finder 去找 Spec
                spec = finder.find_spec(fullname, path, target)
            except AttributeError:
                continue
                
            if spec and spec.loader:
                # 3. 偷梁换柱：把 Loader 换成我们的代理 Loader
                spec.loader = PatchingLoader(spec.loader, HOOK_REGISTRY[fullname])
                return spec
        return None


@register_hook("vllm.distributed.parallel_state")
def patch_parallel_state_pp_comm(module):
    group_cls = getattr(module, "GroupCoordinator", None)
    if group_cls is None:
        logger.warning(f"GroupCoordinator not found in {module.__name__}")
        return
    if getattr(group_cls, "_de_latency_pp_comm_patch_installed", False):
        return

    def _wrap_send(original_func):
        def wrapper(self, tensor_dict, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, tensor_dict, *args, **kwargs)
            dst = kwargs.get("dst")
            if dst is None and args:
                dst = args[0]
            if dst is None:
                try:
                    dst = (self.rank_in_group + 1) % self.world_size
                except Exception:
                    dst = None
            start_ns = time.time_ns()
            try:
                return original_func(self, tensor_dict, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="pp_comm_span",
                    payload=_pp_comm_context_payload(
                        self,
                        op="send_tensor_dict",
                        start_ns=start_ns,
                        end_ns=end_ns,
                        peer_rank=dst,
                        dst=dst,
                        tensor_payload=tensor_dict,
                    ),
                )
        return wrapper

    def _wrap_isend(original_func):
        def wrapper(self, tensor_dict, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, tensor_dict, *args, **kwargs)
            dst = kwargs.get("dst")
            if dst is None and args:
                dst = args[0]
            if dst is None:
                try:
                    dst = (self.rank_in_group + 1) % self.world_size
                except Exception:
                    dst = None
            comm_id = f"pp-async-{os.getpid()}-{next(_ASYNC_PP_COMM_SEQ)}"
            start_ns = time.time_ns()
            handles = None
            try:
                handles = original_func(self, tensor_dict, *args, **kwargs)
            except Exception:
                end_ns = time.time_ns()
                error_payload = _pp_comm_context_payload(
                    self,
                    op="isend_tensor_dict_launch",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    peer_rank=dst,
                    dst=dst,
                    tensor_payload=tensor_dict,
                )
                error_payload.update({
                    "comm_id": comm_id,
                    "async_phase": "launch",
                    "launch_start_ns": start_ns,
                    "launch_end_ns": end_ns,
                    "handle_count": 0,
                    "error": True,
                })
                TraceSender.emit(event_type="pp_comm_span", payload=error_payload)
                raise
            end_ns = time.time_ns()
            launch_payload = _pp_comm_context_payload(
                self,
                op="isend_tensor_dict_launch",
                start_ns=start_ns,
                end_ns=end_ns,
                peer_rank=dst,
                dst=dst,
                tensor_payload=tensor_dict,
            )
            launch_payload.update({
                "comm_id": comm_id,
                "async_phase": "launch",
                "launch_start_ns": start_ns,
                "launch_end_ns": end_ns,
                "handle_count": len(handles or []),
            })
            TraceSender.emit(event_type="pp_comm_span", payload=launch_payload)
            return [
                _TracedCommHandle(handle, launch_payload, "isend_tensor_dict_wait", idx)
                for idx, handle in enumerate(handles or [])
            ]
        return wrapper

    def _wrap_recv(original_func):
        def wrapper(self, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, *args, **kwargs)
            src = kwargs.get("src")
            if src is None and args:
                src = args[0]
            if src is None:
                try:
                    src = (self.rank_in_group - 1) % self.world_size
                except Exception:
                    src = None
            start_ns = time.time_ns()
            result = None
            try:
                result = original_func(self, *args, **kwargs)
                return result
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="pp_comm_span",
                    payload=_pp_comm_context_payload(
                        self,
                        op="recv_tensor_dict",
                        start_ns=start_ns,
                        end_ns=end_ns,
                        peer_rank=src,
                        src=src,
                        tensor_payload=result,
                    ),
                )
        return wrapper

    def _wrap_irecv(original_func):
        def wrapper(self, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, *args, **kwargs)
            src = kwargs.get("src")
            if src is None and args:
                src = args[0]
            if src is None:
                try:
                    src = (self.rank_in_group - 1) % self.world_size
                except Exception:
                    src = None
            comm_id = f"pp-async-{os.getpid()}-{next(_ASYNC_PP_COMM_SEQ)}"
            start_ns = time.time_ns()
            tensor_dict = None
            handles = None
            postprocess = None
            try:
                tensor_dict, handles, postprocess = original_func(self, *args, **kwargs)
            except Exception:
                end_ns = time.time_ns()
                error_payload = _pp_comm_context_payload(
                    self,
                    op="irecv_tensor_dict_launch",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    peer_rank=src,
                    src=src,
                    tensor_payload=tensor_dict,
                )
                error_payload.update({
                    "comm_id": comm_id,
                    "async_phase": "launch",
                    "launch_start_ns": start_ns,
                    "launch_end_ns": end_ns,
                    "handle_count": 0,
                    "postprocess_count": 0,
                    "error": True,
                })
                TraceSender.emit(event_type="pp_comm_span", payload=error_payload)
                raise
            end_ns = time.time_ns()
            launch_payload = _pp_comm_context_payload(
                self,
                op="irecv_tensor_dict_launch",
                start_ns=start_ns,
                end_ns=end_ns,
                peer_rank=src,
                src=src,
                tensor_payload=tensor_dict,
            )
            launch_payload.update({
                "comm_id": comm_id,
                "async_phase": "launch",
                "launch_start_ns": start_ns,
                "launch_end_ns": end_ns,
                "handle_count": len(handles or []),
                "postprocess_count": len(postprocess or []),
            })
            TraceSender.emit(event_type="pp_comm_span", payload=launch_payload)
            return (
                tensor_dict,
                [
                    _TracedCommHandle(handle, launch_payload, "irecv_tensor_dict_wait", idx)
                    for idx, handle in enumerate(handles or [])
                ],
                postprocess,
            )
        return wrapper

    def _wrap_broadcast(original_func):
        def wrapper(self, tensor_dict=None, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, tensor_dict, *args, **kwargs)
            src = kwargs.get("src")
            if src is None and args:
                src = args[0]
            if src is None:
                src = 0
            start_ns = time.time_ns()
            result = None
            try:
                result = original_func(self, tensor_dict, *args, **kwargs)
                return result
            finally:
                end_ns = time.time_ns()
                payload_obj = tensor_dict if tensor_dict is not None else result
                TraceSender.emit(
                    event_type="pp_comm_span",
                    payload=_pp_comm_context_payload(
                        self,
                        op="broadcast_tensor_dict",
                        start_ns=start_ns,
                        end_ns=end_ns,
                        peer_rank=src,
                        src=src,
                        tensor_payload=payload_obj,
                    ),
                )
        return wrapper

    def _wrap_send_tensor(original_func):
        def wrapper(self, tensor, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, tensor, *args, **kwargs)
            dst = kwargs.get("dst")
            if dst is None and args:
                dst = args[0]
            if dst is None:
                try:
                    dst = (self.rank_in_group + 1) % self.world_size
                except Exception:
                    dst = None
            start_ns = time.time_ns()
            try:
                return original_func(self, tensor, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="pp_comm_span",
                    payload=_pp_comm_context_payload(
                        self,
                        op="send_tensor",
                        start_ns=start_ns,
                        end_ns=end_ns,
                        peer_rank=dst,
                        dst=dst,
                        tensor_payload=tensor,
                    ),
                )
        return wrapper

    def _wrap_recv_tensor(original_func):
        def wrapper(self, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, *args, **kwargs)
            src = kwargs.get("src")
            if src is None and len(args) >= 3:
                src = args[2]
            if src is None:
                try:
                    src = (self.rank_in_group - 1) % self.world_size
                except Exception:
                    src = None
            start_ns = time.time_ns()
            result = None
            try:
                result = original_func(self, *args, **kwargs)
                return result
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="pp_comm_span",
                    payload=_pp_comm_context_payload(
                        self,
                        op="recv_tensor",
                        start_ns=start_ns,
                        end_ns=end_ns,
                        peer_rank=src,
                        src=src,
                        tensor_payload=result,
                    ),
                )
        return wrapper

    def _wrap_broadcast_tensor(original_func):
        def wrapper(self, tensor, *args, **kwargs):
            if not monkeypatch_enabled() or not _is_pp_group(self):
                return original_func(self, tensor, *args, **kwargs)
            src = kwargs.get("src")
            if src is None and args:
                src = args[0]
            if src is None:
                src = 0
            start_ns = time.time_ns()
            try:
                return original_func(self, tensor, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="pp_comm_span",
                    payload=_pp_comm_context_payload(
                        self,
                        op="broadcast_tensor",
                        start_ns=start_ns,
                        end_ns=end_ns,
                        peer_rank=src,
                        src=src,
                        tensor_payload=tensor,
                    ),
                )
        return wrapper

    apply_method_patch(group_cls, "send_tensor_dict", _wrap_send)
    apply_method_patch(group_cls, "isend_tensor_dict", _wrap_isend)
    apply_method_patch(group_cls, "recv_tensor_dict", _wrap_recv)
    apply_method_patch(group_cls, "irecv_tensor_dict", _wrap_irecv)
    apply_method_patch(group_cls, "broadcast_tensor_dict", _wrap_broadcast)
    apply_method_patch(group_cls, "send", _wrap_send_tensor)
    apply_method_patch(group_cls, "recv", _wrap_recv_tensor)
    apply_method_patch(group_cls, "broadcast", _wrap_broadcast_tensor)
    group_cls._de_latency_pp_comm_patch_installed = True
    logger.info(">>> [HOOK] Patched PP communication spans on GroupCoordinator")


@register_hook("vllm.v1.worker.gpu_model_runner")
def patch_gpu_model_runner(module):
    """
    针对 vLLM GPU Worker 的核心插桩逻辑。
    目标类: GPUModelRunner
    """
    if not hasattr(module, "GPUModelRunner"):
        logger.warning(f"GPUModelRunner not found in {module}")
        return

    cls = module.GPUModelRunner

    if getattr(module, "_de_latency_phase_scope_patch_installed", False):
        logger.info(">>> [HOOK] GPUModelRunner phase scope patch already installed")
    else:
        original_record_function = module.record_function_or_nullcontext

        def patched_record_function_or_nullcontext(name):
            base_cm = original_record_function(name)
            phase_name = _canonical_gpu_phase_name(name)
            if phase_name is None:
                return base_cm

            class _TracePhaseScope:
                def __init__(self, wrapped_cm, phase_name, raw_phase_name):
                    self._wrapped_cm = wrapped_cm
                    self._phase_name = phase_name
                    self._raw_phase_name = raw_phase_name
                    self._start_ns = None

                def __enter__(self):
                    self._start_ns = time.time_ns()
                    return self._wrapped_cm.__enter__()

                def __exit__(self, exc_type, exc, tb):
                    suppressed = False
                    try:
                        suppressed = self._wrapped_cm.__exit__(exc_type, exc, tb)
                    finally:
                        ctx = _current_gpu_exec_context()
                        if ctx is not None and self._start_ns is not None:
                            end_ns = time.time_ns()
                            ctx = _refresh_gpu_exec_context(ctx)
                            payload = {
                                "phase": self._phase_name,
                                "raw_phase": self._raw_phase_name,
                                "pid": ctx.get("pid", os.getpid()),
                                "tid": ctx.get("tid", _get_native_tid()),
                                "req_ids": list(ctx.get("req_ids", [])),
                                "batch_size": ctx.get("batch_size", 0),
                                "input_type": ctx.get("input_type", "unknown"),
                                "ranks": dict(ctx.get("ranks", {})),
                                "dispatch_key": ctx.get("dispatch_key"),
                                "method_name": ctx.get("method_name"),
                                "start_ns": self._start_ns,
                                "end_ns": end_ns,
                                "timestamp_ns": end_ns,
                            }
                            TraceSender.emit(
                                event_type="gpu_phase_span",
                                payload=payload,
                            )
                            _emit_legacy_gpu_phase_events(self._phase_name, payload)
                    return suppressed

            return _TracePhaseScope(base_cm, phase_name, name)

        module.record_function_or_nullcontext = patched_record_function_or_nullcontext
        module._de_latency_phase_scope_patch_installed = True
        logger.info(">>> [HOOK] Patched gpu_model_runner.record_function_or_nullcontext")

    execute_model_already_patched = getattr(
        cls, "_de_latency_execute_model_patch_installed", False
    )
    if execute_model_already_patched:
        logger.info(">>> [HOOK] GPUModelRunner.execute_model patch already installed")

    def execute_model_wrapper(original_func):
        sig = inspect.signature(original_func)

        def wrapper(self, *args, **kwargs):
            scheduler_output = None
            try:
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                scheduler_output = bound_args.arguments.get("scheduler_output")
            except Exception:
                pass

            start_ns = time.time_ns()
            ctx = _build_gpu_exec_context(
                self,
                scheduler_output=scheduler_output,
                method_name=original_func.__name__,
            )
            _push_gpu_exec_context(ctx)
            try:
                return original_func(self, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                ctx = _pop_gpu_exec_context() or ctx
                ctx = _refresh_gpu_exec_context(ctx)
                TraceSender.emit(
                    event_type="gpu_execute_model",
                    payload={
                        "pid": ctx.get("pid", os.getpid()),
                        "tid": ctx.get("tid", _get_native_tid()),
                        "ranks": dict(ctx.get("ranks", {})),
                        "batch_size": ctx.get("batch_size", 0),
                        "input_type": ctx.get("input_type", "unknown"),
                        "hooked_method": original_func.__name__,
                        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
                        "req_ids": list(ctx.get("req_ids", [])),
                        "input_batch_req_ids": list(ctx.get("input_batch_req_ids", [])),
                        "dispatch_key": ctx.get("dispatch_key"),
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                    },
                )

        return wrapper

    if not execute_model_already_patched and apply_method_patch(cls, "execute_model", execute_model_wrapper):
        cls._de_latency_execute_model_patch_installed = True

    if getattr(cls, "_de_latency_sample_tokens_patch_installed", False):
        logger.info(">>> [HOOK] GPUModelRunner.sample_tokens patch already installed")
    elif hasattr(cls, "sample_tokens"):
        def sample_tokens_wrapper(original_func):
            def wrapper(self, *args, **kwargs):
                start_ns = time.time_ns()
                start_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
                scheduler_output = _extract_scheduler_output_from_runner_state(self)
                ctx = _build_gpu_exec_context(
                    self,
                    scheduler_output=scheduler_output,
                    method_name=original_func.__name__,
                )
                _push_gpu_exec_context(ctx)
                try:
                    return original_func(self, *args, **kwargs)
                finally:
                    ctx = _pop_gpu_exec_context() or ctx
                    ctx = _refresh_gpu_exec_context(ctx)
                    end_ns = time.time_ns()
                    end_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
                    req_ids = list(ctx.get("req_ids", []))
                    if req_ids:
                        TraceSender.emit(
                            event_type="gpu_phase_span",
                            payload={
                                "phase": "GPUModelRunner sample_tokens",
                                "pid": ctx.get("pid", os.getpid()),
                                "tid": ctx.get("tid", _get_native_tid()),
                                "req_ids": req_ids,
                                "ranks": dict(ctx.get("ranks", {})),
                                "dispatch_key": ctx.get("dispatch_key"),
                                "batch_size": ctx.get("batch_size", 0),
                                "input_type": ctx.get("input_type", "unknown"),
                                "method_name": original_func.__name__,
                                "start_ns": start_ns,
                                "end_ns": end_ns,
                            },
                        )
                        TraceSender.emit(
                            event_type="worker_model_execute_span",
                            payload={
                                "req_ids": req_ids,
                                "pid": ctx.get("pid", os.getpid()),
                                "tid": ctx.get("tid", _get_native_tid()),
                                "dispatch_key": ctx.get("dispatch_key"),
                                "ranks": dict(ctx.get("ranks", {})),
                                "batch_size": ctx.get("batch_size", 0),
                                "input_type": ctx.get("input_type", "unknown"),
                                "hooked_method": original_func.__name__,
                                "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
                                "start_ns": start_mono,
                                "end_ns": end_mono,
                                "duration_us": (end_mono - start_mono) // 1000,
                            },
                        )
            return wrapper

        if apply_method_patch(cls, "sample_tokens", sample_tokens_wrapper):
            cls._de_latency_sample_tokens_patch_installed = True

@register_hook("vllm.v1.worker.gpu.model_runner")
def patch_v2_gpu_model_runner(module):
    """
    针对 vLLM V2 GPU Worker (vllm/v1/worker/gpu/model_runner.py) 的插桩。
    V2 没有 record_function_or_nullcontext，不产生 gpu_phase_span。
    只 patch execute_model 和 sample_tokens，产出 gpu_execute_model +
    worker_model_execute_span 两个事件。
    """
    if not hasattr(module, "GPUModelRunner"):
        logger.warning(f"GPUModelRunner not found in {module.__name__}")
        return

    cls = module.GPUModelRunner
    logger.info(f">>> [HOOK] Patching V2 GPUModelRunner: {cls}")

    # ---- execute_model --------------------------------------------------
    def v2_execute_model_wrapper(original_func):
        sig = inspect.signature(original_func)

        def wrapper(self, *args, **kwargs):
            scheduler_output = None
            try:
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                scheduler_output = bound_args.arguments.get("scheduler_output")
            except Exception:
                pass

            start_ns = time.time_ns()
            ctx = _build_gpu_exec_context(
                self,
                scheduler_output=scheduler_output,
                method_name=original_func.__name__,
            )
            # V2 的 ExecuteModelState 不含 scheduler_output，
            # 把 ctx 关键字段暂存到 self 上，供 sample_tokens 使用。
            self._de_v2_dispatch_key = ctx.get("dispatch_key")
            self._de_v2_req_ids = list(ctx.get("req_ids", []))
            self._de_v2_ranks = dict(ctx.get("ranks", {}))
            self._de_v2_batch_size = ctx.get("batch_size", 0)

            _push_gpu_exec_context(ctx)
            try:
                return original_func(self, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                ctx = _pop_gpu_exec_context() or ctx
                ctx = _refresh_gpu_exec_context(ctx)
                TraceSender.emit(
                    event_type="gpu_execute_model",
                    payload={
                        "pid": ctx.get("pid", os.getpid()),
                        "tid": ctx.get("tid", _get_native_tid()),
                        "ranks": dict(ctx.get("ranks", {})),
                        "batch_size": ctx.get("batch_size", 0),
                        "input_type": ctx.get("input_type", "unknown"),
                        "hooked_method": original_func.__name__,
                        "class_path": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
                        "req_ids": list(ctx.get("req_ids", [])),
                        "input_batch_req_ids": list(ctx.get("input_batch_req_ids", [])),
                        "dispatch_key": ctx.get("dispatch_key"),
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                    },
                )

        return wrapper

    if apply_method_patch(cls, "execute_model", v2_execute_model_wrapper):
        cls._de_latency_v2_execute_model_patch_installed = True

    # ---- sample_tokens --------------------------------------------------
    if getattr(cls, "_de_latency_v2_sample_tokens_patch_installed", False):
        logger.info(">>> [HOOK] GPUModelRunner.sample_tokens patch already installed")
    elif hasattr(cls, "sample_tokens"):
        def v2_sample_tokens_wrapper(original_func):
            def wrapper(self, *args, **kwargs):
                start_ns = time.time_ns()
                start_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
                dispatch_key = getattr(self, "_de_v2_dispatch_key", None)
                req_ids = getattr(self, "_de_v2_req_ids", [])
                ranks = getattr(self, "_de_v2_ranks", {})
                batch_size = getattr(self, "_de_v2_batch_size", 0)
                ctx = {
                    "dispatch_key": dispatch_key,
                    "req_ids": req_ids,
                    "ranks": ranks,
                    "batch_size": batch_size,
                    "pid": os.getpid(),
                    "tid": _get_native_tid(),
                    "input_type": "unknown",
                    "method_name": original_func.__name__,
                    "runner": self,
                }
                _push_gpu_exec_context(ctx)
                try:
                    return original_func(self, *args, **kwargs)
                finally:
                    ctx = _pop_gpu_exec_context() or ctx
                    end_ns = time.time_ns()
                    end_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
                    actual_req_ids = list(ctx.get("req_ids", req_ids))
                    if actual_req_ids:
                        TraceSender.emit(
                            event_type="gpu_phase_span",
                            payload={
                                "phase": "GPUModelRunner sample_tokens",
                                "pid": ctx.get("pid", os.getpid()),
                                "tid": ctx.get("tid", _get_native_tid()),
                                "req_ids": actual_req_ids,
                                "ranks": dict(ctx.get("ranks", ranks)),
                                "dispatch_key": ctx.get("dispatch_key", dispatch_key),
                                "batch_size": ctx.get("batch_size", batch_size),
                                "input_type": ctx.get("input_type", "unknown"),
                                "method_name": original_func.__name__,
                                "start_ns": start_ns,
                                "end_ns": end_ns,
                            },
                        )
                        TraceSender.emit(
                            event_type="worker_model_execute_span",
                            payload={
                                "req_ids": actual_req_ids,
                                "pid": ctx.get("pid", os.getpid()),
                                "tid": ctx.get("tid", _get_native_tid()),
                                "dispatch_key": ctx.get("dispatch_key", dispatch_key),
                                "ranks": dict(ctx.get("ranks", ranks)),
                                "batch_size": ctx.get("batch_size", batch_size),
                                "input_type": ctx.get("input_type", "unknown"),
                                "hooked_method": original_func.__name__,
                                "class_path": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
                                "start_ns": start_mono,
                                "end_ns": end_mono,
                                "duration_us": (end_mono - start_mono) // 1000,
                            },
                        )
            return wrapper

        if apply_method_patch(cls, "sample_tokens", v2_sample_tokens_wrapper):
            cls._de_latency_v2_sample_tokens_patch_installed = True

@register_hook("vllm.v1.core.sched.scheduler") 
def patch_v1_scheduler(module):
    # 再次确认类名，防止报错
    if not hasattr(module, "Scheduler"):
        logger.warning(f"Scheduler class not found in {module.__name__}")
        return
    
    cls = module.Scheduler
    logger.info(f">>> [HOOK] Patching Scheduler Implementation: {cls}")

    def add_request_wrapper(original_func):
        def wrapper(self, request):
            # 获取 Request ID
            req_id = getattr(request, "request_id", "unknown")
            
            # 发送入队事件
            TraceSender.emit(
                event_type="req_enqueue_scheduler",
                payload={
                    "req_id": req_id,
                    "event": "enqueue_start",
                    "timestamp_ns": time.time_ns()
                }
            )
            return original_func(self, request)
        return wrapper
    def _safe_int(val, default=None):
        try:
            return int(val)
        except Exception:
            return default

    def _phase_from_counts(num_computed_tokens, num_prompt_tokens):
        if num_computed_tokens is None or num_prompt_tokens is None:
            return None
        return "prefill" if num_computed_tokens < num_prompt_tokens else "decode"

    def _collect_scheduler_request_token_state(scheduler_obj):
        prompt_tokens_by_req = {}
        output_tokens_by_req = {}

        def _rid_keys(rid):
            keys = []
            for key in [rid] + _normalize_req_ids(rid):
                if key and key not in keys:
                    keys.append(key)
            return keys

        try:
            req_map = getattr(scheduler_obj, "requests", {}) or {}
            for rid, req in req_map.items():
                rid_keys = _rid_keys(rid)
                npt = _safe_int(getattr(req, "num_prompt_tokens", None))
                if npt is not None:
                    for key in rid_keys:
                        prompt_tokens_by_req[key] = npt
                notok = _safe_int(
                    getattr(req, "num_output_tokens", None),
                )
                placeholders = _safe_int(
                    getattr(req, "num_output_placeholders", None),
                    0,
                )
                if notok is not None:
                    for key in rid_keys:
                        output_tokens_by_req[key] = notok + (placeholders or 0)
        except Exception:
            pass

        return prompt_tokens_by_req, output_tokens_by_req

    def _build_scheduled_phase_map(scheduler_obj, scheduler_output):
        """
        为当前调度步构建 req_id -> phase 映射。
        vLLM 0.20 的调度语义以 context/generation 区分请求阶段：
        新请求或 cached.is_context_phase(req_id) 为 prefill，否则为 decode。
        旧版缺少该信息时，再回退到 num_computed_tokens < num_prompt_tokens。
        """
        phase_by_req = {}
        phase_reason_by_req = {}
        prompt_tokens_by_req, output_tokens_by_req = (
            _collect_scheduler_request_token_state(scheduler_obj)
        )

        # 新请求：使用 scheduler_output 自身的静态数据判断 phase，
        # 不依赖 _collect_scheduler_request_token_state 的 live 状态，
        # 避免异步调度时下一轮 schedule() 修改了 live 状态导致误判。
        try:
            if hasattr(scheduler_output, "scheduled_new_reqs"):
                for req in scheduler_output.scheduled_new_reqs:
                    rid = getattr(req, "request_id", None) or getattr(req, "req_id", None)
                    if not rid:
                        continue
                    num_computed = _safe_int(getattr(req, "num_computed_tokens", None))
                    prompt_ids = getattr(req, "prompt_token_ids", None)
                    num_prompt = _safe_int(len(prompt_ids)) if prompt_ids is not None else None
                    phase = _phase_from_counts(num_computed, num_prompt)
                    if phase:
                        phase_by_req[rid] = phase
                        phase_reason_by_req[rid] = "new_req_static_token_counts"
        except Exception:
            pass

        # 缓存请求：优先使用 vLLM 0.20 的 is_context_phase(req_id)。
        try:
            if hasattr(scheduler_output, "scheduled_cached_reqs"):
                cached = scheduler_output.scheduled_cached_reqs
                if cached is not None and hasattr(cached, "req_ids"):
                    req_ids = list(cached.req_ids)
                    num_computed_list = list(getattr(cached, "num_computed_tokens", []) or [])
                    for i, rid in enumerate(req_ids):
                        if not rid:
                            continue
                        phase = None
                        num_computed = _safe_int(num_computed_list[i]) if i < len(num_computed_list) else None
                        num_prompt = prompt_tokens_by_req.get(rid)
                        # vLLM's scheduler owns the context/generation state.
                        # Token counts are only a fallback: async PP can enqueue
                        # later SchedulerOutputs while earlier context work is
                        # still draining on downstream PP stages.
                        try:
                            if hasattr(cached, "is_context_phase"):
                                phase = "prefill" if cached.is_context_phase(rid) else "decode"
                                phase_reason_by_req[rid] = "cached_is_context_phase"
                        except Exception:
                            phase = None
                        if phase is None:
                            phase = _phase_from_counts(num_computed, num_prompt)
                            if phase:
                                phase_reason_by_req[rid] = "cached_token_counts_fallback"
                        if phase is None and rid in output_tokens_by_req:
                            phase = "prefill" if output_tokens_by_req.get(rid, 0) == 0 else "decode"
                            phase_reason_by_req[rid] = "cached_output_tokens_fallback"
                        if phase:
                            phase_by_req[rid] = phase
        except Exception:
            pass

        return phase_by_req, phase_reason_by_req

    def _build_decode_step_map(
        scheduler_obj,
        scheduler_output,
        phase_by_req: Dict[str, str],
    ) -> Dict[str, int]:
        """Return req_id -> 1-based decode dispatch index for this step."""
        decode_step_by_req = {}
        prompt_tokens_by_req, _ = _collect_scheduler_request_token_state(scheduler_obj)
        num_computed_by_req, _ = _extract_scheduler_step_maps(scheduler_output)

        for rid in _extract_req_ids_from_scheduler_output(scheduler_output):
            if not rid or phase_by_req.get(rid) != "decode":
                continue

            num_computed = _safe_int(num_computed_by_req.get(rid))
            num_prompt = _safe_int(prompt_tokens_by_req.get(rid))
            if num_computed is None or num_prompt is None:
                continue

            step = num_computed - num_prompt + 1
            if step > 0:
                decode_step_by_req[rid] = step

        return decode_step_by_req

    #这个函数帮助我们将update_from_output函数返回的dict[int, EngineCoreOutputs]中的每个req的时间戳事件提取出来
    def extract_events_from_results(results_dict: Dict[int, Any], scheduled_phase_by_req: Dict[str, str]) -> List[Dict]:
        """
        针对确定的 vLLM V1 结构提取事件
        Input: dict[int, EngineCoreOutputs]
        Output: List of event payloads
        """
        extracted_updates = []
        # 1. 遍历字典 (key 是 engine_index)
        for engine_id, core_outputs_batch in results_dict.items():
            # core_outputs_batch 就是 EngineCoreOutputs 对象
            # 2. 获取该 batch 下的所有请求输出列表
            # 根据定义: outputs: list[EngineCoreOutput] = []
            if not hasattr(core_outputs_batch, "outputs"):
                continue
            #request_outputs_list=outputs: outputs: list[EngineCoreOutput] = []
            request_outputs_list = core_outputs_batch.outputs
            # 3. 遍历每个请求的输出
            for req_output in request_outputs_list:
                req_id = getattr(req_output, "request_id", "unknown")
                # req_output 就是 EngineCoreOutput 对象
                
                # A. 必须要有 events 才有发送价值
                # 定义: events: Optional[list[EngineCoreEvent]] = None
                if not hasattr(req_output, "events") or not req_output.events:
                    continue
                # B. 解析事件列表
                parsed_events = []
                for evt in req_output.events:
                    # EngineCoreEvent 通常有 type 和 timestamp 属性
                    # 有些版本可能是 tuple，做个防御性读取
                    e_type = getattr(evt, "type", None)
                    e_ts = getattr(evt, "timestamp", None)
                    
                    # 如果是 Enum，转字符串
                    if e_type is not None:
                        e_type = str(e_type)

                    if e_type and e_ts:
                        parsed = {
                            "type": e_type,
                            "ts": e_ts
                        }
                        up = e_type.upper()
                        is_scheduled = (e_type == "2") or ("SCHEDULED" in up)
                        if is_scheduled:
                            phase = scheduled_phase_by_req.get(req_id)
                            if phase in ("prefill", "decode"):
                                parsed["phase"] = phase
                        parsed_events.append(parsed)
                
                # C. 如果提取到了事件，加入列表
                if parsed_events:
                    extracted_updates.append({
                        "req_id": req_id,
                        "events": parsed_events,
                        # 还可以带上 engine_id 方便调试
                        "engine_id": engine_id 
                    })

        return extracted_updates

    def update_from_output_wrapper(original_func):
        def wrapper(self, scheduler_output, model_output):
            scheduled_phase_by_req, phase_reason_by_req = _build_scheduled_phase_map(
                self,
                scheduler_output,
            )
            decode_step_by_req = _build_decode_step_map(
                self,
                scheduler_output,
                scheduled_phase_by_req,
            )
            dispatch_key = _make_dispatch_key(
                scheduler_output,
                phase_by_req=scheduled_phase_by_req,
            )
            # 1. 先执行原逻辑 (确保状态更新完毕)
            res = original_func(self, scheduler_output, model_output)
            
            # 2. 采集时间戳：这就是 "Ready for Next Step" 的时间
            # 我们需要知道哪些请求 Ready 了，这就在 scheduler_output 里
            req_ids = []
            try:
                # 复用之前的提取逻辑
                if hasattr(scheduler_output, "scheduled_new_reqs"):
                    for req in scheduler_output.scheduled_new_reqs:
                        if hasattr(req, "request_id"): req_ids.append(req.request_id)
                        elif hasattr(req, "req_id"): req_ids.append(req.req_id)

                if hasattr(scheduler_output, "scheduled_cached_reqs"):
                    cached = scheduler_output.scheduled_cached_reqs
                    if cached and hasattr(cached, "req_ids"):
                        req_ids.extend(cached.req_ids)
            except:
                pass

            if req_ids:
                TraceSender.emit(
                    event_type="req_step_ready", # 事件名：这一步跑完了，准备好跑下一步了
                    payload={
                        "req_ids": req_ids,
                        "phase_by_req": scheduled_phase_by_req,
                        "phase_reason_by_req": phase_reason_by_req,
                        "decode_step_by_req": decode_step_by_req,
                        "dispatch_key": dispatch_key,
                        "event": "step_ready",
                        "num_computed_by_req": _extract_scheduler_step_maps(scheduler_output)[0],
                        "scheduled_tokens_by_req": _extract_scheduler_step_maps(scheduler_output)[1],
                        "timestamp_ns": time.time_ns(),
                    }
                )
            try:
                req_events = extract_events_from_results(res, scheduled_phase_by_req)
                if req_events:
                    #这里发送的是在这一轮update_from_output中多个req的多个时间戳时间
                    #需要在后续进行解析，将每个req的时间平铺
                    TraceSender.emit(
                        event_type="req_metrics_events",
                        payload={
                            "req_events": req_events,
                            "dispatch_key": dispatch_key,
                        }
                    )
            except Exception as e:
                print(f"[Probe Error] Extraction failed: {e}")

            return res
        return wrapper
    
    apply_method_patch(cls, "update_from_output", update_from_output_wrapper)
    apply_method_patch(cls, "add_request", add_request_wrapper)

#某些请求被调度出并rpc的时间点插桩（1.调度结束时间点 2.enginecore向worker广播的开始时间点）
@register_hook("vllm.v1.executor.multiproc_executor")
def patch_v1_executor(module):
    if not hasattr(module, "MultiprocExecutor"):
        return
    
    cls = module.MultiprocExecutor
    # logger.info(f">>> [HOOK] Patching Executor: {cls}")

    apply_method_patch(cls, "execute_model", _make_scheduler_out_rpc_wrapper)


@register_hook("vllm.v1.executor.abstract")
def patch_v1_uniproc_executor(module):
    target_classes = []
    for candidate in ["UniProcExecutor", "Executor"]:
        cls = getattr(module, candidate, None)
        if cls is not None:
            target_classes.append(cls)
    if not target_classes:
        return

    for target_cls in target_classes:
        apply_method_patch(target_cls, "execute_model", _make_scheduler_out_rpc_wrapper)


def _patch_executor_module(module, candidate_names):
    target_classes = []
    for candidate in candidate_names:
        cls = getattr(module, candidate, None)
        if cls is not None:
            target_classes.append(cls)
    if not target_classes:
        return

    for target_cls in target_classes:
        apply_method_patch(target_cls, "execute_model", _make_scheduler_out_rpc_wrapper)


@register_hook("vllm.v1.executor.uniproc_executor")
def patch_v1_uniproc_executor_impl(module):
    _patch_executor_module(
        module,
        ["UniProcExecutor", "ExecutorWithExternalLauncher"],
    )


@register_hook("vllm.v1.executor.ray_executor")
def patch_v1_ray_executor(module):
    _patch_executor_module(
        module,
        ["RayDistributedExecutor"],
    )


@register_hook("vllm.v1.worker.gpu_worker")
def patch_worker_base(module):
    target_cls_name = None
    cls = None
    for candidate in ["Worker", "GPUWorker", "WorkerBase"]:
        if hasattr(module, candidate):
            target_cls_name = candidate
            cls = getattr(module, candidate)
            break

    if cls is None:
        logger.warning(f"⚠️ [HOOK FAIL] worker class not found in {module.__name__}")
        return

    logger.info(f">>> [HOOK SUCCESS] Patching Worker Entry: {cls.__name__}")

    def emit_ready_wrapper_factory(stage):
        def _factory(original_func):
            def _wrapper(self, *args, **kwargs):
                result = original_func(self, *args, **kwargs)
                emit_worker_process_ready(stage=stage)
                return result
            return _wrapper
        return _factory

    def execute_model_wrapper(original_func):
        def wrapper(self, scheduler_output, *args, **kwargs):
            emit_worker_process_ready(stage="execute_model")
            # --- [采集点：Worker 进程入口] ---
            # 无论这是不是 Wrapper，这里都是 Worker 收到任务的第一时间
            start_ns = time.time_ns()
            try:
                tid = threading.get_native_id()
            except AttributeError:
                tid = threading.get_ident()
            # [DEBUG] 确认被调用
            #print(f">>> [WORKER DEBUG] WorkerBase.execute_model HIT! PID={os.getpid()}")

            summary = _summarize_scheduler_output(scheduler_output)
            req_ids = list(summary.get("req_ids") or _extract_req_ids_from_scheduler_output(scheduler_output))
            dispatch_key = _make_dispatch_key(scheduler_output)
            ranks = _get_parallel_rank_info(self)
            ctx = {
                "runner": getattr(self, "model_runner", None),
                "pid": os.getpid(),
                "tid": tid,
                "req_ids": req_ids,
                "batch_size": summary.get("batch_size", 0),
                "input_type": summary.get("input_type", "unknown"),
                "ranks": ranks,
                "dispatch_key": dispatch_key,
            }
            
            if req_ids:
                TraceSender.emit(
                    event_type="worker_preprocess_start",
                    payload={
                        "req_ids": req_ids,
                        "event": "preprocess_start",
                        "dispatch_key": dispatch_key,
                        "ranks": ranks,
                        "timestamp_ns": start_ns,
                        "pid": os.getpid(),
                        "tid": tid
                    }
                )
            # ---------------------------------------
            start_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
            _push_gpu_exec_context(ctx)
            try:
                return original_func(self, scheduler_output, *args, **kwargs)
            finally:
                _pop_gpu_exec_context()
                end_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
                if req_ids:
                    TraceSender.emit(
                        event_type="worker_model_execute_span",
                        payload={
                            "req_ids": req_ids,
                            "pid": os.getpid(),
                            "tid": tid,
                            "dispatch_key": dispatch_key,
                            "ranks": ranks,
                            "start_ns": start_mono,
                            "end_ns": end_mono,
                            "duration_us": (end_mono - start_mono) // 1000
                        }
                    )
        return wrapper

    # 尝试 Hook execute_model
    if hasattr(cls, "execute_model"):
        apply_method_patch(cls, "execute_model", execute_model_wrapper)
    else:
        logger.warning(f"⚠️ [HOOK FAIL] execute_model not found in {target_cls_name}")

    # 尽早发 ready 事件：不同版本方法名可能不同，尽量多兼容几个初始化路径
    for method_name in ["__init__", "init_device", "load_model"]:
        if hasattr(cls, method_name):
            apply_method_patch(cls, method_name, emit_ready_wrapper_factory(method_name))


# --- (示例) 可以在这里添加更多模块的 Hook ---
# @register_hook("vllm.core.scheduler")
# def patch_scheduler(module):
#     ...


def _patch_already_loaded_modules():
    for module_name, hook_func in list(HOOK_REGISTRY.items()):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        try:
            hook_func(module)
        except Exception as exc:
            logger.exception("Failed to patch already loaded module %s: %s", module_name, exc)


#整个程序的入口，一旦执行了这个insert的进程都会在每次import的时候自动执行上方的find_spec
#如果find_spec返回成功则执行exec_module方法
if not _SKIP_TRACE_BOOTSTRAP:
    _patch_already_loaded_modules()
    sys.meta_path.insert(0, VllmUniversalPatcher())
    logger.info(">>> [TRACE SYSTEM] VllmUniversalPatcher installed. Ready to intercept.")
else:
    logger.info(">>> [TRACE SYSTEM] sitecustomize bootstrap skipped for control/utility process.")
