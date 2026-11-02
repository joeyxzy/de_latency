"""Process-local worker state and payload construction; no hook installation."""
import os
import threading

from ..execution_context import current
from ..scheduler_metadata import dispatch_key, request_ids, safe_int, step_maps, normalize_req_ids


class ProcessState:
    last_dispatch_key = None
    ready_pids = set()
    preprocess_done = {}
    lock = threading.Lock()


def ranks(obj=None):
    config = getattr(obj, "parallel_config", None)
    if config is None:
        config = getattr(getattr(obj, "vllm_config", None), "parallel_config", None)
    result = dict.fromkeys(("rank", "dp", "tp", "pp", "dp_size", "tp_size", "pp_size"), -1)
    result.update(is_pp_first=None, is_pp_last=None)
    if config is not None:
        for key, attr in (("rank", "rank"), ("dp", "data_parallel_rank"),
                          ("dp_size", "data_parallel_size"), ("tp_size", "tensor_parallel_size"),
                          ("pp_size", "pipeline_parallel_size")):
            result[key] = safe_int(getattr(config, attr, None), -1)
        if result["rank"] >= 0 and result["tp_size"] > 0:
            result["tp"] = result["rank"] % result["tp_size"]
            if result["pp_size"] > 0:
                result["pp"] = (result["rank"] // result["tp_size"]) % result["pp_size"]
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        pp, tp = get_pp_group(), get_tp_group()
        for name, group in (("pp", pp), ("tp", tp)):
            result[name] = safe_int(getattr(group, "rank_in_group", None), result[name])
            result[name + "_size"] = safe_int(getattr(group, "world_size", None), result[name + "_size"])
        result.update(is_pp_first=bool(pp.is_first_rank), is_pp_last=bool(pp.is_last_rank))
    except Exception:
        pass  # Groups are not initialized during worker construction.
    return result


def summarize(output):
    computed, scheduled = step_maps(output)
    size, kind = 0, "unknown"
    if output is not None:
        size, kind = safe_int(output.total_num_scheduled_tokens, 0), "scheduler_output_v1"
    return {"req_ids": request_ids(output), "batch_size": size, "input_type": kind,
            "num_computed_by_req": computed, "scheduled_tokens_by_req": scheduled}


def refresh(context):
    batch = getattr(context.get("runner"), "input_batch", None)
    ids = normalize_req_ids(getattr(batch, "req_ids", None))
    if ids:
        context["input_batch_req_ids"] = ids
        if not context.get("req_ids"):
            context["req_ids"] = ids
    return context


def runner_output(runner):
    state = getattr(runner, "execute_model_state", None)
    if state is None:
        return None
    if getattr(state, "scheduler_output", None) is not None:
        return state.scheduler_output
    try:
        return state[0]
    except (TypeError, IndexError):
        return None


def build(runner, output=None, method=None):
    summary = summarize(output)
    key = dispatch_key(output) if output is not None else None
    parent = current()
    if output is None and parent is not None:
        key = parent.get("dispatch_key")
        if not summary["req_ids"]:
            summary["req_ids"] = parent.get("req_ids", [])
    context = {**summary, "runner": runner, "pid": os.getpid(), "tid": threading.get_native_id(),
               "ranks": ranks(runner), "dispatch_key": key or dispatch_key(output), "method_name": method}
    return refresh(context)


def fields(context):
    return {key: context.get(key) for key in
            ("pid", "tid", "req_ids", "ranks", "dispatch_key", "batch_size", "input_type")}
