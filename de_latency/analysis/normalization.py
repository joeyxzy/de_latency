"""Shared numerical and clock normalization; no rendering dependencies."""
from collections import defaultdict
from .clocks import to_ns

def count_prefill_decode_tokens(scheduled_tokens_by_req):
    prefill_tokens = 0
    decode_tokens = 0
    if isinstance(scheduled_tokens_by_req, dict):
        for rid, count in scheduled_tokens_by_req.items():
            cnt = int(count) if count is not None else 0
            if cnt == 1:
                decode_tokens += 1
            elif cnt > 1:
                prefill_tokens += cnt
    return prefill_tokens, decode_tokens

def infer_trace_origin_ns(
    worker_events,
    scheduler_events,
    coroutine_events,
    coroutine_exec_events,
    output_socket_sched_events,
    output_socket_exec_events,
    output_handler_sched_events,
    output_handler_exec_events,
    req_stage_events,
    enginecore_loop_events,
    pp_comm_events,
    execute_model_span,
    cupti_events,
):
    """
    推断 wall 时间轴的最小 ns 作为 trace 原点。
    目的是把超大绝对时间戳平移到接近 0，避免浮点在 1e15 us 量级产生 0.25us 量化误差。
    """
    min_ns = None

    def _update(candidate):
        nonlocal min_ns
        ns = to_ns(candidate)
        if ns is None:
            return
        if min_ns is None or ns < min_ns:
            min_ns = ns

    for ev in worker_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in scheduler_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns"))

    for ev in coroutine_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))

    for ev in coroutine_exec_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in output_socket_sched_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("ready_ts_ns"))
        _update(p.get("run_ts_ns"))

    for ev in output_socket_exec_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in output_handler_sched_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("ready_ts_ns"))
        _update(p.get("run_ts_ns"))

    for ev in output_handler_exec_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in req_stage_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in enginecore_loop_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    for ev in pp_comm_events:
        p = ev.get("payload", {})
        _update(p.get("timestamp_ns", ev.get("ts")))
        _update(p.get("start_ns"))
        _update(p.get("end_ns"))

    # CUPTI 当前日志里的 start_ns/end_ns 也是 wall-clock 对齐的 epoch ns。
    # 如果不纳入 trace origin 推断，trace 一开头的 CUDA init/runtime/kernel
    # 会被整体平移到负时间戳，导入 Perfetto 时被 trace_sorter 丢弃。
    for ev in cupti_events:
        _update(ev.get("start_ns"))
        _update(ev.get("end_ns"))

    # NOTE:
    # execute_model_span 当前来自 worker 侧 monotonic 时钟，
    # 而绝大多数其余事件使用 wall clock (time.time_ns)。
    # 这里不能把 mono 时间直接混进 trace origin 推断，
    # 否则会导致 wall 时间事件整体被平移到一个极大的 us 位置，
    # 看起来像“只剩下 worker overlap 可见”。

    return min_ns or 0

def to_plain_data(value):
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_data(v) for v in value]
    if isinstance(value, tuple):
        return [to_plain_data(v) for v in value]
    return value

