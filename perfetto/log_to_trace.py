import json
import sys
from collections import defaultdict, deque
import os
import math
from decimal import Decimal, InvalidOperation

TRACE_TS_ORIGIN_NS = 0

def load_log_lines(filepath):
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                # 兼容某些情况下 json 前面有杂质的情况
                json_start = line.find('{')
                if json_start != -1:
                    events.append(json.loads(line[json_start:]))
            except json.JSONDecodeError:
                continue
    return events

def create_perfetto_event(name, cat, ph, ts, dur, pid, tid, args=None, cname=None):
    # Perfetto 要求时间单位是微秒 (us)
    event = {
        "name": name, "cat": cat, "ph": ph, "ts": (ts - TRACE_TS_ORIGIN_NS) / 1000.0, 
        "pid": pid, "tid": tid, "args": args or {}
    }
    if dur is not None: 
        event["dur"] = dur / 1000.0
    if cname:
        event["cname"] = cname
    return event

def create_flow_event(ph, ts, pid, tid, corr_id):
    return {
        "name": "link", "cat": "flow", "ph": ph, "ts": (ts - TRACE_TS_ORIGIN_NS) / 1000.0, 
        "pid": pid, "tid": tid, "id": corr_id
    }


def emit_thread_name_once(trace_events, emitted_keys, pid, tid, name):
    key = (pid, tid, name)
    if key in emitted_keys:
        return
    emitted_keys.add(key)
    trace_events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "args": {"name": name},
    })


def build_request_lane_tids(req_index):
    # Keep queue/exec producers on dedicated lanes to avoid Perfetto hiding
    # overlapping slices on the same request track.
    base = req_index * 100
    return {
        "label": base + 1,
        "lifecycle": base + 2,
        "vllm_queue": base + 3,
        "coro_queue": base + 4,
        "output_socket_queue": base + 5,
        "output_queue": base + 6,
        "dispatch": base + 7,
        "generate_exec": base + 8,
        "output_socket_exec": base + 9,
        "output_exec": base + 10,
        "os": base + 11,
        "stage": base + 12,
        "task": base + 13,
        "pp_gap": base + 14,
    }


def describe_gpu_phase(phase_name):
    styles = {
        "Preprocess": ("Worker Preprocess", "python", "good"),
        "Forward": ("Model Forward", "python", "olive"),
        "Postprocess": ("Postprocess", "python", "rail_load"),
        "Sample": ("Sampling", "python", "mauve"),
        "GPUModelRunner sample_tokens": ("GPUModelRunner sample_tokens", "python", "mauve"),
        "Bookkeep": ("Bookkeeping/Sync", "python", "cyan"),
        "Draft": ("Draft", "python", "yellow"),
        "EPLB": ("EPLB", "python", "rail_idle"),
        "ModelRunnerOutput": ("Model Runner Output", "python", "rail_response"),
    }
    return styles.get(
        phase_name,
        (f"GPU Phase: {phase_name}", "python", "background"),
    )

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

def nudge_equal_boundaries(
    trace_events,
    target_pid=None,
    epsilon_us=0.001,
    cat_prefixes=None,
    snap_tolerance_us=0.0,
):
    """
    某些可视化器在同一轨道上遇到“前一个块的 end == 后一个块的 start”时，
    可能出现后一个块不稳定渲染。这里做最小量微调：
    - 仅处理 ph='X' 且 dur>0 的区间事件；
    - 同一 (pid, tid) 轨道上，若 start 与上一个 end 完全相等，则将 start + epsilon，
      同时 dur - epsilon（保持 end 基本不变，且不会改成负数）。
    """
    prefixes = tuple(cat_prefixes or ())
    slots = []
    for idx, ev in enumerate(trace_events):
        if ev.get("ph") != "X":
            continue
        if target_pid is not None and ev.get("pid") != target_pid:
            continue
        if prefixes:
            cat = ev.get("cat", "")
            if not any(str(cat).startswith(pre) for pre in prefixes):
                continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if ts is None or dur is None:
            continue
        try:
            ts = float(ts)
            dur = float(dur)
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        slots.append((ev.get("pid"), ev.get("tid"), ts, ts + dur, idx))

    slots.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    last_end_by_track = {}
    nudged = 0

    for pid, tid, start, _end, idx in slots:
        track = (pid, tid)
        prev_end = last_end_by_track.get(track)
        if prev_end is not None:
            gap = start - prev_end
        else:
            gap = None

        # 两类情况都做修正：
        # 1) 完全相等：start == prev_end
        # 2) 近似贴边/微小重叠：|start - prev_end| <= snap_tolerance_us
        need_nudge = (
            prev_end is not None
            and (
                start == prev_end
                or (
                    snap_tolerance_us > 0
                    and gap is not None
                    and -snap_tolerance_us <= gap <= snap_tolerance_us
                )
            )
        )

        if need_nudge:
            ev = trace_events[idx]
            old_dur = float(ev["dur"])
            # 只做最小修正，保持事件仍为正时长
            target_start = prev_end + epsilon_us
            delta = target_start - start
            if delta > 0 and old_dur > delta:
                ev["ts"] = start + delta
                ev["dur"] = old_dur - delta
                nudged += 1
                start = float(ev["ts"])
                _end = start + float(ev["dur"])
        last_end_by_track[track] = _end

    return nudged


def clip_negative_complete_events(trace_events):
    clipped = 0
    dropped = 0
    kept = []
    for ev in trace_events:
        if ev.get("ph") != "X":
            kept.append(ev)
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        try:
            ts = float(ts)
            dur = float(dur)
        except (TypeError, ValueError):
            kept.append(ev)
            continue
        if ts >= 0:
            kept.append(ev)
            continue
        end = ts + dur
        if end <= 0:
            dropped += 1
            continue
        ev["ts"] = 0.0
        ev["dur"] = end
        clipped += 1
        kept.append(ev)
    trace_events[:] = kept
    return clipped, dropped

def to_ns(raw_ts):
    """
    统一时间戳到 ns。
    - 若原值是大整数（>1e14），认为已是 ns。
    - 否则按秒（float）转 ns（兼容 EngineCoreEvent.timestamp）。
    """
    if raw_ts is None:
        return None

    # 优先走整数通道，避免 1e18 量级 ns 经 float 造成 256ns 级精度丢失。
    if isinstance(raw_ts, bool):
        return None
    if isinstance(raw_ts, int):
        val_int = raw_ts
        if val_int <= 0:
            return None
        if val_int > 1e14:
            return val_int
        return int(val_int * 1e9)

    if isinstance(raw_ts, str):
        s = raw_ts.strip()
        if not s:
            return None
        try:
            if s.isdigit() or (s[0] in "+-" and s[1:].isdigit()):
                val_int = int(s)
                if val_int <= 0:
                    return None
                if val_int > 1e14:
                    return val_int
                return int(val_int * 1e9)

            val_dec = Decimal(s)
            if not val_dec.is_finite() or val_dec <= 0:
                return None
            if val_dec > Decimal("1e14"):
                return int(val_dec)
            return int(val_dec * Decimal("1e9"))
        except (ValueError, InvalidOperation):
            return None

    if isinstance(raw_ts, float):
        if not math.isfinite(raw_ts) or raw_ts <= 0:
            return None
        if raw_ts > 1e14:
            return int(raw_ts)
        return int(raw_ts * 1e9)

    try:
        val_int = int(raw_ts)
        if val_int <= 0:
            return None
        if val_int > 1e14:
            return val_int
        return int(val_int * 1e9)
    except (TypeError, ValueError):
        return None

def to_int(raw_val):
    if raw_val is None:
        return None
    try:
        return int(raw_val)
    except (TypeError, ValueError):
        return None

def normalize_request_id(raw_rid):
    if raw_rid is None or isinstance(raw_rid, bool):
        return None

    if isinstance(raw_rid, bytes):
        try:
            raw_rid = raw_rid.decode("utf-8", errors="ignore")
        except Exception:
            raw_rid = str(raw_rid)

    if isinstance(raw_rid, str):
        rid = raw_rid.strip()
        return rid or None

    if isinstance(raw_rid, (int, float, Decimal)):
        rid = str(raw_rid).strip()
        return rid or None

    if isinstance(raw_rid, dict):
        for key in ("request_id", "req_id", "id", "rid", "value"):
            if key in raw_rid:
                candidate = normalize_request_id(raw_rid.get(key))
                if candidate:
                    return candidate
        try:
            rid = json.dumps(raw_rid, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError):
            rid = str(raw_rid)
        rid = rid.strip()
        return rid or None

    if isinstance(raw_rid, (list, tuple, set)):
        if len(raw_rid) == 1:
            return normalize_request_id(next(iter(raw_rid)))
        items = [normalize_request_id(item) for item in raw_rid]
        items = [item for item in items if item]
        if not items:
            return None
        try:
            rid = json.dumps(items, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError):
            rid = str(items)
        rid = rid.strip()
        return rid or None

    rid = str(raw_rid).strip()
    return rid or None

def normalize_request_ids(raw_req_ids):
    if raw_req_ids is None:
        return []
    if isinstance(raw_req_ids, (list, tuple, set)):
        values = raw_req_ids
    else:
        values = [raw_req_ids]
    normalized = []
    for item in values:
        rid = normalize_request_id(item)
        if rid:
            normalized.append(rid)
    return normalized

def normalize_payload_request_fields(payload):
    if not isinstance(payload, dict):
        return payload

    request_name = normalize_request_id(payload.get("request_name"))
    if "request_name" in payload:
        payload["request_name"] = request_name

    if "request_id" in payload:
        raw_request_id = payload.get("request_id")
        normalized_request_id = normalize_request_id(raw_request_id)
        # 有些日志把 prompt/prompt_token_ids 整个对象塞进 request_id。
        # 这类场景优先使用 request_name 作为稳定主键，避免污染汇总输出。
        if isinstance(raw_request_id, (dict, list, tuple, set)) and request_name:
            normalized_request_id = request_name
        payload["request_id"] = normalized_request_id
    if "req_id" in payload:
        raw_req_id = payload.get("req_id")
        normalized_req_id = normalize_request_id(raw_req_id)
        if isinstance(raw_req_id, (dict, list, tuple, set)) and request_name:
            normalized_req_id = request_name
        payload["req_id"] = normalized_req_id
    if "req_ids" in payload:
        payload["req_ids"] = normalize_request_ids(payload.get("req_ids"))

    req_events = payload.get("req_events")
    if isinstance(req_events, list):
        for update in req_events:
            if isinstance(update, dict) and "req_id" in update:
                update["req_id"] = normalize_request_id(update.get("req_id"))

    return payload

def canonicalize_payload_request_fields(payload, resolve_request_id):
    if not isinstance(payload, dict):
        return payload

    if "request_id" in payload:
        payload["request_id"] = resolve_request_id(payload.get("request_id"))
    if "req_id" in payload:
        payload["req_id"] = resolve_request_id(payload.get("req_id"))
    if "request_name" in payload:
        payload["request_name"] = resolve_request_id(payload.get("request_name"))
    if "external_request_id" in payload:
        payload["external_request_id"] = resolve_request_id(payload.get("external_request_id"))
    if "req_ids" in payload:
        payload["req_ids"] = [
            rid for rid in (resolve_request_id(item) for item in payload.get("req_ids", []))
            if rid
        ]

    phase_by_req = payload.get("phase_by_req")
    if isinstance(phase_by_req, dict):
        merged_phase_by_req = {}
        for rid, phase in phase_by_req.items():
            canonical = resolve_request_id(rid)
            if canonical:
                merged_phase_by_req[canonical] = phase
        payload["phase_by_req"] = merged_phase_by_req

    phase_reason_by_req = payload.get("phase_reason_by_req")
    if isinstance(phase_reason_by_req, dict):
        merged_phase_reason_by_req = {}
        for rid, reason in phase_reason_by_req.items():
            canonical = resolve_request_id(rid)
            if canonical:
                merged_phase_reason_by_req[canonical] = reason
        payload["phase_reason_by_req"] = merged_phase_reason_by_req

    decode_step_by_req = payload.get("decode_step_by_req")
    if isinstance(decode_step_by_req, dict):
        merged_decode_step_by_req = {}
        for rid, step in decode_step_by_req.items():
            canonical = resolve_request_id(rid)
            step_int = to_int(step)
            if canonical and step_int is not None:
                merged_decode_step_by_req[canonical] = step_int
        payload["decode_step_by_req"] = merged_decode_step_by_req

    req_events = payload.get("req_events")
    if isinstance(req_events, list):
        for update in req_events:
            if isinstance(update, dict) and "req_id" in update:
                update["req_id"] = resolve_request_id(update.get("req_id"))

    return payload

def normalize_req_event_type(raw_type):
    if raw_type is None:
        return None
    s = str(raw_type).strip()
    up = s.upper()

    if s == "2" or "DEQUEUE" in up or "SCHEDULED" in up or "DISPATCH" in up:
        return "dequeue"
    if s == "3" or "PREEMPT" in up:
        return "preempt"
    if s == "1" or "QUEUE" in up or "ENQUEUE" in up:
        return "enqueue"
    return None

def median_int(values):
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) // 2


def clip_interval(start_ns, end_ns, window_start_ns=None, window_end_ns=None):
    start = to_ns(start_ns)
    end = to_ns(end_ns)
    if start is None or end is None or end <= start:
        return None
    if window_start_ns is not None:
        start = max(start, window_start_ns)
    if window_end_ns is not None:
        end = min(end, window_end_ns)
    if end <= start:
        return None
    return start, end


def merge_intervals(intervals):
    merged = []
    for item in sorted(intervals, key=lambda x: (x[0], x[1])):
        start, end = item
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def sum_interval_pairs(intervals):
    return sum(max(0, end - start) for start, end in intervals)


def uncovered_interval_ns(window_start_ns, window_end_ns, covered_intervals):
    clipped = []
    for start_ns, end_ns in covered_intervals:
        span = clip_interval(start_ns, end_ns, window_start_ns, window_end_ns)
        if span:
            clipped.append(span)
    total_window = max(0, int(window_end_ns) - int(window_start_ns))
    return max(0, total_window - sum_interval_pairs(merge_intervals(clipped)))


def sum_clipped_interval_items(intervals, window_start_ns=None, window_end_ns=None, merge_overlaps=False):
    total_ns = 0
    clipped = []
    for item in intervals:
        span = clip_interval(item.get("start_ns"), item.get("end_ns"), window_start_ns, window_end_ns)
        if not span:
            continue
        clipped.append(span)
    if merge_overlaps:
        clipped = merge_intervals(clipped)
    total_ns = sum_interval_pairs(clipped)
    return total_ns, clipped


def sum_intersections(intervals_a, intervals_b):
    total_ns = 0
    intersections = []
    for item_a in intervals_a:
        start_a = to_ns(item_a.get("start_ns"))
        end_a = to_ns(item_a.get("end_ns"))
        if start_a is None or end_a is None or end_a <= start_a:
            continue
        for item_b in intervals_b:
            start_b = to_ns(item_b.get("start_ns"))
            end_b = to_ns(item_b.get("end_ns"))
            if start_b is None or end_b is None or end_b <= start_b:
                continue
            start = max(start_a, start_b)
            end = min(end_a, end_b)
            if end > start:
                total_ns += end - start
                intersections.append((start, end))
    return total_ns, intersections


def assign_non_overlapping_lanes(intervals, start_key="start_ns", end_key="end_ns"):
    lane_ends = []
    assignments = []
    ordered = sorted(
        enumerate(intervals),
        key=lambda item: (
            to_ns(item[1].get(start_key)) or 0,
            to_ns(item[1].get(end_key)) or 0,
            item[0],
        ),
    )
    for original_idx, item in ordered:
        start = to_ns(item.get(start_key))
        end = to_ns(item.get(end_key))
        if start is None or end is None or end <= start:
            assignments.append((original_idx, 0))
            continue
        lane_idx = None
        for idx, lane_end in enumerate(lane_ends):
            if start >= lane_end:
                lane_idx = idx
                break
        if lane_idx is None:
            lane_idx = len(lane_ends)
            lane_ends.append(end)
        else:
            lane_ends[lane_idx] = end
        assignments.append((original_idx, lane_idx))

    lane_by_original = [0] * len(intervals)
    for original_idx, lane_idx in assignments:
        lane_by_original[original_idx] = lane_idx
    return lane_by_original, len(lane_ends)


def build_phase_merged_dispatch_intervals(intervals, merge_gap_ns=1_000_000):
    """Build request-view phase bands from per-step dispatch intervals.

    The raw per-step intervals intentionally keep PP pipeline overlap.  For the
    request timeline, however, those overlaps make one request fan out into
    multiple visual dispatch lanes.  Merge same-phase intervals into wall-clock
    bands and keep the detailed per-step/rank view on worker lanes.
    """
    grouped = defaultdict(list)
    for interval in intervals:
        start_ns = to_ns(interval.get("start_ns"))
        end_ns = to_ns(interval.get("end_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        phase = interval.get("phase") or "unknown"
        item = dict(interval)
        item["start_ns"] = start_ns
        item["end_ns"] = end_ns
        item["phase"] = phase
        grouped[phase].append(item)

    merged = []
    for phase, items in grouped.items():
        items.sort(key=lambda item: (item["start_ns"], item["end_ns"]))
        current = None
        current_keys = []
        current_batch_req_ids = set()
        current_rpc_wait_ns_values = []

        def _finish_current():
            if current is None:
                return
            keys = list(dict.fromkeys(current_keys))
            current["duration_ns"] = current["end_ns"] - current["start_ns"]
            current["dispatch_key"] = keys[0] if len(keys) == 1 else f"{len(keys)} dispatches"
            current["dispatch_keys"] = keys[:16]
            current["dispatch_key_count"] = len(keys)
            current["batch_req_ids"] = sorted(current_batch_req_ids)
            if current_rpc_wait_ns_values:
                current["max_rpc_to_worker_queue_ns"] = max(current_rpc_wait_ns_values)
            merged.append(current)

        for item in items:
            dispatch_key = item.get("dispatch_key")
            rpc_start_ns = to_ns(item.get("rpc_start_ns"))
            worker_start_ns = to_ns(item.get("worker_start_ns"))
            rpc_wait_ns = None
            if rpc_start_ns is not None and worker_start_ns is not None:
                rpc_wait_ns = max(0, worker_start_ns - rpc_start_ns)

            if current is None or item["start_ns"] > current["end_ns"] + merge_gap_ns:
                _finish_current()
                current = {
                    "start_ns": item["start_ns"],
                    "end_ns": item["end_ns"],
                    "phase": phase,
                    "start_source": "phase_merged_worker_execute_model",
                    "step_count": 0,
                    "worker_span_count": 0,
                }
                current_keys = []
                current_batch_req_ids = set()
                current_rpc_wait_ns_values = []
            else:
                current["end_ns"] = max(current["end_ns"], item["end_ns"])

            current["step_count"] += 1
            current["worker_span_count"] += int(item.get("worker_span_count") or 0)
            if dispatch_key:
                current_keys.append(dispatch_key)
            for req_id in item.get("batch_req_ids") or []:
                if req_id:
                    current_batch_req_ids.add(req_id)
            if rpc_wait_ns is not None:
                current_rpc_wait_ns_values.append(rpc_wait_ns)

        _finish_current()

    return sorted(merged, key=lambda item: (item["start_ns"], item["end_ns"], item["phase"]))


def build_raw_dispatch_display_intervals(intervals):
    """Normalize per-dispatch request intervals for request-lane display."""
    display = []
    for interval in intervals:
        start_ns = to_ns(interval.get("start_ns"))
        end_ns = to_ns(interval.get("end_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        item = dict(interval)
        item["start_ns"] = start_ns
        item["end_ns"] = end_ns
        item["duration_ns"] = end_ns - start_ns
        item["phase"] = item.get("phase") or "unknown"
        item["step_count"] = 1
        dispatch_key = item.get("dispatch_key")
        item["dispatch_keys"] = [dispatch_key] if dispatch_key else []
        item["dispatch_key_count"] = 1 if dispatch_key else 0
        display.append(item)
    return sorted(display, key=lambda item: (item["start_ns"], item["end_ns"], item["phase"]))


def _dispatch_key_matches(dispatch_key, interval):
    interval_key = interval.get("dispatch_key") if isinstance(interval, dict) else None
    return not dispatch_key or not interval_key or interval_key == dispatch_key


def _component_priority(component):
    priority = {
        "pp_gap": 0,
        "model_forward": 10,
        "sampling": 20,
        "postprocess": 30,
        "bookkeeping_sync": 40,
        "worker_preprocess": 50,
        "other_compute": 60,
        "other_exec": 70,
        "other_prefill_exec": 70,
        "pipeline_gap": 100,
    }
    return (priority.get(component, 80), str(component))


def _rank_sort_value(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _attribute_wall_components(component_intervals, window_start_ns, window_end_ns):
    """Assign one dispatch window to non-overlapping wall-time components."""
    window_start = to_ns(window_start_ns)
    window_end = to_ns(window_end_ns)
    if window_start is None or window_end is None or window_end <= window_start:
        return {
            "wall_components_ns": {},
            "wall_components_ms": {},
            "pipeline_gap_ns": 0,
            "pipeline_gap_ms": 0.0,
            "parallel_overlap_ns": 0,
            "parallel_overlap_ms": 0.0,
            "segments": [],
        }

    clipped = []
    boundaries = {window_start, window_end}
    for item in component_intervals:
        span = clip_interval(item.get("start_ns"), item.get("end_ns"), window_start, window_end)
        if not span:
            continue
        component = item.get("component") or "other_compute"
        detail_key = item.get("detail_key")
        clipped.append((span[0], span[1], component, detail_key))
        boundaries.add(span[0])
        boundaries.add(span[1])

    if not clipped:
        duration_ns = window_end - window_start
        wall_components = {"other_compute": duration_ns}
        return {
            "wall_components_ns": wall_components,
            "wall_components_ms": normalize_component_map_ms(wall_components),
            "pipeline_gap_ns": 0,
            "pipeline_gap_ms": ns_to_ms(0),
            "parallel_overlap_ns": 0,
            "parallel_overlap_ms": ns_to_ms(0),
            "segments": [{
                "start_ns": window_start,
                "end_ns": window_end,
                "duration_ns": duration_ns,
                "duration_ms": ns_to_ms(duration_ns),
                "component": "other_compute",
                "active_components": [],
            }],
        }

    wall_components = defaultdict(int)
    parallel_overlap_ns = 0
    segments = []
    sorted_boundaries = sorted(boundaries)

    for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
        if end <= start:
            continue
        active = [
            (component, detail_key)
            for it_start, it_end, component, detail_key in clipped
            if it_start < end and it_end > start
        ]
        dur_ns = end - start
        if not active:
            chosen = "pipeline_gap"
            active_components = []
        else:
            active_components = sorted({component for component, _detail in active}, key=_component_priority)
            chosen = active_components[0]
            parallel_overlap_ns += max(0, len(active) - 1) * dur_ns

        wall_components[chosen] += dur_ns
        segments.append({
            "start_ns": start,
            "end_ns": end,
            "duration_ns": dur_ns,
            "duration_ms": ns_to_ms(dur_ns),
            "component": chosen,
            "active_components": active_components,
        })

    return {
        "wall_components_ns": dict(wall_components),
        "wall_components_ms": normalize_component_map_ms(wall_components),
        "pipeline_gap_ns": wall_components.get("pipeline_gap", 0),
        "pipeline_gap_ms": ns_to_ms(wall_components.get("pipeline_gap", 0)),
        "parallel_overlap_ns": parallel_overlap_ns,
        "parallel_overlap_ms": ns_to_ms(parallel_overlap_ns),
        "segments": segments,
    }


def build_pp_gap_dispatch_breakdown(dispatches, phase_intervals_by_name, step_execution_intervals, phase_component_map=None):
    """Return wall-time execution details with PP gaps from Step Execution windows.

    For TP>1, each PP stage can have one Step Execution per TP rank for the
    same dispatch/batch.  We first merge those TP-local spans into a single
    PP-stage window, then charge only the gap between adjacent PP-stage windows
    as pp_gap.
    """
    phase_component_map = phase_component_map or {}
    rows = []
    compute_by_pp = defaultdict(int)
    raw_compute_by_pp_tp = defaultdict(int)
    raw_compute_by_component = defaultdict(int)
    step_execution_by_pp = defaultdict(int)
    raw_step_execution_by_pp_tp = defaultdict(int)
    pp_gap_by_edge = defaultdict(int)
    dispatch_window_spans = []
    global_component_intervals = []
    total_parallel_overlap_ns = 0

    for dispatch in sorted(dispatches, key=lambda x: (x.get("start_ns") or 0, x.get("end_ns") or 0)):
        dispatch_key = dispatch.get("dispatch_key")
        d_start = to_ns(dispatch.get("start_ns"))
        d_end = to_ns(dispatch.get("end_ns"))
        if d_start is None or d_end is None or d_end <= d_start:
            continue
        dispatch_window_spans.append((d_start, d_end))

        stage_compute = defaultdict(int)
        stage_raw_compute = defaultdict(int)
        stage_raw_compute_by_component = defaultdict(int)
        pp_component_intervals = defaultdict(list)
        component_intervals = []

        for phase_name, intervals in phase_intervals_by_name.items():
            component = phase_component_map.get(phase_name, "other_compute")
            for interval in intervals:
                if not _dispatch_key_matches(dispatch_key, interval):
                    continue
                span = clip_interval(interval.get("start_ns"), interval.get("end_ns"), d_start, d_end)
                if not span:
                    continue
                ranks = interval.get("ranks") if isinstance(interval.get("ranks"), dict) else {}
                pp_rank = ranks.get("pp", "unknown")
                tp_rank = ranks.get("tp", "unknown")
                dur_ns = span[1] - span[0]
                raw_key = f"pp{pp_rank}:tp{tp_rank}:{component}"
                stage_raw_compute[raw_key] += dur_ns
                raw_compute_by_pp_tp[raw_key] += dur_ns
                stage_raw_compute_by_component[component] += dur_ns
                raw_compute_by_component[component] += dur_ns
                pp_component_intervals[(pp_rank, component)].append(span)

        for (pp_rank, component), spans in pp_component_intervals.items():
            wall_key = f"pp{pp_rank}:{component}"
            for start_ns, end_ns in merge_intervals(spans):
                dur_ns = end_ns - start_ns
                stage_compute[wall_key] += dur_ns
                compute_by_pp[wall_key] += dur_ns
                component_intervals.append({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "component": component,
                    "detail_key": wall_key,
                })

        step_spans_by_stage = defaultdict(list)
        step_tp_by_stage = defaultdict(set)
        stage_raw_step = defaultdict(int)
        stage_step_exec = defaultdict(int)
        for interval in step_execution_intervals:
            if not _dispatch_key_matches(dispatch_key, interval):
                continue
            span = clip_interval(interval.get("start_ns"), interval.get("end_ns"), d_start, d_end)
            if not span:
                continue
            ranks = interval.get("ranks") if isinstance(interval.get("ranks"), dict) else {}
            dp_rank = ranks.get("dp", "unknown")
            pp_rank = ranks.get("pp", "unknown")
            tp_rank = ranks.get("tp", "unknown")
            dur_ns = span[1] - span[0]
            stage_key = (dp_rank, pp_rank)
            raw_key = f"dp{dp_rank}:pp{pp_rank}:tp{tp_rank}:step_execution"
            wall_key = f"dp{dp_rank}:pp{pp_rank}:step_execution"
            stage_raw_step[raw_key] += dur_ns
            raw_step_execution_by_pp_tp[raw_key] += dur_ns
            step_spans_by_stage[stage_key].append(span)
            step_tp_by_stage[stage_key].add(tp_rank)
            stage_step_exec[wall_key] += 0

        stage_windows_by_dp = defaultdict(list)
        for (dp_rank, pp_rank), spans in step_spans_by_stage.items():
            merged = merge_intervals(spans)
            if not merged:
                continue
            start_ns = min(start for start, _end in merged)
            end_ns = max(end for _start, end in merged)
            union_ns = sum(end - start for start, end in merged)
            wall_key = f"dp{dp_rank}:pp{pp_rank}:step_execution"
            stage_step_exec[wall_key] += union_ns
            step_execution_by_pp[wall_key] += union_ns
            stage_windows_by_dp[dp_rank].append({
                "dp_rank": dp_rank,
                "pp_rank": pp_rank,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
                "duration_ms": ns_to_ms(end_ns - start_ns),
                "union_duration_ns": union_ns,
                "union_duration_ms": ns_to_ms(union_ns),
                "tp_ranks": sorted(step_tp_by_stage[(dp_rank, pp_rank)], key=_rank_sort_value),
                "span_count": len(spans),
            })

        pp_gap_intervals = []
        row_pp_gap_by_edge = defaultdict(int)
        for dp_rank, windows in stage_windows_by_dp.items():
            windows.sort(key=lambda item: _rank_sort_value(item.get("pp_rank")))
            for prev_stage, next_stage in zip(windows, windows[1:]):
                gap_start = to_ns(prev_stage.get("end_ns"))
                gap_end = to_ns(next_stage.get("start_ns"))
                if gap_start is None or gap_end is None or gap_end <= gap_start:
                    continue
                edge = f"dp{dp_rank}:pp{prev_stage.get('pp_rank')}->pp{next_stage.get('pp_rank')}"
                gap_ns = gap_end - gap_start
                pp_gap_by_edge[edge] += gap_ns
                row_pp_gap_by_edge[edge] += gap_ns
                pp_gap_intervals.append({
                    "start_ns": gap_start,
                    "end_ns": gap_end,
                    "duration_ns": gap_ns,
                    "duration_ms": ns_to_ms(gap_ns),
                    "edge": edge,
                    "dp_rank": dp_rank,
                    "src_pp": prev_stage.get("pp_rank"),
                    "dst_pp": next_stage.get("pp_rank"),
                })
                component_intervals.append({
                    "start_ns": gap_start,
                    "end_ns": gap_end,
                    "component": "pp_gap",
                    "detail_key": edge,
                })

        wall_info = _attribute_wall_components(component_intervals, d_start, d_end)
        row_wall_components = dict(wall_info["wall_components_ns"])
        row_pp_gap_ns = row_wall_components.get("pp_gap", 0)
        global_component_intervals.extend(component_intervals)

        rows.append({
            "dispatch_key": dispatch_key,
            "phase": dispatch.get("phase"),
            "decode_step": dispatch.get("decode_step"),
            "start_ns": d_start,
            "end_ns": d_end,
            "duration_ns": d_end - d_start,
            "duration_ms": ns_to_ms(d_end - d_start),
            "compute_by_pp_phase_ns": dict(stage_compute),
            "compute_by_pp_phase_ms": normalize_component_map_ms(stage_compute),
            "raw_compute_by_pp_tp_phase_ns": dict(stage_raw_compute),
            "raw_compute_by_pp_tp_phase_ms": normalize_component_map_ms(stage_raw_compute),
            "raw_compute_by_component_ns": dict(stage_raw_compute_by_component),
            "raw_compute_by_component_ms": normalize_component_map_ms(stage_raw_compute_by_component),
            "step_execution_by_pp_ns": dict(stage_step_exec),
            "step_execution_by_pp_ms": normalize_component_map_ms(stage_step_exec),
            "raw_step_execution_by_pp_tp_ns": dict(stage_raw_step),
            "raw_step_execution_by_pp_tp_ms": normalize_component_map_ms(stage_raw_step),
            "pp_gap_by_edge_ns": dict(row_pp_gap_by_edge),
            "pp_gap_by_edge_ms": normalize_component_map_ms(row_pp_gap_by_edge),
            "stage_windows": [
                window
                for _dp_rank, windows in sorted(stage_windows_by_dp.items(), key=lambda item: _rank_sort_value(item[0]))
                for window in windows
            ],
            "pp_gap_intervals": pp_gap_intervals,
            "wall_components_ns": row_wall_components,
            "wall_components_ms": normalize_component_map_ms(row_wall_components),
            "pp_gap_ns": row_pp_gap_ns,
            "pp_gap_ms": ns_to_ms(row_pp_gap_ns),
            "raw_pp_gap_ns": sum(gap["duration_ns"] for gap in pp_gap_intervals),
            "raw_pp_gap_ms": ns_to_ms(sum(gap["duration_ns"] for gap in pp_gap_intervals)),
            "unattributed_gap_ns": wall_info["pipeline_gap_ns"],
            "unattributed_gap_ms": wall_info["pipeline_gap_ms"],
            "parallel_overlap_ns": wall_info["parallel_overlap_ns"],
            "parallel_overlap_ms": wall_info["parallel_overlap_ms"],
            "wall_segments": wall_info["segments"],
        })

    wall_components_total = defaultdict(int)
    merged_dispatch_windows = merge_intervals(dispatch_window_spans)
    global_wall_segments = []
    for window_start, window_end in merged_dispatch_windows:
        wall_info = _attribute_wall_components(global_component_intervals, window_start, window_end)
        for key, value in wall_info["wall_components_ns"].items():
            wall_components_total[key] += value
        total_parallel_overlap_ns += wall_info["parallel_overlap_ns"]
        global_wall_segments.extend(wall_info["segments"])

    total_wall_ns = sum_interval_pairs(merged_dispatch_windows)
    total_compute_ns = sum(
        val for key, val in wall_components_total.items()
        if key not in ("pp_gap", "pipeline_gap")
    )
    total_pp_gap_ns = wall_components_total.get("pp_gap", 0)
    total_unattributed_gap_ns = wall_components_total.get("pipeline_gap", 0)

    return {
        "dispatches": rows,
        "compute_by_pp_phase_ns": dict(compute_by_pp),
        "compute_by_pp_phase_ms": normalize_component_map_ms(compute_by_pp),
        "raw_compute_by_pp_tp_phase_ns": dict(raw_compute_by_pp_tp),
        "raw_compute_by_pp_tp_phase_ms": normalize_component_map_ms(raw_compute_by_pp_tp),
        "raw_compute_by_component_ns": dict(raw_compute_by_component),
        "raw_compute_by_component_ms": normalize_component_map_ms(raw_compute_by_component),
        "step_execution_by_pp_ns": dict(step_execution_by_pp),
        "step_execution_by_pp_ms": normalize_component_map_ms(step_execution_by_pp),
        "raw_step_execution_by_pp_tp_ns": dict(raw_step_execution_by_pp_tp),
        "raw_step_execution_by_pp_tp_ms": normalize_component_map_ms(raw_step_execution_by_pp_tp),
        "pp_gap_by_edge_ns": dict(pp_gap_by_edge),
        "pp_gap_by_edge_ms": normalize_component_map_ms(pp_gap_by_edge),
        "wall_components_ns": dict(wall_components_total),
        "wall_components_ms": normalize_component_map_ms(wall_components_total),
        "total_compute_ns": total_compute_ns,
        "total_compute_ms": ns_to_ms(total_compute_ns),
        "total_pp_gap_ns": total_pp_gap_ns,
        "total_pp_gap_ms": ns_to_ms(total_pp_gap_ns),
        "total_unattributed_gap_ns": total_unattributed_gap_ns,
        "total_unattributed_gap_ms": ns_to_ms(total_unattributed_gap_ns),
        "total_wall_ns": total_wall_ns,
        "total_wall_ms": ns_to_ms(total_wall_ns),
        "total_parallel_overlap_ns": total_parallel_overlap_ns,
        "total_parallel_overlap_ms": ns_to_ms(total_parallel_overlap_ns),
        "merged_dispatch_windows": [
            {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
                "duration_ms": ns_to_ms(end_ns - start_ns),
            }
            for start_ns, end_ns in merged_dispatch_windows
        ],
        "wall_segments": global_wall_segments,
        "aggregation_policy": "pp_gap_from_adjacent_pp_step_execution_windows; TP ranks in the same PP stage are merged before computing PP-stage gaps; request-level wall components are attributed over merged dispatch windows so overlapping dispatches/chunked-prefill PP gaps are counted once; raw_step_execution_by_pp_tp and pp_gap_by_edge keep raw per-rank/per-edge sums",
    }


def ns_to_ms(value_ns):
    if value_ns is None:
        return None
    return round(value_ns / 1e6, 6)


def normalize_component_map_ms(component_map_ns):
    return {key: ns_to_ms(val) for key, val in component_map_ns.items()}


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


def derive_sidecar_path(output_file, suffix):
    root, ext = os.path.splitext(output_file)
    if ext == ".gz":
        root, _ = os.path.splitext(root)
    return f"{root}{suffix}"


def _svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_compute_breakdown_svg(svg_path, ttft_components_ms, tpot_components_ms, meta):
    ttft_order = [
        ("preprocess", "#4E79A7", "Preprocess"),
        ("enginecore_queue", "#F28E2B", "EngineCore Queue"),
        ("vllm_queue", "#E15759", "vLLM Queue"),
        ("prefill_exec", "#76B7B2", "Prefill Exec"),
        ("postprocess_transport", "#59A14F", "Postprocess / Transport"),
        ("other_gap", "#BAB0AC", "Other Gap"),
    ]
    tpot_order = [
        ("vllm_scheduling_wait", "#E15759", "vLLM Scheduling Wait"),
        ("worker_preprocess", "#59A14F", "Worker Preprocess"),
        ("model_forward", "#4E79A7", "Model Forward"),
        ("postprocess", "#9C755F", "Postprocess"),
        ("sampling", "#B07AA1", "GPUModelRunner sample_tokens"),
        ("bookkeeping_sync", "#76B7B2", "Bookkeeping / Sync"),
        ("pp_gap", "#F28E2B", "PP Gap"),
        ("other_exec", "#EDC948", "Other Exec"),
        ("tail_transport", "#86BCB6", "Tail Transport"),
        ("other_gap", "#BAB0AC", "Other Gap"),
    ]

    width = 1280
    height = 760
    chart_x = 260
    chart_w = 900
    bar_h = 42
    row_gap = 94
    top = 120
    legend_top = 360
    legend_row_h = 28
    legend_col_gap = 300

    ttft_total = sum(max(0.0, float(ttft_components_ms.get(key, 0.0) or 0.0)) for key, _, _ in ttft_order)
    tpot_total = sum(max(0.0, float(tpot_components_ms.get(key, 0.0) or 0.0)) for key, _, _ in tpot_order)
    max_total = max(ttft_total, tpot_total, 1.0)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAF7F2"/>',
        '<text x="60" y="62" font-size="30" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#1F2937">Compute-Side Breakdown</text>',
        f'<text x="60" y="92" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#4B5563">TTFT and TPOT are split from the compute path only. TPOT uses decode dispatch count as the denominator ({_svg_escape(meta.get("tpot_denominator_label", "decode dispatch count"))}).</text>',
    ]

    def _draw_row(y, title, total_ms, components, order, subtitle):
        lines = [
            f'<text x="60" y="{y + 14}" font-size="22" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">{_svg_escape(title)}</text>',
            f'<text x="60" y="{y + 42}" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#6B7280">{_svg_escape(subtitle)}</text>',
            f'<rect x="{chart_x}" y="{y}" width="{chart_w}" height="{bar_h}" rx="10" fill="#E5E7EB"/>',
        ]
        cursor_x = chart_x
        for key, color, _label in order:
            value = max(0.0, float(components.get(key, 0.0) or 0.0))
            if total_ms <= 0 or value <= 0:
                continue
            seg_w = chart_w * (value / max_total)
            seg_w = max(seg_w, 1.0)
            lines.append(
                f'<rect x="{cursor_x:.2f}" y="{y}" width="{seg_w:.2f}" height="{bar_h}" rx="10" fill="{color}"/>'
            )
            cursor_x += seg_w
        lines.append(
            f'<text x="{chart_x + chart_w + 18}" y="{y + 28}" font-size="18" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">{total_ms:.3f} ms</text>'
        )
        return lines

    svg_lines.extend(_draw_row(
        top,
        "TTFT Avg (compute-side)",
        ttft_total,
        ttft_components_ms,
        ttft_order,
        meta.get("ttft_subtitle", "Average across requests with a closed prefill window."),
    ))
    svg_lines.extend(_draw_row(
        top + row_gap,
        "TPOT Avg (compute-side, ms/token)",
        tpot_total,
        tpot_components_ms,
        tpot_order,
        meta.get("tpot_subtitle", "Token-weighted average using decode dispatch count."),
    ))

    legend_items = []
    for key, color, label in ttft_order + [item for item in tpot_order if item[0] not in {key for key, _, _ in ttft_order}]:
        legend_items.append((color, label))

    legend_x = 60
    legend_y = legend_top
    for idx, (color, label) in enumerate(legend_items):
        col = idx // 6
        row = idx % 6
        x = legend_x + col * legend_col_gap
        y = legend_y + row * legend_row_h
        svg_lines.append(f'<rect x="{x}" y="{y - 12}" width="16" height="16" rx="3" fill="{color}"/>')
        svg_lines.append(
            f'<text x="{x + 26}" y="{y + 1}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{_svg_escape(label)}</text>'
        )

    footer_y = height - 52
    svg_lines.append(
        f'<text x="60" y="{footer_y}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#6B7280">Requests used: TTFT={meta.get("ttft_req_count", 0)}, TPOT={meta.get("tpot_req_count", 0)}; total decode steps={meta.get("tpot_total_decode_steps", 0)}.</text>'
    )
    svg_lines.append("</svg>")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def sanitize_filename(name):
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip("._")
    return result or "request"


def _component_rows_svg(x, y, width, rows, total_ms):
    row_h = 26
    lines = []
    for idx, row in enumerate(rows):
        label = row["label"]
        value_ms = float(row["value_ms"])
        color = row["color"]
        pct = (value_ms / total_ms * 100.0) if total_ms > 0 else 0.0
        ry = y + idx * row_h
        lines.append(f'<rect x="{x}" y="{ry + 6}" width="{width}" height="10" rx="5" fill="#E5E7EB"/>')
        if value_ms > 0 and total_ms > 0:
            bar_w = max(10.0, width * (value_ms / total_ms))
            bar_w = min(bar_w, width)
            lines.append(f'<rect x="{x}" y="{ry + 6}" width="{bar_w:.2f}" height="10" rx="5" fill="{color}"/>')
        lines.append(f'<rect x="{x - 18}" y="{ry + 4}" width="10" height="10" rx="2" fill="{color}"/>')
        lines.append(
            f'<text x="{x + width + 12}" y="{ry + 15}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#111827">{_svg_escape(label)}: {value_ms:.3f} ms ({pct:.1f}%)</text>'
        )
    return lines, y + len(rows) * row_h


def render_request_breakdown_svg(svg_path, rid, request_name, ttft_entry, tpot_entry):
    ttft_rows = []
    if ttft_entry:
        ttft_colors = {
            "preprocess": "#4E79A7",
            "enginecore_queue": "#F28E2B",
            "vllm_queue": "#E15759",
            "prefill_exec": "#76B7B2",
            "postprocess_transport": "#59A14F",
            "other_gap": "#BAB0AC",
        }
        ttft_labels = {
            "preprocess": "Preprocess",
            "enginecore_queue": "EngineCore Queue",
            "vllm_queue": "vLLM Queue",
            "prefill_exec": "Prefill Exec",
            "postprocess_transport": "Postprocess / Transport",
            "other_gap": "Other Gap",
        }
        for key in ["preprocess", "enginecore_queue", "vllm_queue", "prefill_exec", "postprocess_transport", "other_gap"]:
            ttft_rows.append({
                "label": ttft_labels[key],
                "value_ms": ttft_entry["components_ms"].get(key, 0.0),
                "color": ttft_colors[key],
            })

    tpot_rows = []
    if tpot_entry:
        tpot_colors = {
            "vllm_scheduling_wait": "#E15759",
            "worker_preprocess": "#59A14F",
            "model_forward": "#4E79A7",
            "postprocess": "#9C755F",
            "sampling": "#B07AA1",
            "bookkeeping_sync": "#76B7B2",
            "pp_gap": "#F28E2B",
            "other_exec": "#EDC948",
            "tail_transport": "#86BCB6",
            "other_gap": "#BAB0AC",
        }
        tpot_labels = {
            "vllm_scheduling_wait": "vLLM Scheduling Wait",
            "worker_preprocess": "Worker Preprocess",
            "model_forward": "Model Forward",
            "postprocess": "Postprocess",
            "sampling": "GPUModelRunner sample_tokens",
            "bookkeeping_sync": "Bookkeeping / Sync",
            "pp_gap": "PP Gap",
            "other_exec": "Other Exec",
            "tail_transport": "Tail Transport",
            "other_gap": "Other Gap",
        }
        for key in [
            "vllm_scheduling_wait",
            "worker_preprocess",
            "model_forward",
            "postprocess",
            "sampling",
            "bookkeeping_sync",
            "pp_gap",
            "other_exec",
            "tail_transport",
            "other_gap",
        ]:
            tpot_rows.append({
                "label": tpot_labels[key],
                "value_ms": tpot_entry["components_ms_per_token"].get(key, 0.0),
                "color": tpot_colors[key],
            })

    width = 1500
    base_height = 210
    ttft_extra = len(ttft_rows) * 26 + 112 if ttft_rows else 0
    tpot_extra = len(tpot_rows) * 26 + 100 if tpot_rows else 0
    height = base_height + ttft_extra + tpot_extra
    chart_x = 110
    chart_w = 520
    cursor_y = 90

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAF7F2"/>',
        f'<text x="48" y="50" font-size="28" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">{_svg_escape(request_name or rid)}</text>',
        f'<text x="48" y="76" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#4B5563">request_id={_svg_escape(rid)}</text>',
    ]

    if ttft_entry:
        lines.append(f'<text x="48" y="{cursor_y}" font-size="22" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">TTFT</text>')
        lines.append(
            f'<text x="48" y="{cursor_y + 24}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#4B5563">total={ttft_entry["window_ms"]:.3f} ms, start_rule={_svg_escape(ttft_entry.get("ttft_start_rule"))}, end_rule={_svg_escape(ttft_entry.get("ttft_end_rule"))}</text>'
        )
        lines.append(
            f'<text x="48" y="{cursor_y + 46}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#4B5563">prefill_dispatch_count={ttft_entry.get("prefill_dispatch_count", 0)}, phase_reentry_after_decode={_svg_escape(ttft_entry.get("phase_reentry_after_decode", False))}</text>'
        )
        row_lines, end_y = _component_rows_svg(chart_x, cursor_y + 62, chart_w, ttft_rows, max(ttft_entry["window_ms"], 1e-9))
        lines.extend(row_lines)
        breakdown = ttft_entry.get("vllm_queue_breakdown_ms", {})
        lines.append(
            f'<text x="{chart_x}" y="{end_y + 18}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">vLLM queue breakdown: before_first_prefill={float(breakdown.get("before_first_prefill", 0.0)):.3f} ms, between_prefills={float(breakdown.get("between_prefills", 0.0)):.3f} ms</text>'
        )
        prefill_phase = ttft_entry.get("prefill_exec_phase_ms", {})
        lines.append(
            f'<text x="{chart_x}" y="{end_y + 40}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">prefill exec wall breakdown: forward={float(prefill_phase.get("model_forward", 0.0)):.3f} ms, pp_gap={float(prefill_phase.get("pp_gap", 0.0)):.3f} ms, other_exec={float(prefill_phase.get("other_prefill_exec", 0.0)):.3f} ms</text>'
        )
        cursor_y = end_y + 78

    if tpot_entry:
        lines.append(f'<text x="48" y="{cursor_y}" font-size="22" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">TPOT</text>')
        lines.append(
            f'<text x="48" y="{cursor_y + 24}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#4B5563">avg={tpot_entry["avg_ms_per_token"]:.3f} ms/token, decode_dispatch_count={tpot_entry.get("decode_dispatch_count", 0)}, start_rule={_svg_escape(tpot_entry.get("tpot_start_rule"))}, end_rule={_svg_escape(tpot_entry.get("tpot_end_rule"))}</text>'
        )
        lines.append(
            f'<text x="48" y="{cursor_y + 46}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#4B5563">scheduling_wait_total={float(tpot_entry.get("scheduling_wait_total_ms", 0.0)):.3f} ms, exec_total={float(tpot_entry.get("exec_total_ms", 0.0)):.3f} ms, tail_transport={float(tpot_entry.get("tail_transport_ms", 0.0)):.3f} ms</text>'
        )
        row_lines, end_y = _component_rows_svg(chart_x, cursor_y + 62, chart_w, tpot_rows, max(tpot_entry["avg_ms_per_token"], 1e-9))
        lines.extend(row_lines)
        sched = tpot_entry.get("scheduling_wait_breakdown_ms", {})
        sched_ratio = tpot_entry.get("scheduling_wait_breakdown_ratio", {})
        lines.append(
            f'<text x="{chart_x}" y="{end_y + 18}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">scheduling wait breakdown: normal_gap={float(sched.get("normal_gap", 0.0)):.3f} ms ({float(sched_ratio.get("normal_gap_pct", 0.0)):.1f}%), preempt_gap={float(sched.get("preempt_gap", 0.0)):.3f} ms ({float(sched_ratio.get("preempt_gap_pct", 0.0)):.1f}%)</text>'
        )
        cursor_y = end_y + 40

    if not ttft_entry and not tpot_entry:
        lines.append('<text x="48" y="118" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#6B7280">No compute-side TTFT/TPOT breakdown available for this request.</text>')

    lines.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_request_breakdown_index(index_path, request_rows):
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Per-Request Compute Breakdown</title>",
        "<style>body{font-family:Helvetica,Arial,sans-serif;background:#faf7f2;color:#111827;margin:24px;}table{border-collapse:collapse;width:100%;background:#fff;}th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:left;}th{background:#f3f4f6;}a{color:#1d4ed8;text-decoration:none;}a:hover{text-decoration:underline;}</style>",
        "</head><body>",
        "<h1>Per-Request Compute Breakdown</h1>",
        "<table>",
        "<thead><tr><th>Request</th><th>TTFT (ms)</th><th>TPOT (ms/token)</th><th>Prefill Re-entry</th><th>Chart</th></tr></thead><tbody>",
    ]
    for row in request_rows:
        lines.append(
            "<tr>"
            f"<td>{_svg_escape(row['request_name'])}</td>"
            f"<td>{row['ttft_ms']}</td>"
            f"<td>{row['tpot_ms']}</td>"
            f"<td>{_svg_escape(row['reentry'])}</td>"
            f"<td><a href='{_svg_escape(row['file_name'])}'>open</a></td>"
            "</tr>"
        )
    lines.extend(["</tbody></table>", "</body></html>"])
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

#DEBUG: 调试OS调度层的时间对齐问题（将ebpf和execute_model_span的区间在perfetto上画出来）
def export_alignment_debug_trace(worker_spans, ebpf_events, output_filename="debug_align.json"):
    """
    生成一个只包含 Worker 执行区间(Green) 和 eBPF 调度事件(Red) 的 Trace。
    用于肉眼 Debug 时间对齐问题。
    """
    print(f"\n🔍 Generating Debug Trace: {output_filename} ...")
    
    trace_events = []
    
    # 1. 添加 Worker Spans (绿色条)
    for item in worker_spans:
        # 兼容处理：有些 payload 在 item 里，有些 item 本身就是 payload
        p = item.get('payload', item)
        
        pid = p.get('pid', 0)
        tid = p.get('tid', 0)
        start = p.get('start_ns')
        end = p.get('end_ns')
        
        if start and end and tid:
            trace_events.append({
                "name": "Worker Span (Python)",
                "cat": "debug_worker",
                "ph": "X",
                "ts": start / 1000.0,       # ns -> us
                "dur": (end - start) / 1000.0,
                "pid": pid,
                "tid": tid,
                "cname": "good",            # 绿色
                "args": {"req_ids": p.get("req_ids")}
            })

    # 2. 添加 eBPF Events (红色条)
    # 为了防止 eBPF 事件太短看不清，或者为了方便对比，
    # 我们将 eBPF 事件放在同一个 PID/TID 轨道上，或者放在一个专门的 Debug 轨道
    for item in ebpf_events:
        p = item.get('payload', item)
        
        tid = p.get('tid', 0)
        start = p.get('start_ns')
        end = p.get('end_ns')
        dur = p.get('dur_us', 0)
        
        # eBPF 数据通常没有 PID，为了让它能在 Perfetto 里跟 Worker 显示在一起，
        # 我们尝试"借用" Worker 的 PID。这里我们做一个特殊的处理：
        # 如果我们不知道 PID，就设为 TID (Perfetto 会把它们分到不同组，但至少能看见)
        # 或者设为 0。为了对比，最好让它们在同一轨道。
        
        if start and end and tid:
            trace_events.append({
                "name": "OS Latency (eBPF)",
                "cat": "debug_ebpf",
                "ph": "X",
                "ts": start / 1000.0,       # ns -> us
                "dur": (end - start) / 1000.0,
                "pid": 0,                   # 故意设为0 或者一个特定值，方便在最顶层看到
                # "pid": p.get('pid', tid), # 如果你想让它和 Worker 挤在一起，解开这行
                "tid": tid,                 # 关键：TID 必须一致
                "cname": "terrible",        # 红色
                "args": {"reason": p.get("reason")}
            })

    # 3. 输出
    # 按时间排序
    trace_events.sort(key=lambda x: x['ts'])
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({"traceEvents": trace_events}, f)
    
    print(f"✅ Debug Trace Saved: {output_filename}")

def process_logs(input_file, output_file):
    print(f"🔄 Loading logs from {input_file}...")
    raw_events = load_log_lines(input_file)
    trace_events = []

    # --- 1. 数据清洗与分类 ---
    # 我们将所有事件分为三类：Worker生命周期、Scheduler、CUPTI
    worker_events = []
    scheduler_events = []
    cupti_events = []
    req_metric_events = []
    coroutine_events = []
    coroutine_exec_events = []
    output_socket_sched_events = []
    output_socket_exec_events = []
    output_handler_sched_events = []
    output_handler_exec_events = []
    req_stage_events = []
    enginecore_loop_events = []
    pp_comm_events = []
    thread_role_events = []
    execute_model_span = []
    worker_span_mono_to_wall_offsets = []
    ebpf_sched_latency_events = []
    req_name_map = {}
    req_id_aliases = {}
    
    # 辅助字典：存储 Kernel 信息
    all_kernels_map = {}
    eager_kernels = []
    zero_kernel_records = []
    runtime_name_by_corr = {}
    cupti_nonzero_max_end_by_pid = defaultdict(int)

    def cupti_corr_key(payload, corr_id=None, pid_override=None):
        if corr_id is None:
            corr_id = payload.get("correlationId")
        pid = pid_override if pid_override is not None else payload.get("pid")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = 0
        return (pid, corr_id)

    flow_id_by_corr_key = {}

    def cupti_flow_debug_key(payload, corr_id=None, pid_override=None):
        pid, corr_id = cupti_corr_key(
            payload,
            corr_id=corr_id,
            pid_override=pid_override,
        )
        return f"{pid}:{corr_id}"

    def cupti_flow_id(payload, corr_id=None, pid_override=None):
        key = cupti_corr_key(
            payload,
            corr_id=corr_id,
            pid_override=pid_override,
        )
        flow_id = flow_id_by_corr_key.get(key)
        if flow_id is None:
            flow_id = len(flow_id_by_corr_key) + 1
            flow_id_by_corr_key[key] = flow_id
        return flow_id

    for entry in raw_events:
        # 处理可能的 TraceSender 格式差异
        if 'meta' in entry:
            # 如果是 ZeroMQ 发送的原始格式
            meta = entry.get('meta', {})
            payload = meta.get('payload') or entry.get('payload', {})
            src = meta.get('source')
            etype = meta.get('event_type')
            ts = meta.get('timestamp_ns')
        else:
            # 如果是打平后的格式
            payload = entry.get('payload', entry)
            src = entry.get('source', 'monkey_patch')
            etype = entry.get('event_type')
            ts = entry.get('timestamp_ns', payload.get('timestamp_ns'))

        if not payload:
            continue
        payload = normalize_payload_request_fields(payload)
        if not isinstance(payload, dict):
            continue
        rid = payload.get("request_id") or payload.get("req_id")
        rname = payload.get("request_name")
        external_rid = payload.get("external_request_id")
        if rid and external_rid and external_rid != rid:
            req_id_aliases[rid] = external_rid
            req_name_map[external_rid] = external_rid
        elif rid and rname and rname != rid:
            req_id_aliases[rid] = rname
            req_name_map.setdefault(rname, rname)
        if rid and rname:
            # Client-side events can reveal the stable external request id before
            # EngineCore later logs the internal suffixed id as request_name. Keep
            # the external name once discovered so lifecycle and worker phases join.
            if rid not in req_name_map or req_name_map[rid] == rid or rname != rid:
                req_name_map[rid] = rname
        # 兜底：确保 payload 里有 timestamp_ns
        if 'timestamp_ns' not in payload and ts:
            payload['timestamp_ns'] = ts
        
        # 修正：你的代码中 gpu_forward 的 payload 包含 start_ns/end_ns，但 event_type 叫 'gpu_forward'
        # 我们统一整理到 worker_events
        if src == 'monkey_patch':
            # 新增事件类型支持
            if etype in ['worker_preprocess_start', 'gpu_forward_start', 'gpu_forward_end', 
                         'gpu_sample', 'gpu_bookkeeping', 'gpu_execute_model',
                         'gpu_phase_span']:
                worker_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('start_ns', payload.get('timestamp_ns', ts))
                })
            elif etype in ['req_enqueue_scheduler', 'req_scheduler_out_rpc', 'req_step_ready']:
                scheduler_events.append({"type": etype, "payload": payload})
            elif etype=="req_metrics_events":
                req_metric_events.append({
                    "payload": payload,
                    "batch_ts_ns": ts
                })
            elif etype in ['coroutine_start', 'coroutine_end', 'coroutine_sched_latency']:
                coroutine_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', ts)
                })
            elif etype == "coroutine_exec_slice":
                coroutine_exec_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "output_socket_sched_latency":
                output_socket_sched_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('run_ts_ns', ts)),
                })
            elif etype == "output_socket_exec_slice":
                output_socket_exec_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "output_handler_sched_latency":
                output_handler_sched_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('run_ts_ns', ts)),
                })
            elif etype == "output_handler_exec_slice":
                output_handler_exec_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "req_generate_stage":
                req_stage_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', ts),
                })
            elif etype == "enginecore_mainloop_span":
                enginecore_loop_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "pp_comm_span":
                pp_comm_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get('timestamp_ns', payload.get('end_ns', ts)),
                })
            elif etype == "thread_role":
                thread_role_events.append({
                    "type": etype,
                    "payload": payload,
                    "ts": payload.get("timestamp_ns", ts),
                })
            elif etype=="worker_model_execute_span":
                execute_model_span.append({
                    "type": etype,
                    "payload": payload,
                    "ts": ts,
                })
                end_mono_ns = to_ns(payload.get("end_ns"))
                emit_wall_ns = to_ns(ts)
                if (
                    end_mono_ns is not None
                    and emit_wall_ns is not None
                    and emit_wall_ns > end_mono_ns
                ):
                    # 该事件在记录完 monotonic end_ns 后立刻用 wall clock 发出，
                    # 可作为稳定的 mono->wall 对齐样本。
                    worker_span_mono_to_wall_offsets.append(emit_wall_ns - end_mono_ns)

        elif src == 'CUPTI':
            corr_id = payload.get('correlationId')
            pid = to_int(payload.get("pid"))
            start_ns = to_ns(payload.get("start_ns"))
            end_ns = to_ns(payload.get("end_ns"))
            if (
                pid is not None
                and start_ns is not None
                and end_ns is not None
                and start_ns > 0
                and end_ns > 0
                and end_ns > cupti_nonzero_max_end_by_pid[pid]
            ):
                cupti_nonzero_max_end_by_pid[pid] = end_ns
            if etype in ['runtime', 'driver', 'marker']:
                cupti_events.append(payload)
                if etype == 'runtime':
                    runtime_name_by_corr[
                        cupti_corr_key(payload, corr_id=corr_id, pid_override=pid)
                    ] = payload.get("name", "")
            elif etype in ['kernel', 'memset', 'memcpy']:
                kernel_key = cupti_corr_key(payload, corr_id=corr_id)
                if kernel_key not in all_kernels_map:
                    all_kernels_map[kernel_key] = []
                all_kernels_map[kernel_key].append(payload)
                if start_ns is not None and start_ns > 0:
                    eager_kernels.append(payload)
                elif etype == 'kernel':
                    zero_kernel_records.append(payload)

        elif src == 'ebpf':
            if etype == 'sched_latency':
                ebpf_sched_latency_events.append(payload)

    def _resolve_request_alias(rid):
        rid = normalize_request_id(rid)
        if not rid:
            return rid
        seen = set()
        while rid in req_id_aliases and rid not in seen:
            seen.add(rid)
            nxt = normalize_request_id(req_id_aliases.get(rid))
            if not nxt:
                break
            rid = nxt
        return rid

    def _canonicalize_event_payloads(events):
        for ev in events:
            payload = ev.get("payload") if isinstance(ev, dict) else None
            canonicalize_payload_request_fields(payload, _resolve_request_alias)

    for _events in (
        worker_events,
        scheduler_events,
        req_metric_events,
        coroutine_events,
        coroutine_exec_events,
        output_socket_sched_events,
        output_socket_exec_events,
        output_handler_sched_events,
        output_handler_exec_events,
        req_stage_events,
        enginecore_loop_events,
        pp_comm_events,
        thread_role_events,
        execute_model_span,
    ):
        _canonicalize_event_payloads(_events)

    canonical_req_name_map = {}
    for rid, name in req_name_map.items():
        canonical_rid = _resolve_request_alias(rid)
        canonical_name = _resolve_request_alias(name)
        if canonical_rid:
            canonical_req_name_map[canonical_rid] = canonical_name or canonical_rid
    req_name_map = canonical_req_name_map
    if req_id_aliases:
        print(f"Canonicalized {len(req_id_aliases)} internal request ids to external request ids.")

    if zero_kernel_records:
        zero_kernel_count_by_pid = defaultdict(int)
        zero_kernel_graph_count_by_pid = defaultdict(int)
        for payload in zero_kernel_records:
            pid = to_int(payload.get("pid"))
            if pid is None:
                continue
            zero_kernel_count_by_pid[pid] += 1
            runtime_name = runtime_name_by_corr.get(
                cupti_corr_key(payload, pid_override=pid),
                "",
            )
            if "graph" in str(runtime_name).lower():
                zero_kernel_graph_count_by_pid[pid] += 1

        zero_total = sum(zero_kernel_count_by_pid.values())
        zero_graph_total = sum(zero_kernel_graph_count_by_pid.values())
        graph_ratio = (zero_graph_total / zero_total) if zero_total else 0.0
        print(
            "  [warn] Detected "
            f"{zero_total} kernel activity records with start_ns/end_ns <= 0; "
            f"{zero_graph_total} ({graph_ratio:.1%}) are linked to CUDA Graph launches."
        )
        for pid in sorted(
            zero_kernel_count_by_pid,
            key=lambda x: zero_kernel_count_by_pid[x],
            reverse=True,
        ):
            zero_cnt = zero_kernel_count_by_pid[pid]
            graph_cnt = zero_kernel_graph_count_by_pid.get(pid, 0)
            ratio = (graph_cnt / zero_cnt) if zero_cnt else 0.0
            print(
                f"    pid={pid}: zero_ts_kernels={zero_cnt}, "
                f"graph_linked={graph_cnt} ({ratio:.1%})"
            )
        print(
            "    These records cannot be placed on the wall-clock timeline. "
            "Blank later Step Execution spans usually mean CUDA Graph replay, not a simple global offset."
        )

    global TRACE_TS_ORIGIN_NS
    TRACE_TS_ORIGIN_NS = infer_trace_origin_ns(
        worker_events=worker_events,
        scheduler_events=scheduler_events,
        coroutine_events=coroutine_events,
        coroutine_exec_events=coroutine_exec_events,
        output_socket_sched_events=output_socket_sched_events,
        output_socket_exec_events=output_socket_exec_events,
        output_handler_sched_events=output_handler_sched_events,
        output_handler_exec_events=output_handler_exec_events,
        req_stage_events=req_stage_events,
        enginecore_loop_events=enginecore_loop_events,
        pp_comm_events=pp_comm_events,
        execute_model_span=execute_model_span,
        cupti_events=cupti_events,
    )
    if TRACE_TS_ORIGIN_NS:
        print(f"Trace time origin (wall ns): {TRACE_TS_ORIGIN_NS}")

    thread_roles_by_tid = {}
    thread_mono_to_wall_offsets = defaultdict(list)
    eventloop_thread_tids = set()
    for item in thread_role_events:
        payload = item.get("payload", {})
        tid = to_int(payload.get("tid"))
        pid = to_int(payload.get("pid"))
        role = payload.get("role")
        wall_ts_ns = to_ns(payload.get("timestamp_ns", item.get("ts")))
        mono_ts_ns = to_ns(payload.get("mono_timestamp_ns"))
        if tid is None or not role:
            continue

        info = thread_roles_by_tid.setdefault(
            tid,
            {
                "pid": pid,
                "roles": set(),
            },
        )
        if info.get("pid") is None and pid is not None:
            info["pid"] = pid
        info["roles"].add(role)
        if role == "asyncio_eventloop":
            eventloop_thread_tids.add(tid)
        if wall_ts_ns is not None and mono_ts_ns is not None and wall_ts_ns > mono_ts_ns:
            thread_mono_to_wall_offsets[tid].append(wall_ts_ns - mono_ts_ns)

    thread_mono_to_wall_offset_by_tid = {
        tid: median_int(offsets)
        for tid, offsets in thread_mono_to_wall_offsets.items()
        if offsets
    }

    # --- 2. Worker 状态机 (支持细粒度阶段) ---
    print(f"Processing Worker Events ({len(worker_events)})...")
    
    worker_events.sort(key=lambda x: x['ts'])
    has_precise_phase_spans = any(ev.get('type') == 'gpu_phase_span' for ev in worker_events)
    req_worker_phase_intervals = defaultdict(lambda: defaultdict(list))
    req_step_execution_intervals = defaultdict(list)
    req_pp_comm_intervals = defaultdict(list)
    pp_comm_attr_stats = defaultdict(int)

    # 状态存储: Key=(pid, tid), Value={
    #   't_pre_start': ..., 
    #   't_fwd_start': ..., 
    #   't_fwd_end': ..., 
    #   'req_ids': [...],
    #   'batch_size': ..., 'input_type': ...
    # }
    worker_states = {}
    generated_dispatch_slices = []  # 仍可用于 CUPTI 关联（用整个 execute_model 区间）

    for ev in worker_events:
        etype = ev['type']
        payload = ev['payload']
        ts = ev['ts']
        pid = payload.get('pid')
        tid = payload.get('tid')
        key = (pid, tid)

        # ----------------------------------------------------------------
        # [Step 0] 整体 execute_model 区间（容器）
        # ----------------------------------------------------------------
        if etype == 'gpu_execute_model':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                req_ids = payload.get('req_ids', [])
                batch_size = payload.get('batch_size', 0)
                input_type = payload.get('input_type', 'unknown')
                dispatch_key = payload.get("dispatch_key")
                ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
                
                # 记录完整区间（用于 CUPTI 关联）
                generated_dispatch_slices.append({
                    'pid': pid, 'tid': tid,
                    'start': t_start, 'end': t_end,
                    'req_ids': req_ids,
                    'dispatch_key': dispatch_key,
                    'ranks': ranks,
                })
                for rid in normalize_request_ids(req_ids):
                    req_step_execution_intervals[rid].append({
                        "pid": pid,
                        "tid": tid,
                        "start_ns": t_start,
                        "end_ns": t_end,
                        "dispatch_key": dispatch_key,
                        "ranks": ranks,
                        "batch_size": batch_size,
                        "input_type": input_type,
                    })
                
                # 画一个透明容器（浅灰色背景）
                class_path = payload.get("class_path", "") or ""
                if "gpu.model_runner" in class_path:
                    event_name = "GPUModelRunner (V2).execute_model"
                elif "gpu_model_runner" in class_path:
                    event_name = "GPUModelRunner (V1).execute_model"
                else:
                    event_name = "Step Execution"
                trace_events.append(create_perfetto_event(
                    name=event_name, cat="worker", ph="X",
                    ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args={
                        "req_ids": "\n".join(req_ids),
                        "batch_size": batch_size,
                        "class_path": class_path,
                        "dispatch_key": dispatch_key,
                        "dp_rank": ranks.get("dp"),
                        "pp_rank": ranks.get("pp"),
                        "tp_rank": ranks.get("tp"),
                    },
                    cname="background"
                ))
            continue  # 不进入状态机流转

        if etype == 'gpu_phase_span':
            phase_name = payload.get('phase')
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if phase_name and t_start is not None and t_end is not None:
                req_ids = payload.get('req_ids', [])
                batch_size = payload.get('batch_size')
                input_type = payload.get('input_type')
                dispatch_key = payload.get("dispatch_key")
                ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
                for rid in normalize_request_ids(req_ids):
                    req_worker_phase_intervals[rid][phase_name].append({
                        "start_ns": t_start,
                        "end_ns": t_end,
                        "dispatch_key": dispatch_key,
                        "ranks": ranks,
                    })
                display_name, cat, cname = describe_gpu_phase(phase_name)
                args = {
                    "phase": phase_name,
                    "req_ids": "\n".join(req_ids),
                    "dispatch_key": dispatch_key,
                    "dp_rank": ranks.get("dp"),
                    "pp_rank": ranks.get("pp"),
                    "tp_rank": ranks.get("tp"),
                }
                if batch_size is not None:
                    args["batch_size"] = batch_size
                if input_type:
                    args["input_type"] = input_type
                trace_events.append(create_perfetto_event(
                    name=display_name,
                    cat=cat, ph="X", ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args=args,
                    cname=cname
                ))
                state = worker_states.setdefault(key, {})
                if req_ids:
                    state['req_ids'] = req_ids
                if batch_size is not None:
                    state['batch_size'] = batch_size
                if input_type:
                    state['input_type'] = input_type
            continue

        # ----------------------------------------------------------------
        # [Step 1] Preprocess Start
        # ----------------------------------------------------------------
        if not has_precise_phase_spans and etype == 'worker_preprocess_start':
            worker_states[key] = {
                't_pre_start': ts,
                'req_ids': payload.get('req_ids', []),
                'batch_size': payload.get('batch_size'),
                'input_type': payload.get('input_type', 'unknown')
            }

        # ----------------------------------------------------------------
        # [Step 2] Forward Start
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_forward_start':
            state = worker_states.get(key)
            if state and 't_pre_start' in state:
                t1 = state['t_pre_start']
                dur = max(0, ts - t1 - 1000)  # 留白
                for rid in normalize_request_ids(state.get('req_ids', [])):
                    req_worker_phase_intervals[rid]["Preprocess"].append({
                        "start_ns": t1,
                        "end_ns": t1 + dur,
                    })
                trace_events.append(create_perfetto_event(
                    name="Worker Preprocess",
                    cat="python", ph="X", ts=t1, dur=dur,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(state['req_ids']), "desc": "Prepare Inputs"},
                    cname="good"  # 黄绿色
                ))
                state['t_fwd_start'] = ts

        # ----------------------------------------------------------------
        # [Step 3] Forward End
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_forward_end':
            state = worker_states.get(key)
            if state and 't_fwd_start' in state:
                t2 = state['t_fwd_start']
                dur = max(0, ts - t2)
                for rid in normalize_request_ids(state.get('req_ids', [])):
                    req_worker_phase_intervals[rid]["Forward"].append({
                        "start_ns": t2,
                        "end_ns": t2 + dur,
                    })
                trace_events.append(create_perfetto_event(
                    name="Model Forward",
                    cat="python", ph="X", ts=t2, dur=dur,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(state['req_ids']), "batch_size": state.get('batch_size', 0)},
                    cname="olive"  # 橄榄绿
                ))
                state['t_fwd_end'] = ts

        # ----------------------------------------------------------------
        # [Step 4] Sample (区间事件)
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_sample':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                state = worker_states.get(key)
                req_ids = state['req_ids'] if state else payload.get('req_ids', [])
                for rid in normalize_request_ids(req_ids):
                    req_worker_phase_intervals[rid]["Sample"].append({
                        "start_ns": t_start,
                        "end_ns": t_end,
                    })
                trace_events.append(create_perfetto_event(
                    name="Sampling",
                    cat="python", ph="X", ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(req_ids)},
                    cname="mauve"  # 紫红色
                ))

        # ----------------------------------------------------------------
        # [Step 5] Bookkeeping / Sync (区间事件)
        # ----------------------------------------------------------------
        elif not has_precise_phase_spans and etype == 'gpu_bookkeeping':
            t_start = payload.get('start_ns')
            t_end = payload.get('end_ns')
            if t_start is not None and t_end is not None:
                state = worker_states.get(key)
                req_ids = state['req_ids'] if state else payload.get('req_ids', [])
                for rid in normalize_request_ids(req_ids):
                    req_worker_phase_intervals[rid]["Bookkeep"].append({
                        "start_ns": t_start,
                        "end_ns": t_end,
                    })
                trace_events.append(create_perfetto_event(
                    name="Bookkeeping/Sync",
                    cat="python", ph="X", ts=t_start, dur=t_end - t_start,
                    pid=pid, tid=tid,
                    args={"req_ids": "\n".join(req_ids)},
                    cname="cyan"  # 青色
                ))

    dispatch_slices_by_pid = defaultdict(list)
    for span in generated_dispatch_slices:
        span_start = to_ns(span.get("start"))
        span_end = to_ns(span.get("end"))
        if span_start is None or span_end is None or span_end <= span_start:
            continue
        dispatch_slices_by_pid[span.get("pid")].append(span)
    for spans in dispatch_slices_by_pid.values():
        spans.sort(key=lambda x: (to_ns(x.get("start")) or 0, to_ns(x.get("end")) or 0))

    def _rank_value_known(value):
        if value is None:
            return False
        if isinstance(value, str) and value.strip() in ("", "unknown", "-1"):
            return False
        return value != -1

    def _rank_compatible(left, right):
        if not _rank_value_known(left) or not _rank_value_known(right):
            return True
        return str(left) == str(right)

    def _infer_pp_comm_request_context(start_ns, end_ns, pid, ranks, dispatch_key):
        candidates = dispatch_slices_by_pid.get(pid, [])
        if not candidates:
            candidates = generated_dispatch_slices

        best_overlap = 0
        best_spans = []
        for span in candidates:
            span_start = to_ns(span.get("start"))
            span_end = to_ns(span.get("end"))
            if span_start is None or span_end is None or span_end <= span_start:
                continue
            if dispatch_key and span.get("dispatch_key") and span.get("dispatch_key") != dispatch_key:
                continue
            span_ranks = span.get("ranks") if isinstance(span.get("ranks"), dict) else {}
            if not (
                _rank_compatible(ranks.get("pp"), span_ranks.get("pp"))
                and _rank_compatible(ranks.get("tp"), span_ranks.get("tp"))
                and _rank_compatible(ranks.get("dp"), span_ranks.get("dp"))
            ):
                continue
            overlap_start = max(start_ns, span_start)
            overlap_end = min(end_ns, span_end)
            overlap_ns = max(0, overlap_end - overlap_start)
            if overlap_ns <= 0:
                continue
            if overlap_ns > best_overlap:
                best_overlap = overlap_ns
                best_spans = [span]
            elif overlap_ns == best_overlap:
                best_spans.append(span)

        inferred_req_ids = []
        inferred_dispatch_keys = []
        for span in best_spans:
            inferred_req_ids.extend(normalize_request_ids(span.get("req_ids")))
            if span.get("dispatch_key"):
                inferred_dispatch_keys.append(span.get("dispatch_key"))

        inferred_req_ids = sorted(set(inferred_req_ids))
        inferred_dispatch_keys = sorted(set(inferred_dispatch_keys))
        inferred_dispatch_key = inferred_dispatch_keys[0] if len(inferred_dispatch_keys) == 1 else dispatch_key
        return inferred_req_ids, inferred_dispatch_key, best_overlap

    if pp_comm_events:
        print(f"Processing PP Communication Events ({len(pp_comm_events)})...")
    pp_comm_flow_points = defaultdict(dict)
    for ev in sorted(pp_comm_events, key=lambda x: x.get("ts") or 0):
        payload = ev.get("payload", {})
        start_ns = to_ns(payload.get("start_ns"))
        end_ns = to_ns(payload.get("end_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        pid = payload.get("pid")
        tid = payload.get("tid")
        op = payload.get("op", "pp_comm")
        dispatch_key = payload.get("dispatch_key")
        ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
        req_ids = normalize_request_ids(payload.get("req_ids"))
        pp_rank = ranks.get("pp")
        peer_rank = payload.get("peer_rank")
        attribution_source = "payload_req_ids"
        attribution_overlap_ns = 0
        if not req_ids or not dispatch_key:
            inferred_req_ids, inferred_dispatch_key, attribution_overlap_ns = _infer_pp_comm_request_context(
                start_ns,
                end_ns,
                pid,
                ranks,
                dispatch_key,
            )
            if not req_ids and inferred_req_ids:
                req_ids = inferred_req_ids
                attribution_source = "gpu_execute_model_overlap"
            if not dispatch_key and inferred_dispatch_key:
                dispatch_key = inferred_dispatch_key
                if attribution_source == "payload_req_ids":
                    attribution_source = "gpu_execute_model_overlap_dispatch_key"
        if req_ids:
            pp_comm_attr_stats["attributed"] += 1
            if attribution_source != "payload_req_ids":
                pp_comm_attr_stats["overlap_attributed"] += 1
        else:
            pp_comm_attr_stats["unattributed"] += 1
        if op in ("send_tensor_dict", "send_tensor", "isend_tensor_dict_launch"):
            flow_key = (dispatch_key, pp_rank, peer_rank)
            pp_comm_flow_points[flow_key]["send"] = (start_ns, pid, tid)
        elif op in ("recv_tensor_dict", "recv_tensor", "irecv_tensor_dict_launch"):
            flow_key = (dispatch_key, peer_rank, pp_rank)
            pp_comm_flow_points[flow_key]["recv"] = (start_ns, pid, tid)
        interval = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "op": op,
            "async_phase": payload.get("async_phase"),
            "comm_id": payload.get("comm_id"),
            "handle_index": payload.get("handle_index"),
            "launch_start_ns": payload.get("launch_start_ns"),
            "launch_end_ns": payload.get("launch_end_ns"),
            "async_elapsed_ns": payload.get("async_elapsed_ns"),
            "dispatch_key": dispatch_key,
            "ranks": ranks,
            "peer_rank": payload.get("peer_rank"),
            "src": payload.get("src"),
            "dst": payload.get("dst"),
            "tensor_bytes": payload.get("tensor_bytes"),
            "tensor_count": payload.get("tensor_count"),
            "attribution_source": attribution_source,
            "attribution_overlap_ns": attribution_overlap_ns,
        }
        for rid in req_ids:
            req_pp_comm_intervals[rid].append(dict(interval))
        trace_events.append(create_perfetto_event(
            name=f"PP Comm: {op}",
            cat="pp_comm",
            ph="X",
            ts=start_ns,
            dur=end_ns - start_ns,
            pid=pid,
            tid=tid,
            args={
                "dispatch_key": dispatch_key,
                "req_ids": "\n".join(req_ids),
                "attribution_source": attribution_source,
                "attribution_overlap_ns": attribution_overlap_ns,
                "dp_rank": ranks.get("dp"),
                "pp_rank": ranks.get("pp"),
                "tp_rank": ranks.get("tp"),
                "peer_rank": payload.get("peer_rank"),
                "src": payload.get("src"),
                "dst": payload.get("dst"),
                "comm_id": payload.get("comm_id"),
                "async_phase": payload.get("async_phase"),
                "handle_index": payload.get("handle_index"),
                "handle_count": payload.get("handle_count"),
                "async_elapsed_ns": payload.get("async_elapsed_ns"),
                "tensor_count": payload.get("tensor_count"),
                "tensor_bytes": payload.get("tensor_bytes"),
            },
            cname="rail_load",
        ))

    pp_flow_id_base = 700000
    for idx, (flow_key, points) in enumerate(pp_comm_flow_points.items(), start=1):
        send_point = points.get("send")
        recv_point = points.get("recv")
        if not send_point or not recv_point:
            continue
        flow_id = pp_flow_id_base + idx
        send_ts, send_pid, send_tid = send_point
        recv_ts, recv_pid, recv_tid = recv_point
        trace_events.append(create_flow_event("s", send_ts, send_pid, send_tid, flow_id))
        trace_events.append(create_flow_event("f", recv_ts, recv_pid, recv_tid, flow_id))

    if generated_dispatch_slices and cupti_nonzero_max_end_by_pid:
        execute_total_by_pid = defaultdict(int)
        execute_after_last_cupti_by_pid = defaultdict(int)
        for span in generated_dispatch_slices:
            pid = to_int(span.get("pid"))
            start_ns = to_ns(span.get("start"))
            if pid is None or start_ns is None:
                continue
            execute_total_by_pid[pid] += 1
            if start_ns > cupti_nonzero_max_end_by_pid.get(pid, 0):
                execute_after_last_cupti_by_pid[pid] += 1

        affected = {
            pid: execute_after_last_cupti_by_pid[pid]
            for pid in execute_after_last_cupti_by_pid
            if execute_after_last_cupti_by_pid[pid] > 0
        }
        if affected:
            print(
                "  [warn] Some Step Execution spans start after the last non-zero CUPTI event "
                "for the same worker pid."
            )
            for pid in sorted(
                affected,
                key=lambda x: affected[x],
                reverse=True,
            ):
                print(
                    f"    pid={pid}: step_exec_after_last_cupti="
                    f"{affected[pid]}/{execute_total_by_pid.get(pid, 0)}, "
                    f"last_nonzero_cupti_end_ns={cupti_nonzero_max_end_by_pid.get(pid)}"
                )

    # --- 3. 请求生命周期 (AsyncLLM.generate 包装) ---
    print(f"Processing Coroutine Lifecycle Events ({len(coroutine_events)})...")
    coroutine_events.sort(key=lambda x: x.get('ts') or 0)

    active_coroutines = {}
    request_spans = {}
    req_generate_start_map = {}
    req_coro_sched_ns = defaultdict(int)
    req_coro_sched_intervals = defaultdict(list)
    req_generate_task_sched_ns = defaultdict(int)
    req_generate_task_sched_intervals = defaultdict(list)
    req_generate_exec_ns = defaultdict(int)
    req_generate_exec_intervals = defaultdict(list)
    req_output_socket_sched_ns = defaultdict(int)
    req_output_socket_sched_intervals = defaultdict(list)
    req_output_socket_exec_ns = defaultdict(int)
    req_output_socket_exec_intervals = defaultdict(list)
    req_output_handler_sched_ns = defaultdict(int)
    req_output_handler_sched_intervals = defaultdict(list)
    req_output_handler_exec_ns = defaultdict(int)
    req_output_handler_exec_intervals = defaultdict(list)

    for ev in coroutine_events:
        etype = ev['type']
        payload = ev.get('payload', {})
        rid = payload.get('request_id') or payload.get('req_id')
        cid = normalize_request_id(payload.get('coroutine_id'))
        ts_ns = payload.get('timestamp_ns', ev.get('ts'))

        if etype == 'coroutine_sched_latency':
            ready_ts_ns = to_int(payload.get('ready_ts_ns'))
            run_ts_ns = to_int(payload.get('run_ts_ns'))
            queue_ns = to_int(payload.get('queue_ns'))
            task_kind = payload.get("task_kind")
            if task_kind == "generate_task":
                if rid and queue_ns is not None and queue_ns >= 0:
                    req_generate_task_sched_ns[rid] += queue_ns
                if rid and ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                    req_generate_task_sched_intervals[rid].append({
                        "start_ns": ready_ts_ns,
                        "end_ns": run_ts_ns,
                    })
            else:
                if rid and queue_ns is not None and queue_ns >= 0:
                    req_coro_sched_ns[rid] += queue_ns
                if rid and ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                    req_coro_sched_intervals[rid].append({
                        "start_ns": ready_ts_ns,
                        "end_ns": run_ts_ns,
                    })
            continue

        if etype == 'coroutine_start' and rid and cid and ts_ns:
            prev_start = req_generate_start_map.get(rid)
            if prev_start is None or ts_ns < prev_start:
                req_generate_start_map[rid] = ts_ns
            active_coroutines[cid] = {
                "request_id": rid,
                "start_ns": ts_ns,
                "pid": payload.get('pid', 0),
                "tid": payload.get('tid', 0),
            }
            continue

        if etype == 'coroutine_end' and rid and ts_ns:
            start_info = active_coroutines.pop(cid, None) if cid else None
            if start_info:
                start_ns = start_info['start_ns']
                pid = start_info['pid']
                tid = start_info['tid']
            else:
                duration_ms = payload.get('duration_ms')
                duration_ns = int(duration_ms * 1e6) if duration_ms else 0
                start_ns = ts_ns - duration_ns if duration_ns > 0 else ts_ns
                pid = payload.get('pid', 0)
                tid = payload.get('tid', 0)

            if ts_ns > start_ns:
                if rid not in request_spans:
                    request_spans[rid] = {
                        "start_ns": start_ns,
                        "end_ns": ts_ns,
                        "pid": pid,
                        "tid": tid,
                    }
                else:
                    request_spans[rid]["start_ns"] = min(request_spans[rid]["start_ns"], start_ns)
                    request_spans[rid]["end_ns"] = max(request_spans[rid]["end_ns"], ts_ns)

    if req_coro_sched_ns:
        print(f"共统计到 {len(req_coro_sched_ns)} 个请求的协程调度排队时间：")
        for rid, lat_ns in sorted(req_coro_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Coroutine Scheduler Queue)")
    else:
        print("  未统计到协程调度排队时间")

    if req_generate_task_sched_ns:
        print(f"共统计到 {len(req_generate_task_sched_ns)} 个请求的 generate task 运行队列时间：")
        for rid, lat_ns in sorted(req_generate_task_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Generate Task Runnable Queue)")
    else:
        print("  未统计到 generate task 运行队列时间")

    if coroutine_exec_events:
        print(f"Processing Generate Coroutine Exec Events ({len(coroutine_exec_events)})...")
    for ev in coroutine_exec_events:
        payload = ev.get("payload", {})
        rid = payload.get("request_id") or payload.get("req_id")
        if not rid:
            continue
        start_ns = to_int(payload.get("start_ns"))
        end_ns = to_int(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        if dur_ns is None and start_ns is not None and end_ns is not None:
            dur_ns = end_ns - start_ns
        if dur_ns is None or dur_ns < 0:
            continue
        req_generate_exec_ns[rid] += dur_ns
        if start_ns and end_ns and end_ns > start_ns:
            req_generate_exec_intervals[rid].append({
                "start_ns": start_ns,
                "end_ns": end_ns,
                "task_name": payload.get("task_name"),
                "task_kind": payload.get("task_kind"),
            })

    if req_generate_exec_ns:
        print(f"共统计到 {len(req_generate_exec_ns)} 个请求的 generate 协程执行时间：")
        for rid, lat_ns in sorted(req_generate_exec_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Generate Coroutine Exec)")
    elif coroutine_exec_events:
        print("  generate 协程执行事件存在，但未找到可归因的 request_id")

    if output_socket_sched_events:
        print(f"Processing Output Socket Scheduler Events ({len(output_socket_sched_events)})...")
    for ev in output_socket_sched_events:
        payload = ev.get("payload", {})
        ready_ts_ns = to_int(payload.get("ready_ts_ns"))
        run_ts_ns = to_int(payload.get("run_ts_ns"))
        queue_ns = to_int(payload.get("queue_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if not req_ids or queue_ns is None or queue_ns < 0:
            continue
        for rid in req_ids:
            req_output_socket_sched_ns[rid] += queue_ns
            if ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                req_output_socket_sched_intervals[rid].append({
                    "start_ns": ready_ts_ns,
                    "end_ns": run_ts_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_socket_sched_ns:
        print(f"共统计到 {len(req_output_socket_sched_ns)} 个请求的 output_socket 调度排队时间：")
        for rid, lat_ns in sorted(req_output_socket_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Socket Scheduler Queue)")
    elif output_socket_sched_events:
        print("  output_socket 调度事件存在，但未找到可归因的 req_ids")

    if output_socket_exec_events:
        print(f"Processing Output Socket Exec Events ({len(output_socket_exec_events)})...")
    for ev in output_socket_exec_events:
        payload = ev.get("payload", {})
        start_ns = to_int(payload.get("start_ns"))
        end_ns = to_int(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if dur_ns is None and start_ns is not None and end_ns is not None:
            dur_ns = end_ns - start_ns
        if not req_ids or dur_ns is None or dur_ns < 0:
            continue
        for rid in req_ids:
            req_output_socket_exec_ns[rid] += dur_ns
            if start_ns and end_ns and end_ns > start_ns:
                req_output_socket_exec_intervals[rid].append({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_socket_exec_ns:
        print(f"共统计到 {len(req_output_socket_exec_ns)} 个请求的 output_socket 执行时间：")
        for rid, lat_ns in sorted(req_output_socket_exec_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Socket Exec)")
    elif output_socket_exec_events:
        print("  output_socket 执行事件存在，但未找到可归因的 req_ids")

    if output_handler_sched_events:
        print(f"Processing Output Handler Scheduler Events ({len(output_handler_sched_events)})...")
    for ev in output_handler_sched_events:
        payload = ev.get("payload", {})
        ready_ts_ns = to_int(payload.get("ready_ts_ns"))
        run_ts_ns = to_int(payload.get("run_ts_ns"))
        queue_ns = to_int(payload.get("queue_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if not req_ids or queue_ns is None or queue_ns < 0:
            continue
        for rid in req_ids:
            req_output_handler_sched_ns[rid] += queue_ns
            if ready_ts_ns and run_ts_ns and run_ts_ns > ready_ts_ns:
                req_output_handler_sched_intervals[rid].append({
                    "start_ns": ready_ts_ns,
                    "end_ns": run_ts_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_handler_sched_ns:
        print(f"共统计到 {len(req_output_handler_sched_ns)} 个请求的 output_handler 调度排队时间：")
        for rid, lat_ns in sorted(req_output_handler_sched_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Handler Scheduler Queue)")
    elif output_handler_sched_events:
        print("  output_handler 调度事件存在，但未找到可归因的 req_ids")

    if output_handler_exec_events:
        print(f"Processing Output Handler Exec Events ({len(output_handler_exec_events)})...")
    for ev in output_handler_exec_events:
        payload = ev.get("payload", {})
        start_ns = to_int(payload.get("start_ns"))
        end_ns = to_int(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        req_ids = normalize_request_ids(payload.get("req_ids"))
        if dur_ns is None and start_ns is not None and end_ns is not None:
            dur_ns = end_ns - start_ns
        if not req_ids or dur_ns is None or dur_ns < 0:
            continue
        for rid in req_ids:
            req_output_handler_exec_ns[rid] += dur_ns
            if start_ns and end_ns and end_ns > start_ns:
                req_output_handler_exec_intervals[rid].append({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "shared_req_count": len(req_ids),
                    "round_seq": payload.get("round_seq"),
                })

    if req_output_handler_exec_ns:
        print(f"共统计到 {len(req_output_handler_exec_ns)} 个请求的 output_handler 执行时间：")
        for rid, lat_ns in sorted(req_output_handler_exec_ns.items(), key=lambda x: x[1], reverse=True):
            label = req_name_map.get(rid, rid)
            print(f"  - Request {rid} ({label}): {lat_ns / 1e6:.3f} ms (Output Handler Exec)")
    elif output_handler_exec_events:
        print("  output_handler 执行事件存在，但未找到可归因的 req_ids")

    # 为 request dispatch phase 做回退兜底：
    # 从 req_metrics_events 中按时间提取每个请求的 "scheduled/dequeue" phase 顺序。
    phase_hint_queue_by_req = defaultdict(deque)
    req_preempt_ts_by_req = defaultdict(list)
    req_metric_events_sorted = sorted(
        req_metric_events,
        key=lambda x: to_ns(x.get("batch_ts_ns")) or 0,
    )
    for item in req_metric_events_sorted:
        payload = item.get("payload", item)
        updates = payload.get("req_events", [])
        for update in updates:
            rid = update.get("req_id")
            if not rid:
                continue
            for ev in update.get("events", []):
                ev_type = normalize_req_event_type(ev.get("type"))
                ev_ts_ns = to_ns(ev.get("ts") or ev.get("timestamp_ns"))
                if ev_type == "preempt" and ev_ts_ns is not None:
                    req_preempt_ts_by_req[rid].append(ev_ts_ns)
                if ev_type != "dequeue":
                    continue
                phase = ev.get("phase")
                if phase in ("prefill", "decode"):
                    phase_hint_queue_by_req[rid].append(phase)
                break

    # --- 4. Scheduler 处理 (保持原逻辑) ---
    print("Processing Scheduler Events...")
    req_enqueue_map = {}
    queue_intervals_map = defaultdict(list)  # 可视化用（墙上时间）
    req_vllm_queue_ns = defaultdict(int)  # 聚合统计（总排队）
    req_vllm_queue_from_enqueue_ns = defaultdict(int)  # 初次调度等待
    req_vllm_queue_from_step_ready_ns = defaultdict(int)  # 后续step等待
    req_dispatch_intervals_map = defaultdict(list)  # 请求每次被调度执行的区间（out_rpc -> step_ready）
    req_dispatch_phase_ns = defaultdict(lambda: defaultdict(int))
    req_dispatch_phase_count = defaultdict(lambda: defaultdict(int))

    # 3.1 只保留事件到请求级区间的计算，不再生成 category=scheduler 的顶层点/流。
    for item in scheduler_events:
        if item['type'] != 'req_enqueue_scheduler':
            continue
        p = item['payload']
        rid = p.get('req_id')
        ts = p.get('timestamp_ns')
        if rid and ts and rid not in req_enqueue_map:
            req_enqueue_map[rid] = ts

    scheduler_events_sorted = sorted(
        scheduler_events,
        key=lambda x: to_ns(x.get("payload", {}).get("timestamp_ns")) or 0,
    )

    # 3.5 每次调度执行区间：
    # vLLM 0.20 的 async PP 会提前提交多个未来 step，req_scheduler_out_rpc
    # 更接近 RPC/队列提交点，不再可靠代表 worker 实际执行开始。request 级
    # dispatch 可视化以 scheduler_output/dispatch_key 为 join 单位：request 归属来自
    # scheduler out/ready 的 req_ids，worker 执行窗口来自同 dispatch_key 下所有
    # GPUModelRunner.execute_model 的最早开始和最晚结束。这里刻意不按 worker
    # payload.req_ids 再过滤 request，因为 PP/异步路径里 worker 上下文字段可能不完整
    # 或展示被截断；dispatch_key 才是同一 scheduler_output 的稳定连接键。
    worker_exec_window_by_dispatch = {}
    grouped_worker_spans_by_dispatch = defaultdict(list)
    for span in generated_dispatch_slices:
        dispatch_key = span.get("dispatch_key")
        start_ns = to_ns(span.get("start"))
        end_ns = to_ns(span.get("end"))
        if not dispatch_key or start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        grouped_worker_spans_by_dispatch[dispatch_key].append(span)

    for dispatch_key, spans in grouped_worker_spans_by_dispatch.items():
        starts = [to_ns(item.get("start")) for item in spans]
        ends = [to_ns(item.get("end")) for item in spans]
        starts = [value for value in starts if value is not None]
        ends = [value for value in ends if value is not None]
        if not starts or not ends:
            continue
        worker_exec_window_by_dispatch[dispatch_key] = {
            "start_ns": min(starts),
            "end_ns": max(ends),
            "span_count": len(spans),
            "total_span_count": len(spans),
            "boundary_policy": "all_pp_all_tp_execute_model",
            "pids": sorted({
                item.get("pid") for item in spans if item.get("pid") is not None
            }),
            "tids": sorted({
                item.get("tid") for item in spans if item.get("tid") is not None
            }),
        }

    gpu_batch_by_dispatch = {}
    marker_events = [ev for ev in cupti_events if ev.get("type") == "marker"]
    if marker_events:
        markers_by_dk = defaultdict(lambda: defaultdict(list))
        for m in marker_events:
            meta = m.get("name", "")
            parts = meta.split("|", 1)
            if len(parts) != 2:
                continue
            role, dk = parts
            markers_by_dk[dk][role].append(m)
        for dk, roles in markers_by_dk.items():
            starts = []
            ends   = []
            for role in ("exec_start", "sample_start"):
                for m in roles.get(role, []):
                    v = m.get("start_ns", 0)
                    if v > 0:
                        starts.append(v)
            for role in ("exec_end", "sample_end"):
                for m in roles.get(role, []):
                    v = m.get("end_ns", 0)
                    if v > 0:
                        ends.append(v)
            if starts and ends:
                gpu_start = min(starts)
                gpu_end   = max(ends)
                if gpu_end > gpu_start:
                    gpu_batch_by_dispatch[dk] = {
                        "gpu_start_ns": gpu_start,
                        "gpu_end_ns":   gpu_end,
                        "exec_markers": len(roles.get("exec_start", [])) + len(roles.get("exec_end", [])),
                        "samp_markers": len(roles.get("sample_start", [])) + len(roles.get("sample_end", [])),
                    }

    # phase 优先来自 req_step_ready.phase_by_req，缺失时回退到 req_metrics_events 的 phase 队列。
    pending_dispatch_start = defaultdict(deque)

    def _pop_dispatch_start_for_ready(rid, dispatch_key):
        queue = pending_dispatch_start[rid]
        if not queue:
            return None
        if dispatch_key:
            for idx, item in enumerate(queue):
                if item.get("dispatch_key") == dispatch_key:
                    del queue[idx]
                    return item
        return queue.popleft()

    for item in scheduler_events_sorted:
        p = item.get("payload", {})
        ev_type = item.get("type")
        ts_ns = to_ns(p.get("timestamp_ns"))
        if ts_ns is None:
            continue

        if ev_type == "req_scheduler_out_rpc":
            dispatch_key = p.get("dispatch_key")
            batch_req_ids = list(p.get("req_ids", []))
            decode_step_by_req = p.get("decode_step_by_req")
            if not isinstance(decode_step_by_req, dict):
                decode_step_by_req = {}
            for rid in p.get("req_ids", []):
                if rid:
                    pending_dispatch_start[rid].append({
                        "start_ns": ts_ns,
                        "rpc_start_ns": ts_ns,
                        "dispatch_key": dispatch_key,
                        "batch_req_ids": batch_req_ids,
                        "decode_step": to_int(decode_step_by_req.get(rid)),
                    })
            continue

        if ev_type != "req_step_ready":
            continue

        ready_dispatch_key = p.get("dispatch_key")
        phase_by_req = p.get("phase_by_req")
        if not isinstance(phase_by_req, dict):
            phase_by_req = {}
        phase_reason_by_req = p.get("phase_reason_by_req")
        if not isinstance(phase_reason_by_req, dict):
            phase_reason_by_req = {}
        decode_step_by_req = p.get("decode_step_by_req")
        if not isinstance(decode_step_by_req, dict):
            decode_step_by_req = {}

        for rid in p.get("req_ids", []):
            if not rid:
                continue
            start_item = _pop_dispatch_start_for_ready(rid, ready_dispatch_key)
            if start_item is None:
                continue
            dispatch_key = ready_dispatch_key or start_item.get("dispatch_key")
            worker_window = worker_exec_window_by_dispatch.get(dispatch_key)
            rpc_start_ns = to_ns(start_item.get("rpc_start_ns") or start_item.get("start_ns"))
            start_ns = to_ns(start_item.get("start_ns"))
            end_ns = ts_ns
            start_source = "scheduler_out_rpc"
            end_source = "scheduler_step_ready"
            worker_start_ns = None
            worker_end_ns = None
            worker_span_count = 0
            worker_span_count_total = 0
            worker_boundary_policy = None

            if worker_window:
                worker_start_ns = to_ns(worker_window.get("start_ns"))
                worker_end_ns = to_ns(worker_window.get("end_ns"))
                worker_span_count = worker_window.get("span_count", 0)
                worker_span_count_total = worker_window.get("total_span_count", worker_span_count)
                worker_boundary_policy = worker_window.get("boundary_policy")

            gpu_window = gpu_batch_by_dispatch.get(dispatch_key)
            if gpu_window:
                start_ns = gpu_window["gpu_start_ns"]
                end_ns   = gpu_window["gpu_end_ns"]
                start_source = "gpu_marker"
                end_source   = "gpu_marker"
            elif worker_window:
                if worker_start_ns is not None:
                    start_ns = worker_start_ns
                    start_source = "worker_execute_model_min"
                if worker_end_ns is not None:
                    end_ns = worker_end_ns
                    end_source = "worker_execute_model_max"

            if start_ns is None:
                continue
            if end_ns <= start_ns:
                continue

            phase = phase_by_req.get(rid)
            phase_reason = phase_reason_by_req.get(rid)
            if phase not in ("prefill", "decode"):
                if phase_hint_queue_by_req[rid]:
                    phase = phase_hint_queue_by_req[rid].popleft()
                    phase_reason = "req_metrics_events_fallback"
                else:
                    phase = "unknown"
                    phase_reason = "missing_phase"
            else:
                # 若 hint 队列头和当前 phase 一致，则消费掉，保持两路数据大致对齐。
                if phase_hint_queue_by_req[rid] and phase_hint_queue_by_req[rid][0] == phase:
                    phase_hint_queue_by_req[rid].popleft()

            decode_step = None
            if phase == "decode":
                decode_step = to_int(decode_step_by_req.get(rid))
                if decode_step is None:
                    decode_step = to_int(start_item.get("decode_step"))

            dur_ns = end_ns - start_ns
            req_dispatch_intervals_map[rid].append({
                "start_ns": start_ns,
                "end_ns": end_ns,
                "phase": phase,
                "phase_reason": phase_reason,
                "decode_step": decode_step,
                "duration_ns": dur_ns,
                "dispatch_key": dispatch_key,
                "batch_req_ids": start_item.get("batch_req_ids", []),
                "rpc_start_ns": rpc_start_ns,
                "scheduler_ready_ns": ts_ns,
                "start_source": start_source,
                "end_source": end_source,
                "worker_start_ns": worker_start_ns,
                "worker_end_ns": worker_end_ns,
                "worker_span_count": worker_span_count,
                "worker_span_count_total": worker_span_count_total,
                "worker_boundary_policy": worker_boundary_policy,
            })
            req_dispatch_phase_ns[rid][phase] += dur_ns
            req_dispatch_phase_count[rid][phase] += 1

    unmatched_dispatch = sum(len(q) for q in pending_dispatch_start.values())
    if unmatched_dispatch > 0:
        print(f"  [warn] 存在 {unmatched_dispatch} 个未闭合 dispatch 起点（可能是日志截断或进程中止）")

    # Newer vLLM paths can surface the same scheduler step through multiple
    # executor wrappers.  Keep the visual/event detail, but compute per-request
    # phase totals from de-duplicated wall-clock intervals so parallel/duplicate
    # decode windows are not charged multiple times to one request.
    deduped_dispatch_intervals_map = defaultdict(list)
    for rid, intervals in req_dispatch_intervals_map.items():
        seen = set()
        for interval in sorted(intervals, key=lambda x: (x.get("start_ns") or 0, x.get("end_ns") or 0)):
            start_ns = to_ns(interval.get("start_ns"))
            end_ns = to_ns(interval.get("end_ns"))
            if start_ns is None or end_ns is None or end_ns <= start_ns:
                continue
            phase = interval.get("phase") or "unknown"
            dispatch_key = interval.get("dispatch_key")
            dedupe_key = (phase, dispatch_key, start_ns, end_ns)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            item = dict(interval)
            item["start_ns"] = start_ns
            item["end_ns"] = end_ns
            item["phase"] = phase
            item["duration_ns"] = end_ns - start_ns
            deduped_dispatch_intervals_map[rid].append(item)

    req_dispatch_intervals_map = deduped_dispatch_intervals_map
    for rid, intervals in req_dispatch_intervals_map.items():
        next_decode_step = 1
        for interval in sorted(intervals, key=lambda x: (x["start_ns"], x["end_ns"])):
            if interval.get("phase") != "decode":
                continue
            decode_step = to_int(interval.get("decode_step"))
            if decode_step is None or decode_step <= 0:
                interval["decode_step"] = next_decode_step
                next_decode_step += 1
            else:
                interval["decode_step"] = decode_step
                next_decode_step = max(next_decode_step, decode_step + 1)
    req_dispatch_phase_ns = defaultdict(lambda: defaultdict(int))
    req_dispatch_phase_count = defaultdict(lambda: defaultdict(int))
    for rid, intervals in req_dispatch_intervals_map.items():
        spans_by_phase = defaultdict(list)
        dispatch_keys_by_phase = defaultdict(set)
        for interval in intervals:
            phase = interval.get("phase") or "unknown"
            spans_by_phase[phase].append((interval["start_ns"], interval["end_ns"]))
            dispatch_keys_by_phase[phase].add(
                interval.get("dispatch_key") or (interval["start_ns"], interval["end_ns"])
            )
        for phase, spans in spans_by_phase.items():
            req_dispatch_phase_ns[rid][phase] = sum_interval_pairs(merge_intervals(spans))
            req_dispatch_phase_count[rid][phase] = len(dispatch_keys_by_phase[phase])

    # Derive vLLM queue intervals from the canonical dispatch sequence.
    # This keeps queue slices in the gaps between dispatch slices and prevents
    # visually impossible overlap such as step_ready->out_rpc appearing inside
    # a Decode Dispatch lane.
    overlapped_dispatch_pairs = 0
    for rid, dispatches_for_req in req_dispatch_intervals_map.items():
        dispatches_sorted = sorted(
            dispatches_for_req,
            key=lambda x: (x.get("start_ns") or 0, x.get("end_ns") or 0),
        )
        if not dispatches_sorted:
            continue

        enqueue_ts = to_ns(req_enqueue_map.get(rid))
        first_start = to_ns(dispatches_sorted[0].get("start_ns"))
        if enqueue_ts is not None and first_start is not None and first_start > enqueue_ts:
            queue_ns = first_start - enqueue_ts
            req_vllm_queue_ns[rid] += queue_ns
            req_vllm_queue_from_enqueue_ns[rid] += queue_ns
            queue_intervals_map[rid].append({
                "start_ns": enqueue_ts,
                "end_ns": first_start,
                "reason": "scheduled_enqueue_wait",
                "scheduled_from": "enqueue",
                "gap_kind": "normal_wait",
                "next_dispatch_key": dispatches_sorted[0].get("dispatch_key"),
            })

        for prev_dispatch, next_dispatch in zip(dispatches_sorted, dispatches_sorted[1:]):
            prev_end = to_ns(prev_dispatch.get("end_ns"))
            next_start = to_ns(next_dispatch.get("start_ns"))
            if prev_end is None or next_start is None:
                continue
            if next_start < prev_end:
                overlapped_dispatch_pairs += 1
                continue
            if next_start == prev_end:
                continue
            queue_ns = next_start - prev_end
            req_vllm_queue_ns[rid] += queue_ns
            req_vllm_queue_from_step_ready_ns[rid] += queue_ns
            preempt_hits = [
                ts for ts in req_preempt_ts_by_req.get(rid, [])
                if prev_end <= ts <= next_start
            ]
            gap_kind = "preempt_wait" if preempt_hits else "normal_wait"
            queue_intervals_map[rid].append({
                "start_ns": prev_end,
                "end_ns": next_start,
                "reason": "scheduled_preempt_wait" if preempt_hits else "scheduled_step_ready_wait",
                "scheduled_from": "step_ready",
                "gap_kind": gap_kind,
                "preempt_ts_ns": preempt_hits,
                "prev_dispatch_key": prev_dispatch.get("dispatch_key"),
                "next_dispatch_key": next_dispatch.get("dispatch_key"),
            })

    if overlapped_dispatch_pairs:
        print(
            f"  [warn] 存在 {overlapped_dispatch_pairs} 对同请求 dispatch 时间重叠，"
            "已跳过对应 queue gap；建议检查 dispatch_key 或日志时序"
        )

    if req_vllm_queue_ns:
        print(f"共统计到 {len(req_vllm_queue_ns)} 个请求的 vLLM 内部排队时间（Dispatch gap 口径）：")
        for rid, lat_ns in sorted(req_vllm_queue_ns.items(), key=lambda x: x[1], reverse=True):
            enqueue_ns = req_vllm_queue_from_enqueue_ns.get(rid, 0)
            step_ready_ns = req_vllm_queue_from_step_ready_ns.get(rid, 0)
            print(
                f"  - Request {rid}: {lat_ns / 1e6:.3f} ms "
                f"(scheduled-enqueue={enqueue_ns / 1e6:.3f} ms, "
                f"scheduled-step_ready={step_ready_ns / 1e6:.3f} ms)"
            )
    else:
        print("  未统计到 vLLM 内部排队时间（Dispatch gap 口径）")

    if req_dispatch_intervals_map:
        print(f"共统计到 {len(req_dispatch_intervals_map)} 个请求的 dispatch 区间（prefill/decode）:")
        for rid in sorted(
            req_dispatch_intervals_map.keys(),
            key=lambda x: req_dispatch_phase_ns[x].get("prefill", 0) + req_dispatch_phase_ns[x].get("decode", 0),
            reverse=True
        ):
            prefill_ms = req_dispatch_phase_ns[rid].get("prefill", 0) / 1e6
            decode_ms = req_dispatch_phase_ns[rid].get("decode", 0) / 1e6
            unknown_ms = req_dispatch_phase_ns[rid].get("unknown", 0) / 1e6
            print(
                f"  - Request {rid}: prefill_exec={prefill_ms:.3f} ms "
                f"(count={req_dispatch_phase_count[rid].get('prefill', 0)}), "
                f"decode_exec={decode_ms:.3f} ms "
                f"(count={req_dispatch_phase_count[rid].get('decode', 0)}), "
                f"unknown_exec={unknown_ms:.3f} ms"
            )
    else:
        print("  未统计到可配对的 dispatch 区间（prefill/decode）")

    # --- 4.5 generate->enqueue 阶段耗时拆分（统计 + 部分阶段可视化数据准备） ---
    print(f"Processing Request Stage Events ({len(req_stage_events)})...")
    req_stage_ns = defaultdict(lambda: defaultdict(int))
    req_stage_counts = defaultdict(lambda: defaultdict(int))
    req_stage_intervals = defaultdict(list)

    for ev in req_stage_events:
        payload = ev.get("payload", {})
        rid = payload.get("request_id") or payload.get("req_id")
        if not rid:
            continue
        rname = payload.get("request_name")
        if rname:
            req_name_map[rid] = rname
        stage = payload.get("stage")
        if not stage:
            continue
        start_ns = to_ns(payload.get("start_ns"))
        end_ns = to_ns(payload.get("end_ns"))
        dur_ns = to_int(payload.get("duration_ns"))
        if dur_ns is None:
            if start_ns is None or end_ns is None or end_ns < start_ns:
                continue
            dur_ns = end_ns - start_ns
        if dur_ns < 0:
            continue
        req_stage_ns[rid][stage] += dur_ns
        req_stage_counts[rid][stage] += 1
        if start_ns is not None and end_ns is not None and end_ns > start_ns:
            req_stage_intervals[rid].append({
                "stage": stage,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
            })

    if req_stage_ns:
        stage_order = [
            "async_llm.generate_wrapper_setup",
            "async_llm._run_output_handler",
            "processor.process_inputs",
            "output_processor.add_request",
            "engine_core.add_request_async",
            "engine_core.preprocess_add_request",
            "engine_core.input_queue_wait_after_preprocess",
            "engine_core.handle_client_add",
            "engine_core.add_request",
            "async_llm._add_request",
            "async_llm.add_request",
        ]
        all_stage_reqs = set(req_stage_ns.keys()) | set(req_generate_start_map.keys()) | set(req_enqueue_map.keys())
        print("=== Generate -> Scheduler Enqueue Breakdown ===")
        for rid in sorted(
            all_stage_reqs,
            key=lambda x: (
                (to_ns(req_enqueue_map.get(x)) or 0) - (to_ns(req_generate_start_map.get(x)) or 0)
                if (to_ns(req_enqueue_map.get(x)) and to_ns(req_generate_start_map.get(x)))
                else sum(req_stage_ns.get(x, {}).values())
            ),
            reverse=True,
        ):
            stage_map = req_stage_ns.get(rid, {})
            if not stage_map:
                continue
            known_ns = sum(stage_map.values())
            gen_start_ns = to_ns(req_generate_start_map.get(rid))
            enqueue_ns = to_ns(req_enqueue_map.get(rid))
            pre_enqueue_total_ns = None
            if gen_start_ns is not None and enqueue_ns is not None and enqueue_ns >= gen_start_ns:
                pre_enqueue_total_ns = enqueue_ns - gen_start_ns
            uncovered_ns = None
            if pre_enqueue_total_ns is not None:
                uncovered_ns = max(0, pre_enqueue_total_ns - known_ns)

            label = req_name_map.get(rid, rid)
            total_str = f"{pre_enqueue_total_ns / 1e6:.3f} ms" if pre_enqueue_total_ns is not None else "N/A"
            uncovered_str = f"{uncovered_ns / 1e6:.3f} ms" if uncovered_ns is not None else "N/A"
            print(f"  - {rid} ({label}): total(generate->enqueue)={total_str}, known_stages={known_ns / 1e6:.3f} ms, uncovered={uncovered_str}")

            printed = set()
            for stage in stage_order:
                if stage in stage_map:
                    print(
                        f"      * {stage}: {stage_map[stage] / 1e6:.3f} ms "
                        f"(count={req_stage_counts[rid].get(stage, 0)})"
                    )
                    printed.add(stage)
            for stage, dur_ns in sorted(stage_map.items(), key=lambda x: x[1], reverse=True):
                if stage in printed:
                    continue
                print(
                    f"      * {stage}: {dur_ns / 1e6:.3f} ms "
                    f"(count={req_stage_counts[rid].get(stage, 0)})"
                )
    else:
        print("  未统计到 generate->enqueue 阶段事件")

    # --- 4.6 EngineCore 主循环可视化（单独轨道） ---
    print(f"Processing EngineCore Main Loop Events ({len(enginecore_loop_events)})...")
    if enginecore_loop_events:
        ENGINECORE_PID = 12000
        trace_events.append({"name": "process_name", "ph": "M", "pid": ENGINECORE_PID, "args": {"name": "EngineCore Main Loop"}})
        tid_alias_map = {}
        next_tid_alias = 1
        for item in sorted(enginecore_loop_events, key=lambda x: x.get("ts") or 0):
            payload = item.get("payload", {})
            phase = payload.get("phase", "unknown")
            start_ns = to_ns(payload.get("start_ns"))
            end_ns = to_ns(payload.get("end_ns"))
            pid = payload.get("pid")
            tid = payload.get("tid")
            if start_ns is None or end_ns is None or end_ns <= start_ns:
                continue

            key = (pid, tid)
            if key not in tid_alias_map:
                tid_alias_map[key] = next_tid_alias
                trace_events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": ENGINECORE_PID,
                    "tid": next_tid_alias,
                    "args": {"name": f"pid={pid} tid={tid}"},
                })
                next_tid_alias += 1
            tid_alias = tid_alias_map[key]

            display_name = "EngineCore: Input Queue Phase" if phase == "_process_input_queue" else "EngineCore: Engine Step Phase"
            trace_events.append(create_perfetto_event(
                name=display_name,
                cat="enginecore_loop",
                ph="X",
                ts=start_ns,
                dur=end_ns - start_ns,
                pid=ENGINECORE_PID,
                tid=tid_alias,
                args={
                    "phase": phase,
                    "source_pid": pid,
                    "source_tid": tid,
                    "duration_ms": round((end_ns - start_ns) / 1e6, 3),
                },
                cname="rail_idle" if phase == "_process_input_queue" else "rail_response",
            ))


    # --- 4. CUPTI 关联 (使用生成的 Dispatch Slices) ---
    print(f"Linking Runtime events ({len(cupti_events)}) to Dispatch Slices...")
    # 按时间排序 Runtime 事件
    cupti_events.sort(key=lambda x: x['start_ns'])

    cupti_api_lane_ends_by_pid = defaultdict(list)
    cupti_api_lane_names_emitted = set()
    cupti_kernel_flow_id_by_obj = {}
    cupti_next_flow_id = 1_000_000_000

    def allocate_cupti_api_tid(pid, start_ns, end_ns, raw_tid):
        tid = to_int(raw_tid)
        if tid is not None and tid > 0:
            return tid

        # Older logs did not persist CUpti_ActivityAPI.threadId, so every API
        # call landed on tid=0. Split those records into virtual lanes to keep
        # Perfetto from dropping partially-overlapping complete events.
        lanes = cupti_api_lane_ends_by_pid[pid]
        lane_idx = None
        for idx, lane_end in enumerate(lanes):
            if start_ns >= lane_end:
                lane_idx = idx
                break
        if lane_idx is None:
            lane_idx = len(lanes)
            lanes.append(end_ns)
        else:
            lanes[lane_idx] = end_ns

        virtual_tid = 10_000_000 + lane_idx
        emit_thread_name_once(
            trace_events,
            cupti_api_lane_names_emitted,
            pid,
            virtual_tid,
            f"CUPTI API lane {lane_idx + 1} (tid unknown)",
        )
        return virtual_tid

    def next_cupti_flow_id():
        nonlocal cupti_next_flow_id
        flow_id = cupti_next_flow_id
        cupti_next_flow_id += 1
        return flow_id
    
    for rt in cupti_events:
        rt_start = rt['start_ns']
        rt_end = rt['end_ns']
        rt_pid = rt.get('pid')
        
        # 如果 payload 里没存 pid，尝试从生成的 slice 里找（备选方案）
        if not rt_pid and generated_dispatch_slices:
            rt_pid = generated_dispatch_slices[0]['pid']
        rt_tid = allocate_cupti_api_tid(rt_pid, rt_start, rt_end, rt.get('tid'))

        corr_id = rt.get('correlationId')
        func_name = rt.get('name', 'cuda_runtime')
        
        # 查找同一进程内关联的 Kernel。CUPTI correlationId 不是跨进程全局唯一，
        # DP 多进程下必须用 (pid, correlationId) 做键，避免一个 runtime
        # 错连到另一个 EngineCore 进程的 GPU kernel。
        runtime_corr_key = cupti_corr_key(rt, corr_id=corr_id, pid_override=rt_pid)
        runtime_flow_debug_key = cupti_flow_debug_key(
            rt,
            corr_id=corr_id,
            pid_override=rt_pid,
        )
        kernels = all_kernels_map.get(runtime_corr_key, [])
        k_names = [k.get('name', 'unknown') for k in kernels]
        
        args = {
            "correlationId": corr_id,
            "correlationKey": runtime_flow_debug_key,
            "kernels": "\n".join(k_names[:5]) + ("..." if len(k_names)>5 else ""),
            "is_graph": "graph" in func_name.lower()
        }
        
        # 只要有 pid 和 tid，就直接在 Worker 轨道绘制 Runtime 条
        # 它会自动出现在 Python "Model Dispatch" 条的下方，因为它们共享 PID/TID
        trace_events.append(create_perfetto_event(
            name=func_name, cat="cuda_runtime", ph="X", ts=rt_start, dur=rt_end-rt_start,
            pid=rt_pid, tid=rt_tid, args=args
        ))
        
        # 画 Flow 线 (Runtime -> Kernel)。一个 runtime correlation 可能对应
        # CUDA Graph replay 的数百个 kernel；Perfetto 要求每条 flow 的 id 只
        # 对应一对 start/finish，因此这里为每个 kernel 单独分配 flow id。
        for k in kernels:
            if k.get('start_ns', 0) <= 0:
                continue
            kernel_obj_id = id(k)
            if kernel_obj_id in cupti_kernel_flow_id_by_obj:
                continue
            flow_id = next_cupti_flow_id()
            cupti_kernel_flow_id_by_obj[kernel_obj_id] = flow_id
            trace_events.append(create_flow_event("s", rt_start, rt_pid, rt_tid, flow_id))

    # --- 5. 画 GPU Kernel ---
    print(f"Processing Eager Kernels ({len(eager_kernels)})...")
    gpu_track_pid_by_key = {}
    gpu_track_names_emitted = set()

    def get_gpu_track_pid(source_pid, device):
        # CUPTI 的 deviceId 是进程内 local ordinal。DP 多进程时，两个
        # EngineCore 可能都看到 deviceId=0，所以轨道必须同时按 pid 区分。
        try:
            source_pid = int(source_pid)
        except (TypeError, ValueError):
            source_pid = 0
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = 0

        key = (source_pid, device)
        if key not in gpu_track_pid_by_key:
            gpu_track_pid_by_key[key] = 9000 + len(gpu_track_pid_by_key)
        return gpu_track_pid_by_key[key], key

    for k in eager_kernels:
        start = k['start_ns']
        end = k['end_ns']
        corr_id = k.get('correlationId')
        stream = k.get('streamId', 0)
        device = k.get('deviceId', 0)
        source_pid = k.get('pid', 0)
        
        # 虚拟 PID 用于区分 GPU 执行轨道。这里用 (source_pid, local device)
        # 而不是单独 deviceId，避免 DP0/DP1 的 local GPU 0 被合并到一条轴。
        gpu_pid, gpu_key = get_gpu_track_pid(source_pid, device)
        
        trace_events.append(create_perfetto_event(
            name=k.get('name', 'kernel'), cat="gpu_kernel", ph="X", ts=start, dur=end-start,
            pid=gpu_pid, tid=stream, args=k
        ))
        
        # 命名轨道
        if gpu_key not in gpu_track_names_emitted:
            source_pid_label, device_label = gpu_key
            trace_events.append({
                "name": "process_name",
                "ph": "M",
                "pid": gpu_pid,
                "args": {
                    "name": f"GPU local {device_label} / source pid {source_pid_label}",
                },
            })
            gpu_track_names_emitted.add(gpu_key)
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": gpu_pid,
            "tid": stream,
            "args": {"name": f"Stream {stream}"},
        })

        # Flow 终点
        flow_id = cupti_kernel_flow_id_by_obj.get(id(k))
        if flow_id is not None:
            trace_events.append(create_flow_event(
                "f",
                start,
                gpu_pid,
                stream,
                flow_id,
            ))

    print(f"Processing Request Metric Events ({len(req_metric_events)})...")
    req_metric_mono_to_wall_offsets = []

    for item in req_metric_events:
        payload = item.get('payload', item)
        batch_ts_ns = to_ns(item.get('batch_ts_ns'))
        updates = payload.get('req_events', [])

        # 仅用于 mono->wall 偏移估计（供 OS overlap 画到 wall 时间轴）
        batch_mono_ns = []
        for update in updates:
            for ev in update.get('events', []):
                ev_ns = to_ns(ev.get('ts'))
                if ev_ns:
                    batch_mono_ns.append(ev_ns)

        if batch_ts_ns and batch_mono_ns:
            # 注意：req_metrics_events 的 wall 时间发生在一批事件被整理并发出之后，
            # 这里会混入 Python 处理与发送延迟，因此只能作为最后兜底。
            req_metric_mono_to_wall_offsets.append(batch_ts_ns - max(batch_mono_ns))

    explicit_mono_to_wall_offsets = []
    explicit_mono_to_wall_offsets.extend(thread_mono_to_wall_offset_by_tid.values())
    explicit_mono_to_wall_offsets.extend(worker_span_mono_to_wall_offsets)

    global_mono_to_wall_offset = None
    global_mono_to_wall_offset_source = None

    if explicit_mono_to_wall_offsets:
        global_mono_to_wall_offset = median_int(explicit_mono_to_wall_offsets)
        global_mono_to_wall_offset_source = "explicit_thread_or_worker_span"
    elif req_metric_mono_to_wall_offsets:
        global_mono_to_wall_offset = median_int(req_metric_mono_to_wall_offsets)
        global_mono_to_wall_offset_source = "req_metrics_fallback"

    if global_mono_to_wall_offset is not None:
        print(
            f"Estimated mono->wall offset: {global_mono_to_wall_offset} ns "
            f"(source={global_mono_to_wall_offset_source})"
        )

    if explicit_mono_to_wall_offsets and req_metric_mono_to_wall_offsets:
        explicit_offset = median_int(explicit_mono_to_wall_offsets)
        req_metric_offset = median_int(req_metric_mono_to_wall_offsets)
        drift_ns = req_metric_offset - explicit_offset
        if abs(drift_ns) > 1_000_000:
            print(
                "  [warn] req_metrics 推断的 mono->wall offset 与显式样本偏差较大: "
                f"{drift_ns / 1e6:.3f} ms "
                f"(explicit={explicit_offset}, req_metrics={req_metric_offset})"
            )


    print("\n=== 正在计算 OS 调度/抢占延迟 (eBPF) ===")
    # 1. 预处理 eBPF 数据：按 TID 分组并按时间排序，方便快速查找
    # ebpf_map: { tid: [ {start, end, dur}, ... sorted by start ] }
    export_alignment_debug_trace(execute_model_span, ebpf_sched_latency_events, output_filename="debug_os_latency_align.json")
    ebpf_map = defaultdict(list)
    for ev in ebpf_sched_latency_events:
        # 提取 payload 里的字段，兼容性处理
        payload = ev.get('payload', ev) 
        tid = payload.get('tid')
        start = payload.get('start_ns')
        end = payload.get('end_ns')
        reason = payload.get('reason')
        
        if tid and start and end:
            ebpf_map[tid].append({
                'start': start,
                'end': end,
                'dur': end - start,
                'reason': reason,
            })

    # 对每个 TID 的事件按开始时间排序
    for tid in ebpf_map:
        ebpf_map[tid].sort(key=lambda x: x['start'])

    EVENTLOOP_OS_PID = 12000
    if eventloop_thread_tids:
        trace_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": EVENTLOOP_OS_PID,
            "args": {"name": "EventLoop OS Scheduling"},
        })
    for tid in sorted(eventloop_thread_tids):
        os_events = ebpf_map.get(tid, [])
        if not os_events:
            continue
        mono_to_wall_offset = thread_mono_to_wall_offset_by_tid.get(tid, global_mono_to_wall_offset)
        if mono_to_wall_offset is None:
            print(f"  [warn] eventloop tid={tid} 缺少 mono->wall offset，跳过 OS 轨道绘制")
            continue

        role_info = thread_roles_by_tid.get(tid, {})
        source_pid = role_info.get("pid")
        role_names = sorted(role_info.get("roles", []))
        track_name = f"eventloop tid={tid}"
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": EVENTLOOP_OS_PID,
            "tid": tid,
            "args": {"name": track_name},
        })
        for os_ev in os_events:
            start_wall = os_ev["start"] + mono_to_wall_offset
            end_wall = os_ev["end"] + mono_to_wall_offset
            if end_wall <= TRACE_TS_ORIGIN_NS:
                continue
            if start_wall < TRACE_TS_ORIGIN_NS:
                start_wall = TRACE_TS_ORIGIN_NS
            if end_wall <= start_wall:
                continue
            trace_events.append(create_perfetto_event(
                name="OS Runnable Wait",
                cat="eventloop_os",
                ph="X",
                ts=start_wall,
                dur=end_wall - start_wall,
                pid=EVENTLOOP_OS_PID,
                tid=tid,
                args={
                    "source_tid": tid,
                    "source_pid": source_pid,
                    "reason": os_ev.get("reason"),
                    "roles": ",".join(role_names),
                },
                cname="terrible",
            ))

    # 2. 结果容器：Key=req_id, Value=critical-path total_os_delay_ns
    req_latency_map = defaultdict(int)
    req_os_overlap_intervals = defaultdict(list)  # mono 时间轴，后续可转 wall 画图

    # 多卡场景下，一个请求属于某个 DP group；该 DP 内所有 worker 的 OS delay
    # 都可能阻塞该请求。先从 worker execute span 反推 dp->worker 与 req->window。
    worker_tids_by_group = defaultdict(set)
    req_windows_by_group = defaultdict(lambda: defaultdict(list))
    req_groups = defaultdict(set)

    def worker_group_key(payload, tid):
        ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
        dp_rank = to_int(ranks.get("dp"))
        if dp_rank is not None and dp_rank >= 0:
            return ("dp", dp_rank), dp_rank
        # 旧日志可能没有 dp 信息；此时保留原来的单 worker 语义。
        return ("worker_tid", tid), None

    for span in execute_model_span:
        payload = span.get('payload', span)
        tid = to_int(payload.get('tid'))
        span_start = to_ns(payload.get('start_ns'))
        span_end = to_ns(payload.get('end_ns'))
        req_ids = normalize_request_ids(payload.get('req_ids', []))

        if tid is None or span_start is None or span_end is None or span_end <= span_start or not req_ids:
            continue

        group_key, _dp_rank = worker_group_key(payload, tid)
        worker_tids_by_group[group_key].add(tid)
        for rid in req_ids:
            req_windows_by_group[group_key][rid].append((span_start, span_end))
            req_groups[rid].add(group_key)

    raw_req_os_intervals = defaultdict(list)
    for group_key, req_windows in req_windows_by_group.items():
        candidate_tids = sorted(tid for tid in worker_tids_by_group.get(group_key, set()) if tid in ebpf_map)
        if not candidate_tids:
            continue
        dp_rank = group_key[1] if group_key[0] == "dp" else None
        for rid, windows in req_windows.items():
            for window_start, window_end in merge_intervals(windows):
                for worker_tid in candidate_tids:
                    for os_ev in ebpf_map[worker_tid]:
                        if os_ev['start'] >= window_end:
                            break
                        if os_ev['end'] <= window_start:
                            continue

                        overlap_start = max(window_start, os_ev['start'])
                        overlap_end = min(window_end, os_ev['end'])
                        if overlap_end > overlap_start:
                            raw_req_os_intervals[rid].append({
                                "start_mono_ns": overlap_start,
                                "end_mono_ns": overlap_end,
                                "worker_tid": worker_tid,
                                "dp_rank": dp_rank,
                                "reason": os_ev.get("reason"),
                            })

    # 对每个请求按时间轴合并所有 worker 的 OS delay。这里算的是关键路径上的
    # wall-clock 等效阻塞时间，而不是把多个 worker 同时发生的 delay 直接相加。
    for rid, intervals in raw_req_os_intervals.items():
        merged = []
        for it in sorted(intervals, key=lambda x: (x["start_mono_ns"], x["end_mono_ns"])):
            start = it["start_mono_ns"]
            end = it["end_mono_ns"]
            if end <= start:
                continue
            if not merged or start > merged[-1]["end_mono_ns"]:
                merged.append({
                    "start_mono_ns": start,
                    "end_mono_ns": end,
                    "worker_tids": {it["worker_tid"]},
                    "dp_ranks": {it["dp_rank"]} if it.get("dp_rank") is not None else set(),
                    "reasons": {it["reason"]} if it.get("reason") else set(),
                })
            else:
                merged[-1]["end_mono_ns"] = max(merged[-1]["end_mono_ns"], end)
                merged[-1]["worker_tids"].add(it["worker_tid"])
                if it.get("dp_rank") is not None:
                    merged[-1]["dp_ranks"].add(it["dp_rank"])
                if it.get("reason"):
                    merged[-1]["reasons"].add(it["reason"])

        total_ns = 0
        for it in merged:
            total_ns += it["end_mono_ns"] - it["start_mono_ns"]
            req_os_overlap_intervals[rid].append({
                "start_mono_ns": it["start_mono_ns"],
                "end_mono_ns": it["end_mono_ns"],
                "worker_tids": sorted(it["worker_tids"]),
                "dp_ranks": sorted(it["dp_ranks"]),
                "reasons": sorted(it["reasons"]),
            })
        if total_ns > 0:
            req_latency_map[rid] = total_ns

    multi_group_reqs = [rid for rid, groups in req_groups.items() if len(groups) > 1]
    if multi_group_reqs:
        print(
            f"  [warn] {len(multi_group_reqs)} 个请求出现在多个 worker/dp group 中；"
            "OS delay 已按所有相关 group 的合并关键路径统计"
        )
    print(
        f"  Worker OS attribution groups: {len(worker_tids_by_group)} groups, "
        f"{sum(len(tids) for tids in worker_tids_by_group.values())} worker tids"
    )

    # 5. 打印结果
    print(f"共统计到 {len(req_latency_map)} 个请求受到 OS 调度影响：")
    
    # 按延迟从高到低排序
    sorted_reqs = sorted(req_latency_map.items(), key=lambda x: x[1], reverse=True)
    
    for rid, lat_ns in sorted_reqs:
        lat_ms = lat_ns / 1e6
        print(f"  - Request {rid}: {lat_ms:.3f} ms (OS Scheduling/Preemption Delay)")
            
    if not sorted_reqs:
        print("  未统计到调度延迟")

    # --- 6. 按请求聚合可视化：生命周期 + vLLM排队 + OS调度 ---
    print("\n=== Building Request Lifecycle/Queue/OS Timeline ===")
    REQUEST_PID = 11000
    trace_events.append({"name": "process_name", "ph": "M", "pid": REQUEST_PID, "args": {"name": "Request Interference"}})

    all_req_ids = (
        set(request_spans.keys())
        | set(req_coro_sched_ns.keys())
        | set(req_generate_task_sched_ns.keys())
        | set(req_generate_exec_ns.keys())
        | set(req_output_socket_sched_ns.keys())
        | set(req_output_socket_exec_ns.keys())
        | set(req_output_handler_sched_ns.keys())
        | set(req_output_handler_exec_ns.keys())
        | set(req_vllm_queue_ns.keys())
        | set(req_latency_map.keys())
        | set(req_stage_ns.keys())
        | set(req_dispatch_intervals_map.keys())
        | set(req_step_execution_intervals.keys())
    )
    req_os_intervals_wall = defaultdict(list)
    if global_mono_to_wall_offset is not None:
        for rid, intervals in req_os_overlap_intervals.items():
            for it in intervals:
                s_wall = it["start_mono_ns"] + global_mono_to_wall_offset
                e_wall = it["end_mono_ns"] + global_mono_to_wall_offset
                if e_wall > s_wall:
                    req_os_intervals_wall[rid].append({
                        "start_ns": s_wall,
                        "end_ns": e_wall,
                        "worker_tids": list(it.get("worker_tids") or []),
                        "dp_ranks": list(it.get("dp_ranks") or []),
                        "reasons": list(it.get("reasons") or []),
                    })
    if global_mono_to_wall_offset is None and req_os_overlap_intervals:
        print("  [warn] 无法估计 mono->wall 偏移，OS overlap 仅做统计，不绘制到请求生命周期轨道")

    phase_exec_name_map = {
        "Preprocess": "worker_preprocess",
        "Forward": "model_forward",
        "Postprocess": "postprocess",
        "Sample": "sampling",
        "GPUModelRunner sample_tokens": "sampling",
        "Bookkeep": "bookkeeping_sync",
        "Draft": "draft",
        "EPLB": "eplb",
        "ModelRunnerOutput": "model_runner_output",
    }
    compute_breakdown = {
        "version": 4,
        "tpot_denominator": "decode_dispatch_count",
        "wall_time_policy": "TTFT prefill_exec and TPOT exec phases use TP-aware wall-time attribution; pp_gap is measured from adjacent PP Step Execution windows within the same dispatch/batch after merging TP ranks in each PP stage.",
        "scheduling_wait_policy": "preempt_gap is classified only from explicit vLLM per-request preempt events; waiting behind another running batch remains normal_gap.",
        "requests": {},
        "summary": {},
        "skipped_requests": {
            "ttft_missing": [],
            "tpot_missing": [],
        },
        "phase_reentry_after_decode_requests": [],
    }
    ttft_component_totals_ns = defaultdict(int)
    ttft_vllm_queue_breakdown_totals_ns = defaultdict(int)
    ttft_prefill_exec_breakdown_totals_ns = defaultdict(int)
    ttft_req_count = 0
    tpot_component_totals_ns = defaultdict(int)
    tpot_req_count = 0
    tpot_total_decode_steps = 0
    tpot_per_req_values_ns = []

    for rid in sorted(all_req_ids):
        dispatches = sorted(req_dispatch_intervals_map.get(rid, []), key=lambda x: (x["start_ns"], x["end_ns"]))
        prefill_dispatches = [d for d in dispatches if d.get("phase") == "prefill"]
        decode_dispatches = [d for d in dispatches if d.get("phase") == "decode"]
        first_decode_index = next((idx for idx, d in enumerate(dispatches) if d.get("phase") == "decode"), None)
        if first_decode_index is None:
            initial_prefill_dispatches = prefill_dispatches
            phase_reentry_after_decode = False
        else:
            initial_prefill_dispatches = [d for d in dispatches[:first_decode_index] if d.get("phase") == "prefill"]
            phase_reentry_after_decode = any(d.get("phase") == "prefill" for d in dispatches[first_decode_index + 1:])
        if phase_reentry_after_decode:
            compute_breakdown["phase_reentry_after_decode_requests"].append(rid)

        dispatch_phase_by_start = {}
        for d in dispatches:
            dispatch_phase_by_start.setdefault(d["start_ns"], d.get("phase", "unknown"))

        queue_intervals = []
        for q in sorted(queue_intervals_map.get(rid, []), key=lambda x: (x["start_ns"], x["end_ns"])):
            q_copy = dict(q)
            q_copy["phase"] = dispatch_phase_by_start.get(q_copy.get("end_ns"))
            queue_intervals.append(q_copy)
        request_phase_intervals = req_worker_phase_intervals.get(rid, {})
        request_step_execution_intervals = req_step_execution_intervals.get(rid, [])

        ttft_entry = None
        generate_setup_intervals = [
            s for s in req_stage_intervals.get(rid, [])
            if s.get("stage") == "async_llm.generate_wrapper_setup"
        ]
        generate_setup_start_ns = None
        if generate_setup_intervals:
            generate_setup_start_ns = min(to_ns(s.get("start_ns")) for s in generate_setup_intervals if to_ns(s.get("start_ns")) is not None)
        ttft_start_ns = generate_setup_start_ns or to_ns(req_generate_start_map.get(rid))
        generate_consume_intervals = sorted(
            [
                it for it in req_generate_exec_intervals.get(rid, [])
                if it.get("task_kind") == "generate_consume"
            ],
            key=lambda x: (x["start_ns"], x["end_ns"]),
        )
        first_generate_consume_end_ns = None
        if generate_consume_intervals:
            first_generate_consume_end_ns = to_ns(generate_consume_intervals[0].get("end_ns"))

        if ttft_start_ns is not None and initial_prefill_dispatches and first_generate_consume_end_ns is not None:
            ttft_end_ns = first_generate_consume_end_ns
            if ttft_end_ns > ttft_start_ns:
                enginecore_queue_intervals = [
                    s for s in req_stage_intervals.get(rid, [])
                    if s.get("stage") == "engine_core.input_queue_wait_after_preprocess"
                ]
                enginecore_queue_ns, _enginecore_queue_pairs = sum_clipped_interval_items(
                    enginecore_queue_intervals,
                    ttft_start_ns,
                    ttft_end_ns,
                )
                enginecore_queue_start_ns = None
                enginecore_queue_end_ns = None
                for s in sorted(enginecore_queue_intervals, key=lambda x: (x["start_ns"], x["end_ns"])):
                    span = clip_interval(s.get("start_ns"), s.get("end_ns"), ttft_start_ns, ttft_end_ns)
                    if not span:
                        continue
                    if enginecore_queue_start_ns is None:
                        enginecore_queue_start_ns = span[0]
                    enginecore_queue_end_ns = span[1]

                initial_prefill_dispatches_sorted = sorted(
                    initial_prefill_dispatches,
                    key=lambda x: (x["start_ns"], x["end_ns"]),
                )
                first_initial_prefill_start_ns = to_ns(initial_prefill_dispatches_sorted[0]["start_ns"])
                last_initial_prefill_end_ns = max(d["end_ns"] for d in initial_prefill_dispatches_sorted)
                if first_initial_prefill_start_ns is None or last_initial_prefill_end_ns is None:
                    compute_breakdown["skipped_requests"]["ttft_missing"].append(rid)
                else:
                    preprocess_boundary_ns = enginecore_queue_start_ns
                    if preprocess_boundary_ns is None:
                        enqueue_ns = to_ns(req_enqueue_map.get(rid))
                        candidates = [val for val in [enqueue_ns, first_initial_prefill_start_ns, ttft_end_ns] if val is not None]
                        preprocess_boundary_ns = min(candidates) if candidates else ttft_start_ns
                    preprocess_boundary_ns = max(ttft_start_ns, min(preprocess_boundary_ns, ttft_end_ns))
                    preprocess_ns = max(0, preprocess_boundary_ns - ttft_start_ns)

                    if enginecore_queue_end_ns is None:
                        enginecore_queue_end_ns = preprocess_boundary_ns
                    enginecore_queue_end_ns = max(preprocess_boundary_ns, min(enginecore_queue_end_ns, first_initial_prefill_start_ns))

                    vllm_queue_before_first_prefill_ns = max(0, first_initial_prefill_start_ns - enginecore_queue_end_ns)
                    prefill_schedule_gap_ns = 0
                    for prev_dispatch, next_dispatch in zip(initial_prefill_dispatches_sorted, initial_prefill_dispatches_sorted[1:]):
                        prev_end_ns = to_ns(prev_dispatch.get("end_ns"))
                        next_start_ns = to_ns(next_dispatch.get("start_ns"))
                        if prev_end_ns is None or next_start_ns is None:
                            continue
                        prefill_schedule_gap_ns += max(0, next_start_ns - prev_end_ns)
                    vllm_queue_ns = vllm_queue_before_first_prefill_ns + prefill_schedule_gap_ns

                    prefill_exec_ns, prefill_exec_pairs = sum_clipped_interval_items(
                        initial_prefill_dispatches_sorted,
                        first_initial_prefill_start_ns,
                        last_initial_prefill_end_ns,
                        merge_overlaps=True,
                    )

                    ttft_total_ns = ttft_end_ns - ttft_start_ns
                    postprocess_transport_ns = max(0, ttft_end_ns - last_initial_prefill_end_ns)
                    ttft_other_gap_ns = max(
                        0,
                        ttft_total_ns - preprocess_ns - enginecore_queue_ns - vllm_queue_ns - prefill_exec_ns - postprocess_transport_ns,
                    )

                    prefill_pp_gap_breakdown = build_pp_gap_dispatch_breakdown(
                        initial_prefill_dispatches_sorted,
                        request_phase_intervals,
                        request_step_execution_intervals,
                        phase_component_map=phase_exec_name_map,
                    )
                    prefill_exec_phase_ns = defaultdict(int)
                    for key, value in prefill_pp_gap_breakdown.get("wall_components_ns", {}).items():
                        prefill_exec_phase_ns[key] += value
                    if "other_compute" in prefill_exec_phase_ns:
                        prefill_exec_phase_ns["other_prefill_exec"] += prefill_exec_phase_ns.pop("other_compute")
                    if "pipeline_gap" in prefill_exec_phase_ns:
                        prefill_exec_phase_ns["other_prefill_exec"] += prefill_exec_phase_ns.pop("pipeline_gap")

                    ttft_components_ns = {
                        "preprocess": preprocess_ns,
                        "enginecore_queue": enginecore_queue_ns,
                        "vllm_queue": vllm_queue_ns,
                        "prefill_exec": prefill_exec_ns,
                        "postprocess_transport": postprocess_transport_ns,
                        "other_gap": ttft_other_gap_ns,
                    }
                    ttft_entry = {
                        "window_ns": ttft_total_ns,
                        "window_ms": ns_to_ms(ttft_total_ns),
                        "start_ns": ttft_start_ns,
                        "end_ns": ttft_end_ns,
                        "initial_prefill_end_ns": last_initial_prefill_end_ns,
                        "initial_prefill_end_ms": ns_to_ms(last_initial_prefill_end_ns),
                        "ttft_start_rule": "generate_wrapper_setup_start" if generate_setup_start_ns is not None else "coroutine_start",
                        "prefill_dispatch_count": len(initial_prefill_dispatches_sorted),
                        "ttft_end_rule": "first_generate_consume_end",
                        "phase_reentry_after_decode": phase_reentry_after_decode,
                        "components_ns": ttft_components_ns,
                        "components_ms": normalize_component_map_ms(ttft_components_ns),
                        "vllm_queue_breakdown_ns": {
                            "before_first_prefill": vllm_queue_before_first_prefill_ns,
                            "between_prefills": prefill_schedule_gap_ns,
                        },
                        "vllm_queue_breakdown_ms": {
                            "before_first_prefill": ns_to_ms(vllm_queue_before_first_prefill_ns),
                            "between_prefills": ns_to_ms(prefill_schedule_gap_ns),
                        },
                        "prefill_exec_phase_ns": dict(prefill_exec_phase_ns),
                        "prefill_exec_phase_ms": normalize_component_map_ms(prefill_exec_phase_ns),
                        "pp_gap_breakdown": prefill_pp_gap_breakdown,
                    }
                    ttft_req_count += 1
                    for key, val in ttft_components_ns.items():
                        ttft_component_totals_ns[key] += val
                    ttft_vllm_queue_breakdown_totals_ns["before_first_prefill"] += vllm_queue_before_first_prefill_ns
                    ttft_vllm_queue_breakdown_totals_ns["between_prefills"] += prefill_schedule_gap_ns
                    for key, val in prefill_exec_phase_ns.items():
                        ttft_prefill_exec_breakdown_totals_ns[key] += val
            else:
                compute_breakdown["skipped_requests"]["ttft_missing"].append(rid)
        else:
            compute_breakdown["skipped_requests"]["ttft_missing"].append(rid)

        tpot_entry = None
        if ttft_entry and decode_dispatches:
            tpot_window_start_ns = to_ns(ttft_entry.get("initial_prefill_end_ns"))
            last_generate_consume_end_ns = None
            if generate_consume_intervals:
                last_generate_consume_end_ns = max(
                    to_ns(it.get("end_ns"))
                    for it in generate_consume_intervals
                    if to_ns(it.get("end_ns")) is not None
                )

            post_start_dispatches = [d for d in dispatches if d["end_ns"] > tpot_window_start_ns]
            post_start_prefill_dispatches = [d for d in post_start_dispatches if d.get("phase") == "prefill"]
            post_start_decode_dispatches = [d for d in post_start_dispatches if d.get("phase") == "decode"]
            decode_steps = len(post_start_decode_dispatches)
            last_decode_end_ns = max((d["end_ns"] for d in post_start_decode_dispatches), default=None)
            tpot_window_end_ns = last_generate_consume_end_ns
            if (
                tpot_window_start_ns is not None
                and decode_steps > 0
                and last_decode_end_ns is not None
                and tpot_window_end_ns is not None
                and last_decode_end_ns > tpot_window_start_ns
                and tpot_window_end_ns >= last_decode_end_ns
            ):
                tpot_exec_end_ns = last_decode_end_ns
                tpot_exec_dispatches = []
                for d in post_start_dispatches:
                    span = clip_interval(d.get("start_ns"), d.get("end_ns"), tpot_window_start_ns, tpot_exec_end_ns)
                    if not span:
                        continue
                    d_copy = dict(d)
                    d_copy["start_ns"], d_copy["end_ns"] = span
                    tpot_exec_dispatches.append(d_copy)
                tpot_exec_ns, tpot_exec_pairs = sum_clipped_interval_items(
                    tpot_exec_dispatches,
                    tpot_window_start_ns,
                    tpot_exec_end_ns,
                    merge_overlaps=True,
                )
                tpot_recompute_prefill_exec_ns, _ = sum_clipped_interval_items(
                    [d for d in tpot_exec_dispatches if d.get("phase") == "prefill"],
                    tpot_window_start_ns,
                    tpot_exec_end_ns,
                    merge_overlaps=True,
                )
                tpot_decode_exec_ns, _ = sum_clipped_interval_items(
                    [d for d in tpot_exec_dispatches if d.get("phase") == "decode"],
                    tpot_window_start_ns,
                    tpot_exec_end_ns,
                    merge_overlaps=True,
                )

                post_start_queue_intervals = [
                    q for q in queue_intervals
                    if q.get("phase") in ("prefill", "decode")
                ]
                scheduling_wait_normal_ns = 0
                scheduling_wait_preempt_ns = 0
                scheduling_wait_preempt_events = []
                tpot_queue_pairs = []
                for q in post_start_queue_intervals:
                    span = clip_interval(q.get("start_ns"), q.get("end_ns"), tpot_window_start_ns, tpot_exec_end_ns)
                    if not span:
                        continue
                    start_ns, end_ns = span
                    tpot_queue_pairs.append(span)
                    preempt_hits = [
                        ts for ts in q.get("preempt_ts_ns", [])
                        if start_ns <= ts <= end_ns
                    ]
                    if q.get("gap_kind") == "preempt_wait" and preempt_hits:
                        scheduling_wait_preempt_ns += end_ns - start_ns
                        scheduling_wait_preempt_events.extend(preempt_hits)
                    else:
                        scheduling_wait_normal_ns += end_ns - start_ns

                tpot_pp_gap_breakdown = build_pp_gap_dispatch_breakdown(
                    tpot_exec_dispatches,
                    request_phase_intervals,
                    request_step_execution_intervals,
                    phase_component_map=phase_exec_name_map,
                )
                tpot_exec_phase_ns = defaultdict(int)
                for key, value in tpot_pp_gap_breakdown.get("wall_components_ns", {}).items():
                    tpot_exec_phase_ns[key] += value
                if "other_compute" in tpot_exec_phase_ns:
                    tpot_exec_phase_ns["other_exec"] += tpot_exec_phase_ns.pop("other_compute")
                if "pipeline_gap" in tpot_exec_phase_ns:
                    tpot_exec_phase_ns["other_exec"] += tpot_exec_phase_ns.pop("pipeline_gap")

                tail_transport_ns = max(0, tpot_window_end_ns - last_decode_end_ns)
                tpot_total_ns = tpot_window_end_ns - tpot_window_start_ns
                tpot_other_gap_ns = uncovered_interval_ns(
                    tpot_window_start_ns,
                    tpot_window_end_ns,
                    tpot_queue_pairs + tpot_exec_pairs + [(last_decode_end_ns, tpot_window_end_ns)],
                )
                scheduling_wait_total_ns = scheduling_wait_normal_ns + scheduling_wait_preempt_ns
                scheduling_wait_normal_pct = (
                    scheduling_wait_normal_ns * 100.0 / scheduling_wait_total_ns
                    if scheduling_wait_total_ns > 0 else 0.0
                )
                scheduling_wait_preempt_pct = (
                    scheduling_wait_preempt_ns * 100.0 / scheduling_wait_total_ns
                    if scheduling_wait_total_ns > 0 else 0.0
                )
                tpot_components_ns = {
                    "vllm_scheduling_wait": scheduling_wait_total_ns,
                    "worker_preprocess": tpot_exec_phase_ns.get("worker_preprocess", 0),
                    "model_forward": tpot_exec_phase_ns.get("model_forward", 0),
                    "postprocess": tpot_exec_phase_ns.get("postprocess", 0),
                    "sampling": tpot_exec_phase_ns.get("sampling", 0),
                    "bookkeeping_sync": tpot_exec_phase_ns.get("bookkeeping_sync", 0),
                    "pp_gap": tpot_exec_phase_ns.get("pp_gap", 0),
                    "other_exec": tpot_exec_phase_ns.get("other_exec", 0),
                    "tail_transport": tail_transport_ns,
                    "other_gap": tpot_other_gap_ns,
                }
                tpot_components_ms_per_token = {
                    key: round(val / decode_steps / 1e6, 6)
                    for key, val in tpot_components_ns.items()
                }
                tpot_entry = {
                    "window_ns": tpot_total_ns,
                    "window_ms": ns_to_ms(tpot_total_ns),
                    "start_ns": tpot_window_start_ns,
                    "end_ns": tpot_window_end_ns,
                    "decode_dispatch_count": decode_steps,
                    "tpot_start_rule": "initial_prefill_end",
                    "tpot_end_rule": "last_generate_consume_end",
                    "last_decode_end_ns": last_decode_end_ns,
                    "last_decode_end_ms": ns_to_ms(last_decode_end_ns),
                    "exec_total_ns": tpot_exec_ns,
                    "exec_total_ms": ns_to_ms(tpot_exec_ns),
                    "recompute_prefill_exec_ns": tpot_recompute_prefill_exec_ns,
                    "recompute_prefill_exec_ms": ns_to_ms(tpot_recompute_prefill_exec_ns),
                    "decode_exec_ns": tpot_decode_exec_ns,
                    "decode_exec_ms": ns_to_ms(tpot_decode_exec_ns),
                    "scheduling_wait_total_ns": scheduling_wait_total_ns,
                    "scheduling_wait_total_ms": ns_to_ms(scheduling_wait_total_ns),
                    "tail_transport_ns": tail_transport_ns,
                    "tail_transport_ms": ns_to_ms(tail_transport_ns),
                    "components_ns_total": tpot_components_ns,
                    "components_ms_total": normalize_component_map_ms(tpot_components_ns),
                    "components_ms_per_token": tpot_components_ms_per_token,
                    "exec_phase_ns": dict(tpot_exec_phase_ns),
                    "exec_phase_ms": normalize_component_map_ms(tpot_exec_phase_ns),
                    "pp_gap_breakdown": tpot_pp_gap_breakdown,
                    "scheduling_wait_breakdown_ns": {
                        "normal_gap": scheduling_wait_normal_ns,
                        "preempt_gap": scheduling_wait_preempt_ns,
                        "total": scheduling_wait_total_ns,
                    },
                    "scheduling_wait_breakdown_ms": {
                        "normal_gap": ns_to_ms(scheduling_wait_normal_ns),
                        "preempt_gap": ns_to_ms(scheduling_wait_preempt_ns),
                        "total": ns_to_ms(scheduling_wait_total_ns),
                    },
                    "scheduling_wait_breakdown_ratio": {
                        "normal_gap_pct": round(scheduling_wait_normal_pct, 6),
                        "preempt_gap_pct": round(scheduling_wait_preempt_pct, 6),
                    },
                    "scheduling_wait_classification": {
                        "policy": "explicit_req_preempt_events_only",
                        "preempt_event_count": len(scheduling_wait_preempt_events),
                        "preempt_ts_ns": scheduling_wait_preempt_events,
                    },
                    "avg_ms_per_token": round(tpot_total_ns / decode_steps / 1e6, 6),
                    "denominator_kind": "decode_dispatch_count",
                    "phase_reentry_after_decode": phase_reentry_after_decode,
                }
                tpot_req_count += 1
                tpot_total_decode_steps += decode_steps
                tpot_per_req_values_ns.append(tpot_total_ns / decode_steps)
                for key, val in tpot_components_ns.items():
                    tpot_component_totals_ns[key] += val
            else:
                compute_breakdown["skipped_requests"]["tpot_missing"].append(rid)
        else:
            compute_breakdown["skipped_requests"]["tpot_missing"].append(rid)

        compute_breakdown["requests"][rid] = {
            "request_name": req_name_map.get(rid, rid),
            "ttft": ttft_entry,
            "tpot": tpot_entry,
        }

    ttft_avg_components_ms = {}
    ttft_vllm_queue_breakdown_avg_ms = {}
    ttft_prefill_exec_breakdown_avg_ms = {}
    if ttft_req_count > 0:
        ttft_avg_components_ms = {
            key: round(ttft_component_totals_ns.get(key, 0) / ttft_req_count / 1e6, 6)
            for key in ["preprocess", "enginecore_queue", "vllm_queue", "prefill_exec", "postprocess_transport", "other_gap"]
        }
        ttft_vllm_queue_breakdown_avg_ms = {
            key: round(ttft_vllm_queue_breakdown_totals_ns.get(key, 0) / ttft_req_count / 1e6, 6)
            for key in ["before_first_prefill", "between_prefills"]
        }
        ttft_prefill_exec_keys = [
            "worker_preprocess",
            "model_forward",
            "postprocess",
            "sampling",
            "bookkeeping_sync",
            "draft",
            "eplb",
            "model_runner_output",
            "pp_gap",
            "other_prefill_exec",
        ]
        ttft_prefill_exec_keys.extend(
            sorted(set(ttft_prefill_exec_breakdown_totals_ns.keys()) - set(ttft_prefill_exec_keys))
        )
        ttft_prefill_exec_breakdown_avg_ms = {
            key: round(ttft_prefill_exec_breakdown_totals_ns.get(key, 0) / ttft_req_count / 1e6, 6)
            for key in ttft_prefill_exec_keys
            if ttft_prefill_exec_breakdown_totals_ns.get(key, 0) > 0 or key == "pp_gap"
        }
    tpot_avg_components_ms = {}
    if tpot_total_decode_steps > 0:
        tpot_avg_components_ms = {
            key: round(tpot_component_totals_ns.get(key, 0) / tpot_total_decode_steps / 1e6, 6)
            for key in [
                "vllm_scheduling_wait",
                "worker_preprocess",
                "model_forward",
                "postprocess",
                "sampling",
                "bookkeeping_sync",
                "draft",
                "eplb",
                "model_runner_output",
                "pp_gap",
                "other_exec",
                "tail_transport",
                "other_gap",
            ]
        }

    compute_breakdown["summary"] = {
        "ttft_request_count": ttft_req_count,
        "ttft_avg_components_ms": ttft_avg_components_ms,
        "ttft_vllm_queue_breakdown_avg_ms": ttft_vllm_queue_breakdown_avg_ms,
        "ttft_prefill_exec_breakdown_avg_ms": ttft_prefill_exec_breakdown_avg_ms,
        "ttft_avg_ms": round(sum(ttft_avg_components_ms.values()), 6) if ttft_avg_components_ms else None,
        "tpot_request_count": tpot_req_count,
        "tpot_total_decode_steps": tpot_total_decode_steps,
        "tpot_avg_components_ms_per_token": tpot_avg_components_ms,
        "tpot_avg_ms_per_token": round(sum(tpot_avg_components_ms.values()), 6) if tpot_avg_components_ms else None,
        "tpot_mean_request_ms_per_token": round(sum(tpot_per_req_values_ns) / len(tpot_per_req_values_ns) / 1e6, 6) if tpot_per_req_values_ns else None,
        "phase_reentry_after_decode_request_count": len(compute_breakdown["phase_reentry_after_decode_requests"]),
    }

    compute_json_path = derive_sidecar_path(output_file, ".compute_breakdown.json")
    with open(compute_json_path, "w", encoding="utf-8") as f:
        json.dump(to_plain_data(compute_breakdown), f, ensure_ascii=False, indent=2)

    print("=== Compute-Side Breakdown Summary ===")
    if ttft_avg_components_ms:
        print(
            f"TTFT Avg ({ttft_req_count} reqs, compute-side): "
            f"{compute_breakdown['summary']['ttft_avg_ms']:.3f} ms"
        )
        for key in ["preprocess", "enginecore_queue", "vllm_queue", "prefill_exec", "postprocess_transport", "other_gap"]:
            print(f"  - {key}: {ttft_avg_components_ms.get(key, 0.0):.3f} ms")
        print(
            "  - vllm_queue_breakdown: "
            f"before_first_prefill={ttft_vllm_queue_breakdown_avg_ms.get('before_first_prefill', 0.0):.3f} ms, "
            f"between_prefills={ttft_vllm_queue_breakdown_avg_ms.get('between_prefills', 0.0):.3f} ms"
        )
        if ttft_prefill_exec_breakdown_avg_ms:
            detail = ", ".join(
                f"{key}={ttft_prefill_exec_breakdown_avg_ms.get(key, 0.0):.3f} ms"
                for key in [
                    "worker_preprocess",
                    "model_forward",
                    "postprocess",
                    "sampling",
                    "bookkeeping_sync",
                    "pp_gap",
                    "other_prefill_exec",
                ]
                if key in ttft_prefill_exec_breakdown_avg_ms
            )
            print(f"  - prefill_exec_breakdown: {detail}")
    else:
        print("TTFT Avg (compute-side): 无可用请求")

    if tpot_avg_components_ms:
        print(
            f"TPOT Avg ({tpot_req_count} reqs, {tpot_total_decode_steps} decode steps, compute-side): "
            f"{compute_breakdown['summary']['tpot_avg_ms_per_token']:.3f} ms/token "
            f"(denominator=decode_dispatch_count)"
        )
        for key in [
            "vllm_scheduling_wait",
            "worker_preprocess",
            "model_forward",
            "postprocess",
            "sampling",
            "bookkeeping_sync",
            "draft",
            "eplb",
            "model_runner_output",
            "pp_gap",
            "other_exec",
            "tail_transport",
            "other_gap",
        ]:
            print(f"  - {key}: {tpot_avg_components_ms.get(key, 0.0):.3f} ms/token")
    else:
        print("TPOT Avg (compute-side): 无可用 decode 请求")

    if compute_breakdown["phase_reentry_after_decode_requests"]:
        print(
            f"[warn] observed {len(compute_breakdown['phase_reentry_after_decode_requests'])} requests with prefill re-entry after decode; "
            "TTFT keeps only the initial prefill segment, and the later recompute-prefill cost is attributed to TPOT."
        )
    print(f"Saved compute breakdown JSON: {compute_json_path}")

    def req_sort_key(rid):
        span = request_spans.get(rid)
        if span and span.get("start_ns"):
            return span["start_ns"]
        q = queue_intervals_map.get(rid)
        if q:
            return q[0]["start_ns"]
        tq = req_generate_task_sched_intervals.get(rid)
        if tq:
            return tq[0]["start_ns"]
        osq = req_output_socket_sched_intervals.get(rid)
        if osq:
            return osq[0]["start_ns"]
        o = req_os_intervals_wall.get(rid)
        if o:
            return o[0]["start_ns"]
        s = req_stage_intervals.get(rid)
        if s:
            return min(it["start_ns"] for it in s)
        oh = req_output_handler_sched_intervals.get(rid)
        if oh:
            return oh[0]["start_ns"]
        gx = req_generate_exec_intervals.get(rid)
        if gx:
            return gx[0]["start_ns"]
        sx = req_output_socket_exec_intervals.get(rid)
        if sx:
            return sx[0]["start_ns"]
        ox = req_output_handler_exec_intervals.get(rid)
        if ox:
            return ox[0]["start_ns"]
        return float("inf")

    sorted_req_ids = sorted(all_req_ids, key=req_sort_key)

    per_request_dir = derive_sidecar_path(output_file, ".compute_breakdown_requests")
    os.makedirs(per_request_dir, exist_ok=True)
    request_rows = []
    for req_index, rid in enumerate(sorted_req_ids, start=1):
        req_entry = compute_breakdown["requests"].get(rid, {})
        ttft_entry = req_entry.get("ttft")
        tpot_entry = req_entry.get("tpot")
        request_name = req_entry.get("request_name", rid)
        file_name = f"{req_index:04d}_{sanitize_filename(request_name)[:80] or sanitize_filename(rid)[:80]}.svg"
        svg_path = os.path.join(per_request_dir, file_name)
        render_request_breakdown_svg(svg_path, rid, request_name, ttft_entry, tpot_entry)
        request_rows.append({
            "request_name": request_name,
            "ttft_ms": f"{float(ttft_entry['window_ms']):.3f}" if ttft_entry else "-",
            "tpot_ms": f"{float(tpot_entry['avg_ms_per_token']):.3f}" if tpot_entry else "-",
            "reentry": str(bool((ttft_entry and ttft_entry.get('phase_reentry_after_decode')) or (tpot_entry and tpot_entry.get('phase_reentry_after_decode')))),
            "file_name": file_name,
        })

    per_request_index_path = os.path.join(per_request_dir, "index.html")
    render_request_breakdown_index(per_request_index_path, request_rows)
    print(f"Saved per-request compute charts: {per_request_dir}")
    print(f"Saved per-request chart index:  {per_request_index_path}")

    def _request_pp_gap_intervals(rid):
        req_entry = compute_breakdown["requests"].get(rid, {})
        intervals = []
        for section_name in ("ttft", "tpot"):
            section = req_entry.get(section_name) or {}
            breakdown = section.get("pp_gap_breakdown") or {}
            for dispatch in breakdown.get("dispatches", []):
                for gap in dispatch.get("pp_gap_intervals", []):
                    item = dict(gap)
                    item["section"] = section_name
                    item["dispatch_key"] = dispatch.get("dispatch_key")
                    item["phase"] = dispatch.get("phase")
                    intervals.append(item)
        return sorted(intervals, key=lambda item: (item.get("start_ns") or 0, item.get("end_ns") or 0))

    def _request_pp_gap_wall_ns(rid):
        pairs = []
        for item in _request_pp_gap_intervals(rid):
            span = clip_interval(item.get("start_ns"), item.get("end_ns"))
            if span:
                pairs.append(span)
        return sum_interval_pairs(merge_intervals(pairs))

    for req_index, rid in enumerate(sorted_req_ids, start=1):
        lane_tids = build_request_lane_tids(req_index)
        dispatch_intervals = list(req_dispatch_intervals_map.get(rid, []))
        dispatch_display_intervals = build_raw_dispatch_display_intervals(dispatch_intervals)
        dispatch_lane_indices, dispatch_lane_count = assign_non_overlapping_lanes(dispatch_display_intervals)
        pp_gap_intervals = _request_pp_gap_intervals(rid)
        pp_gap_lane_indices, pp_gap_lane_count = assign_non_overlapping_lanes(pp_gap_intervals)
        lane_names = {
            "label": f"{rid[:12]}",
            "lifecycle": f"{rid[:12]} | lifecycle",
            "vllm_queue": f"{rid[:12]} | vllm queue",
            "coro_queue": f"{rid[:12]} | coroutine queue",
            "output_socket_queue": f"{rid[:12]} | output socket queue",
            "output_queue": f"{rid[:12]} | output queue",
            "dispatch": f"{rid[:12]} | dispatch",
            "generate_exec": f"{rid[:12]} | generate exec",
            "output_socket_exec": f"{rid[:12]} | output socket exec",
            "output_exec": f"{rid[:12]} | output exec",
            "os": f"{rid[:12]} | os",
            "stage": f"{rid[:12]} | stage",
            "task": f"{rid[:12]} | task",
            "pp_gap": f"{rid[:12]} | pp gap",
        }
        if dispatch_lane_count > 1:
            lane_names["dispatch"] = f"{rid[:12]} | dispatch 1"
            for lane_idx in range(1, dispatch_lane_count):
                key = f"dispatch_{lane_idx}"
                lane_tids[key] = req_index * 100 + 50 + lane_idx
                lane_names[key] = f"{rid[:12]} | dispatch {lane_idx + 1}"
        if pp_gap_lane_count > 1:
            lane_names["pp_gap"] = f"{rid[:12]} | pp gap 1"
            for lane_idx in range(1, pp_gap_lane_count):
                key = f"pp_gap_{lane_idx}"
                lane_tids[key] = req_index * 100 + 70 + lane_idx
                lane_names[key] = f"{rid[:12]} | pp gap {lane_idx + 1}"
        for lane_key, lane_tid in lane_tids.items():
            trace_events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": REQUEST_PID,
                "tid": lane_tid,
                "args": {"name": lane_names[lane_key]},
            })

        coro_sched_queue_ms = req_coro_sched_ns.get(rid, 0) / 1e6
        generate_task_sched_queue_ms = req_generate_task_sched_ns.get(rid, 0) / 1e6
        generate_exec_ms = req_generate_exec_ns.get(rid, 0) / 1e6
        output_socket_sched_queue_ms = req_output_socket_sched_ns.get(rid, 0) / 1e6
        output_socket_exec_ms = req_output_socket_exec_ns.get(rid, 0) / 1e6
        output_handler_sched_queue_ms = req_output_handler_sched_ns.get(rid, 0) / 1e6
        output_handler_exec_ms = req_output_handler_exec_ns.get(rid, 0) / 1e6
        vllm_queue_ms = req_vllm_queue_ns.get(rid, 0) / 1e6
        pp_gap_ms = _request_pp_gap_wall_ns(rid) / 1e6
        vllm_queue_from_enqueue_ms = req_vllm_queue_from_enqueue_ns.get(rid, 0) / 1e6
        vllm_queue_from_step_ready_ms = req_vllm_queue_from_step_ready_ns.get(rid, 0) / 1e6
        os_delay_ms = req_latency_map.get(rid, 0) / 1e6
        engine_input_queue_wait_ms = req_stage_ns.get(rid, {}).get(
            "engine_core.input_queue_wait_after_preprocess", 0) / 1e6
        prefill_exec_ms = req_dispatch_phase_ns.get(rid, {}).get("prefill", 0) / 1e6
        decode_exec_ms = req_dispatch_phase_ns.get(rid, {}).get("decode", 0) / 1e6
        unknown_exec_ms = req_dispatch_phase_ns.get(rid, {}).get("unknown", 0) / 1e6

        span = request_spans.get(rid)
        if span and span.get("end_ns", 0) > span.get("start_ns", 0):
            trace_events.append(create_perfetto_event(
                name=f"Req {rid[:8]} Lifecycle",
                cat="request_lifecycle",
                ph="X",
                ts=span["start_ns"],
                dur=span["end_ns"] - span["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids["lifecycle"],
                args={
                    "request_id": rid,
                    "request_name": req_name_map.get(rid, rid),
                    "coro_sched_queue_ms": round(coro_sched_queue_ms, 3),
                    "generate_task_sched_queue_ms": round(generate_task_sched_queue_ms, 3),
                    "generate_exec_ms": round(generate_exec_ms, 3),
                    "output_socket_sched_queue_ms": round(output_socket_sched_queue_ms, 3),
                    "output_socket_exec_ms": round(output_socket_exec_ms, 3),
                    "output_handler_sched_queue_ms": round(output_handler_sched_queue_ms, 3),
                    "output_handler_exec_ms": round(output_handler_exec_ms, 3),
                    "vllm_queue_ms": round(vllm_queue_ms, 3),
                    "pp_gap_ms": round(pp_gap_ms, 3),
                    "vllm_queue_from_enqueue_ms": round(vllm_queue_from_enqueue_ms, 3),
                    "vllm_queue_from_step_ready_ms": round(vllm_queue_from_step_ready_ms, 3),
                    "engine_input_queue_wait_ms": round(engine_input_queue_wait_ms, 3),
                    "os_sched_delay_ms": round(os_delay_ms, 3),
                    "prefill_exec_ms": round(prefill_exec_ms, 3),
                    "decode_exec_ms": round(decode_exec_ms, 3),
                    "unknown_exec_ms": round(unknown_exec_ms, 3),
                },
                cname="good",
            ))

        for dispatch_idx, d in enumerate(dispatch_display_intervals):
            if d["end_ns"] <= d["start_ns"]:
                continue
            dispatch_lane_idx = dispatch_lane_indices[dispatch_idx] if dispatch_idx < len(dispatch_lane_indices) else 0
            dispatch_lane_key = "dispatch" if dispatch_lane_idx == 0 else f"dispatch_{dispatch_lane_idx}"
            phase = d.get("phase", "unknown")
            if phase == "prefill":
                d_name = "Prefill Dispatch"
                d_cat = "request_dispatch_prefill"
                d_color = "olive"
            elif phase == "decode":
                d_name = "Decode Dispatch"
                d_cat = "request_dispatch_decode"
                d_color = "rail_response"
            else:
                d_name = "Unknown Dispatch"
                d_cat = "request_dispatch_unknown"
                d_color = "background"
            trace_events.append(create_perfetto_event(
                name=d_name,
                cat=d_cat,
                ph="X",
                ts=d["start_ns"],
                dur=d["end_ns"] - d["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids.get(dispatch_lane_key, lane_tids["dispatch"]),
                args={
                    "request_id": rid,
                    "phase": phase,
                    "phase_reason": d.get("phase_reason"),
                    "decode_step": d.get("decode_step"),
                    "dispatch_key": d.get("dispatch_key"),
                    "dispatch_keys": d.get("dispatch_keys"),
                    "dispatch_key_count": d.get("dispatch_key_count"),
                    "step_count": d.get("step_count"),
                    "batch_req_count": len(d.get("batch_req_ids") or []),
                    "duration_ms": round((d["end_ns"] - d["start_ns"]) / 1e6, 3),
                    "start_source": d.get("start_source"),
                    "end_source": d.get("end_source"),
                    "scheduler_ready_lag_ms": (
                        round((d["scheduler_ready_ns"] - d["worker_end_ns"]) / 1e6, 3)
                        if d.get("scheduler_ready_ns") is not None and d.get("worker_end_ns") is not None
                        else None
                    ),
                    "rpc_to_worker_queue_ms": (
                        round(d["max_rpc_to_worker_queue_ns"] / 1e6, 3)
                        if d.get("max_rpc_to_worker_queue_ns") is not None
                        else round((d["worker_start_ns"] - d["rpc_start_ns"]) / 1e6, 3)
                        if d.get("worker_start_ns") is not None and d.get("rpc_start_ns") is not None
                        else None
                    ),
                    "worker_span_count": d.get("worker_span_count"),
                    "worker_span_count_total": d.get("worker_span_count_total"),
                    "worker_boundary_policy": d.get("worker_boundary_policy"),
                },
                cname=d_color,
            ))

        for gap_idx, gap in enumerate(pp_gap_intervals):
            if gap["end_ns"] > gap["start_ns"]:
                pp_gap_lane_idx = pp_gap_lane_indices[gap_idx] if gap_idx < len(pp_gap_lane_indices) else 0
                pp_gap_lane_key = "pp_gap" if pp_gap_lane_idx == 0 else f"pp_gap_{pp_gap_lane_idx}"
                trace_events.append(create_perfetto_event(
                    name="PP Gap",
                    cat="request_pp_gap",
                    ph="X",
                    ts=gap["start_ns"],
                    dur=gap["end_ns"] - gap["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids.get(pp_gap_lane_key, lane_tids["pp_gap"]),
                    args={
                        "request_id": rid,
                        "dispatch_key": gap.get("dispatch_key"),
                        "phase": gap.get("phase"),
                        "section": gap.get("section"),
                        "edge": gap.get("edge"),
                        "dp_rank": gap.get("dp_rank"),
                        "src_pp": gap.get("src_pp"),
                        "dst_pp": gap.get("dst_pp"),
                        "duration_ms": gap.get("duration_ms"),
                    },
                    cname="rail_load",
                ))

        for q0 in req_coro_sched_intervals.get(rid, []):
            if q0["end_ns"] > q0["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Coroutine Scheduler Queue Wait",
                    cat="request_coro_sched_queue",
                    ph="X",
                    ts=q0["start_ns"],
                    dur=q0["end_ns"] - q0["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["coro_queue"],
                    args={"request_id": rid, "request_name": req_name_map.get(rid, rid)},
                    cname="yellow",
                ))

        for tq in req_generate_task_sched_intervals.get(rid, []):
            if tq["end_ns"] > tq["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Generate Task Runnable Queue",
                    cat="request_generate_task_queue",
                    ph="X",
                    ts=tq["start_ns"],
                    dur=tq["end_ns"] - tq["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["task"],
                    args={"request_id": rid, "request_name": req_name_map.get(rid, rid)},
                    cname="yellow",
                ))

        for gx in req_generate_exec_intervals.get(rid, []):
            if gx["end_ns"] > gx["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Generate Coroutine Exec",
                    cat="request_generate_exec",
                    ph="X",
                    ts=gx["start_ns"],
                    dur=gx["end_ns"] - gx["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["generate_exec"],
                    args={
                        "request_id": rid,
                        "request_name": req_name_map.get(rid, rid),
                        "task_name": gx.get("task_name"),
                    },
                    cname="rail_idle",
                ))

        for sq in req_output_socket_sched_intervals.get(rid, []):
            if sq["end_ns"] > sq["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Socket Scheduler Queue Wait",
                    cat="request_output_socket_sched_queue",
                    ph="X",
                    ts=sq["start_ns"],
                    dur=sq["end_ns"] - sq["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_socket_queue"],
                    args={
                        "request_id": rid,
                        "request_name": req_name_map.get(rid, rid),
                        "shared_req_count": sq.get("shared_req_count"),
                        "round_seq": sq.get("round_seq"),
                    },
                    cname="yellow",
                ))

        for sx in req_output_socket_exec_intervals.get(rid, []):
            if sx["end_ns"] > sx["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Socket Exec (Attributed)",
                    cat="request_output_socket_exec",
                    ph="X",
                    ts=sx["start_ns"],
                    dur=sx["end_ns"] - sx["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_socket_exec"],
                    args={
                        "request_id": rid,
                        "request_name": req_name_map.get(rid, rid),
                        "shared_req_count": sx.get("shared_req_count"),
                        "round_seq": sx.get("round_seq"),
                    },
                    cname="cyan",
                ))

        for qh in req_output_handler_sched_intervals.get(rid, []):
            if qh["end_ns"] > qh["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Handler Scheduler Queue Wait",
                    cat="request_output_handler_sched_queue",
                    ph="X",
                    ts=qh["start_ns"],
                    dur=qh["end_ns"] - qh["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_queue"],
                    args={
                        "request_id": rid,
                        "request_name": req_name_map.get(rid, rid),
                        "shared_req_count": qh.get("shared_req_count"),
                        "round_seq": qh.get("round_seq"),
                    },
                    cname="yellow",
                ))

        for ox in req_output_handler_exec_intervals.get(rid, []):
            if ox["end_ns"] > ox["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="Output Handler Exec (Attributed)",
                    cat="request_output_handler_exec",
                    ph="X",
                    ts=ox["start_ns"],
                    dur=ox["end_ns"] - ox["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["output_exec"],
                    args={
                        "request_id": rid,
                        "request_name": req_name_map.get(rid, rid),
                        "shared_req_count": ox.get("shared_req_count"),
                        "round_seq": ox.get("round_seq"),
                    },
                    cname="cyan",
                ))

        for q in queue_intervals_map.get(rid, []):
            if q["end_ns"] > q["start_ns"]:
                scheduled_from = q.get("scheduled_from")
                gap_kind = q.get("gap_kind", "normal_wait")
                if gap_kind == "preempt_wait":
                    q_name = "vLLM Preempt Wait (explicit preempt)"
                    q_cat = "request_queue_preempt"
                    q_color = "terrible"
                elif scheduled_from == "enqueue":
                    q_name = "vLLM Queue Wait (scheduled-enqueue)"
                    q_cat = "request_queue_scheduled_enqueue"
                    q_color = "yellow"
                elif scheduled_from == "step_ready":
                    q_name = "vLLM Scheduling Wait (step_ready->out_rpc)"
                    q_cat = "request_queue_scheduled_step_ready"
                    q_color = "yellow"
                else:
                    q_name = "vLLM Queue Wait"
                    q_cat = "request_queue"
                    q_color = "yellow"
                trace_events.append(create_perfetto_event(
                    name=q_name,
                    cat=q_cat,
                    ph="X",
                    ts=q["start_ns"],
                    dur=q["end_ns"] - q["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["vllm_queue"],
                    args={
                        "request_id": rid,
                        "reason": q["reason"],
                        "scheduled_from": scheduled_from,
                        "gap_kind": gap_kind,
                        "preempt_ts_ns": q.get("preempt_ts_ns", []),
                        "prev_dispatch_key": q.get("prev_dispatch_key"),
                        "next_dispatch_key": q.get("next_dispatch_key"),
                    },
                    cname=q_color,
                ))

        for o in req_os_intervals_wall.get(rid, []):
            if o["end_ns"] > o["start_ns"]:
                trace_events.append(create_perfetto_event(
                    name="OS Sched Delay (Attributed)",
                    cat="request_os",
                    ph="X",
                    ts=o["start_ns"],
                    dur=o["end_ns"] - o["start_ns"],
                    pid=REQUEST_PID,
                    tid=lane_tids["os"],
                    args={
                        "request_id": rid,
                        "worker_tids": ",".join(str(tid) for tid in o.get("worker_tids", [])),
                        "dp_ranks": ",".join(str(dp) for dp in o.get("dp_ranks", [])),
                        "reasons": ",".join(str(reason) for reason in o.get("reasons", [])),
                    },
                    cname="terrible",
                ))

        for s in req_stage_intervals.get(rid, []):
            if s["stage"] != "engine_core.input_queue_wait_after_preprocess":
                continue
            trace_events.append(create_perfetto_event(
                name="EngineCore Input Queue Wait",
                cat="request_engine_core_input_queue",
                ph="X",
                ts=s["start_ns"],
                dur=s["end_ns"] - s["start_ns"],
                pid=REQUEST_PID,
                tid=lane_tids["stage"],
                args={
                    "request_id": rid,
                    "stage": s["stage"],
                    "duration_ms": round((s["end_ns"] - s["start_ns"]) / 1e6, 3),
                },
                cname="yellow",
            ))

    print("=== Request Interference Summary ===")
    for rid in sorted(
        all_req_ids,
        key=lambda x: (
            req_coro_sched_ns.get(x, 0)
            + req_generate_task_sched_ns.get(x, 0)
            + req_generate_exec_ns.get(x, 0)
            + req_output_socket_sched_ns.get(x, 0)
            + req_output_socket_exec_ns.get(x, 0)
            + req_output_handler_sched_ns.get(x, 0)
            + req_output_handler_exec_ns.get(x, 0)
            + req_vllm_queue_ns.get(x, 0)
            + _request_pp_gap_wall_ns(x)
            + req_latency_map.get(x, 0)
        ),
        reverse=True,
    ):
        lifecycle_ms = None
        if rid in request_spans:
            lifecycle_ms = (request_spans[rid]["end_ns"] - request_spans[rid]["start_ns"]) / 1e6
        pp_gap_ms = _request_pp_gap_wall_ns(rid) / 1e6
        print(
            f"  - {rid}: "
            f"lifecycle={f'{lifecycle_ms:.3f} ms' if lifecycle_ms is not None else 'N/A'}, "
            f"coro_sched_queue={req_coro_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"generate_task_queue={req_generate_task_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"generate_exec={req_generate_exec_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_socket_sched_queue={req_output_socket_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_socket_exec={req_output_socket_exec_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_handler_sched_queue={req_output_handler_sched_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"output_handler_exec={req_output_handler_exec_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"vllm_queue={req_vllm_queue_ns.get(rid, 0) / 1e6:.3f} ms "
            f"(scheduled-enqueue={req_vllm_queue_from_enqueue_ns.get(rid, 0) / 1e6:.3f} ms, "
            f"scheduled-step_ready={req_vllm_queue_from_step_ready_ns.get(rid, 0) / 1e6:.3f} ms), "
            f"pp_gap={pp_gap_ms:.3f} ms, "
            f"os_delay={req_latency_map.get(rid, 0) / 1e6:.3f} ms, "
            f"prefill_exec={req_dispatch_phase_ns.get(rid, {}).get('prefill', 0) / 1e6:.3f} ms, "
            f"decode_exec={req_dispatch_phase_ns.get(rid, {}).get('decode', 0) / 1e6:.3f} ms"
        )

    # 避免同轨道 end==start 的零缝边界导致 Perfetto 局部漏渲染。
    # 只修正 queue/dispatch 这两类相邻阶段，避免影响 lifecycle/OS 等重叠语义事件。
    nudged_count = nudge_equal_boundaries(
        trace_events,
        target_pid=REQUEST_PID,
        epsilon_us=1.0,
        cat_prefixes=(
            "request_queue",
            "request_dispatch",
            "request_coro_sched_queue",
            "request_generate_task_queue",
            "request_output_socket_sched_queue",
            "request_output_handler_sched_queue",
            "request_generate_exec",
            "request_output_socket_exec",
            "request_output_handler_exec",
            "request_pp_gap",
        ),
        snap_tolerance_us=1.0,
    )
    if nudged_count > 0:
        print(f"Applied boundary nudge for request tracks: {nudged_count} events (epsilon=1.0us)")

    # GPU 同一 stream 上理论上应串行；这里观测到的 overlap 普遍只有几百 ns，
    # 更像 CUPTI 记录边界/量化误差。对小重叠做最小修正，避免 Perfetto 丢 slice。
    gpu_nudged_count = nudge_equal_boundaries(
        trace_events,
        epsilon_us=0.001,
        cat_prefixes=("gpu_kernel",),
        snap_tolerance_us=1.0,
    )
    if gpu_nudged_count > 0:
        print(
            "Applied boundary nudge for GPU kernel tracks: "
            f"{gpu_nudged_count} events (epsilon=0.001us, tolerance=1.0us)"
        )

    clipped_negative_count, dropped_negative_count = clip_negative_complete_events(trace_events)
    if clipped_negative_count or dropped_negative_count:
        print(
            "Clipped negative complete events before Perfetto export: "
            f"clipped={clipped_negative_count}, dropped={dropped_negative_count}"
        )

    # --- 输出 ---
    print("✨ Sorting all trace events...")
    trace_events.sort(key=lambda x: x.get('ts', 0))
    
    output_json = {"traceEvents": trace_events}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f)
    print(f"✅ Done. Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_to_trace.py <input.log> <output.json>")
        sys.exit(1)
    process_logs(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "trace.json")
