"""Clock calibration and worker/request attribution for eBPF runnable waits."""
from collections import defaultdict
from dataclasses import dataclass

from .clocks import to_ns, to_int, median_int
from .requests import normalize_request_ids


@dataclass
class OsAttribution:
    offset_ns: int | None
    by_tid: dict
    overlaps: dict
    dispatch_delay_ns: dict
    request_delay_ns: dict


def estimate_offset(thread_offsets, worker_offsets, metrics):
    explicit = list(thread_offsets.values()) + list(worker_offsets)
    if explicit:
        return median_int(explicit)
    candidates = []
    for item in metrics:
        payload = item.get("payload", item)
        timestamps = [to_ns(event.get("ts")) for update in payload.get("req_events", [])
                      for event in update.get("events", []) if to_ns(event.get("ts"))]
        wall = to_ns(item.get("batch_ts_ns"))
        if wall and timestamps:
            candidates.append(wall - max(timestamps))
    return median_int(candidates) if candidates else None


def attribute_os_waits(events, dispatches, worker_spans, offset_ns):
    by_tid = defaultdict(list)
    for event in events:
        payload = event.get("payload", event)
        tid, start, end = payload.get("tid"), payload.get("start_ns"), payload.get("end_ns")
        if tid and start and end:
            by_tid[tid].append({"start": start, "end": end, "dur": end - start, "reason": payload.get("reason")})
    for waits in by_tid.values():
        waits.sort(key=lambda item: item["start"])
    windows = defaultdict(list)
    for dispatch in dispatches:
        key, tid = dispatch.get("dispatch_key"), dispatch.get("tid")
        start, end = dispatch.get("start"), dispatch.get("end")
        if not key or tid is None or start is None or end is None or end <= start:
            continue
        ranks = dispatch.get("ranks") if isinstance(dispatch.get("ranks"), dict) else {}
        pp, tp = to_int(ranks.get("pp")), to_int(ranks.get("tp"))
        windows[tid].append({"dk": key, "pp_rank": pp if pp is not None else 0,
            "tp_rank": tp if tp is not None else 0, "wall_start": start, "wall_end": end,
            "req_ids": normalize_request_ids(dispatch.get("req_ids", []))})
    if offset_ns is not None:
        for item in worker_spans:
            payload = item.get("payload", item)
            key, tid, end = payload.get("dispatch_key"), to_int(payload.get("tid")), to_ns(payload.get("end_ns"))
            if key and tid is not None and end is not None:
                for window in windows.get(tid, []):
                    if window["dk"] == key:
                        window["wall_end"] = max(window["wall_end"], end + offset_ns)
    overlaps = defaultdict(list)
    fragments = defaultdict(lambda: defaultdict(list))
    # Historical policy: match each wait against the first overlapping worker
    # window. Keep this stable independently of the v2 TTFT union partition.
    if offset_ns is not None:
        for tid, worker_windows in windows.items():
            worker_windows.sort(key=lambda item: (item["wall_start"], item["wall_end"]))
            index = 0
            for event in by_tid.get(tid, []):
                start, end = event["start"] + offset_ns, event["end"] + offset_ns
                if end <= start:
                    continue
                while index < len(worker_windows) and worker_windows[index]["wall_end"] <= start:
                    index += 1
                if index >= len(worker_windows):
                    break
                window = worker_windows[index]
                left, right = max(start, window["wall_start"]), min(end, window["wall_end"])
                if right <= left:
                    continue
                key, pp, tp = window["dk"], window["pp_rank"], window["tp_rank"]
                fragments[key][pp, tp].append(right - left)
                overlaps[key].append({"start_mono_ns": left - offset_ns, "end_mono_ns": right - offset_ns,
                    "worker_tid": tid, "pp_rank": pp, "tp_rank": tp, "reason": event.get("reason"),
                    "req_ids": window["req_ids"]})
    dispatch_delay = {}
    for key, per_rank in fragments.items():
        per_stage = defaultdict(int)
        for (pp, tp), waits in per_rank.items():
            per_stage[pp] = max(per_stage[pp], sum(waits))
        if total := int(sum(per_stage.values())):
            dispatch_delay[key] = total
    by_request = defaultdict(lambda: defaultdict(float))
    for key, waits in overlaps.items():
        ids, per_stage = set(), defaultdict(float)
        for wait in waits:
            ids.update(normalize_request_ids(wait.get("req_ids", [])))
            duration = (wait.get("end_mono_ns") or 0) - (wait.get("start_mono_ns") or 0)
            if duration > 0:
                per_stage[wait.get("pp_rank", 0)] = max(per_stage[wait.get("pp_rank", 0)], duration)
        total = sum(per_stage.values())
        if total > 0:
            for rid in ids:
                by_request[rid][key] = max(by_request[rid].get(key, 0), total)
    request_delay = defaultdict(int, {rid: int(sum(delays.values())) for rid, delays in by_request.items()})
    return OsAttribution(offset_ns, by_tid, overlaps, dispatch_delay, request_delay)
