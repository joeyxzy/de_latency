"""Index worker and GPU marker windows by dispatch key."""
from collections import defaultdict
from dataclasses import dataclass
from .clocks import to_ns


@dataclass
class DispatchWindows:
    worker: dict
    gpu: dict


def index_dispatch_windows(dispatch_slices: list, cupti_events: list) -> DispatchWindows:
    worker_exec_window_by_dispatch = {}
    grouped_worker_spans_by_dispatch = defaultdict(list)
    for span in dispatch_slices:
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

    return DispatchWindows(worker_exec_window_by_dispatch, gpu_batch_by_dispatch)
