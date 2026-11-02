"""Non-overlapping wall-time attribution for concurrent TP/PP components."""
from collections import defaultdict

from .clocks import to_ns
from .intervals import clip_interval


def ns_to_ms(value_ns):
    return None if value_ns is None else round(value_ns / 1e6, 6)


def normalize_component_map_ms(components):
    return {key: ns_to_ms(value) for key, value in components.items()}


def _component_priority(component):
    priority = {"pp_gap": 0, "model_forward": 10, "sampling": 20,
                "postprocess": 30, "bookkeeping_sync": 40, "worker_preprocess": 50,
                "other_compute": 60, "other_exec": 70, "other_prefill_exec": 70,
                "pipeline_gap": 100}
    return priority.get(component, 80), str(component)


def attribute_wall_components(component_intervals, window_start_ns, window_end_ns):
    """Partition the window; overlapping ranks never multiply wall time."""
    start, end = to_ns(window_start_ns), to_ns(window_end_ns)
    components = defaultdict(int)
    segments = []
    overlap = 0
    if start is not None and end is not None and end > start:
        clipped = []
        boundaries = {start, end}
        for item in component_intervals:
            span = clip_interval(item.get("start_ns"), item.get("end_ns"), start, end)
            if span:
                clipped.append((*span, item.get("component") or "other_compute"))
                boundaries.update(span)
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            active = [component for a, b, component in clipped if a < right and b > left]
            active_components = sorted(set(active), key=_component_priority)
            chosen = active_components[0] if active else ("pipeline_gap" if clipped else "other_compute")
            duration = right - left
            components[chosen] += duration
            overlap += max(0, len(active) - 1) * duration
            segments.append({"start_ns": left, "end_ns": right, "duration_ns": duration,
                             "duration_ms": ns_to_ms(duration), "component": chosen,
                             "active_components": active_components})
    gap = components.get("pipeline_gap", 0)
    return {"wall_components_ns": dict(components),
            "wall_components_ms": normalize_component_map_ms(components),
            "pipeline_gap_ns": gap, "pipeline_gap_ms": ns_to_ms(gap),
            "parallel_overlap_ns": overlap, "parallel_overlap_ms": ns_to_ms(overlap),
            "segments": segments}
