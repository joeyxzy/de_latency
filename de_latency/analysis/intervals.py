"""Interval algebra shared by attribution components (integer nanoseconds)."""
from .clocks import to_ns


def clip_interval(start_ns, end_ns, window_start_ns=None, window_end_ns=None):
    start, end = to_ns(start_ns), to_ns(end_ns)
    if start is None or end is None or end <= start:
        return None
    start = max(start, window_start_ns) if window_start_ns is not None else start
    end = min(end, window_end_ns) if window_end_ns is not None else end
    return (start, end) if end > start else None


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(span) for span in merged]


def sum_interval_pairs(intervals):
    return sum(max(0, end - start) for start, end in intervals)


def uncovered_interval_ns(window_start_ns, window_end_ns, covered_intervals):
    clipped = [span for start, end in covered_intervals
               if (span := clip_interval(start, end, window_start_ns, window_end_ns))]
    return max(0, window_end_ns - window_start_ns - sum_interval_pairs(merge_intervals(clipped)))


def sum_clipped_interval_items(intervals, window_start_ns=None, window_end_ns=None, merge_overlaps=False):
    clipped = [span for item in intervals
               if (span := clip_interval(item.get("start_ns"), item.get("end_ns"), window_start_ns, window_end_ns))]
    if merge_overlaps:
        clipped = merge_intervals(clipped)
    return sum_interval_pairs(clipped), clipped


def sum_intersections(intervals_a, intervals_b):
    intersections = []
    for a in intervals_a:
        aa = clip_interval(a.get("start_ns"), a.get("end_ns"))
        if aa is None:
            continue
        for b in intervals_b:
            span = clip_interval(b.get("start_ns"), b.get("end_ns"), *aa)
            if span:
                intersections.append(span)
    return sum_interval_pairs(intersections), intersections
