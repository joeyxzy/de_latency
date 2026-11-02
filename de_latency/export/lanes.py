"""Display-only lane allocation; never changes attribution intervals."""
from de_latency.analysis.clocks import to_ns


def assign_non_overlapping_lanes(intervals, start_key="start_ns", end_key="end_ns"):
    lane_ends = []
    assignments = [0] * len(intervals)
    ordered = sorted(enumerate(intervals), key=lambda pair: (
        to_ns(pair[1].get(start_key)) or 0,
        to_ns(pair[1].get(end_key)) or 0,
        pair[0],
    ))
    for index, item in ordered:
        start, end = to_ns(item.get(start_key)), to_ns(item.get(end_key))
        if start is None or end is None or end <= start:
            continue
        lane = next((i for i, boundary in enumerate(lane_ends) if start >= boundary), None)
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(end)
        else:
            lane_ends[lane] = end
        assignments[index] = lane
    return assignments, len(lane_ends)
