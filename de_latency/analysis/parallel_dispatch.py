"""TP-aware execution accounting and gaps between adjacent PP stage windows."""
from collections import defaultdict

from .clocks import to_ns
from .intervals import clip_interval, merge_intervals, sum_interval_pairs
from .wall_time import attribute_wall_components, ns_to_ms, normalize_component_map_ms

POLICY = "pp_gap_from_adjacent_pp_step_execution_windows; TP ranks in the same PP stage are merged before computing PP-stage gaps; request-level wall components are attributed over merged dispatch windows so overlapping dispatches/chunked-prefill PP gaps are counted once; raw_step_execution_by_pp_tp and pp_gap_by_edge keep raw per-rank/per-edge sums"


def _rank(value):
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _matches(key, interval):
    other = interval.get("dispatch_key")
    return not key or not other or key == other


def _units(result, **values):
    for name, value in values.items():
        is_map = isinstance(value, dict)
        result[name + "_ns"] = dict(value) if is_map else value
        result[name + "_ms"] = normalize_component_map_ms(value) if is_map else ns_to_ms(value)
    return result


def build_pp_gap_dispatch_breakdown(dispatches, phase_intervals_by_name, step_execution_intervals,
                                     phase_component_map=None):
    phase_component_map = phase_component_map or {}
    rows, windows, global_intervals = [], [], []
    compute, raw_compute, raw_components = defaultdict(int), defaultdict(int), defaultdict(int)
    steps, raw_steps, gaps = defaultdict(int), defaultdict(int), defaultdict(int)
    for dispatch in sorted(dispatches, key=lambda item: (item.get("start_ns") or 0, item.get("end_ns") or 0)):
        key = dispatch.get("dispatch_key")
        start, end = to_ns(dispatch.get("start_ns")), to_ns(dispatch.get("end_ns"))
        if start is None or end is None or end <= start:
            continue
        windows.append((start, end))
        stage_compute, stage_raw, stage_components = defaultdict(int), defaultdict(int), defaultdict(int)
        by_component, intervals = defaultdict(list), []
        for phase, phase_intervals in phase_intervals_by_name.items():
            component = phase_component_map.get(phase, "other_compute")
            for item in phase_intervals:
                span = clip_interval(item.get("start_ns"), item.get("end_ns"), start, end)
                if not span or not _matches(key, item):
                    continue
                ranks = item.get("ranks") if isinstance(item.get("ranks"), dict) else {}
                pp, tp = ranks.get("pp", "unknown"), ranks.get("tp", "unknown")
                duration = span[1] - span[0]
                raw_key = f"pp{pp}:tp{tp}:{component}"
                stage_raw[raw_key] += duration
                raw_compute[raw_key] += duration
                stage_components[component] += duration
                raw_components[component] += duration
                by_component[(pp, component)].append(span)
        for (pp, component), spans in by_component.items():
            wall_key = f"pp{pp}:{component}"
            for left, right in merge_intervals(spans):
                stage_compute[wall_key] += right - left
                compute[wall_key] += right - left
                intervals.append({"start_ns": left, "end_ns": right,
                                  "component": component, "detail_key": wall_key})
        step_spans, tp_by_stage = defaultdict(list), defaultdict(set)
        stage_raw_steps, stage_steps = defaultdict(int), defaultdict(int)
        for item in step_execution_intervals:
            span = clip_interval(item.get("start_ns"), item.get("end_ns"), start, end)
            if not span or not _matches(key, item):
                continue
            ranks = item.get("ranks") if isinstance(item.get("ranks"), dict) else {}
            dp, pp, tp = (ranks.get(rank, "unknown") for rank in ("dp", "pp", "tp"))
            raw_key = f"dp{dp}:pp{pp}:tp{tp}:step_execution"
            stage_raw_steps[raw_key] += span[1] - span[0]
            raw_steps[raw_key] += span[1] - span[0]
            step_spans[(dp, pp)].append(span)
            tp_by_stage[(dp, pp)].add(tp)
            stage_steps[f"dp{dp}:pp{pp}:step_execution"] += 0
        stage_windows = defaultdict(list)
        for (dp, pp), spans in step_spans.items():
            merged = merge_intervals(spans)
            left, right = min(a for a, b in merged), max(b for a, b in merged)
            union = sum_interval_pairs(merged)
            wall_key = f"dp{dp}:pp{pp}:step_execution"
            stage_steps[wall_key] += union
            steps[wall_key] += union
            stage_windows[dp].append(_units({
                "dp_rank": dp, "pp_rank": pp, "start_ns": left, "end_ns": right,
                "tp_ranks": sorted(tp_by_stage[(dp, pp)], key=_rank), "span_count": len(spans),
            }, duration=right - left, union_duration=union))
        gap_intervals, row_gaps = [], defaultdict(int)
        for dp, stage in stage_windows.items():
            stage.sort(key=lambda item: _rank(item["pp_rank"]))
            for previous, following in zip(stage, stage[1:]):
                left, right = previous["end_ns"], following["start_ns"]
                if right <= left:
                    continue
                edge = f"dp{dp}:pp{previous['pp_rank']}->pp{following['pp_rank']}"
                gaps[edge] += right - left
                row_gaps[edge] += right - left
                gap_intervals.append(_units({"start_ns": left, "end_ns": right,
                    "edge": edge, "dp_rank": dp, "src_pp": previous["pp_rank"],
                    "dst_pp": following["pp_rank"]}, duration=right - left))
                intervals.append({"start_ns": left, "end_ns": right,
                                  "component": "pp_gap", "detail_key": edge})
        wall = attribute_wall_components(intervals, start, end)
        global_intervals.extend(intervals)
        rows.append(_units({
            "dispatch_key": key, "phase": dispatch.get("phase"), "decode_step": dispatch.get("decode_step"),
            "start_ns": start, "end_ns": end,
            "stage_windows": [window for dp, stage in sorted(stage_windows.items(), key=lambda pair: _rank(pair[0]))
                              for window in stage],
            "pp_gap_intervals": gap_intervals, "wall_segments": wall["segments"],
        }, duration=end - start, compute_by_pp_phase=stage_compute,
            raw_compute_by_pp_tp_phase=stage_raw, raw_compute_by_component=stage_components,
            step_execution_by_pp=stage_steps, raw_step_execution_by_pp_tp=stage_raw_steps,
            pp_gap_by_edge=row_gaps, wall_components=wall["wall_components_ns"],
            pp_gap=wall["wall_components_ns"].get("pp_gap", 0),
            raw_pp_gap=sum(item["duration_ns"] for item in gap_intervals),
            unattributed_gap=wall["pipeline_gap_ns"], parallel_overlap=wall["parallel_overlap_ns"]))
    total, segments, overlap = defaultdict(int), [], 0
    merged_windows = merge_intervals(windows)
    for start, end in merged_windows:
        wall = attribute_wall_components(global_intervals, start, end)
        for key, value in wall["wall_components_ns"].items():
            total[key] += value
        overlap += wall["parallel_overlap_ns"]
        segments.extend(wall["segments"])
    return _units({
        "dispatches": rows, "wall_segments": segments, "aggregation_policy": POLICY,
        "merged_dispatch_windows": [_units({"start_ns": a, "end_ns": b}, duration=b - a)
                                    for a, b in merged_windows],
    }, compute_by_pp_phase=compute, raw_compute_by_pp_tp_phase=raw_compute,
        raw_compute_by_component=raw_components, step_execution_by_pp=steps,
        raw_step_execution_by_pp_tp=raw_steps, pp_gap_by_edge=gaps, wall_components=total,
        total_compute=sum(value for key, value in total.items() if key not in ("pp_gap", "pipeline_gap")),
        total_pp_gap=total.get("pp_gap", 0), total_unattributed_gap=total.get("pipeline_gap", 0),
        total_wall=sum_interval_pairs(merged_windows), total_parallel_overlap=overlap)
