"""Write all legacy-named artifacts from an AnalysisResult."""
import json
from pathlib import Path

from .perfetto import nudge_equal_boundaries, clip_negative_complete_events
from .reports import sanitize_filename, render_request_breakdown_svg, render_request_breakdown_index
from .trace_io import artifact_stem, json_output, overview_path, write_trace
from .overview import write_overview


def export_result(result, output_file, *, overview=False):
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(output)
    sidecar = lambda suffix: Path(str(stem) + suffix)
    # Only renderer copies undergo boundary repair; attribution stays exact.
    events = [dict(event) for event in result.trace_events]
    nudge_equal_boundaries(events, target_pid=result.request_track_pid, epsilon_us=1.0,
        cat_prefixes=("request_queue", "request_dispatch", "request_coro_sched_queue",
                      "request_generate_task_queue", "request_output_socket_sched_queue",
                      "request_output_handler_sched_queue", "request_generate_exec",
                      "request_output_socket_exec", "request_output_handler_exec", "request_pp_gap"),
        snap_tolerance_us=1.0)
    nudge_equal_boundaries(events, epsilon_us=.001, cat_prefixes=("gpu_kernel",), snap_tolerance_us=1.0)
    clip_negative_complete_events(events)
    events.sort(key=lambda event: event.get("ts", 0))
    write_trace(output, events)
    if overview:
        write_overview(events, overview_path(output))
    for path, value in ((sidecar(".compute_breakdown.json"), result.compute_breakdown),
                        (sidecar(".summary.json"), result.summary),
                        (sidecar(".alignment.json"), {"traceEvents": result.alignment_events})):
        with json_output(path) as stream:
            json.dump(value, stream, ensure_ascii=False)
    directory = sidecar(".compute_breakdown_requests")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, rid in enumerate(result.request_order, 1):
        entry = result.compute_breakdown["requests"].get(rid, {})
        ttft, tpot = entry.get("ttft"), entry.get("tpot")
        name = entry.get("request_name", rid)
        filename = f"{index:04d}_{sanitize_filename(name)[:80]}.svg"
        render_request_breakdown_svg(directory / filename, rid, name, ttft, tpot)
        rows.append({"request_name": name, "file_name": filename,
                     "ttft_ms": f'{ttft["window_ms"]:.3f}' if ttft else "-",
                     "tpot_ms": f'{tpot["avg_ms_per_token"]:.3f}' if tpot else "-",
                     "reentry": bool((ttft or {}).get("phase_reentry_after_decode") or (tpot or {}).get("phase_reentry_after_decode"))})
    render_request_breakdown_index(directory / "index.html", rows)
    print(f"Saved trace and analysis artifacts: {output}")
