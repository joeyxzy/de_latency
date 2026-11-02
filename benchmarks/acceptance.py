"""Acceptance gates for real captures, including producer loss and attribution."""
import json
import math

from de_latency.protocol import LogDiagnostics, iter_log
from de_latency.export.trace_io import iter_trace_events


def validate_capture(session, prompts, required_sources, *, trace_stem="trace"):
    errors = []
    stats = LogDiagnostics()
    invalid_kernels = 0
    kernel_pids = set()
    marker_pids = set()
    graph_launches = 0
    ranks = set()
    manifest = json.loads((session / "manifest.json").read_text())
    for event in iter_log(session / "events.log", stats, strict=True):
        if event["schema_version"] != 2:
            errors.append("non-v2 event in new capture")
            break
        if event["source"] == "cupti" and event["event_type"] == "kernel":
            kernel_pids.add(event["pid"])
            payload = event["payload"]
            if not 0 < payload.get("start_ns", 0) < payload.get("end_ns", 0):
                invalid_kernels += 1
        if event["source"] == "cupti" and event["event_type"] == "marker":
            marker_pids.add(event["pid"])
        if event["source"] == "cupti" and event["event_type"] in ("runtime", "driver"):
            graph_launches += "GraphLaunch" in event["payload"].get("name", "")
        if event["event_type"] == "gpu_phase_span":
            rank = event["payload"].get("ranks", {})
            if "tp" in rank and "pp" in rank:
                ranks.add((rank["tp"], rank["pp"]))
    missing = required_sources - stats.sources.keys()
    if missing:
        errors.append(f"missing sources: {sorted(missing)}")
    if stats.sessions != {session.name}:
        errors.append(f"unexpected sessions: {sorted(stats.sessions)}")
    for kind in ("cupti:kernel", "cupti:marker", "monkey_patch:gpu_phase_span"):
        if not stats.event_types[kind]:
            errors.append(f"missing {kind}")
    if invalid_kernels:
        errors.append(f"invalid kernel intervals: {invalid_kernels}")
    collector = json.loads((session / "events.log.stats.json").read_text())
    if collector["sources"] != dict(stats.sources):
        errors.append("collector counts do not match persisted log")
    if collector["invalid_events"]:
        errors.append(f"collector rejected {collector['invalid_events']} events")
    drops = {key: value for key, value in collector["producer_dropped"].items() if value}
    if drops:
        errors.append("producer reported dropped events")
    ebpf_stats = None
    if "ebpf" in required_sources:
        stats_file = session / "ebpf.stats.json"
        if not stats_file.exists():
            errors.append("missing final eBPF status")
        else:
            ebpf_stats = json.loads(stats_file.read_text())
            if any(ebpf_stats.get(key) != 0 for key in
                   ("ring_dropped", "map_dropped", "send_dropped", "exit_code")):
                errors.append("eBPF monitor reported failure or loss")
    statuses = json.loads((session / "cupti_off.json").read_text())
    expected_workers = manifest["tp"] * manifest["pp"]
    if len(statuses) != expected_workers or len(kernel_pids) != expected_workers or kernel_pids != marker_pids:
        errors.append("incomplete per-worker CUPTI kernel/marker coverage")
    if manifest["graph"] and not graph_launches:
        errors.append("Graph mode produced no CUDA Graph launches")
    if expected_workers > 1:
        expected_ranks = {(tp, pp) for tp in range(manifest["tp"]) for pp in range(manifest["pp"])}
        if ranks != expected_ranks:
            errors.append(f"incomplete TP/PP phase coverage: {sorted(ranks)}")
    if "ebpf" in required_sources:
        summary = json.loads((session / f"{trace_stem}.summary.json").read_text())
        if not any(value > 0 for value in summary["os_delay_ms_by_request"].values()):
            errors.append("no eBPF scheduling delay attributed to requests")
    for worker, response in statuses.items():
        fields = dict(part.split("=", 1) for part in response.split() if "=" in part)
        if not response.startswith("ok ") or fields.get("state") != "off":
            errors.append(f"{worker}: CUPTI did not stop and drain")
        for key in ("ring_dropped", "zmq_dropped", "cupti_dropped"):
            if fields.get(key) != "0":
                errors.append(f"{worker}: {key}={fields.get(key, 'missing')}")
    breakdown = json.loads((session / f"{trace_stem}.compute_breakdown.json").read_text())
    requests = breakdown["requests"]
    if len(requests) != prompts:
        errors.append(f"expected {prompts} requests, analyzed {len(requests)}")
    for rid, request in requests.items():
        for phase, components_key, exec_key in (
            ("ttft", "components_ns", "prefill_exec"),
            ("tpot", "components_ns_total", "decode_exec"),
        ):
            entry = request.get(phase)
            if not entry:
                errors.append(f"{rid}: missing {phase}")
                continue
            components = entry[components_key]
            if any(not math.isfinite(value) or value < 0 for value in components.values()):
                errors.append(f"{rid}: invalid {phase} component")
            if abs(sum(components.values()) - entry["window_ns"]) > 1:
                errors.append(f"{rid}: {phase} components do not close the window")
            if components[exec_key] <= 0:
                errors.append(f"{rid}: missing {phase} execution attribution")
    trace_path = session / f"{trace_stem}.json.gz"
    if not trace_path.exists():
        trace_path = session / f"{trace_stem}.json"
    complete_count, invalid_complete = 0, False
    for event in iter_trace_events(trace_path):
        if event.get("ph") != "X":
            continue
        complete_count += 1
        invalid_complete |= (not math.isfinite(event["ts"]) or not math.isfinite(event["dur"]) or event["dur"] < 0)
    if not complete_count or invalid_complete:
        errors.append("invalid or empty Perfetto intervals")
    return {"passed": not errors, "errors": errors, "trace_stem": trace_stem, "sources": dict(stats.sources),
            "event_types": dict(stats.event_types), "request_count": len(requests),
            "required_sources": sorted(required_sources), "producer_dropped": drops,
            "cupti_stop_status": statuses, "ebpf_status": ebpf_stats,
            "graph_launches": graph_launches, "kernel_worker_pids": sorted(kernel_pids),
            "phase_ranks": sorted(ranks),
            "perfetto_complete_events": complete_count}
