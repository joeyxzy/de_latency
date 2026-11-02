"""CUPTI host/GPU tracks and one-to-one launch flows for Perfetto."""
from collections import defaultdict

from de_latency.analysis.clocks import to_int
from de_latency.analysis.correlation import cupti_corr_key, kernels_for_launch


def append_gpu_trace(events, writer, api_events, kernel_index, kernels, fallback_pid=None):
    api_events.sort(key=lambda item: item["start_ns"])
    lane_ends = defaultdict(list)
    kernel_flows = {}
    for api in api_events:
        start, end = api["start_ns"], api["end_ns"]
        pid = api.get("pid")
        if not pid and fallback_pid is not None:
            pid = fallback_pid
        tid = to_int(api.get("tid"))
        if tid is None or tid <= 0:
            lanes = lane_ends[pid]
            index = next((i for i, boundary in enumerate(lanes) if start >= boundary), None)
            if index is None:
                index = len(lanes)
                lanes.append(end)
                events.append({"name": "thread_name", "ph": "M", "pid": pid,
                               "tid": 10_000_000 + index,
                               "args": {"name": f"CUPTI API lane {index + 1} (tid unknown)"}})
            else:
                lanes[index] = end
            tid = 10_000_000 + index
        matched = kernels_for_launch(kernel_index, api, pid)
        names = [kernel.get("name", "unknown") for kernel in matched]
        source_pid, correlation = cupti_corr_key(api, pid_override=pid)
        name = api.get("name", "cuda_runtime")
        events.append(writer.event(name=name, cat="cuda_runtime", ph="X", ts=start,
            dur=end - start, pid=pid, tid=tid,
            args={"correlationId": api.get("correlationId"),
                  "correlationKey": f"{source_pid}:{correlation}",
                  "kernels": "\n".join(names[:5]) + ("..." if len(names) > 5 else ""),
                  "is_graph": "graph" in name.lower()}))
        for kernel in matched:
            if kernel.get("start_ns", 0) <= 0 or id(kernel) in kernel_flows:
                continue
            flow = 1_000_000_000 + len(kernel_flows)
            kernel_flows[id(kernel)] = flow
            events.append(writer.flow("s", start, pid, tid, flow))
    tracks, streams = {}, set()
    for kernel in kernels:
        start, end = kernel["start_ns"], kernel["end_ns"]
        key = (to_int(kernel.get("pid")) or 0, to_int(kernel.get("deviceId")) or 0)
        if key not in tracks:
            tracks[key] = 9000 + len(tracks)
            events.append({"name": "process_name", "ph": "M", "pid": tracks[key],
                           "args": {"name": f"GPU local {key[1]} / source pid {key[0]}"}})
        pid, stream = tracks[key], kernel.get("streamId", 0)
        events.append(writer.event(name=kernel.get("name", "kernel"), cat="gpu_kernel", ph="X",
                                   ts=start, dur=end - start, pid=pid, tid=stream, args=kernel))
        if (pid, stream) not in streams:
            streams.add((pid, stream))
            events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": stream,
                           "args": {"name": f"Stream {stream}"}})
        if id(kernel) in kernel_flows:
            events.append(writer.flow("f", start, pid, stream, kernel_flows[id(kernel)]))
