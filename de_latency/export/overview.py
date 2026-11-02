"""Display-only overview: never changes raw capture or numerical attribution."""
from pathlib import Path

from .trace_io import iter_trace_events, overview_path, write_trace

MARKER_ROLES = ("exec_start|", "exec_end|", "sample_start|", "sample_end|")
OVERVIEW_METADATA = {
    "view": "overview",
    "omitted": "non-marker GPU activity, CUDA APIs and flow arrows; source and compute breakdown unchanged",
}


def keep_overview_event(event):
    if event.get("cat") == "flow" or event.get("ph") in ("s", "f", "t"):
        return False
    if event.get("cat") == "gpu_kernel":
        return "de_marker" in (event.get("name") or "")
    if event.get("cat") == "cuda_runtime":
        return (event.get("name") or "").startswith(MARKER_ROLES)
    return True


def write_overview(events, output):
    return write_trace(output, (event for event in events if keep_overview_event(event)),
                       metadata=OVERVIEW_METADATA)


def overview_from_trace(source, output=None):
    source = Path(source).resolve()
    output = Path(output).resolve() if output is not None else overview_path(source)
    if source == output:
        raise ValueError("overview output must not overwrite its full source trace")
    count = write_overview(iter_trace_events(source), output)
    return output, count
