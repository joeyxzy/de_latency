"""Results are independent of output paths and renderer state."""
from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    trace_events: list
    compute_breakdown: dict
    summary: dict
    request_order: list
    alignment_events: list = field(default_factory=list)
    request_track_pid: int = 0
