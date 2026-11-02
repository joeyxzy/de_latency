"""Public analysis entry points and explicit stage orchestration."""
import sys
from .normalization import to_plain_data
from de_latency.export.request_layout import build_alignment_debug_trace
# Explicit compatibility exports for former pipeline helper callers.
from .normalization import count_prefill_decode_tokens, infer_trace_origin_ns
from de_latency.export.request_layout import (
    emit_thread_name_once, build_request_lane_tids, build_raw_dispatch_display_intervals,
)
from de_latency.export.styles import describe_gpu_phase
from .ingestion import load_log
from .worker_intervals import analyze_workers
from .request_timing import analyze_request_timing
from .dispatch import analyze_dispatches
from .request_stages import analyze_request_stages
from .request_index import index_requests
from .breakdown import build_compute_breakdown
from de_latency.export.workers import append_worker_trace
from de_latency.export.core import append_core_trace
from de_latency.export.requests import append_request_trace
from .os_scheduling import estimate_offset, attribute_os_waits
from .model import AnalysisResult
from de_latency.export.perfetto import PerfettoWriter
from de_latency.export.gpu import append_gpu_trace
from de_latency.export.os_scheduling import append_eventloop_waits


def analyze_logs(input_file):
    log_data = load_log(input_file)
    trace_events = []
    writer = PerfettoWriter(log_data.TRACE_TS_ORIGIN_NS)
    worker_analysis = analyze_workers(log_data)
    request_timing = analyze_request_timing(log_data)
    dispatch_analysis = analyze_dispatches(log_data, request_timing, worker_analysis)
    request_stages = analyze_request_stages(dispatch_analysis, log_data, request_timing)
    offset = estimate_offset(log_data.thread_mono_to_wall_offset_by_tid,
                             log_data.worker_span_mono_to_wall_offsets, log_data.req_metric_events)
    os_result = attribute_os_waits(log_data.ebpf_sched_latency_events, worker_analysis.generated_dispatch_slices,
                                  log_data.execute_model_span, offset)
    request_index = index_requests(
        dispatch_analysis, os_result, request_stages, request_timing, worker_analysis,
    )
    compute_analysis = build_compute_breakdown(
        dispatch_analysis, log_data, os_result, request_index, request_stages, request_timing,
    )
    append_worker_trace(trace_events, writer, worker_analysis)
    append_core_trace(log_data, trace_events, writer)
    append_gpu_trace(trace_events, writer, log_data.cupti_events, log_data.all_kernels_map, log_data.eager_kernels,
                     worker_analysis.generated_dispatch_slices[0]['pid'] if worker_analysis.generated_dispatch_slices else None)
    append_eventloop_waits(trace_events, writer, os_result.by_tid, log_data.eventloop_thread_tids,
                          log_data.thread_mono_to_wall_offset_by_tid, offset,
                          log_data.thread_roles_by_tid, log_data.TRACE_TS_ORIGIN_NS)
    sorted_req_ids = append_request_trace(
        compute_analysis, dispatch_analysis, log_data, os_result, request_index,
        request_stages, request_timing, worker_analysis, trace_events, writer,
    )
    summary = {
        "schema_version": 2,
        "input": log_data.diagnostics.as_dict(),
        "request_count": len(request_index.all_req_ids),
        "os_delay_ms_by_request": {rid: ns / 1e6 for rid, ns in os_result.request_delay_ns.items()},
    }
    alignment = build_alignment_debug_trace(log_data.execute_model_span, log_data.ebpf_sched_latency_events)
    return AnalysisResult(trace_events, to_plain_data(compute_analysis.compute_breakdown), summary,
                          sorted_req_ids, alignment, 11000)


def process_logs(input_file, output_file="trace.json.gz", *, overview=False):
    from de_latency.export.bundle import export_result
    result = analyze_logs(input_file)
    export_result(result, output_file, overview=overview)
    return result.summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m de_latency.analysis.pipeline <input.log> [output.json.gz]")
        sys.exit(1)
    process_logs(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "trace.json.gz")
