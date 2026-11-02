import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from de_latency.analysis.ingestion import load_log
from de_latency.analysis.worker_intervals import analyze_workers
from de_latency.analysis.pipeline import analyze_logs
from de_latency.export.perfetto import PerfettoWriter
from de_latency.export.workers import append_worker_trace
from de_latency.protocol import make_event


def sample_events():
    def event(kind, timestamp, **payload):
        return make_event("monkey_patch", kind, {"timestamp_ns": timestamp, "pid": 10, "tid": 11, **payload})
    rid = "request"
    events = [
        event("coroutine_start", 100, request_id=rid, coroutine_id=1),
        event("req_generate_stage", 110, request_id=rid, stage="async_llm.generate_wrapper_setup", start_ns=100, end_ns=110),
        event("req_enqueue_scheduler", 120, req_id=rid),
        event("enginecore_mainloop_span", 125, phase="_process_input_queue", start_ns=120, end_ns=125),
    ]
    for key, phase, rpc, start, end, consume in (("first", "prefill", 130, 140, 170, 180), ("second", "decode", 190, 200, 220, 230)):
        common = {"dispatch_key": key, "req_ids": [rid]}
        events += [
            event("scheduler_schedule", rpc, start_ns=rpc-5, end_ns=rpc, dispatch_key=key),
            event("req_scheduler_out_rpc", rpc, **common),
            event("gpu_execute_model", end, start_ns=start, end_ns=end, **common,
                  ranks={"pp": 0, "tp": 0}, scheduled_tokens_by_req={rid: 2 if phase == "prefill" else 1}),
            event("gpu_phase_span", end, start_ns=start, end_ns=end, phase="Forward", **common),
            event("pp_comm_span", end, start_ns=start+1, end_ns=start+2, op="send_tensor_dict", **common,
                  ranks={"pp": 0, "tp": 0}, peer_rank=1),
            event("pp_comm_span", end, start_ns=start+3, end_ns=start+4, op="recv_tensor_dict", **common,
                  ranks={"pp": 1, "tp": 0}, peer_rank=0),
            event("scheduler_sample_tokens", end, dispatch_key=key),
            event("req_step_ready", end+1, **common, phase_by_req={rid: phase}, scheduler_before_ns=end),
            event("coroutine_exec_slice", consume, request_id=rid, start_ns=consume-2, end_ns=consume,
                  task_kind="generate_consume"),
        ]
    events.append(event("coroutine_end", 240, request_id=rid, coroutine_id=1))
    return events


class AnalysisStagesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name) / "events.log"
        self.log.write_text("".join(json.dumps(event) + "\n" for event in sample_events()))

    def test_nonempty_pipeline_renders_without_shadowing_stage_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze_logs(self.log)
        self.assertEqual(result.summary["request_count"], 1)
        request = result.compute_breakdown["requests"]["request"]
        self.assertEqual(request["ttft"]["window_ns"], 80)
        self.assertEqual(request["tpot"]["decode_dispatch_count"], 1)
        categories = {event.get("cat") for event in result.trace_events}
        self.assertTrue({"worker", "python", "pp_comm", "enginecore_loop", "request_lifecycle"} <= categories)
        self.assertEqual(sum(event.get("id", 0) == 700001 for event in result.trace_events), 2)

    def test_worker_stage_is_independent_of_writer_and_rendering_is_repeatable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_log(self.log)
            workers = analyze_workers(data)
        self.assertEqual(len(workers.generated_dispatch_slices), 2)
        self.assertEqual(len(workers.req_pp_comm_intervals["request"]), 4)
        first, second = [], []
        writer = PerfettoWriter(data.TRACE_TS_ORIGIN_NS)
        append_worker_trace(first, writer, workers)
        append_worker_trace(second, writer, workers)
        self.assertEqual(first, second)
        self.assertEqual(workers.timeline[0].start_ns, 140)
