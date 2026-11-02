"""Reject loss reported only at shutdown, after the last logged event."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.acceptance import validate_capture
from de_latency.protocol import make_event
from de_latency.export.trace_io import write_trace


class AcceptanceTests(unittest.TestCase):
    def test_shutdown_loss_cannot_be_hidden_by_clean_collector_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)

            def write(name, value):
                (session / name).write_text(json.dumps(value))

            write("manifest.json", {"tp": 1, "pp": 1, "graph": False})
            events = []
            for source, kind in (("cupti", "kernel"), ("cupti", "marker"),
                                 ("monkey_patch", "gpu_phase_span")):
                event = make_event(source, kind, {"pid": 123, "start_ns": 10, "end_ns": 20,
                                                  "ranks": {"tp": 0, "pp": 0}})
                event["session_id"] = session.name
                events.append(json.dumps(event))
            (session / "events.log").write_text("\n".join(events) + "\n")
            write("events.log.stats.json", {"invalid_events": 0, "producer_dropped": {},
                                           "sources": {"cupti": 2, "monkey_patch": 1}})
            write("trace.compute_breakdown.json", {"requests": {"r": {
                "ttft": {"window_ns": 10, "components_ns": {"prefill_exec": 10}},
                "tpot": {"window_ns": 10, "components_ns_total": {"decode_exec": 10}},
            }}})
            write("trace.json", {"traceEvents": [{"ph": "X", "ts": 0, "dur": 1}]})
            status = "ok state=off ring_dropped=0 zmq_dropped=0 cupti_dropped=0"
            write("cupti_off.json", {"worker": status})
            required = {"monkey_patch", "cupti"}
            self.assertTrue(validate_capture(session, 1, required)["passed"])
            write_trace(session / "trace.json.gz", [{"ph": "X", "ts": 0, "dur": 1}])
            write("trace.json", {"traceEvents": []})  # Old output must not shadow the new compressed result.
            self.assertTrue(validate_capture(session, 1, required)["passed"])
            for field in ("ring_dropped", "zmq_dropped", "cupti_dropped"):
                with self.subTest(field=field):
                    write("cupti_off.json", {"worker": status.replace(f"{field}=0", f"{field}=1")})
                    report = validate_capture(session, 1, required)
                    self.assertFalse(report["passed"])
                    self.assertIn(f"worker: {field}=1", report["errors"])
