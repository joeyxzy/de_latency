import io
import json
import tempfile
import unittest
from pathlib import Path

from de_latency.protocol import make_event, validate_event, decode_record, analysis_record, iter_log, LogDiagnostics
from de_latency.analysis.intervals import merge_intervals, uncovered_interval_ns
from de_latency.export.perfetto import PerfettoWriter
from de_latency.analysis.clocks import to_ns


class ProtocolTests(unittest.TestCase):
    def test_monotonic_ns_after_recent_boot_is_not_seconds(self):
        self.assertEqual(to_ns(123456789), 123456789)
        self.assertEqual(to_ns("123456789"), 123456789)
        self.assertEqual(to_ns(1.25), 1250000000)
    def test_legacy_collector_and_cupti_envelopes(self):
        payload = {"pid": 10, "start_ns": 100, "end_ns": 200}
        wrapped = {"meta": {"source": "CUPTI", "event_type": "kernel", "payload": payload}}
        flat = {"source": "CUPTI", "event_type": "kernel", "payload": payload}
        self.assertEqual(decode_record(wrapped), decode_record(flat))
        self.assertEqual(analysis_record(decode_record(flat))["source"], "cupti")

    def test_cupti_clock_conversion_preserves_missing_timestamp(self):
        event = make_event("cupti", "kernel", {"start_ns": 0, "end_ns": 0}, clock_domain="cupti")
        event["clock_offset_ns"] = 1000
        self.assertEqual(analysis_record(event)["payload"]["start_ns"], 0)
        event["payload"].update(start_ns=100, end_ns=200)
        self.assertEqual(analysis_record(event)["payload"]["start_ns"], 1100)
        del event["clock_offset_ns"]
        with self.assertRaises(ValueError):
            analysis_record(event)

    def test_v2_rejects_float_nanoseconds_and_unknown_version(self):
        event = make_event("ebpf", "sched_latency", {"start_ns": 1.5}, clock_domain="monotonic")
        with self.assertRaises(ValueError):
            validate_event(event)
        with self.assertRaises(ValueError):
            decode_record({"schema_version": 3})

    def test_bad_lines_are_diagnosed_and_strict_mode_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.log"
            path.write_text(json.dumps(make_event("ebpf", "sched_latency", {}, clock_domain="monotonic")) + "\ninvalid\n")
            stats = LogDiagnostics()
            self.assertEqual(len(list(iter_log(path, stats))), 1)
            self.assertEqual(stats.invalid_lines, 1)
            with self.assertRaises(ValueError):
                list(iter_log(path, strict=True))

    def test_overlapping_intervals_are_not_double_counted(self):
        base = 1_800_000_000_000_000_000
        covered = [(base, base + 10), (base + 5, base + 15)]
        self.assertEqual(merge_intervals(covered), [(base, base + 15)])
        self.assertEqual(uncovered_interval_ns(base, base + 20, covered), 5)

    def test_renderers_do_not_share_time_origin(self):
        first, second = PerfettoWriter(1000), PerfettoWriter(2000)
        args = ("span", "test", "X", 3000, 1000, 1, 1)
        self.assertEqual(first.event(*args)["ts"], 2)
        self.assertEqual(second.event(*args)["ts"], 1)
        self.assertEqual(first.event(*args)["ts"], 2)


if __name__ == "__main__":
    unittest.main()
