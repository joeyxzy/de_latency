import unittest

from de_latency.analysis.os_scheduling import attribute_os_waits, estimate_offset


class OsSchedulingTests(unittest.TestCase):
    def test_explicit_clock_samples_take_precedence_over_metric_fallback(self):
        metrics = [{"batch_ts_ns": 1000, "payload": {"req_events": [{"events": [{"ts": 100}]}]}}]
        self.assertEqual(estimate_offset({10: 200}, [202, 204], metrics), 202)
        self.assertEqual(estimate_offset({}, [], metrics), 900)
        self.assertIsNone(estimate_offset({}, [], []))

    def test_worker_extension_and_tp_max_pp_sum_preserve_historical_policy(self):
        dispatches = [
            {"dispatch_key": "step", "tid": tid, "start": 1100, "end": 1120,
             "req_ids": ["r"], "ranks": {"pp": pp, "tp": tp}}
            for tid, pp, tp in ((10, 0, 0), (11, 0, 1), (12, 1, 0))
        ]
        waits = [{"tid": tid, "start_ns": start, "end_ns": end, "reason": "preempt"}
                 for tid, start, end in ((10, 105, 115), (11, 105, 120), (12, 115, 125))]
        extension = [{"payload": {"dispatch_key": "step", "tid": 12, "end_ns": 130}}]
        result = attribute_os_waits(waits, dispatches, extension, 1000)
        self.assertEqual(result.dispatch_delay_ns, {"step": 25})
        self.assertEqual(result.request_delay_ns, {"r": 25})
        self.assertEqual(len(result.overlaps["step"]), 3)
        self.assertEqual(result.overlaps["step"][-1]["end_mono_ns"], 125)

    def test_missing_calibration_does_not_mix_monotonic_and_wall_time(self):
        result = attribute_os_waits([{"tid": 1, "start_ns": 10, "end_ns": 20}], [], [], None)
        self.assertEqual(result.by_tid[1][0]["dur"], 10)
        self.assertEqual(result.overlaps, {})
        self.assertEqual(result.dispatch_delay_ns, {})
