import unittest

from de_latency.analysis.ttft import partition_ttft


class TtftAttributionTests(unittest.TestCase):
    def test_os_wait_is_subtracted_from_execution_and_rank_duplicates_merge(self):
        result = partition_ttft(100, 200, 120, [(120, 130)], [(130, 140)],
                                [(140, 150)], [(150, 190), (150, 190)],
                                [(160, 170), (165, 175)], 190)
        self.assertEqual(result, {
            "preprocess": 20, "enginecore_queue": 10, "cpu_sched_wait": 10,
            "gpu_dispatch_wait": 10, "prefill_exec": 25, "os_sched_wait": 15,
            "postprocess_transport": 10, "other_gap": 0,
        })

    def test_execution_after_first_token_is_clipped_not_added(self):
        result = partition_ttft(100, 200, 120, [], [], [(120, 160)],
                                [(150, 240)], [(195, 210)], 240)
        self.assertEqual(sum(result.values()), 100)
        self.assertEqual(result["prefill_exec"], 45)
        self.assertEqual(result["os_sched_wait"], 5)
        self.assertEqual(result["gpu_dispatch_wait"], 30)
        self.assertEqual(result["postprocess_transport"], 0)

    def test_os_wait_outside_execution_is_not_charged(self):
        result = partition_ttft(100, 200, 130, [], [], [], [(140, 180)],
                                [(110, 120), (185, 195)], 180)
        self.assertEqual(result["os_sched_wait"], 0)
        self.assertEqual(result["other_gap"], 10)
        self.assertEqual(sum(result.values()), 100)

    def test_large_integer_timestamps_preserve_nanoseconds(self):
        base = 1_800_000_000_000_000_000
        result = partition_ttft(base, base + 9, base + 2, [], [], [],
                                [(base + 2, base + 7)], [(base + 3, base + 4)], base + 7)
        self.assertEqual(result["os_sched_wait"], 1)
        self.assertEqual(sum(result.values()), 9)
