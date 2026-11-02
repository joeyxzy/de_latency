import unittest

from de_latency.analysis.parallel_dispatch import build_pp_gap_dispatch_breakdown


class ParallelDispatchTests(unittest.TestCase):
    def test_tp_merge_then_pp_gap_and_overlapping_dispatch_deduplication(self):
        def span(start, end, pp, tp):
            return {"start_ns": start, "end_ns": end, "dispatch_key": "batch",
                    "ranks": {"dp": 0, "pp": pp, "tp": tp}}

        intervals = [span(100, 140, 0, 0), span(105, 145, 0, 1),
                     span(155, 195, 1, 0), span(160, 200, 1, 1)]
        dispatch = {"start_ns": 100, "end_ns": 200, "dispatch_key": "batch", "phase": "prefill"}
        result = build_pp_gap_dispatch_breakdown([dispatch], {"Forward": intervals}, intervals,
                                                  {"Forward": "model_forward"})
        self.assertEqual(result["total_wall_ns"], 100)
        self.assertEqual(result["total_compute_ns"], 90)
        self.assertEqual(result["total_pp_gap_ns"], 10)
        self.assertEqual(result["raw_compute_by_component_ns"]["model_forward"], 160)
        self.assertEqual(result["step_execution_by_pp_ns"]["dp0:pp0:step_execution"], 45)
        self.assertEqual(result["dispatches"][0]["stage_windows"][0]["tp_ranks"], [0, 1])
        duplicate = build_pp_gap_dispatch_breakdown([dispatch, dict(dispatch)], {"Forward": intervals},
                                                    intervals, {"Forward": "model_forward"})
        self.assertEqual(duplicate["total_wall_ns"], 100)
        self.assertEqual(duplicate["total_pp_gap_ns"], 10)
        self.assertEqual(duplicate["total_compute_ns"], 90)
        self.assertEqual(duplicate["pp_gap_by_edge_ns"]["dp0:pp0->pp1"], 20)

    def test_empty_and_invalid_windows(self):
        for dispatches in ([], [{"start_ns": 20, "end_ns": 10}]):
            with self.subTest(dispatches=dispatches):
                result = build_pp_gap_dispatch_breakdown(dispatches, {}, [])
                self.assertEqual(result["total_wall_ns"], 0)
                self.assertEqual(result["dispatches"], [])
