import hashlib
import json
from types import SimpleNamespace
import unittest

from de_latency.instrumentation.scheduler_metadata import dispatch_key, request_ids, step_maps


class SchedulerMetadataTests(unittest.TestCase):
    def output(self):
        return SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id="new", prompt_token_ids=[1, 2, 3], num_computed_tokens=0)],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"], num_computed_tokens=[12]),
            num_scheduled_tokens={"new": 2, "cached": 1}, total_num_scheduled_tokens=3,
        )

    def test_metadata_and_existing_dispatch_hash_are_preserved(self):
        output = self.output()
        self.assertEqual(request_ids(output), ["new", "cached"])
        self.assertEqual(step_maps(output), ({"new": 0, "cached": 12}, {"new": 2, "cached": 1}))
        parts = {"req_ids": ["new", "cached"], "num_computed_by_req": {"new": 0, "cached": 12},
                 "scheduled_tokens_by_req": {"new": 2, "cached": 1}, "total_scheduled_tokens": 3}
        raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        expected = "sched:" + hashlib.sha1(raw.encode()).hexdigest()[:16]
        self.assertEqual(dispatch_key(output), expected)
        # Fresh instances in worker/executor processes must yield the same ID.
        self.assertEqual(dispatch_key(self.output()), expected)
        output.scheduled_cached_reqs.num_computed_tokens[0] += 1
        self.assertNotEqual(dispatch_key(output), expected)

    def test_missing_scheduler_output_is_valid_for_runner_fallback(self):
        self.assertEqual(request_ids(None), [])
        self.assertEqual(step_maps(None), ({}, {}))
        self.assertTrue(dispatch_key(None).startswith("sched:"))
