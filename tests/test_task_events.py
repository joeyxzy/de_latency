import unittest
from unittest.mock import patch

from de_latency.instrumentation import task_events


class TaskEventTests(unittest.TestCase):
    def test_batch_wait_and_exec_payloads_are_consistent(self):
        for component in ("output_handler", "output_socket"):
            with self.subTest(component=component), patch.object(task_events, "send_event") as send:
                wait = getattr(task_events, f"emit_{component}_sched_latency")
                execute = getattr(task_events, f"emit_{component}_exec_slice")
                wait(100, 150, 50, ["r1", "r2"], task_id=42, round_seq=3)
                execute(150, 180, ["r1", "r2"], task_id=42, round_seq=3)
                first, second = [call.kwargs for call in send.call_args_list]
                self.assertEqual(first["event_type"], f"{component}_sched_latency")
                self.assertEqual(second["event_type"], f"{component}_exec_slice")
                for key in ("req_ids", "batch_size", "task_id", "round_seq", "pid", "tid"):
                    self.assertEqual(first["payload"][key], second["payload"][key])
                self.assertEqual(first["payload"]["timestamp_ns"], 150)
                self.assertEqual(second["payload"]["duration_ns"], 30)

    def test_invalid_spans_or_empty_batch_do_not_emit(self):
        with patch.object(task_events, "send_event") as send:
            task_events.emit_output_handler_sched_latency(100, 90, -10, ["r"])
            task_events.emit_output_socket_exec_slice(100, 90, ["r"])
            task_events.emit_output_socket_sched_latency(100, 150, 50, [])
            task_events.emit_coroutine_exec_slice(100, 100)
            send.assert_not_called()

    def test_stage_timing_can_be_disabled_and_preserves_extra(self):
        with patch.object(task_events, "send_event") as send:
            with patch.object(task_events, "ENABLE_REQ_STAGE_TIMING", False):
                task_events.emit_stage_duration("stage", 100, 200)
                send.assert_not_called()
            with patch.object(task_events, "ENABLE_REQ_STAGE_TIMING", True):
                task_events.emit_stage_duration("stage", 100, 200, "r", "request", {"source_method": "method"})
                payload = send.call_args.kwargs["payload"]
                self.assertEqual(payload["duration_ns"], 100)
                self.assertEqual(payload["source_method"], "method")
