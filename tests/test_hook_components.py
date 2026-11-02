import asyncio
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from de_latency.instrumentation.registry import apply_method_patch
from de_latency.instrumentation.requests.state import TaskRegistry, tasks
from de_latency.instrumentation.requests.rounds import begin_round, begin_slice, end_round, register_output_task
from de_latency.instrumentation.requests.scheduling import scheduled
from de_latency.instrumentation.workers import executor, parallel


class WorkerComponentTests(unittest.TestCase):
    def test_inherited_executor_wrapper_is_not_wrapped_again(self):
        class Base:
            def execute_model(self, scheduler_output):
                return scheduler_output
        class Child(Base):
            pass
        apply_method_patch(Base, "execute_model", executor.execute)
        apply_method_patch(Child, "execute_model", executor.execute)
        apply_method_patch(Child, "execute_model", executor.execute)
        self.assertEqual(str(inspect.signature(Child.execute_model)), "(self, scheduler_output)")
        output = SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                                 num_scheduled_tokens={}, total_num_scheduled_tokens=0)
        with patch.object(executor.TraceSender, "emit") as emit:
            self.assertIs(Child().execute_model(output), output)
        emit.assert_not_called()

    def test_pp_async_receive_preserves_handles_postprocessing_and_errors(self):
        class Handle:
            attribute = 7
            def wait(self):
                return "done"
        tensors, postprocess = {"hidden": "tensor"}, [lambda: None]
        group = SimpleNamespace(rank_in_group=1, world_size=2, ranks=[0, 1])
        original = lambda self, src=None: (tensors, [Handle()], postprocess)
        wrapper = parallel.communication("irecv_tensor_dict", "src", 0, False, True)(original)
        with patch.object(parallel, "is_enabled", return_value=True), patch.object(parallel, "is_pp", return_value=True), \
                patch.object(parallel, "ranks", return_value={}), patch.object(parallel.TraceSender, "emit") as emit:
            received, handles, callbacks = wrapper(group, 0)
            self.assertIs(received, tensors)
            self.assertIs(callbacks, postprocess)
            self.assertEqual(handles[0].attribute, 7)
            self.assertEqual(handles[0].wait(), "done")
            self.assertEqual([call.args[1]["async_phase"] for call in emit.call_args_list], ["launch", "wait"])
            self.assertEqual(emit.call_args_list[0].args[1]["comm_id"], emit.call_args_list[1].args[1]["comm_id"])
            failing = parallel.communication("isend_tensor_dict", "dst", 1, True, True)(
                lambda *args: (_ for _ in ()).throw(ValueError("send failed")))
            with self.assertRaisesRegex(ValueError, "send failed"):
                failing(group, tensors)
            self.assertTrue(emit.call_args_list[-1].args[1]["error"])

    def test_pp_disabled_wrapper_preserves_exact_return(self):
        result = object()
        wrapper = parallel.communication("isend_tensor_dict", "dst", 1, True, True)(lambda *args: result)
        with patch.object(parallel, "is_enabled", return_value=False), patch.object(parallel.TraceSender, "emit") as emit:
            self.assertIs(wrapper(object(), {}), result)
        emit.assert_not_called()


class RequestComponentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        tasks.unregister(asyncio.current_task())

    async def test_registry_context_and_done_cleanup(self):
        registry = TaskRegistry()
        async def run():
            return 1
        task = asyncio.create_task(run(), name="request-task")
        registry.register(task, "request", "display")
        self.assertEqual(registry.lookup(id(task))["request_name"], "display")
        await task
        await asyncio.sleep(0)
        self.assertIsNone(registry.lookup(id(task)))

    async def test_pending_waits_follow_remaining_output_slices(self):
        task = asyncio.current_task()
        register_output_task(task, "output_handler", "output")
        slices = [[SimpleNamespace(request_id="a")], [SimpleNamespace(request_id="b")],
                  [SimpleNamespace(request_id="a"), SimpleNamespace(request_id="c")]]
        sequence = begin_round(slices, task)
        with patch("de_latency.instrumentation.requests.rounds.emit_output_handler_sched_latency") as emit:
            for index, expected in enumerate((["a", "b", "c"], ["b", "a", "c"], ["a", "c"])):
                tasks.lookup(id(task))["pending_sched_events"] = [{"ready_ts_ns": 10, "run_ts_ns": 20, "queue_ns": 10}]
                ids, current_sequence = begin_slice(slices[index], "output_handler")
                self.assertEqual(ids, [item.request_id for item in slices[index]])
                self.assertEqual(current_sequence, sequence)
                self.assertEqual(emit.call_args.args[3], expected)
        end_round(task)
        self.assertEqual(tasks.lookup(id(task))["round_suffix_req_ids"], [])
        self.assertIsNone(tasks.lookup(id(task))["active_round_seq"])

    async def test_callback_scheduling_buffers_output_wait_and_preserves_exception(self):
        task = asyncio.current_task()
        register_output_task(task, "output_socket", "socket")
        def fail():
            raise ValueError("callback failed")
        callback = scheduled(fail, 100, id(task), None, None)
        with patch.object(tasks, "announce_role"), patch("de_latency.instrumentation.requests.scheduling.time.time_ns", side_effect=[120, 125]):
            with self.assertRaisesRegex(ValueError, "callback failed"):
                callback()
        self.assertEqual(tasks.lookup(id(task))["pending_sched_events"],
                         [{"ready_ts_ns": 100, "run_ts_ns": 120, "queue_ns": 20}])
