from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

from de_latency.instrumentation.execution_context import current, push, pop


class ExecutionContextTests(unittest.TestCase):
    def test_nested_dispatch_and_threads_do_not_share_state(self):
        self.assertIsNone(current())
        outer, inner = {"dispatch_key": "outer"}, {"dispatch_key": "inner"}
        push(outer)
        try:
            push(inner)
            self.assertIs(current(), inner)
            self.assertIs(pop(), inner)
            self.assertIs(current(), outer)
            barrier = threading.Barrier(2)

            def work(key):
                self.assertIsNone(current())
                context = {"dispatch_key": key}
                push(context)
                barrier.wait(timeout=5)
                self.assertIs(current(), context)
                self.assertIs(pop(), context)
                self.assertIsNone(current())

            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(work, ["worker-1", "worker-2"]))
            self.assertIs(current(), outer)
        finally:
            pop()
        self.assertIsNone(pop())
