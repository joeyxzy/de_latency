import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import zmq

from de_latency.protocol import make_event
from de_latency.instrumentation.registry import apply_method_patch


class RuntimeTests(unittest.TestCase):
    def test_patch_is_idempotent_and_preserves_return_value(self):
        class Example:
            def run(self, value):
                return value + 1
        calls = []
        def factory(original):
            def wrapper(self, value):
                calls.append(value)
                return original(self, value)
            return wrapper
        apply_method_patch(Example, "run", factory)
        apply_method_patch(Example, "run", factory)
        self.assertEqual(Example().run(5), 6)
        self.assertEqual(calls, [5])

    def test_collector_drains_and_discovers_targets(self):
        with tempfile.TemporaryDirectory(prefix="de-latency-") as directory:
            folder = Path(directory)
            env = {**os.environ, "DE_LATENCY_DISABLE_SITECUSTOMIZE": "1",
                   "TRACER_ZMQ_ADDR": f"ipc://{directory}/events.sock",
                   "DE_LATENCY_LOG_PATH": str(folder / "events.log"),
                   "TRACER_WORKER_PID_FILE": str(folder / "targets")}
            proc = subprocess.Popen([sys.executable, "-m", "de_latency.collector"], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            context = zmq.Context()
            sender = context.socket(zmq.PUSH)
            sender.setsockopt(zmq.SNDTIMEO, 3000)
            sender.setsockopt(zmq.LINGER, 1000)
            try:
                deadline = time.monotonic() + 10
                while not (folder / "events.sock").exists():
                    if time.monotonic() > deadline or proc.poll() is not None:
                        self.fail("collector failed to bind")
                    time.sleep(.05)
                sender.connect(env["TRACER_ZMQ_ADDR"])
                for source, kind, payload in [
                    ("monkey_patch", "worker_process_ready", {"pid": 123, "tid": 123}),
                    ("cupti", "kernel", {"pid": 123, "start_ns": 10, "end_ns": 20}),
                    ("ebpf", "sched_latency", {"tid": 123, "start_ns": 11, "end_ns": 12}),
                ]:
                    sender.send_json(make_event(source, kind, payload))
                sender.close()
                context.term()
                time.sleep(.2)
                proc.send_signal(signal.SIGTERM)
                output, _ = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 0, output.decode())
                records = [json.loads(line) for line in (folder / "events.log").read_text().splitlines()]
                self.assertEqual({record["source"] for record in records}, {"monkey_patch", "cupti", "ebpf"})
                self.assertEqual((folder / "targets").read_text(), "123\n")
            finally:
                if not sender.closed:
                    sender.close(0)
                    context.term()
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()


if __name__ == "__main__":
    unittest.main()
