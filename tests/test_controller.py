"""Lifecycle checks using real controller subprocesses, without CUDA workloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import signal
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "build/controller").exists(), "run make first")
class ControllerTests(unittest.TestCase):
    def run_target(self, folder, code, **extra):
        env = {**os.environ, "DE_LATENCY_DISABLE_SITECUSTOMIZE": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
               "TRACER_CUPTI_START_ENABLED": "0", "TRACER_EBPF_START_ENABLED": "0",
               "TRACER_ZMQ_ADDR": f"ipc://{folder}/events.sock",
               "TRACER_WORKER_PID_FILE": str(folder / "targets"),
               "DE_LATENCY_LOG_PATH": str(folder / "events.log"),
               "DE_LATENCY_CONTROL_STATE_FILE": str(folder / "control"),
               "TRACER_CUPTI_CONTROL_DIR": str(folder),
               "TRACER_DEBUG_LOG_PATH": str(folder / "cupti.log"), **extra}
        return subprocess.run([str(ROOT / "build/controller"), sys.executable, "-c", code],
                              cwd=folder, env=env, capture_output=True, text=True, timeout=15)

    def test_target_exit_code_and_collector_drain(self):
        with tempfile.TemporaryDirectory(prefix="de-controller-") as tmp:
            folder = Path(tmp)
            completed = self.run_target(folder, "raise SystemExit(7)")
            self.assertEqual(completed.returncode, 7, completed.stderr)
            stats = json.loads((folder / "events.log.stats.json").read_text())
            self.assertEqual(stats["invalid_events"], 0)
            self.assertFalse((folder / "control").exists())

    def test_requested_monitor_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory(prefix="de-controller-") as tmp:
            completed = self.run_target(Path(tmp), "import time; time.sleep(60)",
                TRACER_EBPF_START_ENABLED="1", DE_LATENCY_EBPF_MONITOR_BIN="/bin/false")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("eBPF monitor failed to start", completed.stderr)

    def test_collector_shutdown_failure_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="de-controller-") as tmp:
            folder = Path(tmp)
            collector = folder / "bad_collector.py"
            collector.write_text(
                "import os, signal, time, zmq\n"
                "def stop(*args): raise SystemExit(3)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "context=zmq.Context(); socket=context.socket(zmq.PULL)\n"
                "socket.bind(os.environ['TRACER_ZMQ_ADDR'])\n"
                "while True: time.sleep(.1)\n"
            )
            completed = self.run_target(folder, "pass", DE_LATENCY_COLLECTOR_SCRIPT=str(collector))
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertIn("collector did not exit successfully", completed.stderr)

    def test_disabled_cupti_does_not_preload_tracer(self):
        with tempfile.TemporaryDirectory(prefix="de-controller-") as tmp:
            completed = self.run_target(Path(tmp),
                "import os; assert 'libtracer.so' not in os.environ.get('LD_PRELOAD', '')",
                TRACER_CUPTI_PRELOAD="0", LD_PRELOAD="")
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_requested_shutdown_cannot_mask_collector_failure(self):
        with tempfile.TemporaryDirectory(prefix="de-controller-") as tmp:
            folder = Path(tmp)
            collector = folder / "bad_collector.py"
            collector.write_text(
                "import os, signal, time, zmq\n"
                "def stop(*args): raise SystemExit(3)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "context=zmq.Context(); socket=context.socket(zmq.PULL)\n"
                "socket.bind(os.environ['TRACER_ZMQ_ADDR'])\n"
                "while True: time.sleep(.1)\n"
            )
            env = {**os.environ, "DE_LATENCY_DISABLE_SITECUSTOMIZE": "1",
                   "TRACER_CUPTI_PRELOAD": "0", "TRACER_EBPF_START_ENABLED": "0",
                   "TRACER_ZMQ_ADDR": f"ipc://{folder}/events.sock",
                   "DE_LATENCY_COLLECTOR_SCRIPT": str(collector),
                   "DE_LATENCY_CONTROL_STATE_FILE": str(folder / "control")}
            proc = subprocess.Popen([str(ROOT / "build/controller"), sys.executable, "-c",
                                     "import time; time.sleep(60)"], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (folder / "control").exists():
                    if proc.poll() is not None or time.monotonic() > deadline:
                        self.fail("controller did not start")
                    time.sleep(.05)
                proc.send_signal(signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 1, stdout + stderr)
                self.assertIn("collector did not exit successfully", stderr)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.communicate(timeout=10)
