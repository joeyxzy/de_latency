import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from de_latency.runtime import transport


class RuntimeShutdownTests(unittest.TestCase):
    def test_transport_close_is_idempotent_and_fork_reset_does_not_close_inherited_socket(self):
        old = transport._local
        self.addCleanup(setattr, transport, "_local", old)
        transport._reset_child()
        socket, context = Mock(), Mock()
        transport._local.pid = os.getpid()
        transport._local.socket, transport._local.context = socket, context
        transport.close()
        transport.close()
        socket.close.assert_called_once_with()
        context.term.assert_called_once_with()
        socket.reset_mock()
        transport._local.pid = os.getpid()
        transport._reset_child()
        transport.close()
        socket.close.assert_not_called()

    def test_control_idle_client_cannot_block_shutdown_or_restart(self):
        with tempfile.TemporaryDirectory(prefix="de-control-") as directory:
            env = {**os.environ, "DE_LATENCY_DISABLE_SITECUSTOMIZE": "1",
                   "TRACER_MONKEYPATCH_CONTROL_DIR": directory,
                   "TRACER_MONKEYPATCH_CONTROL_POLL_TIMEOUT_S": "0.05"}
            code = """
import json, socket, time
from pathlib import Path
from de_latency.runtime import control
path = control.start_control_server()
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(path)
time.sleep(.05)
thread = control._server_thread
start = time.monotonic()
control.stop_control_server()
assert not thread.is_alive()
assert not Path(path).exists()
assert time.monotonic() - start < 1
client.close()
path = control.start_control_server()
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(1)
    client.connect(path)
    client.sendall(b'status')
    assert client.recv(1024).startswith(b'ok state=')
control.stop_control_server()
print(json.dumps({'passed': True}))
"""
            result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["passed"])
