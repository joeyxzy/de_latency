import argparse
import contextlib
import importlib
import io
import unittest
from unittest.mock import patch


def load_cli(name):
    return importlib.import_module(f"de_latency.runtime.{name}")


class ControlCliTests(unittest.TestCase):
    def test_cupti_rejected_command_returns_failure(self):
        cli = load_cli("cupti_cli")
        args = argparse.Namespace(pid=123, socket_path="/unused", command="off",
                                  mode="graceful", timeout=60)
        with patch.object(cli, "parse_args", return_value=args), \
             patch.object(cli, "send_command", return_value="error=disable_or_drain_failed state=off"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(), 1)

    def test_group_command_reports_partial_failure(self):
        cli = load_cli("control_cli")
        with patch.object(cli.os.path, "exists", return_value=True), \
             patch.object(cli, "send_socket_command", side_effect=["ok state=off", "error=disable_failed"]):
            results, failures = cli.control_socket_group("cupti", [1, 2], "off", 60, str)
        self.assertEqual(results, [(1, "ok state=off")])
        self.assertEqual(failures, [(2, "error=disable_failed")])
