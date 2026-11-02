"""Compatibility bootstrap for worker-side hooks (vLLM 0.20 only)."""
import logging
import os
import sys

from de_latency.runtime.control import start_control_server
from .registry import install
from .workers import engine_core, entry, executor, gpu_runner, parallel, scheduler


def install_worker_hooks():
    if os.getenv("DE_LATENCY_DISABLE_SITECUSTOMIZE", "0") == "1" or os.path.basename(sys.argv[0]) in {
        "collector.py", "cupti_ctl.py", "trace_ctl.py",
    }:
        return
    try:
        start_control_server()
    except OSError as exc:
        logging.getLogger(__name__).warning("worker control bootstrap failed: %s", exc)
    install()


install_worker_hooks()
