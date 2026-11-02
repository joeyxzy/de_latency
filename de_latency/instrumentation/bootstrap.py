"""Installation boundaries: worker import hooks and API request hooks."""
import importlib
import importlib.metadata
import re


def check_version():
    version = importlib.metadata.version("vllm")
    if not re.match(r"^0\.20\.", version):
        raise RuntimeError(f"de_latency requires vLLM 0.20.x; found {version}")
    return version


def install_worker_hooks():
    check_version()
    importlib.import_module("de_latency.instrumentation.worker_hooks")


def install_request_hooks():
    install_worker_hooks()
    importlib.import_module("de_latency.instrumentation.request_hooks")
