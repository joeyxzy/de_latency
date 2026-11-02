"""Launch the user's vLLM API server with optional request instrumentation."""
import os
import runpy
import sys


def main():
    if os.getenv("DE_LATENCY_PYTHON_ENABLED", "1") != "0":
        from .bootstrap import install_request_hooks
        install_request_hooks()
    sys.argv = ["vllm.entrypoints.openai.api_server", *sys.argv[1:]]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
