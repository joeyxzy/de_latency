"""Python startup entry point, including vLLM spawn workers."""
import os

if os.getenv("DE_LATENCY_DISABLE_SITECUSTOMIZE", "0") != "1":
    from de_latency.instrumentation.bootstrap import install_worker_hooks
    install_worker_hooks()
