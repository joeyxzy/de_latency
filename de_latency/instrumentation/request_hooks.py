"""Compatibility bootstrap for API-side vLLM 0.20 instrumentation."""
import logging

from de_latency.runtime.control import start_control_server
from .requests import scheduling, output, stages, lifecycle


def patch_vllm():
    from vllm.v1.engine import async_llm, core_client
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
    cls = async_llm.AsyncLLM
    if getattr(cls, "_de_latency_req_stage_patch_installed", False):
        return
    output.install(async_llm, OutputProcessor, RequestOutputCollector, core_client)
    stages.install(cls, InputProcessor, OutputProcessor, core_client)
    lifecycle.install(cls)
    cls._de_latency_req_stage_patch_installed = True


patch_asyncio_scheduler = scheduling.install


def install_request_hooks():
    try:
        start_control_server()
    except OSError as exc:
        logging.getLogger(__name__).warning("request control bootstrap failed: %s", exc)
    patch_asyncio_scheduler()
    patch_vllm()


install_request_hooks()
