import sys
import importlib
import logging
import os
import threading
import inspect
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VLLM_HOOK")

class VllmWorkerPatcher(MetaPathFinder, Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "vllm.v1.worker.gpu_model_runner":
            return spec_from_loader(fullname, self)
        return None

    def exec_module(self, module):
        # 1. 恢复 sys.meta_path 并重新加载原模块
        sys.meta_path = [x for x in sys.meta_path if x is not self]
        importlib.reload(importlib.import_module("vllm.v1.worker.gpu_model_runner"))
        sys.meta_path.insert(0, self)
        
        target_module = sys.modules["vllm.v1.worker.gpu_model_runner"]
        
        # --- 调试信息：打印实际加载的文件路径 ---
        logger.info(f">>> [HOOK DEBUG] Module loaded from: {target_module.__file__}")
        
        if hasattr(target_module, "GPUModelRunner"):
            self._apply_patch(target_module.GPUModelRunner)
        else:
            logger.warning(f">>> [HOOK] GPUModelRunner not found in module!")

    def _apply_patch(self, cls):
        import time
        import zmq
        import json
        
        # --- 智能寻找目标函数 ---
        target_method_name = None
        if hasattr(cls, "_model_forward"):
            target_method_name = "_model_forward"
        elif hasattr(cls, "forward"):
            target_method_name = "forward"
        elif hasattr(cls, "execute_model"):
            target_method_name = "execute_model"
            
        if target_method_name is None:
            logger.error(f">>> [HOOK FATAL] Attributes: {dir(cls)}")
            return

        logger.info(f">>> [HOOK] Targeted method: '{target_method_name}'")
        original_func = getattr(cls, target_method_name)

        # ZMQ Setup (保持不变)
        _thread_local = threading.local()
        def _get_socket():
            if not hasattr(_thread_local, "sock"):
                try:
                    ctx = zmq.Context()
                    sock = ctx.socket(zmq.PUSH)
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.connect("ipc:///tmp/tracer.sock") 
                    _thread_local.sock = sock
                except Exception:
                    _thread_local.sock = None
            return _thread_local.sock

        def _send_trace(payload):
            sock = _get_socket()
            if sock:
                try:
                    meta = {
                        "source": "monkey_patch",
                        "event_type": "gpu_forward",
                        "timestamp_ns": time.time_ns(),
                        "payload": payload
                    }
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    sock.send_multipart([meta_bytes, b""], flags=zmq.DONTWAIT)
                except Exception:
                    pass

        # --- Patch 主体 ---
        def patched_func(self, *args, **kwargs):
            start_ns = time.time_ns()
            pid = os.getpid()
            try:
                tid = threading.get_native_id()
            except AttributeError:
                tid = threading.get_ident()

            # 1. 执行原函数
            res = original_func(self, *args, **kwargs)
            
            # 2. 采集数据
            try:
                end_ns = time.time_ns()
                
                # 解析参数
                sig = inspect.signature(original_func)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                all_args = bound_args.arguments

                actual_batch_size = 0
                input_type = "unknown"
                
                # --- 新增：针对 execute_model 的 scheduler_output 解析 ---
                if "scheduler_output" in all_args:
                    sched_out = all_args["scheduler_output"]
                    # 尝试从 scheduler_output 获取 Token 总数
                    if hasattr(sched_out, "total_num_scheduled_tokens"):
                        actual_batch_size = sched_out.total_num_scheduled_tokens
                        input_type = "scheduler_output"
                    elif hasattr(sched_out, "num_scheduled_tokens"):
                        # 有些版本是字典 {req_id: num}
                        if isinstance(sched_out.num_scheduled_tokens, dict):
                            actual_batch_size = sum(sched_out.num_scheduled_tokens.values())
                        else:
                            # 有些版本是列表或数字
                            try:
                                actual_batch_size = sum(sched_out.num_scheduled_tokens)
                            except:
                                actual_batch_size = -1
                        input_type = "scheduler_output"

                # --- 兼容：针对 _model_forward 的 Tensor 解析 ---
                # 如果没从 scheduler_output 拿到，再试着找 Tensor
                if actual_batch_size == 0:
                    if "input_ids" in all_args and all_args["input_ids"] is not None:
                        actual_batch_size = all_args["input_ids"].shape[0]
                        input_type = "input_ids"
                    elif "intermediate_tensors" in all_args and all_args["intermediate_tensors"] is not None:
                        actual_batch_size = all_args["intermediate_tensors"].shape[0]
                        input_type = "intermediate_tensors"

                req_ids = []
                if hasattr(self, "input_batch") and self.input_batch:
                    if hasattr(self.input_batch, "req_ids"):
                        req_ids = list(self.input_batch.req_ids)

                p_cfg = getattr(self, "parallel_config", None)
                ranks = {}
                if p_cfg:
                    ranks = {
                        "dp": getattr(p_cfg, 'data_parallel_rank', -1),
                        "tp": getattr(p_cfg, 'tensor_parallel_rank', -1)
                    }

                _send_trace({
                    "pid": pid,
                    "tid": tid,
                    "ranks": ranks,
                    "batch_size": actual_batch_size,
                    "input_type": input_type,
                    "hooked_method": target_method_name,
                    "req_ids": req_ids,
                    "start_ns": start_ns,
                    "end_ns": end_ns
                })
            except Exception:
                pass
            
            return res

        setattr(cls, target_method_name, patched_func)
        logger.info(f">>> [HOOK] Successfully patched {target_method_name}")

sys.meta_path.insert(0, VllmWorkerPatcher())