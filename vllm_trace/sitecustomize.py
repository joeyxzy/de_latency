import sys
import importlib
import logging
import os
import threading
import inspect
import time
import zmq
import json
from typing import Dict, List, Any
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRACE] %(message)s")
logger = logging.getLogger("VLLM_HOOK")

# =============================================================================
# Part 1: 基础设施层 (Infrastructure)
# 负责：ZMQ通信、Hook注册表、通用模块拦截器
# =============================================================================

class TraceSender:
    """
    负责将采集到的数据发送给 Log 收集进程 (C++ Collector 或其他消费者)。
    使用 thread_local 确保多线程下的 ZMQ Socket 安全。
    """
    _thread_local = threading.local()
    # ZMQ 地址，需与接收端保持一致
    ZMQ_ADDR = "ipc:///tmp/tracer.sock"

    @classmethod
    def get_socket(cls):
        if not hasattr(cls._thread_local, "sock"):
            try:
                ctx = zmq.Context()
                sock = ctx.socket(zmq.PUSH)
                # LINGER=0 确保关闭时不会挂起等待发送
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(cls.ZMQ_ADDR)
                cls._thread_local.sock = sock
            except Exception as e:
                # 仅报错一次，避免刷屏
                if not hasattr(cls._thread_local, "error_logged"):
                    logger.error(f"ZMQ Init Failed: {e}")
                    cls._thread_local.error_logged = True
                cls._thread_local.sock = None
        return cls._thread_local.sock

    @classmethod
    def emit(cls, event_type, payload, source="monkey_patch"):
        """
        发送 Trace 事件。
        flags=zmq.DONTWAIT: 关键设置，如果队列满则丢弃，绝对不阻塞 vLLM 推理线程。
        """
        sock = cls.get_socket()
        if sock:
            try:
                meta = {
                    "source": source,
                    "event_type": event_type,
                    "timestamp_ns": time.time_ns(),
                    "payload": payload
                }
                # 发送两帧：Header(JSON) + Empty Body
                meta_bytes = json.dumps(meta).encode("utf-8")
                sock.send_multipart([meta_bytes, b""], flags=zmq.DONTWAIT)
            except zmq.error.Again:
                # 队列满，丢弃日志，不阻塞业务
                pass
            except Exception:
                pass

# --- Hook 注册表 ---
HOOK_REGISTRY = {}

def register_hook(module_name):
    """装饰器：将函数注册为指定模块的 Patch 逻辑"""
    def decorator(func):
        HOOK_REGISTRY[module_name] = func
        return func
    return decorator

def apply_method_patch(cls, method_name, wrapper_factory):
    """
    辅助工具：将 wrapper 应用到类的方法上。
    wrapper_factory: 接收 (original_func) 返回 (patched_func) 的闭包工厂。
    """
    if not hasattr(cls, method_name):
        # 很多时候 vLLM 版本差异导致函数名不同，这里记录 warn 即可，不要 crash
        logger.debug(f"Method '{method_name}' not found in {cls.__name__}, skipping.")
        return False

    original_func = getattr(cls, method_name)
    patched_func = wrapper_factory(original_func)
    setattr(cls, method_name, patched_func)
    logger.info(f"Successfully patched: {cls.__name__}.{method_name}")
    return True

# =============================================================================
# [升级版] 鲁棒的通用拦截器 (Proxy Pattern)
# 解决嵌套导入时 Hook 失效的问题
# =============================================================================

class PatchingLoader:
    """代理加载器：包装原始 Loader，在加载完成后执行 Hook"""
    def __init__(self, original_loader, patch_func):
        self.original_loader = original_loader
        self.patch_func = patch_func

    # 代理所有未定义的方法给原始 Loader (如 get_source, is_package 等)
    def __getattr__(self, name):
        return getattr(self.original_loader, name)

    def create_module(self, spec):
        if hasattr(self.original_loader, 'create_module'):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        # 1. 执行原始加载逻辑 (此时 sys.meta_path 里依然有 Patcher，嵌套导入会被捕获)
        if hasattr(self.original_loader, 'exec_module'):
            self.original_loader.exec_module(module)
        
        # 2. 模块加载完毕，执行插桩
        try:
            self.patch_func(module)
        except Exception as e:
            logger.error(f"Patch failed for {module.__name__}: {e}", exc_info=True)

class VllmUniversalPatcher(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # 1. 只处理注册过的模块
        if fullname not in HOOK_REGISTRY:
            return None
        
        # 2. 遍历 sys.meta_path 寻找真正的加载器 (跳过自己)
        for finder in sys.meta_path:
            if finder is self:
                continue
            
            try:
                # 让后续的 Finder 去找 Spec
                spec = finder.find_spec(fullname, path, target)
            except AttributeError:
                continue
                
            if spec and spec.loader:
                # 3. 偷梁换柱：把 Loader 换成我们的代理 Loader
                spec.loader = PatchingLoader(spec.loader, HOOK_REGISTRY[fullname])
                return spec
        return None



@register_hook("vllm.v1.worker.gpu_model_runner")
def patch_gpu_model_runner(module):
    """
    针对 vLLM GPU Worker 的核心插桩逻辑。
    目标类: GPUModelRunner
    """
    if not hasattr(module, "GPUModelRunner"):
        logger.warning(f"GPUModelRunner not found in {module}")
        return

    cls = module.GPUModelRunner
    method_names = [m for m in dir(cls) ]
    class ModelProxy:
        def __init__(self, model):
            self._model = model

        def __call__(self, *args, **kwargs):
            # [Timeline Point] Forward 开始
            start_ns = time.time_ns()
            TraceSender.emit(
                event_type="gpu_forward_start",
                payload={
                    "pid": os.getpid(),
                    "tid": getattr(threading, "get_native_id", threading.get_ident)(),
                    "hooked_method": "model.__call__",
                    "timestamp_ns": start_ns,
                }
            )
            
            try:
                # 执行真正的模型 Forward
                return self._model(*args, **kwargs)
            finally:
                # [Timeline Point] Forward 结束 (这也是 Postprocess 的开始)
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="gpu_forward_end",
                    payload={
                        "pid": os.getpid(),
                        "tid": getattr(threading, "get_native_id", threading.get_ident)(),
                        "timestamp_ns": end_ns,
                        # 可选：如果你想直接看 forward 耗时，可以把 start_ns 也带上
                        # "duration_ns": end_ns - start_ns 
                    }
                )

        def __getattr__(self, name):
            return getattr(self._model, name)
    #定义excute_model
    def excute_model_wrapper(original_func):
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_model_wrapped"):
                if hasattr(self, "model") and self.model is not None:
                    self.model = ModelProxy(self.model)
                    self._model_wrapped = True
                    logger.info(f">>> [HOOK] GPUModelRunner.model wrapped with ModelProxy")
                else:
                    # 安全兜底
                    logger.warning("GPUModelRunner instance has no 'model' attribute yet")
            # [Stage 1] 记录开始信息
            start_ns = time.time_ns()
            pid = os.getpid()
            try:
                tid = threading.get_native_id()
            except AttributeError:
                tid = threading.get_ident()

            # [Stage 2] 执行原函数 (这是真正的 GPU 推理触发点)
            res = original_func(self, *args, **kwargs)
            
            # [Stage 3] 采集数据 (放在 try 块中以防分析逻辑崩溃影响推理)
            try:
                end_ns = time.time_ns()
                
                # --- 参数内省 (Introspection) ---
                # 使用 inspect 绑定参数，无论用户是位置参数还是关键字参数调用都能拿到
                sig = inspect.signature(original_func)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                all_args = bound_args.arguments

                actual_batch_size = 0
                input_type = "unknown"
                
                # 尝试从 scheduler_output 解析 Batch Size
                if "scheduler_output" in all_args:
                    sched_out = all_args["scheduler_output"]
                    # vLLM 不同版本属性名可能不同，做兼容处理
                    if hasattr(sched_out, "total_num_scheduled_tokens"):
                        actual_batch_size = sched_out.total_num_scheduled_tokens
                        input_type = "scheduler_output_v1"
                    elif hasattr(sched_out, "num_scheduled_tokens"):
                        val = sched_out.num_scheduled_tokens
                        if isinstance(val, dict):
                            actual_batch_size = sum(val.values())
                        elif isinstance(val, (list, tuple)):
                            actual_batch_size = sum(val)
                        else:
                            actual_batch_size = int(val)
                        input_type = "scheduler_output_v2"

                # 兜底：尝试从 Tensor 解析
                if actual_batch_size == 0:
                    for key in ["input_ids", "intermediate_tensors"]:
                        if key in all_args and all_args[key] is not None:
                            # 假设第一维是 batch
                            actual_batch_size = getattr(all_args[key], "shape", [0])[0]
                            input_type = f"tensor_{key}"
                            break

                # 获取 Request IDs (上下文关联的关键)
                req_ids = []
                if hasattr(self, "input_batch") and self.input_batch:
                    if hasattr(self.input_batch, "req_ids"):
                        req_ids = list(self.input_batch.req_ids)

                # 获取并行配置 (DP/TP Rank)
                ranks = {"dp": -1, "tp": -1}
                p_cfg = getattr(self, "parallel_config", None)
                if p_cfg:
                    ranks["dp"] = getattr(p_cfg, 'data_parallel_rank', -1)
                    ranks["tp"] = getattr(p_cfg, 'tensor_parallel_rank', -1)

                # 发送数据
                TraceSender.emit(
                    event_type="gpu_execute_model",
                    payload={
                        "pid": pid,
                        "tid": tid,
                        "ranks": ranks,
                        "batch_size": actual_batch_size,
                        "input_type": input_type,
                        "hooked_method": original_func.__name__,
                        "req_ids": req_ids,
                        "start_ns": start_ns,
                        "end_ns": end_ns
                    }
                )

            except Exception:
                # 生产环境建议 pass，调试时可 logger.error
                pass
            
            return res
        return wrapper
    
    def sample_wrapper(original_func):
        def wrapper(self, *args, **kwargs):
            start_ns = time.time_ns()
            # 执行原有的 _sample
            try:
                return original_func(self, *args, **kwargs)
            finally:
                # 即使报错也要记录结束时间
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="gpu_sample",
                    payload={
                        "pid": os.getpid(),
                        "tid": getattr(threading, "get_native_id", threading.get_ident)(),
                        "start_ns": start_ns,
                        "end_ns": end_ns
                    }
                )
        return wrapper
    
    def bookkeeping_wrapper(original_func):
        def wrapper(self, *args, **kwargs):
            start_ns = time.time_ns()
            # 执行原有的 _bookkeeping_sync
            # 这里的耗时 = GPU计算剩余时间 + 数据拷贝时间
            try:
                return original_func(self, *args, **kwargs)
            finally:
                end_ns = time.time_ns()
                TraceSender.emit(
                    event_type="gpu_bookkeeping",
                    payload={
                        "pid": os.getpid(),
                        "tid": getattr(threading, "get_native_id", threading.get_ident)(),
                        "start_ns": start_ns,
                        "end_ns": end_ns
                    }
                )
        return wrapper
    
    apply_method_patch(cls, "execute_model", excute_model_wrapper)
    apply_method_patch(cls, "_sample", sample_wrapper)
    apply_method_patch(cls, "_bookkeeping_sync", bookkeeping_wrapper)

@register_hook("vllm.v1.core.sched.scheduler") 
def patch_v1_scheduler(module):
    # 再次确认类名，防止报错
    if not hasattr(module, "Scheduler"):
        logger.warning(f"Scheduler class not found in {module.__name__}")
        return
    
    cls = module.Scheduler
    logger.info(f">>> [HOOK] Patching Scheduler Implementation: {cls}")

    def add_request_wrapper(original_func):
        def wrapper(self, request):
            # 获取 Request ID
            req_id = getattr(request, "request_id", "unknown")
            
            # 发送入队事件
            TraceSender.emit(
                event_type="req_enqueue_scheduler",
                payload={
                    "req_id": req_id,
                    "event": "enqueue_start",
                    "timestamp_ns": time.time_ns()
                }
            )
            return original_func(self, request)
        return wrapper
    #这个函数帮助我们将update_from_output函数返回的dict[int, EngineCoreOutputs]中的每个req的时间戳事件提取出来
    def extract_events_from_results(results_dict: Dict[int, Any]) -> List[Dict]:
        """
        针对确定的 vLLM V1 结构提取事件
        Input: dict[int, EngineCoreOutputs]
        Output: List of event payloads
        """
        extracted_updates = []
        # 1. 遍历字典 (key 是 engine_index)
        for engine_id, core_outputs_batch in results_dict.items():
            # core_outputs_batch 就是 EngineCoreOutputs 对象
            # 2. 获取该 batch 下的所有请求输出列表
            # 根据定义: outputs: list[EngineCoreOutput] = []
            if not hasattr(core_outputs_batch, "outputs"):
                continue
            #request_outputs_list=outputs: outputs: list[EngineCoreOutput] = []
            request_outputs_list = core_outputs_batch.outputs
            # 3. 遍历每个请求的输出
            for req_output in request_outputs_list:
                # req_output 就是 EngineCoreOutput 对象
                
                # A. 必须要有 events 才有发送价值
                # 定义: events: Optional[list[EngineCoreEvent]] = None
                if not hasattr(req_output, "events") or not req_output.events:
                    continue
                # B. 解析事件列表
                parsed_events = []
                for evt in req_output.events:
                    # EngineCoreEvent 通常有 type 和 timestamp 属性
                    # 有些版本可能是 tuple，做个防御性读取
                    e_type = getattr(evt, "type", None)
                    e_ts = getattr(evt, "timestamp", None)
                    
                    # 如果是 Enum，转字符串
                    if e_type is not None:
                        e_type = str(e_type)

                    if e_type and e_ts:
                        parsed_events.append({
                            "type": e_type,
                            "ts": e_ts
                        })
                
                # C. 如果提取到了事件，加入列表
                if parsed_events:
                    extracted_updates.append({
                        "req_id": getattr(req_output, "request_id", "unknown"),
                        "events": parsed_events,
                        # 还可以带上 engine_id 方便调试
                        "engine_id": engine_id 
                    })

        return extracted_updates

    def update_from_output_wrapper(original_func):
        def wrapper(self, scheduler_output, model_output):
            # 1. 先执行原逻辑 (确保状态更新完毕)
            res = original_func(self, scheduler_output, model_output)
            
            # 2. 采集时间戳：这就是 "Ready for Next Step" 的时间
            # 我们需要知道哪些请求 Ready 了，这就在 scheduler_output 里
            req_ids = []
            try:
                # 复用之前的提取逻辑
                if hasattr(scheduler_output, "scheduled_new_reqs"):
                    for req in scheduler_output.scheduled_new_reqs:
                        if hasattr(req, "request_id"): req_ids.append(req.request_id)
                        elif hasattr(req, "req_id"): req_ids.append(req.req_id)

                if hasattr(scheduler_output, "scheduled_cached_reqs"):
                    cached = scheduler_output.scheduled_cached_reqs
                    if cached and hasattr(cached, "req_ids"):
                        req_ids.extend(cached.req_ids)
            except:
                pass

            if req_ids:
                TraceSender.emit(
                    event_type="req_step_ready", # 事件名：这一步跑完了，准备好跑下一步了
                    payload={
                        "req_ids": req_ids,
                        "event": "step_ready",
                        "timestamp_ns": time.time_ns(),
                    }
                )
            try:
                req_events = extract_events_from_results(res)
                if req_events:
                    #这里发送的是在这一轮update_from_output中多个req的多个时间戳时间
                    #需要在后续进行解析，将每个req的时间平铺
                    TraceSender.emit(
                        event_type="req_metrics_events",
                        payload={
                            "req_events": req_events,
                        }
                    )
            except Exception as e:
                print(f"[Probe Error] Extraction failed: {e}")

            return res
        return wrapper
    
    apply_method_patch(cls, "update_from_output", update_from_output_wrapper)
    apply_method_patch(cls, "add_request", add_request_wrapper)

#某些请求被调度出并rpc的时间点插桩（1.调度结束时间点 2.enginecore向worker广播的开始时间点）
@register_hook("vllm.v1.executor.multiproc_executor")
def patch_v1_executor(module):
    if not hasattr(module, "MultiprocExecutor"):
        return
    
    cls = module.MultiprocExecutor
    # logger.info(f">>> [HOOK] Patching Executor: {cls}")

    def execute_model_wrapper(original_func):
        def wrapper(self, scheduler_output, *args, **kwargs):
            # --- [开始采集] ---
            req_ids = []
            try:
                # 1. 提取 Prefill 阶段的新请求 (这是一个 List)
                # SchedulerOutput.scheduled_new_reqs: list[NewRequestData]
                if hasattr(scheduler_output, "scheduled_new_reqs"):
                    # 这里依然需要遍历，因为每个 NewRequestData 是独立的
                    for req in scheduler_output.scheduled_new_reqs:
                        # 尝试获取 request_id
                        if hasattr(req, "request_id"):
                            req_ids.append(req.request_id)
                        elif hasattr(req, "req_id"):
                            req_ids.append(req.req_id)

                # 2. 提取 Decode 阶段的缓存请求 (这是一个 Object)
                # SchedulerOutput.scheduled_cached_reqs: CachedRequestData
                if hasattr(scheduler_output, "scheduled_cached_reqs"):
                    cached_data = scheduler_output.scheduled_cached_reqs
                    # 确保不是 None (有些时候可能为空)
                    if cached_data is not None:
                        # 直接获取里面的 req_ids 列表，不需要遍历 cached_data
                        if hasattr(cached_data, "req_ids"):
                            req_ids.extend(cached_data.req_ids)

            except Exception:
                # 生产环境保持静默，避免影响性能
                pass

            # 发送数据
            if req_ids:
                TraceSender.emit(
                    event_type="req_scheduler_out_rpc",
                    payload={
                        "req_ids": req_ids,
                        "event": "execute_start", # 代表 RPC 发送前一瞬间
                        "batch_size": len(req_ids),
                        "timestamp_ns": time.time_ns()
                    }
                )
            # --- [结束采集] ---

            return original_func(self, scheduler_output, *args, **kwargs)
        return wrapper

    apply_method_patch(cls, "execute_model", execute_model_wrapper)


@register_hook("vllm.v1.worker.gpu_worker")
def patch_worker_base(module):
    # 修改目标为 WorkerBase
    target_cls_name = "Worker"
    
    if not hasattr(module, target_cls_name):
        logger.warning(f"⚠️ [HOOK FAIL] {target_cls_name} not found in {module.__name__}")
        return
    
    cls = getattr(module, target_cls_name)
    logger.info(f">>> [HOOK SUCCESS] Patching Worker Entry: {cls.__name__}")

    def execute_model_wrapper(original_func):
        def wrapper(self, scheduler_output, *args, **kwargs):
            # --- [采集点：Worker 进程入口] ---
            # 无论这是不是 Wrapper，这里都是 Worker 收到任务的第一时间
            start_ns = time.time_ns()
            try:
                tid = threading.get_native_id()
            except AttributeError:
                tid = threading.get_ident()
            # [DEBUG] 确认被调用
            #print(f">>> [WORKER DEBUG] WorkerBase.execute_model HIT! PID={os.getpid()}")

            req_ids = []
            try:
                # 提取 req_ids (保持之前的逻辑)
                if hasattr(scheduler_output, "scheduled_new_reqs"):
                    for req in scheduler_output.scheduled_new_reqs:
                        if hasattr(req, "request_id"): req_ids.append(req.request_id)
                        elif hasattr(req, "req_id"): req_ids.append(req.req_id)

                if hasattr(scheduler_output, "scheduled_cached_reqs"):
                    cached = scheduler_output.scheduled_cached_reqs
                    if cached is not None and hasattr(cached, "req_ids"):
                        req_ids.extend(cached.req_ids)
            except:
                pass
            
            if req_ids:
                TraceSender.emit(
                    event_type="worker_preprocess_start",
                    payload={
                        "req_ids": req_ids,
                        "event": "preprocess_start",
                        "timestamp_ns": start_ns,
                        "pid": os.getpid(),
                        "tid": tid
                    }
                )
            # ---------------------------------------
            start_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
            try:
                return original_func(self, scheduler_output, *args, **kwargs)
            finally:
                end_mono = int(time.clock_gettime_ns(time.CLOCK_MONOTONIC))
                if req_ids:
                    TraceSender.emit(
                        event_type="worker_model_execute_span",
                        payload={
                            "req_ids": req_ids,
                            "pid": os.getpid(),
                            "tid": tid,
                            "start_ns": start_mono,
                            "end_ns": end_mono,
                            "duration_us": (end_mono - start_mono) // 1000
                        }
                    )
        return wrapper

    # 尝试 Hook execute_model
    if hasattr(cls, "execute_model"):
        apply_method_patch(cls, "execute_model", execute_model_wrapper)
    else:
        logger.warning(f"⚠️ [HOOK FAIL] execute_model not found in {target_cls_name}")


# --- (示例) 可以在这里添加更多模块的 Hook ---
# @register_hook("vllm.core.scheduler")
# def patch_scheduler(module):
#     ...


#整个程序的入口，一旦执行了这个insert的进程都会在每次import的时候自动执行上方的find_spec
#如果find_spec返回成功则执行exec_module方法
sys.meta_path.insert(0, VllmUniversalPatcher())

logger.info(">>> [TRACE SYSTEM] VllmUniversalPatcher installed. Ready to intercept.")