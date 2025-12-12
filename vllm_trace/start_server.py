import os
import sys
import runpy

print(f"✅ [PATCH_LOADER] LD_PRELOAD = {os.getenv('LD_PRELOAD')}")
print("✅ [PATCH_LOADER] Loading custom vLLM patch...")
import trace_patch
print("✅ [PATCH_LOADER] Patch loaded successfully.")


def start_server():
    # 保持原来的 API server 启动逻辑
    original_argv = sys.argv
    sys.argv = ["vllm.entrypoints.openai.api_server", *original_argv[1:]]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")

def start_bench():
    # 获取外部传入的参数 (如 --model ...)
    # sys.argv[0] 是脚本本身路径，sys.argv[1:] 是参数
    args = sys.argv[1:]
    
    # 2. 构造伪造的命令行参数
    # 我们要模拟执行: vllm bench latency [args...]
    # 所以 argv[0] 是 'vllm' (或者是入口模块名), 后面紧跟 'bench', 'latency', 然后是原来的参数
    sys.argv = ["vllm", "bench", "latency"] + args
    
    print(f"✅ [BENCH_RUNNER] Invoking vLLM CLI with args: {sys.argv}")
    
    # 3. 调用 vLLM 的 CLI 主入口
    # 'vllm' 命令实际上对应的就是 vllm.entrypoints.cli.main
    try:
        runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
    except ImportError:
        # 兼容旧版本 vLLM，如果上面的路径不对，可能是 vllm.scripts.vllm
        print("⚠️ [WARN] vllm.entrypoints.cli.main not found, trying vllm.scripts.vllm")
        runpy.run_module("vllm.scripts.vllm", run_name="__main__")

# ============================================================
# 4. 根据模式选择启动
# ============================================================
if __name__ == "__main__":
    #正常启动vllm服务
    start_server()
    #运行latency的benchmark(取消注释开启)
    #start_bench()
