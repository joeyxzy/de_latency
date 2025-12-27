import os
import sys
import runpy

print("✅ [PATCH_LOADER START...]")
#引入monkeypatch模块，这个模块会被执行，执行完vllm的相关函数就被修改了
import trace_patch
print("✅ [PATCH_LOADER SUCCESS!]")


def start_server():
    # 保持原来的 API server 启动逻辑
    original_argv = sys.argv
    sys.argv = ["vllm.entrypoints.openai.api_server", *original_argv[1:]]
    #patch只对当前这个进程有效，所以在这里直接运行模块启动vllm，这个修改才可见
    #但是可以预知的，vLLM的后续启动的进程是无法感知到这个patch的
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
        print("[WARN] vllm.entrypoints.cli.main not found, trying vllm.scripts.vllm")
        runpy.run_module("vllm.scripts.vllm", run_name="__main__")

# ============================================================
# 4. 根据模式选择启动
# ============================================================
if __name__ == "__main__":
    #正常启动vllm服务
    start_server()
    #运行latency的benchmark(取消注释开启)
    #start_bench()
