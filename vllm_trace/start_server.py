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

def start_bench(subcommand="latency"):
    # 获取外部传入的参数 (如 --model ...)
    # sys.argv[0] 是脚本本身路径，sys.argv[1:] 是参数
    args = sys.argv[1:]
    
    # 2. 构造伪造的命令行参数
    # 我们要模拟执行: vllm bench <subcommand> [args...]
    # 所以 argv[0] 是 'vllm' (或者是入口模块名), 后面紧跟 'bench', '<subcommand>', 然后是原来的参数
    sys.argv = ["vllm", "bench", subcommand] + args
    
    print(f"✅ [BENCH_RUNNER] Invoking vLLM CLI with args: {sys.argv}")
    
    # 3. 调用 vLLM 的 CLI 主入口
    # 'vllm' 命令实际上对应的就是 vllm.entrypoints.cli.main
    try:
        runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
    except ImportError:
        # 兼容旧版本 vLLM，如果上面的路径不对，可能是 vllm.scripts.vllm
        print("[WARN] vllm.entrypoints.cli.main not found, trying vllm.scripts.vllm")
        runpy.run_module("vllm.scripts.vllm", run_name="__main__")


def _pick_mode(argv):
    # 环境变量优先，便于一键切换
    env_mode = os.getenv("DE_LATENCY_VLLM_MODE", "").strip().lower()
    if env_mode in {"serve", "bench"}:
        return env_mode

    # 支持显式参数：python start_server.py --mode bench ...
    if len(argv) >= 3 and argv[1] == "--mode":
        mode = argv[2].strip().lower()
        if mode in {"serve", "bench"}:
            del argv[1:3]
            return mode

    # 默认行为：服务模式
    return "serve"


def _pick_bench_subcommand(argv):
    # 环境变量优先，便于无侵入切换
    env_subcmd = os.getenv("DE_LATENCY_BENCH_SUBCOMMAND", "").strip().lower()
    if env_subcmd:
        return env_subcmd

    # 显式参数：python start_server.py --bench-subcommand serve ...
    for flag in ("--bench-subcommand", "--bench-mode"):
        if len(argv) >= 3 and argv[1] == flag:
            subcmd = argv[2].strip().lower()
            if subcmd:
                del argv[1:3]
                return subcmd

    # 默认保持原行为
    return "latency"

# ============================================================
# 4. 根据模式选择启动
# ============================================================
if __name__ == "__main__":
    mode = _pick_mode(sys.argv)
    print(f"✅ [MODE] start_server.py mode={mode}")
    if mode == "bench":
        bench_subcommand = _pick_bench_subcommand(sys.argv)
        print(f"✅ [MODE] bench subcommand={bench_subcommand}")
        start_bench(subcommand=bench_subcommand)
    else:
        start_server()
