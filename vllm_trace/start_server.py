# start_server.py (v2 - Correct version using runpy)

# ======================================================================
# 1. 导入补丁模块 (这一步保持不变，且至关重要)
# ======================================================================
print("✅ [PATCH_LOADER] Loading custom vLLM patch...")
#import vllm_patch
import trace_patch
print("✅ [PATCH_LOADER] Patch loaded successfully.")


# ======================================================================
# 2. 导入 runpy 和 sys，准备执行 vLLM 的主模块
# ======================================================================
import runpy
import sys

def start_patched_server():
    print("🚀 [PATCH_LOADER] Starting patched vLLM server via runpy...")

    # 3. 设置命令行参数
    #    runpy.run_module 会像命令行一样读取 sys.argv
    #    我们需要确保 sys.argv 包含了我们想传递的所有参数
    #    第一个参数应该是被执行的模块的路径，后面跟着所有的命令行标志
    original_argv = sys.argv
    sys.argv = [
        # runpy 会用这个作为脚本名
        "vllm.entrypoints.openai.api_server",
        # 从原始命令行中继承所有参数 (除了我们自己的脚本名 'start_server.py')
        *original_argv[1:]
    ]

    # 4. 使用 runpy.run_module 来执行 vLLM 的入口点
    #    这和在命令行运行 `python -m vllm.entrypoints.openai.api_server` 的效果几乎完全一样
    #    但它是在我们的补丁已经被加载之后才执行的
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    start_patched_server()