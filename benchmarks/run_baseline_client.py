#!/usr/bin/env python3
"""调用 vLLM 自带的 benchmark，运行固定长度、泊松到达的请求。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, help="本地模型路径")
parser.add_argument("--qps", type=float, default=1, help="平均每秒请求数")
parser.add_argument("--num-prompts", type=int, default=1000, help="请求总数")
parser.add_argument("--input-len", type=int, default=1024, help="输入 token 数")
parser.add_argument("--output-len", type=int, default=256, help="输出 token 数")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--port", type=int, default=8000, help="服务端口")
parser.add_argument("--result-dir", default="results/baseline", help="结果保存目录")
args = parser.parse_args()

# 客户端不需要安装我们工具的自动插桩。
env = os.environ.copy()
env["DE_LATENCY_DISABLE_SITECUSTOMIZE"] = "1"

# 使用当前 Python 环境里的 vLLM，连接本机已经启动的服务。
# served-model-name 需要与服务端的 --served-model-name 一致。
command = [
    sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
    "--backend", "openai",
    "--host", "127.0.0.1",
    "--port", str(args.port),
    "--endpoint", "/v1/completions",
    "--model", str(Path(args.model).expanduser()),
    "--served-model-name", "qwen3-32b",

    # random 数据集：不添加共享前缀，输入和输出长度不随机波动。
    "--dataset-name", "random",
    "--random-input-len", str(args.input_len),
    "--random-output-len", str(args.output_len),
    "--random-range-ratio", "0",
    "--random-prefix-len", "0",

    # 忽略 EOS，生成指定数量的 token；使用贪心采样。
    # 此 benchmark 后端默认发送流式请求，用于统计 TTFT 和 ITL。
    "--ignore-eos",
    "--temperature", "0",

    # burstiness=1 表示泊松到达；QPS 是平均到达率。
    "--request-rate", str(args.qps),
    "--burstiness", "1",
    "--num-prompts", str(args.num_prompts),
    "--seed", str(args.seed),

    # 在终端显示进度，并保存汇总和逐请求结果。
    # 不指定文件名，由 vLLM 自动生成带时间戳的结果文件。
    "--percentile-metrics", "ttft,tpot,itl,e2el",
    "--metric-percentiles", "50,95,99",
    "--save-result",
    "--save-detailed",
    "--result-dir", args.result_dir,
]

# 不自动预热、不管理服务，仅执行这一轮压测。
subprocess.run(command, env=env, check=True)
