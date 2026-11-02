# 辅助工具与手动示例

这些脚本不参与默认服务启动、采集和数值归因。在仓库根目录执行：

| 工具 | 用途 |
|---|---|
| `python tools/plot_results.py [results_dir]` | 绘制 `python -m benchmarks.experiments` 的 CPU 核数/GPU 利用率实验结果；另需 matplotlib、numpy |
| `python tools/examples/chat_client.py` | OpenAI 聊天请求示例，保留 vLLM 上游许可标记 |
| `python tools/examples/batching_requests.py` | 手动并发请求示例 |
| `python tools/examples/concurrent_stress.py` | 长文本并发压力实验 |
| `python tools/examples/preemption_stress.py` | 预生成 prompts，尝试制造抢占 |

请求示例的模型、端口、并发与当前正式 benchmark 不同，使用前检查脚本配置。用户现有客户端仍是 `benchmarks/run_baseline_client.py`，没有改写。

日常入口统一用 `de-latency serve/ebpf/analyze/overview`；低层控制命令是 `python -m de_latency.runtime.control_cli` 和 `python -m de_latency.runtime.cupti_cli`。
