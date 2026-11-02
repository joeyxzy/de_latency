# 功能范围与入口迁移

本次重构以 vLLM 0.20 本机源码为运行基线。0.11 兼容和旧 demo 的移除是明确的范围调整。

2026-11-02 按职责统一布局后，不再保留旧文件路径兼容；保留的是采集、分析和数值输出行为。产品 Python 代码统一在 `de_latency/`，原生代码在 `native/`。旧入口到新模块的对应关系见 [项目布局](project-layout.md)。

| 重构前能力 | 当前实现与检查 |
|---|---|
| make 构建三个原生产物 | `build/controller`、`build/ebpf_monitor`、`build/libtracer.so` |
| 启动、控制、收集、转换 | `de-latency`、`de_latency.instrumentation.launch`、`de_latency.runtime.*_cli`、`de_latency.collector`、`de_latency.analysis.pipeline`；原文件路径取消 |
| Python 请求、调度、worker 阶段 | instrumentation 注册表、请求和 worker hooks；语义对照与真实采集 |
| CUPTI kernel/API/传输/marker | native/cupti；默认过滤查询，`TRACER_CUPTI_CAPTURE_QUERIES=1` 可恢复完整查询记录 |
| eBPF 调度等待 | native/ebpf；实时三路日志与请求 OS delay 归因 |
| eager、CUDA Graph、TP/PP | 真实 benchmark 已分别覆盖，见 validation.md |
| 历史日志与 TTFT/TPOT 分解 | 约 197 万条 v1 历史事件数值回归；逐请求、汇总、缺失项、re-entry 一致；v2 的重复 OS 等待和窗口越界修正有独立版本标识 |
| Perfetto、JSON sidecar、逐请求图表 | `AnalysisResult` 和独立 export 模块 |
| CPU 配额实验与 GPU 曲线数据 | `python -m benchmarks.experiments`、独立 session runner、兼容 GPU CSV；绘图在 `tools/plot_results.py` |
| runtime on/off/status/flush/markers-only | 参数与控制协议保留，命令迁入 runtime 模块；CUDA 控制实测覆盖重启、模式恢复、排空、客户端提前断开；服务端错误返回失败退出码 |

结构拆分与功能兼容分别验收：worker/request 入口只安装按组件划分的 hooks；CUPTI 的 lifecycle、activity、transport、control server、marker 和 serializer 各自独立；分析按 ingestion、worker、request timing、dispatch/window/queue、stage、OS 和 breakdown 交换具名结果；所有轨道构建归属 export。阶段接口及扩展边界见 architecture.md。功能兼容仍由行为与输出回归判断，不能仅凭文件变小宣称完成。

受测模型为 Qwen1.5-4B-Chat。已覆盖单卡 eager、单卡 CUDA Graph、TP=2/PP=2 eager，以及真实 Python+CUPTI+eBPF 联合采集（含 128 请求、约 797 万条事件的有界负载）。当前检查不证明任意模型、所有 0.20 提交或无限时长负载下的兼容性。完整查询模式经过小负载 CUDA 控制测试，但其高负载零丢包尚未验收。
