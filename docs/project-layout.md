# 项目布局与发布清单

更新：2026-11-02。布局按职责划分，不再围绕旧 demo 和历史路径保留兼容包装。

## 1. 目录职责

```text
.
├── de_latency/                 Python 产品代码，唯一实现位置
│   ├── cli.py                  serve / ebpf / analyze / overview
│   ├── __main__.py             python -m de_latency
│   ├── collector.py            三路事件接收、JSONL、worker 发现与统计
│   ├── protocol.py             事件协议、校验和历史日志解码
│   ├── runtime/                session、发送、控制和请求上下文
│   ├── instrumentation/        vLLM 0.20 插桩与服务启动
│   │   ├── launch.py           API 服务启动，原参数透传
│   │   ├── startup/            采集进程专用 sitecustomize
│   │   ├── workers/            EngineCore / executor / GPU / PP / scheduler
│   │   └── requests/           请求生命周期 / callback / 输出轮次
│   ├── analysis/               离线归因及 TTFT/TPOT 分解
│   └── export/                 Perfetto、gzip、概览、数值报告与图表
├── native/                     自研 C / C++ / CUDA / eBPF
│   ├── controller/             服务进程组、注入与停止排空
│   ├── cupti/                  GPU 活动、markers、控制与发送
│   ├── ebpf/                   内核调度探针与用户态 monitor
│   └── include/                自研共享头：tracer_comm.h / sched_event.h
├── third_party/                第三方材料，不与自研源码混放
│   ├── libbpf/include/bpf/     已有 libbpf 头文件
│   ├── libbpf/lib/             已有动态库与软链接，系统依赖的回退
│   └── linux/vmlinux.h         历史生成头；当前构建不使用
├── benchmarks/                 客户端、自动验收、回归与专项实验
├── tools/                      绘图与手动请求示例
│   └── examples/               聊天、并发、压力、抢占示例
├── tests/                      单元与生命周期测试
├── docs/                       当前文档
│   └── history/                历史环境与实验清理审计记录
├── Makefile                    原生构建
├── pyproject.toml              Python 包、依赖和 console 入口
├── README.md                   安装与使用入口
└── .gitignore                  本地生成物与数据隔离
```

这里的“双语言分区”是有意的：`de_latency/` 是 Python 包，`native/` 是本项目原生实现，`third_party/` 是外部依赖；它们不是三份重复产品代码。根目录不再放零散启动、分析、绘图或兼容脚本。

## 2. Python 各模块负责什么

### `runtime/`：基础设施

| 文件 | 作用 |
|---|---|
| `session.py` | session 环境、服务参数、进程身份、控制目标和排空边界 |
| `transport.py` | 线程独立 Python PUSH socket、fork 后重置 |
| `control.py` | Python 采集运行时开关服务 |
| `socket_client.py` | Unix socket 控制客户端 |
| `request_context.py` | 请求 ID/名称/上下文 |
| `control_cli.py`、`cupti_cli.py` | 开发用低层控制命令；不是额外产品实现 |

### `instrumentation/`：只做插桩和启动适配

| 文件/目录 | 作用 |
|---|---|
| `launch.py` | 安装可选 API hooks，运行 vLLM API server；不运行 benchmark |
| `startup/sitecustomize.py` | spawn worker 的自动 hooks；只有采集环境才把该目录加入 PYTHONPATH |
| `bootstrap.py`、`registry.py` | 0.20 版本门禁、延迟导入和幂等包装 |
| `markers.py`、`execution_context.py` | GPU marker 桥接、线程隔离的嵌套 dispatch 上下文 |
| `scheduler_metadata.py`、`task_events.py` | 调度元信息和共享任务事件 |
| `output_loop.py` | 对照 vLLM 0.20 输出循环的插桩实现 |
| `worker_hooks.py`、`request_hooks.py` | 按组件安装 hooks 的薄入口 |
| `workers/context.py`、`entry.py`、`engine_core.py` | worker 状态、就绪/执行边界、EngineCore 阶段 |
| `workers/executor.py`、`gpu_runner.py`、`parallel.py`、`scheduler.py` | dispatch、GPU execute/sample、PP 通信、调度决策 |
| `requests/state.py`、`scheduling.py` | task/context 注册和 callback 调度 |
| `requests/lifecycle.py`、`stages.py`、`output.py`、`rounds.py` | generate 生命周期、阶段包装、输出循环和轮次归因 |

### `analysis/`：离线数值，不依赖 vLLM/GPU/root

| 文件 | 作用 |
|---|---|
| `pipeline.py`、`model.py` | 阶段编排和分析结果 |
| `ingestion.py`、`normalization.py`、`requests.py` | 日志分类、标准化、历史请求 ID/别名 |
| `clocks.py`、`intervals.py` | 时间单位和区间运算 |
| `worker_intervals.py` | worker 状态机及 PP 请求关联 |
| `request_timing.py`、`request_stages.py` | 生命周期、协程/输出任务和请求阶段 |
| `dispatch_windows.py`、`dispatch.py`、`dispatch_queues.py` | 窗口索引、RPC/ready/marker 关联和队列间隔 |
| `parallel_dispatch.py`、`wall_time.py` | TP 合并、PP gap 和重叠墙钟分配 |
| `correlation.py` | CUPTI correlation key / launch epoch |
| `os_scheduling.py`、`request_index.py` | eBPF 时钟校准、OS 等待、请求集合与区间投影 |
| `ttft.py`、`tpot.py`、`breakdown.py` | TTFT/TPOT 独占区间分解与汇总 |
| `helpers.py` | 已有 Python helper 导出；不承担独立分析流水线 |

### `export/`：显示和序列化，不回写归因

| 文件 | 作用 |
|---|---|
| `bundle.py`、`trace_io.py`、`overview.py` | 全量/sidecar 输出、原子 gzip、流式概览 |
| `perfetto.py`、`styles.py`、`lanes.py` | 事件构造、显示修正、颜色和轨道 |
| `request_layout.py`、`requests.py` | 请求布局、生命周期、排队/执行和对齐视图 |
| `workers.py`、`core.py` | worker、PP、EngineCore、scheduler 轨道 |
| `gpu.py`、`os_scheduling.py` | GPU/API/flow 与 event-loop OS 轨道 |
| `reports.py` | 逐请求 SVG/HTML |

每层的 `__init__.py` 是包边界，不应当作空文件随意删除。

## 3. 原生、实验、工具和文档

### 原生代码

- `native/controller/main.c`：管理本次服务进程组；默认用 `python -m de_latency.collector` 启动 collector。
- `native/cupti/tracer.cu`：生命周期和策略；`activity.*` 解码活动，`transport.*` 发送/排空，`control_server.*` 提供控制服务，`markers.*` 生成 GPU marker，`serialize.*` 序列化，`support.*` 提供环境/日志辅助。
- `native/ebpf/sched_latency.bpf.c`、`sched_latency.c`：内核探针及用户态加载/目标筛选/发送。
- `native/include/`：自研的通信与调度事件定义。
- `third_party/`：保留原第三方材料及版权头；详见其 README。移动文件不等于已解决再分发许可。

### 实验与工具

| 路径 | 作用 |
|---|---|
| `benchmarks/run_baseline_client.py` | 用户原客户端与负载定义，未改写 |
| `benchmarks/validate.py`、`acceptance.py` | 自动真实验收及请求/来源/丢包/归因闭合检查 |
| `benchmarks/check_vllm_semantics.py`、`check_cupti_controls.py` | vLLM 语义对照与 CUDA 控制测试 |
| `benchmarks/replay_regression.py` | 历史日志数值回归 |
| `benchmarks/experiments.py`、`gpu_monitor.py` | CPU 核数实验及 GPU CSV |
| `benchmarks/ebpf_session.sh` | 自动验收使用的独立 eBPF 启动 |
| `benchmarks/backpressure.sh` | `max_pending_dispatches` 专项实验，需相应参数/定制实现 |
| `tools/plot_results.py` | 实验绘图，另需 matplotlib、numpy |
| `tools/examples/*.py` | 非正式手动请求/压力示例，负载不是当前 benchmark |

### 测试与文档

- `tests/`：协议、发送/collector、controller、CLI 生命周期、hooks、上下文、元信息、归因、TP/PP、OS、gzip/概览和验收；`test_layout.py` 额外检查包内启动和 spawn 自动插桩路径。
- `docs/commands.md`：日常完整命令；`architecture.md`：模块边界；`protocol.md`：事件格式；`validation.md`：验收记录；`functional-parity.md`：功能范围和入口变更。
- `docs/history/environment-2026-04-13.md`：旧机器/vLLM 0.11 环境快照，不是当前安装说明。
- `docs/history/cleanup-validation-2026-10-30.json`：旧中间 session 清理审计；里面的路径描述历史，不需要随源码搬迁改写。

## 4. 入口迁移：不再保留旧路径

| 原路径 | 现在使用 |
|---|---|
| `vllm_trace/start_server.py`、`trace_patch.py` | 产品用 `de-latency serve`；内部为 `python -m de_latency.instrumentation.launch` |
| `vllm_trace/sitecustomize.py` | `de_latency/instrumentation/startup/sitecustomize.py`，由 session 自动配置 |
| `vllm_trace/trace_ctl.py` | `python -m de_latency.runtime.control_cli` |
| `vllm_trace/cupti_ctl.py` | `python -m de_latency.runtime.cupti_cli` |
| `vllm_trace/monkeypatch_runtime.py`、`request_context.py` | 直接导入 `de_latency.runtime.control` / `request_context`，删除转发壳 |
| `src/FINAL/collector.py` | `python -m de_latency.collector` |
| `perfetto/log_to_trace.py` | `python -m de_latency.analysis.pipeline events.log trace.json.gz` |
| `run_experiments.py` | `python -m benchmarks.experiments` |
| 根目录 `plot_results.py`、`bench_gpu_backpressure.sh` | `tools/plot_results.py`、`benchmarks/backpressure.sh` |
| `vllm_trace/test/`、聊天示例 | `tools/examples/` |
| 根目录 `include/`、`lib/` | 自研头到 `native/include/`；第三方头/库到 `third_party/` |

产品命令 `de-latency serve/ebpf/analyze/overview` 不变；全量采集、默认 `trace.json.gz`、可选概览和数值 sidecar 命名不变。已有实验目录和数据没有随源码布局迁移，历史 manifest 的命令路径只作为记录，不用于启动新 session。

## 5. 哪些不提交、哪些能清理

| 路径 | 性质 | 清理原则 |
|---|---|---|
| `build/` | 原生二进制、`.o`/`.d`、BPF skeleton、打包中间文件 | 停止本项目服务/构建后可删；采集前重新 `make -j4` |
| `de_latency.egg-info/` | 安装元信息 | 可重建；之后 `pip install -e .` |
| `__pycache__/`、`.pyc` | Python 缓存 | 可删 |
| `dist/`、`.pytest_cache/` | 打包/测试产物 | 可删 |
| `plots/`、调试 alignment JSON | 生成视图 | 有源数据且可重建时可删 |
| `bench_results/` | 原始采集、配置、统计、分析及回归基线 | 不提交，不默认删 |
| `results/` | 客户端指标、纯净对照和专项实验结果 | 不提交，不默认删 |
| `.log`、历史 trace | 可能是唯一原始数据 | 不能只看后缀就删，先确认备份/回归需求 |
| `third_party/linux/vmlinux.h` | 未用于当前构建的历史生成头 | 可作为后续精简候选，本轮只迁移保留 |
| `.git/` | Git 历史和配置 | 不删除 |

这些本地结果不属于提交的源码树。旧的 `perfetto/`、`tmp/` 若本机还有内容，也按数据/调试资料处理，不再作为产品代码入口。此次重排期间发现旧 Perfetto 数据、压缩脚本和 tmp 已不在工作区；没有恢复或覆盖这些外部变化。

## 6. GitHub 发布检查

1. **许可仍需确认**：根目录尚无项目许可证；不能替第三方代码任意授权。核对 libbpf、vLLM 示例及其他上游实现来源、版权和许可文本。
2. **第三方二进制**：明确是否继续分发 `third_party/libbpf/lib/`，或另行改成仅系统依赖；不能把移动文件误当作依赖/许可问题已解决。
3. **本机信息**：`docs/history/` 和命令文档有本机路径/环境信息，公开前确认是否脱敏。未对全部 Git 历史做凭据或大对象审计。
4. **别漏掉新文件**：大量新源码还未跟踪，`git commit -am ...` 不会添加它们；源码搬迁的新增与删除要一起审阅。
5. **忽略不等于删除历史**：大实验数据无需为了上传而从本机删除；`.gitignore` 也不能清除已存在于提交历史中的大文件或秘密。JSON、gzip、HTML、SVG 的全局忽略规则可能影响未来想提交的示例；文档历史 JSON 已有例外。
6. **不要盲目回滚删除记录**：此前重构删除的 demo、本轮移走的旧入口，以及工作期间外部删除的文件需分别识别，不能用 reset/revert 整体覆盖。

只查看的提交预览：

```bash
git status --short
git add --dry-run -A
git diff --check
git diff --cached --stat
git diff --cached --name-status
```

更新环境并检查：

```bash
pip install -e .
make -j4
PYTHONDONTWRITEBYTECODE=1 DE_LATENCY_DISABLE_SITECUSTOMIZE=1 \
  python -m unittest discover -s tests -v
```

提交和推送仍需审阅后执行；源码整理本身不执行 commit/push，也不扩大到实验数据清理。
