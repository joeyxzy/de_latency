# vLLM latency decomposition

联合采集 Python 请求/调度阶段、CUPTI GPU 活动和 eBPF CPU 调度等待，输出 JSONL、Perfetto trace 与逐请求 TTFT/TPOT 分解。

**支持 vLLM 0.20.x。** 当前验收基线是本机 `0.20.2.dev0+g132765e35.d20260512`，Python 3.12、PyTorch 2.11.0+cu128、CUDA 12.8。插桩依赖 vLLM 内部接口；升级 0.20 源码后也应重新运行语义检查和真实 benchmark。

## 项目结构

```text
de_latency/       Python 产品代码：CLI、collector、插桩、归因、导出
native/           C/C++/CUDA/eBPF 源码及共享头文件
third_party/      第三方头文件、库与历史生成头，和自研代码分开
benchmarks/       benchmark 客户端、自动验收及专项实验
tools/            绘图和手动请求示例，不参与产品启动
tests/            单元与生命周期测试
docs/             使用、架构、协议与验收文档；history/ 保存旧记录
build/            编译产物（不提交）
bench_results/    本机采集与分析结果（不提交）
results/, plots/  本机客户端结果与图片（不提交）
```

产品入口统一为 `de-latency` 或 `python -m de_latency`。不再维护旧路径包装脚本；目录职责与迁移表见 [项目布局](docs/project-layout.md)。

## 安装与构建

在已有 vLLM 0.20 环境中执行：

```bash
pip install -e .
make -j4
```

系统依赖：C/C++ 编译器、make、clang（BPF 后端）、bpftool、libbpf、libelf、zlib、ZeroMQ、json-c 和 CUDA/CUPTI 开发文件。Ubuntu 可安装：

```bash
sudo apt-get install build-essential clang bpftool pkg-config \
  libbpf-dev libelf-dev zlib1g-dev libzmq3-dev libjson-c-dev
```

CUDA 路径默认 `/usr/local/cuda-12.8`，可用 `make CUDA_INSTALL_PATH=/path/to/cuda` 覆盖。产物名称保持为：

ZeroMQ/json-c 安装在自定义目录时，构建先从 pkg-config、再从编译器的库搜索路径获取位置并写入 RUNPATH。也可显式指定 `ZMQ_LIB_PATH` 和 `JSONC_LIB_PATH`；这样 sudo 清理 `LD_LIBRARY_PATH` 后 eBPF monitor 仍能加载依赖。已有产物需要用 `make -B build/ebpf_monitor` 重新链接。

```text
build/controller
build/ebpf_monitor
build/libtracer.so
```

## 日常使用：服务、eBPF、benchmark、分析各自独立

不需要手工 export，不会替你运行 benchmark，也不会改写已有 benchmark 命令。

完整的插桩/纯净对照和分析命令见 [指令速查](docs/commands.md)。下列示例的 session 目录必须不存在；已有 `qps2` 数据请勿复用该名字。

### 1. 启动自己的服务

把原来的 `vllm serve MODEL ...` 换为以下入口。`--` 后保留原有模型和服务参数：

```bash
de-latency serve --session bench_results/qps2-01 --gpu 0,1,2,3 -- \
  ~/models/Qwen3-32B \
  --served-model-name qwen3-32b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 \
  --max-model-len 8192 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --no-enable-prefix-caching
```

默认安装 Python hooks，并在服务健康、worker 就绪后启用 CUPTI；collector 自动启动。看到 `[de-latency] READY` 后可发请求。不会额外发送预热/测试请求，不会添加 eager、显存比例、长度或并行度等隐式服务参数。

启动时在 `--` 前添加 `--python off` 或 `--cupti off`，可分别关闭对应组件。Python off 不安装 hooks；CUPTI off 不由工具预加载 tracer。只开启部分来源时不能获得完整的逐请求 TTFT/TPOT 归因。若已有未插桩的服务占用同一端口，需要先由其所有者正常停止并重启；工具不会接管或杀掉它。

### 2. 可选：另一终端单独启动 eBPF

```bash
de-latency ebpf --session bench_results/qps2-01
```

命令仅为 native monitor 请求 sudo，不会以 root 运行服务或 collector。看到 `eBPF Tracer Running` 后再运行需要三路采集的 benchmark。本 session 的 monitor 可在服务启动过程中启动，也可以在 READY 后启动；服务结束时自动停止。

### 3. 原有 benchmark 命令保持不变

例如继续执行你自己的客户端：

```bash
python benchmarks/run_baseline_client.py \
  --model ~/models/Qwen3-32B --qps 2 --num-prompts 100 \
  --input-len 1024 --output-len 256 --seed 42 \
  --result-dir results/qps2
```

工具不改客户端、不替换数据集/到达分布、不自动开始或结束压测。benchmark 完成后，在 **serve 终端按 Ctrl+C**；工具停止采集、保存 native 终态计数、停止 eBPF，并排空 collector。等命令完全退出。

### 4. 一个命令转换分析

```bash
de-latency analyze bench_results/qps2-01
```

**默认仍导出全量事件，主时间线改为真正 gzip 压缩的 `trace.json.gz`，不是只改文件后缀。** 时间线直接写入 gzip，不先落一份巨大的未压缩 JSON。数值结果继续使用 `trace.compute_breakdown.json`、`trace.summary.json`，请求图表继续使用 `trace.compute_breakdown_requests/index.html`。运行中的 session 会被拒绝分析，避免读取仍在追加的日志。

需要同时提供浏览器易打开的压缩概览时：

```bash
de-latency analyze bench_results/qps2-01 --overview
```

它在全量 `trace.json.gz` 之外额外生成 `trace.overview.json.gz`。概览保留请求、Python/worker、scheduler、PP 通信、OS 轨道和 GPU markers，仅省略普通 GPU kernel/内存操作、CUDA API 及 flow 箭头；**不改变全量采集、原始日志或 TTFT/TPOT 归因**。gzip 本身只节约文件体积，不减少全量 trace 的事件数与浏览器解析成本。

如果已有全量 trace，只补概览，不重读 `events.log`、不重新计算分解：

```bash
de-latency overview bench_results/qps2
# 也可直接指定 trace.json 或 trace.json.gz：
de-latency overview bench_results/qps2/trace.json
```

session 输入优先读取 `trace.json.gz`，兼容原有 `trace.json`；输出默认为同目录 `trace.overview.json.gz`。此路径流式读取大 trace，不会 `json.load` 整个文件，也不会改写源文件。明确指定 `analyze --output path.json` 时仍可得到未压缩 JSON；默认 `.json.gz` 和显式 `.json.gz` 都是真 gzip。程序不会自动删除以前的未压缩文件。

session 目录必须不存在，重复实验请换名字；目录路径不能过长（Unix socket 限制）。`manifest.json` 记录原始服务参数、组件开关和采集退出情况；`cupti_off.json`、`events.log.stats.json`、`ebpf.stats.json` 保存终态统计。手动服务模式不生成自动压测的 `benchmark.json` 或严格验收的 `validation.json`，也不把转换成功等同于采集无损。

若刚更新源码，请重新执行 `pip install -e .` 注册命令；也可把 `de-latency` 换为 `python -m de_latency`。目前默认就绪检查针对本机 HTTP `/health` 服务；HTTPS 使用 Python 默认 CA 校验，自签名证书需另行配置可信 CA。

## 开发者自动验收（自动启动 server 和 benchmark）

### Python + CUPTI

```bash
export DE_LATENCY_MODEL=/path/to/model
DE_LATENCY_DISABLE_SITECUSTOMIZE=1 python benchmarks/validate.py \
  --model "$DE_LATENCY_MODEL" --gpu 0 \
  --session bench_results/eager-01 --ebpf off
```

默认 eager、16 个请求、64 输入 token、32 输出 token。session 目录必须不存在，且路径足够短以容纳 Unix socket。脚本使用独立的端口、socket、日志、目标文件和进程组。

CUDA Graph 与多卡示例：

```bash
DE_LATENCY_DISABLE_SITECUSTOMIZE=1 python benchmarks/validate.py \
  --model "$DE_LATENCY_MODEL" --gpu 0 --graph \
  --gpu-memory-utilization 0.65 --session bench_results/graph-01 --ebpf off

DE_LATENCY_DISABLE_SITECUSTOMIZE=1 python benchmarks/validate.py \
  --model "$DE_LATENCY_MODEL" --gpu 0,1,2,3 --tp 2 --pp 2 \
  --gpu-memory-utilization 0.35 --session bench_results/tp2pp2-01 --ebpf off
```

显存比例需要按模型和可用显存调整。GPU 使用数字索引，runner 设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，并将 GPU UUID 写入 manifest 便于核对。

### 三路采集：单独授权 eBPF

终端 A（普通用户）：

```bash
DE_LATENCY_DISABLE_SITECUSTOMIZE=1 python benchmarks/validate.py \
  --model "$DE_LATENCY_MODEL" --gpu 0 \
  --session bench_results/three-01 --ebpf external
```

在 runner 等待 eBPF ready 时，在终端 B 输入密码授权：

```bash
sudo bash benchmarks/ebpf_session.sh bench_results/three-01
```

helper 只跟踪该 session 的目标，看到 `stop` 文件后退出。若当前终端已具有可用的 sudo 授权，可以使用 `--ebpf sudo`，runner 将执行 `sudo -n`。

### 验收产物

每个 session 包含：

- `manifest.json`：模型、设备、并行度、采集窗口。
- `benchmark.json`：vLLM benchmark 原始指标。
- `events.log`、`events.log.stats.json`：schema v2 JSONL 与 collector 统计。
- `cupti_on.json`、`cupti_off.json`：每个 worker 的控制响应及最终丢包计数。
- `ebpf.stats.json`：启用 eBPF 时的最终 kernel ring/map/send 丢包统计。
- `trace.json.gz`：全量 gzip Perfetto trace；可在 <https://ui.perfetto.dev/> 打开。旧 session 的 `trace.json` 仍受支持。
- `trace.overview.json.gz`：使用 `analyze --overview` 或独立 `overview` 命令额外生成的轻量视图。
- `trace.compute_breakdown.json`、`trace.summary.json`、`trace.alignment.json`。
- `trace.compute_breakdown_requests/index.html`：逐请求图表。
- `validation.json`：验收结果；失败时进程返回非零。

验收要求全部请求成功、各 worker kernel/marker 完整、无上报丢包、TTFT/TPOT 分量闭合且执行时间为正、Perfetto JSON 可解析。Graph 模式还要求真实 GraphLaunch 记录；TP/PP 要求所有 rank 的阶段记录；eBPF 要求实际请求调度等待归因。

## 开发与低层入口

普通使用只需要 `de-latency`。需要独立运行组件或开发控制时：

```bash
python -m de_latency.analysis.pipeline events.log trace.json.gz
python -m de_latency.runtime.cupti_cli --help
python -m de_latency.runtime.control_cli --help
python -m benchmarks.experiments --help
```

collector 使用 `python -m de_latency.collector`；API 服务启动使用 `python -m de_latency.instrumentation.launch`。spawn worker 的自动插桩文件位于 `de_latency/instrumentation/startup/sitecustomize.py`，只有采集进程的 `PYTHONPATH` 才加入这个专用目录，不会在仓库根目录放一个全局自动执行的 `sitecustomize.py`。

手动运行 controller（此示例关闭 eBPF 自动启动）：

```bash
PYTHONPATH="$PWD/de_latency/instrumentation/startup:$PWD" TRACER_EBPF_START_ENABLED=0 \
  build/controller "$(command -v python)" -m de_latency.instrumentation.launch \
  --model "$DE_LATENCY_MODEL" --enforce-eager --port 8001
```

多 session 建议使用上述 runner。手动运行需分别设置 `TRACER_ZMQ_ADDR`、`TRACER_WORKER_PID_FILE`、`DE_LATENCY_LOG_PATH`、`DE_LATENCY_CONTROL_STATE_FILE` 及控制 socket 目录。

## 运行时配置

| 变量 | 用途 |
|---|---|
| `DE_LATENCY_SESSION_ID` | 三路事件共享的 session 标识 |
| `DE_LATENCY_DISABLE_SITECUSTOMIZE=1` | 禁用自动插桩，用于 benchmark 客户端、分析和测试 |
| `TRACER_ZMQ_ADDR` | 三路共用的 ZeroMQ 地址 |
| `DE_LATENCY_LOG_PATH` | collector JSONL 路径 |
| `TRACER_WORKER_PID_FILE` | worker PID 与 `tid:<native_tid>` 目标文件 |
| `TRACER_CUPTI_CONTROL_DIR` / `TRACER_MONKEYPATCH_CONTROL_DIR` | 控制 socket 目录 |
| `TRACER_CUPTI_START_ENABLED=0` | 启动时暂停 CUPTI，模型就绪后启用 |
| `TRACER_CUPTI_CAPTURE_QUERIES=1` | 保留设备信息/状态查询记录；默认过滤以降低采集流量 |
| `TRACER_DEBUG_LOG_PATH` | CUPTI 调试日志；runner 使用 session 内路径 |
| `DE_LATENCY_MARKER_DEBUG=1` | 记录 GPU marker 调用错误 |

默认过滤的 API 列表见 `native/cupti/activity.cpp:is_query`。过滤量单独报告为 `filtered_queries`，不计为丢包；kernel、marker、launch、同步、分配和数据传输仍采集。完整查询采集的高负载零丢包尚未验收。

## 检查与架构

```bash
python -m unittest discover -s tests -v
DE_LATENCY_DISABLE_SITECUSTOMIZE=1 python benchmarks/check_vllm_semantics.py
CUDA_VISIBLE_DEVICES=0 DE_LATENCY_DISABLE_SITECUSTOMIZE=1 python benchmarks/check_cupti_controls.py
python benchmarks/replay_regression.py historical.log baseline.compute_breakdown.json replay.json
```

语义检查直接对照当前安装的 vLLM 原实现，覆盖正常输出、取消、异常及输出循环，不加载模型权重。CUPTI 控制检查需要 GPU，但不加载模型，验证 on/off/flush/markers-only 与重启后的真实 CUDA 事件。数值回归比较逐请求分解、汇总、缺失项和 prefill re-entry。

新采集的 TTFT 分解使用 version 5 的独占墙钟区间，避免 OS 等待重复叠加及 worker 执行窗口越过首 token；旧 v1 日志保留 version 4 的历史数值口径。

- [模块与扩展边界](docs/architecture.md)
- [目录用途、清理建议与 GitHub 发布清单](docs/project-layout.md)
- [统一事件协议](docs/protocol.md)
- [本机验收记录](docs/validation.md)
- [功能兼容范围](docs/functional-parity.md)
