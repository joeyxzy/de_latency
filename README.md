# vLLM Framework Tracing

v0.2

这个仓库用于对 vLLM 服务过程进行联合追踪，包含三条数据链路：

1. Python monkey patch / `sitecustomize` 发出的调度与请求阶段事件
2. `libtracer.so` 基于 CUPTI 采集的 GPU 事件
3. `ebpf_monitor` 基于 eBPF 采集的调度事件

最终这些事件会被 `collector.py` 汇总到日志，再由 `perfetto/log_to_trace.py` 转成 Perfetto 可查看的 `trace.json`。

## 目录结构

```text
de_latency/
├── build/                  # make 产物：controller / ebpf_monitor / libtracer.so
├── include/
├── lib/
├── perfetto/
│   └── log_to_trace.py
├── src/
│   └── FINAL/
│       ├── main.c          # controller
│       ├── sched_latency.c # ebpf monitor
│       ├── sched_latency.bpf.c
│       ├── tracer.cu       # CUPTI tracer
│       └── collector.py    # 日志汇总
├── vllm_trace/
│   ├── start_server.py
│   ├── trace_patch.py
│   ├── sitecustomize.py
│   └── test/
└── Makefile
```

## 功能概览

- `build/controller`
  - 启动 `collector.py`
  - 注入 `libtracer.so`
  - 拉起目标程序
  - 自动启动 `build/ebpf_monitor`
- `build/libtracer.so`
  - 通过 `LD_PRELOAD` 注入
  - 采集 CUDA / CUPTI 事件
- `build/ebpf_monitor`
  - 采集 worker 线程的调度信息
- `src/FINAL/collector.py`
  - 将三路事件写入统一日志
- `perfetto/log_to_trace.py`
  - 将日志转成 Perfetto trace

## 环境依赖

### 系统依赖

至少需要以下包：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential make gcc g++ clang bpftool pkg-config \
  libzmq3-dev libjson-c-dev libbpf-dev libelf-dev zlib1g-dev
```

如果你使用系统默认动态库路径，上面这些就够了；如果你把 ZeroMQ 或 json-c 安装在自定义目录，编译时可通过后文的 `ZMQ_LIB_PATH` 和 `JSONC_LIB_PATH` 指定。

### CUDA / CUPTI

需要：

- NVIDIA Driver
- CUDA Toolkit
- CUPTI

默认 `Makefile` 使用：

```bash
CUDA_INSTALL_PATH=/usr/local/cuda-12.8
CUPTI_INSTALL_PATH=$CUDA_INSTALL_PATH/extras/CUPTI
```

如果你的 CUDA 不在这个路径，编译时显式传参即可：

```bash
make CUDA_INSTALL_PATH=/usr/local/cuda-12.8
```

### Python 环境

推荐在 conda 环境中运行：

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

当前项目验证过的一组核心包版本是：

```bash
pip install \
  vllm==0.11.0 \
  torch==2.8.0 \
  openai==2.6.1 \
  pyzmq==27.1.0 \
  numpy==2.2.6 \
  transformers==4.57.1 \
  tokenizers==0.22.1 \
  triton==3.4.0 \
  xformers==0.0.32.post1 \
  fastapi==0.121.0 \
  uvicorn==0.38.0 \
  pydantic==2.12.3 \
  aiohttp==3.13.2 \
  sentencepiece==0.2.1
```

如果需要运行 `vllm bench`，再补：

```bash
pip install pandas datasets
```

## 编译

在仓库根目录执行：

```bash
conda activate vllm
make clean
make
```

编译成功后会得到：

```text
build/controller
build/ebpf_monitor
build/libtracer.so
```

如果你的库不在系统默认路径，可以这样编译：

```bash
make \
  CUDA_INSTALL_PATH=/usr/local/cuda-12.8 \
  ZMQ_LIB_PATH=/path/to/zeromq/lib \
  JSONC_LIB_PATH=/path/to/json-c/lib
```

## 路径与环境变量

项目已经去掉了对固定用户目录的硬编码，运行时优先读取环境变量。

### 常用环境变量

```bash
export REPO_ROOT=/path/to/de_latency
export PYTHONPATH="$REPO_ROOT/vllm_trace"
export DE_LATENCY_MODEL=/path/to/model
```

### 可选环境变量

```bash
export DE_LATENCY_COLLECTOR_PY="$(which python)"
export DE_LATENCY_COLLECTOR_SCRIPT="$REPO_ROOT/src/FINAL/collector.py"
export DE_LATENCY_EBPF_MONITOR_BIN="$REPO_ROOT/build/ebpf_monitor"
export DE_LATENCY_LIBTRACER_SO="$REPO_ROOT/build/libtracer.so"

export TRACER_ZMQ_ADDR="ipc:///tmp/tracer.sock"
export TRACER_WORKER_PID_FILE="/tmp/tracer_worker_pids"
export DE_LATENCY_LOG_PATH="$REPO_ROOT/perfetto/de_latency.log"
```

### 编译期可选变量

```bash
export ZMQ_LIB_PATH=/path/to/zeromq/lib
export JSONC_LIB_PATH=/path/to/json-c/lib
```

### 这些变量分别控制什么

- `PYTHONPATH`
  - 让 `start_server.py` 能导入 `trace_patch.py`
- `DE_LATENCY_MODEL`
  - 压测脚本默认使用的模型路径或模型名
- `DE_LATENCY_COLLECTOR_PY`
  - `controller` 启动 `collector.py` 时使用的 Python
- `DE_LATENCY_COLLECTOR_SCRIPT`
  - `collector.py` 的位置
- `DE_LATENCY_EBPF_MONITOR_BIN`
  - `ebpf_monitor` 二进制位置
- `DE_LATENCY_LIBTRACER_SO`
  - 需要注入的 `libtracer.so` 位置
- `TRACER_ZMQ_ADDR`
  - Python / CUPTI / eBPF 三路事件的 ZMQ 地址
- `TRACER_WORKER_PID_FILE`
  - worker pid 自动发现文件
- `DE_LATENCY_LOG_PATH`
  - 汇总日志输出文件
- `ZMQ_LIB_PATH` / `JSONC_LIB_PATH`
  - 编译时追加的库目录

### 默认行为

如果不设置这些变量：

- `controller` 会优先复用你传给它的 Python 解释器来启动 `collector.py`
- `controller` 会按自身所在目录自动寻找：
  - 同目录下的 `ebpf_monitor`
  - 同目录下的 `libtracer.so`
  - 上一级 `src/FINAL/collector.py`
- ZMQ 地址默认使用 `ipc:///tmp/tracer.sock`
- worker pid 文件默认使用 `/tmp/tracer_worker_pids`
- 日志默认写到 `perfetto/de_latency.log`

## 快速开始

### 1. 准备环境

```bash
conda activate vllm
export REPO_ROOT=/path/to/de_latency
export PYTHONPATH="$REPO_ROOT/vllm_trace"
export DE_LATENCY_MODEL=/path/to/model
```

### 2. 编译

```bash
cd "$REPO_ROOT"
make clean
make
```

### 3. 启动 trace 版 vLLM 服务

```bash
cd "$REPO_ROOT"
sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --model "$DE_LATENCY_MODEL" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8001
```

说明：

- `controller` 会自动拉起 `collector.py`
- `controller` 会自动注入 `libtracer.so`
- `controller` 会自动拉起 `ebpf_monitor`
- 日志默认写入 `perfetto/de_latency.log`

### 4. 发送请求测试

可以直接用测试脚本：

```bash
cd "$REPO_ROOT/vllm_trace/test"
python request_test.py
```

也可以自己调用 OpenAI 兼容接口。

### 5. 转换为 Perfetto trace

```bash
cd "$REPO_ROOT/perfetto"
python log_to_trace.py de_latency.log trace.json
```

## 运行方式

### A. 只启动带 patch 的 vLLM 服务

这种方式只加载 Python 侧 patch，不启用 controller / eBPF / CUPTI：

```bash
cd "$REPO_ROOT/vllm_trace"
python start_server.py \
  --model "$DE_LATENCY_MODEL" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8001
```

### B. 启动完整 trace 链路

推荐方式：

```bash
cd "$REPO_ROOT"
sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --model "$DE_LATENCY_MODEL" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8001
```

### C. 指定可见 GPU

```bash
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=1,2 sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --model "$DE_LATENCY_MODEL" \
  --tensor-parallel-size 2 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8001
```

## Bench 模式

`vllm_trace/start_server.py` 支持两种模式：

- `serve`
- `bench`

默认是 `serve`。可以用命令行或环境变量切换。

### bench latency

```bash
cd "$REPO_ROOT"
sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --mode bench \
  --bench-subcommand latency \
  --model "$DE_LATENCY_MODEL" \
  --tensor-parallel-size 1 \
  --input-len 32 \
  --output-len 1 \
  --max-model-len 4096 \
  --enforce-eager \
  --load-format dummy
```

### bench serve

```bash
cd "$REPO_ROOT"
sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --mode bench \
  --bench-subcommand serve \
  --backend vllm \
  --model "$DE_LATENCY_MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 200
```

### 环境变量方式切换 bench

```bash
export DE_LATENCY_VLLM_MODE=bench
export DE_LATENCY_BENCH_SUBCOMMAND=serve
```

然后直接运行：

```bash
cd "$REPO_ROOT"
sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --backend vllm \
  --model "$DE_LATENCY_MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 200
```

## 压测脚本

测试脚本默认读取：

```bash
DE_LATENCY_MODEL
```

示例：

```bash
export DE_LATENCY_MODEL=/path/to/model
cd "$REPO_ROOT/vllm_trace/test"
python request_test.py
python highpres_test.py
python high_para.py
```

## 纯净启动对照

如果你想对照没有 trace 的原始 vLLM：

```bash
vllm serve "$DE_LATENCY_MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 4096
```

## 常见问题

### 1. 换机器后找不到 `collector.py` / `ebpf_monitor` / `libtracer.so`

优先检查：

```bash
echo "$DE_LATENCY_COLLECTOR_SCRIPT"
echo "$DE_LATENCY_EBPF_MONITOR_BIN"
echo "$DE_LATENCY_LIBTRACER_SO"
```

如果没设置，默认要求你已经执行过 `make`，并且使用的是仓库里的 `build/controller`。

### 2. `trace_patch.py` 导入失败

说明 `PYTHONPATH` 没有包含 `vllm_trace`：

```bash
export PYTHONPATH="$REPO_ROOT/vllm_trace"
```

### 3. 自定义 ZeroMQ / json-c 目录无法链接

编译前设置：

```bash
export ZMQ_LIB_PATH=/path/to/zeromq/lib
export JSONC_LIB_PATH=/path/to/json-c/lib
make clean
make
```

### 4. eBPF 相关功能无法工作

通常需要：

- Linux 环境
- root 权限
- 可用的 eBPF / BTF 支持
- `bpftool`、`clang`、`libbpf`

## 一条最常用的完整命令

```bash
conda activate vllm
export REPO_ROOT=/path/to/de_latency
export PYTHONPATH="$REPO_ROOT/vllm_trace"
export DE_LATENCY_MODEL=/path/to/model

cd "$REPO_ROOT"
make clean && make

sudo -E "$REPO_ROOT/build/controller" \
  "$(which python)" \
  "$REPO_ROOT/vllm_trace/start_server.py" \
  --model "$DE_LATENCY_MODEL" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8001
```
