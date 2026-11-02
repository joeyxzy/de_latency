# 常用指令：Qwen3-32B / QPS=2

以下命令在仓库根目录、`vllm-0200` 环境执行。服务、eBPF、客户端和分析分别运行；工具不替你发请求。

## 0. 首次安装或更新依赖

```bash
cd /home/joeyxzy/de_latency/de_latency
conda activate vllm-0200
pip install -e .
make -j4
```

之后使用 `de-latency`。未注册 console 命令时可使用 `python -m de_latency` 替代。每个终端都使用同一个 Python 环境。

## 1. 插桩服务：终端 A

先确认 GPU 可用、8000 端口没有旧服务。`bench_results/qps2-01` 必须不存在，重复实验请修改 session 名称；不能复用已有的 `bench_results/qps2`。

```bash
de-latency serve --session bench_results/qps2-01 --gpu 0,1,2,3 -- \
  ~/models/Qwen3-32B \
  --served-model-name qwen3-32b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

默认开启 Python hooks 和 CUPTI 全量活动输出（仍保留既有查询过滤规则），collector 自动启动。等 `[de-latency] READY` 后再发请求。不需要 export；`--` 后的服务参数原样透传。

若需要关闭组件，在 `--` **之前**加 `--python off` 或 `--cupti off`。这不是概览开关；关闭组件会失去相应来源，影响完整逐请求归因。导出概览不需要关闭任何采集源。

## 2. 可选：独立 eBPF，终端 B

```bash
de-latency ebpf --session bench_results/qps2-01
```

需要时仅为 monitor 请求 sudo；服务和 collector 不以 root 运行。等 `eBPF Tracer Running` 后再执行三路实验。若不启动它，本轮只有 Python+CUPTI，没有 eBPF OS 等待归因。

## 3. 原 benchmark：终端 C

客户端脚本和负载参数不变：100 请求、输入 1024、输出 256、忽略 EOS、流式、泊松到达、seed=42。

```bash
python benchmarks/run_baseline_client.py \
  --model ~/models/Qwen3-32B \
  --qps 2 \
  --num-prompts 100 \
  --input-len 1024 \
  --output-len 256 \
  --seed 42 \
  --result-dir results/qps2
```

本轮请求全部结束后，在 **终端 A 按 Ctrl+C**，等待 serve 完全退出。CUPTI、Python、独立 eBPF 和 collector 会按顺序停止/排空，不需要手动操作各 worker。

## 4. 分析：默认全量、默认 gzip

仅生成全量压缩时间线及数值结果：

```bash
de-latency analyze bench_results/qps2-01
```

全量之外，同时提供压缩概览（推荐日常使用这个命令）：

```bash
de-latency analyze bench_results/qps2-01 --overview
```

主要产物：

| 文件 | 内容 |
|---|---|
| `trace.json.gz` | 全量时间线；完整保留导出事件和字段，真正 gzip 压缩 |
| `trace.overview.json.gz` | 可选概览；请求、worker、调度、PP、OS 和 GPU marker，省略普通 kernel/API/flow |
| `trace.compute_breakdown.json` | TTFT/TPOT 数值分解，不因视图筛选而改变 |
| `trace.summary.json` | 分析汇总 |
| `trace.alignment.json` | 时钟对齐调试视图 |
| `trace.compute_breakdown_requests/index.html` | 逐请求图表 |

小型数值 sidecar 保持普通 JSON，方便读取和比较。两个 `.json.gz` 时间线都可拖入 Perfetto；全量 gzip 的事件数没有减少，浏览器仍可能需要大量内存。概览只是显示过滤，不是 markers-only 采集。

## 5. 已有大文件：只补概览，不重新分析日志

对已有的 `bench_results/qps2`：

```bash
de-latency overview bench_results/qps2
```

优先读取 `trace.json.gz`，没有则读取旧的 `trace.json`，流式生成 `trace.overview.json.gz`。源文件、`events.log`、数值结果和图表不动。

也可直接指定文件或自定义输出：

```bash
de-latency overview bench_results/qps2/trace.json.gz \
  --output bench_results/qps2/trace.preview.json.gz
```

若确实需要未压缩 JSON（例如兼容其他工具）：

```bash
de-latency analyze bench_results/qps2-01 \
  --output bench_results/qps2-01/trace.json
```

重新导出会更新同名分析产物，但不会自动删除旧 `.json`、原始日志或采集记录。

## 6. 纯净对照：相同 benchmark、不加载插桩

先正常停止插桩服务和 eBPF，释放同一端口及 GPU。在终端 A 执行：

```bash
env -u LD_PRELOAD -u PYTHONPATH \
  DE_LATENCY_DISABLE_SITECUSTOMIZE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve ~/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching
```

服务就绪后，在另一个终端执行相同客户端，只分开存储结果：

```bash
env -u LD_PRELOAD -u PYTHONPATH \
  DE_LATENCY_DISABLE_SITECUSTOMIZE=1 \
  python benchmarks/run_baseline_client.py \
    --model ~/models/Qwen3-32B \
    --qps 2 \
    --num-prompts 100 \
    --input-len 1024 \
    --output-len 256 \
    --seed 42 \
    --result-dir results/qps2-clean
```

纯净对照不启动 collector/eBPF，也不执行工具的分析命令；直接比较客户端结果。两轮都要核对实际成功请求数、输入/输出 token 数和服务参数，不能把采集开销与不同工作负载混为一谈。
