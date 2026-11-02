# 本机验收记录

最后更新：2026-11-02。重构验收模型：`Qwen1.5-4B-Chat`；GPU：RTX 3090。

环境：Python 3.12.12、vLLM `0.20.2.dev0+g132765e35.d20260512`、torch `2.11.0+cu128`、pyzmq 27.1.0、CUDA 12.8。

清理说明：按用户选择，已删除被最终验收替代的 12 个成功中间 session，包含下表的 `final-live`、`complete-graph` 及回归使用过的 `split-workers`。它们的验收结果、manifest 和 collector 统计保存在 [清理验收快照](history/cleanup-validation-2026-10-30.json)，原始日志与 trace 不再保留。该次清理保留了最新三次成功验收、128 请求的 `complete-tp`、全部失败/未完成 session 和原有历史数据；记录描述的是当时的状态，不保证后续手动清理后文件仍存在。

## 收尾结果

| 检查 | 结果 |
|---|---|
| `make -B -j4` | controller / CUPTI / eBPF 全量重建成功；自定义库目录同时写入 C 与 NVCC 产物 RUNPATH |
| 无 `LD_LIBRARY_PATH` 的 eBPF 启动 | `ebpf_monitor --help` 加载成功，不依赖 sudo 保留用户环境 |
| 单元/生命周期/验收门禁测试 | 见下方结构收尾复验；原有 29 项继续保留 |
| 本地 vLLM 语义对照 | 7 个 generate 场景、2 个 output-loop 场景通过 |
| 原生 CUDA 控制检查 | on/off/flush、markers-only、模式恢复、重启、客户端提前断开通过；默认过滤及完整查询模式均有小负载检查 |
| 历史 v1 数值回归 | 1,973,432 条事件、11 请求；请求/汇总/缺失/re-entry 完全一致，包含最新 GPU 导出抽取 |
| 单卡实时三路：`bench_results/final-live` | 16/16 请求；5,979 Python、247,570 CUPTI、365 eBPF 事件；103,899 kernel、800 marker；终态丢包全 0 |
| 最终 GPU 导出复验：`bench_results/final-live/final.validation.json` | 通过；与同次采集原分析的逐请求数值及 summary 一致，重复 stream metadata 已去重 |
| CUDA Graph：`bench_results/complete-graph` | 16/16 请求；131,108 kernel、1,103 marker、257 GraphLaunch；CUPTI 终态丢包全 0 |
| 四卡实时三路：`bench_results/complete-tp` | TP=2/PP=2、128/128 请求；109,887 Python、7,861,224 CUPTI、3,965 eBPF 事件；4 worker 的 kernel/marker 与 rank 覆盖完整；CUPTI/eBPF 终态丢包全 0 |
| Python wheel | `pip wheel --no-deps --no-build-isolation` 构建成功；新增分析及插桩模块包含在包中 |

真实采集门禁检查：请求全部成功；collector 终态统计与落盘源计数一致；每个 worker 都有 kernel/marker；请求 TTFT/TPOT 分量闭合且有正执行归因；eBPF 有请求 OS delay 归因；Perfetto JSON 有有限、非负的完整区间。Graph 模式另检查真实 GraphLaunch。

128 请求是约 797 万条事件的有界负载检查，不等同于数小时稳定性或任意模型覆盖。完整查询模式在高负载下的零丢包未验收。Python 线程最后一次发送失败后立即退出时，附带于后续事件的计数仍存在可观测性边界，详见 architecture.md。

## 结构收尾复验

本轮将 native、worker/request hooks 和分析阶段实际拆开，不能用上表拆分前的通过结果替代本轮验收。

- `bench_results/refactor-three`：最新 TP=2/PP=2 实时 Python+CUPTI+eBPF，32/32 请求；27,395 Python、2,031,579 CUPTI、1,395 eBPF 事件；496,514 kernel、6,384 marker；4 个 worker 的 rank/活动覆盖完整。4 份 CUPTI 终态 drop 和 eBPF ring/map/send drop 全 0，collector 正常退出；`validation.json` 通过。
- 本轮历史回归仍覆盖原 1,973,432 条事件和 11 请求，逐请求/汇总/缺失项/re-entry 与原基线完全一致；另对 `split-workers` 的现代 v2 日志作严格数值回归。
- 新增组件检查覆盖非空完整分析链路、worker 分析与渲染分离、继承 executor 包装幂等、PP 异步 handle/postprocess/异常契约、task 清理、output suffix 等待归因、callback 异常、Python socket 关闭与空闲控制客户端 shutdown/restart。
- 原生 CUDA 控制测试增加了空闲客户端保持连接时的退出期限检查。凭据不写入仓库；eBPF 仍独立提权。
- `bench_results/refactor-eager2`：16/16 请求；5,900 Python、242,306 CUPTI 事件；101,781 kernel、784 marker；验收通过，CUPTI 终态 drop 全 0。
- `bench_results/refactor-graph2`：16/16 请求；7,565 Python、193,090 CUPTI 事件；131,109 kernel、1,104 marker、257 GraphLaunch；验收通过，CUPTI 终态 drop 全 0。
- 最终单元/生命周期/组件检查 **42 项通过**；本地 vLLM 语义对照 9 场景通过。`make -B -j4`、清理 `LD_LIBRARY_PATH` 后 eBPF 加载、带空闲客户端的原生 CUDA 控制检查和 wheel 构建通过；wheel 包含本轮新增模块。
- 分析和渲染分离后，历史日志生成的 **2,941,141 个 Perfetto event** 与拆分前逐字段一致（忽略 event 数组顺序，不忽略字段）。

未通过的原 session 不会改写成通过。本轮是短负载结构回归，不替代上轮 128 请求记录，也不证明多小时稳定性或完整 query 模式高负载零丢包。

## 用户命令入口复验

`de-latency serve / ebpf / analyze` 不自动运行或修改客户端。以下使用 Qwen1.5-4B-Chat、小请求负载和未修改的 `benchmarks/run_baseline_client.py` 验证接口；served-model-name 使用客户端要求的 `qwen3-32b` 仅作为 API 别名，实际模型不是 Qwen3-32B。

| Session | 检查与结果 |
|---|---|
| `cli-live` | Python+CUPTI；独立客户端 2/2 请求，450 Python、20,191 CUPTI 事件；Ctrl+C 后自动排空，单独 analyze 得到完整 TTFT/TPOT |
| `cli-three2` | 独立启动 eBPF；4/4 请求，1,624 Python、78,416 CUPTI、165 eBPF 事件；三路采集后单独分析成功；native 终态丢包全 0 |
| `cli-python2` | `--cupti off`；2/2 请求，仅有 823 Python 事件，未由工具注入 tracer |
| `cli-cupti2` | `--python off`；2/2 请求，仅有 36,629 CUPTI 事件，可生成 GPU trace；没有请求 hooks/完整请求分解 |

用户入口阶段单元/生命周期检查 **52 项通过**，含原服务参数透传、四种组件开关组合、独立 eBPF 权限边界、原客户端不受管理、活跃日志拒绝分析、失效 socket 过滤及 Ctrl+C 不掩盖 collector 退出失败。9 个 vLLM 语义场景通过，wheel 包含 CLI 模块和 console entry point。

`cli-cupti` 是本入口开发期间的失败记录：CUPTI-only 发现模型加载临时子进程残留 socket，连接被拒绝。现只选择仍存活且属于本服务进程树的 socket，`cli-cupti2` 复验通过；原失败记录未改写。

这些检查不等于已经运行用户的 Qwen3-32B、TP=2/PP=2、100 请求、1024/256 token 正式实验。用户负载参数不会被 smoke-test 默认值覆盖，正式实验需按实际可用 GPU/显存运行。

## 全量 gzip 与可选概览（2026-11-02）

- 默认分析/自动验收时间线改为 `trace.json.gz`；实际 gzip 压缩且保留全部事件/字段。显式 `.json`、旧 session 的未压缩输出继续兼容；数值 sidecar 名称不变。
- `analyze --overview` 同时输出全量与概览；`overview` 可从已有 trace 流式补概览，不读取 raw log、不改变分解结果。默认采集模式未改成 markers-only。
- **62 项单元/生命周期测试通过**，新增检查包括 gzip 文件头、全量事件无损、ns 大整数、Unicode、概览保留/过滤规则、两个输入格式、侧车命名、写出失败不截断旧文件、同源输出保护和压缩验收兼容。
- 对 `cli-live` 原始日志重新分析到 `/tmp/opencode/gzip-cli-check/trace.json.gz`：解压后的完整 trace、TTFT/TPOT、summary、alignment 与原分析逐字段一致。gzip replay regression 通过。
- 使用正式 `overview` 命令读取用户已有 `qps2/trace.json`（6,627,798,946 字节、22,324,574 事件）：保留 429,824 个概览事件，写到 `/tmp/opencode/qps2-export-check/trace.overview.json.gz`；峰值 RSS 23,552 KiB，耗时约 90 秒。没有重算该正式实验的 raw log，也没有改写它的全量 trace、数值结果或客户端。
- 正式命令生成的概览与此前 `qps2/trace.overview.json.gz` 逐事件流式 SHA-256 一致；最终补充截断输入保护和零时长事件检查后，62 项测试再次全部通过。

## 按职责统一布局（2026-11-02）

- Python 产品实现统一在 `de_latency/`，native 自研头文件归入 `native/include/`，第三方头/库归入 `third_party/`，绘图与请求示例归入 `tools/`，旧环境/清理记录归入 `docs/history/`。原路径包装不再保留。
- API 服务、collector、离线分析和低层控制改用包内模块启动；CLI、native controller、自动验收及测试均已更新引用。worker 启动钩子位于 `instrumentation/startup/`，由采集环境显式加入 PYTHONPATH；新增测试覆盖开关、参数透传、异地工作目录和 multiprocessing spawn。
- 本工作区完整 `make -j4` 成功；**67 项单元/生命周期测试通过**；9 个 vLLM 0.20 语义场景通过（异常输出场景中的错误日志为预期测试行为）。`cli-live` 原始日志 gzip 回放的数值分解与既有基线一致。
- 在 `/tmp/opencode/layout-clean-jlb54xj4` 创建仅含新布局源码、不含旧目录/旧 build 的独立快照：重新编译三个原生产物、运行 67 项测试、构建 wheel 均通过。wheel 含新 launcher、startup 和控制模块；保留 `de-latency` console entry。
- 第三方头文件、历史 vmlinux 头和版本库均与原 Git 内容逐字节核对；保留原软链接，并补齐加载器所需的 `libbpf.so.1` SONAME 软链接。monitor 的 RUNPATH/动态加载路径已检查，`--help` 可正常运行；本轮没有重新执行 GPU 模型负载或实时 eBPF attach，不将布局检查当作新的三路 GPU 验收。
- 日常四个 CLI 命令、benchmark 客户端、全量 gzip/概览和数值结果命名未改；没有移动或清理实验数据。工作期间发现外部移除的旧 Perfetto 数据/压缩脚本及 tmp，未恢复这些变化。

## 保留的失败记录与修复

- `parity-final`：127/128 成功，客户端记录一条 Bad Gateway；不计为通过。后续 runner 为 loopback 显式补齐 NO_PROXY，并保存详细请求结果。
- `parity-v2`：128 请求完成，但 collector 缺失最终统计；不计为通过。collector 的排空与外部强杀期限原来都是 30 秒，现已为 runner 和 controller 留出独立退出余量，并检查退出码及落盘数量。
- `parity-v3`：采集完整、终态丢包为 0，但 4 个请求的 TTFT 分量不闭合。原因是 OS 等待叠加在已含等待的执行窗口上，以及部分 worker 窗口越过首 token。新 v2 分解按真实区间独占归因并裁剪到首 token；该日志重新分析后 128 个请求全部闭合，`reanalyzed.validation.json` 通过，原失败产物保留。
- CUDA 控制测试暴露首个 marker 序号为 0，导致零尺寸 CUDA grid；序号改从 1 开始，失败 metadata 清理。markers-only 在 off 状态只改变配置，不提前启用活动；活动切换失败不再返回成功；断开的控制客户端不向推理进程发送 SIGPIPE。
- 自定义 ZeroMQ/json-c 原来只依赖 shell 库搜索环境，sudo 下缺库；现由 pkg-config 或编译器查找库路径，C 与 NVCC 分别使用合适的链接参数。全量重建和环境清理检查通过。
- `refactor-eager`、`refactor-graph`：本轮采集完成，但分析阶段拆分时的导出局部变量与输入对象重名，导致转换失败；原 session 不计通过。现已使用具名阶段输入避免该冲突，并增加非空完整分析测试；最终重跑使用 `refactor-eager2`、`refactor-graph2`。

## 口径

v1 历史日志保留原 compute breakdown version 4 的数值，以支持严格回归。v2 新采集使用 version 5，显式标注 `ttft_partition_policy`。OS 等待不再与包含它的执行时间重复相加，因此现代分量与旧错误口径不要求逐值相同；请求总窗口不改变。

不将历史 eBPF 回放、双路采集或仅编译成功视为实时三路验收。保留 session 的原始日志、benchmark、控制响应、统计和分析产物在对应 `bench_results/` 目录中，目录不纳入版本管理；已清理 session 仅保留上述小型验收快照。
