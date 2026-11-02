# 模块与扩展边界

```text
de_latency/
  cli.py / __main__.py        serve / ebpf / analyze / overview 用户入口；不管理 benchmark
  protocol.py                v2 校验、历史日志解码、CUPTI 时钟转换
  collector.py               单帧接收、JSONL 持久化、目标发现、统计
  runtime/
    control_cli.py / cupti_cli.py 包内低层控制命令，不依赖外置包装脚本
    transport.py             线程独立的 Python PUSH socket、fork 重置
    control.py               Python 运行时开关
    request_context.py       请求名称和上下文
    socket_client.py         Unix socket 控制客户端
    session.py               独立 session 配置、参数透传、进程身份与排空边界
  instrumentation/
    launch.py                API 服务启动；参数透传，不启动 benchmark
    startup/sitecustomize.py 仅在采集进程 PYTHONPATH 中启用的 spawn 自动插桩
    bootstrap.py             vLLM 0.20 版本门禁
    registry.py              幂等方法包装、导入钩子
    markers.py               可选 libtracer GPU marker 桥接
    output_loop.py           对照 0.20 的输出循环与轮次边界
    task_events.py           请求阶段、协程、output handler/socket 的共享事件构造
    scheduler_metadata.py    0.20 scheduler output 的请求集、token 状态和稳定 dispatch key
    execution_context.py     嵌套且线程隔离的 worker dispatch 上下文
    worker_hooks.py          worker 侧薄安装入口
    workers/                context、EngineCore、entry、executor、GPU runner、PP、scheduler
    request_hooks.py         API 侧薄安装入口
    requests/               task registry、callback 调度、输出轮次、output、stages、generate 生命周期
  analysis/
    clocks.py                时钟单位转换
    intervals.py             合并、裁剪、交集
    requests.py              历史请求 ID 规范化与别名映射
    wall_time.py             重叠 TP/PP 区间的墙钟时间分配
    ttft.py                  TTFT 独占墙钟区间分解；历史 v1 数值回放
    parallel_dispatch.py     TP 合并、PP stage gap 和逐 dispatch 执行分解
    correlation.py           进程内 CUPTI correlation key 与 launch epoch 筛选
    ingestion.py             日志分类、请求别名、时钟样本 → LogData
    worker_intervals.py      worker 状态机与 PP 关联 → WorkerAnalysis
    request_timing.py        请求生命周期和 task 区间 → RequestTiming
    dispatch_windows.py     worker/GPU marker 的 dispatch 索引 → DispatchWindows
    dispatch.py              RPC/ready 配对、phase 与区间去重 → DispatchAnalysis
    dispatch_queues.py       CPU/GPU queue gap → QueueAnalysis
    request_stages.py        generate/enqueue 阶段聚合 → RequestStages
    os_scheduling.py         mono/wall 校准、OS 等待归因 → OsAttribution
    request_index.py         请求集合、OS 区间投影 → RequestIndex
    tpot.py / breakdown.py   TPOT 独立计算、TTFT/TPOT 汇总 → ComputeBreakdown
    model.py                 对外 AnalysisResult
    pipeline.py              显式阶段调用及结果组装；保留旧入口
  export/
    workers.py / core.py      worker、PP flow、EngineCore、scheduler 轨道
    requests.py              请求生命周期、排队、执行、OS 轨道
    gpu.py / os_scheduling.py GPU/API/flow 与 event-loop OS 轨道
    request_layout.py        请求 lane、dispatch 显示区间及 alignment debug
    perfetto.py / styles.py  ns→trace、显示修正及配色
    reports.py / bundle.py   SVG/HTML、JSON sidecar 和文件输出
    trace_io.py / overview.py 原子 JSON/gzip 输出、大文件流式概览；不改数值归因
native/
  include/                   自研 native 共享协议/事件头
  controller/main.c          子进程组与采集生命周期
  cupti/
    tracer.cu                生命周期与控制命令策略
    activity.cpp             CUPTI buffer、活动解码、query 过滤及 driver drop 计数
    transport.cpp            有界发送队列、排空、独占 ZMQ socket 的 sender 线程
    control_server.cpp       Unix socket 服务；命令处理/tick 回调与策略分离
    markers.cu               GPU marker 与 metadata 生命周期
    support.cpp / serialize.cpp 环境/日志与 JSON-C 序列化
  ebpf/                      内核调度探针与用户态消费
benchmarks/                  独立 session、真实验收、语义/历史回归
tools/                       绘图、手动请求示例，不参与产品启动
third_party/                 libbpf 头/库和历史 Linux 生成头
tests/                       单元及生命周期测试
docs/history/                历史环境/清理记录，不作为当前路径说明
```

## 数据流

三路 producer → 单帧 JSON PUSH → collector → JSONL → `analyze_logs()` → `AnalysisResult` → `export_result()`。

`process_logs(input, output)` 内部依次分析和导出。导出端对 Perfetto slice 边界的显示修正作用于副本，不改变数值归因。不同分析实例有独立的时间原点。controller 通过 `python -m de_latency.collector` 启动接收端，CLI 通过包内 `instrumentation.launch` 启动 API 服务；不再经过旧目录的兼容脚本。

## 扩展方式

1. 新事件使用 `protocol.make_event` 或对应 native serializer；标明源时钟域，时间使用整数 ns。
2. worker 方法使用 `register_hook` + `apply_method_patch`，避免重复包装。
3. 请求业务循环替换需对照本地 0.20 源码，运行 `check_vllm_semantics.py` 和真实 benchmark。
4. 新归因规则优先写成纯区间函数，由 pipeline 调用；新格式只在 export 层处理文件。
5. 修改归因后运行历史数值回归。仅编译成功不代表采集链路通过。

## 当前边界

`pipeline.py` 只编排上述阶段。阶段间通过具名 dataclass 交换结果，不传 `locals()`、不动态执行源码，也不依赖原长函数的隐式局部变量。分析阶段不调用 Perfetto writer；worker 状态机产生整数 ns 的 `WorkerTimelineSpan`，由 export 转成轨道。渲染修改只作用于输出数据，不回写归因结果。

`LogData` 的归一化 payload 和区间结果属于一次分析调用；阶段可以排序事件和消费 phase hint 队列，因此需要重跑时重新 `load_log()`。导出可对同一 worker 结果重复调用。公开 `AnalysisResult` 仍包含兼容的 trace、数值 sidecar 和请求顺序。

worker/request 安装入口已薄化。两种 0.20 GPU runner 共用 execute/sample 包装，只保留上下文恢复差异；PP 同步/异步通信共享包装，并保持返回 tensor、handle、postprocess callback 及异常传播。executor 的继承包装不会构造自引用 `__wrapped__`。请求 output handler 等待按尚未处理的 suffix 请求集归因，轮次状态独立于 task registry 和 callback 调度器。

native 组件显式 shutdown，进程级状态不依赖共享库静态析构顺序。CUPTI socket 的创建、发送与关闭始终属于 sender 线程；空闲控制客户端有超时，不能无限阻塞 shutdown。

schema-v2 新采集的 TTFT 使用独占区间分解（compute breakdown version 5）：裁剪到首个 generate consume 结束，合并重复 rank，OS 等待从执行时间中分出。旧 v1 日志保留 version 4 的数值口径，便于严格历史回归；两种口径明确区分，不能将 bug 修正后的分量与旧口径逐项混比。

## 生命周期

日常使用由 `de-latency serve` 管理 collector 和用户指定参数的服务：默认安装 Python hooks、预加载但暂不开启 CUPTI，HTTP `/health` 就绪且 worker 已发现后开启 CUPTI 并提示 READY。用户在另一终端运行原 benchmark，工具不发推理请求、不改压测参数。`--python off` 跳过 hook 安装；`--cupti off` 禁止 controller 注入 tracer，而不仅是把采集暂时设为 off。

`de-latency ebpf --session ...` 独立启动仅跟踪该 session 的 native monitor，需要时仅为 monitor 请求 sudo。Python 开启时使用 collector 的 worker/TID 目标表；Python 关闭时改为本服务进程树的主线程扫描，不跨 session 猜测 worker。后者没有请求上下文，不提供完整逐请求归因。

用户结束 benchmark 后在 serve 终端 Ctrl+C：先停止/排空 CUPTI、关闭 Python 发送，再停止独立 eBPF，最后由 controller 关闭服务及排空 collector。`manifest.json` 保存原始参数、组件配置和退出错误；`de-latency analyze` 拒绝读取运行中或未排空的 session。新入口不依赖或修改用户的客户端脚本；`benchmarks/validate.py` 仍是独立的自动验收入口。

分析默认输出完整事件集的 `trace.json.gz`，直接逐事件写入真实 gzip；显式 `.json` 输出仍兼容。小型数值 sidecar 和图表命名不变。`analyze --overview` 额外输出压缩概览而非替换全量结果；独立 `overview` 命令用 ijson 流式读取已有 `.json`/`.json.gz`，仅过滤展示层的普通 GPU/API/flow，保留请求、worker、调度、PP、OS 和 markers。它不重算归因、不改原始日志，也不把原本的全量采集改成 markers-only。输出使用临时文件完成后原子替换，解析/写出失败不会留下截断的目标文件。

runner 先启动 collector，再启动服务及可选 monitor。服务健康后对已发现 worker 开启 CUPTI；benchmark 完成后逐 worker 停止并排空 CUPTI，再停止服务、eBPF、collector。只清理由本 session 创建的进程组。

eBPF 单独提权，API 服务和 collector 可保持普通用户。runner 和 controller 均为 collector 留出排空期限及统计写出余量，失败退出不能静默报告成功。eBPF ring/map/send 和 CUPTI ring/ZMQ/driver 丢包均有终态统计。Python 发送失败计数附在该线程后续成功事件中；若线程在最后一次发送失败后退出，这种附带计数不能证明尾部绝对无损。
