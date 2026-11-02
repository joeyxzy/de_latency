# 事件协议 v2

每条 ZeroMQ 消息只有一个 UTF-8 JSON frame，collector 原样写一行 JSONL。新 producer 只写 v2；历史 multipart/envelope 日志的兼容处理限定在读端，不接受旧 producer 接入新 collector。

```json
{
  "schema_version": 2,
  "session_id": "example",
  "source": "cupti",
  "event_type": "kernel",
  "clock_domain": "cupti",
  "clock_offset_ns": 1000000,
  "timestamp_ns": 2000,
  "pid": 123,
  "tid": 0,
  "producer_dropped": 0,
  "payload": {"pid": 123, "start_ns": 1000, "end_ns": 2000, "name": "kernel"}
}
```

- `source`：`monkey_patch`、`cupti`、`ebpf`。
- `clock_domain`：`realtime`、`monotonic`、`cupti`。
- `session_id`：来自 `DE_LATENCY_SESSION_ID`；单次分析拒绝混合 session。
- 所有 `*_ns` 为整数纳秒。`duration_ns` 始终是时长，不施加时钟偏移。
- CUPTI `clock_offset_ns`：启用时测量的 realtime − CUPTI offset。读端将非零 CUPTI start/end 转换为 realtime；0 保留为缺失值。
- Python `worker_model_execute_span` 使用 monotonic start/end，并附带 `emit_wall_ns` 校准样本。
- eBPF 使用 `bpf_ktime_get_ns()`，与 Python monotonic span 做区间交集。调度事件只保留大于 5 µs 的等待。
- CUPTI API 使用系统 native TID；kernel 无 host TID 时为 0，通过关联 ID、stream、marker 和 dispatch 建立关系。
- `producer_dropped` 是 producer 累计观测到的失败数。collector 按 source/PID/TID 保留最大值；不能跨线程直接相加，因为 CUPTI 计数是进程级的。
- `filtered_queries` 是 CUPTI 控制响应中的主动过滤量，与失败丢包分开。
- CUPTI `zmq_dropped` 包括所有最终发送错误及发送前的序列化/socket 准备失败；正常发送与退出排空共用计数路径。

历史日志读端支持旧的 `meta` envelope、大小写 source 和 EngineCore 秒浮点时间。新采集不允许浮点纳秒。`events.log.stats.json` 记录被拒绝的消息数量及各源事件数；最终丢包还应检查 CUPTI off 响应与 eBPF stats 文件。

事件 schema 版本与分析产物版本独立。v1 历史日志的 compute breakdown version 4 保留历史数值；v2 采集输出 version 5，并标注 `ttft_partition_policy`，修正 OS 等待重复计时及首 token 之后的执行越界。TTFT 分量在 v5 中逐纳秒闭合到请求窗口。
