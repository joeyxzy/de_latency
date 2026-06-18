#!/usr/bin/env python3
"""
compress_trace.py - 后处理压缩 Perfetto trace JSON

压缩策略:
  1. args key 名缩短 (KEY_MAP 映射)
  2. compact 序列化 (无空格)
  3. 浮点 ts/dur 去 .0 尾缀 (保留亚微秒精度)

用法:
  python compress_trace.py <input.json> [-o output.json] [--stats] [--reverse]
"""

import json
import sys
import argparse
import os


# ── args key 名压缩映射 ───────────────────────────────────────────────
KEY_MAP = {
    # ── Request Lifecycle ──
    "request_id":                       "rid",
    "request_name":                     "rname",
    "coro_sched_queue_ms":              "coro_sq",
    "generate_task_sched_queue_ms":     "gen_sq",
    "generate_exec_ms":                 "gen_ex",
    "output_socket_sched_queue_ms":     "osk_sq",
    "output_socket_exec_ms":            "osk_ex",
    "output_handler_sched_queue_ms":    "oh_sq",
    "output_handler_exec_ms":           "oh_ex",
    "vllm_queue_ms":                    "vq",
    "cpu_sched_queue_ms":               "csq",
    "pp_gap_ms":                        "ppg",
    "vllm_queue_from_enqueue_ms":       "vq_enq",
    "cpu_sched_queue_from_enqueue_ms":  "csq_enq",
    "vllm_queue_from_step_ready_ms":    "vq_sr",
    "cpu_sched_queue_from_step_ready_ms":"csq_sr",
    "gpu_dispatch_queue_ms":            "gdq",
    "engine_input_queue_wait_ms":       "eiq_w",
    "os_sched_delay_ms":                "os_sd",
    "prefill_exec_ms":                  "pf_ex",
    "decode_exec_ms":                   "dc_ex",
    "unknown_exec_ms":                  "uk_ex",

    # ── Dispatch ──
    "phase":                            "phs",
    "phase_reason":                     "ph_r",
    "decode_step":                      "dc_st",
    "dispatch_key":                     "dk",
    "dispatch_keys":                    "dks",
    "dispatch_key_count":               "dkc",
    "step_count":                       "sc",
    "batch_req_ids":                    "b_rids",
    "batch_req_count":                  "brc",
    "duration_ms":                      "dur_ms",
    "start_source":                     "ss",
    "end_source":                       "es",
    "gpu_marker_start_ns":              "gms",
    "gpu_marker_end_ns":                "gme",
    "gpu_queue_ns":                     "gqn",
    "rpc_start_ns":                     "rpc_s",
    "scheduler_ready_lag_ms":           "srl",
    "rpc_to_worker_queue_ms":           "rwq",
    "worker_span_count":                "wsc",
    "worker_span_count_total":          "wsct",
    "worker_boundary_policy":           "wbp",

    # ── GPU / Worker Phase ──
    "req_ids":                          "rids",
    "batch_size":                       "bs",
    "input_type":                       "inp",
    "class_path":                       "cpath",
    "dp_rank":                          "dpr",
    "pp_rank":                          "ppr",
    "tp_rank":                          "tpr",

    # ── PP Comm ──
    "attribution_source":               "asrc",
    "attribution_overlap_ns":           "aolp",
    "peer_rank":                        "prnk",
    "src":                              "src",
    "dst":                              "dst",
    "comm_id":                          "cid",
    "async_phase":                      "aphs",
    "async_elapsed_ns":                 "aens",
    "handle_index":                     "hidex",
    "handle_count":                     "hcnt",
    "tensor_count":                     "tcnt",
    "tensor_bytes":                     "tbyt",

    # ── Scheduler ──
    "num_requests":                     "nr",
    "num_running":                      "nrn",
    "num_waiting":                      "nw",

    # ── Request Queue ──
    "reason":                           "rsn",
    "scheduled_from":                   "sch_f",
    "gap_kind":                         "gk",
    "preempt_ts_ns":                    "pe_ts",
    "prev_dispatch_key":                "pdk",
    "next_dispatch_key":                "ndk",

    # ── Output / Generate ──
    "task_name":                        "tn",
    "shared_req_count":                 "shrc",
    "round_seq":                        "rseq",

    # ── OS Sched ──
    "worker_tids":                      "wtids",
    "dp_ranks":                         "dpks",
    "reasons":                          "rsns",

    # ── Stage ──
    "stage":                            "stg",

    # ── CUDA Runtime ──
    "correlationId":                    "corr_id",
    "correlationKey":                   "corr_key",
    "kernels":                          "knls",
    "is_graph":                         "is_gr",

    # ── PP Gap ──
    "section":                          "sec",
    "edge":                             "edg",
    "src_pp":                           "spp",
    "dst_pp":                           "dpp",
}

# 反向映射 (用于 --reverse)
REVERSE_MAP = {v: k for k, v in KEY_MAP.items()}

# 这些 key 出现在 metadata 事件的 args 中, 不能缩短 (Perfetto 依赖它们渲染)
PROTECTED_ARGS_KEYS = {"name"}


def compress_number(v):
    """去掉浮点数无意义的 .0 尾缀, 保留亚微秒精度"""
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def compress_args(args, mapping):
    """递归压缩 args dict 中的 key 名和数值"""
    if not isinstance(args, dict):
        return args
    result = {}
    for k, v in args.items():
        if k in PROTECTED_ARGS_KEYS:
            result[k] = v
        else:
            new_k = mapping.get(k, k)
            if isinstance(v, dict):
                result[new_k] = compress_args(v, mapping)
            elif isinstance(v, list):
                result[new_k] = [compress_args(item, mapping) if isinstance(item, dict) else compress_number(item) for item in v]
            else:
                result[new_k] = compress_number(v)
    return result


def compress_trace_events(trace_events, mapping):
    """压缩 traceEvents 中的所有事件"""
    compressed = []
    for event in trace_events:
        ev = {}

        # 处理 top-level fields
        for key, val in event.items():
            if key == "args":
                # ph=="M" 的 metadata 事件不压缩 args key (Perfetto 依赖原名)
                if event.get("ph") == "M":
                    ev["args"] = val
                else:
                    ev["args"] = compress_args(val, mapping) if val else {}
            elif key in ("ts", "dur"):
                ev[key] = compress_number(val)
            elif key == "dur" and val is None:
                continue  # 不输出 None 的 dur
            else:
                ev[key] = val

        # 如果 dur 是 None 或 0，去掉 dur field (非必须)
        if ev.get("dur") == 0:
            del ev["dur"]

        compressed.append(ev)
    return compressed


def main():
    parser = argparse.ArgumentParser(
        description="压缩 Perfetto trace JSON: key名缩短 + compact序列化 + 去.0尾缀"
    )
    parser.add_argument("input", help="输入 Perfetto trace JSON 文件")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件路径 (默认: input_compressed.json)")
    parser.add_argument("--stats", action="store_true",
                        help="打印压缩统计")
    parser.add_argument("--reverse", action="store_true",
                        help="反向: 将缩短的 key 还原为原名 (用于 debug)")
    parser.add_argument("--no-shorten-keys", action="store_true",
                        help="不缩短 key 名，只做 compact + 去.0 尾缀")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        if args.reverse:
            args.output = f"{base}_restored{ext}"
        else:
            args.output = f"{base}_compressed{ext}"

    input_size = os.path.getsize(args.input)

    print(f"Reading: {args.input} ({_size_str(input_size)})")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])

    if args.reverse:
        mapping = REVERSE_MAP
        print(f"🔄  Reverse mode: restoring {len(REVERSE_MAP)} key names")
    elif args.no_shorten_keys:
        mapping = {}
        print("⚠️   Key shortening disabled")
    else:
        mapping = KEY_MAP
        print(f"🔑  Shortening {len(KEY_MAP)} key names")

    compressed_events = compress_trace_events(trace_events, mapping)
    print(f"📦  Processed {len(compressed_events)} trace events")

    output_data = {"traceEvents": compressed_events}

    if not args.reverse:
        print(f"📝  Writing compact JSON to: {args.output}")
    else:
        print(f"📝  Writing restored JSON to: {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, separators=(",", ":"), ensure_ascii=False)

    output_size = os.path.getsize(args.output)
    ratio = (1.0 - output_size / input_size) * 100 if input_size > 0 else 0

    if args.stats:
        print()
        print("── Compression Stats ──")
        print(f"  Input:  {_size_str(input_size)}")
        print(f"  Output: {_size_str(output_size)}")
        print(f"  Ratio:  {ratio:.1f}%  reduced")
        print(f"  Saved:  {_size_str(input_size - output_size)}")
    else:
        print(f"✅  {_size_str(input_size)} → {_size_str(output_size)}  ({ratio:.1f}% reduced)")


def _size_str(size):
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    elif size >= 1_000_000:
        return f"{size / 1_000_000:.2f} MB"
    elif size >= 1_000:
        return f"{size / 1_000:.2f} KB"
    else:
        return f"{size} B"


if __name__ == "__main__":
    main()
