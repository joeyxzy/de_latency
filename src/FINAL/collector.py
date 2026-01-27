import zmq
import json
from pathlib import Path
from datetime import datetime

SOCK_ADDR = "ipc:///tmp/tracer.sock"

LOG_DIR = Path.cwd()
LOG_DIR.mkdir(parents=True, exist_ok=True)
CUPTI_LOG = Path("/home/joeyxzy/de_latency/de_latency/perfetto/de_latency.log")

# 1. 修改这里：增加 src 参数
def _log(src, meta, payload):
    # 将记录写入 JSON
    rec = {
        "src": src,  # <--- 修改这里：使用传入的 src，而不是写死 "cupti"
        "meta": meta,
        "payload": (
            {"_bytes_len": len(payload)}
            if isinstance(payload, (bytes, bytearray)) else payload
        )
    }
    with open(CUPTI_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _get_src(meta):
    src = meta.get("source")
    return src.lower() if isinstance(src, str) else "unknown"

def handle_frames(frames):
    # ... 前面代码不变 ...
    
    meta = None
    payload = None

    try:
        # 第一帧总是 JSON
        meta_json = frames[0].decode()
        meta = json.loads(meta_json)
    except Exception as e:
        print(f"[collector] invalid JSON: {e}", flush=True)
        return

    # 情况 A: 双帧 (CUPTI / Python) -> Payload 在第二帧
    if len(frames) == 2:
        try:
            decoded = frames[1].decode()
            payload = json.loads(decoded)
        except Exception:
            payload = frames[1] # 二进制保留

    # 情况 B: 单帧 (eBPF 新逻辑) -> Payload 就在 Meta 里
    elif len(frames) == 1:
        # 尝试从 meta 中提取 payload
        payload = meta.get("payload")
    
    else:
        print(f"[ERROR] Unexpected multipart size: {len(frames)}", flush=True)
        return
    
    src = _get_src(meta)
    
    # 统一记录逻辑
    if src == "ebpf" or src == "cupti" or src == "monkey_patch":
        # 如果 payload 已经在 meta 里了 (eBPF)，_log_cupti 会把它再包一层
        # 这里的 meta 其实就是包含 payload 的完整对象
        # 我们可以稍微调整 _log_cupti 的调用方式，或者保持现状
        
        # 现状：_log_cupti 会写成 {"src":..., "meta": meta, "payload": payload}
        # 对于 eBPF (单帧)，此时 meta 包含 payload，而上面的变量 payload 也是那个 payload
        # 结果会有冗余：{"meta": {"payload": {...}}, "payload": {...}}
        # 但这不影响数据完整性，只是稍微浪费空间。
        
        # 优化：如果是单帧且 meta 里已有 payload，调用时 payload 传 None 或者处理一下
        _log(src, meta, payload) 
        return
    
    # ...

def collector():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(SOCK_ADDR)

    print("[collector] started, listening on", SOCK_ADDR)

    while True:
        frames = sock.recv_multipart()
        handle_frames(frames)

if __name__ == "__main__":
    CUPTI_LOG.write_text("", encoding="utf-8")
    collector()