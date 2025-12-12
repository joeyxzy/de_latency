import zmq
import json
from pathlib import Path
from datetime import datetime

SOCK_ADDR = "ipc:///tmp/tracer.sock"

LOG_DIR = Path.cwd()
LOG_DIR.mkdir(parents=True, exist_ok=True)
CUPTI_LOG = LOG_DIR / "de_latency.log"

def _log_cupti(meta, payload):
    # 将 CUPTI 记录为一行 JSON；二进制 payload 只记录长度，避免文件巨大
    rec = {
        "src": "cupti",
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
    if not frames:
        print("[collector] empty frames", flush=True)
        return
    try:
        meta_json = frames[0].decode()
        meta = json.loads(meta_json)
    except Exception as e:
        print(f"[collector] invalid JSON: {e}", flush=True)
        return
    payload = None
    if len(frames) == 2:
        payload = frames[1]
        # 若第二帧本身就是 JSON 文本，可按需解析：
        try:
            # 尝试将二进制 payload 解析为 JSON（不是 JSON 时保留原始 bytes）
            decoded = frames[1].decode()
            payload = json.loads(decoded)
        except Exception:
            pass
    elif len(frames) > 2:
        print(f"[ERROR] Unexpected multipart size: {len(frames)}", flush=True)
        return
    src = _get_src(meta)
    if src == "ebpf":
        return  # 暂时忽略 eBPF 消息，太多了
    if src == "cupti" or src == "monkey_patch":
        _log_cupti(meta, payload)
        return
    if payload is not None:
        print(src, meta, payload, flush=True)
    else:
        print(src, meta, flush=True)

def collector():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(SOCK_ADDR)

    print("[collector] started, listening on", SOCK_ADDR)

    while True:
        frames = sock.recv_multipart()   # 支持 1 到 N 帧
        handle_frames(frames)
        # if len(frames) == 1:
        #     # 单帧 JSON → CUPTI / eBPF
        #     meta_json = frames[0].decode()
        #     meta = json.loads(meta_json)
        #     payload = None
        #     print("[CUPTI/eBPF]", meta)

        # elif len(frames) == 2:
        #     # Python multipart → meta + payload
        #     meta_json = frames[0].decode()
        #     payload = frames[1]      # 二进制
        #     meta = json.loads(meta_json)
        #     print("[PYTHON]", meta, "payload_size=", len(payload))

        # else:
        #     # 不期望出现多帧
        #     print("[ERROR] Unexpected multipart size:", len(frames))
        #     continue

        # 你可以在这里进一步处理 meta + payload
        # process(meta, payload)

if __name__ == "__main__":
    CUPTI_LOG.write_text("", encoding="utf-8")
    collector()
