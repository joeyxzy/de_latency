import json
import os
from pathlib import Path

import zmq

SOCK_ADDR = "ipc:///tmp/tracer.sock"


def _env_int(name, default, min_value):
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(min_value, value)


RCVHWM = _env_int("TRACER_ZMQ_RCVHWM", 200000, 1000)
LOG_FLUSH_EVERY = _env_int("TRACER_LOG_FLUSH_EVERY", 1000, 1)

LOG_DIR = Path.cwd()
LOG_DIR.mkdir(parents=True, exist_ok=True)
CUPTI_LOG = Path("/home/joeyxzy/de_latency/de_latency/perfetto/de_latency.log")
WORKER_PID_FILE = Path("/tmp/tracer_worker_pids")
WORKER_PID_EVENTS = {
    "worker_process_ready",
    "worker_preprocess_start",
    "worker_model_execute_span",
    "gpu_forward_start",
    "gpu_forward_end",
    "gpu_execute_model",
}
_known_worker_pids = set()


def _log(log_file, src, meta, payload):
    rec = {
        "src": src,
        "meta": meta,
        "payload": (
            {"_bytes_len": len(payload)}
            if isinstance(payload, (bytes, bytearray)) else payload
        ),
    }
    log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _get_src(meta):
    src = meta.get("source")
    return src.lower() if isinstance(src, str) else "unknown"


def _extract_dict_payload(meta, payload):
    if isinstance(payload, dict):
        return payload
    p = meta.get("payload")
    if isinstance(p, dict):
        return p
    return None


def _coerce_pid(value):
    try:
        pid = int(value)
    except Exception:
        return None
    return pid if pid > 0 else None


def _extract_worker_pid(src, meta, payload):
    if src != "monkey_patch":
        return None

    event_type = meta.get("event_type")
    if event_type not in WORKER_PID_EVENTS:
        return None

    p = _extract_dict_payload(meta, payload)
    if not p:
        return None
    return _coerce_pid(p.get("pid"))


def _persist_worker_pid(pid):
    if pid is None or pid in _known_worker_pids:
        return

    _known_worker_pids.add(pid)
    WORKER_PID_FILE.write_text(
        "\n".join(str(x) for x in sorted(_known_worker_pids)) + "\n",
        encoding="utf-8",
    )
    print(f"[collector] discovered worker pid={pid}", flush=True)


def handle_frames(frames, log_file):
    meta = None
    payload = None

    try:
        meta_json = frames[0].decode()
        meta = json.loads(meta_json)
    except Exception as e:
        print(f"[collector] invalid JSON: {e}", flush=True)
        return False

    if len(frames) == 2:
        try:
            decoded = frames[1].decode()
            payload = json.loads(decoded)
        except Exception:
            payload = frames[1]
    elif len(frames) == 1:
        payload = meta.get("payload")
    else:
        print(f"[ERROR] Unexpected multipart size: {len(frames)}", flush=True)
        return False

    src = _get_src(meta)
    _persist_worker_pid(_extract_worker_pid(src, meta, payload))

    if src == "ebpf" or src == "cupti" or src == "monkey_patch":
        _log(log_file, src, meta, payload)
        return True
    return False


def collector():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, RCVHWM)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(SOCK_ADDR)

    print("[collector] started, listening on", SOCK_ADDR, f"(rcvhwm={RCVHWM}, flush_every={LOG_FLUSH_EVERY})")

    written = 0
    with open(CUPTI_LOG, "a", encoding="utf-8") as log_file:
        while True:
            frames = sock.recv_multipart()
            if handle_frames(frames, log_file):
                written += 1
                if written % LOG_FLUSH_EVERY == 0:
                    log_file.flush()


if __name__ == "__main__":
    CUPTI_LOG.write_text("", encoding="utf-8")
    WORKER_PID_FILE.write_text("", encoding="utf-8")
    collector()
