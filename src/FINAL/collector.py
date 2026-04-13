import json
import os
import signal
from pathlib import Path

import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = REPO_ROOT / "perfetto" / "de_latency.log"
SOCK_ADDR = os.getenv("TRACER_ZMQ_ADDR", "ipc:///tmp/tracer.sock")
DEFAULT_TARGET_FILE = "/tmp/tracer_worker_pids"


def _env_int(name, default, min_value):
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(min_value, value)


RCVHWM = _env_int("TRACER_ZMQ_RCVHWM", 2000000, 1000)
RCVTIMEO_MS = _env_int("TRACER_ZMQ_RCVTIMEO_MS", 200, 1)
LOG_FLUSH_EVERY = _env_int("TRACER_LOG_FLUSH_EVERY", 1000, 1)

LOG_DIR = Path.cwd()
LOG_DIR.mkdir(parents=True, exist_ok=True)
CUPTI_LOG = Path(os.getenv("DE_LATENCY_LOG_PATH", str(DEFAULT_LOG_PATH)))
TARGET_FILE = Path(
    os.getenv(
        "TRACER_TARGET_FILE",
        os.getenv("TRACER_WORKER_PID_FILE", DEFAULT_TARGET_FILE),
    )
)
CUPTI_LOG.parent.mkdir(parents=True, exist_ok=True)
TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
WORKER_PID_EVENTS = {
    "worker_process_ready",
    "worker_preprocess_start",
    "worker_model_execute_span",
    "gpu_forward_start",
    "gpu_forward_end",
    "gpu_execute_model",
}
THREAD_TARGET_EVENTS = {
    "thread_role",
}
THREAD_TARGET_ROLES = {
    "asyncio_eventloop",
}
_known_worker_pids = set()
_known_exact_tids = set()
_stop_requested = False


def _handle_stop_signal(signum, _frame):
    global _stop_requested
    _stop_requested = True
    print(f"[collector] stop requested by signal {signum}", flush=True)


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


def _log_raw_json_line(log_file, raw_frame):
    try:
        line = raw_frame.decode("utf-8")
    except UnicodeDecodeError:
        line = raw_frame.decode("utf-8", errors="replace")
    log_file.write(line)
    if not line.endswith("\n"):
        log_file.write("\n")


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


def _persist_target_file():
    lines = [str(x) for x in sorted(_known_worker_pids)]
    lines.extend(f"tid:{x}" for x in sorted(_known_exact_tids))
    content = ("\n".join(lines) + "\n") if lines else ""
    TARGET_FILE.write_text(content, encoding="utf-8")


def _persist_worker_pid(pid):
    if pid is None or pid in _known_worker_pids:
        return

    _known_worker_pids.add(pid)
    _persist_target_file()
    print(f"[collector] discovered worker pid={pid}", flush=True)


def _extract_exact_tid(src, meta, payload):
    if src != "monkey_patch":
        return None

    event_type = meta.get("event_type")
    if event_type not in THREAD_TARGET_EVENTS:
        return None

    p = _extract_dict_payload(meta, payload)
    if not p:
        return None
    if p.get("role") not in THREAD_TARGET_ROLES:
        return None
    return _coerce_pid(p.get("tid"))


def _persist_exact_tid(tid):
    if tid is None or tid in _known_exact_tids:
        return

    _known_exact_tids.add(tid)
    _persist_target_file()
    print(f"[collector] discovered exact tid={tid}", flush=True)


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
    _persist_exact_tid(_extract_exact_tid(src, meta, payload))

    if src == "ebpf" or src == "cupti" or src == "monkey_patch":
        # CUPTI 是单帧 JSON，直接落盘原始行，避免 json.loads+json.dumps 的双重开销。
        if src == "cupti" and len(frames) == 1:
            _log_raw_json_line(log_file, frames[0])
        else:
            _log(log_file, src, meta, payload)
        return True
    return False


def collector():
    global _stop_requested
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, RCVHWM)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, RCVTIMEO_MS)
    sock.bind(SOCK_ADDR)

    print("[collector] started, listening on", SOCK_ADDR, f"(rcvhwm={RCVHWM}, rcvtimeo_ms={RCVTIMEO_MS}, flush_every={LOG_FLUSH_EVERY})")

    written = 0
    with open(CUPTI_LOG, "a", encoding="utf-8") as log_file:
        while True:
            try:
                frames = sock.recv_multipart()
            except zmq.Again:
                # 收到停止信号后，等到 socket 至少空闲一个超时周期再退出，尽量排空队列。
                if _stop_requested:
                    break
                continue
            except KeyboardInterrupt:
                if _stop_requested:
                    break
                _stop_requested = True
                continue

            if handle_frames(frames, log_file):
                written += 1
                if written % LOG_FLUSH_EVERY == 0:
                    log_file.flush()

            if _stop_requested:
                # 进入 draining 状态后继续读，直到下一个 recv 超时才退出。
                continue

        log_file.flush()
    sock.close(0)
    ctx.term()
    print(f"[collector] stopped, total_written={written}", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    CUPTI_LOG.write_text("", encoding="utf-8")
    TARGET_FILE.write_text("", encoding="utf-8")
    collector()
