"""Receive canonical events, persist JSONL and publish eBPF target discovery."""
import json
import os
import signal
import time
from collections import Counter
from pathlib import Path

import zmq

from de_latency.protocol import validate_event

WORKER_EVENTS = frozenset({
    "worker_process_ready", "worker_preprocess_start", "worker_model_execute_span",
    "gpu_forward_start", "gpu_forward_end", "gpu_execute_model",
})


def env_int(name, default, minimum=1):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


class TargetRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.pids = set()
        self.tids = set()

    def persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        text = "".join(f"{pid}\n" for pid in sorted(self.pids))
        text += "".join(f"tid:{tid}\n" for tid in sorted(self.tids))
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def observe(self, event):
        if event["source"] != "monkey_patch":
            return
        payload = event["payload"]
        changed = False
        if event["event_type"] in WORKER_EVENTS:
            pid = event.get("pid") or payload.get("pid")
            if isinstance(pid, int) and pid > 0 and pid not in self.pids:
                self.pids.add(pid)
                changed = True
        if event["event_type"] == "thread_role" and payload.get("role") == "asyncio_eventloop":
            tid = event.get("tid") or payload.get("tid")
            if isinstance(tid, int) and tid > 0 and tid not in self.tids:
                self.tids.add(tid)
                changed = True
        if changed:
            self.persist()


def main():
    root = Path(__file__).resolve().parents[1]
    log_path = Path(os.getenv("DE_LATENCY_LOG_PATH", str(root / "perfetto/de_latency.log")))
    targets = TargetRegistry(os.getenv("TRACER_TARGET_FILE", os.getenv("TRACER_WORKER_PID_FILE", "/tmp/tracer_worker_pids")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    targets.persist()
    stop_at = None

    def stop(signum, frame):
        nonlocal stop_at
        if stop_at is None:
            stop_at = time.monotonic()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    counts = Counter()
    invalid = 0
    producer_drops = {}
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, env_int("TRACER_ZMQ_RCVHWM", 2000000))
    sock.setsockopt(zmq.RCVTIMEO, env_int("TRACER_ZMQ_RCVTIMEO_MS", 200))
    sock.setsockopt(zmq.LINGER, 0)
    flush_every = env_int("TRACER_LOG_FLUSH_EVERY", 1000)
    drain_limit = env_int("TRACER_DRAIN_TIMEOUT_SEC", 30)
    try:
        sock.bind(os.getenv("TRACER_ZMQ_ADDR", "ipc:///tmp/tracer.sock"))
        print(f"[collector] ready log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            while True:
                try:
                    frames = sock.recv_multipart()
                except zmq.Again:
                    if stop_at is not None:
                        break
                    continue
                try:
                    if len(frames) != 1:
                        raise ValueError("v2 requires a single JSON frame")
                    event = validate_event(json.loads(frames[0]))
                    targets.observe(event)
                    counts[event["source"]] += 1
                    key = f'{event["source"]}:{event.get("pid")}:{event.get("tid")}'
                    producer_drops[key] = max(producer_drops.get(key, 0), event.get("producer_dropped", 0))
                    stream.write(frames[0].decode("utf-8").rstrip("\n") + "\n")
                    if sum(counts.values()) % flush_every == 0:
                        stream.flush()
                except (ValueError, TypeError, KeyError) as exc:
                    invalid += 1
                    if invalid <= 10:
                        print(f"[collector] invalid event: {exc}", flush=True)
                if stop_at is not None and time.monotonic() - stop_at > drain_limit:
                    raise RuntimeError("collector drain deadline exceeded: producers still active")
    finally:
        sock.close()
        ctx.term()
        summary = {"schema_version": 2, "sources": dict(counts), "invalid_events": invalid,
                   "producer_dropped": producer_drops}
        log_path.with_suffix(log_path.suffix + ".stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[collector] stopped {summary['sources']} invalid={invalid}", flush=True)
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
