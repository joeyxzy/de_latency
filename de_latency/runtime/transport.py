"""Shared nonblocking Python sender, with one socket per process/thread."""
import atexit
import json
import os
import threading
import time

from de_latency.protocol import make_event

_local = threading.local()


def _reset_child():
    global _local
    # Never use or close an inherited libzmq socket from a fork child.
    _local = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_child)


def get_socket():
    import zmq
    if getattr(_local, "pid", None) != os.getpid():
        _local.pid = os.getpid()
        _local.context = zmq.Context()
        _local.socket = _local.context.socket(zmq.PUSH)
        _local.socket.setsockopt(zmq.LINGER, 1000)
        _local.socket.setsockopt(zmq.SNDHWM, int(os.getenv("TRACER_ZMQ_SNDHWM", "100000")))
        _local.socket.connect(os.getenv("TRACER_ZMQ_ADDR", "ipc:///tmp/tracer.sock"))
        _local.dropped = 0
    return _local.socket


def send_event(source, event_type, payload=None, extra=None):
    from de_latency.runtime.control import is_enabled
    if not is_enabled():
        return
    try:
        import zmq
        payload = dict(payload or {})
        domain = "realtime"
        if event_type == "worker_model_execute_span":
            domain = "monotonic"
            payload["emit_wall_ns"] = time.time_ns()
        event = make_event(source, event_type, payload, clock_domain=domain, extra=extra)
        socket = get_socket()
        event["producer_dropped"] = _local.dropped
        socket.send(json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode(), zmq.DONTWAIT)
    except Exception:
        # Instrumentation errors must not break inference. Counts are attached
        # to the next successful event, rather than silently disappearing.
        _local.dropped = getattr(_local, "dropped", 0) + 1


class TraceSender:
    @staticmethod
    def emit(event_type, payload, source="monkey_patch"):
        send_event(source, event_type, payload)


def close():
    if getattr(_local, "pid", None) == os.getpid():
        _local.socket.close()
        _local.context.term()
        _local.pid = None


atexit.register(close)
