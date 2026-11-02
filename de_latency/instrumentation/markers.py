"""Optional GPU batch markers exported by the preloaded CUPTI library."""
import ctypes
import logging
import os

logger = logging.getLogger(__name__)
try:
    _push = ctypes.CDLL(None).de_marker_push
    _push.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _push.restype = ctypes.c_int
except AttributeError:
    _push = None


def mark(stream, role, dispatch_key):
    if _push is None or dispatch_key is None:
        return
    try:
        result = _push(int(stream.cuda_stream), f"{role}|{dispatch_key}".encode())
        if result and os.getenv("DE_LATENCY_MARKER_DEBUG") == "1":
            logger.warning("marker %s failed: %s", role, result)
    except (AttributeError, TypeError, ValueError) as exc:
        if os.getenv("DE_LATENCY_MARKER_DEBUG") == "1":
            logger.warning("marker %s failed: %s", role, exc)
