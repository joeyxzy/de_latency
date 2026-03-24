import threading
from contextvars import ContextVar
from typing import Dict, Optional, Tuple


# Per-coroutine request identity. ContextVar ensures each asyncio context
# sees its own value without cross-request contamination.
_ctx_request_id: ContextVar[Optional[str]] = ContextVar("ctx_request_id", default=None)
_ctx_request_name: ContextVar[Optional[str]] = ContextVar("ctx_request_name", default=None)

# Global cache: request_id -> request_name, shared across threads/process callbacks.
_lock = threading.Lock()
_request_name_by_id: Dict[str, str] = {}


def _normalize(value) -> Optional[str]:
    """Normalize input to a non-empty string, filtering placeholder values.

    Returns None for blank/unknown-like values so callers can use clear fallback rules.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"unknown", "none", "null"}:
        return None
    return text or None


def bind_request_context(request_id, request_name=None) -> Tuple[object, object]:
    """Bind request identity into current ContextVar scope.

    Also updates the global id->name cache when both fields are valid.
    Returns tokens for later reset in finally blocks.
    """
    rid = _normalize(request_id)
    rname = _normalize(request_name) or rid
    t1 = _ctx_request_id.set(rid)
    t2 = _ctx_request_name.set(rname)
    if rid and rname:
        with _lock:
            _request_name_by_id[rid] = rname
    return t1, t2


def reset_request_context(tokens: Tuple[object, object]) -> None:
    """Restore previous ContextVar values using tokens from bind_request_context."""
    t1, t2 = tokens
    _ctx_request_id.reset(t1)
    _ctx_request_name.reset(t2)


def get_current_request_id() -> Optional[str]:
    """Get request_id in the current coroutine/thread context."""
    return _ctx_request_id.get()


def get_current_request_name() -> Optional[str]:
    """Get request_name in the current coroutine/thread context."""
    return _ctx_request_name.get()


def get_request_ctx_from_context(ctx=None) -> Tuple[Optional[str], Optional[str]]:
    """Read request context from a specific context object or current context.

    When ctx is provided (e.g. callback's captured context), this extracts
    _ctx_request_id/_ctx_request_name from that context snapshot.
    """
    if ctx is None:
        return get_current_request_id(), get_current_request_name()
    try:
        rid = ctx.get(_ctx_request_id)
    except Exception:
        rid = None
    try:
        rname = ctx.get(_ctx_request_name)
    except Exception:
        rname = None
    return rid, rname


def remember_request_name(request_id, request_name=None) -> Optional[str]:
    """Persist request_id -> request_name mapping with fallback chain.

    Fallback priority:
    1) explicit request_name
    2) current context request_name
    3) request_id itself
    """
    rid = _normalize(request_id)
    rname = _normalize(request_name) or get_current_request_name() or rid
    if rid and rname:
        with _lock:
            _request_name_by_id[rid] = rname
    return rname


def resolve_request_name(request_id, fallback=None) -> Optional[str]:
    """Resolve the best request_name for a request_id.

    Lookup priority:
    1) global cache (_request_name_by_id)
    2) caller-provided fallback
    3) current context request_name
    4) request_id itself
    """
    rid = _normalize(request_id)
    if rid:
        with _lock:
            name = _request_name_by_id.get(rid)
        if name:
            return name
    return _normalize(fallback) or get_current_request_name() or rid
