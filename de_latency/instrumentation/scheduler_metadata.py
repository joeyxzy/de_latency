"""vLLM 0.20 scheduler metadata shared by scheduler, executor and worker hooks."""
import hashlib
import json


def safe_int(value, default=None):
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_req_ids(req_ids):
    if req_ids is None:
        return []
    values = req_ids if isinstance(req_ids, (list, tuple, set)) else [req_ids]
    normalized = []
    for item in values:
        if item is not None:
            rid = getattr(item, "request_id", None) or getattr(item, "req_id", None) or item
            if text := str(rid).strip():
                normalized.append(text)
    return normalized


def request_ids(output):
    if output is None:
        return []
    ids = [rid for request in output.scheduled_new_reqs for rid in normalize_req_ids(request)]
    return ids + normalize_req_ids(output.scheduled_cached_reqs.req_ids)


def step_maps(output):
    computed, scheduled = {}, {}
    if output is None:
        return computed, scheduled
    for request in output.scheduled_new_reqs:
        ids = normalize_req_ids(request)
        if not ids:
            continue
        rid = ids[0]
        computed[rid] = safe_int(request.num_computed_tokens)
        tokens = request.prompt_token_ids
        if tokens is not None:
            scheduled[rid] = len(tokens)
    cached = output.scheduled_cached_reqs
    computed.update((rid, safe_int(count)) for rid, count in
                    zip(normalize_req_ids(cached.req_ids), cached.num_computed_tokens))
    scheduled.update((normalize_req_ids(rid)[0], safe_int(count, 0))
                     for rid, count in output.num_scheduled_tokens.items() if normalize_req_ids(rid))
    return computed, scheduled


def dispatch_key(output, phase_by_req=None):
    computed, scheduled = step_maps(output)
    parts = {"req_ids": request_ids(output), "num_computed_by_req": computed,
             "scheduled_tokens_by_req": scheduled,
             "total_scheduled_tokens": safe_int(getattr(output, "total_num_scheduled_tokens", None))}
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sched:" + hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
