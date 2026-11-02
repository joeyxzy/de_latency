"""Request identity normalization at the historical-log boundary."""
import json
from decimal import Decimal

from .clocks import to_int


def normalize_request_id(raw):
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, (str, int, float, Decimal)):
        return str(raw).strip() or None
    if isinstance(raw, dict):
        for key in ("request_id", "req_id", "id", "rid", "value"):
            candidate = normalize_request_id(raw.get(key))
            if candidate:
                return candidate
        try:
            return json.dumps(raw, sort_keys=True, ensure_ascii=True, separators=(",", ":")).strip() or None
        except (TypeError, ValueError):
            return str(raw).strip() or None
    if isinstance(raw, (list, tuple, set)):
        if len(raw) == 1:
            return normalize_request_id(next(iter(raw)))
        items = normalize_request_ids(raw)
        return json.dumps(items, ensure_ascii=True, separators=(",", ":")) if items else None
    return str(raw).strip() or None


def normalize_request_ids(raw):
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return [rid for item in values if (rid := normalize_request_id(item))]


def normalize_payload_request_fields(payload):
    if not isinstance(payload, dict):
        return payload
    name = normalize_request_id(payload.get("request_name"))
    if "request_name" in payload:
        payload["request_name"] = name
    for key in ("request_id", "req_id"):
        if key in payload:
            raw = payload[key]
            # Historical producers sometimes put a prompt object in the ID.
            payload[key] = name if name and isinstance(raw, (dict, list, tuple, set)) else normalize_request_id(raw)
    if "req_ids" in payload:
        payload["req_ids"] = normalize_request_ids(payload["req_ids"])
    _map_event_ids(payload, normalize_request_id)
    return payload


def _map_event_ids(payload, resolve):
    updates = payload.get("req_events")
    if isinstance(updates, list):
        for update in updates:
            if isinstance(update, dict) and "req_id" in update:
                update["req_id"] = resolve(update["req_id"])


def canonicalize_payload_request_fields(payload, resolve_request_id):
    if not isinstance(payload, dict):
        return payload
    for key in ("request_id", "req_id", "request_name", "external_request_id"):
        if key in payload:
            payload[key] = resolve_request_id(payload[key])
    if "req_ids" in payload:
        payload["req_ids"] = [rid for item in payload["req_ids"] if (rid := resolve_request_id(item))]
    for key in ("phase_by_req", "phase_reason_by_req", "decode_step_by_req"):
        if isinstance(payload.get(key), dict):
            mapped = {}
            for rid, value in payload[key].items():
                canonical = resolve_request_id(rid)
                if key == "decode_step_by_req":
                    value = to_int(value)
                    if value is None:
                        continue
                if canonical:
                    mapped[canonical] = value
            payload[key] = mapped
    _map_event_ids(payload, resolve_request_id)
    return payload


def normalize_req_event_type(raw_type):
    if raw_type is None:
        return None
    text = str(raw_type).strip().upper()
    for number, needles, kind in (
        ("2", ("DEQUEUE", "SCHEDULED", "DISPATCH"), "dequeue"),
        ("3", ("PREEMPT",), "preempt"),
        ("1", ("QUEUE", "ENQUEUE"), "enqueue"),
    ):
        if text == number or any(needle in text for needle in needles):
            return kind
    return None
