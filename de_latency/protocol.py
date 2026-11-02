"""Wire protocol v2. One UTF-8 JSON object per ZMQ message / log line.

All time fields ending in ``_ns`` are integer nanoseconds. ``clock_domain``
describes payload timestamps, not collector receive time. Legacy decoding is
deliberately confined to this module; new producers must emit v2.
"""
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field

SCHEMA_VERSION = 2
SOURCES = frozenset({"monkey_patch", "cupti", "ebpf"})
CLOCKS = frozenset({"realtime", "monotonic", "cupti"})


def make_event(source, event_type, payload, *, clock_domain="realtime", extra=None):
    payload = dict(payload or {})
    event = {
        "schema_version": SCHEMA_VERSION,
        "session_id": os.environ.get("DE_LATENCY_SESSION_ID", "default"),
        "source": source.lower(),
        "event_type": event_type,
        "clock_domain": clock_domain,
        "timestamp_ns": payload.get("timestamp_ns", payload.get("end_ns", time.time_ns())),
        "pid": payload.get("pid", os.getpid()),
        "tid": payload.get("tid"),
        "payload": payload,
    }
    if extra:
        event["extra"] = extra
    return event


def validate_event(event):
    if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("expected schema_version=2 object")
    if event.get("source") not in SOURCES:
        raise ValueError("unknown event source")
    if event.get("clock_domain") not in CLOCKS:
        raise ValueError("unknown clock domain")
    if not isinstance(event.get("event_type"), str) or not event["event_type"]:
        raise ValueError("missing event_type")
    if not isinstance(event.get("session_id"), str) or not event["session_id"]:
        raise ValueError("missing session_id")
    if not isinstance(event.get("payload"), dict):
        raise ValueError("payload must be an object")
    for obj in (event, event["payload"]):
        for key, value in obj.items():
            if key.endswith("_ns") and value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{key} must be integer nanoseconds")
    return event


def decode_record(record):
    """Accept v2 and historical collector/raw records, returning one envelope."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if record.get("schema_version") == SCHEMA_VERSION:
        return validate_event(record)
    if record.get("schema_version", 1) not in (None, 1):
        raise ValueError("unsupported schema version")
    meta = record.get("meta", record)
    payload = meta.get("payload") or record.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("legacy payload must be an object")
    source = str(meta.get("source", record.get("src", "monkey_patch"))).lower()
    kind = meta.get("event_type")
    domain = "monotonic" if source == "ebpf" or kind == "worker_model_execute_span" else "realtime"
    event = {
        "schema_version": 1,
        "session_id": "legacy",
        "source": source,
        "event_type": kind,
        "clock_domain": domain,
        "timestamp_ns": meta.get("timestamp_ns", payload.get("timestamp_ns")),
        "pid": payload.get("pid"),
        "tid": payload.get("tid"),
        "payload": payload,
    }
    return event


def analysis_record(event):
    """Adapt the canonical envelope to the established attribution payloads.

The worker monotonic span uses emit_wall_ns only as a calibration sample;
its start/end stay monotonic for OS overlap attribution.
"""
    payload = dict(event["payload"])
    for key in ("pid", "tid"):
        if key not in payload and event.get(key) is not None:
            payload[key] = event[key]
    ts = event.get("timestamp_ns")
    if event["event_type"] == "worker_model_execute_span":
        ts = payload.get("emit_wall_ns", ts)
    if event["source"] == "cupti" and event["clock_domain"] == "cupti":
        offset = event.get("clock_offset_ns")
        if not isinstance(offset, int):
            raise ValueError("CUPTI event lacks clock_offset_ns calibration")
        for key in ("start_ns", "end_ns", "timestamp_ns"):
            if payload.get(key, 0) > 0:
                payload[key] += offset
        ts = ts + offset if ts else ts
    return {
        "source": event["source"], "event_type": event["event_type"],
        "timestamp_ns": ts, "payload": payload,
    }


@dataclass
class LogDiagnostics:
    sources: Counter = field(default_factory=Counter)
    event_types: Counter = field(default_factory=Counter)
    sessions: set = field(default_factory=set)
    schema_versions: set = field(default_factory=set)
    invalid_lines: int = 0
    errors: list = field(default_factory=list)

    def as_dict(self):
        return {
            "sources": dict(self.sources), "event_types": dict(self.event_types),
            "sessions": sorted(self.sessions), "invalid_lines": self.invalid_lines,
            "errors": self.errors,
        }


def iter_log(path, diagnostics=None, *, strict=False):
    stats = diagnostics if diagnostics is not None else LogDiagnostics()
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                # Old logs occasionally contain a text prefix before JSON.
                record = json.loads(line[line.index("{"):])
                event = decode_record(record)
                if event["source"] not in SOURCES:
                    raise ValueError("unknown source")
                stats.sources[event["source"]] += 1
                stats.event_types[f'{event["source"]}:{event["event_type"]}'] += 1
                stats.sessions.add(event["session_id"])
                stats.schema_versions.add(event["schema_version"])
                yield event
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                stats.invalid_lines += 1
                if len(stats.errors) < 10:
                    stats.errors.append(f"line {number}: {exc}")
                if strict:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
