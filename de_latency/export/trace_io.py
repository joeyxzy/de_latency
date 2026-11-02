"""Atomic JSON/gzip IO and bounded-memory iteration over Perfetto events."""
from contextlib import contextmanager
import gzip
import io
import json
import os
from pathlib import Path
import tempfile


def artifact_stem(path):
    path = Path(path)
    if path.suffix == ".gz":
        path = path.with_suffix("")
    return path.with_suffix("")


def overview_path(path):
    return Path(str(artifact_stem(path)) + ".overview.json.gz")


def open_json_input(path):
    path = Path(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_json(path):
    with open_json_input(path) as stream:
        return json.load(stream)


@contextmanager
def json_output(path):
    """Never leave a half-written target or mistake a .gz suffix for compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as raw:
            if path.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                        yield stream
            else:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                    yield stream
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_trace(path, events, *, metadata=None):
    """Serialize each event once, without an intermediate JSON string/list."""
    count = 0
    with json_output(path) as stream:
        stream.write('{"traceEvents":[')
        for event in events:
            if count:
                stream.write(",")
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            count += 1
        stream.write("]")
        if metadata is not None:
            stream.write(',"otherData":')
            stream.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
        stream.write("}")
    return count


def iter_trace_events(path):
    """Stream plain/gzipped traces. Do not json.load a multi-GB trace."""
    import ijson

    with open_json_input(path) as stream:
        try:
            # Validate the root/container, including the valid empty-array case.
            parser = ijson.parse(stream, use_float=True)
            if next(parser, None) != ("", "start_map", None):
                raise ValueError("expected a Perfetto object with a traceEvents array")
            for prefix, kind, value in parser:
                if prefix == "traceEvents":
                    if kind != "start_array":
                        raise ValueError("traceEvents must be an array")
                    break
            else:
                raise ValueError("missing traceEvents array")
            del parser
            stream.seek(0)
            for event in ijson.items(stream, "traceEvents.item", use_float=True):
                if not isinstance(event, dict):
                    raise ValueError("traceEvents entries must be objects")
                yield event
        except (ijson.JSONError, EOFError) as exc:
            raise ValueError(f"invalid or truncated trace {path}: {exc}") from exc
