import json

# 当前 metadata schema 版本
SCHEMA_VERSION = 1

def make_metadata(source, event_type, extra=None, schema_version=None):
    """
    构造 metadata dict。可序列化为 JSON 并作为第一帧发送。
    extra: 可选字典，放任意小量元信息（例如 pid, tid, record_size 等）
    """
    if schema_version is None:
        schema_version = SCHEMA_VERSION
    meta = {
        "source": source,
        "event_type": event_type,
        "schema_version": schema_version,
    }
    if extra:
        meta["extra"] = extra
    return meta

def metadata_to_bytes(meta_dict):
    # 使用 json dumps; 如果需要性能可替换为 msgpack
    return json.dumps(meta_dict, separators=(",", ":" )).encode()

def metadata_from_bytes(b):
    import json
    return json.loads(b.decode())