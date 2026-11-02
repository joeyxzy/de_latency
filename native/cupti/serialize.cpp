#include "serialize.h"
#include <json-c/json.h>
#include <cstdlib>
#include <cstring>

bool serialize_record(const UnifiedTraceRecord *r, const char *type, int64_t offset,
                      uint64_t dropped, char *buffer, size_t capacity) {
    json_object *event = json_object_new_object();
    json_object *p = json_object_new_object();
    auto text = [](json_object *o, const char *k, const char *v) {
        json_object_object_add(o, k, json_object_new_string(v));
    };
    auto integer = [](json_object *o, const char *k, int64_t v) {
        json_object_object_add(o, k, json_object_new_int64(v));
    };
    const char *session = std::getenv("DE_LATENCY_SESSION_ID");
    integer(event, "schema_version", 2);
    text(event, "session_id", session ? session : "default");
    text(event, "source", "cupti");
    text(event, "event_type", type);
    text(event, "clock_domain", "cupti");
    integer(event, "clock_offset_ns", offset);
    integer(event, "timestamp_ns", r->end_ns);
    integer(event, "pid", r->pid);
    integer(event, "tid", r->tid);
    integer(event, "producer_dropped", dropped);
    integer(p, "pid", r->pid);
    integer(p, "tid", r->tid);
    text(p, "type", type);
    integer(p, "start_ns", r->start_ns);
    integer(p, "end_ns", r->end_ns);
    integer(p, "correlationId", r->correlationId);
    if (r->type == RECORD_TYPE_DRIVER || r->type == RECORD_TYPE_RUNTIME) {
        integer(p, "cbid", r->cbid);
        text(p, "name", r->name[0] ? r->name : type);
    } else {
        integer(p, "deviceId", r->deviceId);
        integer(p, "contextId", r->contextId);
        integer(p, "streamId", r->streamId);
        if (r->type == RECORD_TYPE_MEMCPY) {
            text(p, "copyKind", r->name);
            integer(p, "runtimeCorrelationId", r->memcpy_runtimeCorrelationId);
        } else if (r->type == RECORD_TYPE_MEMSET) {
            integer(p, "value", r->memset_value);
        } else {
            text(p, "name", r->name);
        }
        if (r->type == RECORD_TYPE_KERNEL) {
            integer(p, "gridX", r->gridX); integer(p, "gridY", r->gridY); integer(p, "gridZ", r->gridZ);
            integer(p, "blockX", r->blockX); integer(p, "blockY", r->blockY); integer(p, "blockZ", r->blockZ);
            integer(p, "staticSharedMemory", r->staticSharedMemory);
            integer(p, "dynamicSharedMemory", r->dynamicSharedMemory);
        }
    }
    json_object_object_add(event, "payload", p);
    const char *json = json_object_to_json_string_ext(event, JSON_C_TO_STRING_PLAIN);
    size_t length = std::strlen(json);
    bool ok = length < capacity;
    if (ok) std::memcpy(buffer, json, length + 1);
    json_object_put(event);
    return ok;
}
