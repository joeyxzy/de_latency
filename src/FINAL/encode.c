// encode.c

#include "encode.h"
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------
// 构造 metadata JSON dict：
// Python 等价：
// {
//     "source": "...",
//     "event_type": "...",
//     "schema_version": 1,
//     "extra": {...} (可选)
// }
// ----------------------------------------------------------------------
struct json_object *make_metadata(
    const char *source,
    const char *event_type,
    struct json_object *extra,
    int schema_version
) {
    if (schema_version < 0) {
        schema_version = SCHEMA_VERSION;
    }

    struct json_object *meta = json_object_new_object();

    json_object_object_add(meta, "source",
        json_object_new_string(source));

    json_object_object_add(meta, "event_type",
        json_object_new_string(event_type));

    json_object_object_add(meta, "schema_version",
        json_object_new_int(schema_version));

    if (extra != NULL) {
        json_object_object_add(meta, "extra", extra);
    }

    return meta;
}

// ----------------------------------------------------------------------
// metadata_to_bytes：输出 JSON 字符串（C 字符串）
// Python 使用 json.dumps(meta, separators=(",", ":"))
//
// json-c 默认输出就是紧凑格式，不含空格
// ----------------------------------------------------------------------
const char *metadata_to_bytes(struct json_object *meta) {
    return json_object_to_json_string_ext(meta, JSON_C_TO_STRING_PLAIN);
}

// ----------------------------------------------------------------------
// metadata_from_bytes：JSON字符串 → json_object *
// ----------------------------------------------------------------------
struct json_object *metadata_from_bytes(const char *json_str) {
    struct json_object *obj = json_tokener_parse(json_str);
    return obj;
}

