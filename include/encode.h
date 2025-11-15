// encode.h

#ifndef ENCODE_H
#define ENCODE_H

#include <json-c/json.h>
#include <stddef.h>

// 当前 schema 版本
#define SCHEMA_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

struct json_object* make_metadata(const char *a, const char *b,
                                  struct json_object* c, int d);

const char* metadata_to_bytes(struct json_object*);
struct json_object* metadata_from_bytes(const char*);

#ifdef __cplusplus
}
#endif

#endif
