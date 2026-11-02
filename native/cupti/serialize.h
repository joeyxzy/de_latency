#pragma once
#include <stddef.h>
#include <stdint.h>
#include "tracer_comm.h"
bool serialize_record(const UnifiedTraceRecord *, const char *type, int64_t offset_ns,
                      uint64_t dropped, char *buffer, size_t capacity);
