#pragma once
#include "tracer_comm.h"
#include <cstddef>

using RecordEncoder = bool (*)(const UnifiedTraceRecord *, char *, size_t);
bool transport_start(RecordEncoder encoder);
bool transport_ready();
void transport_enqueue(const UnifiedTraceRecord *record);
bool transport_drain(int timeout_ms);
void transport_stop();
unsigned long long transport_ring_dropped();
unsigned long long transport_send_dropped();
