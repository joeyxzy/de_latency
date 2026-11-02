#pragma once
#include <atomic>
#include <cupti.h>

struct ActivityState {
    std::atomic<bool> enabled{false}, markers_only{false};
    std::atomic<unsigned long long> filtered_queries{0}, driver_dropped{0};
    bool capture_queries = false;
};
ActivityState &activity_state();
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *max_records);
void CUPTIAPI bufferCompleted(CUcontext context, uint32_t stream, uint8_t *buffer, size_t size, size_t valid);
