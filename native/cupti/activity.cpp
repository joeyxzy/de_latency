#include "activity.h"
#include "markers.h"
#include "support.h"
#include "transport.h"
#include <cupti_callbacks.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

ActivityState &activity_state() { static ActivityState *instance = new ActivityState; return *instance; }

namespace {
constexpr size_t buffer_size = 1024 * 1024;
constexpr size_t alignment = 8;

void free_buffer(uint8_t *buffer) { if (buffer) free(reinterpret_cast<void **>(buffer)[-1]); }

bool is_query(CUpti_ActivityKind kind, uint32_t id) {
    if (kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        switch (id) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaPeekAtLastError_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetAttribute_v5000:
            case CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamIsCapturing_v10000: return true;
            default: return false;
        }
    }
    switch (id) {
        case CUPTI_DRIVER_TRACE_CBID_cuEventQuery:
        case CUPTI_DRIVER_TRACE_CBID_cuStreamQuery:
        case CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxGetState:
        case CUPTI_DRIVER_TRACE_CBID_cuCtxGetCurrent:
        case CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction:
        case CUPTI_DRIVER_TRACE_CBID_cuFuncGetAttribute:
        case CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: return true;
        default: return false;
    }
}

template<class Activity>
void common_gpu(UnifiedTraceRecord &result, const Activity &activity) {
    result.start_ns = activity.start; result.end_ns = activity.end;
    result.correlationId = activity.correlationId;
    result.deviceId = activity.deviceId; result.contextId = activity.contextId;
    result.streamId = activity.streamId;
}

void decode(CUpti_Activity *record) {
    auto &state = activity_state();
    bool kernel = record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    bool marker = kernel && strstr(reinterpret_cast<CUpti_ActivityKernel4 *>(record)->name, "de_marker");
    if (state.markers_only.load() && !marker) return;
    UnifiedTraceRecord result{};
    result.pid = getpid();
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto &activity = *reinterpret_cast<CUpti_ActivityKernel4 *>(record);
            result.type = RECORD_TYPE_KERNEL;
            common_gpu(result, activity);
            result.gridX = activity.gridX; result.gridY = activity.gridY; result.gridZ = activity.gridZ;
            result.blockX = activity.blockX; result.blockY = activity.blockY; result.blockZ = activity.blockZ;
            result.staticSharedMemory = activity.staticSharedMemory;
            result.dynamicSharedMemory = activity.dynamicSharedMemory;
            std::string metadata = marker ? consume_marker(activity.gridX) : std::string();
            if (metadata.empty()) strncpy(result.name, activity.name, sizeof(result.name) - 1);
            else snprintf(result.name, sizeof(result.name), "%s:%s", activity.name, metadata.c_str());
            transport_enqueue(&result);
            if (!marker || metadata.empty()) return;
            // Marker records historically carry geometry/correlation as zero.
            result = {};
            result.pid = getpid(); result.type = RECORD_TYPE_MARKER;
            result.start_ns = activity.start; result.end_ns = activity.end;
            result.streamId = activity.streamId; result.deviceId = activity.deviceId;
            result.contextId = activity.contextId;
            strncpy(result.name, metadata.c_str(), sizeof(result.name) - 1);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto &activity = *reinterpret_cast<CUpti_ActivityMemcpy *>(record);
            result.type = RECORD_TYPE_MEMCPY;
            common_gpu(result, activity);
            const char *name = activity.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD ? "HtoD" :
                activity.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH ? "DtoH" :
                activity.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOD ? "DtoD" : "<unknown>";
            strncpy(result.name, name, sizeof(result.name) - 1);
            result.memcpy_runtimeCorrelationId = activity.runtimeCorrelationId;
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto &activity = *reinterpret_cast<CUpti_ActivityMemset *>(record);
            result.type = RECORD_TYPE_MEMSET;
            common_gpu(result, activity);
            result.memset_value = activity.value;
            break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto &activity = *reinterpret_cast<CUpti_ActivityAPI *>(record);
            if (!state.capture_queries && is_query(record->kind, activity.cbid)) {
                state.filtered_queries.fetch_add(1);
                return;
            }
            bool runtime = record->kind == CUPTI_ACTIVITY_KIND_RUNTIME;
            result.type = runtime ? RECORD_TYPE_RUNTIME : RECORD_TYPE_DRIVER;
            result.start_ns = activity.start; result.end_ns = activity.end;
            result.correlationId = activity.correlationId;
            result.pid = activity.processId; result.tid = activity.threadId; result.cbid = activity.cbid;
            const char *name = nullptr;
            if (cuptiGetCallbackName(runtime ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API,
                                     activity.cbid, &name) == CUPTI_SUCCESS && name)
                strncpy(result.name, name, sizeof(result.name) - 1);
            break;
        }
        default: return;
    }
    transport_enqueue(&result);
}
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *max_records) {
    uint8_t *raw = static_cast<uint8_t *>(malloc(buffer_size + alignment + sizeof(void *)));
    if (!raw) { tracer_log("FATAL ERROR: out of memory in bufferRequested"); exit(-1); }
    uintptr_t address = (reinterpret_cast<uintptr_t>(raw + sizeof(void *)) + alignment - 1) & ~(alignment - 1);
    *buffer = reinterpret_cast<uint8_t *>(address);
    reinterpret_cast<void **>(*buffer)[-1] = raw;
    *size = buffer_size;
    *max_records = 0;
}

void CUPTIAPI bufferCompleted(CUcontext context, uint32_t stream, uint8_t *buffer, size_t, size_t valid) {
    if (!activity_state().enabled.load() || !valid) { free_buffer(buffer); return; }
    CUpti_Activity *record = nullptr;
    CUptiResult status;
    while ((status = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) decode(record);
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) tracer_log("Activity decoding failed: %d", int(status));
    size_t dropped = 0;
    if (cuptiActivityGetNumDroppedRecords(context, stream, &dropped) == CUPTI_SUCCESS && dropped) {
        activity_state().driver_dropped.fetch_add(dropped);
        tracer_log("CUPTI dropped %zu records on stream %u", dropped, stream);
    }
    free_buffer(buffer);
}
