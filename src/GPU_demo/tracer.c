// tracer.c (Simplified Version - Time Measurement Only)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <cupti_activity.h>
#include <cupti.h>
#include <cuda.h>

#define CUPTI_CALL(call) do { \
    CUptiResult _status = call; \
    if (_status != CUPTI_SUCCESS) { \
        const char *errstr; \
        cuptiGetResultString(_status, &errstr); \
        fprintf(stderr, "[TRACER] CUPTI error %s:%d: %s\n", __FILE__, __LINE__, errstr); \
        exit(1); \
    } \
} while(0)

// --- Activity Buffer Callbacks ---

// This function allocates a buffer for CUPTI to store activity records.
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    // Allocate a 16MB buffer.
    *size = 16 * 1024 * 1024;
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0; // Let CUPTI manage the number of records.
    if (!*buffer) {
        fprintf(stderr, "[TRACER] Failed to allocate CUPTI activity buffer\n");
        exit(1);
    }
}

// This function is called when a buffer is full or when tracing is flushed.
// It processes the collected activity records.
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                              uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    CUptiResult status;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                switch(record->kind) {
                    case CUPTI_ACTIVITY_KIND_KERNEL: {
                        CUpti_ActivityKernel4 *k = (CUpti_ActivityKernel4*)record;
                        unsigned long long exec_ns = k->end > k->start ? k->end - k->start : 0;
                        printf("[KERNEL] stream=%u name=%s exec_ns=%llu\n",
                            k->streamId, k->name, exec_ns);
                        break;
                    }
                    case CUPTI_ACTIVITY_KIND_MEMCPY: {
                        CUpti_ActivityMemcpy2 *m = (CUpti_ActivityMemcpy2*)record;
                        unsigned long long exec_ns = m->end > m->start ? m->end - m->start : 0;
                        printf("[MEMCPY] kind=%u stream=%u size=%llu exec_ns=%llu\n",
                            m->copyKind, m->streamId, (unsigned long long)m->bytes, exec_ns);
                        break;
                    }
                    default:
                        // Ignore other activity kinds.
                        break;
                }
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                // This error means the buffer is full, which is normal.
                break;
            } else {
                CUPTI_CALL(status);
            }
        } while (1);
    }
    free(buffer);
}

// --- Constructor and Destructor ---

// This function is automatically called when the library is loaded (e.g., via LD_PRELOAD).
void __attribute__((constructor)) tracer_init(void) {
    fprintf(stderr, "[TRACER] Initializing CUPTI Tracer (Time Measurement Only)...\n");
    
    // 1. Register callbacks for activity buffer management.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // 2. Enable the activity kinds we want to trace.
    // We are no longer enabling CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION.
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

    // NOTE: All API callback subscription code has been removed as it's not needed
    // for simple time measurement.

    fprintf(stderr, "[TRACER] CUPTI Tracer Initialized.\n");
}

// This function is automatically called when the library is unloaded.
void __attribute__((destructor)) tracer_fini(void) {
    fprintf(stderr, "[TRACER] Flushing CUPTI data and shutting down...\n");
    // Force CUPTI to deliver any remaining buffered data.
    CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    fprintf(stderr, "[TRACER] Shutdown complete.\n");
}