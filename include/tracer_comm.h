#ifndef TRACER_COMM_H
#define TRACER_COMM_H

#include <stdint.h>

// The enum you requested to identify the record type
typedef enum {
    RECORD_TYPE_KERNEL,
    RECORD_TYPE_MEMCPY,
    RECORD_TYPE_RUNTIME,
    RECORD_TYPE_DRIVER,
    RECORD_TYPE_MEMSET,
    RECORD_TYPE_MARKER,           // GPU-side batch marker (de_marker kernel)
    // We still need a few types for internal communication
    RECORD_TYPE_METADATA_TID_MAP, // For sending the TID mapping
    RECORD_TYPE_METADATA_FLUSH_COMPLETE // To signal the end of data
} RecordType;

// The unified structure containing all requested fields
typedef struct {
    RecordType type;

    // Common Fields
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t pid;
    uint32_t tid; // This will hold the OS TID for API calls
    uint32_t correlationId;

    // GPU Activity Fields (used by KERNEL, MEMCPY, MEMSET)
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;

    // API Activity Fields (used by DRIVER, RUNTIME)
    uint32_t cbid;

    // --- Type-Specific Data ---
    char name[256];                     // For KERNEL name or MEMCPY kind
    uint32_t memset_value;              // For MEMSET
    uint32_t memcpy_runtimeCorrelationId; // For MEMCPY

    // For KERNEL geometry
    uint32_t gridX, gridY, gridZ;
    uint32_t blockX, blockY, blockZ;
    uint32_t staticSharedMemory;
    uint32_t dynamicSharedMemory;

} UnifiedTraceRecord;

#endif // TRACER_COMM_H