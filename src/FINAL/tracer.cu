#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <mutex>
#include <stdarg.h> // For va_list, va_start, etc.
#include <fcntl.h>  // For fcntl
#include <errno.h>  // For errno
#include <limits.h>
#include <sys/stat.h>
#include <cuda.h>
#include <cupti.h>
#include <cupti_callbacks.h>
#include <zmq.h>

#include "tracer_comm.h"
#include "encode.h"

// --- Logger ---
// A robust logging function that writes to a dedicated file
void tracer_log(const char* format, ...) {
    FILE* log_file = fopen("/tmp/libtracer.log", "a");
    if (!log_file) return;

    // Add PID prefix to each log entry
    fprintf(log_file, "[TRACER PID %d] ", getpid());

    va_list args;
    va_start(args, format);
    vfprintf(log_file, format, args);
    va_end(args);

    fprintf(log_file, "\n");
    fclose(log_file);
}


// --- CUPTI MACROS AND HELPERS ---
// Modified CUPTI_CALL to use our logger
#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      tracer_log("FATAL ERROR: %s:%d: function %s failed with error %s.", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1); /* In a shared library, exit might be too drastic, but for debugging it's fine */ \
    }                                                                   \
  } while (0)

#define BUF_SIZE (64 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
  // Add other kinds if needed
  default: break;
  }
  return "<unknown>";
}

// --- GLOBAL STATE ---
static std::map<uint32_t, pid_t> g_corr_to_ostid_map;
static std::mutex g_corr_mutex;
static CUpti_SubscriberHandle g_subscriber;
static void *g_zmq_ctx = NULL;
static void *g_zmq_push = NULL;
static char g_zmq_addr[256] = {0};

static void init_zmq_from_env() {
    const char *addr = getenv("TRACER_ZMQ_ADDR");
    if (!addr || !*addr) {
        tracer_log("TRACER_ZMQ_ADDR not set, skip ZMQ");
        return;
    }
    strncpy(g_zmq_addr, addr, sizeof(g_zmq_addr)-1);
    g_zmq_ctx = zmq_ctx_new();
    if (!g_zmq_ctx) {
        tracer_log("zmq_ctx_new failed: %s", strerror(errno));
        return;
    }
    g_zmq_push = zmq_socket(g_zmq_ctx, ZMQ_PUSH);
    if (!g_zmq_push) {
        tracer_log("zmq_socket failed: %s", strerror(errno));
        zmq_ctx_term(g_zmq_ctx); g_zmq_ctx=NULL;
        return;
    }
    if (zmq_connect(g_zmq_push, g_zmq_addr) != 0) {
        tracer_log("zmq_connect(%s) failed: %s", g_zmq_addr, zmq_strerror(errno));
        zmq_close(g_zmq_push); zmq_ctx_term(g_zmq_ctx);
        g_zmq_push=NULL; g_zmq_ctx=NULL;
        return;
    }
    tracer_log("ZMQ connected: %s", g_zmq_addr);
}

static const char *record_type_str(uint32_t t) {
    switch (t) {
        case RECORD_TYPE_KERNEL: return "kernel";
        case RECORD_TYPE_MEMCPY: return "memcpy";
        case RECORD_TYPE_MEMSET: return "memset";
        case RECORD_TYPE_DRIVER: return "driver";
        case RECORD_TYPE_RUNTIME: return "runtime";
        default: return "unknown";
    }
}

static void send_record_payload(const UnifiedTraceRecord *rec) {
    if (!g_zmq_push) {tracer_log("!g_zmq_push\n"); return;}
  
    struct json_object *payload = json_object_new_object();
  
    // 公共字段
    json_object_object_add(payload, "pid", json_object_new_int((int)rec->pid));
    if (rec->tid) {
        json_object_object_add(payload, "tid", json_object_new_int((int)rec->tid));
    }
    json_object_object_add(payload, "type", json_object_new_string(record_type_str(rec->type)));
    json_object_object_add(payload, "start_ns", json_object_new_int64((long long)rec->start_ns));
    json_object_object_add(payload, "end_ns", json_object_new_int64((long long)rec->end_ns));
    json_object_object_add(payload, "correlationId", json_object_new_int((int)rec->correlationId));
  
    // 根据类型添加字段
    switch (rec->type) {
        case RECORD_TYPE_KERNEL: {
            json_object_object_add(payload, "deviceId", json_object_new_int((int)rec->deviceId));
            json_object_object_add(payload, "contextId", json_object_new_int((int)rec->contextId));
            json_object_object_add(payload, "streamId", json_object_new_int((int)rec->streamId));
            if (rec->name[0])
                json_object_object_add(payload, "name", json_object_new_string(rec->name));
            json_object_object_add(payload, "gridX", json_object_new_int((int)rec->gridX));
            json_object_object_add(payload, "gridY", json_object_new_int((int)rec->gridY));
            json_object_object_add(payload, "gridZ", json_object_new_int((int)rec->gridZ));
            json_object_object_add(payload, "blockX", json_object_new_int((int)rec->blockX));
            json_object_object_add(payload, "blockY", json_object_new_int((int)rec->blockY));
            json_object_object_add(payload, "blockZ", json_object_new_int((int)rec->blockZ));
            json_object_object_add(payload, "staticSharedMemory", json_object_new_int((int)rec->staticSharedMemory));
            json_object_object_add(payload, "dynamicSharedMemory", json_object_new_int((int)rec->dynamicSharedMemory));
            break;
        }
        case RECORD_TYPE_MEMCPY: {
            json_object_object_add(payload, "deviceId", json_object_new_int((int)rec->deviceId));
            json_object_object_add(payload, "contextId", json_object_new_int((int)rec->contextId));
            json_object_object_add(payload, "streamId", json_object_new_int((int)rec->streamId));
            if (rec->name[0])
                json_object_object_add(payload, "copyKind", json_object_new_string(rec->name));
            json_object_object_add(payload, "runtimeCorrelationId",
                json_object_new_int((int)rec->memcpy_runtimeCorrelationId));
            break;
        }
        case RECORD_TYPE_MEMSET: {
            json_object_object_add(payload, "deviceId", json_object_new_int((int)rec->deviceId));
            json_object_object_add(payload, "contextId", json_object_new_int((int)rec->contextId));
            json_object_object_add(payload, "streamId", json_object_new_int((int)rec->streamId));
            json_object_object_add(payload, "value", json_object_new_int((int)rec->memset_value));
            break;
        }
        case RECORD_TYPE_DRIVER:
        case RECORD_TYPE_RUNTIME: {
            json_object_object_add(payload, "cbid", json_object_new_int((int)rec->cbid));
            if (rec->tid)
                json_object_object_add(payload, "tid", json_object_new_int((int)rec->tid));
            break;
        }
        default:
            // unknown: 只发送公共字段
            break;
    }
    struct json_object *meta = make_metadata("CUPTI", record_type_str(rec->type), NULL, -1);
    json_object_object_add(meta, "payload", payload);
    const char *js = metadata_to_bytes(meta);
    int rc = zmq_send(g_zmq_push, js, strlen(js), ZMQ_DONTWAIT);
    if (rc < 0) {
        tracer_log("zmq_send failed: %s", zmq_strerror(errno));
    }
    json_object_put(meta);
  }

// --- CUPTI CALLBACKS ---

// This callback is triggered when a CUDA API is entered or exited.
void CUPTIAPI api_callback(
  void *userdata,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cbid,
  const void *cbdata_void)
{
  const CUpti_CallbackData *cbdata = (const CUpti_CallbackData *)cbdata_void;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
      uint32_t correlationId = cbdata->correlationId;
      pid_t os_tid = syscall(SYS_gettid);
      
      // Log that we are capturing a correlation
      // tracer_log("API_ENTER: cbid=%u, corrId=%u, tid=%d", cbid, correlationId, os_tid);
      
      {
          std::lock_guard<std::mutex> lock(g_corr_mutex);
          g_corr_to_ostid_map[correlationId] = os_tid;
      }
  }
}

// This callback is triggered when CUPTI needs a buffer to store activity records.
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    tracer_log("FATAL ERROR: out of memory in bufferRequested");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

// This callback is triggered when a buffer of activity records is ready to be processed.
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    tracer_log("bufferCompleted called. validSize = %zu", validSize);
    if (!g_zmq_push) {
        tracer_log("Warning: Dropping buffer of size %zu because ZMQ socket is not initialized.", validSize);
        if (buffer) free(buffer);
        return;
    }
    if (validSize == 0) {
        if (buffer) free(buffer);
        return;
    }

    CUptiResult status;
    CUpti_Activity *record = NULL;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status != CUPTI_SUCCESS) {
            if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) { // This error is expected at the end
                 const char *errstr;
                 cuptiGetResultString(status, &errstr);
                 tracer_log("ERROR: cuptiActivityGetNextRecord failed with %s", errstr);
            }
            break;
        }

        UnifiedTraceRecord rec = {}; // Zero-initialize
        rec.pid = getpid();
        bool record_valid = true;
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                auto *k = (CUpti_ActivityKernel4 *)record;
                rec.type = RECORD_TYPE_KERNEL;
                rec.start_ns = k->start;
                rec.end_ns = k->end;
                rec.correlationId = k->correlationId;
                rec.deviceId = k->deviceId;
                rec.contextId = k->contextId;
                rec.streamId = k->streamId;
                strncpy(rec.name, k->name, sizeof(rec.name) - 1);
                rec.gridX = k->gridX; rec.gridY = k->gridY; rec.gridZ = k->gridZ;
                rec.blockX = k->blockX; rec.blockY = k->blockY; rec.blockZ = k->blockZ;
                rec.staticSharedMemory = k->staticSharedMemory;
                rec.dynamicSharedMemory = k->dynamicSharedMemory;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto *m = (CUpti_ActivityMemcpy *)record;
                rec.type = RECORD_TYPE_MEMCPY;
                rec.start_ns = m->start;
                rec.end_ns = m->end;
                rec.correlationId = m->correlationId;
                rec.deviceId = m->deviceId;
                rec.contextId = m->contextId;
                rec.streamId = m->streamId;
                strncpy(rec.name, getMemcpyKindString((CUpti_ActivityMemcpyKind)m->copyKind), sizeof(rec.name) - 1);
                rec.memcpy_runtimeCorrelationId = m->runtimeCorrelationId;
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                auto *m = (CUpti_ActivityMemset *)record;
                rec.type = RECORD_TYPE_MEMSET;
                rec.start_ns = m->start;
                rec.end_ns = m->end;
                rec.correlationId = m->correlationId;
                rec.deviceId = m->deviceId;
                rec.contextId = m->contextId;
                rec.streamId = m->streamId;
                rec.memset_value = m->value;
                break;
            }
            case CUPTI_ACTIVITY_KIND_DRIVER:
            case CUPTI_ACTIVITY_KIND_RUNTIME: {
                auto *api = (CUpti_ActivityAPI *)record;
                rec.type = (record->kind == CUPTI_ACTIVITY_KIND_DRIVER) ? RECORD_TYPE_DRIVER : RECORD_TYPE_RUNTIME;
                rec.start_ns = api->start;
                rec.end_ns = api->end;
                rec.correlationId = api->correlationId;
                rec.pid = api->processId;
                rec.cbid = api->cbid;
                
                std::lock_guard<std::mutex> lock(g_corr_mutex);
                auto it = g_corr_to_ostid_map.find(api->correlationId);
                if (it != g_corr_to_ostid_map.end()) {
                    rec.tid = it->second;
                } else {
                    rec.tid = 0; // TID not found
                    // tracer_log("Warning: OS TID for corrId %u not found in map.", api->correlationId);
                }
                break;
            }
            default: 
                record_valid = false;
                tracer_log("Skipping activity kind=%u", record->kind);
                break;
        }

        if (record_valid) {
            send_record_payload(&rec);
        }

    } while (status == CUPTI_SUCCESS);

    free(buffer);
}

// --- CONSTRUCTOR & DESTRUCTOR ---
__attribute__((constructor))
void initCuptiTracer() {
    tracer_log("--- libtracer.so loaded in PID %d ---", getpid());
    init_zmq_from_env();

    tracer_log("Initializing CUPTI...");
    CUPTI_CALL(cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)api_callback, nullptr));
    tracer_log("cuptiSubscribe OK.");

    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    tracer_log("Enabled RUNTIME_API domain.");
    CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    tracer_log("Enabled DRIVER_API domain.");
    
    tracer_log("Registering Activity callbacks...");
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    tracer_log("Activity callbacks registered.");
    
    tracer_log("Enabling Activity kinds...");
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    tracer_log("Activity kinds enabled.");

    tracer_log("--- CUPTI Initialization complete ---");
}

__attribute__((destructor))
void deinitCuptiTracer() {
    tracer_log("--- libtracer.so unloading from PID %d ---", getpid());
    CUPTI_CALL(cuptiActivityFlushAll(0));
    tracer_log("Flushed all remaining CUPTI activity.");
    if (g_zmq_push) {
        zmq_close(g_zmq_push);
        g_zmq_push=NULL;
    }
    if (g_zmq_ctx) {
        zmq_ctx_term(g_zmq_ctx);
        g_zmq_ctx=NULL;
    }
    CUPTI_CALL(cuptiUnsubscribe(g_subscriber));
    tracer_log("Unsubscribed from CUPTI.");
}