#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <mutex>
#include <climits>
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

#define BUF_SIZE (1024 * 1024)
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
static std::mutex g_zmq_send_mutex;
static CUpti_SubscriberHandle g_subscriber;
static void *g_zmq_ctx = NULL;
static void *g_zmq_push = NULL;
static char g_zmq_addr[256] = {0};
static int g_zmq_blocking_send = 1;
static int g_zmq_sndtimeo_ms = -1;
static int g_zmq_linger_ms = 30000;
static int g_zmq_send_retry = 200;
static int g_zmq_send_retry_us = 50;

static int read_env_int(const char *key, int default_value, int min_value, int max_value) {
    const char *s = getenv(key);
    if (!s || !*s) return default_value;

    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        tracer_log("Invalid integer for %s=%s, fallback to %d", key, s, default_value);
        return default_value;
    }
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    return (int)v;
}

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
    int sndhwm = read_env_int("TRACER_ZMQ_SNDHWM", 200000, 1000, INT_MAX);
    if (zmq_setsockopt(g_zmq_push, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm)) != 0) {
        tracer_log("zmq_setsockopt(ZMQ_SNDHWM=%d) failed: %s", sndhwm, zmq_strerror(errno));
    }
    int immediate = 1;
    if (zmq_setsockopt(g_zmq_push, ZMQ_IMMEDIATE, &immediate, sizeof(immediate)) != 0) {
        tracer_log("zmq_setsockopt(ZMQ_IMMEDIATE=1) failed: %s", zmq_strerror(errno));
    }
    g_zmq_blocking_send = read_env_int("TRACER_ZMQ_BLOCKING_SEND", 1, 0, 1);
    g_zmq_sndtimeo_ms = read_env_int("TRACER_ZMQ_SNDTIMEO_MS", -1, -1, INT_MAX);
    g_zmq_linger_ms = read_env_int("TRACER_ZMQ_LINGER_MS", 30000, 0, INT_MAX);
    if (zmq_setsockopt(g_zmq_push, ZMQ_SNDTIMEO, &g_zmq_sndtimeo_ms, sizeof(g_zmq_sndtimeo_ms)) != 0) {
        tracer_log("zmq_setsockopt(ZMQ_SNDTIMEO=%d) failed: %s", g_zmq_sndtimeo_ms, zmq_strerror(errno));
    }
    if (zmq_setsockopt(g_zmq_push, ZMQ_LINGER, &g_zmq_linger_ms, sizeof(g_zmq_linger_ms)) != 0) {
        tracer_log("zmq_setsockopt(ZMQ_LINGER=%d) failed: %s", g_zmq_linger_ms, zmq_strerror(errno));
    }
    g_zmq_send_retry = read_env_int("TRACER_ZMQ_SEND_RETRY", 200, 0, 100000);
    g_zmq_send_retry_us = read_env_int("TRACER_ZMQ_SEND_RETRY_US", 50, 0, 1000000);
    if (zmq_connect(g_zmq_push, g_zmq_addr) != 0) {
        tracer_log("zmq_connect(%s) failed: %s", g_zmq_addr, zmq_strerror(errno));
        zmq_close(g_zmq_push); zmq_ctx_term(g_zmq_ctx);
        g_zmq_push=NULL; g_zmq_ctx=NULL;
        return;
    }
    tracer_log("ZMQ connected: %s (sndhwm=%d, blocking_send=%d, sndtimeo_ms=%d, linger_ms=%d, retry=%d, retry_us=%d)",
               g_zmq_addr, sndhwm, g_zmq_blocking_send, g_zmq_sndtimeo_ms, g_zmq_linger_ms,
               g_zmq_send_retry, g_zmq_send_retry_us);
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
            if (rec->name[0] != '\0') {
                json_object_object_add(payload, "name", json_object_new_string(rec->name));
            } else {
                // 如果为空，说明采集时解析失败，给个默认值
                json_object_object_add(payload, "name", json_object_new_string(
                    rec->type == RECORD_TYPE_RUNTIME ? "runtime_unknown" : "driver_unknown"
                ));
            }
            break;
        }
        default:
            // unknown: 只发送公共字段
            break;
    }
    struct json_object *meta = make_metadata("CUPTI", record_type_str(rec->type), NULL, -1);
    json_object_object_add(meta, "payload", payload);
    const char *js = metadata_to_bytes(meta);
    int rc = -1;
    int saved_errno = 0;
    if (g_zmq_blocking_send) {
        std::lock_guard<std::mutex> lock(g_zmq_send_mutex);
        rc = zmq_send(g_zmq_push, js, strlen(js), 0);
        saved_errno = errno;
    } else {
        for (int attempt = 0; attempt <= g_zmq_send_retry; ++attempt) {
            {
                std::lock_guard<std::mutex> lock(g_zmq_send_mutex);
                rc = zmq_send(g_zmq_push, js, strlen(js), ZMQ_DONTWAIT);
                saved_errno = errno;
            }
            if (rc >= 0) break;
            if (saved_errno != EAGAIN || attempt == g_zmq_send_retry) break;
            if (g_zmq_send_retry_us > 0) usleep((useconds_t)g_zmq_send_retry_us);
        }
    }
    if (rc < 0) {
        if (!g_zmq_blocking_send && saved_errno == EAGAIN) {
            tracer_log("zmq_send failed after retries (%d): %s",
                       g_zmq_send_retry, zmq_strerror(saved_errno));
        } else {
            tracer_log("zmq_send failed (blocking_send=%d): %s",
                       g_zmq_blocking_send, zmq_strerror(saved_errno));
        }
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
                const char* funcName = NULL;
                CUpti_CallbackDomain domain = (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) 
                                              ? CUPTI_CB_DOMAIN_RUNTIME_API 
                                              : CUPTI_CB_DOMAIN_DRIVER_API;
                                              
                // 调用 CUPTI API 解析 cbid
                CUptiResult res = cuptiGetCallbackName(domain, api->cbid, &funcName);
                
                if (res == CUPTI_SUCCESS && funcName) {
                    // 安全复制到 rec.name (假设 name 是 char[256])
                    strncpy(rec.name, funcName, sizeof(rec.name) - 1);
                    rec.name[sizeof(rec.name) - 1] = '\0'; // 确保以 null 结尾
                } else {
                    rec.name[0] = '\0'; // 获取失败则置空
                }
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

    size_t dropped = 0;
    CUptiResult drop_status = cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (drop_status == CUPTI_SUCCESS) {
        if (dropped > 0) {
            tracer_log("Warning: CUPTI dropped %zu activity records (streamId=%u).",
                       dropped, streamId);
        }
    } else if (drop_status != CUPTI_ERROR_INVALID_PARAMETER) {
        const char *errstr = NULL;
        cuptiGetResultString(drop_status, &errstr);
        tracer_log("Warning: cuptiActivityGetNumDroppedRecords failed: %s",
                   errstr ? errstr : "unknown");
    }

    free(buffer);
}

//__attribute__((constructor))保证这个共享库在被加载到任何一个进程时，initCuptiTracer函数会被自动调用
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

// 假设这些是你在全局作用域或类成员中定义的
// static CUpti_SubscriberHandle g_subscriber;
// static void *g_zmq_ctx = NULL;
// static void *g_zmq_push = NULL;

//这个是在卸载共享库时自动调用的清理函数
__attribute__((destructor))
void deinitCuptiTracer() {
    tracer_log("--- [DEINIT] libtracer.so unloading from PID %d ---", getpid());

    // --------------------------------------------------------------------
    // 步骤 1: 禁用所有 Activity Kinds
    // 这是最先要做的事情之一，告诉CUPTI不要再产生新的活动记录。
    // 这与初始化的启用顺序相反。
    // --------------------------------------------------------------------
    tracer_log("[DEINIT] Disabling Activity kinds...");
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    tracer_log("[DEINIT] Activity kinds disabled.");


    // --------------------------------------------------------------------
    // 步骤 2: Flush 所有剩余的 Activity 记录
    // 确保在关闭追踪之前，处理完所有已经产生但在缓冲区中的数据。
    // --------------------------------------------------------------------
    tracer_log("[DEINIT] Flushing all remaining CUPTI activity...");
    CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    tracer_log("[DEINIT] Flushed all activity.");


    // --------------------------------------------------------------------
    // 步骤 3: 取消订阅 Callback API
    // 告诉CUPTI，我们不再对任何同步回调感兴趣。
    // 这是非常关键的一步，防止卸载后出现悬空指针调用。
    // --------------------------------------------------------------------
    // 检查 g_subscriber 是否有效，防止在初始化失败的情况下调用
    if (g_subscriber != NULL) {
        tracer_log("[DEINIT] Unsubscribing from CUPTI callbacks...");
        CUPTI_CALL(cuptiUnsubscribe(g_subscriber));
        g_subscriber = NULL; // 置空，好习惯
        tracer_log("[DEINIT] Unsubscribed from CUPTI.");
    }


    // --------------------------------------------------------------------
    // 步骤 4: 清理外部资源，例如 ZMQ
    // 此时所有CUPTI活动都已停止，可以安全地关闭网络连接。
    // --------------------------------------------------------------------
    if (g_zmq_push) {
        tracer_log("[DEINIT] Closing ZMQ socket...");
        // 留出 linger 窗口，尽量把发送队列中的事件冲刷到 collector
        zmq_setsockopt(g_zmq_push, ZMQ_LINGER, &g_zmq_linger_ms, sizeof(g_zmq_linger_ms));
        zmq_close(g_zmq_push);
        g_zmq_push = NULL;
        tracer_log("[DEINIT] ZMQ socket closed.");
    }
    if (g_zmq_ctx) {
        tracer_log("[DEINIT] Terminating ZMQ context...");
        zmq_ctx_term(g_zmq_ctx);
        g_zmq_ctx = NULL;
        tracer_log("[DEINIT] ZMQ context terminated.");
    }
    
    tracer_log("--- [DEINIT] libtracer.so cleanup complete for PID %d ---", getpid());
}
