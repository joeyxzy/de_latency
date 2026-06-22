#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <map>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <climits>
#include <stdarg.h> // For va_list, va_start, etc.
#include <fcntl.h>  // For fcntl
#include <errno.h>  // For errno
#include <strings.h>
#include <limits.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <atomic>
#include <string>
#include <ctime>
#include <stdint.h>
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

#define BUF_SIZE (1 * 1024 * 1024)
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
void CUPTIAPI api_callback(
  void *userdata,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cbid,
  const void *cbdata_void);
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

#define CONTROL_SOCKET_PATH_MAX 108

static std::map<uint32_t, pid_t> g_corr_to_ostid_map;
static std::mutex g_corr_mutex;
static std::mutex g_zmq_send_mutex;
static std::mutex g_cupti_state_mutex;
static CUpti_SubscriberHandle g_subscriber = NULL;
static void *g_zmq_ctx = NULL;
static void *g_zmq_push = NULL;
static char g_zmq_addr[256] = {0};
static int g_zmq_blocking_send = 0;
static int g_zmq_sndtimeo_ms = -1;
static int g_zmq_linger_ms = 30000;
static int g_zmq_send_retry = 200;
static int g_zmq_send_retry_us = 50;
static int g_flush_after_sync_api = 0;
static bool g_activity_callbacks_registered = false;
static std::atomic<bool> g_cupti_enabled{false};
static std::atomic<bool> g_cupti_markers_only{false};
static std::atomic<unsigned long long> g_cupti_generation{0};
static pthread_t g_control_thread;
static std::atomic<bool> g_control_thread_started{false};
static std::atomic<bool> g_control_thread_running{false};
static std::atomic<bool> g_destroying{false};
static std::atomic<int> g_control_listen_fd{-1};
static char g_control_socket_path[CONTROL_SOCKET_PATH_MAX] = {0};
static int g_control_poll_timeout_ms = 500;
static int g_control_backlog = 8;

#define SEND_RING_SIZE (64 * 1024)
static UnifiedTraceRecord *g_send_ring[SEND_RING_SIZE];
static volatile unsigned int g_send_ring_head = 0;
static volatile unsigned int g_send_ring_tail = 0;
static std::atomic<unsigned long long> g_ring_dropped{0};
static std::atomic<unsigned long long> g_zmq_send_dropped{0};
static std::atomic<unsigned long long> g_cupti_driver_dropped{0};
static pthread_mutex_t g_send_ring_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_send_ring_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_send_thread;
static volatile bool g_send_thread_running = false;
static volatile bool g_send_thread_stop = false;

static int g_cupti_start_delay_sec = 15;
static pthread_t g_cupti_delay_thread;

static std::atomic<uint32_t> g_marker_seq{0};
static std::unordered_map<uint32_t, std::string> g_marker_map;
static std::mutex g_marker_map_mutex;

static unsigned int g_enabled_activity_kinds = 0;
#define ACTIVITY_KIND_KERNEL              (1u << 0)
#define ACTIVITY_KIND_MEMCPY              (1u << 1)
#define ACTIVITY_KIND_MEMSET              (1u << 2)
#define ACTIVITY_KIND_DRIVER              (1u << 3)
#define ACTIVITY_KIND_RUNTIME             (1u << 4)
#define ACTIVITY_KIND_CONCURRENT_KERNEL   (1u << 5)
static unsigned int g_default_activity_kinds = ACTIVITY_KIND_CONCURRENT_KERNEL | ACTIVITY_KIND_MEMCPY | ACTIVITY_KIND_MEMSET | ACTIVITY_KIND_DRIVER | ACTIVITY_KIND_RUNTIME;

static bool cupti_enable_locked(const char *reason);
static bool cupti_disable_locked(bool graceful, const char *reason);
static bool cupti_flush_locked(const char *reason);
static bool enable_activity_kinds_locked(unsigned int kinds, const char *phase);
static bool disable_activity_kinds_locked(unsigned int kinds, const char *phase);

static bool ring_push(UnifiedTraceRecord *rec) {
    unsigned int next_tail = (g_send_ring_tail + 1) % SEND_RING_SIZE;
    if (next_tail == g_send_ring_head) {
        g_ring_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_send_ring[g_send_ring_tail] = rec;
    g_send_ring_tail = next_tail;
    return true;
}

static UnifiedTraceRecord *ring_pop() {
    if (g_send_ring_head == g_send_ring_tail) return NULL;
    UnifiedTraceRecord *rec = g_send_ring[g_send_ring_head];
    g_send_ring_head = (g_send_ring_head + 1) % SEND_RING_SIZE;
    return rec;
}

static bool ring_empty() {
    return g_send_ring_head == g_send_ring_tail;
}

static bool cupti_ok(CUptiResult status, const char *expr, const char *phase) {
    if (status == CUPTI_SUCCESS) {
        return true;
    }

    const char *errstr = "unknown";
    cuptiGetResultString(status, &errstr);
    tracer_log("[%s] %s failed with error %s.",
               phase ? phase : "CUPTI",
               expr ? expr : "<expr>",
               errstr ? errstr : "unknown");
    return false;
}

static bool should_flush_after_api(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        switch (cbid) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020:
            case CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020:
                return true;
            default:
                break;
        }
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        switch (cbid) {
            case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
            case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
            case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
            case CUPTI_DRIVER_TRACE_CBID_cuEventQuery:
            case CUPTI_DRIVER_TRACE_CBID_cuStreamQuery:
                return true;
            default:
                break;
        }
    }
    return false;
}

static void flush_cupti_after_sync_api(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
    if (!g_flush_after_sync_api) {
        return;
    }
    if (!g_cupti_enabled.load()) {
        return;
    }
    if (!should_flush_after_api(domain, cbid)) {
        return;
    }

    CUptiResult status = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (status != CUPTI_SUCCESS) {
        const char *errstr = NULL;
        cuptiGetResultString(status, &errstr);
        tracer_log("Activity flush after sync API failed for domain=%u cbid=%u: %s",
                   (unsigned)domain, (unsigned)cbid, errstr ? errstr : "unknown");
        return;
    }
}

static uint8_t *allocate_cupti_buffer(size_t size) {
    const size_t alloc_size = size + ALIGN_SIZE + sizeof(void *);
    uint8_t *raw = (uint8_t *)malloc(alloc_size);
    if (raw == NULL) {
        return NULL;
    }

    uintptr_t aligned_addr = (uintptr_t)(raw + sizeof(void *));
    aligned_addr = (aligned_addr + (ALIGN_SIZE - 1)) & ~(uintptr_t)(ALIGN_SIZE - 1);
    uint8_t *aligned = (uint8_t *)aligned_addr;
    ((void **)aligned)[-1] = raw;
    return aligned;
}

static void free_cupti_buffer(uint8_t *buffer) {
    if (buffer == NULL) {
        return;
    }
    void *raw = ((void **)buffer)[-1];
    free(raw);
}

static void try_sync_current_cuda_context() {
    CUcontext ctx = nullptr;
    CUresult get_ctx_res = cuCtxGetCurrent(&ctx);
    if (get_ctx_res != CUDA_SUCCESS) {
        tracer_log("[DEINIT] cuCtxGetCurrent failed with error %d", (int)get_ctx_res);
        return;
    }
    if (ctx == nullptr) {
        tracer_log("[DEINIT] No current CUDA context, skip cuCtxSynchronize.");
        return;
    }

    tracer_log("[DEINIT] Synchronizing current CUDA context before flushing CUPTI activity...");
    CUresult sync_res = cuCtxSynchronize();
    if (sync_res != CUDA_SUCCESS) {
        tracer_log("[DEINIT] cuCtxSynchronize failed with error %d", (int)sync_res);
        return;
    }
    tracer_log("[DEINIT] Current CUDA context synchronized.");
}

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

static int read_env_bool(const char *key, int default_value) {
    const char *s = getenv(key);
    if (!s || !*s) {
        return default_value;
    }
    if (strcmp(s, "0") == 0 ||
        strcasecmp(s, "false") == 0 ||
        strcasecmp(s, "off") == 0 ||
        strcasecmp(s, "no") == 0) {
        return 0;
    }
    if (strcmp(s, "1") == 0 ||
        strcasecmp(s, "true") == 0 ||
        strcasecmp(s, "on") == 0 ||
        strcasecmp(s, "yes") == 0) {
        return 1;
    }
    tracer_log("Invalid boolean for %s=%s, fallback to %d", key, s, default_value);
    return default_value;
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
    g_zmq_blocking_send = read_env_int("TRACER_ZMQ_BLOCKING_SEND", 0, 0, 1);
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

static bool register_activity_callbacks_locked(const char *reason) {
    if (g_activity_callbacks_registered) {
        return true;
    }

    if (!cupti_ok(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted),
                  "cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted)",
                  reason)) {
        return false;
    }

    g_activity_callbacks_registered = true;
    tracer_log("[%s] Activity callbacks registered.", reason ? reason : "CUPTI");
    return true;
}

static bool cupti_flush_locked(const char *reason) {
    return cupti_ok(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                    "cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)",
                    reason);
}

static bool cupti_disable_locked(bool graceful, const char *reason) {
    const char *phase = reason ? reason : "cupti_off";
    const bool was_enabled = g_cupti_enabled.load();
    bool ok = true;

    if (!was_enabled) {
        tracer_log("[%s] CUPTI already disabled.", phase);
        return true;
    }

    if (graceful && was_enabled) {
        try_sync_current_cuda_context();
    }

    tracer_log("[%s] Flushing all remaining CUPTI activity...", phase);
    ok = cupti_flush_locked(phase) && ok;

    tracer_log("[%s] Disabling Activity kinds...", phase);
    ok = disable_activity_kinds_locked(ACTIVITY_KIND_KERNEL | ACTIVITY_KIND_CONCURRENT_KERNEL | ACTIVITY_KIND_MEMCPY | ACTIVITY_KIND_MEMSET | ACTIVITY_KIND_DRIVER | ACTIVITY_KIND_RUNTIME, phase) && ok;

    tracer_log("[%s] Final CUPTI flush after disabling activity kinds...", phase);
    ok = cupti_flush_locked(phase) && ok;

    g_cupti_enabled.store(false);
    tracer_log("[%s] CUPTI disabled.", phase);
    return ok;
}

static bool cupti_enable_locked(const char *reason) {
    const char *phase = reason ? reason : "cupti_on";
    bool ok = true;

    if (g_cupti_enabled.load()) {
        tracer_log("[%s] CUPTI already enabled.", phase);
        return true;
    }

    if (!register_activity_callbacks_locked(phase)) {
        return false;
    }

    unsigned int kinds = g_enabled_activity_kinds;
    if (kinds == 0) {
        kinds = g_default_activity_kinds;
    }
    if (g_cupti_markers_only.load()) {
        unsigned int orig = kinds;
        kinds &= (ACTIVITY_KIND_KERNEL | ACTIVITY_KIND_CONCURRENT_KERNEL);
        if (kinds != orig) {
            tracer_log("[%s] markers_only active: activity kinds reduced from 0x%x to 0x%x",
                       phase, orig, kinds);
        }
    }
    tracer_log("[%s] Enabling activity kinds mask=0x%x", phase, kinds);

    ok = enable_activity_kinds_locked(kinds, phase);

    if (!ok) {
        tracer_log("[%s] CUPTI enable sequence hit an error, attempting cleanup.", phase);
        disable_activity_kinds_locked(kinds, "enable_rollback");
        return false;
    }

    unsigned long long generation = g_cupti_generation.fetch_add(1) + 1;
    g_cupti_enabled.store(true);
    tracer_log("[%s] CUPTI enabled. generation=%llu kinds=0x%x", phase, generation, kinds);
    return true;
}

static unsigned int parse_activity_kinds_env() {
    const char *s = getenv("TRACER_CUPTI_ACTIVITY_KINDS");
    if (!s || !*s) {
        return g_default_activity_kinds;
    }

    unsigned int kinds = 0;
    std::string value(s);
    size_t pos = 0;
    while (pos < value.size()) {
        size_t comma = value.find(',', pos);
        std::string token = value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        while (!token.empty() && (token[0] == ' ' || token[0] == '\t')) token.erase(0, 1);
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) token.pop_back();

        if (token == "kernel")  kinds |= ACTIVITY_KIND_KERNEL;
        else if (token == "memcpy")  kinds |= ACTIVITY_KIND_MEMCPY;
        else if (token == "memset")  kinds |= ACTIVITY_KIND_MEMSET;
        else if (token == "driver")  kinds |= ACTIVITY_KIND_DRIVER;
        else if (token == "runtime") kinds |= ACTIVITY_KIND_RUNTIME;
        else if (token == "concurrent_kernel") kinds |= ACTIVITY_KIND_CONCURRENT_KERNEL;
        else if (token == "all")     kinds = ACTIVITY_KIND_KERNEL | ACTIVITY_KIND_CONCURRENT_KERNEL | ACTIVITY_KIND_MEMCPY | ACTIVITY_KIND_MEMSET | ACTIVITY_KIND_DRIVER | ACTIVITY_KIND_RUNTIME;

        pos = comma == std::string::npos ? value.size() : comma + 1;
    }
    if (kinds == 0) kinds = g_default_activity_kinds;
    return kinds;
}

static bool enable_activity_kinds_locked(unsigned int kinds, const char *phase) {
    bool ok = true;

    if (kinds & ACTIVITY_KIND_KERNEL) {
        ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL),
                      "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_MEMCPY) {
        ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),
                      "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_MEMSET) {
        ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET),
                      "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_DRIVER) {
        ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER),
                      "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_RUNTIME) {
        ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),
                      "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_CONCURRENT_KERNEL) {
        ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
                      "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)", phase) && ok;
    }
    return ok;
}

static bool disable_activity_kinds_locked(unsigned int kinds, const char *phase) {
    bool ok = true;

    if (kinds & ACTIVITY_KIND_KERNEL) {
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_MEMCPY) {
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_MEMSET) {
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_DRIVER) {
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_RUNTIME) {
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME)", phase) && ok;
    }
    if (kinds & ACTIVITY_KIND_CONCURRENT_KERNEL) {
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)", phase) && ok;
    }
    return ok;
}

static const char *record_type_str(uint32_t t);

static bool record_to_json(const UnifiedTraceRecord *rec, char *buf, size_t bufsz) {
    int len = 0;
    const char *type_str = record_type_str(rec->type);

    len = snprintf(buf, bufsz,
        "{\"source\":\"CUPTI\",\"event_type\":\"%s\",\"schema_version\":1,"
        "\"payload\":{\"pid\":%u,\"type\":\"%s\",\"start_ns\":%llu,"
        "\"end_ns\":%llu,\"correlationId\":%u",
        type_str, rec->pid, type_str,
        (unsigned long long)rec->start_ns,
        (unsigned long long)rec->end_ns,
        rec->correlationId);
    if ((size_t)len >= bufsz) return false;

    if (rec->tid)
        len += snprintf(buf + len, bufsz - len, ",\"tid\":%u", rec->tid);

    switch (rec->type) {
        case RECORD_TYPE_KERNEL:
            len += snprintf(buf + len, bufsz - len,
                ",\"deviceId\":%u,\"contextId\":%u,\"streamId\":%u",
                rec->deviceId, rec->contextId, rec->streamId);
            if (rec->name[0])
                len += snprintf(buf + len, bufsz - len, ",\"name\":\"%s\"", rec->name);
            len += snprintf(buf + len, bufsz - len,
                ",\"gridX\":%u,\"gridY\":%u,\"gridZ\":%u,"
                "\"blockX\":%u,\"blockY\":%u,\"blockZ\":%u,"
                "\"staticSharedMemory\":%u,\"dynamicSharedMemory\":%u",
                rec->gridX, rec->gridY, rec->gridZ,
                rec->blockX, rec->blockY, rec->blockZ,
                rec->staticSharedMemory, rec->dynamicSharedMemory);
            break;
        case RECORD_TYPE_MEMCPY:
            len += snprintf(buf + len, bufsz - len,
                ",\"deviceId\":%u,\"contextId\":%u,\"streamId\":%u",
                rec->deviceId, rec->contextId, rec->streamId);
            if (rec->name[0])
                len += snprintf(buf + len, bufsz - len, ",\"copyKind\":\"%s\"", rec->name);
            len += snprintf(buf + len, bufsz - len,
                ",\"runtimeCorrelationId\":%u", rec->memcpy_runtimeCorrelationId);
            break;
        case RECORD_TYPE_MEMSET:
            len += snprintf(buf + len, bufsz - len,
                ",\"deviceId\":%u,\"contextId\":%u,\"streamId\":%u,\"value\":%u",
                rec->deviceId, rec->contextId, rec->streamId, rec->memset_value);
            break;
        case RECORD_TYPE_MARKER:
            len += snprintf(buf + len, bufsz - len,
                ",\"deviceId\":%u,\"contextId\":%u,\"streamId\":%u",
                rec->deviceId, rec->contextId, rec->streamId);
            if (rec->name[0])
                len += snprintf(buf + len, bufsz - len, ",\"name\":\"%s\"", rec->name);
            break;
        case RECORD_TYPE_DRIVER:
        case RECORD_TYPE_RUNTIME:
            len += snprintf(buf + len, bufsz - len, ",\"cbid\":%u", rec->cbid);
            if (rec->name[0])
                len += snprintf(buf + len, bufsz - len, ",\"name\":\"%s\"", rec->name);
            else
                len += snprintf(buf + len, bufsz - len, ",\"name\":\"%s\"",
                    rec->type == RECORD_TYPE_RUNTIME ? "runtime_unknown" : "driver_unknown");
            break;
        default:
            break;
    }

    if ((size_t)len >= bufsz) return false;
    len += snprintf(buf + len, bufsz - len, "}}");
    return (size_t)len < bufsz;
}

static bool _zmq_send_with_retry(void *sock, const char *buf, size_t len) {
    int retries = g_zmq_send_retry;
    for (int i = 0; i <= retries; i++) {
        int rc = zmq_send(sock, buf, len, ZMQ_DONTWAIT);
        if (rc >= 0) return true;
        if (errno == EAGAIN) {
            if (i < retries) {
                usleep(g_zmq_send_retry_us);
                continue;
            }
            g_zmq_send_dropped.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    return false;
}

static void _report_drop_stats() {
    static time_t last_report = 0;
    time_t now = time(NULL);
    if (now - last_report >= 10) {
        last_report = now;
        unsigned long long rd = g_ring_dropped.exchange(0, std::memory_order_relaxed);
        unsigned long long sd = g_zmq_send_dropped.exchange(0, std::memory_order_relaxed);
        unsigned long long cd = g_cupti_driver_dropped.exchange(0, std::memory_order_relaxed);
        if (rd || sd || cd) {
            tracer_log("[drop_stats] ring_dropped=%llu zmq_dropped=%llu cupti_dropped=%llu",
                       rd, sd, cd);
        }
    }
}

static void *send_thread_main(void *) {
    g_send_thread_running = true;
    tracer_log("[send_thread] started (ring=%u entries, zmq_retry=%d/%d us)",
               SEND_RING_SIZE, g_zmq_send_retry, g_zmq_send_retry_us);

    while (!g_send_thread_stop) {
        UnifiedTraceRecord *rec = NULL;
        pthread_mutex_lock(&g_send_ring_mutex);
        while (ring_empty() && !g_send_thread_stop) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 1;
            pthread_cond_timedwait(&g_send_ring_cond, &g_send_ring_mutex, &ts);
        }
        rec = ring_pop();
        pthread_mutex_unlock(&g_send_ring_mutex);

        if (rec) {
            char buf[8192];
            if (g_zmq_push && record_to_json(rec, buf, sizeof(buf))) {
                std::lock_guard<std::mutex> lock(g_zmq_send_mutex);
                _zmq_send_with_retry(g_zmq_push, buf, strlen(buf));
            }
            free(rec);
        }
        _report_drop_stats();
    }

    pthread_mutex_lock(&g_send_ring_mutex);
    UnifiedTraceRecord *rec;
    while ((rec = ring_pop()) != NULL) {
        pthread_mutex_unlock(&g_send_ring_mutex);
        char buf[8192];
        if (g_zmq_push && record_to_json(rec, buf, sizeof(buf))) {
            std::lock_guard<std::mutex> lock(g_zmq_send_mutex);
            _zmq_send_with_retry(g_zmq_push, buf, strlen(buf));
        }
        free(rec);
        pthread_mutex_lock(&g_send_ring_mutex);
    }
    pthread_mutex_unlock(&g_send_ring_mutex);

    g_send_thread_running = false;
    tracer_log("[send_thread] stopped ring_dropped=%llu zmq_dropped=%llu cupti_dropped=%llu",
               g_ring_dropped.load(), g_zmq_send_dropped.load(), g_cupti_driver_dropped.load());
    return NULL;
}

static void start_send_thread() {
    if (g_send_thread_running) return;
    g_send_thread_stop = false;
    pthread_create(&g_send_thread, NULL, send_thread_main, NULL);
}

static void stop_send_thread() {
    g_send_thread_stop = true;
    pthread_cond_broadcast(&g_send_ring_cond);
    if (g_send_thread_running || g_send_thread != 0) {
        pthread_join(g_send_thread, NULL);
        g_send_thread = 0;
    }
}

static void build_control_socket_path() {
    const char *dir = getenv("TRACER_CUPTI_CONTROL_DIR");
    if (!dir || !*dir) {
        dir = "/tmp";
    }

    int written = snprintf(g_control_socket_path,
                           sizeof(g_control_socket_path),
                           "%s/de_latency_cupti_%d.sock",
                           dir,
                           getpid());
    if (written < 0 || written >= (int)sizeof(g_control_socket_path)) {
        snprintf(g_control_socket_path,
                 sizeof(g_control_socket_path),
                 "/tmp/de_latency_cupti_%d.sock",
                 getpid());
    }
}

static std::string trim_ascii(const char *text) {
    std::string value = text ? text : "";
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::string build_status_response_locked(const char *prefix) {
    char buf[512];
    snprintf(buf,
             sizeof(buf),
             "%s state=%s markers_only=%s generation=%llu subscriber=%s activity_callbacks=%s pid=%d socket=%s\n",
             prefix ? prefix : "ok",
             g_cupti_enabled.load() ? "on" : "off",
             g_cupti_markers_only.load() ? "yes" : "no",
             (unsigned long long)g_cupti_generation.load(),
             g_subscriber != NULL ? "yes" : "no",
             g_activity_callbacks_registered ? "yes" : "no",
             getpid(),
             g_control_socket_path[0] ? g_control_socket_path : "<disabled>");
    return std::string(buf);
}

static std::string handle_control_command(const std::string &command) {
    std::lock_guard<std::mutex> lock(g_cupti_state_mutex);

    if (command == "status") {
        return build_status_response_locked("ok");
    }
    if (command == "on" || command == "enable") {
        return build_status_response_locked(
            cupti_enable_locked("control:on") ? "ok" : "error=enable_failed");
    }
    if (command == "off" || command == "disable" ||
        command == "off graceful" || command == "disable graceful") {
        return build_status_response_locked(
            cupti_disable_locked(true, "control:off") ? "ok" : "error=disable_failed");
    }
    if (command == "off fast" || command == "disable fast") {
        return build_status_response_locked(
            cupti_disable_locked(false, "control:off_fast") ? "ok" : "error=disable_failed");
    }
    if (command == "flush") {
        if (!g_cupti_enabled.load() && g_subscriber == NULL) {
            return build_status_response_locked("ok");
        }
        return build_status_response_locked(
            cupti_flush_locked("control:flush") ? "ok" : "error=flush_failed");
    }
    if (command == "markers_only on") {
        g_cupti_markers_only.store(true);
        disable_activity_kinds_locked(
            ACTIVITY_KIND_DRIVER | ACTIVITY_KIND_RUNTIME |
            ACTIVITY_KIND_MEMCPY | ACTIVITY_KIND_MEMSET,
            "markers_only_on");
        tracer_log("[control] markers_only enabled (non-kernel activity kinds disabled)");
        return build_status_response_locked("ok");
    }
    if (command == "markers_only off") {
        g_cupti_markers_only.store(false);
        enable_activity_kinds_locked(
            ACTIVITY_KIND_DRIVER | ACTIVITY_KIND_RUNTIME |
            ACTIVITY_KIND_MEMCPY | ACTIVITY_KIND_MEMSET,
            "markers_only_off");
        tracer_log("[control] markers_only disabled (all activity kinds restored)");
        return build_status_response_locked("ok");
    }
    return std::string("error=unknown_command\n");
}

static void handle_control_client(int client_fd) {
    char buf[256] = {0};
    ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n < 0) {
        tracer_log("[control] recv failed: %s", strerror(errno));
        return;
    }

    std::string command = trim_ascii(buf);
    if (command.empty()) {
        command = "status";
    }
    tracer_log("[control] received command '%s'", command.c_str());

    const std::string response = handle_control_command(command);
    const char *data = response.c_str();
    size_t remaining = response.size();
    while (remaining > 0) {
        ssize_t sent = send(client_fd, data, remaining, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            tracer_log("[control] send failed: %s", strerror(errno));
            break;
        }
        data += sent;
        remaining -= (size_t)sent;
    }
}

static void *control_thread_main(void *) {
    g_control_thread_running.store(true);
    tracer_log("[control] listening on %s", g_control_socket_path);

    while (!g_destroying.load()) {
        const int listen_fd = g_control_listen_fd.load();
        if (listen_fd < 0) {
            break;
        }

        struct pollfd pfd = {};
        pfd.fd = listen_fd;
        pfd.events = POLLIN;

        const int poll_rc = poll(&pfd, 1, g_control_poll_timeout_ms);
        if (poll_rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            tracer_log("[control] poll failed: %s", strerror(errno));
            break;
        }
        if (poll_rc == 0) {
            continue;
        }
        if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            if (!g_destroying.load()) {
                tracer_log("[control] poll revents=%d, stopping control loop", pfd.revents);
            }
            break;
        }
        if ((pfd.revents & POLLIN) == 0) {
            continue;
        }

        const int client_fd = accept(listen_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (g_destroying.load() && (errno == EBADF || errno == EINVAL)) {
                break;
            }
            tracer_log("[control] accept failed: %s", strerror(errno));
            continue;
        }

        handle_control_client(client_fd);
        close(client_fd);
    }

    g_control_thread_running.store(false);
    tracer_log("[control] thread stopped");
    return NULL;
}

static void start_control_server() {
    build_control_socket_path();
    unlink(g_control_socket_path);

    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        tracer_log("[control] socket() failed: %s", strerror(errno));
        g_control_socket_path[0] = '\0';
        return;
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_control_socket_path, sizeof(addr.sun_path) - 1);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        tracer_log("[control] bind(%s) failed: %s", g_control_socket_path, strerror(errno));
        close(fd);
        unlink(g_control_socket_path);
        g_control_socket_path[0] = '\0';
        return;
    }
    if (listen(fd, g_control_backlog) != 0) {
        tracer_log("[control] listen(%s) failed: %s", g_control_socket_path, strerror(errno));
        close(fd);
        unlink(g_control_socket_path);
        g_control_socket_path[0] = '\0';
        return;
    }
    chmod(g_control_socket_path, 0600);

    g_control_listen_fd.store(fd);
    if (pthread_create(&g_control_thread, NULL, control_thread_main, NULL) != 0) {
        tracer_log("[control] pthread_create failed: %s", strerror(errno));
        close(fd);
        unlink(g_control_socket_path);
        g_control_listen_fd.store(-1);
        g_control_socket_path[0] = '\0';
        return;
    }
    g_control_thread_started.store(true);
}

static void stop_control_server() {
    const int fd = g_control_listen_fd.exchange(-1);
    if (fd >= 0) {
        close(fd);
    }
    if (g_control_thread_started.exchange(false)) {
        pthread_join(g_control_thread, NULL);
    }
    if (g_control_socket_path[0]) {
        unlink(g_control_socket_path);
        g_control_socket_path[0] = '\0';
    }
}

static const char *record_type_str(uint32_t t) {
    switch (t) {
        case RECORD_TYPE_KERNEL: return "kernel";
        case RECORD_TYPE_MEMCPY: return "memcpy";
        case RECORD_TYPE_MEMSET: return "memset";
        case RECORD_TYPE_DRIVER: return "driver";
        case RECORD_TYPE_RUNTIME: return "runtime";
        case RECORD_TYPE_MARKER: return "marker";
        default: return "unknown";
    }
}

static void send_record_payload(const UnifiedTraceRecord *rec) {
    if (!g_zmq_push) return;
    UnifiedTraceRecord *copy = (UnifiedTraceRecord *)malloc(sizeof(UnifiedTraceRecord));
    if (!copy) return;
    memcpy(copy, rec, sizeof(UnifiedTraceRecord));
    pthread_mutex_lock(&g_send_ring_mutex);
    if (!ring_push(copy)) {
        free(copy);
    }
    pthread_mutex_unlock(&g_send_ring_mutex);
    pthread_cond_signal(&g_send_ring_cond);
}

// --- CUPTI CALLBACKS ---

// This callback is triggered when a CUDA API is entered or exited.
void CUPTIAPI api_callback(
  void *userdata,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cbid,
  const void *cbdata_void)
{
  if (!g_cupti_enabled.load()) {
      return;
  }
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
      return;
  }

  if (cbdata->callbackSite == CUPTI_API_EXIT) {
      // 在同步/查询 API 返回时，相关 GPU 工作通常已经完成，且 CUDA context 仍然有效。
      // 这里主动 flush，可避免等到进程退出时才拿到尾部 activity，进而减少 start/end=0 的记录。
      flush_cupti_after_sync_api(domain, cbid);
  }
}

// This callback is triggered when CUPTI needs a buffer to store activity records.
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = allocate_cupti_buffer(BUF_SIZE);
  if (bfr == NULL) {
    tracer_log("FATAL ERROR: out of memory in bufferRequested");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = bfr;
  *maxNumRecords = 0;
}

// This callback is triggered when a buffer of activity records is ready to be processed.
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    if (!g_cupti_enabled.load()) {
        free_cupti_buffer(buffer);
        return;
    }
    if (!g_zmq_push) {
        tracer_log("Warning: Dropping buffer of size %zu because ZMQ socket is not initialized.", validSize);
        free_cupti_buffer(buffer);
        return;
    }
    if (validSize == 0) {
        free_cupti_buffer(buffer);
        return;
    }

    static std::atomic<unsigned long long> bc_count{0};
    unsigned long long count = bc_count.fetch_add(1) + 1;
    if (count % 1000 == 1) {
        tracer_log("[bufferCompleted] call #%llu validSize=%zu",
                   count, validSize);
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

        // markers-only 模式：CUPTI 驱动层已禁用 runtime/driver/memcpy/memset，
        // 此处作为安全网二次过滤，跳过非 de_marker 的 kernel 记录。
        if (g_cupti_markers_only.load()) {
            bool is_marker_kernel = false;
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                auto *k = (CUpti_ActivityKernel4 *)record;
                if (strstr(k->name, "de_marker") != NULL) {
                    is_marker_kernel = true;
                }
            }
            if (!is_marker_kernel) {
                continue;
            }
        }

        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                auto *k = (CUpti_ActivityKernel4 *)record;
                if (strstr(k->name, "de_marker") != NULL) {
                    // 1. 先查 metadata，用于 kernel 名称和 marker 记录
                    std::string meta_str;
                    {
                        std::lock_guard<std::mutex> lk(g_marker_map_mutex);
                        auto it = g_marker_map.find(k->gridX);
                        if (it != g_marker_map.end()) {
                            meta_str = it->second;
                            g_marker_map.erase(it);
                        }
                    }

                    // 2. 发 kernel 记录，name 中加入 role|dispatch_key 便于观察
                    UnifiedTraceRecord kern_copy = {};
                    kern_copy.type     = RECORD_TYPE_KERNEL;
                    kern_copy.start_ns = k->start;
                    kern_copy.end_ns   = k->end;
                    kern_copy.correlationId = k->correlationId;
                    kern_copy.deviceId      = k->deviceId;
                    kern_copy.contextId     = k->contextId;
                    kern_copy.streamId      = k->streamId;
                    kern_copy.pid           = getpid();
                    if (meta_str.empty()) {
                        strncpy(kern_copy.name, k->name, sizeof(kern_copy.name) - 1);
                    } else {
                        snprintf(kern_copy.name, sizeof(kern_copy.name),
                                 "%s:%s", k->name, meta_str.c_str());
                    }
                    kern_copy.gridX  = k->gridX;  kern_copy.gridY  = k->gridY;  kern_copy.gridZ  = k->gridZ;
                    kern_copy.blockX = k->blockX; kern_copy.blockY = k->blockY; kern_copy.blockZ = k->blockZ;
                    kern_copy.staticSharedMemory  = k->staticSharedMemory;
                    kern_copy.dynamicSharedMemory = k->dynamicSharedMemory;
                    send_record_payload(&kern_copy);

                    // 3. 发 marker 记录（metadata 专用于批次定界）
                    rec.type     = RECORD_TYPE_MARKER;
                    rec.start_ns = k->start;
                    rec.end_ns   = k->end;
                    rec.streamId = k->streamId;
                    rec.deviceId = k->deviceId;
                    rec.contextId = k->contextId;
                    rec.pid      = getpid();
                    if (!meta_str.empty()) {
                        strncpy(rec.name, meta_str.c_str(), sizeof(rec.name) - 1);
                    }
                    if (rec.name[0]) {
                        record_valid = true;
                    } else {
                        record_valid = false;
                    }
                    break;
                }
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
                rec.tid = api->threadId;
                rec.cbid = api->cbid;
                const char* funcName = NULL;
                CUpti_CallbackDomain domain = (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) 
                                              ? CUPTI_CB_DOMAIN_RUNTIME_API 
                                              : CUPTI_CB_DOMAIN_DRIVER_API;
                                              
                // 调用 CUPTI API 解析 cbid
                CUptiResult res = cuptiGetCallbackName(domain, api->cbid, &funcName);
                
                if (res == CUPTI_SUCCESS && funcName) {
                    strncpy(rec.name, funcName, sizeof(rec.name) - 1);
                    rec.name[sizeof(rec.name) - 1] = '\0';
                } else {
                    rec.name[0] = '\0';
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
            g_cupti_driver_dropped.fetch_add(dropped, std::memory_order_relaxed);
            tracer_log("Warning: CUPTI dropped %zu activity records (streamId=%u) total=%llu.",
                       dropped, streamId, g_cupti_driver_dropped.load());
        }
    } else if (drop_status != CUPTI_ERROR_INVALID_PARAMETER) {
        const char *errstr = NULL;
        cuptiGetResultString(drop_status, &errstr);
        tracer_log("Warning: cuptiActivityGetNumDroppedRecords failed: %s",
                   errstr ? errstr : "unknown");
    }

    free_cupti_buffer(buffer);
}

static void *cupti_delayed_enable_thread(void *) {
    tracer_log("[delay] sleeping %d seconds before CUPTI enable", g_cupti_start_delay_sec);
    sleep(g_cupti_start_delay_sec);

    if (g_destroying.load()) {
        tracer_log("[delay] process exiting, skipping CUPTI enable");
        return NULL;
    }

    std::lock_guard<std::mutex> lock(g_cupti_state_mutex);
    if (!cupti_enable_locked("delayed")) {
        tracer_log("[delay] CUPTI enable failed");
    }
    return NULL;
}

// --- GPU-side batch marker kernel and export ---
extern "C" __global__ void de_marker() {}

extern "C" int de_marker_push(void* stream_ptr, const char* meta) {
    if (meta == nullptr) return -1;
    uint32_t id = g_marker_seq.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_marker_map_mutex);
        g_marker_map[id] = std::string(meta);
    }
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (s == nullptr) s = 0;
    de_marker<<<id, 1, 0, s>>>();
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

//__attribute__((constructor))保证这个共享库在被加载到任何一个进程时，initCuptiTracer函数会被自动调用
__attribute__((constructor))
void initCuptiTracer() {
    tracer_log("--- libtracer.so loaded in PID %d ---", getpid());
    g_destroying.store(false);
    g_control_poll_timeout_ms = read_env_int("TRACER_CUPTI_CONTROL_POLL_TIMEOUT_MS", 500, 50, 60000);
    g_flush_after_sync_api = read_env_bool("TRACER_CUPTI_FLUSH_AFTER_SYNC_API", 0);
    g_cupti_start_delay_sec = read_env_int("TRACER_CUPTI_START_DELAY_SEC", 15, 0, 300);
    g_cupti_markers_only.store(
        read_env_bool("TRACER_CUPTI_MARKERS_ONLY", 0));
    g_enabled_activity_kinds = parse_activity_kinds_env();
    tracer_log("[INIT] Activity kinds mask=0x%x", g_enabled_activity_kinds);
    init_zmq_from_env();
    start_send_thread();
    start_control_server();

    if (read_env_bool("TRACER_CUPTI_START_ENABLED", 1)) {
        if (g_cupti_start_delay_sec > 0) {
            tracer_log("[INIT] CUPTI will enable after %d seconds (triton JIT warmup)",
                       g_cupti_start_delay_sec);
            pthread_create(&g_cupti_delay_thread, NULL, cupti_delayed_enable_thread, NULL);
        } else {
            std::lock_guard<std::mutex> lock(g_cupti_state_mutex);
            if (!cupti_enable_locked("constructor")) {
                tracer_log("[INIT] Initial CUPTI enable failed, continuing with CUPTI off.");
            }
        }
    } else {
        tracer_log("[INIT] TRACER_CUPTI_START_ENABLED=0, starting with CUPTI off.");
    }

    tracer_log("--- CUPTI runtime control initialization complete ---");
}

//这个是在卸载共享库时自动调用的清理函数
__attribute__((destructor))
void deinitCuptiTracer() {
    tracer_log("--- [DEINIT] libtracer.so unloading from PID %d ---", getpid());
    g_destroying.store(true);
    stop_control_server();

    {
        std::lock_guard<std::mutex> lock(g_cupti_state_mutex);
        if (!cupti_disable_locked(true, "deinit")) {
            tracer_log("[DEINIT] CUPTI disable reported errors.");
        }
    }

    tracer_log("[DEINIT] Stopping send thread and draining ring buffer...");
    stop_send_thread();

    if (g_zmq_push) {
        tracer_log("[DEINIT] Closing ZMQ socket...");
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
