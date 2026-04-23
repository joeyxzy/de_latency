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
#include <strings.h>
#include <limits.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <atomic>
#include <string>
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
static int g_zmq_blocking_send = 1;
static int g_zmq_sndtimeo_ms = -1;
static int g_zmq_linger_ms = 30000;
static int g_zmq_send_retry = 200;
static int g_zmq_send_retry_us = 50;
static bool g_activity_callbacks_registered = false;
static std::atomic<bool> g_cupti_enabled{false};
static std::atomic<unsigned long long> g_cupti_generation{0};
static pthread_t g_control_thread;
static std::atomic<bool> g_control_thread_started{false};
static std::atomic<bool> g_control_thread_running{false};
static std::atomic<bool> g_destroying{false};
static std::atomic<int> g_control_listen_fd{-1};
static char g_control_socket_path[CONTROL_SOCKET_PATH_MAX] = {0};
static int g_control_poll_timeout_ms = 500;
static int g_control_backlog = 8;

static bool cupti_enable_locked(const char *reason);
static bool cupti_disable_locked(bool graceful, const char *reason);
static bool cupti_flush_locked(const char *reason);

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
    const bool had_subscriber = (g_subscriber != NULL);
    const bool should_disable_activities = was_enabled || had_subscriber;
    bool ok = true;

    if (!was_enabled && !had_subscriber) {
        tracer_log("[%s] CUPTI already disabled.", phase);
        return true;
    }

    if (graceful && was_enabled) {
        try_sync_current_cuda_context();
    }

    if (should_disable_activities) {
        tracer_log("[%s] Flushing all remaining CUPTI activity...", phase);
        ok = cupti_flush_locked(phase) && ok;

        tracer_log("[%s] Disabling Activity kinds...", phase);
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME)",
                      phase) && ok;
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER)",
                      phase) && ok;
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET)",
                      phase) && ok;
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY)",
                      phase) && ok;
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL),
                      "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL)",
                      phase) && ok;

        tracer_log("[%s] Final CUPTI flush after disabling activity kinds...", phase);
        ok = cupti_flush_locked(phase) && ok;
    }

    if (g_subscriber != NULL) {
        cupti_ok(cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API),
                 "cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API)",
                 phase);
        cupti_ok(cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API),
                 "cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API)",
                 phase);
        ok = cupti_ok(cuptiUnsubscribe(g_subscriber),
                      "cuptiUnsubscribe(g_subscriber)",
                      phase) && ok;
        g_subscriber = NULL;
    }

    {
        std::lock_guard<std::mutex> lock(g_corr_mutex);
        g_corr_to_ostid_map.clear();
    }
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

    if (g_subscriber == NULL) {
        CUpti_SubscriberHandle subscriber = NULL;
        if (!cupti_ok(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)api_callback, nullptr),
                      "cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)api_callback, nullptr)",
                      phase)) {
            return false;
        }
        g_subscriber = subscriber;
        tracer_log("[%s] cuptiSubscribe OK.", phase);
    }

    ok = cupti_ok(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API),
                  "cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API)",
                  phase) && ok;
    ok = cupti_ok(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API),
                  "cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API)",
                  phase) && ok;
    ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)",
                  phase) && ok;
    ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)",
                  phase) && ok;
    ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET)",
                  phase) && ok;
    ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER)",
                  phase) && ok;
    ok = cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),
                  "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)",
                  phase) && ok;

    if (!ok) {
        tracer_log("[%s] CUPTI enable sequence hit an error, attempting cleanup.", phase);
        cupti_disable_locked(false, "enable_rollback");
        return false;
    }

    unsigned long long generation = g_cupti_generation.fetch_add(1) + 1;
    g_cupti_enabled.store(true);
    tracer_log("[%s] CUPTI enabled. generation=%llu", phase, generation);
    return true;
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
             "%s state=%s generation=%llu subscriber=%s activity_callbacks=%s pid=%d socket=%s\n",
             prefix ? prefix : "ok",
             g_cupti_enabled.load() ? "on" : "off",
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
    if (!g_cupti_enabled.load()) {
        if (buffer) free(buffer);
        return;
    }
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
    g_destroying.store(false);
    g_control_poll_timeout_ms = read_env_int("TRACER_CUPTI_CONTROL_POLL_TIMEOUT_MS", 500, 50, 60000);
    init_zmq_from_env();
    start_control_server();

    if (read_env_bool("TRACER_CUPTI_START_ENABLED", 1)) {
        std::lock_guard<std::mutex> lock(g_cupti_state_mutex);
        if (!cupti_enable_locked("constructor")) {
            tracer_log("[INIT] Initial CUPTI enable failed, continuing with CUPTI off.");
        }
    } else {
        tracer_log("[INIT] TRACER_CUPTI_START_ENABLED=0, starting with CUPTI off.");
    }

    tracer_log("--- CUPTI runtime control initialization complete ---");
}

// 假设这些是你在全局作用域或类成员中定义的
// static CUpti_SubscriberHandle g_subscriber;
// static void *g_zmq_ctx = NULL;
// static void *g_zmq_push = NULL;

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
