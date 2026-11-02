// CUPTI lifecycle and command policy. IO, activity decoding and GPU markers
// belong to independent components; no transport socket is used on this thread.
#include "activity.h"
#include "control_server.h"
#include "serialize.h"
#include "support.h"
#include "transport.h"
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace {
enum Kind { Kernel = 1, Memcpy = 2, Memset = 4, Driver = 8, RuntimeApi = 16, Concurrent = 32 };
constexpr unsigned all_kinds = 63, kernel_kinds = Kernel | Concurrent;
constexpr unsigned default_kinds = Concurrent | Memcpy | Memset | Driver | RuntimeApi;
struct Runtime {
    std::mutex mutex;
    std::condition_variable changed;
    std::thread delayed;
    bool destroying = false, registered = false;
    unsigned kinds = default_kinds;
    unsigned long long generation = 0;
    std::atomic<int64_t> clock_offset{0};
};
Runtime &runtime() { static Runtime *instance = new Runtime; return *instance; }

bool check(CUptiResult status, const char *operation) {
    if (status == CUPTI_SUCCESS) return true;
    const char *text = "unknown";
    cuptiGetResultString(status, &text);
    tracer_log("%s failed: %s", operation, text);
    return false;
}

bool set_kinds(unsigned kinds, bool enabled) {
    const CUpti_ActivityKind activities[] = {CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL};
    bool ok = true;
    for (unsigned index = 0; index < 6; ++index)
        if (kinds & (1u << index))
            ok = check(enabled ? cuptiActivityEnable(activities[index]) : cuptiActivityDisable(activities[index]),
                       enabled ? "cuptiActivityEnable" : "cuptiActivityDisable") && ok;
    return ok;
}

bool flush() { return check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "cuptiActivityFlushAll"); }

bool disable(bool graceful) {
    auto &state = activity_state();
    if (!state.enabled.load()) return true;
    if (graceful) {
        CUcontext context = nullptr;
        if (cuCtxGetCurrent(&context) == CUDA_SUCCESS && context) cuCtxSynchronize();
    }
    bool ok = flush();
    ok = set_kinds(all_kinds, false) && ok;
    ok = flush() && ok;
    state.enabled.store(false);
    return ok;
}

bool enable() {
    auto &state = activity_state();
    auto &owner = runtime();
    if (state.enabled.load()) return true;
    if (!check(cuptiSetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM), "cuptiSetThreadIdType")) return false;
    if (!owner.registered) {
        if (!check(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted), "cuptiActivityRegisterCallbacks"))
            return false;
        owner.registered = true;
    }
    timespec before{}, after{};
    uint64_t timestamp = 0;
    clock_gettime(CLOCK_REALTIME, &before);
    if (!check(cuptiGetTimestamp(&timestamp), "cuptiGetTimestamp")) return false;
    clock_gettime(CLOCK_REALTIME, &after);
    int64_t left = int64_t(before.tv_sec) * 1000000000LL + before.tv_nsec;
    int64_t right = int64_t(after.tv_sec) * 1000000000LL + after.tv_nsec;
    owner.clock_offset.store(left + (right - left) / 2 - int64_t(timestamp));
    unsigned kinds = state.markers_only.load() ? owner.kinds & kernel_kinds : owner.kinds;
    if (!set_kinds(kinds, true)) { set_kinds(kinds, false); return false; }
    state.enabled.store(true);
    ++owner.generation;
    return true;
}

std::string status(const char *prefix) {
    auto &owner = runtime();
    auto &state = activity_state();
    char text[512];
    snprintf(text, sizeof(text),
        "%s state=%s markers_only=%s generation=%llu subscriber=no activity_callbacks=%s pid=%d socket=%s ring_dropped=%llu zmq_dropped=%llu cupti_dropped=%llu filtered_queries=%llu\n",
        prefix, state.enabled.load() ? "on" : "off", state.markers_only.load() ? "yes" : "no",
        owner.generation, owner.registered ? "yes" : "no", getpid(),
        control_socket_path()[0] ? control_socket_path() : "<disabled>", transport_ring_dropped(),
        transport_send_dropped(), state.driver_dropped.load(), state.filtered_queries.load());
    return text;
}

std::string command(const std::string &text) {
    std::lock_guard<std::mutex> lock(runtime().mutex);
    auto &state = activity_state();
    if (text == "status") return status("ok");
    if (text == "on" || text == "enable") return status(enable() ? "ok" : "error=enable_failed");
    if (text == "off" || text == "disable" || text == "off graceful" || text == "disable graceful") {
        bool ok = disable(true);
        bool drained = transport_drain(30000);
        return status(ok && drained ? "ok" : "error=disable_or_drain_failed");
    }
    if (text == "off fast" || text == "disable fast") return status(disable(false) ? "ok" : "error=disable_failed");
    if (text == "flush") return status(!state.enabled.load() || flush() ? "ok" : "error=flush_failed");
    if (text == "markers_only on" || text == "markers_only off") {
        bool desired = text == "markers_only on";
        bool ok = true;
        if (desired != state.markers_only.load() && state.enabled.load())
            ok = set_kinds(runtime().kinds & ~kernel_kinds, !desired);
        if (ok) state.markers_only.store(desired);
        return status(ok ? "ok" : "error=markers_mode_failed");
    }
    return "error=unknown_command\n";
}

void tick() {
    std::unique_lock<std::mutex> lock(runtime().mutex, std::try_to_lock);
    if (lock.owns_lock() && activity_state().enabled.load())
        check(cuptiActivityFlushAll(0), "periodic flush");
}

bool encode(const UnifiedTraceRecord *record, char *buffer, size_t size) {
    const char *names[] = {"kernel", "memcpy", "runtime", "driver", "memset", "marker"};
    const char *name = unsigned(record->type) < 6 ? names[record->type] : "unknown";
    return serialize_record(record, name, runtime().clock_offset.load(), transport_ring_dropped() +
                            transport_send_dropped() + activity_state().driver_dropped.load(), buffer, size);
}

unsigned configured_kinds() {
    const char *text = getenv("TRACER_CUPTI_ACTIVITY_KINDS");
    if (!text || !*text) return default_kinds;
    unsigned result = 0;
    std::stringstream stream(text);
    std::string token;
    const char *names[] = {"kernel", "memcpy", "memset", "driver", "runtime", "concurrent_kernel"};
    while (std::getline(stream, token, ',')) {
        size_t first = token.find_first_not_of(" \t"), last = token.find_last_not_of(" \t");
        if (first == std::string::npos) continue;
        token = token.substr(first, last - first + 1);
        if (token == "all") result = all_kinds;
        for (unsigned index = 0; index < 6; ++index) if (token == names[index]) result |= 1u << index;
    }
    return result ? result : default_kinds;
}
}

__attribute__((constructor)) void initCuptiTracer() {
    auto &owner = runtime();
    auto &state = activity_state();
    state.capture_queries = read_env_bool("TRACER_CUPTI_CAPTURE_QUERIES", 0);
    state.markers_only.store(read_env_bool("TRACER_CUPTI_MARKERS_ONLY", 0));
    owner.kinds = configured_kinds();
    transport_start(encode);
    control_start(command, tick);
    if (!read_env_bool("TRACER_CUPTI_START_ENABLED", 1)) return;
    int delay = read_env_int("TRACER_CUPTI_START_DELAY_SEC", 15, 0, 300);
    if (!delay) { std::lock_guard<std::mutex> lock(owner.mutex); enable(); return; }
    owner.delayed = std::thread([delay] {
        auto &owner = runtime();
        std::unique_lock<std::mutex> lock(owner.mutex);
        if (!owner.changed.wait_for(lock, std::chrono::seconds(delay), [&] { return owner.destroying; })) enable();
    });
}

__attribute__((destructor)) void deinitCuptiTracer() {
    auto &owner = runtime();
    { std::lock_guard<std::mutex> lock(owner.mutex); owner.destroying = true; }
    owner.changed.notify_all();
    if (owner.delayed.joinable()) owner.delayed.join();
    control_stop();
    { std::lock_guard<std::mutex> lock(owner.mutex); disable(true); }
    transport_stop();
}
