#include "transport.h"
#include "support.h"
#include <atomic>
#include <chrono>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <zmq.h>

namespace {
constexpr size_t capacity = 1024 * 1024;
struct Sender {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<UnifiedTraceRecord *> ring;
    size_t head = 0, tail = 0;
    bool stop = false, inflight = false, initialized = false;
    std::atomic<bool> ready{false};
    std::atomic<unsigned long long> ring_dropped{0}, send_dropped{0};
    std::thread thread;
    RecordEncoder encode = nullptr;
};

Sender &sender() {
    // Explicit shutdown avoids C++ static-destruction order interacting with
    // CUDA's shared-library destructor and late CUPTI callbacks.
    static Sender *instance = new Sender;
    return *instance;
}

void send_record(void *socket, const UnifiedTraceRecord *record, int retries, int retry_us) {
    auto &state = sender();
    char json[8192];
    if (!socket || !state.encode(record, json, sizeof(json))) {
        state.send_dropped.fetch_add(1);
        return;
    }
    for (int attempt = 0; attempt <= retries; ++attempt) {
        if (zmq_send(socket, json, strlen(json), ZMQ_DONTWAIT) >= 0) return;
        if (errno != EAGAIN || attempt == retries) break;
        std::this_thread::sleep_for(std::chrono::microseconds(retry_us));
    }
    state.send_dropped.fetch_add(1);
}

void run() {
    auto &state = sender();
    void *context = zmq_ctx_new();
    void *socket = context ? zmq_socket(context, ZMQ_PUSH) : nullptr;
    const char *address = getenv("TRACER_ZMQ_ADDR");
    int hwm = read_env_int("TRACER_ZMQ_SNDHWM", 200000, 1000, INT_MAX);
    int immediate = 1;
    int timeout = read_env_int("TRACER_ZMQ_SNDTIMEO_MS", -1, -1, INT_MAX);
    int linger = read_env_int("TRACER_ZMQ_LINGER_MS", 30000, 0, INT_MAX);
    int retries = read_env_int("TRACER_ZMQ_SEND_RETRY", 200, 0, 100000);
    int retry_us = read_env_int("TRACER_ZMQ_SEND_RETRY_US", 50, 0, 1000000);
    bool connected = false;
    if (socket && address && *address) {
        zmq_setsockopt(socket, ZMQ_SNDHWM, &hwm, sizeof(hwm));
        zmq_setsockopt(socket, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));
        zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
        zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
        connected = zmq_connect(socket, address) == 0;
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.ready.store(connected);
        state.initialized = true;
    }
    state.changed.notify_all();
    if (connected) {
        tracer_log("ZMQ connected: %s (sndhwm=%d, linger_ms=%d, retry=%d/%dus)",
                   address, hwm, linger, retries, retry_us);
        while (true) {
            UnifiedTraceRecord *record;
            {
                std::unique_lock<std::mutex> lock(state.mutex);
                state.changed.wait(lock, [&] { return state.stop || state.head != state.tail; });
                if (state.head == state.tail && state.stop) break;
                record = state.ring[state.head];
                state.head = (state.head + 1) % capacity;
                state.inflight = true;
            }
            send_record(socket, record, retries, retry_us);
            free(record);
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.inflight = false;
            }
            state.changed.notify_all();
        }
    } else {
        tracer_log("Cannot initialize ZMQ transport: %s", address ? address : "address not configured");
    }
    if (socket) zmq_close(socket);
    if (context) zmq_ctx_term(context);
    state.ready.store(false);
    tracer_log("[send_thread] stopped ring_dropped=%llu zmq_dropped=%llu",
               state.ring_dropped.load(), state.send_dropped.load());
}
}

bool transport_start(RecordEncoder encoder) {
    auto &state = sender();
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.thread.joinable()) return state.ready.load();
    state.ring.resize(capacity);
    state.encode = encoder;
    state.stop = false;
    state.initialized = false;
    state.thread = std::thread(run);
    state.changed.wait(lock, [&] { return state.initialized; });
    return state.ready.load();
}

bool transport_ready() { return sender().ready.load(); }
unsigned long long transport_ring_dropped() { return sender().ring_dropped.load(); }
unsigned long long transport_send_dropped() { return sender().send_dropped.load(); }

void transport_enqueue(const UnifiedTraceRecord *record) {
    auto &state = sender();
    if (!state.ready.load()) { state.send_dropped.fetch_add(1); return; }
    auto *copy = static_cast<UnifiedTraceRecord *>(malloc(sizeof(*record)));
    if (!copy) { state.ring_dropped.fetch_add(1); return; }
    *copy = *record;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        size_t next = (state.tail + 1) % capacity;
        if (next == state.head || state.stop) {
            state.ring_dropped.fetch_add(1);
            free(copy);
            return;
        }
        state.ring[state.tail] = copy;
        state.tail = next;
    }
    state.changed.notify_one();
}

bool transport_drain(int timeout_ms) {
    auto &state = sender();
    std::unique_lock<std::mutex> lock(state.mutex);
    return state.changed.wait_for(lock, std::chrono::milliseconds(timeout_ms),
        [&] { return state.head == state.tail && !state.inflight; });
}

void transport_stop() {
    auto &state = sender();
    { std::lock_guard<std::mutex> lock(state.mutex); state.stop = true; }
    state.changed.notify_all();
    if (state.thread.joinable()) state.thread.join();
}
