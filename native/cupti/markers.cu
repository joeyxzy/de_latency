#include "markers.h"
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace {
struct Markers {
    std::atomic<uint32_t> sequence{1};
    std::mutex mutex;
    std::unordered_map<uint32_t, std::string> metadata;
};
Markers &markers() { static Markers *instance = new Markers; return *instance; }
}

std::string consume_marker(uint32_t id) {
    auto &state = markers();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto found = state.metadata.find(id);
    if (found == state.metadata.end()) return {};
    std::string text = std::move(found->second);
    state.metadata.erase(found);
    return text;
}

extern "C" __global__ void de_marker() {}

extern "C" int de_marker_push(void *stream_ptr, const char *metadata) {
    if (!metadata) return -1;
    auto &state = markers();
    uint32_t id = state.sequence.fetch_add(1);
    if (!id) id = state.sequence.fetch_add(1);  // gridX cannot be zero
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.metadata[id] = metadata;
    }
    de_marker<<<id, 1, 0, reinterpret_cast<cudaStream_t>(stream_ptr)>>>();
    if (cudaGetLastError() == cudaSuccess) return 0;
    consume_marker(id);
    return -1;
}
