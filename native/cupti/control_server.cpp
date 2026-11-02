#include "control_server.h"
#include "support.h"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace {
struct Server {
    std::atomic<bool> stop{false};
    int fd = -1;
    char path[108]{};
    std::thread thread;
    ControlCommand command = nullptr;
    ControlTick tick = nullptr;
};
Server &server() { static Server *instance = new Server; return *instance; }

void handle_client(int fd) {
    // A client that never sends must not prevent library shutdown.
    timeval timeout{1, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    char buffer[256]{};
    ssize_t count = recv(fd, buffer, sizeof(buffer) - 1, 0);
    if (count < 0) return;
    std::string text(buffer, count);
    size_t first = text.find_first_not_of(" \t\r\n");
    text = first == std::string::npos ? "status" : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
    tracer_log("[control] received command '%s'", text.c_str());
    std::string response = server().command(text);
    size_t sent = 0;
    while (sent < response.size()) {
        ssize_t result = send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) break;
        sent += result;
    }
}

void run() {
    auto &state = server();
    int timeout = read_env_int("TRACER_CUPTI_CONTROL_POLL_TIMEOUT_MS", 500, 50, 60000);
    tracer_log("[control] listening on %s", state.path);
    while (!state.stop.load()) {
        pollfd poll_fd{state.fd, POLLIN, 0};
        int result = poll(&poll_fd, 1, timeout);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0 || (poll_fd.revents & (POLLERR | POLLHUP | POLLNVAL))) break;
        if (result == 0) { if (state.tick) state.tick(); continue; }
        if (!(poll_fd.revents & POLLIN)) continue;
        int client = accept(state.fd, nullptr, nullptr);
        if (client >= 0) { handle_client(client); close(client); }
    }
    tracer_log("[control] thread stopped");
}
}

const char *control_socket_path() { return server().path; }

void control_start(ControlCommand command, ControlTick tick) {
    auto &state = server();
    if (state.thread.joinable()) return;
    const char *directory = getenv("TRACER_CUPTI_CONTROL_DIR");
    int size = snprintf(state.path, sizeof(state.path), "%s/de_latency_cupti_%d.sock",
                        directory && *directory ? directory : "/tmp", getpid());
    if (size < 0 || size >= static_cast<int>(sizeof(state.path)))
        snprintf(state.path, sizeof(state.path), "/tmp/de_latency_cupti_%d.sock", getpid());
    state.fd = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, state.path, sizeof(address.sun_path) - 1);
    unlink(state.path);
    if (state.fd < 0 || bind(state.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ||
        listen(state.fd, 8)) {
        tracer_log("[control] startup failed: %s", strerror(errno));
        if (state.fd >= 0) close(state.fd);
        unlink(state.path);
        state.path[0] = '\0';
        state.fd = -1;
        return;
    }
    chmod(state.path, 0600);
    state.command = command;
    state.tick = tick;
    state.stop.store(false);
    state.thread = std::thread(run);
}

void control_stop() {
    auto &state = server();
    state.stop.store(true);
    if (state.fd >= 0) shutdown(state.fd, SHUT_RDWR);
    if (state.thread.joinable()) state.thread.join();
    if (state.fd >= 0) close(state.fd);
    state.fd = -1;
    if (state.path[0]) unlink(state.path);
    state.path[0] = '\0';
}
