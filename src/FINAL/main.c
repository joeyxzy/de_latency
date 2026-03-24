#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <zmq.h>

static volatile sig_atomic_t exiting = 0;
static pid_t child_pid = 0;
static pid_t collector_pid = 0;
static pid_t ebpf_pid = 0;

#define COLLECTOR_PY "/home/joeyxzy/miniconda3/envs/vllm/bin/python"
#define COLLECTOR_SCRIPT "../src/FINAL/collector.py"
#define ZMQ_ADDR "ipc:///tmp/tracer.sock"
#define EBPF_MONITOR_BIN "./ebpf_monitor"
#define WORKER_PID_FILE "/tmp/tracer_worker_pids"

// --- Helper: wait for collector socket ---
static int wait_for_path(const char *path, int timeout_ms) {
    int waited = 0;
    const int step_ms = 50;
    struct stat st;
    while (waited < timeout_ms) {
        if (stat(path, &st) == 0) return 0;
        usleep(step_ms * 1000);
        waited += step_ms;
    }
    return -1;
}

// --- Start collector process ---
static int start_collector_process(void) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork collector");
        return 0;
    }
    if (pid == 0) {
        setenv("TRACER_ZMQ_ADDR", ZMQ_ADDR, 1);
        execl(COLLECTOR_PY, COLLECTOR_PY, COLLECTOR_SCRIPT, "--bind", "ipc:///tmp/tracer.sock", (char *)NULL);
        perror("execl collector");
        _exit(127);
    }
    if (wait_for_path("/tmp/tracer.sock", 2000) != 0) {
        fprintf(stderr, "warning: collector socket not ready after timeout\n");
        kill(pid, SIGTERM);
        return 0;
    }
    return pid;
}

// --- Start eBPF monitor in auto-discovery mode ---
static pid_t start_ebpf_monitor_process(pid_t root_pid) {
    pid_t pid;
    char root_pid_arg[32];
    int status;

    if (root_pid <= 0) return 0;

    pid = fork();
    if (pid < 0) {
        perror("fork ebpf_monitor");
        return 0;
    }

    if (pid == 0) {
        snprintf(root_pid_arg, sizeof(root_pid_arg), "%d", (int)root_pid);
        execl(EBPF_MONITOR_BIN,
              EBPF_MONITOR_BIN,
              "--auto",
              "--root-pid",
              root_pid_arg,
              "--worker-pid-file",
              WORKER_PID_FILE,
              (char *)NULL);
        perror("execl ebpf_monitor");
        _exit(127);
    }

    // If the monitor exits immediately, treat it as startup failure but keep controller running.
    usleep(200 * 1000);
    if (waitpid(pid, &status, WNOHANG) == pid) {
        fprintf(stderr, "warning: ebpf_monitor exited during startup (status=%d)\n", status);
        return 0;
    }

    return pid;
}

// --- ZMQ cleanup ---
void shutdown_zmq_sender(void *ctx, void *sock) {
    if (sock) zmq_close(sock);
    if (ctx) zmq_ctx_term(ctx);
}

// --- Signal handler ---
static void sigint_handler(int sig) {
    exiting = 1;
    if (child_pid > 0) {
        kill(child_pid, SIGTERM);
    }
    if (ebpf_pid > 0) {
        kill(ebpf_pid, SIGTERM);
    }
}

// --- Main Application ---
int main(int argc, char **argv) {
    void *zmq_ctx = NULL;
    void *zmq_push = NULL;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // 1. Start collector
    collector_pid = start_collector_process();
    if (collector_pid == 0) {
        fprintf(stderr, "Failed to start collector process\n");
        goto cleanup;
    }

    // 2. Initialize ZMQ sender (for libtracer.so to use)
    zmq_ctx = zmq_ctx_new();
    if (!zmq_ctx) {
        perror("zmq_ctx_new");
        goto cleanup;
    }
    zmq_push = zmq_socket(zmq_ctx, ZMQ_PUSH);
    if (!zmq_push) {
        perror("zmq_socket");
        goto cleanup;
    }
    int linger = 0;
    zmq_setsockopt(zmq_push, ZMQ_LINGER, &linger, sizeof(linger));
    int rc = zmq_connect(zmq_push, ZMQ_ADDR);
    if (rc != 0) {
        fprintf(stderr, "zmq_connect failed: %s\n", zmq_strerror(zmq_errno()));
        // Not fatal — libtracer will handle it
    }

    // 3. Fork and execute target application
    child_pid = fork();
    if (child_pid < 0) {
        perror("fork");
        goto cleanup;
    }

    if (child_pid == 0) {
        // Child: inject libtracer.so and exec
        setenv("LD_PRELOAD", "./libtracer.so", 1);
        setenv("TRACER_ZMQ_ADDR", ZMQ_ADDR, 1);
        execvp(argv[1], &argv[1]);
        perror("execvp");
        _exit(127);
    }

    printf("Launched target process PID: %d\n", child_pid);
    printf("Collector PID: %d\n", collector_pid);

    // 4. Start standalone eBPF monitor automatically
    ebpf_pid = start_ebpf_monitor_process(child_pid);
    if (ebpf_pid > 0) {
        printf("eBPF monitor PID: %d (auto mode)\n", ebpf_pid);
    } else {
        fprintf(stderr, "warning: eBPF monitor failed to start; continue without scheduler tracing\n");
    }

    printf("Waiting for target to finish...\n");

    // 5. Wait for child to exit
    int status;
    while (!exiting) {
        if (waitpid(child_pid, &status, WNOHANG) == child_pid) {
            break;
        }
        usleep(100000); // 100ms
    }

cleanup:
    // Cleanup ZMQ
    shutdown_zmq_sender(zmq_ctx, zmq_push);

    // Ensure child is cleaned up
    if (child_pid > 0) {
        kill(child_pid, SIGTERM);
        waitpid(child_pid, NULL, 0);
    }
    if (collector_pid > 0) {
        kill(collector_pid, SIGTERM);
        waitpid(collector_pid, NULL, 0);
    }
    if (ebpf_pid > 0) {
        kill(ebpf_pid, SIGTERM);
        waitpid(ebpf_pid, NULL, 0);
    }

    printf("Controller exiting.\n");
    return 0;
}
