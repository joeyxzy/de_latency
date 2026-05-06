#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <zmq.h>

static volatile sig_atomic_t exiting = 0;
static pid_t child_pid = 0;
static pid_t collector_pid = 0;
static pid_t ebpf_pid = 0;

#define DEFAULT_COLLECTOR_PY "python3"
#define DEFAULT_COLLECTOR_SCRIPT "../src/FINAL/collector.py"
#define DEFAULT_ZMQ_ADDR "ipc:///tmp/tracer.sock"
#define DEFAULT_EBPF_MONITOR_BIN "ebpf_monitor"
#define DEFAULT_LIBTRACER_SO "libtracer.so"
#define DEFAULT_WORKER_PID_FILE "/tmp/tracer_worker_pids"
#define DEFAULT_CONTROL_STATE_FILE "/tmp/de_latency_control_state"

static const char *env_or_default(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return (value && value[0]) ? value : fallback;
}

static int env_enabled(const char *name, int default_value) {
    const char *value = getenv(name);

    if (!value || !value[0]) {
        return default_value;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "off") == 0 ||
        strcasecmp(value, "no") == 0) {
        return 0;
    }
    if (strcmp(value, "1") == 0 ||
        strcasecmp(value, "true") == 0 ||
        strcasecmp(value, "on") == 0 ||
        strcasecmp(value, "yes") == 0) {
        return 1;
    }
    return default_value;
}

static const char *pick_collector_python(const char *target_program) {
    const char *configured = getenv("DE_LATENCY_COLLECTOR_PY");
    const char *base = NULL;

    if (configured && configured[0]) {
        return configured;
    }
    if (!target_program || !target_program[0]) {
        return DEFAULT_COLLECTOR_PY;
    }

    base = strrchr(target_program, '/');
    base = base ? base + 1 : target_program;
    if (strstr(base, "python") != NULL) {
        return target_program;
    }
    return DEFAULT_COLLECTOR_PY;
}

static int resolve_self_dir(char *buf, size_t size) {
    ssize_t n = readlink("/proc/self/exe", buf, size - 1);
    char *slash = NULL;

    if (n > 0 && (size_t)n < size) {
        buf[n] = '\0';
        slash = strrchr(buf, '/');
        if (slash) {
            *slash = '\0';
            return 0;
        }
    }

    if (getcwd(buf, size) != NULL) {
        return 0;
    }
    return -1;
}

static int join_path(char *buf, size_t size, const char *dir, const char *name) {
    int written = snprintf(buf, size, "%s/%s", dir, name);
    return (written < 0 || (size_t)written >= size) ? -1 : 0;
}

static int build_path_from_env_or_default(
    char *buf,
    size_t size,
    const char *env_name,
    const char *base_dir,
    const char *default_rel_path) {
    const char *configured = getenv(env_name);
    int written;

    if (configured && configured[0]) {
        written = snprintf(buf, size, "%s", configured);
        return (written < 0 || (size_t)written >= size) ? -1 : 0;
    }
    return join_path(buf, size, base_dir, default_rel_path);
}

static const char *ipc_path_from_addr(const char *addr) {
    if (!addr) {
        return NULL;
    }
    return strncmp(addr, "ipc://", 6) == 0 ? addr + 6 : NULL;
}

static int write_control_state_file(
    const char *path,
    pid_t controller_pid,
    pid_t target_pid,
    pid_t collector_pid_value,
    pid_t ebpf_pid_value,
    const char *zmq_addr,
    const char *worker_pid_file,
    const char *ebpf_monitor_bin) {
    char tmp_path[PATH_MAX];
    FILE *fp = NULL;

    if (!path || !path[0]) {
        return -1;
    }

    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path) < 0 ||
        strlen(tmp_path) >= sizeof(tmp_path)) {
        return -1;
    }

    fp = fopen(tmp_path, "w");
    if (!fp) {
        return -1;
    }

    fprintf(fp, "state_version=1\n");
    fprintf(fp, "controller_pid=%d\n", (int)controller_pid);
    fprintf(fp, "target_pid=%d\n", (int)target_pid);
    fprintf(fp, "collector_pid=%d\n", (int)collector_pid_value);
    fprintf(fp, "ebpf_pid=%d\n", (int)ebpf_pid_value);
    fprintf(fp, "zmq_addr=%s\n", zmq_addr ? zmq_addr : "");
    fprintf(fp, "worker_pid_file=%s\n", worker_pid_file ? worker_pid_file : "");
    fprintf(fp, "ebpf_monitor_bin=%s\n", ebpf_monitor_bin ? ebpf_monitor_bin : "");

    if (fclose(fp) != 0) {
        unlink(tmp_path);
        return -1;
    }

    if (rename(tmp_path, path) != 0) {
        unlink(tmp_path);
        return -1;
    }
    return 0;
}

static void remove_control_state_file(const char *path) {
    if (path && path[0]) {
        unlink(path);
    }
}

static pid_t read_ebpf_pid_from_control_state(const char *path) {
    FILE *fp = NULL;
    char line[256];

    if (!path || !path[0]) {
        return 0;
    }

    fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }

    while (fgets(line, sizeof(line), fp)) {
        int value = 0;
        if (sscanf(line, "ebpf_pid=%d", &value) == 1 && value > 0) {
            fclose(fp);
            return (pid_t)value;
        }
    }

    fclose(fp);
    return 0;
}

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
static int start_collector_process(
    const char *collector_python,
    const char *collector_script,
    const char *zmq_addr) {
    const char *socket_path = ipc_path_from_addr(zmq_addr);
    pid_t pid;

    if (socket_path && unlink(socket_path) != 0 && errno != ENOENT) {
        fprintf(stderr, "warning: failed to remove stale collector socket %s: %s\n",
                socket_path, strerror(errno));
    }

    pid = fork();
    if (pid < 0) {
        perror("fork collector");
        return 0;
    }
    if (pid == 0) {
        setenv("TRACER_ZMQ_ADDR", zmq_addr, 1);
        setenv("DE_LATENCY_DISABLE_SITECUSTOMIZE", "1", 1);
        execlp(collector_python, collector_python, collector_script, (char *)NULL);
        perror("execl collector");
        _exit(127);
    }
    if (socket_path && wait_for_path(socket_path, 2000) != 0) {
        fprintf(stderr, "warning: collector socket not ready after timeout\n");
        kill(pid, SIGTERM);
        return 0;
    }
    return pid;
}

// --- Start eBPF monitor in auto-discovery mode ---
static pid_t start_ebpf_monitor_process(
    const char *ebpf_monitor_bin,
    pid_t root_pid,
    const char *worker_pid_file) {
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
        execl(ebpf_monitor_bin,
              ebpf_monitor_bin,
              "--auto",
              "--root-pid",
              root_pid_arg,
              "--worker-pid-file",
              worker_pid_file,
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
        kill(-child_pid, SIGTERM);
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
    char controller_dir[PATH_MAX];
    char collector_script[PATH_MAX];
    char ebpf_monitor_bin[PATH_MAX];
    char libtracer_path[PATH_MAX];
    const char *collector_python;
    const char *zmq_addr;
    const char *worker_pid_file;
    const char *control_state_file;
    int ebpf_start_enabled;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        return 1;
    }

    collector_python = pick_collector_python(argv[1]);
    zmq_addr = env_or_default("TRACER_ZMQ_ADDR", DEFAULT_ZMQ_ADDR);
    worker_pid_file = env_or_default("TRACER_WORKER_PID_FILE", DEFAULT_WORKER_PID_FILE);
    control_state_file = env_or_default("DE_LATENCY_CONTROL_STATE_FILE", DEFAULT_CONTROL_STATE_FILE);
    ebpf_start_enabled = env_enabled("TRACER_EBPF_START_ENABLED", 1);

    if (resolve_self_dir(controller_dir, sizeof(controller_dir)) != 0) {
        perror("resolve_self_dir");
        return 1;
    }
    if (build_path_from_env_or_default(
            collector_script,
            sizeof(collector_script),
            "DE_LATENCY_COLLECTOR_SCRIPT",
            controller_dir,
            DEFAULT_COLLECTOR_SCRIPT) != 0 ||
        build_path_from_env_or_default(
            ebpf_monitor_bin,
            sizeof(ebpf_monitor_bin),
            "DE_LATENCY_EBPF_MONITOR_BIN",
            controller_dir,
            DEFAULT_EBPF_MONITOR_BIN) != 0 ||
        build_path_from_env_or_default(
            libtracer_path,
            sizeof(libtracer_path),
            "DE_LATENCY_LIBTRACER_SO",
            controller_dir,
            DEFAULT_LIBTRACER_SO) != 0) {
        fprintf(stderr, "Failed to resolve runtime helper paths\n");
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // 1. Start collector
    collector_pid = start_collector_process(collector_python, collector_script, zmq_addr);
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
    int rc = zmq_connect(zmq_push, zmq_addr);
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
        if (setpgid(0, 0) != 0) {
            perror("setpgid child");
        }
        setenv("LD_PRELOAD", libtracer_path, 1);
        setenv("TRACER_ZMQ_ADDR", zmq_addr, 1);
        execvp(argv[1], &argv[1]);
        perror("execvp");
        _exit(127);
    }
    if (setpgid(child_pid, child_pid) != 0 && errno != EACCES) {
        perror("setpgid parent");
    }

    printf("Launched target process PID: %d\n", child_pid);
    printf("Collector PID: %d\n", collector_pid);

    // 4. Optionally start standalone eBPF monitor automatically
    if (ebpf_start_enabled) {
        ebpf_pid = start_ebpf_monitor_process(ebpf_monitor_bin, child_pid, worker_pid_file);
        if (ebpf_pid > 0) {
            printf("eBPF monitor PID: %d (auto mode)\n", ebpf_pid);
        } else {
            fprintf(stderr, "warning: eBPF monitor failed to start; continue without scheduler tracing\n");
        }
    } else {
        ebpf_pid = 0;
        printf("eBPF monitor auto-start disabled by TRACER_EBPF_START_ENABLED=%s\n",
               getenv("TRACER_EBPF_START_ENABLED"));
    }
    if (write_control_state_file(
            control_state_file,
            getpid(),
            child_pid,
            collector_pid,
            ebpf_pid,
            zmq_addr,
            worker_pid_file,
            ebpf_monitor_bin) != 0) {
        fprintf(stderr, "warning: failed to write control state file: %s\n", control_state_file);
    } else {
        printf("Control state file: %s\n", control_state_file);
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
        kill(-child_pid, SIGTERM);
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
    {
        pid_t current_ebpf_pid = read_ebpf_pid_from_control_state(control_state_file);
        if (current_ebpf_pid > 0 && current_ebpf_pid != ebpf_pid) {
            kill(current_ebpf_pid, SIGTERM);
            waitpid(current_ebpf_pid, NULL, 0);
        }
    }
    remove_control_state_file(control_state_file);

    printf("Controller exiting.\n");
    return 0;
}
