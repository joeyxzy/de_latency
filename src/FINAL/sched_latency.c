// trace_latency.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "sched_latency.skel.h"

// Minimal libzmq declarations to avoid requiring zmq.h at build time.
#define ZMQ_PUSH 8
extern void *zmq_ctx_new(void);
extern int zmq_ctx_term(void *context);
extern void *zmq_socket(void *context, int type);
extern int zmq_close(void *socket);
extern int zmq_connect(void *socket, const char *endpoint);
extern const char *zmq_strerror(int errnum);
extern int zmq_send(void *socket, const void *buf, size_t len, int flags);

// ZMQ Configuration
#define DEFAULT_ZMQ_ADDR "ipc:///tmp/tracer.sock"
static void *g_zmq_ctx = NULL;
static void *g_zmq_sock = NULL;

static volatile bool exiting = false;

static const char *get_zmq_addr(void)
{
    const char *addr = getenv("TRACER_ZMQ_ADDR");
    return (addr && addr[0]) ? addr : DEFAULT_ZMQ_ADDR;
}

static void sig_handler(int sig)
{
    exiting = true;
}

struct event_data {
    __u64 start_ns;
    __u64 end_ns;
    __u32 tid;
    __u8  type;
};

struct config {
    bool auto_mode;
    pid_t root_pid;
    const char *worker_pattern;
    const char *worker_pid_file;
    int scan_interval_ms;
};

static bool is_numeric_name(const char *s)
{
    if (!s || !*s) return false;
    for (; *s; s++) {
        if (!isdigit((unsigned char)*s)) return false;
    }
    return true;
}

static int parse_u32_str(const char *s, __u32 *out)
{
    char *endptr = NULL;
    unsigned long v;

    if (!s || !*s || !out) return -1;
    v = strtoul(s, &endptr, 10);
    if (*endptr != '\0' || v == 0 || v > UINT32_MAX) return -1;
    *out = (__u32)v;
    return 0;
}

static int parse_int_str(const char *s, int *out)
{
    char *endptr = NULL;
    long v;

    if (!s || !*s || !out) return -1;
    v = strtol(s, &endptr, 10);
    if (*endptr != '\0' || v <= 0 || v > INT32_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int read_first_line(const char *path, char *buf, size_t size)
{
    FILE *fp = NULL;

    if (!path || !buf || size == 0) return -1;
    fp = fopen(path, "r");
    if (!fp) return -1;

    if (!fgets(buf, size, fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    buf[strcspn(buf, "\r\n")] = '\0';
    return 0;
}

static int read_ppid(pid_t pid, pid_t *ppid_out)
{
    char path[64];
    FILE *fp = NULL;
    char line[256];

    if (!ppid_out || pid <= 0) return -1;
    snprintf(path, sizeof(path), "/proc/%d/status", pid);
    fp = fopen(path, "r");
    if (!fp) return -1;

    while (fgets(line, sizeof(line), fp)) {
        int ppid;
        if (sscanf(line, "PPid:\t%d", &ppid) == 1) {
            fclose(fp);
            *ppid_out = (pid_t)ppid;
            return 0;
        }
    }

    fclose(fp);
    return -1;
}

static bool is_descendant_or_self(pid_t pid, pid_t root_pid)
{
    pid_t cur = pid;

    if (root_pid <= 0) return true;
    while (cur > 1) {
        if (cur == root_pid) return true;
        if (read_ppid(cur, &cur) != 0) return false;
    }
    return false;
}

static int read_cmdline(pid_t pid, char *buf, size_t size)
{
    char path[64];
    FILE *fp = NULL;
    size_t n;

    if (!buf || size < 2 || pid <= 0) return -1;

    snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);
    fp = fopen(path, "r");
    if (!fp) return -1;

    n = fread(buf, 1, size - 1, fp);
    fclose(fp);

    if (n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        if (buf[i] == '\0') buf[i] = ' ';
    }
    buf[n] = '\0';
    return 0;
}

static bool process_matches_worker(pid_t pid, const char *pattern)
{
    char path[64];
    char comm[256] = {0};
    char cmdline[4096] = {0};

    if (!pattern || !*pattern) return true;

    snprintf(path, sizeof(path), "/proc/%d/comm", pid);
    if (read_first_line(path, comm, sizeof(comm)) == 0) {
        if (strcasestr(comm, pattern)) return true;
    }
    if (read_cmdline(pid, cmdline, sizeof(cmdline)) == 0) {
        if (strcasestr(cmdline, pattern)) return true;
    }
    return false;
}

static int add_tid_to_bpf_map(int tid_map_fd, __u32 tid, bool verbose)
{
    __u8 val = 1;
    __u8 existing = 0;

    if (bpf_map_lookup_elem(tid_map_fd, &tid, &existing) == 0) {
        return 0; // already tracked
    }

    if (bpf_map_update_elem(tid_map_fd, &tid, &val, BPF_ANY) != 0) {
        fprintf(stderr, "Failed to add TID %u: %s\n", tid, strerror(errno));
        return -1;
    }

    if (verbose) {
        printf("Tracing TID %u\n", tid);
    }
    return 1;
}

static int track_main_tid_of_pid(pid_t pid, int tid_map_fd)
{
    if (pid <= 0 || pid > INT32_MAX) return 0;
    return add_tid_to_bpf_map(tid_map_fd, (__u32)pid, true) > 0 ? 1 : 0;
}

static int auto_scan_worker_pid_file(const struct config *cfg, int tid_map_fd, int *matched_workers)
{
    FILE *fp = NULL;
    char line[128];
    int workers = 0;
    int added_tids = 0;

    if (!cfg->worker_pid_file || !cfg->worker_pid_file[0]) {
        if (matched_workers) *matched_workers = 0;
        return 0;
    }

    fp = fopen(cfg->worker_pid_file, "r");
    if (!fp) {
        if (errno != ENOENT) {
            fprintf(stderr, "Failed to open worker pid file %s: %s\n",
                    cfg->worker_pid_file, strerror(errno));
        }
        if (matched_workers) *matched_workers = 0;
        return 0;
    }

    while (fgets(line, sizeof(line), fp)) {
        __u32 pid_u32;
        pid_t pid;

        line[strcspn(line, "\r\n")] = '\0';
        if (parse_u32_str(line, &pid_u32) != 0) continue;

        pid = (pid_t)pid_u32;
        if (cfg->root_pid > 0 && !is_descendant_or_self(pid, cfg->root_pid)) continue;

        workers++;
        added_tids += track_main_tid_of_pid(pid, tid_map_fd);
    }

    fclose(fp);
    if (matched_workers) *matched_workers = workers;
    return added_tids;
}

static int auto_scan_workers(const struct config *cfg, int tid_map_fd, int *matched_workers)
{
    if (cfg->worker_pid_file && cfg->worker_pid_file[0]) {
        return auto_scan_worker_pid_file(cfg, tid_map_fd, matched_workers);
    }

    DIR *proc_dir = NULL;
    struct dirent *ent;
    int added_tids = 0;
    int workers = 0;

    proc_dir = opendir("/proc");
    if (!proc_dir) {
        perror("opendir(/proc)");
        if (matched_workers) *matched_workers = 0;
        return 0;
    }

    while ((ent = readdir(proc_dir)) != NULL) {
        pid_t pid;

        if (!is_numeric_name(ent->d_name)) continue;
        pid = (pid_t)atoi(ent->d_name);
        if (pid <= 0) continue;

        if (!is_descendant_or_self(pid, cfg->root_pid)) continue;
        if (!process_matches_worker(pid, cfg->worker_pattern)) continue;

        workers++;
        added_tids += track_main_tid_of_pid(pid, tid_map_fd);
    }

    closedir(proc_dir);

    if (matched_workers) *matched_workers = workers;
    return added_tids;
}

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <TID1> [TID2] ...\n", prog);
    fprintf(stderr, "  %s --auto [--root-pid <pid>] [--worker-pattern <pattern>] [--worker-pid-file <path>] [--scan-interval-ms <ms>]\n", prog);
    fprintf(stderr, "  %s   (no args defaults to --auto)\n", prog);
}

// Initialize ZMQ PUSH socket
static int init_zmq() {
    g_zmq_ctx = zmq_ctx_new();
    if (!g_zmq_ctx) {
        perror("zmq_ctx_new");
        return -1;
    }
    
    // Using PUSH socket to connect to Python's PULL
    g_zmq_sock = zmq_socket(g_zmq_ctx, ZMQ_PUSH);
    if (!g_zmq_sock) {
        perror("zmq_socket");
        return -1;
    }

    // Since Python binds, we connect
    if (zmq_connect(g_zmq_sock, get_zmq_addr()) != 0) {
        fprintf(stderr, "zmq_connect failed: %s\n", zmq_strerror(errno));
        return -1;
    }
    
    printf("ZMQ connected to %s\n", get_zmq_addr());
    return 0;
}

static void cleanup_zmq() {
    if (g_zmq_sock) zmq_close(g_zmq_sock);
    if (g_zmq_ctx) zmq_ctx_term(g_zmq_ctx);
}

// Send event via ZMQ
static int handle_event(void *ctx, void *data, size_t data_sz)
{
    const struct event_data *e = data;
    char msg[512];
    const char *reason;
    int len;

    if (data_sz != sizeof(*e)) {
        fprintf(stderr, "Invalid event size\n");
        return 1;
    }

    reason = (e->type == 0) ? "Wakeup" : "Preempt";
    len = snprintf(
        msg,
        sizeof(msg),
        "{\"source\":\"ebpf\",\"event_type\":\"sched_latency\",\"timestamp\":%llu,"
        "\"payload\":{\"tid\":%u,\"start_ns\":%llu,\"end_ns\":%llu,\"dur_us\":%llu,\"reason\":\"%s\"}}",
        (unsigned long long)e->end_ns,
        (unsigned int)e->tid,
        (unsigned long long)e->start_ns,
        (unsigned long long)e->end_ns,
        (unsigned long long)(e->end_ns - e->start_ns),
        reason
    );
    if (len < 0) {
        return 1;
    }
    if ((size_t)len >= sizeof(msg)) {
        len = (int)(sizeof(msg) - 1);
    }
    zmq_send(g_zmq_sock, msg, (size_t)len, 0);

    return 0;
}

int main(int argc, char **argv)
{
    struct sched_latency_bpf *skel = NULL;
    struct ring_buffer *rb = NULL;
    int i, matched_workers = 0;
    int poll_ms = 100;
    int scan_tick = 0;
    __u32 manual_tids[1024];
    int manual_tid_count = 0;
    struct config cfg = {
        .auto_mode = (argc == 1),
        .root_pid = 0,
        .worker_pattern = "worker",
        .worker_pid_file = NULL,
        .scan_interval_ms = 500,
    };

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--auto") == 0) {
            cfg.auto_mode = true;
        } else if (strcmp(argv[i], "--root-pid") == 0) {
            int root_pid;
            if (++i >= argc || parse_int_str(argv[i], &root_pid) != 0) {
                fprintf(stderr, "Invalid --root-pid value\n");
                print_usage(argv[0]);
                return 1;
            }
            cfg.root_pid = (pid_t)root_pid;
            cfg.auto_mode = true;
        } else if (strcmp(argv[i], "--worker-pattern") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Invalid --worker-pattern value\n");
                print_usage(argv[0]);
                return 1;
            }
            cfg.worker_pattern = argv[i];
            cfg.auto_mode = true;
        } else if (strcmp(argv[i], "--worker-pid-file") == 0) {
            if (++i >= argc || !argv[i][0]) {
                fprintf(stderr, "Invalid --worker-pid-file value\n");
                print_usage(argv[0]);
                return 1;
            }
            cfg.worker_pid_file = argv[i];
            cfg.auto_mode = true;
        } else if (strcmp(argv[i], "--scan-interval-ms") == 0) {
            int interval_ms;
            if (++i >= argc || parse_int_str(argv[i], &interval_ms) != 0) {
                fprintf(stderr, "Invalid --scan-interval-ms value\n");
                print_usage(argv[0]);
                return 1;
            }
            cfg.scan_interval_ms = interval_ms;
            cfg.auto_mode = true;
        } else {
            __u32 tid;
            if (parse_u32_str(argv[i], &tid) != 0) {
                fprintf(stderr, "Invalid TID: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
            if (manual_tid_count >= (int)(sizeof(manual_tids) / sizeof(manual_tids[0]))) {
                fprintf(stderr, "Too many manual TIDs (max=%zu)\n",
                        sizeof(manual_tids) / sizeof(manual_tids[0]));
                return 1;
            }
            manual_tids[manual_tid_count++] = tid;
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Init ZMQ
    if (init_zmq() != 0) {
        return 1;
    }

    struct rlimit rl = { .rlim_cur = RLIM_INFINITY, .rlim_max = RLIM_INFINITY };
    setrlimit(RLIMIT_MEMLOCK, &rl);

    skel = sched_latency_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open and load BPF skeleton\n");
        cleanup_zmq();
        return 1;
    }

    int tid_map_fd = bpf_map__fd(skel->maps.target_tid_map);
    if (manual_tid_count > 0) {
        cfg.auto_mode = false;
        for (i = 0; i < manual_tid_count; i++) {
            if (add_tid_to_bpf_map(tid_map_fd, manual_tids[i], true) < 0) {
                goto cleanup;
            }
        }
    } else {
        int added = auto_scan_workers(&cfg, tid_map_fd, &matched_workers);
        printf("Auto mode enabled: root_pid=%d, scan_interval=%dms\n",
               cfg.root_pid, cfg.scan_interval_ms);
        if (cfg.worker_pid_file && cfg.worker_pid_file[0]) {
            printf("Worker source: pid file '%s'\n", cfg.worker_pid_file);
        } else {
            printf("Worker source: name pattern '%s'\n", cfg.worker_pattern);
        }
        if (added == 0) {
            printf("No worker TIDs found yet, waiting for worker process...\n");
        }
    }

    if (sched_latency_bpf__attach(skel) != 0) {
        fprintf(stderr, "Failed to attach BPF skeleton\n");
        goto cleanup;
    }

    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "Failed to create ring buffer\n");
        goto cleanup;
    }

    printf("\n--- eBPF Tracer Running (Sending to %s) ---\n", get_zmq_addr());

    while (!exiting) {
        int ret = ring_buffer__poll(rb, poll_ms);
        if (ret < 0 && ret != -EINTR) {
            fprintf(stderr, "ring_buffer__poll failed: %d\n", ret);
            break;
        }

        if (cfg.auto_mode) {
            int every_n_polls = cfg.scan_interval_ms / poll_ms;
            if (every_n_polls < 1) every_n_polls = 1;
            scan_tick++;
            if (scan_tick >= every_n_polls) {
                int newly_added = auto_scan_workers(&cfg, tid_map_fd, &matched_workers);
                if (newly_added > 0) {
                    printf("Auto-discovered %d new worker main TID(s), worker process count=%d\n",
                           newly_added, matched_workers);
                }
                scan_tick = 0;
            }
        }
    }

cleanup:
    ring_buffer__free(rb);
    sched_latency_bpf__destroy(skel);
    cleanup_zmq();
    printf("\nExiting...\n");
    return 0;
}
