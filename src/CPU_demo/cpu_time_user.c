// cpu_time_user.c (Final Corrected Version without rodata & with correct TGID map update)
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "cpu_time.skel.h"

static volatile sig_atomic_t exiting = 0;
static void sigint(int sig) { exiting = 1; }

// Print map values safely
static void print_all_thread_times(int map_fd) {
    __u32 key, next_key;
    __u64 time_ns;

    printf("\n--- CPU Time Per Thread ---\n");
    printf("%-10s %-20s\n", "TID", "CPU Time (ms)");

    if (bpf_map_get_next_key(map_fd, NULL, &key) != 0) {
        printf("Map is empty. No data collected.\n");
        printf("---------------------------\n");
        return;
    }

    while (1) {
        if (bpf_map_lookup_elem(map_fd, &key, &time_ns) == 0) {
            printf("%-10u %-20.3f\n", key, time_ns / 1000000.0);
        }
        if (bpf_map_get_next_key(map_fd, &key, &next_key) != 0) {
            break;
        }
        key = next_key;
    }

    printf("---------------------------\n");
}

int main(int argc, char **argv) {
    struct cpu_time_bpf *skel;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint);
    signal(SIGTERM, sigint);

    skel = cpu_time_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    if (cpu_time_bpf__load(skel)) {
        fprintf(stderr, "Failed to load and verify BPF skeleton\n");
        cpu_time_bpf__destroy(skel);
        return 1;
    }

    pid_t child_pid = fork();
    if (child_pid < 0) {
        perror("fork");
        cpu_time_bpf__destroy(skel);
        return 1;
    }
    if (child_pid == 0) {
        execvp(argv[1], &argv[1]);
        perror("execvp");
        _exit(127);
    }

    printf("Tracking process group with TGID: %d\n", child_pid);

    // ✅ Correctly set TGID via map
    int tgid_map_fd = bpf_map__fd(skel->maps.target_tgid_map);
    if (tgid_map_fd < 0) {
        fprintf(stderr, "Failed to get target_tgid_map FD\n");
    } else {
        __u32 key = 0;
        __u32 val = (uint32_t) child_pid;
        if (bpf_map_update_elem(tgid_map_fd, &key, &val, BPF_ANY) != 0) {
            perror("bpf_map_update_elem(target_tgid_map)");
        }
    }

    if (cpu_time_bpf__attach(skel) != 0) {
        fprintf(stderr, "Failed to attach BPF skeleton\n");
        cpu_time_bpf__destroy(skel);
        return 1;
    }

    int status;
    while (!exiting) {
        pid_t w = waitpid(child_pid, &status, 0);
        if (w == -1) {
            if (errno == EINTR) continue;
            perror("waitpid");
            break;
        }
        if (w == child_pid) break;
    }

    printf("Target program finished. Reading data from BPF map...\n");

    int map_total_fd = bpf_map__fd(skel->maps.total_ns);
    if (map_total_fd < 0) {
        fprintf(stderr, "Failed to get map FD\n");
    } else {
        print_all_thread_times(map_total_fd);
    }

    cpu_time_bpf__destroy(skel);
    return 0;
}
