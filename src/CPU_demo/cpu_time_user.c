// cpu_time_user.c
// gcc -O2 -g cpu_time_user.c -o cpu_time_user -lbpf -lelf -lz -pthread

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>

#include "bpf/libbpf.h"
#include "bpf/bpf.h"

static volatile sig_atomic_t exiting = 0;
static void sigint(int sig) { exiting = 1; }

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint);
    signal(SIGTERM, sigint);

    // 1) open & load bpf object
    struct bpf_object *obj;
    int prog_fd;
    int err;

    obj = bpf_object__open_file("cpu_time.bpf.o", NULL);
    if (!obj) {
        fprintf(stderr, "failed to open BPF object\n");
        return 1;
    }
    err = bpf_object__load(obj);
    if (err) {
        fprintf(stderr, "failed to load BPF object: %d\n", err);
        return 1;
    }

    // find map fds
    int map_targets_fd = bpf_object__find_map_fd_by_name(obj, "targets");
    int map_total_fd   = bpf_object__find_map_fd_by_name(obj, "total_ns");
    if (map_targets_fd < 0 || map_total_fd < 0) {
        fprintf(stderr, "failed to find maps in object\n");
        return 1;
    }

    // attach tracepoint prog (libbpf auto-attaches using section name)
    // ensure the tracepoint program is attached by iterating programs and attaching
    struct bpf_program *prog;
    bpf_object__for_each_program(prog, obj) {
        const char *sec = bpf_program__section_name(prog);
        // only attach tracepoint programs (our section is tracepoint/sched/sched_switch)
        if (sec && strstr(sec, "tracepoint/sched/sched_switch")) {
            struct bpf_link *link = bpf_program__attach(prog);
            if (!link) {
                fprintf(stderr, "failed to attach program %s\n", sec);
                // continue trying other progs
            } else {
                // keep the link; libbpf will free on object close
            }
        }
    }

    // 2) fork & exec the target program
    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 1;
    }
    if (child == 0) {
        // child: execute the requested program
        execvp(argv[1], &argv[1]);
        perror("execvp");
        _exit(127);
    }

    // parent: we have child's pid; insert into targets map
    __u32 key = (uint32_t)child;
    __u8 val = 1;
    if (bpf_map_update_elem(map_targets_fd, &key, &val, BPF_ANY) != 0) {
        fprintf(stderr, "failed to add pid %u to targets map: %s\n", key, strerror(errno));
        // still continue to wait and then cleanup
    } else {
        printf("Tracking pid %u\n", key);
    }

    // wait for child to exit
    int status;
    while (!exiting) {
        pid_t w = waitpid(child, &status, 0);
        if (w == -1) {
            if (errno == EINTR) continue;
            perror("waitpid");
            break;
        }
        if (w == child) break;
    }

    // child exited; read total running ns from map
    __u64 total_ns = 0;
    if (bpf_map_lookup_elem(map_total_fd, &key, &total_ns) != 0) {
        // if there is no entry, 0 is OK
        total_ns = 0;
    }

    double total_ms = (double)total_ns / 1e6;
    printf("PID %u total CPU running time: %.3f ms (total_ns=%llu)\n", key, total_ms, (unsigned long long)total_ns);

    // clean up: remove target entry (best-effort)
    bpf_map_delete_elem(map_targets_fd, &key);

    bpf_object__close(obj);
    return 0;
}
