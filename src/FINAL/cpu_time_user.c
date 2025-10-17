#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "cpu_time.skel.h"
#include "tracer_comm.h" // You must have this shared header file

// --- Global State ---
static volatile sig_atomic_t exiting = 0;
static pid_t child_pid = 0;

// --- Signal Handler ---
static void sigint_handler(int sig) {
    exiting = 1;
    // If the child is still running, send it a signal to terminate
    if (child_pid > 0) {
        kill(child_pid, SIGTERM);
    }
}

// --- eBPF Data Printing ---
static void print_all_thread_times(int map_fd) {
    __u32 key, next_key;
    __u64 time_ns;

    printf("\n--- CPU Time Per Thread (from eBPF) ---\n");
    printf("%-10s %-20s\n", "TID", "CPU Time (ms)");

    if (bpf_map_get_next_key(map_fd, NULL, &key) != 0) {
        printf("eBPF map is empty. No CPU data collected.\n");
        printf("---------------------------------------\n");
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
    printf("---------------------------------------\n");
}

// --- CUPTI Data Processing and Printing ---
void process_cupti_record(UnifiedTraceRecord *rec) {
    // Note: Timestamps are raw from CUPTI and are not aligned to the system clock.
    switch (rec->type) {
        case RECORD_TYPE_KERNEL:
            printf("KERNEL \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
                   rec->name, rec->start_ns, rec->end_ns, rec->deviceId, rec->contextId, rec->streamId, rec->correlationId);
            printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
                   rec->gridX, rec->gridY, rec->gridZ, rec->blockX, rec->blockY, rec->blockZ,
                   rec->staticSharedMemory, rec->dynamicSharedMemory);
            break;
        case RECORD_TYPE_MEMCPY:
            printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
                   rec->name, rec->start_ns, rec->end_ns, rec->deviceId, rec->contextId, rec->streamId,
                   rec->correlationId, rec->memcpy_runtimeCorrelationId);
            break;
        case RECORD_TYPE_MEMSET:
            printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
                   rec->memset_value, rec->start_ns, rec->end_ns, rec->deviceId, rec->contextId, rec->streamId,
                   rec->correlationId);
            break;
        case RECORD_TYPE_DRIVER:
            printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                   rec->cbid, rec->start_ns, rec->end_ns, rec->pid, rec->tid, rec->correlationId);
            break;
        case RECORD_TYPE_RUNTIME:
            printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                   rec->cbid, rec->start_ns, rec->end_ns, rec->pid, rec->tid, rec->correlationId);
            break;
        case RECORD_TYPE_METADATA_TID_MAP:
            // This is metadata. We can print it for debugging or ignore it for cleaner output.
            // printf("[INFO] Tracer attached to PID %u, TID %u\n", rec->pid, rec->tid);
            break;
        default: 
            break;
    }
}

// --- Main Application ---
int main(int argc, char **argv) {
    struct cpu_time_bpf *skel = NULL;
    int comm_pipe[2] = {-1, -1};

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // 1. Initialize BPF
    skel = cpu_time_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }
    if (cpu_time_bpf__load(skel)) {
        fprintf(stderr, "Failed to load BPF skeleton\n");
        goto cleanup;
    }

    // 2. Create communication pipe
    if (pipe(comm_pipe) == -1) {
        perror("pipe");
        goto cleanup;
    }

    // 3. Fork and execute the target application
    child_pid = fork();
    if (child_pid < 0) {
        perror("fork");
        goto cleanup;
    }

    if (child_pid == 0) {
        // Child process: setup environment and exec
        close(comm_pipe[0]); // Child only writes, so close read end

        char pipe_fd_str[16];
        snprintf(pipe_fd_str, sizeof(pipe_fd_str), "%d", comm_pipe[1]);
        setenv("CUPTI_COMM_FD", pipe_fd_str, 1);
        setenv("LD_PRELOAD", "./libtracer.so", 1); // Ensure libtracer.so is in the current directory

        printf("exec!!!:%s\n",argv[1]);
        execvp(argv[1], &argv[1]);

        // If execvp returns, it must have failed
        perror("execvp");
        close(comm_pipe[1]);
        _exit(127);
    }

    // Parent process (Controller)
    close(comm_pipe[1]); // Parent only reads, so close write end
    printf("Tracking process group with TGID: %d\n", child_pid);

    // Set the target TGID in the BPF map
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

    // Attach BPF probes
    if (cpu_time_bpf__attach(skel) != 0) {
        fprintf(stderr, "Failed to attach BPF skeleton\n");
        goto cleanup;
    }

    // 4. Main event loop: read from the pipe
    printf("--- Real-time CUPTI Trace (timestamps are relative to tracer start) ---\n");

    // 4. =========== 主事件循环 (重写逻辑) ===========
    int cupti_fd = comm_pipe[0];
    int pipe_closed = 0;
    while (!pipe_closed) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(cupti_fd, &read_fds);

        // 我们不再需要频繁地 waitpid，让 select 来处理等待
        // 设置一个合理的超时，比如1秒，或者干脆阻塞等待
        struct timeval timeout = {1, 0}; // 1 second timeout

        int ret = select(cupti_fd + 1, &read_fds, NULL, NULL, &timeout);

        if (ret < 0) {
            // select 出错
            if (errno == EINTR) continue; // 被信号中断，继续循环
            perror("select");
            break;
        }

        if (ret == 0) {
            // 超时，检查一下孩子是否还在运行
            int status;
            if (waitpid(child_pid, &status, WNOHANG) == child_pid) {
                // 子进程已经退出了，但管道可能还有数据，我们继续循环直到管道关闭
            }
            continue; // 继续下一次循环
        }

        if (FD_ISSET(cupti_fd, &read_fds)) {
            UnifiedTraceRecord rec;
            // 一次性读取所有可用数据，避免 select 多次唤醒
            while (1) {
                ssize_t bytes_read = read(cupti_fd, &rec, sizeof(rec));
                if (bytes_read == sizeof(rec)) {
                    if (rec.type == RECORD_TYPE_METADATA_FLUSH_COMPLETE) {
                        // 这是一个可选的提前退出信号
                        // pipe_closed = 1; 
                        // break;
                    } else {
                        process_cupti_record(&rec);
                    }
                } else if (bytes_read == 0) {
                    // 这是最重要的信号：管道的写端被关闭了 (子进程已退出)
                    // 我们可以安全地退出了
                    pipe_closed = 1;
                    break;
                } else {
                    // 读取出错 (比如被信号中断) 或者管道里暂时没数据了
                    if (errno != EAGAIN) { // EAGAIN 表示暂时没数据
                        perror("read from pipe");
                        pipe_closed = 1;
                    }
                    break; // 跳出内层循环，回到 select
                }
            }
        }
    } // 主循环结束

    printf("\n--- Child process finished, all data processed. ---\n");

    // 6. Print final eBPF results
    int map_total_fd = bpf_map__fd(skel->maps.total_ns);
    if (map_total_fd > 0) {
        print_all_thread_times(map_total_fd);
    } else {
        fprintf(stderr, "Failed to get BPF map FD for final results.\n");
    }

cleanup:
    // 7. Cleanup resources
    if (comm_pipe[0] != -1) close(comm_pipe[0]);
    if (comm_pipe[1] != -1) close(comm_pipe[1]); // Should already be closed
    if (skel) cpu_time_bpf__destroy(skel);
    
    // Ensure the child process is cleaned up properly
    if (child_pid > 0) {
        waitpid(child_pid, NULL, 0);
    }

    printf("Controller exiting.\n");
    return 0;
}