#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/epoll.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <cupti.h>
#include<time.h>
#include "cpu_time.skel.h"
#include "tracer_comm.h" // You must have this shared header file

// --- Global State ---
static volatile sig_atomic_t exiting = 0;
static pid_t child_pid = 0;
uint64_t offset=0;

// --- Signal Handler ---
static void sigint_handler(int sig) {
    exiting = 1;
    // If the child is still running, send it a signal to terminate
    if (child_pid > 0) {
        kill(child_pid, SIGTERM);
    }
}

static inline uint64_t gt2ct(uint64_t gpu_time_ns)
{
    return gpu_time_ns+offset;
}

static inline uint64_t cpu_get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
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

// libbpf 会在 ring buffer 中有数据时自动调用这个函数
int handle_cpu_event(void *ctx, void *data, size_t size) {
    // 从 BPF 内核代码中复制 cpu_event 结构体的定义过来
    struct cpu_event {
        __u64 start_ns;
        __u64 end_ns;
        __u32 pid; // 这是TID
        __u32 tgid;
    };

    const struct cpu_event *event = data;
    
    // 安全检查
    if (size != sizeof(*event)) {
        fprintf(stderr, "Warning: Invalid cpu_event size received\n");
        return 1;
    }
    
    // 打印实时收到的 CPU 时间区间
    // 为了和CUPTI的输出区分开，我们加个前缀
    printf("eBPF-CPU [TID %-5u] ns range: [%llu, %llu]\n",
        event->pid,
        event->start_ns,
        event->end_ns);

    return 0;
}

// --- CUPTI Data Processing and Printing ---
void process_cupti_record(UnifiedTraceRecord *rec) {
    // Note: Timestamps are raw from CUPTI and are not aligned to the system clock.
    switch (rec->type) {
        case RECORD_TYPE_KERNEL:
            printf("KERNEL \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
                   rec->name, gt2ct(rec->start_ns), gt2ct(rec->end_ns), rec->deviceId, rec->contextId, rec->streamId, rec->correlationId);
            printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
                   rec->gridX, rec->gridY, rec->gridZ, rec->blockX, rec->blockY, rec->blockZ,
                   rec->staticSharedMemory, rec->dynamicSharedMemory);
            break;
        case RECORD_TYPE_MEMCPY:
            printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
                   rec->name, gt2ct(rec->start_ns), gt2ct(rec->end_ns), rec->deviceId, rec->contextId, rec->streamId,
                   rec->correlationId, rec->memcpy_runtimeCorrelationId);
            break;
        case RECORD_TYPE_MEMSET:
            printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
                   rec->memset_value, gt2ct(rec->start_ns), gt2ct(rec->end_ns), rec->deviceId, rec->contextId, rec->streamId,
                   rec->correlationId);
            break;
        case RECORD_TYPE_DRIVER:
            printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                   rec->cbid, gt2ct(rec->start_ns), gt2ct(rec->end_ns), rec->pid, rec->tid, rec->correlationId);
            break;
        case RECORD_TYPE_RUNTIME:
            printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
                   rec->cbid, gt2ct(rec->start_ns), gt2ct(rec->end_ns), rec->pid, rec->tid, rec->correlationId);
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
    struct ring_buffer *rb = NULL;
    int ringbuf_fd = -1;
    int comm_pipe[2] = {-1, -1};
    //增加一个同步管道，用于保证子进程启动目标程序在父进程attach ebpf程序以后
    int sync_pipe[2]={-1,-1};

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // 1.初始化bpf
    //1.1填充skel
    skel = cpu_time_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }
    //1.2加载ebpf到内核中，这个阶段验证器会工作
    if (cpu_time_bpf__load(skel)) {
        fprintf(stderr, "Failed to load BPF skeleton\n");
        goto cleanup;
    }

    // 2.创建管道
    if (pipe(comm_pipe) == -1) {
        perror("pipe");
        goto cleanup;
    }
    if(pipe(sync_pipe)==-1)
    {
        perror("sync_pipe");
        goto cleanup;
    }

    // 3. Fork and execute the target application
    child_pid = fork();
    if (child_pid < 0) {
        perror("fork");
        goto cleanup;
    }

    if (child_pid == 0) {
        //子进程
        close(comm_pipe[0]); // Child only writes, so close read end
        close(sync_pipe[1]);
        //此处是将pipe管道的描述符字符串化写到pipe_fd_str里
        char pipe_fd_str[16];
        snprintf(pipe_fd_str, sizeof(pipe_fd_str), "%d", comm_pipe[1]);
        setenv("CUPTI_COMM_FD", pipe_fd_str, 1);
        //进程独立的环境变量
        setenv("LD_PRELOAD", "./libtracer.so", 1); // Ensure libtracer.so is in the current directory

        char sync_temp;
        read(sync_pipe[0],&sync_temp,sizeof(sync_temp));
        close(sync_pipe[0]);

        //此处子进程启动测试程序，启动的时候动态链接器会发现LD_PRELOAD设置的环境变量，所以就会装载我们的CUPTI程序
        execvp(argv[1], &argv[1]);

        //exec不会退出，退出就证明fail了
        perror("execvp");
        close(comm_pipe[1]);
        _exit(127);
    }

    // Parent process (Controller)
    close(comm_pipe[1]); // Parent only reads, so close write end
    close(sync_pipe[0]);
    printf("Tracking process group with TGID: %d\n", child_pid);

    //父进程上传子进程PID，即线程组号，用于ebpf后续找到自己要采的线程
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

    //attach之后ebpf程序开始立即执行
    //!!需要注意的是：有可能存在某种情况，目标程序已经开始运行了，但是ebpf还没有开始采（虽然可能性不大，因为exec显然要慢很多，ebpf可能会率先等好）
    if (cpu_time_bpf__attach(skel) != 0) {
        fprintf(stderr, "Failed to attach BPF skeleton\n");
        goto cleanup;
    }
    uint64_t cpu_ref = cpu_get_time_ns();  // CPU reference
    uint64_t gpu_ref;
    cuptiGetTimestamp(&gpu_ref); // GPU reference
    offset = (int64_t)cpu_ref - (int64_t)gpu_ref;

    close(sync_pipe[1]);//父进程关闭写端就会通知子进程的读端，读端的阻塞就会结束

    //为时间区间的ringbuffer注册回调打印函数
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_cpu_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "Failed to create ring buffer\n");
        goto cleanup;
    }
    // 获取 ring buffer 的文件描述符，用于 select/epoll
    ringbuf_fd = ring_buffer__epoll_fd(rb);
    if (ringbuf_fd < 0) {
        fprintf(stderr, "Failed to get ring buffer epoll fd\n");
        goto cleanup;
    }

    int cupti_fd=comm_pipe[0];
    //创建epoll实例
    int epoll_fd=epoll_create1(0);
    if(epoll_fd<0)
    {
        perror("epoll_create1");
        goto cleanup;
    }

    // --- 1. 为 cupti_fd 准备并添加事件 ---
    struct epoll_event cupti_event;
    cupti_event.events = EPOLLIN;
    cupti_event.data.fd = cupti_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, cupti_fd, &cupti_event) < 0) {
        perror("epoll_ctl add cupti_fd");
        close(epoll_fd);
        goto cleanup;
    }
    
    // --- 2. 为 ringbuf_fd 准备并添加事件 ---
    struct epoll_event ringbuf_event;
    ringbuf_event.events = EPOLLIN;
    ringbuf_event.data.fd = ringbuf_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ringbuf_fd, &ringbuf_event) < 0) {
        perror("epoll_ctl add ringbuf_fd");
        close(epoll_fd);
        goto cleanup;
    }
    
    // 4. Main event loop: read from the pipe
    printf("--- Real-time CUPTI Trace (timestamps are relative to tracer start) ---\n");

    // 4. =========== 主事件循环 (重写逻辑) ===========
    // 替换主循环：更稳健地处理 epoll + ringbuf + 子进程退出
    int pipe_closed = 0;
    int child_exited = 0;
    struct epoll_event active_events[4];

    while (1) {
        if (exiting) {
            // 收到 SIGINT/SIGTERM，优雅退出前继续处理剩余 ringbuf 事件
            fprintf(stderr, "[main] exiting flag set\n");
        }

        int n_events = epoll_wait(epoll_fd, active_events, 4, 1000); // 1s 超时
        if (n_events < 0) {
            if (errno == EINTR) {
                // 被信号打断，重试或检查退出条件
                continue;
            }
            perror("epoll_wait");
            break;
        }

        // 如果 epoll 没有事件，也要尝试 poll ring buffer（防止漏掉）
        if (n_events == 0) {
            // 轮询 ringbuf，处理可能缓存在内核的事件
            int processed = ring_buffer__poll(rb, 0);
            // 可选 debug:
            // printf("[main] epoll timeout, ring_buffer__poll returned %d\n", processed);
        }

        for (int i = 0; i < n_events; i++) {
            int fd = active_events[i].data.fd;
            uint32_t ev = active_events[i].events;
            // debug 打印（删除或定为 --debug 模式）
            // printf("[main] epoll event fd=%d ev=0x%x\n", fd, ev);

            if (fd == cupti_fd) {
                // 处理 CUPTI 管道（非阻塞 fd）
                UnifiedTraceRecord rec;
                while (1) {
                    ssize_t bytes_read = read(cupti_fd, &rec, sizeof(rec));
                    if (bytes_read == sizeof(rec)) {
                        process_cupti_record(&rec);
                    } else if (bytes_read == 0) {
                        // 管道写端关闭（子进程或 tracer lib 已关闭）
                        pipe_closed = 1;
                        // printf("[main] cupti pipe EOF\n");
                        break;
                    } else {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            // 非阻塞，暂时无数据
                            break;
                        }
                        // 其他错误：记录并视为管道关闭/出问题
                        perror("read from cupti pipe");
                        pipe_closed = 1;
                        break;
                    }
                }
            } else if (fd == ringbuf_fd) {
                // ring buffer 的 eventfd 触发，去 poll 并处理事件
                // 注意：ring_buffer__poll 会调用你的 handle_cpu_event 回调
                // 我们把返回值打印出来以便 debug
                int processed = ring_buffer__poll(rb, 0);
                // printf("[main] ring_buffer__poll returned %d\n", processed);
            } else {
                // 未知的 fd：根据实际情况处理或忽略
                // printf("[main] unknown fd %d\n", fd);
            }
        }

        // 每个循环都尝试非阻塞地处理残余 ringbuf 事件，确保不漏样本
        ring_buffer__poll(rb, 0);

        // 检查子进程是否退出（非阻塞）
        if (!child_exited) {
            int status;
            pid_t r = waitpid(child_pid, &status, WNOHANG);
            if (r == child_pid) {
                child_exited = 1;
                // printf("[main] child exited, status=%d\n", status);
            }
        }

        // 退出条件：外部退出或子进程已退出并且管道已见 EOF
        // 这里我们要求：子进程已退出 或 (pipe 已关闭)
        // 但为了保险起见，还要确保 ring buffer 中没有立即可处理的事件
        if (exiting) {
            break;
        }
        if (child_exited && pipe_closed) {
            // 做一次最终的 ringbuf 处理，确保不漏最后数据
            int more;
            do {
                more = ring_buffer__poll(rb, 100); // 等待最多 100ms 来吸尽残留事件
            } while (more > 0);
            break;
        }
    }


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
    if(rb) ring_buffer__free(rb);
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