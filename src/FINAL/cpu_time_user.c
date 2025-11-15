#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdint.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <cupti.h>
#include <time.h>
#include <fcntl.h>
#include <zmq.h>
#include "cpu_time.skel.h"
#include "tracer_comm.h"
#include "encode.h"

static volatile sig_atomic_t exiting = 0;
static pid_t child_pid = 0;
static pid_t collector_pid = 0;
uint64_t offset=0;
__u64 cpu_walltime_start=UINT64_MAX;
__u64 cpu_walltime_end=0;
__u64 cpu_walltim=0;
__u64 cpu_devicetime=0;

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

#define COLLECTOR_PY "/home/joeyxzy/miniconda3/envs/vllm/bin/python"
#define COLLECTOR_SCRIPT "../src/FINAL/collector.py"
#define ZMQ_ADDR "ipc:///tmp/tracer.sock"

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

static int start_collector_process(void) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork collector");
        return 0;
    }
    if (pid == 0) {
        //子进程：exec collector.py
        setenv("TRACER_ZMQ_ADDR", ZMQ_ADDR, 1);
        execl(COLLECTOR_PY, COLLECTOR_PY, COLLECTOR_SCRIPT, "--bind", "ipc:///tmp/tracer.sock", (char *)NULL);
        //失败
        perror("execl collector");
        _exit(127);
    }
    //父进程：等待 collector 启动并将 socket 准备好
    if (wait_for_path("/tmp/tracer.sock", 2000) != 0) {
        fprintf(stderr, "warning: collector socket not ready after timeout\n");
        //杀死子进程
        kill(pid, SIGTERM);
        return 0;
    }
    return pid;
}

//用于zmq通信的全局变量
void *g_zmq_ctx = NULL;
void *g_zmq_push = NULL;

//初始化zmq传输器socket，在collector成功启动后调用
int init_zmq_sender(const char *addr) {
    g_zmq_ctx = zmq_ctx_new();
    if (!g_zmq_ctx) { perror("zmq_ctx_new"); return -1; }
    g_zmq_push = zmq_socket(g_zmq_ctx, ZMQ_PUSH);
    if (!g_zmq_push) { perror("zmq_socket"); return -1; }
    int linger = 0;
    zmq_setsockopt(g_zmq_push, ZMQ_LINGER, &linger, sizeof(linger));
    int rc = zmq_connect(g_zmq_push, addr);
    if (rc != 0) {
        fprintf(stderr, "zmq_connect failed: %s\n", zmq_strerror(zmq_errno()));
        // 可以 retry 或继续
    }
    return 0;
}

void shutdown_zmq_sender() {
    if (g_zmq_push) zmq_close(g_zmq_push);
    if (g_zmq_ctx) zmq_ctx_term(g_zmq_ctx);
}

// send_metadata_payload takes ownership of payload (will free it)
static void send_metadata_payload(const char *event_type, struct json_object *payload) {
    if (!g_zmq_push || !payload) return;
    struct json_object *meta = make_metadata("eBPF", event_type, NULL, -1);
    json_object_get(payload);  // increase ref
    json_object_object_add(meta, "payload", payload);
    const char *js = metadata_to_bytes(meta);
    zmq_send(g_zmq_push, js, strlen(js), ZMQ_DONTWAIT);
    json_object_put(meta);
}


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

void develop_cpu_metrics(__u64 start_ns,__u64 end_ns)
{
    cpu_walltime_start=min(start_ns,cpu_walltime_start);
    cpu_walltime_end=max(end_ns,cpu_walltime_end);
    cpu_walltim=cpu_walltime_end-cpu_walltime_start;
    cpu_devicetime+=end_ns-start_ns;
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
    // printf("eBPF-CPU [TID %-5u] ns range: [%llu, %llu]\n",
    //     event->pid,
    //     event->start_ns,
    //     event->end_ns);
    develop_cpu_metrics(event->start_ns,event->end_ns);
    struct json_object *payload = json_object_new_object();
    json_object_object_add(payload, "tgid", json_object_new_int((int)event->tgid));
    json_object_object_add(payload, "tid", json_object_new_int((int)event->pid));
    json_object_object_add(payload, "start_ns", json_object_new_int64((long long)event->start_ns));
    json_object_object_add(payload, "end_ns", json_object_new_int64((long long)event->end_ns));
    json_object_object_add(payload, "offset", json_object_new_int64((long long)offset));
    send_metadata_payload("cpu_interval", payload); // payload 在 meta 内释放
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
            printf("[INFO] Tracer attached to PID %u, TID %u\n", rec->pid, rec->tid);
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
    //int comm_pipe[2] = {-1, -1};//？
    //创建匿名管道
    //const char* fifo_path="/tmp/cupti_trace.fifo";
    int fifo_rfd=-1;int fifo_wfd=-1;
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
    // if (pipe(comm_pipe) == -1) {
    //     perror("pipe");
    //     goto cleanup;
    // }//？

    //unlink(fifo_path); // 保证是新的
    // if (mkfifo(fifo_path, 0666) == -1 && errno != EEXIST) {
    //     perror("mkfifo");
    //     goto cleanup;
    // }
    // 打开读端为非阻塞
    // fifo_rfd = open(fifo_path, O_RDONLY | O_NONBLOCK);
    // if (fifo_rfd == -1) {
    //     perror("open fifo read");
    //     goto cleanup;
    // }
    // 防止没有写端时 read 立即返回 EOF，保持一个写端占位
    // fifo_wfd = open(fifo_path, O_WRONLY | O_NONBLOCK);
    // if (fifo_wfd == -1) {
    //     perror("open fifo keep write");
    //     // 不致命
    // }

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
        printf("Child process started\n");
        //子进程
        close(sync_pipe[1]);
        //进程独立的环境变量
        setenv("LD_PRELOAD", "./libtracer.so", 1); // Ensure libtracer.so is in the current directory
        setenv("TRACER_ZMQ_ADDR", ZMQ_ADDR, 1);
        //此处是阻塞等待父进程的通知，确保父进程已经attach好了ebpf程序
        char sync_temp;
        printf("before sync read...\n");
        ssize_t n = read(sync_pipe[0],&sync_temp,sizeof(sync_temp));
        close(sync_pipe[0]);
        if (n == 1) {
            printf("sync received, launching target\n");
        } else if (n == 0) {
            printf("sync pipe EOF (父进程未写字节直接关闭)\n");
        } else {
            perror("read sync_pipe");
        }
        printf("yes!!\n");
        //此处子进程启动测试程序，启动的时候动态链接器会发现LD_PRELOAD设置的环境变量，所以就会装载我们的CUPTI程序
        execvp(argv[1], &argv[1]);

        //exec不会退出，退出就证明fail了
        perror("execvp");
        _exit(127);
    }

    close(sync_pipe[0]);

    // 4.初始化zmq传输器socket
    collector_pid = start_collector_process();
    if (collector_pid == 0) {
        fprintf(stderr, "Failed to start collector process\n");
        goto cleanup;
    }
    int init_zmq_=init_zmq_sender(ZMQ_ADDR);
    if (init_zmq_ != 0) {
        fprintf(stderr, "Failed to initialize ZMQ sender\n");
        //不致命
    }

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

    //使用更稳健的方式，因为主进程在fork的时候可能会复制管道，所以期待所有写端关闭不稳健，直接通过写的方式确定
    char notify = 1;
    if (write(sync_pipe[1], &notify, 1) != 1) {
        perror("write sync_pipe");
    }
    close(sync_pipe[1]);
    
    //为时间区间的ringbuffer注册回调打印函数
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_cpu_event, NULL, NULL);
    if (!rb) { fprintf(stderr, "Failed to create ring buffer\n");goto cleanup; }
    // 获取 ring buffer 的文件描述符，用于 select/epoll
    ringbuf_fd = ring_buffer__epoll_fd(rb);
    if (ringbuf_fd < 0) {fprintf(stderr, "Failed to get ring buffer epoll fd\n");goto cleanup; }

    printf("Tracking TGID: %d\n", child_pid);
    printf("--- eBPF CPU intervals collecting (JSON 已通过 ZMQ 发送) ---\n");

    int child_exited = 0;
    while (!exiting) {
        // 轮询 ring buffer
        ring_buffer__poll(rb, 200); // 200ms
        if (!child_exited) {
            int status;
            pid_t r = waitpid(child_pid, &status, WNOHANG);
            if (r == child_pid) child_exited = 1;
        }
        if (child_exited) {
            // 再多 poll 几次吸尽剩余事件
            for (int i=0;i<5;i++) ring_buffer__poll(rb, 100);
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

    printf("CPU WALL TIME: %llu ms\n",(cpu_walltime_end-cpu_walltime_start)/1000000);
    printf("CPU DEVICE TIME: %llu ms\n",cpu_devicetime/1000000);

cleanup:
    // 7. Cleanup resources
    shutdown_zmq_sender();
    if(rb) ring_buffer__free(rb);
    if (skel) cpu_time_bpf__destroy(skel);
    
    // Ensure the child process is cleaned up properly
    if (child_pid > 0) {
        waitpid(child_pid, NULL, 0);
    }
    if (collector_pid > 0) {
        kill(collector_pid, SIGTERM);
        waitpid(collector_pid, NULL, 0);
    }
    printf("Controller exiting.\n");
    return 0;
}