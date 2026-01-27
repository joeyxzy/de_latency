// trace_latency.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <zmq.h>
#include <json-c/json.h>

#include "sched_latency.skel.h"

// ZMQ Configuration
#define ZMQ_ADDR "ipc:///tmp/tracer.sock"
static void *g_zmq_ctx = NULL;
static void *g_zmq_sock = NULL;

static volatile bool exiting = false;

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
    if (zmq_connect(g_zmq_sock, ZMQ_ADDR) != 0) {
        fprintf(stderr, "zmq_connect failed: %s\n", zmq_strerror(errno));
        return -1;
    }
    
    printf("ZMQ connected to %s\n", ZMQ_ADDR);
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
    if (data_sz != sizeof(*e)) {
        fprintf(stderr, "Invalid event size\n");
        return 1;
    }

    // 1. 创建 Payload JSON 对象
    struct json_object *payload_obj = json_object_new_object();
    json_object_object_add(payload_obj, "tid", json_object_new_int((int)e->tid));
    json_object_object_add(payload_obj, "start_ns", json_object_new_int64((int64_t)e->start_ns));
    json_object_object_add(payload_obj, "end_ns", json_object_new_int64((int64_t)e->end_ns));
    json_object_object_add(payload_obj, "dur_us", json_object_new_int64((int64_t)(e->end_ns - e->start_ns))); // 这里修正了单位转换逻辑，假设原本是ns差值
    json_object_object_add(payload_obj, "reason", json_object_new_string((e->type == 0) ? "Wakeup" : "Preempt"));

    // 2. 创建 Meta JSON 对象 (最外层)
    struct json_object *meta_obj = json_object_new_object();
    json_object_object_add(meta_obj, "source", json_object_new_string("ebpf"));
    json_object_object_add(meta_obj, "event_type", json_object_new_string("sched_latency"));
    json_object_object_add(meta_obj, "timestamp", json_object_new_int64(e->end_ns));
    
    // 关键点：将 payload 放入 meta 中
    json_object_object_add(meta_obj, "payload", payload_obj);

    // 3. 序列化并发送 (单帧)
    const char *full_msg_str = json_object_to_json_string(meta_obj);
    size_t full_msg_len = strlen(full_msg_str);

    // 发送单帧 (无 SNDMORE)
    zmq_send(g_zmq_sock, full_msg_str, full_msg_len, 0);

    // 释放内存
    // 注意：json_object_put(meta_obj) 会递归释放它包含的 payload_obj，
    // 所以不需要单独释放 payload_obj
    json_object_put(meta_obj);

    return 0;
}

int main(int argc, char **argv)
{
    struct sched_latency_bpf *skel;
    struct ring_buffer *rb;
    int i;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <TID1> [TID2] ... [TIDN]\n", argv[0]);
        return 1;
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
    __u8 val = 1;
    for (i = 1; i < argc; i++) {
        char *endptr;
        unsigned long tid = strtoul(argv[i], &endptr, 10);
        if (*endptr != '\0' || tid == 0 || tid > UINT32_MAX) {
            fprintf(stderr, "Invalid TID: %s\n", argv[i]);
            goto cleanup;
        }
        bpf_map_update_elem(tid_map_fd, &(__u32){tid}, &val, BPF_ANY);
        printf("Tracing TID %lu\n", tid);
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

    printf("\n--- eBPF Tracer Running (Sending to %s) ---\n", ZMQ_ADDR);

    while (!exiting) {
        ring_buffer__poll(rb, 100); 
    }

cleanup:
    ring_buffer__free(rb);
    sched_latency_bpf__destroy(skel);
    cleanup_zmq();
    printf("\nExiting...\n");
    return 0;
}