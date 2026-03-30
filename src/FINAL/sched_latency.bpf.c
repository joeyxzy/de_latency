// SPDX-License-Identifier: GPL-2.0
typedef unsigned char __u8;
typedef signed char __s8;
typedef unsigned short __u16;
typedef signed short __s16;
typedef unsigned int __u32;
typedef signed int __s32;
typedef unsigned long long __u64;
typedef signed long long __s64;
typedef __u16 __be16;
typedef __u32 __be32;
typedef __u32 __wsum;
typedef _Bool bool;

#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

#define BPF_MAP_TYPE_HASH 1
#define BPF_MAP_TYPE_RINGBUF 27
#define BPF_ANY 0

#define TASK_COMM_LEN 16

struct trace_event_raw_sched_wakeup {
    __u16 common_type;
    __u8 common_flags;
    __u8 common_preempt_count;
    __s32 common_pid;
    char comm[TASK_COMM_LEN];
    __s32 pid;
    __s32 prio;
    __s32 success;
    __s32 target_cpu;
};

struct trace_event_raw_sched_wakeup_new {
    __u16 common_type;
    __u8 common_flags;
    __u8 common_preempt_count;
    __s32 common_pid;
    char comm[TASK_COMM_LEN];
    __s32 pid;
    __s32 prio;
    __s32 success;
    __s32 target_cpu;
};

struct trace_event_raw_sched_switch {
    __u16 common_type;
    __u8 common_flags;
    __u8 common_preempt_count;
    __s32 common_pid;
    char prev_comm[TASK_COMM_LEN];
    __s32 prev_pid;
    __s32 prev_prio;
    long prev_state;
    char next_comm[TASK_COMM_LEN];
    __s32 next_pid;
    __s32 next_prio;
};

// ==========================================
// 1. Map 定义
// ==========================================

/* 目标 TID 白名单: target_tid_map[TID] = 1 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);   // TID
    __type(value, __u8);  // dummy=1
} target_tid_map SEC(".maps");

/* 记录进程变为 Runnable 的时刻: runnable_ts[TID] = { timestamp, type } */
struct runnable_info {
    __u64 ts;
    __u8  type; // 0=Wakeup (睡眠唤醒), 1=Preempt (被强占)
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, struct runnable_info);
} runnable_ts SEC(".maps");

/* 输出 Ring Buffer */
struct latency_event {
    __u64 start_ns;
    __u64 end_ns;
    __u32 tid;
    __u8  type; // 0=Wakeup, 1=Preempt
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// ==========================================
// 2. 辅助函数 (Internal Helpers)
// ==========================================

static __always_inline bool is_tracked_tid(__u32 tid)
{
    return bpf_map_lookup_elem(&target_tid_map, &tid) != NULL;
}

static __always_inline void record_runnable(__u32 tid, __u64 ts, __u8 type)
{
    struct runnable_info val = { .ts = ts, .type = type };
    bpf_map_update_elem(&runnable_ts, &tid, &val, BPF_ANY);
}

static __always_inline void send_event(__u32 tid, __u64 start, __u64 end, __u8 type)
{
    struct latency_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return;
    
    e->start_ns = start;
    e->end_ns = end;
    e->tid = tid;
    e->type = type;
    
    bpf_ringbuf_submit(e, 0);
}

// 提取出的公共逻辑，解决 BPF_PROG 调用问题
static __always_inline int trace_wakeup_common(__u32 tid)
{
    if (is_tracked_tid(tid)) {
        __u64 now = bpf_ktime_get_ns();
        record_runnable(tid, now, 0); // Type 0: Wakeup from sleep
    }
    return 0;
}

// ==========================================
// 3. BPF Programs (Hooks)
// ==========================================

/*
 * Hook 点 1: 进程被唤醒 (加入运行队列)
 */
SEC("tracepoint/sched/sched_wakeup")
int handle_sched_wakeup(struct trace_event_raw_sched_wakeup *ctx)
{
    return trace_wakeup_common((__u32)ctx->pid);
}

/*
 * Hook 点 2: 新进程创建时的唤醒
 */
SEC("tracepoint/sched/sched_wakeup_new")
int handle_sched_wakeup_new(struct trace_event_raw_sched_wakeup_new *ctx)
{
    return trace_wakeup_common((__u32)ctx->pid);
}

/*
 * Hook 点 3: 进程切换 (Context Switch)
 */
SEC("tracepoint/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    __u64 now = bpf_ktime_get_ns();

    // -----------------------------------------------------------
    // Part A: 处理切出进程 (Prev)
    // 如果它是 Running 状态被切出，说明是“被抢占”，开始计算延迟
    // -----------------------------------------------------------
    __u32 prev_tid = (__u32)ctx->prev_pid;
    long prev_state = ctx->prev_state;

    if (is_tracked_tid(prev_tid) && prev_state == 0) { // 0 == TASK_RUNNING
        record_runnable(prev_tid, now, 1); // Type 1: Preempted
    }

    // -----------------------------------------------------------
    // Part B: 处理切入进程 (Next)
    // 它终于拿到了 CPU，如果之前记录过开始时间，现在就是结束时间
    // -----------------------------------------------------------
    __u32 next_tid = (__u32)ctx->next_pid;
    
    if (is_tracked_tid(next_tid)) {
        struct runnable_info *info = bpf_map_lookup_elem(&runnable_ts, &next_tid);
        if (info && info->ts > 0) {
            __u64 dur_ns = now - info->ts;
            
            // 过滤极小的调度噪音 (> 5us)
            if (dur_ns > 5000) {
                send_event(next_tid, info->ts, now, info->type);
            }
            
            // 必须删除，防止 Map 爆满和逻辑错误的计算
            bpf_map_delete_elem(&runnable_ts, &next_tid);
        }
    }

    return 0;
}
