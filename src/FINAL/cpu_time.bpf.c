// cpu_time.bpf.c  (modified: use map for target_tgid + exit handler)
#include "vmlinux.h"
#include "bpf/bpf_helpers.h"
#include "bpf/bpf_core_read.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

/* target tgid stored in a small array map at key 0 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u32);   // tgid
    __type(value, __u8);  // dummy=1
} target_tgid_map SEC(".maps");

//中间数据结构，某个线程的上一次上CPU时间点
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);   // TID
    __type(value, __u64);
} ts_in SEC(".maps");

//每个线程的累计时间
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);   // TID
    __type(value, __u64);
} total_ns SEC(".maps");

//时间区间数据结构
struct cpu_event {
    __u64 start_ns;
    __u64 end_ns;
    __u32 pid; // 这是TID
    __u32 tgid;
};

// 判定是否跟踪
static __always_inline bool is_tracked_tgid(__u32 tgid)
{
    __u8 *v = bpf_map_lookup_elem(&target_tgid_map, &tgid);
    return v != 0;
}

//用于传递时间区间的map
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

/* helper to get current target_tgid from map; returns 0 if unset */
static __always_inline __u32 get_target_tgid(void)
{
    __u32 idx = 0;
    __u32 *val = bpf_map_lookup_elem(&target_tgid_map, &idx);
    if (!val) return 0;
    return *val;
}

//传递时间区间
static __always_inline void
send_cpu_event(__u32 tid, __u32 tgid, __u64 start_ns, __u64 end_ns)
{
    bpf_printk("send_cpu_event called for TID %u\n", tid);
    struct cpu_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return;
    }
    event->start_ns = start_ns;
    event->end_ns = end_ns;
    event->pid = tid;
    event->tgid = tgid;
    bpf_printk("Submitting event for TID %u\n", tid);
    bpf_ringbuf_submit(event, 0);
}

SEC("raw_tracepoint/sched_switch")
int handle_sched_switch(struct bpf_raw_tracepoint_args *ctx)
{
    struct task_struct *prev = (struct task_struct *)ctx->args[1];
    struct task_struct *next = (struct task_struct *)ctx->args[2];
    __u64 now = bpf_ktime_get_ns();

    __u32 prev_pid  = BPF_CORE_READ(prev, pid);
    __u32 prev_tgid = BPF_CORE_READ(prev, tgid);

    if (is_tracked_tgid(prev_tgid)) {
        __u64 *t_in = bpf_map_lookup_elem(&ts_in, &prev_pid);
        if (t_in) {
            send_cpu_event(prev_pid, prev_tgid, *t_in, now);
            __u64 dur = now - *t_in;
            __u64 *acc = bpf_map_lookup_elem(&total_ns, &prev_pid);
            if (acc) {
                __u64 new_total = *acc + dur;
                bpf_map_update_elem(&total_ns, &prev_pid, &new_total, BPF_ANY);
            } else {
                bpf_map_update_elem(&total_ns, &prev_pid, &dur, BPF_ANY);
            }
            bpf_map_delete_elem(&ts_in, &prev_pid);
        }
    }

    __u32 next_pid  = BPF_CORE_READ(next, pid);
    __u32 next_tgid = BPF_CORE_READ(next, tgid);
    if (is_tracked_tgid(next_tgid)) {
        bpf_map_update_elem(&ts_in, &next_pid, &now, BPF_ANY);
    }
    return 0;
}

SEC("raw_tracepoint/sched_process_exit")
int handle_process_exit(struct bpf_raw_tracepoint_args *ctx)
{
    struct task_struct *task = (struct task_struct *)ctx->args[0];
    __u64 now = bpf_ktime_get_ns();
    __u32 pid  = BPF_CORE_READ(task, pid);
    __u32 tgid = BPF_CORE_READ(task, tgid);

    if (is_tracked_tgid(tgid)) {
        __u64 *t_in = bpf_map_lookup_elem(&ts_in, &pid);
        if (t_in) {
            send_cpu_event(pid, tgid, *t_in, now);
            __u64 dur = now - *t_in;
            __u64 *acc = bpf_map_lookup_elem(&total_ns, &pid);
            if (acc) {
                __u64 new_total = *acc + dur;
                bpf_map_update_elem(&total_ns, &pid, &new_total, BPF_ANY);
            } else {
                bpf_map_update_elem(&total_ns, &pid, &dur, BPF_ANY);
            }
            bpf_map_delete_elem(&ts_in, &pid);
        }
        // 可选：进程退出后删除 tgid，避免 map 膨胀
        bpf_map_delete_elem(&target_tgid_map, &tgid);
    }
    return 0;
}

SEC("raw_tracepoint/sched_process_fork")
int handle_process_fork(struct bpf_raw_tracepoint_args *ctx)
{
    struct task_struct *parent = (struct task_struct *)ctx->args[0];
    struct task_struct *child  = (struct task_struct *)ctx->args[1];

    __u32 parent_tgid = BPF_CORE_READ(parent, tgid);
    __u32 child_tgid  = BPF_CORE_READ(child, tgid);

    if (is_tracked_tgid(parent_tgid) && !is_tracked_tgid(child_tgid)) {
        __u8 one = 1;
        bpf_map_update_elem(&target_tgid_map, &child_tgid, &one, BPF_ANY);
    }
    return 0;
}