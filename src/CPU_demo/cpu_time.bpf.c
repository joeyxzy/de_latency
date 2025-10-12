// cpu_time.bpf.c
// clang -O2 -g -target bpf -c cpu_time.bpf.c -o cpu_time.bpf.o

#include "vmlinux.h"
#include "bpf/bpf_helpers.h"
#include "bpf/bpf_core_read.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

/* maps */

// set of PIDs we track (key = pid (u32) -> value = u8 (1))
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u8);
} targets SEC(".maps");

// per-pid last switch-in timestamp (ns)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} ts_in SEC(".maps");

// per-pid accumulated running time in ns
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} total_ns SEC(".maps");

/* tracepoint handler for sched_switch */
SEC("tracepoint/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    __u32 prev_pid = ctx->prev_pid;
    __u32 next_pid = ctx->next_pid;
    __u64 now = bpf_ktime_get_ns();

    // --- handle prev (switched out): if prev is a target and we recorded ts_in, accumulate ---
    __u8 *is_prev_tracked = bpf_map_lookup_elem(&targets, &prev_pid);
    if (is_prev_tracked) {
        __u64 *t_in = bpf_map_lookup_elem(&ts_in, &prev_pid);
        if (t_in) {
            __u64 dur = now - *t_in;

            __u64 *acc = bpf_map_lookup_elem(&total_ns, &prev_pid);
            if (acc) {
                __u64 new = *acc + dur;
                bpf_map_update_elem(&total_ns, &prev_pid, &new, BPF_ANY);
            } else {
                __u64 init = dur;
                bpf_map_update_elem(&total_ns, &prev_pid, &init, BPF_ANY);
            }
            bpf_map_delete_elem(&ts_in, &prev_pid);
        }
    }

    // --- handle next (switched in): if next is target, set ts_in if not already set ---
    __u8 *is_next_tracked = bpf_map_lookup_elem(&targets, &next_pid);
    if (is_next_tracked) {
        __u64 now_copy = now;
        bpf_map_update_elem(&ts_in, &next_pid, &now_copy, BPF_ANY);
    }

    return 0;
}
