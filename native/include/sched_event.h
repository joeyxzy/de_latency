#ifndef DE_LATENCY_SCHED_EVENT_H
#define DE_LATENCY_SCHED_EVENT_H
/* Shared BPF/userspace ABI; __u* types are provided by each compilation unit. */
struct latency_event {
    __u64 start_ns;
    __u64 end_ns;
    __u32 tid;
    __u8 type; /* 0=wakeup, 1=preemption */
};
#endif
