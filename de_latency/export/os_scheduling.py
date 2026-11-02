"""Event-loop scheduling waits, rendered independently of OS attribution."""


def append_eventloop_waits(events, writer, by_tid, tids, offsets, fallback_offset, roles, origin):
    pid = 12000
    if tids:
        events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": "EventLoop OS Scheduling"}})
    for tid in sorted(tids):
        waits, offset = by_tid.get(tid, []), offsets.get(tid, fallback_offset)
        if not waits or offset is None:
            continue
        role = roles.get(tid, {})
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                       "args": {"name": f"eventloop tid={tid}"}})
        for wait in waits:
            start, end = max(origin, wait["start"] + offset), wait["end"] + offset
            if end > start:
                events.append(writer.event("OS Runnable Wait", "eventloop_os", "X", start, end - start,
                    pid, tid, args={"source_tid": tid, "source_pid": role.get("pid"),
                        "reason": wait.get("reason"), "roles": ",".join(sorted(role.get("roles", [])))}, cname="terrible"))
