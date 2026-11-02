"""Per-analysis Perfetto event factory; origin is not process-global."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PerfettoWriter:
    origin_ns: int

    def event(self, name, cat, ph, ts, dur, pid, tid, args=None, cname=None):
        event = {"name": name, "cat": cat, "ph": ph,
                 "ts": (ts - self.origin_ns) / 1000.0,
                 "pid": pid, "tid": tid, "args": args or {}}
        if dur is not None:
            event["dur"] = dur / 1000.0
        if cname:
            event["cname"] = cname
        return event

    def flow(self, ph, ts, pid, tid, corr_id):
        return {"name": "link", "cat": "flow", "ph": ph,
                "ts": (ts - self.origin_ns) / 1000.0,
                "pid": pid, "tid": tid, "id": corr_id}


def nudge_equal_boundaries(events, target_pid=None, epsilon_us=.001, cat_prefixes=None, snap_tolerance_us=0):
    slots = []
    for index, event in enumerate(events):
        if event.get("ph") != "X" or (target_pid is not None and event.get("pid") != target_pid):
            continue
        if cat_prefixes and not str(event.get("cat", "")).startswith(tuple(cat_prefixes)):
            continue
        try:
            start, duration = float(event["ts"]), float(event["dur"])
        except (KeyError, TypeError, ValueError):
            continue
        if duration > 0:
            slots.append((event.get("pid"), event.get("tid"), start, start + duration, index))
    last_end = {}
    changed = 0
    for pid, tid, start, end, index in sorted(slots):
        previous = last_end.get((pid, tid))
        if previous is not None and (start == previous or (snap_tolerance_us > 0 and abs(start - previous) <= snap_tolerance_us)):
            delta = previous + epsilon_us - start
            duration = float(events[index]["dur"])
            if 0 < delta < duration:
                events[index]["ts"] = start + delta
                events[index]["dur"] = duration - delta
                end = events[index]["ts"] + events[index]["dur"]
                changed += 1
        last_end[(pid, tid)] = end
    return changed


def clip_negative_complete_events(events):
    clipped = dropped = 0
    kept = []
    for event in events:
        if event.get("ph") == "X":
            try:
                start, duration = float(event["ts"]), float(event["dur"])
            except (KeyError, ValueError, TypeError):
                kept.append(event)
                continue
            if start < 0:
                if start + duration <= 0:
                    dropped += 1
                    continue
                event["ts"], event["dur"] = 0.0, start + duration
                clipped += 1
        kept.append(event)
    events[:] = kept
    return clipped, dropped
