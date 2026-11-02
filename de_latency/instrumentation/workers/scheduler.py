"""Scheduler decisions, per-request phases, and step completion metadata."""
import time

from ...runtime.transport import TraceSender
from ..registry import register_hook, apply_method_patch
from ..scheduler_metadata import normalize_req_ids, request_ids, step_maps, dispatch_key, safe_int
from .context import ProcessState


def token_state(scheduler):
    prompt, output = {}, {}
    for rid, request in (getattr(scheduler, "requests", {}) or {}).items():
        keys = list(dict.fromkeys([rid] + normalize_req_ids(rid)))
        prompt_count = safe_int(getattr(request, "num_prompt_tokens", None))
        output_count = safe_int(getattr(request, "num_output_tokens", None))
        placeholders = safe_int(getattr(request, "num_output_placeholders", None), 0) or 0
        for key in keys:
            if prompt_count is not None:
                prompt[key] = prompt_count
            if output_count is not None:
                output[key] = output_count + placeholders
    return prompt, output


def phase_from_counts(computed, prompt):
    if computed is None or prompt is None:
        return None
    return "prefill" if computed < prompt else "decode"


def phase_metadata(scheduler, scheduled):
    phases, reasons, decode_steps = {}, {}, {}
    prompts, outputs = token_state(scheduler)
    for request in scheduled.scheduled_new_reqs:
        rid = getattr(request, "request_id", None) or request.req_id
        tokens = request.prompt_token_ids
        phase = phase_from_counts(safe_int(request.num_computed_tokens), len(tokens) if tokens is not None else None)
        if phase:
            phases[rid], reasons[rid] = phase, "new_req_static_token_counts"
    cached = scheduled.scheduled_cached_reqs
    for index, rid in enumerate(cached.req_ids):
        if not rid:
            continue
        phase = None
        computed = safe_int(cached.num_computed_tokens[index]) if index < len(cached.num_computed_tokens) else None
        if hasattr(cached, "is_context_phase"):
            try:
                phase = "prefill" if cached.is_context_phase(rid) else "decode"
                reasons[rid] = "cached_is_context_phase"
            except Exception:
                pass
        if phase is None:
            phase = phase_from_counts(computed, prompts.get(rid))
            if phase:
                reasons[rid] = "cached_token_counts_fallback"
        if phase is None and rid in outputs:
            phase = "prefill" if outputs[rid] == 0 else "decode"
            reasons[rid] = "cached_output_tokens_fallback"
        if phase:
            phases[rid] = phase
    computed, _ = step_maps(scheduled)
    for rid in request_ids(scheduled):
        if phases.get(rid) != "decode" or computed.get(rid) is None or prompts.get(rid) is None:
            continue
        step = computed[rid] - prompts[rid] + 1
        if step > 0:
            decode_steps[rid] = step
    return phases, reasons, decode_steps


def result_events(results, phases):
    updates = []
    for engine, batch in results.items():
        for output in getattr(batch, "outputs", []):
            events = []
            rid = getattr(output, "request_id", "unknown")
            for event in getattr(output, "events", None) or []:
                kind, timestamp = getattr(event, "type", None), getattr(event, "timestamp", None)
                if kind is None or not timestamp:
                    continue
                kind = str(kind)
                item = {"type": kind, "ts": timestamp}
                if (kind == "2" or "SCHEDULED" in kind.upper()) and phases.get(rid) in ("prefill", "decode"):
                    item["phase"] = phases[rid]
                events.append(item)
            if events:
                updates.append({"req_id": rid, "events": events, "engine_id": engine})
    return updates


@register_hook("vllm.v1.core.sched.scheduler")
def patch_v1_scheduler(module):
    cls = module.Scheduler

    def add(original):
        def wrapper(self, request):
            TraceSender.emit("req_enqueue_scheduler", {"req_id": getattr(request, "request_id", "unknown"),
                "event": "enqueue_start", "timestamp_ns": time.time_ns()})
            return original(self, request)
        return wrapper

    def update(original):
        def wrapper(self, scheduler_output, model_output):
            phases, reasons, steps = phase_metadata(self, scheduler_output)
            key = dispatch_key(scheduler_output)
            before = time.time_ns()
            result = original(self, scheduler_output, model_output)
            ids = request_ids(scheduler_output)
            if ids:
                computed, scheduled = step_maps(scheduler_output)
                TraceSender.emit("req_step_ready", {"req_ids": ids, "phase_by_req": phases,
                    "phase_reason_by_req": reasons, "decode_step_by_req": steps, "dispatch_key": key,
                    "event": "step_ready", "num_computed_by_req": computed, "scheduled_tokens_by_req": scheduled,
                    "timestamp_ns": time.time_ns(), "scheduler_before_ns": before})
            events = result_events(result, phases)
            if events:
                TraceSender.emit("req_metrics_events", {"req_events": events, "dispatch_key": key})
            return result
        return wrapper

    def schedule(original):
        def wrapper(self, *args, **kwargs):
            start = time.time_ns()
            result = original(self, *args, **kwargs)
            end = time.time_ns()
            key = dispatch_key(result)
            ProcessState.last_dispatch_key = key
            TraceSender.emit("scheduler_schedule", {"start_ns": start, "end_ns": end,
                "duration_ns": end - start, "num_requests": len(getattr(self, "requests", [])),
                "num_running": len(getattr(self, "running", [])), "num_waiting": len(getattr(self, "waiting", [])),
                "scheduled_batch_size": len(request_ids(result)),
                "total_scheduled_tokens": safe_int(result.total_num_scheduled_tokens, 0),
                "timestamp_ns": end, "dispatch_key": key})
            return result
        return wrapper

    apply_method_patch(cls, "add_request", add)
    apply_method_patch(cls, "update_from_output", update)
    apply_method_patch(cls, "schedule", schedule)
