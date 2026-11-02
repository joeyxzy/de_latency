"""CUPTI correlation IDs are process-local and can wrap within a process."""


def cupti_corr_key(payload, corr_id=None, pid_override=None):
    if corr_id is None:
        corr_id = payload.get("correlationId")
    pid = pid_override if pid_override is not None else payload.get("pid")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        pid = 0
    return pid, corr_id


def kernels_for_launch(kernel_index, api, pid):
    candidates = kernel_index.get(cupti_corr_key(api, pid_override=pid), [])
    start = api.get("start_ns")
    if candidates and start is not None:
        # Permit clock calibration drift, but reject unrelated launch epochs.
        return [kernel for kernel in candidates
                if int(start) - 1_000_000_000 <= (kernel.get("start_ns") or 0)
                <= int(start) + 30_000_000_000]
    return candidates
