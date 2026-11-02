"""PP communication spans; common wrappers preserve tensor/handle return contracts."""
import itertools
import os
import threading
import time

from ...runtime.control import is_enabled
from ...runtime.transport import TraceSender
from ..execution_context import current
from ..registry import register_hook, apply_method_patch
from .context import ranks

_sequence = itertools.count(1)


def tensor_summary(value, limit=32):
    result = {"tensor_count": 0, "bytes": 0, "items": []}

    def visit(item, path):
        if len(result["items"]) >= limit:
            return
        if hasattr(item, "numel") and hasattr(item, "element_size"):
            try:
                size = max(0, int(item.numel()) * int(item.element_size()))
            except Exception:
                size = 0
            try:
                shape = list(item.shape)
            except Exception:
                shape = None
            result["tensor_count"] += 1
            result["bytes"] += size
            result["items"].append({"path": path, "shape": shape, "dtype": str(getattr(item, "dtype", "")),
                                     "device": str(getattr(item, "device", "")), "bytes": size})
        elif isinstance(item, dict):
            for key, child in list(item.items())[:limit]:
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item[:limit]):
                visit(child, f"{path}[{index}]")
    visit(value, "")
    return result


def is_pp(group):
    try:
        from vllm.distributed.parallel_state import get_pp_group
        pp = get_pp_group()
        name = str(getattr(group, "unique_name", "") or "").lower()
        return group is pp or name.startswith("pp") or ":pp" in name or name.endswith("pp") or (
            list(group.ranks) == list(pp.ranks) and group.world_size > 1)
    except Exception:
        return False


def payload(group, op, start, end, peer, direction, tensors):
    context = current() or {}
    summary = tensor_summary(tensors) if tensors is not None else {}
    return {"op": op, "pid": os.getpid(), "tid": threading.get_native_id(),
        "dispatch_key": context.get("dispatch_key"), "req_ids": list(context.get("req_ids") or []),
        "ranks": dict(context.get("ranks") or ranks()), "group_rank": group.rank_in_group,
        "group_world_size": group.world_size, "group_ranks": list(group.ranks),
        "peer_rank": peer, "src": peer if direction == "src" else None,
        "dst": peer if direction == "dst" else None, "tensor_count": summary.get("tensor_count", 0),
        "tensor_bytes": summary.get("bytes", 0), "tensor_items": summary.get("items", []),
        "start_ns": start, "end_ns": end, "duration_ns": max(0, end - start), "timestamp_ns": end}


class TracedHandle:
    def __init__(self, handle, launch, op, index):
        self.handle, self.launch, self.op, self.index = handle, dict(launch), op, index

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def wait(self, *args, **kwargs):
        start = time.time_ns()
        try:
            return self.handle.wait(*args, **kwargs)
        finally:
            end = time.time_ns()
            TraceSender.emit("pp_comm_span", {**self.launch, "op": self.op, "tid": threading.get_native_id(),
                "start_ns": start, "end_ns": end, "duration_ns": max(0, end - start), "timestamp_ns": end,
                "async_phase": "wait", "handle_index": self.index,
                "async_elapsed_ns": max(0, end - self.launch.get("launch_start_ns", start))})


def communication(op, direction, peer_index, has_input, asynchronous=False):
    def factory(original):
        def wrapper(self, *args, **kwargs):
            if not is_enabled() or not is_pp(self):
                return original(self, *args, **kwargs)
            peer = kwargs.get(direction)
            if peer is None and len(args) > peer_index:
                peer = args[peer_index]
            if peer is None:
                peer = 0 if "broadcast" in op else (self.rank_in_group + (1 if direction == "dst" else -1)) % self.world_size
            tensors = (args[0] if args else kwargs.get("tensor_dict", kwargs.get("tensor"))) if has_input else None
            comm_id = f"pp-async-{os.getpid()}-{next(_sequence)}" if asynchronous else None
            start, result, failed = time.time_ns(), None, True
            try:
                result = original(self, *args, **kwargs)
                failed = False
            finally:
                end = time.time_ns()
                received = result[0] if asynchronous and direction == "src" and result is not None else result
                data = payload(self, op + ("_launch" if asynchronous else ""), start, end,
                               peer, direction, tensors if tensors is not None else received)
                if asynchronous:
                    handles = (result[1] if direction == "src" else result) if result is not None else []
                    data.update(comm_id=comm_id, async_phase="launch", launch_start_ns=start,
                                launch_end_ns=end, handle_count=len(handles or []))
                    if direction == "src":
                        data["postprocess_count"] = len(result[2] or []) if result is not None else 0
                    if failed:
                        data["error"] = True
                TraceSender.emit("pp_comm_span", data)
            if not asynchronous:
                return result
            wrapped = [TracedHandle(handle, data, op + "_wait", index) for index, handle in enumerate(handles or [])]
            return (result[0], wrapped, result[2]) if direction == "src" else wrapped
        return wrapper
    return factory


@register_hook("vllm.distributed.parallel_state")
def patch_parallel_state_pp_comm(module):
    cls = module.GroupCoordinator
    if getattr(cls, "_de_latency_pp_comm_patch_installed", False):
        return
    for method, op, direction, peer_index, has_input, asynchronous in (
        ("send_tensor_dict", "send_tensor_dict", "dst", 1, True, False),
        ("recv_tensor_dict", "recv_tensor_dict", "src", 0, False, False),
        ("broadcast_tensor_dict", "broadcast_tensor_dict", "src", 1, True, False),
        ("send", "send_tensor", "dst", 1, True, False),
        ("recv", "recv_tensor", "src", 2, False, False),
        ("broadcast", "broadcast_tensor", "src", 1, True, False),
        ("isend_tensor_dict", "isend_tensor_dict", "dst", 1, True, True),
        ("irecv_tensor_dict", "irecv_tensor_dict", "src", 0, False, True),
    ):
        apply_method_patch(cls, method, communication(op, direction, peer_index, has_input, asynchronous))
    cls._de_latency_pp_comm_patch_installed = True
