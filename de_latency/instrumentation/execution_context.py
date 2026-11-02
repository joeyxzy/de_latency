"""Nested, thread-local worker dispatch context shared by phase/PP hooks."""
import threading

_local = threading.local()


def push(context):
    if not hasattr(_local, "stack"):
        _local.stack = []
    _local.stack.append(context)


def pop():
    stack = getattr(_local, "stack", None)
    return stack.pop() if stack else None


def current():
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None
