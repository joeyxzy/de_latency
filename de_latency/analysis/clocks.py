"""Integer timestamps are ns; legacy EngineCore float timestamps are seconds."""
from decimal import Decimal, InvalidOperation
import math


def to_ns(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value <= 0:
            return None
        return int(value if value > 1e14 else value * 1e9)
    try:
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            number = int(value)
            return number if number > 0 else None
        number = Decimal(str(value))
        if not number.is_finite() or number <= 0:
            return None
        return int(number if number > Decimal("1e14") else number * 1_000_000_000)
    except (ValueError, TypeError, InvalidOperation):
        return None


def to_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def median_int(values):
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) // 2
