from __future__ import annotations


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    result: dict[str, dict[str, float | bool]] = {}
    still_rejecting = True
    for index, (name, p_value) in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        reject = still_rejecting and p_value <= threshold
        still_rejecting = still_rejecting and reject
        result[name] = {"p_value": p_value, "holm_threshold": threshold, "reject": reject}
    return result
