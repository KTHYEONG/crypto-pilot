"""Small causal alignment helper used by the strategy data adapter."""

from __future__ import annotations

from typing import Any


def compute_multi_alignment_info(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    embargo: int,
) -> dict[str, Any] | None:
    """Return common start offsets and effective length for aligned panels."""
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    starts: dict[str, Any] = {}
    for symbol in symbols:
        frame = data_maps.get(symbol, {}).get(tf)
        if frame is None or frame.empty or "datetime" not in frame.columns:
            continue
        starts[symbol] = frame["datetime"].iloc[0]
    if not starts:
        return None
    common_start = max(starts.values())
    offsets: dict[str, int] = {}
    lengths: list[int] = []
    for symbol in symbols:
        frame = data_maps.get(symbol, {}).get(tf)
        if frame is None or frame.empty or "datetime" not in frame.columns:
            continue
        offset = int(frame["datetime"].searchsorted(common_start)) + embargo
        if offset < len(frame):
            offsets[symbol] = offset
            lengths.append(len(frame) - offset)
    if not lengths or min(lengths) < 200:
        return None
    return {
        "common_is_start_dt": common_start,
        "alignment_offsets": offsets,
        "eff_ref_len": min(lengths),
    }
