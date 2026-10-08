"""Shared execution-quality shard reader for live recording and lab evidence.

The loader is pure parquet reads with no live dependency, so both the live
daemon (recording side) and the research lab (forward-evidence side) share it
without the lab importing live.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_all_frames(history_dir: Path) -> pd.DataFrame | None:
    shards = sorted(history_dir.glob("*.parquet"))
    if not shards:
        return None
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: S112
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_execution_quality_records(history_dir: Path | str) -> pd.DataFrame:
    """Load every execution-quality shard into one frame (forward evidence).

    Legacy shards without ``strategy_digest``/``observed_at`` load with nulls
    (preserved, excluded from certification) instead of failing.
    """
    loaded = _load_all_frames(Path(history_dir))
    frame = pd.DataFrame() if loaded is None else loaded.copy()
    frame["strategy_digest"] = frame.get("strategy_digest", None)
    frame["observed_at"] = pd.to_datetime(frame.get("observed_at", pd.NaT), utc=True, errors="coerce")
    if "decision_time" in frame.columns:
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    return frame
