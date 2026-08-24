"""MHS 배포 목표비중 아티팩트 소비와 인과성 게이트.

I-SIGNAL-FIDELITY: 이 모듈은 알파를 재계산하지 않고 parquet 산출물만 읽는다.
I-CAUSAL-EXEC: 결정 시각 T의 신호는 kline open-time 규약상 T+1h에야 관측 가능하다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.errors import CausalityViolation

_SIGNAL_LAG = pd.Timedelta(hours=1)


def _as_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        raise ValueError("timestamp must be tz-aware UTC")
    return ts.tz_convert("UTC")


def latest_target_weights(artifact_path: Path, decision_time: pd.Timestamp) -> pd.Series:
    """deployed_target_weights.parquet에서 정확히 decision_time 행을 읽는다."""
    decision_ts = _as_utc(decision_time)
    if (decision_ts.hour, decision_ts.minute, decision_ts.second) != (0, 0, 0):
        raise ValueError("decision_time must lie on the 24h grid (00:00 UTC)")
    frame = pd.read_parquet(artifact_path)
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise DataIntegrityError("target weights index must be tz-aware UTC")
    if decision_ts not in index:
        raise DataIntegrityError(
            f"decision_time {decision_ts} not present in target weights artifact"
        )
    row = frame.loc[decision_ts]
    return pd.Series(row, index=frame.columns, dtype="float64", name=decision_ts)


def assert_signal_available(decision_time: pd.Timestamp, now: pd.Timestamp) -> None:
    """now < decision_time + 1h 이면 look-ahead 위반이다."""
    decision_ts = _as_utc(decision_time)
    now_ts = _as_utc(now)
    if now_ts < decision_ts + _SIGNAL_LAG:
        raise CausalityViolation(
            f"orders for {decision_ts} cannot be created before "
            f"{decision_ts + _SIGNAL_LAG}; got now={now_ts}"
        )
