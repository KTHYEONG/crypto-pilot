"""MHS 배포 목표비중 아티팩트 소비와 인과성 게이트.

I-SIGNAL-FIDELITY: 이 모듈은 알파를 재계산하지 않고 parquet 산출물만 읽는다.
I-CAUSAL-EXEC: 결정 시각 T의 신호는 kline open-time 규약상 T+1h에야 관측 가능하다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.crypto import derive_key, read_sealed_parquet
from src.live.errors import ArtifactSealError, CausalityViolation, StaleSignalError

_SIGNAL_LAG = pd.Timedelta(hours=1)


def _as_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        raise ValueError("timestamp must be tz-aware UTC")
    return ts.tz_convert("UTC")


def latest_target_weights(
    artifact_path: Path,
    decision_time: pd.Timestamp,
    *,
    artifact_key: SecretStr | None = None,
) -> pd.Series:
    """deployed_target_weights.parquet(.enc)에서 정확히 decision_time 행을 읽는다."""
    decision_ts = _as_utc(decision_time)
    if (decision_ts.hour, decision_ts.minute, decision_ts.second) != (0, 0, 0):
        raise ValueError("decision_time must lie on the 24h grid (00:00 UTC)")
    if artifact_path.suffix == ".enc":
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {artifact_path}")
        frame = read_sealed_parquet(artifact_path, derive_key(artifact_key))
    else:
        try:
            frame = pd.read_parquet(artifact_path)
        except (FileNotFoundError, OSError) as exc:
            # 데몬 bare except 가 삼키지 않도록 계약 예외로 승격한다.
            raise DataIntegrityError(f"target weights artifact missing: {artifact_path}") from exc
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


def assert_signal_fresh(
    decision_time: pd.Timestamp, now: pd.Timestamp, max_staleness: pd.Timedelta
) -> None:
    """now > decision_time + max_staleness 이면 StaleSignalError(하한 게이트의 대칭 상한)."""
    decision_ts = _as_utc(decision_time)
    now_ts = _as_utc(now)
    if now_ts > decision_ts + max_staleness:
        raise StaleSignalError(
            f"signal for {decision_ts} is stale: now={now_ts} "
            f"max_staleness={max_staleness}"
        )
