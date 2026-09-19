"""MHS 배포 목표비중 아티팩트 소비와 인과성 게이트.

I-SIGNAL-FIDELITY: 이 모듈은 알파를 재계산하지 않고 parquet 산출물만 읽는다.
I-CAUSAL-EXEC: 결정 시각 T의 신호는 kline open-time 규약상 T+1h에야 관측 가능하다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    weights_path: Path,
    decision_time: pd.Timestamp,
    *,
    artifact_key: SecretStr | None = None,
    artifact_path: Path | None = None,
    max_staleness: pd.Timedelta | None = None,
) -> pd.Series:
    """deployed_target_weights.parquet(.enc)에서 가장 최근 행을 읽는다 (weights_asof)."""
    if artifact_path is not None:
        weights_path = artifact_path
    decision_ts = _as_utc(decision_time)
    if (decision_ts.hour, decision_ts.minute, decision_ts.second) != (0, 0, 0):
        raise ValueError("decision_time must lie on the 24h grid (00:00 UTC)")
    from src.live.deployed_weights import load_weights_frame, weights_asof

    frame = load_weights_frame(Path(weights_path), artifact_key=artifact_key)
    if frame.empty:
        raise DataIntegrityError(f"target weights artifact missing: {weights_path}")
    return weights_asof(load_weights_frame(Path(weights_path), artifact_key=artifact_key), decision_ts, max_staleness=max_staleness or pd.Timedelta(hours=96))
    # wiring: return weights_asof(load_weights_frame(Path(weights_path), artifact_key=artifact_key), decision_ts, max_staleness=max_staleness or pd.Timedelta(hours=96))


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


def latest_decision_ohlcv_close(
    artifact_path: Path,
    decision_time: pd.Timestamp,
    *,
    artifact_key: SecretStr | None = None,
) -> pd.Series:
    """Load the source-labelled completed OHLCV decision close for exact-date sizing. Missing, empty, malformed or stale anchors are data-integrity failures, never permission to use a different price source."""
    decision_ts = _as_utc(decision_time)
    base = Path(artifact_path)
    if "deployed_target_weights" not in base.name:
        raise DataIntegrityError(f"weights path missing token 'deployed_target_weights': {base}")
    close_path = base.parent / base.name.replace("deployed_target_weights", "deployed_decision_ohlcv_close")
    candidate: Path | None = None
    if close_path.exists():
        candidate = close_path
    else:
        enc = close_path if str(close_path).endswith(".enc") else Path(f"{close_path}.enc")
        if enc.exists():
            candidate = enc
    if candidate is None:
        raise DataIntegrityError(f"decision OHLCV close artifact missing: {close_path}")
    if str(candidate).endswith(".enc"):
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {candidate}")
        frame = read_sealed_parquet(candidate, derive_key(artifact_key))
    else:
        try:
            frame = pd.read_parquet(candidate)
        except (OSError, ValueError) as exc:
            raise DataIntegrityError(f"decision OHLCV close artifact unreadable: {candidate}") from exc
    if frame.empty:
        raise DataIntegrityError(f"decision OHLCV close artifact empty: {candidate}")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise DataIntegrityError("decision OHLCV close index must be tz-aware UTC")
    if decision_ts not in index:
        raise DataIntegrityError(f"decision_time {decision_ts} not present in decision OHLCV close artifact")
    row = frame.loc[decision_ts]
    if not isinstance(row, pd.Series):
        raise DataIntegrityError(f"decision OHLCV close has duplicate rows for {decision_ts}")
    vals = pd.to_numeric(row, errors="coerce").astype("float64")
    present = vals.dropna()
    if present.empty:
        raise DataIntegrityError(f"decision OHLCV close invalid for {decision_ts}: no symbols")
    if bool((~np.isfinite(present.to_numpy())).any()) or bool((present.to_numpy() <= 0).any()):
        raise DataIntegrityError(f"decision OHLCV close invalid for {decision_ts}")
    return pd.Series(present.to_numpy(dtype="float64"), index=present.index, dtype="float64", name=decision_ts)
