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


def latest_decision_marks(
    artifact_path: Path,
    decision_time: pd.Timestamp,
    *,
    artifact_key: SecretStr | None = None,
) -> pd.Series | None:
    """deployed_decision_marks 아티팩트에서 decision_time 행을 읽는다."""
    decision_ts = _as_utc(decision_time)
    base = Path(artifact_path)
    if "deployed_target_weights" not in base.name:
        return None
    # 파일명만 치환: deployed_target_weights -> deployed_decision_marks
    marks_name = base.name.replace("deployed_target_weights", "deployed_decision_marks")
    marks_path = base.parent / marks_name
    # handle .enc vs plain: check both existence
    candidate: Path | None = None
    if marks_path.exists():
        candidate = marks_path
    elif artifact_key is not None:
        # try .enc variant if not already
        enc = marks_path if str(marks_path).endswith(".enc") else Path(f"{marks_path}.enc")
        if enc.exists():
            candidate = enc
        else:
            # also try direct marks_path as enc when base was .enc but name replacement lost .enc?
            pass
    if candidate is None:
        # also check if plain path with .enc exists even without key? fallback
        enc_alt = marks_path if str(marks_path).endswith(".enc") else Path(f"{marks_path}.enc")
        if enc_alt.exists():
            candidate = enc_alt
        else:
            return None
    if candidate is None:
        return None
    if str(candidate).endswith(".enc"):
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {candidate}")
        frame = read_sealed_parquet(candidate, derive_key(artifact_key))
    else:
        try:
            frame = pd.read_parquet(candidate)
        except (FileNotFoundError, OSError) as exc:
            raise DataIntegrityError(f"decision marks artifact missing: {candidate}") from exc
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise DataIntegrityError("decision marks index must be tz-aware UTC")
    # normalize decision_ts to match index tz
    if decision_ts not in index:
        raise DataIntegrityError(
            f"decision_time {decision_ts} not present in decision marks artifact"
        )
    row = frame.loc[decision_ts]
    return pd.Series(row, index=frame.columns, dtype="float64", name=decision_ts)


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
