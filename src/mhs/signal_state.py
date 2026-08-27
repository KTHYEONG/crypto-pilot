"""Signal state persistence: frozen params + path-dependent state.

Atomic persistence mirrors src.live.ledger.save_ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.errors import ArtifactSealError

SCHEMA_VERSION: int = 1

BOUND_FLAGS: frozenset[str] = frozenset(
    {
        "committee_capital",
        "committee_evidence_weighting",
        "committee_kelly_sizing",
        "committee_member_set",
        "committee_regime_adaptive_tranche",
        "committee_tranche_smoothing",
        "committee_target_gross",
        "beta_neutralize",
        "trend_sleeve",
        "trend_sleeve_gross",
        "trend_efficiency_overlay",
        "rebalance_filter",
        "fast_book_mode",
        "slow_book_mode",
        "ensemble_signal",
        "execution_universe_size",
        "funding_carry_sleeve",
        "funding_carry_weight",
        "pnl_vol_target_mode",
        "growth_envelope",
        "exposure_scale_two_sided",
        "exposure_drawdown_brake",
        "fill_mark_parity_gate",
    }
)


@dataclass(frozen=True, slots=True)
class FrozenSignalParams:
    slow_horizon_hours: int
    committee_member_weights: dict[str, float]
    admitted_members: tuple[str, ...]
    growth_budget_target_vol: float
    exposure_cap: float
    growth_envelope: str
    execution_universe_size: int
    pnl_vol_target_mode: str
    # 배포 시점 실제 request 의 BOUND_FLAGS 해석값 스냅샷(단, committee_target_gross
    # 는 sentinel 이 아니라 _resolved_committee_target_gross 해석값). refresh_signal_row
    # 가 MhsDiagnosticRequest 를 재구성할 때 이 dict 를 그대로 풀어 넣어, 기본값 추정에
    # 의존하지 않고 배포 당시와 동일한 유효 설정을 재현한다.
    deployed_flags: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SignalState:
    schema_version: int
    params_digest: str
    flags_digest: str
    frozen: FrozenSignalParams
    last_decision_time: pd.Timestamp
    held_target_row: dict[str, float]
    reference_daily_returns: pd.Series


def compute_params_digest() -> str:
    from src.mhs.params import (
        COMMITTEE_MEMBER_SETS,
        COMMITTEE_OOS_START,
        COMMITTEE_TARGET_GROSS,
        COMMITTEE_TARGET_VOL,
        EXECUTION_ROSTER_EXIT_MULTIPLIER,
        GROWTH_RISK_ENVELOPES,
        PNL_VOL_TARGET_BURN_IN_DAYS,
        PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
        PNL_VOL_TARGET_MAX_SCALE,
        PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS,
        PNL_VOL_TARGET_SCALE_FLOOR,
        PNL_VOL_TARGET_WINDOW_DAYS,
        REBALANCE_DEADBAND_POSITION_FRACTION,
        REGIME_CASH_MEDIAN_WINDOW_HOURS,
        REGIME_CASH_SCALE_FLOOR,
        SIGNAL_PANEL_WINDOW_DAYS,
        SIGNAL_RETURN_TAIL_DAYS,
    )

    raw: dict[str, Any] = {
        "COMMITTEE_MEMBER_SETS": {k: list(v) for k, v in sorted(COMMITTEE_MEMBER_SETS.items())},
        "COMMITTEE_OOS_START": COMMITTEE_OOS_START.isoformat(),
        "COMMITTEE_TARGET_GROSS": COMMITTEE_TARGET_GROSS,
        "COMMITTEE_TARGET_VOL": COMMITTEE_TARGET_VOL,
        "EXECUTION_ROSTER_EXIT_MULTIPLIER": EXECUTION_ROSTER_EXIT_MULTIPLIER,
        "GROWTH_RISK_ENVELOPES": {
            k: {
                "horizon_years": float(v.horizon_years),
                "leverage_ceiling": float(v.leverage_ceiling),
                "max_drawdown": float(v.max_drawdown),
                "max_drawdown_prob": float(v.max_drawdown_prob),
                "max_ruin_prob": float(v.max_ruin_prob),
                "ruin_fraction": float(v.ruin_fraction),
            }
            for k, v in sorted(GROWTH_RISK_ENVELOPES.items())
        },
        "PNL_VOL_TARGET_BURN_IN_DAYS": PNL_VOL_TARGET_BURN_IN_DAYS,
        "PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS": PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
        "PNL_VOL_TARGET_MAX_SCALE": PNL_VOL_TARGET_MAX_SCALE,
        "PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS": PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS,
        "PNL_VOL_TARGET_SCALE_FLOOR": PNL_VOL_TARGET_SCALE_FLOOR,
        "PNL_VOL_TARGET_WINDOW_DAYS": PNL_VOL_TARGET_WINDOW_DAYS,
        "REBALANCE_DEADBAND_POSITION_FRACTION": REBALANCE_DEADBAND_POSITION_FRACTION,
        "REGIME_CASH_MEDIAN_WINDOW_HOURS": REGIME_CASH_MEDIAN_WINDOW_HOURS,
        "REGIME_CASH_SCALE_FLOOR": REGIME_CASH_SCALE_FLOOR,
        "SIGNAL_PANEL_WINDOW_DAYS": SIGNAL_PANEL_WINDOW_DAYS,
        "SIGNAL_RETURN_TAIL_DAYS": SIGNAL_RETURN_TAIL_DAYS,
    }
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_flags_digest(flags: Mapping[str, Any]) -> str:
    filtered = {k: flags[k] for k in sorted(flags) if k in BOUND_FLAGS}
    # Use json with sorted keys for determinism; default=str for non-serializable
    # but flags values are primitives.
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (set, tuple, frozenset)):
            return sorted(obj) if isinstance(obj, set) else list(obj)
        return str(obj)

    payload = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _serialize_state(state: SignalState) -> dict[str, Any]:
    # Reference series ordered iso -> float
    ref = state.reference_daily_returns
    # Ensure sorted by index
    if not ref.empty:
        ref = ref.sort_index()
        ref_dict = {pd.Timestamp(k).isoformat(): float(v) for k, v in ref.items()}
    else:
        ref_dict = {}
    return {
        "schema_version": int(state.schema_version),
        "params_digest": str(state.params_digest),
        "flags_digest": str(state.flags_digest),
        "frozen": {
            "slow_horizon_hours": int(state.frozen.slow_horizon_hours),
            "committee_member_weights": {str(k): float(v) for k, v in state.frozen.committee_member_weights.items()},
            "admitted_members": [str(m) for m in state.frozen.admitted_members],
            "growth_budget_target_vol": float(state.frozen.growth_budget_target_vol),
            "exposure_cap": float(state.frozen.exposure_cap),
            "growth_envelope": str(state.frozen.growth_envelope),
            "execution_universe_size": int(state.frozen.execution_universe_size),
            "pnl_vol_target_mode": str(state.frozen.pnl_vol_target_mode),
            "deployed_flags": dict(state.frozen.deployed_flags),
        },
        "last_decision_time": state.last_decision_time.isoformat(),
        "held_target_row": {str(k): float(v) for k, v in state.held_target_row.items()},
        "reference_daily_returns": ref_dict,
    }


def _deserialize_state(raw: dict[str, Any]) -> SignalState:
    if not isinstance(raw, dict):
        raise DataIntegrityError("signal state must be a JSON object")
    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise DataIntegrityError(f"unknown schema_version {schema!r}")
    params_digest = raw.get("params_digest")
    flags_digest = raw.get("flags_digest")
    if not isinstance(params_digest, str) or not isinstance(flags_digest, str):
        raise DataIntegrityError("signal state missing digests")
    frozen_raw = raw.get("frozen")
    if not isinstance(frozen_raw, dict):
        raise DataIntegrityError("signal state missing frozen")
    try:
        frozen = FrozenSignalParams(
            slow_horizon_hours=int(frozen_raw["slow_horizon_hours"]),
            committee_member_weights={str(k): float(v) for k, v in dict(frozen_raw.get("committee_member_weights") or {}).items()},
            admitted_members=tuple(str(m) for m in frozen_raw.get("admitted_members") or []),
            growth_budget_target_vol=float(frozen_raw["growth_budget_target_vol"]),
            exposure_cap=float(frozen_raw["exposure_cap"]),
            growth_envelope=str(frozen_raw["growth_envelope"]),
            execution_universe_size=int(frozen_raw["execution_universe_size"]),
            pnl_vol_target_mode=str(frozen_raw["pnl_vol_target_mode"]),
            deployed_flags=dict(frozen_raw.get("deployed_flags") or {}),
        )
    except Exception as exc:
        raise DataIntegrityError(f"signal state frozen field invalid: {exc}") from exc

    last_raw = raw.get("last_decision_time")
    if not isinstance(last_raw, str):
        raise DataIntegrityError("signal state missing last_decision_time")
    try:
        last_ts = pd.Timestamp(last_raw)
    except Exception as exc:
        raise DataIntegrityError(f"signal state last_decision_time invalid: {exc}") from exc
    if last_ts.tzinfo is None:
        raise DataIntegrityError("last_decision_time must be tz-aware")
    last_ts = last_ts.tz_convert("UTC")

    held_raw = raw.get("held_target_row")
    if not isinstance(held_raw, dict):
        raise DataIntegrityError("signal state held_target_row must be an object")
    held: dict[str, float] = {}
    for k, v in held_raw.items():
        if isinstance(v, bool):
            # bool is a numeric subtype in Python; float(True) would silently succeed.
            raise DataIntegrityError(f"signal state held_target_row entry non-numeric {k!r}")
        try:
            held[str(k)] = float(v)
        except Exception as exc:
            raise DataIntegrityError(f"signal state held_target_row entry non-numeric {k!r}") from exc
    ref_raw = raw.get("reference_daily_returns")
    if not isinstance(ref_raw, dict):
        raise DataIntegrityError("signal state reference_daily_returns must be an object")
    # Build series
    if ref_raw:
        idx: list[pd.Timestamp] = []
        vals: list[float] = []
        for iso, val in ref_raw.items():
            try:
                ts = pd.Timestamp(iso)
            except Exception as exc:
                raise DataIntegrityError(f"signal state return timestamp invalid {iso!r}") from exc
            if ts.tzinfo is None:
                raise DataIntegrityError("reference_daily_returns index must be tz-aware")
            ts = ts.tz_convert("UTC")
            try:
                fval = float(val)
            except Exception as exc:
                raise DataIntegrityError(f"signal state return entry non-numeric {iso!r}") from exc
            if isinstance(val, bool):
                raise DataIntegrityError(f"signal state return entry non-numeric {iso!r}")
            idx.append(ts)
            vals.append(fval)
        # Preserve insertion order as stored (which is sorted)
        series = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype="float64")
        # Ensure sorted for consistency
        series = series.sort_index()
    else:
        series = pd.Series(dtype="float64")

    return SignalState(
        schema_version=int(schema),
        params_digest=str(params_digest),
        flags_digest=str(flags_digest),
        frozen=frozen,
        last_decision_time=last_ts,
        held_target_row=held,
        reference_daily_returns=series,
    )


def save_signal_state(path: Path, state: SignalState, *, artifact_key: SecretStr | None = None) -> Path:
    path = Path(path)
    payload = _serialize_state(state)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if artifact_key is not None:
        from src.live.crypto import derive_key, seal_bytes

        dest = path if str(path).endswith(".enc") else Path(str(path) + ".enc")
        dest.parent.mkdir(parents=True, exist_ok=True)
        sealed = seal_bytes(data, derive_key(artifact_key))
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(sealed)
        os.replace(tmp, dest)
        return dest
    # plaintext
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def load_signal_state(path: Path, *, artifact_key: SecretStr | None = None) -> SignalState:
    path = Path(path)
    is_enc = str(path).endswith(".enc")
    if is_enc:
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {path}")
        if not path.exists():
            raise DataIntegrityError(f"signal state file missing: {path}")
        try:
            from src.live.crypto import derive_key, open_bytes

            blob = path.read_bytes()
            plain = open_bytes(blob, derive_key(artifact_key))
            raw = json.loads(plain.decode("utf-8"))
        except ArtifactSealError:
            raise
        except Exception as exc:
            raise DataIntegrityError(f"signal state file corrupt: {path}: {exc}") from exc
    else:
        if not path.exists():
            # Also try .enc variant if artifact_key provided? For convenience, check sibling
            if artifact_key is not None:
                enc_candidate = Path(str(path) + ".enc")
                if enc_candidate.exists():
                    return load_signal_state(enc_candidate, artifact_key=artifact_key)
            raise DataIntegrityError(f"signal state file missing: {path}")
        try:
            raw_text = path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"signal state file corrupt: {path}") from exc
        except Exception as exc:
            raise DataIntegrityError(f"signal state file corrupt: {path}: {exc}") from exc

    if isinstance(raw, bytes):
        raise DataIntegrityError(f"signal state file corrupt: {path}")
    return _deserialize_state(raw)


def assert_state_binding(state: SignalState, flags: Mapping[str, Any]) -> None:
    current_params = compute_params_digest()
    if current_params != state.params_digest:
        raise DataIntegrityError(f"params_digest mismatch: expected {state.params_digest!r} got {current_params!r}")
    current_flags = compute_flags_digest(flags)
    if current_flags != state.flags_digest:
        raise DataIntegrityError(f"flags_digest mismatch: expected {state.flags_digest!r} got {current_flags!r}")
