# ruff: noqa
"""Live strategy params: immutable sealed definition emitted locally."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.errors import ArtifactSealError

PARAMS_SNAPSHOT_KEYS: tuple[str, ...] = (
    "SIGNAL_PANEL_WINDOW_DAYS",
    "SIGNAL_REPLAY_WARMUP_DAYS",
    "SIGNAL_RETURN_TAIL_DAYS",
    "SIGNAL_OVERLAP_TOLERANCE",
    "FOLD_PANEL_WARMUP_HOURS",
    "COMMITTEE_PURGE_HOURS",
    "COMMITTEE_OOS_START",
    "PNL_VOL_TARGET_SCALE_FLOOR",
)

STRATEGY_PARAMS_FILENAME: str = "strategy_params.json"
STRATEGY_BOOTSTRAP_FILENAME: str = "strategy_bootstrap.parquet"

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


def capture_params_snapshot() -> dict[str, Any]:
    from src.mhs import params as mhs_params

    snap: dict[str, Any] = {}
    for key in PARAMS_SNAPSHOT_KEYS:
        val = getattr(mhs_params, key)
        if key == "COMMITTEE_OOS_START":
            snap[key] = pd.Timestamp(val).isoformat()
        else:
            snap[key] = val
    return snap


def snapshot_value(params: LiveStrategyParams, key: str) -> Any:
    snap = params.params_snapshot
    if key not in snap:
        raise DataIntegrityError(f"params_snapshot missing key {key!r}")
    raw = snap[key]
    if key == "COMMITTEE_OOS_START":
        try:
            ts = pd.Timestamp(raw)
        except Exception as exc:
            raise DataIntegrityError(f"params_snapshot COMMITTEE_OOS_START invalid: {exc}") from exc
        if ts.tzinfo is None:
            raise DataIntegrityError("COMMITTEE_OOS_START must be tz-aware")
        return ts.tz_convert("UTC")
    return raw


@dataclass(frozen=True, slots=True)
class LiveStrategyParams:
    schema_version: int
    strategy_digest: str
    backtest_window: tuple[pd.Timestamp, pd.Timestamp]
    created_at: pd.Timestamp
    slow_horizon_hours: int
    committee_member_weights: dict[str, float]
    admitted_members: tuple[str, ...]
    growth_budget_target_vol: float
    exposure_cap: float
    growth_envelope: str
    execution_universe_size: int
    pnl_vol_target_mode: str
    deployed_flags: dict[str, Any]
    params_snapshot: dict[str, Any]
    bootstrap_held_row: dict[str, float]


def _canonical_for_digest(params: LiveStrategyParams | dict[str, Any]) -> dict[str, Any]:
    if isinstance(params, LiveStrategyParams):
        data: dict[str, Any] = {
            "schema_version": int(params.schema_version),
            "backtest_window": [
                pd.Timestamp(params.backtest_window[0]).tz_convert("UTC").isoformat(),
                pd.Timestamp(params.backtest_window[1]).tz_convert("UTC").isoformat(),
            ],
            "slow_horizon_hours": int(params.slow_horizon_hours),
            "committee_member_weights": {str(k): float(v) for k, v in sorted(params.committee_member_weights.items())},
            "admitted_members": [str(m) for m in params.admitted_members],
            "growth_budget_target_vol": float(params.growth_budget_target_vol),
            "exposure_cap": float(params.exposure_cap),
            "growth_envelope": str(params.growth_envelope),
            "execution_universe_size": int(params.execution_universe_size),
            "pnl_vol_target_mode": str(params.pnl_vol_target_mode),
            "deployed_flags": {str(k): v for k, v in sorted(params.deployed_flags.items())},
            "params_snapshot": {str(k): v for k, v in sorted((params.params_snapshot or {}).items())},
            "bootstrap_held_row": {str(k): float(v) for k, v in sorted(params.bootstrap_held_row.items())},
        }
    else:
        # dict version (from raw json)
        bw = params.get("backtest_window")
        if isinstance(bw, (list, tuple)) and len(bw) == 2:
            bw_iso = [pd.Timestamp(bw[0]).tz_convert("UTC").isoformat() if pd.Timestamp(bw[0]).tzinfo is not None else pd.Timestamp(bw[0]).tz_localize("UTC").isoformat(), pd.Timestamp(bw[1]).tz_convert("UTC").isoformat() if pd.Timestamp(bw[1]).tzinfo is not None else pd.Timestamp(bw[1]).tz_localize("UTC").isoformat()]
        else:
            bw_iso = bw  # type: ignore[assignment]
        data = {
            "schema_version": params.get("schema_version"),
            "backtest_window": bw_iso,
            "slow_horizon_hours": params.get("slow_horizon_hours"),
            "committee_member_weights": {str(k): float(v) for k, v in sorted((params.get("committee_member_weights") or {}).items())},
            "admitted_members": [str(m) for m in (params.get("admitted_members") or [])],
            "growth_budget_target_vol": float(params.get("growth_budget_target_vol", 0)),
            "exposure_cap": float(params.get("exposure_cap", 0)),
            "growth_envelope": str(params.get("growth_envelope", "")),
            "execution_universe_size": int(params.get("execution_universe_size", 0)),
            "pnl_vol_target_mode": str(params.get("pnl_vol_target_mode", "")),
            "deployed_flags": {str(k): v for k, v in sorted((params.get("deployed_flags") or {}).items())},
            "params_snapshot": {str(k): v for k, v in sorted((params.get("params_snapshot") or {}).items())},
            "bootstrap_held_row": {str(k): float(v) for k, v in sorted((params.get("bootstrap_held_row") or {}).items())},
        }
    return data


def _compute_strategy_digest(params: LiveStrategyParams | dict[str, Any]) -> str:
    canonical = _canonical_for_digest(params)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _serialize_params(params: LiveStrategyParams) -> dict[str, Any]:
    return {
        "schema_version": int(params.schema_version),
        "strategy_digest": str(params.strategy_digest),
        "backtest_window": [
            pd.Timestamp(params.backtest_window[0]).isoformat(),
            pd.Timestamp(params.backtest_window[1]).isoformat(),
        ],
        "created_at": pd.Timestamp(params.created_at).isoformat(),
        "slow_horizon_hours": int(params.slow_horizon_hours),
        "committee_member_weights": {str(k): float(v) for k, v in params.committee_member_weights.items()},
        "admitted_members": [str(m) for m in params.admitted_members],
        "growth_budget_target_vol": float(params.growth_budget_target_vol),
        "exposure_cap": float(params.exposure_cap),
        "growth_envelope": str(params.growth_envelope),
        "execution_universe_size": int(params.execution_universe_size),
        "pnl_vol_target_mode": str(params.pnl_vol_target_mode),
        "deployed_flags": dict(params.deployed_flags),
        "params_snapshot": dict(params.params_snapshot) if params.params_snapshot else {},
        "bootstrap_held_row": {str(k): float(v) for k, v in params.bootstrap_held_row.items()},
    }


def _deserialize_params(raw: dict[str, Any]) -> LiveStrategyParams:
    if not isinstance(raw, dict):
        raise DataIntegrityError("strategy params must be a JSON object")
    schema = raw.get("schema_version")
    digest = raw.get("strategy_digest")
    if not isinstance(digest, str) or not digest:
        raise DataIntegrityError("strategy params missing strategy_digest")
    bw_raw = raw.get("backtest_window")
    if not isinstance(bw_raw, (list, tuple)) or len(bw_raw) != 2:
        raise DataIntegrityError("backtest_window must be 2-element list")
    try:
        bw0 = pd.Timestamp(bw_raw[0])
        bw1 = pd.Timestamp(bw_raw[1])
    except Exception as exc:
        raise DataIntegrityError(f"backtest_window invalid: {exc}") from exc
    for ts in (bw0, bw1):
        if ts.tzinfo is None:
            raise DataIntegrityError("backtest_window must be tz-aware")
    bw0 = bw0.tz_convert("UTC")
    bw1 = bw1.tz_convert("UTC")
    created_raw = raw.get("created_at")
    if not isinstance(created_raw, str):
        raise DataIntegrityError("missing created_at")
    try:
        created_at = pd.Timestamp(created_raw)
    except Exception as exc:
        raise DataIntegrityError(f"created_at invalid: {exc}") from exc
    if created_at.tzinfo is None:
        raise DataIntegrityError("created_at must be tz-aware")
    created_at = created_at.tz_convert("UTC")
    # compute digest verification before constructing object (using raw dict)
    expected = _compute_strategy_digest(raw)
    if expected != digest:
        raise DataIntegrityError(f"strategy_digest mismatch: expected {expected!r} got {digest!r}")
    return LiveStrategyParams(
        schema_version=int(schema) if schema is not None else 1,
        strategy_digest=str(digest),
        backtest_window=(bw0, bw1),
        created_at=created_at,
        slow_horizon_hours=int(raw["slow_horizon_hours"]),
        committee_member_weights={str(k): float(v) for k, v in dict(raw.get("committee_member_weights") or {}).items()},
        admitted_members=tuple(str(m) for m in raw.get("admitted_members") or []),
        growth_budget_target_vol=float(raw["growth_budget_target_vol"]),
        exposure_cap=float(raw["exposure_cap"]),
        growth_envelope=str(raw["growth_envelope"]),
        execution_universe_size=int(raw["execution_universe_size"]),
        pnl_vol_target_mode=str(raw["pnl_vol_target_mode"]),
        deployed_flags=dict(raw.get("deployed_flags") or {}),
        params_snapshot=dict(raw.get("params_snapshot") or {}),
        bootstrap_held_row={str(k): float(v) for k, v in dict(raw.get("bootstrap_held_row") or {}).items()},
    )


def save_strategy_params(path: Path, params: LiveStrategyParams, *, artifact_key: SecretStr | None = None) -> Path:
    path = Path(path)
    # compute digest deterministically (excluding strategy_digest and created_at)
    digest = _compute_strategy_digest(params)
    # create params with computed digest
    params_with_digest = LiveStrategyParams(
        schema_version=params.schema_version,
        strategy_digest=digest,
        backtest_window=params.backtest_window,
        created_at=params.created_at,
        slow_horizon_hours=params.slow_horizon_hours,
        committee_member_weights=dict(params.committee_member_weights),
        admitted_members=tuple(params.admitted_members),
        growth_budget_target_vol=float(params.growth_budget_target_vol),
        exposure_cap=float(params.exposure_cap),
        growth_envelope=str(params.growth_envelope),
        execution_universe_size=int(params.execution_universe_size),
        pnl_vol_target_mode=str(params.pnl_vol_target_mode),
        deployed_flags=dict(params.deployed_flags),
        params_snapshot=dict(params.params_snapshot) if params.params_snapshot else {},
        bootstrap_held_row=dict(params.bootstrap_held_row),
    )
    payload = _serialize_params(params_with_digest)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if artifact_key is not None:
        from src.live.crypto import derive_key, seal_bytes

        dest = path if str(path).endswith(".enc") else Path(str(path) + ".enc")
        dest.parent.mkdir(parents=True, exist_ok=True)
        sealed = seal_bytes(data, derive_key(artifact_key))
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(sealed)
        os.replace(tmp, dest)
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def load_strategy_params(path: Path, *, artifact_key: SecretStr | None = None) -> LiveStrategyParams:
    path = Path(path)
    is_enc = str(path).endswith(".enc")
    if is_enc:
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {path}")
        if not path.exists():
            raise DataIntegrityError(f"strategy params not found: {path}")
        try:
            from src.live.crypto import derive_key, open_bytes

            blob = path.read_bytes()
            plain = open_bytes(blob, derive_key(artifact_key))
            raw = json.loads(plain.decode("utf-8"))
        except ArtifactSealError:
            raise
        except Exception as exc:
            raise DataIntegrityError(f"strategy params file corrupt: {path}: {exc}") from exc
    else:
        if not path.exists():
            if artifact_key is not None:
                enc = Path(str(path) + ".enc")
                if enc.exists():
                    return load_strategy_params(enc, artifact_key=artifact_key)
            raise DataIntegrityError(f"strategy params not found: {path}")
        try:
            raw_text = path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"strategy params file corrupt: {path}") from exc
        except Exception as exc:
            raise DataIntegrityError(f"strategy params file corrupt: {path}: {exc}") from exc
    if isinstance(raw, bytes):
        raise DataIntegrityError(f"strategy params file corrupt: {path}")
    return _deserialize_params(raw)


def assert_deployment_eligible(report: Any, *, reference_report_path: Path | None = None) -> None:
    if getattr(report, "status", None) != "OK":
        raise DataIntegrityError("deployment ineligible: report status not OK")
    rg = getattr(report, "research_go", None)
    if rg is None or not getattr(rg, "eligible", False):
        raise DataIntegrityError("deployment ineligible: research_go not eligible")
    blend = getattr(report, "blend", None)
    if blend is None:
        raise DataIntegrityError("deployment ineligible: blend is None")
    tw = getattr(blend, "target_weights", None)
    if tw is None or (hasattr(tw, "empty") and tw.empty) or (hasattr(tw, "__len__") and len(tw) == 0):
        raise DataIntegrityError("deployment ineligible: blend target_weights empty")
    if reference_report_path is not None:
        ref_path = Path(reference_report_path)
        if ref_path.exists():
            try:
                ref_raw = json.loads(ref_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DataIntegrityError(f"reference report corrupt: {exc}") from exc
            ref_flags = ref_raw.get("flags")
            if ref_flags is not None:
                # cur flags from report if available
                cur_flags = getattr(report, "flags", None)
                if cur_flags is None:
                    cur_flags = {}
                    # attempt to derive from deployed_flags if report has it? fallback empty
                cur_payload = json.dumps(cur_flags, sort_keys=True, separators=(",", ":"), default=str)
                ref_payload = json.dumps(ref_flags, sort_keys=True, separators=(",", ":"), default=str)
                cur_digest = hashlib.sha256(cur_payload.encode("utf-8")).hexdigest()
                ref_digest = hashlib.sha256(ref_payload.encode("utf-8")).hexdigest()
                if cur_digest != ref_digest:
                    raise DataIntegrityError("deployment ineligible: flags digest drift")
