# ruff: noqa
"""Live strategy params: immutable sealed definition emitted locally (schema v2)."""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.errors import ArtifactSealError
from src.mhs.deployment_policy import MhsDeploymentPolicy, SignalWindowPolicy, SizingPolicy, TargetWeightPolicy

PARAMS_SNAPSHOT_KEYS: tuple[str, ...] = (
    "SIGNAL_PANEL_WINDOW_DAYS",
    "SIGNAL_REPLAY_WARMUP_DAYS",
    "SIGNAL_RETURN_TAIL_DAYS",
    "SIGNAL_OVERLAP_TOLERANCE",
    "FOLD_PANEL_WARMUP_HOURS",
    "COMMITTEE_PURGE_HOURS",
    "COMMITTEE_OOS_START",
    "PNL_VOL_TARGET_SCALE_FLOOR",
    "COMMITTEE_KELLY_WINDOW_DAYS",
    "COMMITTEE_KELLY_FRACTION",
    "COMMITTEE_KELLY_LCB_Z",
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

_ALLOWED_PARAMS_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "strategy_digest",
        "backtest_window",
        "created_at",
        "policy",
        "bootstrap_sha256",
        "bootstrap_held_row",
    }
)

_ALLOWED_POLICY_KEYS: frozenset[str] = frozenset(
    {
        "target_weights",
        "sizing",
        "signal_window",
        "slow_horizon_hours",
        "committee_member_weights",
        "admitted_members",
    }
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


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


@dataclass(frozen=True, slots=True)
class LiveStrategyParams:
    schema_version: int
    strategy_digest: str
    backtest_window: tuple[pd.Timestamp, pd.Timestamp]
    created_at: pd.Timestamp
    policy: MhsDeploymentPolicy
    bootstrap_sha256: str
    bootstrap_held_row: dict[str, float]


def _canonical_policy(policy: MhsDeploymentPolicy) -> dict[str, Any]:
    tw = policy.target_weights
    tw_dict: dict[str, Any] = {
        "beta_neutralize": bool(tw.beta_neutralize),
        "committee_capital": bool(tw.committee_capital),
        "committee_member_set": str(tw.committee_member_set),
        "committee_regime_adaptive_tranche": bool(tw.committee_regime_adaptive_tranche),
        "committee_target_gross": None if tw.committee_target_gross is None else float(tw.committee_target_gross),
        "committee_tranche_smoothing": bool(tw.committee_tranche_smoothing),
        "crash_regime_tilt_alpha": None if tw.crash_regime_tilt_alpha is None else float(tw.crash_regime_tilt_alpha),
        "ensemble_signal": str(tw.ensemble_signal),
        "execution_timeframe": str(tw.execution_timeframe),
        "execution_universe_size": int(tw.execution_universe_size),
        "fast_book_mode": str(tw.fast_book_mode),
        "fill_mark_parity_gate": bool(tw.fill_mark_parity_gate),
        "funding_carry_sleeve": bool(tw.funding_carry_sleeve),
        "funding_carry_weight": float(tw.funding_carry_weight),
        "rebalance_filter": str(tw.rebalance_filter),
        "slow_book_mode": str(tw.slow_book_mode),
        "trend_efficiency_overlay": bool(tw.trend_efficiency_overlay),
        "trend_sleeve": bool(tw.trend_sleeve),
        "trend_sleeve_gross": float(tw.trend_sleeve_gross),
    }
    sz = policy.sizing
    sz_dict: dict[str, Any] = {
        "drawdown_brake": bool(sz.drawdown_brake),
        "exposure_cap": float(sz.exposure_cap),
        "kelly_blend_weight": float(sz.kelly_blend_weight),
        "kelly_enabled": bool(sz.kelly_enabled),
        "kelly_fraction": float(sz.kelly_fraction),
        "kelly_lcb_z": float(sz.kelly_lcb_z),
        "kelly_window_days": int(sz.kelly_window_days),
        "mode": str(sz.mode),
        "scale_floor": float(sz.scale_floor),
        "target_annual_vol": float(sz.target_annual_vol),
    }
    sw = policy.signal_window
    sw_dict: dict[str, Any] = {
        "bootstrap_return_tail_days": int(sw.bootstrap_return_tail_days),
        "committee_oos_start": pd.Timestamp(sw.committee_oos_start).tz_convert("UTC").isoformat(),
        "committee_purge_hours": int(sw.committee_purge_hours),
        "fold_panel_warmup_hours": int(sw.fold_panel_warmup_hours),
        "panel_window_days": int(sw.panel_window_days),
    }
    return {
        "admitted_members": [str(m) for m in sorted(policy.admitted_members)],
        "committee_member_weights": {str(k): float(v) for k, v in sorted(policy.committee_member_weights.items())},
        "signal_window": sw_dict,
        "sizing": sz_dict,
        "slow_horizon_hours": int(policy.slow_horizon_hours),
        "target_weights": tw_dict,
    }


def _canonical_for_digest(params: LiveStrategyParams | dict[str, Any]) -> dict[str, Any]:
    if isinstance(params, LiveStrategyParams):
        data: dict[str, Any] = {
            "schema_version": int(params.schema_version),
            "backtest_window": [
                pd.Timestamp(params.backtest_window[0]).tz_convert("UTC").isoformat(),
                pd.Timestamp(params.backtest_window[1]).tz_convert("UTC").isoformat(),
            ],
            "policy": _canonical_policy(params.policy),
            "bootstrap_sha256": str(params.bootstrap_sha256).lower(),
            "bootstrap_held_row": {str(k): float(v) for k, v in sorted(params.bootstrap_held_row.items())},
        }
    else:
        bw = params.get("backtest_window")
        if isinstance(bw, (list, tuple)) and len(bw) == 2:
            bw_iso = [
                pd.Timestamp(bw[0]).tz_convert("UTC").isoformat() if pd.Timestamp(bw[0]).tzinfo is not None else pd.Timestamp(bw[0]).tz_localize("UTC").isoformat(),
                pd.Timestamp(bw[1]).tz_convert("UTC").isoformat() if pd.Timestamp(bw[1]).tzinfo is not None else pd.Timestamp(bw[1]).tz_localize("UTC").isoformat(),
            ]
        else:
            bw_iso = bw  # type: ignore[assignment]
        raw_policy = params.get("policy") or {}
        data = {
            "schema_version": params.get("schema_version"),
            "backtest_window": bw_iso,
            "policy": {
                "admitted_members": [str(m) for m in sorted(raw_policy.get("admitted_members") or [])],
                "committee_member_weights": {str(k): float(v) for k, v in sorted((raw_policy.get("committee_member_weights") or {}).items())},
                "signal_window": {
                    str(k): v for k, v in sorted((raw_policy.get("signal_window") or {}).items())
                } if not isinstance(raw_policy.get("signal_window"), dict) else {
                    "bootstrap_return_tail_days": int(raw_policy["signal_window"].get("bootstrap_return_tail_days", 0)),
                    "committee_oos_start": (pd.Timestamp(raw_policy["signal_window"].get("committee_oos_start")).tz_convert("UTC").isoformat() if pd.Timestamp(raw_policy["signal_window"].get("committee_oos_start")).tzinfo is not None else pd.Timestamp(raw_policy["signal_window"].get("committee_oos_start")).tz_localize("UTC").isoformat()),
                    "committee_purge_hours": int(raw_policy["signal_window"].get("committee_purge_hours", 0)),
                    "fold_panel_warmup_hours": int(raw_policy["signal_window"].get("fold_panel_warmup_hours", 0)),
                    "panel_window_days": int(raw_policy["signal_window"].get("panel_window_days", 0)),
                },
                "sizing": {str(k): v for k, v in sorted((raw_policy.get("sizing") or {}).items())},
                "slow_horizon_hours": raw_policy.get("slow_horizon_hours"),
                "target_weights": {str(k): v for k, v in sorted((raw_policy.get("target_weights") or {}).items())},
            },
            "bootstrap_sha256": str(params.get("bootstrap_sha256", "")).lower(),
            "bootstrap_held_row": {str(k): float(v) for k, v in sorted((params.get("bootstrap_held_row") or {}).items())},
        }
    return data


def _compute_strategy_digest(params: LiveStrategyParams | dict[str, Any]) -> str:
    canonical = _canonical_for_digest(params)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _serialize_policy(policy: MhsDeploymentPolicy) -> dict[str, Any]:
    tw = policy.target_weights
    return {
        "target_weights": {
            "execution_timeframe": str(tw.execution_timeframe),
            "execution_universe_size": int(tw.execution_universe_size),
            "fast_book_mode": str(tw.fast_book_mode),
            "slow_book_mode": str(tw.slow_book_mode),
            "rebalance_filter": str(tw.rebalance_filter),
            "beta_neutralize": bool(tw.beta_neutralize),
            "ensemble_signal": str(tw.ensemble_signal),
            "trend_efficiency_overlay": bool(tw.trend_efficiency_overlay),
            "trend_sleeve": bool(tw.trend_sleeve),
            "trend_sleeve_gross": float(tw.trend_sleeve_gross),
            "crash_regime_tilt_alpha": None if tw.crash_regime_tilt_alpha is None else float(tw.crash_regime_tilt_alpha),
            "committee_capital": bool(tw.committee_capital),
            "committee_member_set": str(tw.committee_member_set),
            "committee_tranche_smoothing": bool(tw.committee_tranche_smoothing),
            "committee_regime_adaptive_tranche": bool(tw.committee_regime_adaptive_tranche),
            "committee_target_gross": None if tw.committee_target_gross is None else float(tw.committee_target_gross),
            "funding_carry_sleeve": bool(tw.funding_carry_sleeve),
            "funding_carry_weight": float(tw.funding_carry_weight),
            "fill_mark_parity_gate": bool(tw.fill_mark_parity_gate),
        },
        "sizing": {
            "mode": str(policy.sizing.mode),
            "target_annual_vol": float(policy.sizing.target_annual_vol),
            "exposure_cap": float(policy.sizing.exposure_cap),
            "scale_floor": float(policy.sizing.scale_floor),
            "kelly_enabled": bool(policy.sizing.kelly_enabled),
            "kelly_window_days": int(policy.sizing.kelly_window_days),
            "kelly_fraction": float(policy.sizing.kelly_fraction),
            "kelly_lcb_z": float(policy.sizing.kelly_lcb_z),
            "kelly_blend_weight": float(policy.sizing.kelly_blend_weight),
            "drawdown_brake": bool(policy.sizing.drawdown_brake),
        },
        "signal_window": {
            "panel_window_days": int(policy.signal_window.panel_window_days),
            "bootstrap_return_tail_days": int(policy.signal_window.bootstrap_return_tail_days),
            "fold_panel_warmup_hours": int(policy.signal_window.fold_panel_warmup_hours),
            "committee_purge_hours": int(policy.signal_window.committee_purge_hours),
            "committee_oos_start": pd.Timestamp(policy.signal_window.committee_oos_start).tz_convert("UTC").isoformat(),
        },
        "slow_horizon_hours": int(policy.slow_horizon_hours),
        "committee_member_weights": {str(k): float(v) for k, v in policy.committee_member_weights.items()},
        "admitted_members": [str(m) for m in policy.admitted_members],
    }


def _deserialize_policy(raw: Any) -> MhsDeploymentPolicy:
    if not isinstance(raw, dict):
        raise DataIntegrityError("policy must be a JSON object")
    unknown = set(raw) - set(_ALLOWED_POLICY_KEYS)
    if unknown:
        raise DataIntegrityError(f"unknown policy key {sorted(unknown)!r}")
    missing = set(_ALLOWED_POLICY_KEYS) - set(raw)
    if missing:
        raise DataIntegrityError(f"policy missing keys {sorted(missing)!r}")
    tw_raw = raw["target_weights"]
    sz_raw = raw["sizing"]
    sw_raw = raw["signal_window"]
    if not isinstance(tw_raw, dict) or not isinstance(sz_raw, dict) or not isinstance(sw_raw, dict):
        raise DataIntegrityError("policy sections must be objects")
    target = TargetWeightPolicy(**{k: v for k, v in tw_raw.items()})
    sizing = SizingPolicy(**{k: v for k, v in sz_raw.items()})
    sw = dict(sw_raw)
    sw["committee_oos_start"] = pd.Timestamp(sw["committee_oos_start"])
    window = SignalWindowPolicy(**{k: v for k, v in sw.items()})
    return MhsDeploymentPolicy(
        target_weights=target,
        sizing=sizing,
        signal_window=window,
        slow_horizon_hours=int(raw["slow_horizon_hours"]),
        committee_member_weights={str(k): float(v) for k, v in dict(raw["committee_member_weights"]).items()},
        admitted_members=tuple(str(m) for m in raw["admitted_members"]),
    )


def _serialize_params(params: LiveStrategyParams) -> dict[str, Any]:
    return {
        "schema_version": int(params.schema_version),
        "strategy_digest": str(params.strategy_digest),
        "backtest_window": [
            pd.Timestamp(params.backtest_window[0]).isoformat(),
            pd.Timestamp(params.backtest_window[1]).isoformat(),
        ],
        "created_at": pd.Timestamp(params.created_at).isoformat(),
        "policy": _serialize_policy(params.policy),
        "bootstrap_sha256": str(params.bootstrap_sha256).lower(),
        "bootstrap_held_row": {str(k): float(v) for k, v in params.bootstrap_held_row.items()},
    }


def _validate_bootstrap_sha256(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value.lower()):
        raise DataIntegrityError("bootstrap_sha256 must be 64-char lowercase hex")
    return value.lower()


def _deserialize_params(raw: dict[str, Any]) -> LiveStrategyParams:
    if not isinstance(raw, dict):
        raise DataIntegrityError("strategy params must be a JSON object")
    unknown = set(raw) - set(_ALLOWED_PARAMS_KEYS)
    if unknown:
        raise DataIntegrityError(f"unknown strategy params key {sorted(unknown)!r}")
    missing = set(_ALLOWED_PARAMS_KEYS) - set(raw)
    if missing:
        raise DataIntegrityError(f"strategy params missing keys {sorted(missing)!r}")
    schema = raw.get("schema_version")
    if schema != 2:
        raise DataIntegrityError(f"unsupported schema_version {schema!r}; schema_version must be 2")
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
    bootstrap_sha256 = _validate_bootstrap_sha256(raw.get("bootstrap_sha256"))
    held_raw = raw.get("bootstrap_held_row")
    if not isinstance(held_raw, dict):
        raise DataIntegrityError("bootstrap_held_row must be object")
    held = {str(k): float(v) for k, v in held_raw.items()}
    policy = _deserialize_policy(raw.get("policy"))
    expected = _compute_strategy_digest(raw)
    if not hmac.compare_digest(expected, digest):
        raise DataIntegrityError(f"strategy_digest mismatch: expected {expected!r} got {digest!r}")
    return LiveStrategyParams(
        schema_version=2,
        strategy_digest=str(digest),
        backtest_window=(bw0, bw1),
        created_at=created_at,
        policy=policy,
        bootstrap_sha256=bootstrap_sha256,
        bootstrap_held_row=held,
    )


def save_strategy_params(path: Path, params: LiveStrategyParams, *, artifact_key: SecretStr | None = None) -> Path:
    path = Path(path)
    if int(params.schema_version) != 2:
        raise DataIntegrityError(f"unsupported schema_version {params.schema_version!r}; schema_version must be 2")
    _validate_bootstrap_sha256(params.bootstrap_sha256)
    digest = _compute_strategy_digest(params)
    params_with_digest = LiveStrategyParams(
        schema_version=2,
        strategy_digest=digest,
        backtest_window=params.backtest_window,
        created_at=params.created_at,
        policy=params.policy,
        bootstrap_sha256=str(params.bootstrap_sha256).lower(),
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


def _validate_bootstrap_series(reference_daily_returns: pd.Series) -> pd.Series:
    if not isinstance(reference_daily_returns, pd.Series):
        raise ValueError("reference_daily_returns must be a Series")
    if reference_daily_returns.empty:
        raise ValueError("reference_daily_returns must be non-empty")
    idx = reference_daily_returns.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("reference_daily_returns index must be a UTC DatetimeIndex")
    if idx.tz is None:
        raise ValueError("reference_daily_returns index must be tz-aware UTC")
    converted = idx.tz_convert("UTC")
    if not converted.is_unique:
        raise ValueError("reference_daily_returns index must be unique")
    if not converted.is_monotonic_increasing:
        raise ValueError("reference_daily_returns index must be strictly increasing")
    values = reference_daily_returns.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("reference_daily_returns must be finite")
    out = pd.Series(values, index=converted, dtype="float64", name="reference_daily_return")
    return out


def _bootstrap_plaintext_bytes(reference_daily_returns: pd.Series) -> bytes:
    series = _validate_bootstrap_series(reference_daily_returns)
    frame = pd.DataFrame({"reference_daily_return": series})
    buf = io.BytesIO()
    frame.to_parquet(buf, index=True)
    return buf.getvalue()


def save_strategy_bootstrap(path: Path, reference_daily_returns: pd.Series, *, artifact_key: SecretStr | None = None) -> tuple[Path, str]:
    path = Path(path)
    plaintext = _bootstrap_plaintext_bytes(reference_daily_returns)
    digest = hashlib.sha256(plaintext).hexdigest().lower()
    if artifact_key is not None:
        from src.live.crypto import derive_key, seal_bytes

        dest = path if str(path).endswith(".enc") else Path(str(path) + ".enc")
        dest.parent.mkdir(parents=True, exist_ok=True)
        sealed = seal_bytes(plaintext, derive_key(artifact_key))
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(sealed)
        os.replace(tmp, dest)
        return dest, digest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(plaintext)
    os.replace(tmp, path)
    return path, digest


def load_strategy_bootstrap(path: Path, *, expected_sha256: str, artifact_key: SecretStr | None = None) -> pd.Series:
    path = Path(path)
    candidate = path
    if str(candidate).endswith(".enc"):
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {candidate}")
        if not candidate.exists():
            raise DataIntegrityError(f"strategy bootstrap not found: {candidate}")
        from src.live.crypto import derive_key, open_bytes

        try:
            plaintext = open_bytes(candidate.read_bytes(), derive_key(artifact_key))
        except ArtifactSealError:
            raise
        except Exception as exc:
            raise DataIntegrityError(f"strategy bootstrap file corrupt: {candidate}: {exc}") from exc
    else:
        if not candidate.exists():
            if artifact_key is not None:
                enc = Path(str(candidate) + ".enc")
                if enc.exists():
                    return load_strategy_bootstrap(enc, expected_sha256=expected_sha256, artifact_key=artifact_key)
            raise DataIntegrityError(f"strategy bootstrap not found: {candidate}")
        try:
            plaintext = candidate.read_bytes()
        except Exception as exc:
            raise DataIntegrityError(f"strategy bootstrap file corrupt: {candidate}: {exc}") from exc
        if artifact_key is not None and plaintext[:8] == b"CPSEAL01":
            from src.live.crypto import derive_key, open_bytes

            plaintext = open_bytes(plaintext, derive_key(artifact_key))
    if not isinstance(expected_sha256, str) or not _SHA256_HEX_RE.match(expected_sha256.lower()):
        raise DataIntegrityError("expected_sha256 must be 64-char lowercase hex")
    actual = hashlib.sha256(plaintext).hexdigest().lower()
    if not hmac.compare_digest(actual, expected_sha256.lower()):
        raise DataIntegrityError(f"bootstrap_sha256 mismatch: expected {expected_sha256.lower()!r} got {actual!r}")
    try:
        frame = pd.read_parquet(io.BytesIO(plaintext))
    except Exception as exc:
        raise DataIntegrityError(f"strategy bootstrap parquet decode failed: {exc}") from exc
    if "reference_daily_return" not in frame.columns:
        raise DataIntegrityError("strategy bootstrap missing column 'reference_daily_return'")
    series = frame["reference_daily_return"]
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataIntegrityError("strategy bootstrap index must be a DatetimeIndex")
    if series.index.tz is None:
        raise DataIntegrityError("strategy bootstrap index must be tz-aware UTC")
    series = series.tz_convert("UTC")
    try:
        values = series.to_numpy(dtype="float64")
    except Exception as exc:
        raise DataIntegrityError(f"strategy bootstrap dtype invalid: {exc}") from exc
    if not np.isfinite(values).all():
        raise DataIntegrityError("strategy bootstrap values must be finite")
    loaded = pd.Series(values, index=series.index, dtype="float64", name="reference_daily_return")
    try:
        inferred_freq = pd.infer_freq(loaded.index)
    except ValueError:
        inferred_freq = None
    if inferred_freq is not None:
        loaded.index.freq = inferred_freq
    return loaded


def assert_deployment_eligible(report: Any, *, reference_report_path: Path | None = None) -> None:
    if getattr(report, "status", None) != "COMPLETE":
        raise DataIntegrityError("deployment ineligible: report status not COMPLETE")
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
                cur_flags = getattr(report, "flags", None)
                if cur_flags is None:
                    cur_flags = {}
                cur_payload = json.dumps(cur_flags, sort_keys=True, separators=(",", ":"), default=str)
                ref_payload = json.dumps(ref_flags, sort_keys=True, separators=(",", ":"), default=str)
                cur_digest = hashlib.sha256(cur_payload.encode("utf-8")).hexdigest()
                ref_digest = hashlib.sha256(ref_payload.encode("utf-8")).hexdigest()
                if not hmac.compare_digest(cur_digest, ref_digest):
                    raise DataIntegrityError("deployment ineligible: flags digest drift")
