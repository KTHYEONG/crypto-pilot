"""Incremental signal refresh: forward scoring over a bounded rolling window.

I-NO-DISCOVERY: this module never widens or re-opens the sealed holdout
window. It reuses the existing bounded-window builder
(``_build_fold_target_weights``) with frozen parameters injected, and never
calls ``resolve_evaluation_end`` or any discovery/fold-selection/evidence-gate
code path.
"""

from __future__ import annotations

import contextlib
import gc
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.mhs.evidence import AnchoredPurgedFold
from src.mhs.params import (
    COMMITTEE_OOS_START,
    COMMITTEE_PURGE_HOURS,
    FOLD_PANEL_WARMUP_HOURS,
    PNL_VOL_TARGET_SCALE_FLOOR,
    SIGNAL_OVERLAP_TOLERANCE,
    SIGNAL_PANEL_WINDOW_DAYS,
    SIGNAL_REPLAY_WARMUP_DAYS,
    SIGNAL_RETURN_TAIL_DAYS,
)
from src.mhs.signal_state import BOUND_FLAGS, SignalState

# Allowed imports at module level are minimal to satisfy I-NO-DISCOVERY: no
# discovery/fold-selection/replay stage modules are imported here.


@dataclass(frozen=True, slots=True)
class SignalRefreshReport:
    status: str
    reason: str | None
    decision_time: pd.Timestamp
    n_symbols: int
    gross_exposure: float
    exposure_scale: float
    elapsed_seconds: float


def _current_deployed_flags() -> dict[str, Any]:
    """The flags a fresh bootstrap would currently resolve to, restricted to
    BOUND_FLAGS. Used only to detect drift against ``state.flags_digest``;
    raises DataIntegrityError instead of silently degrading to an empty
    mapping (an empty mapping would trivially match a stale digest)."""
    from src.mhs.pipeline.config import MhsRunConfig

    cfg = MhsRunConfig()
    flags: dict[str, Any] = {}
    for name in BOUND_FLAGS:
        if not hasattr(cfg, name):
            raise DataIntegrityError(f"MhsRunConfig missing bound flag {name!r}")
        flags[name] = getattr(cfg, name)
    return flags


def _load_artifact_frame(artifact_path: Path, artifact_key: SecretStr | None) -> pd.DataFrame | None:
    """Returns None only when the artifact genuinely does not exist yet
    (first-ever refresh). A corrupt/malformed EXISTING artifact must fail
    closed via DataIntegrityError, never be silently treated as absent."""
    path = Path(artifact_path)
    candidate: Path | None = None
    if path.exists():
        candidate = path
    elif artifact_key is not None:
        enc = path if str(path).endswith(".enc") else Path(f"{path}.enc")
        if enc.exists():
            candidate = enc
    if candidate is None:
        return None
    if str(candidate).endswith(".enc"):
        if artifact_key is None:
            raise DataIntegrityError(f"sealed artifact requires a key: {candidate}")
        from src.live.crypto import derive_key, read_sealed_parquet

        return read_sealed_parquet(candidate, derive_key(artifact_key))
    try:
        return pd.read_parquet(candidate)
    except Exception as exc:
        raise DataIntegrityError(f"artifact read failed: {candidate}: {exc}") from exc


def _save_artifact_frame(frame: pd.DataFrame, artifact_path: Path, artifact_key: SecretStr | None) -> None:
    path = Path(artifact_path)
    is_enc = str(path).endswith(".enc")
    if artifact_key is not None or is_enc:
        from src.live.crypto import derive_key, seal_bytes

        dest = path if is_enc else Path(f"{path}.enc")
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=True)
        if artifact_key is None:
            raise DataIntegrityError(f"sealed artifact path requires a key: {dest}")
        sealed = seal_bytes(buffer.getvalue(), derive_key(artifact_key))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(sealed)
        os.replace(tmp, dest)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=True)
    os.replace(tmp, path)


def _synthetic_fold(decision_time: pd.Timestamp) -> AnchoredPurgedFold:
    """A rolling [decision_time - SIGNAL_PANEL_WINDOW_DAYS, decision_time]
    validation window. ``vs - FOLD_PANEL_WARMUP_HOURS`` (computed inside
    _build_fold_target_weights as ``panel_start``) lands exactly at
    ``decision_time - SIGNAL_PANEL_WINDOW_DAYS`` by construction."""
    ve = pd.Timestamp(decision_time)
    ve = ve.tz_localize("UTC") if ve.tzinfo is None else ve.tz_convert("UTC")
    vs = ve - pd.Timedelta(days=SIGNAL_PANEL_WINDOW_DAYS) + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS)
    if vs >= ve:
        raise DataIntegrityError(
            f"SIGNAL_PANEL_WINDOW_DAYS ({SIGNAL_PANEL_WINDOW_DAYS}d) too small for "
            f"FOLD_PANEL_WARMUP_HOURS ({FOLD_PANEL_WARMUP_HOURS}h)"
        )
    train_end = vs - pd.Timedelta(hours=COMMITTEE_PURGE_HOURS) - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=365)
    if not (train_start < train_end < vs < ve):
        raise DataIntegrityError(f"signal refresh fold bounds not ascending for decision_time={decision_time}")
    return AnchoredPurgedFold(
        train_start=train_start, train_end=train_end,
        validation_start=vs, validation_end=ve,
        forward_dependency_hours=24, purge_hours=COMMITTEE_PURGE_HOURS,
    )


def _assert_member_parity_precondition(fold: AnchoredPurgedFold, state: SignalState) -> None:
    """I-MEMBER-PARITY.

    The invariant this guards: with ``coverage_cutoff=COMMITTEE_OOS_START``,
    ``build_feature_books`` restricts its per-feature coverage audit to rows
    strictly before the cutoff (``evaluation.py``/``features.py``
    ``feature_coverage_audit``). A rolling window whose validation_start is
    already after ``COMMITTEE_OOS_START`` slices that audit to an EMPTY frame,
    and ``feature_coverage_audit`` on an empty slice returns ``{}`` -- so
    ``any(cov < min_coverage for cov in {}.values())`` is unconditionally
    False and every member of the pinned set is admitted, deterministically
    (verified empirically; see docs/specs/mhs_incremental_signal_refresh.md
    §1.3). This function asserts that precondition holds for THIS fold rather
    than re-deriving the admitted set (which _build_fold_target_weights does
    not expose) or trusting it silently.
    """
    if fold.validation_start <= COMMITTEE_OOS_START:
        raise DataIntegrityError(
            f"member-parity precondition violated: fold.validation_start={fold.validation_start} "
            f"<= COMMITTEE_OOS_START={COMMITTEE_OOS_START}; the pinned admitted-member "
            "guarantee does not hold for this window"
        )
    if not state.frozen.admitted_members:
        raise DataIntegrityError("signal state carries an empty admitted_members set")


def _request_from_deployed_flags(state: SignalState, data_root: str | None) -> Any:
    """Reconstruct the exact deployed MhsDiagnosticRequest by unpacking the
    verbatim resolved-flag snapshot captured at bootstrap -- never a fresh
    default-guessing reconstruction. Any failure fails closed."""
    from src.application.research.mhs.contracts import MhsDiagnosticRequest

    try:
        request = MhsDiagnosticRequest(**state.frozen.deployed_flags)
    except Exception as exc:
        raise DataIntegrityError(
            f"failed to reconstruct MhsDiagnosticRequest from frozen deployed_flags: {exc}"
        ) from exc
    if data_root is not None:
        request = _dataclass_replace(request, data_root=data_root)
    return request


def _dataclass_replace(request: Any, **overrides: Any) -> Any:
    import dataclasses

    return dataclasses.replace(request, **overrides)


def refresh_signal_row(
    state_path: Path,
    artifact_path: Path,
    decision_time: pd.Timestamp,
    *,
    data_root: str | None = None,
    artifact_key: SecretStr | None = None,
    now: pd.Timestamp | None = None,
) -> SignalRefreshReport:
    """The forward scorer. See docs/specs/mhs_incremental_signal_refresh.md."""
    del now  # reserved for future staleness diagnostics; not used by scoring
    t0 = time.perf_counter()
    from src.mhs.signal_state import assert_state_binding, load_signal_state, save_signal_state

    # 1) I-STATE-BINDING: load + verify before touching anything else.
    state = load_signal_state(Path(state_path), artifact_key=artifact_key)
    assert_state_binding(state, _current_deployed_flags())

    dt = pd.Timestamp(decision_time)
    if dt.tzinfo is None:
        raise DataIntegrityError("decision_time must be tz-aware")
    dt = dt.tz_convert("UTC")

    # 2) I-APPEND-ONLY.
    if dt <= state.last_decision_time:
        return SignalRefreshReport(
            status="NOOP", reason="decision_time not after last_decision_time",
            decision_time=dt, n_symbols=0, gross_exposure=0.0, exposure_scale=0.0,
            elapsed_seconds=time.perf_counter() - t0,
        )

    artifact_frame = _load_artifact_frame(Path(artifact_path), artifact_key)
    if artifact_frame is not None and not artifact_frame.empty:
        idx = pd.DatetimeIndex(artifact_frame.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            artifact_frame = artifact_frame.set_axis(idx)
        if dt in idx:
            return SignalRefreshReport(
                status="NOOP", reason="decision_time already in artifact",
                decision_time=dt, n_symbols=int(artifact_frame.shape[1]),
                gross_exposure=0.0, exposure_scale=0.0,
                elapsed_seconds=time.perf_counter() - t0,
            )

    # 3) Bounded-window forward scoring via the EXISTING builder (I-REUSE-BUILDER).
    from src.application.research.mhs.evaluation import _build_fold_target_weights

    fold = _synthetic_fold(dt)
    _assert_member_parity_precondition(fold, state)
    request = _request_from_deployed_flags(state, data_root)
    root_str = str(data_root) if data_root is not None else ""

    funding_by_symbol = _load_funding_by_symbol(root_str, data_root)

    seed_row = pd.Series(state.held_target_row, dtype="float64") if state.held_target_row else None
    target_weights, signal_available_at, _minute_roster, grid_1h = _build_fold_target_weights(
        root_str, fold, request, funding_by_symbol,
        slow_horizon_override=int(state.frozen.slow_horizon_hours),
        committee_member_weights=dict(state.frozen.committee_member_weights),
        deadband_seed_row=seed_row,
    )

    # 4) I-OVERLAP-PARITY: any already-published row must match exactly.
    if artifact_frame is not None and not artifact_frame.empty:
        _assert_overlap_parity(artifact_frame, target_weights)

    # 5) Release the panel before the replay (mirror the backtest's own
    # release discipline around this exact call, e.g. evaluation.py's
    # ``del log_close`` / ``gc.collect()`` pairing).
    with contextlib.suppress(NameError):
        del grid_1h
    gc.collect()

    # 6) I-REFERENCE-FIDELITY / I-STATELESS-REPLAY: rebuild the reference
    # daily returns with the SAME unscaled-book replay the backtest uses.
    usable = _rolling_reference_returns(target_weights, signal_available_at, fold, request, funding_by_symbol, root_str)

    # 7) Exposure scale, pinned to the frozen target_vol/cap.
    exposure_scale = _resolve_exposure_scale(state, usable)

    if dt not in target_weights.index:
        raise DataIntegrityError(f"decision_time {dt} not in scored window")
    raw_row = target_weights.loc[dt]
    scaled_row = raw_row * exposure_scale

    new_frame = _append_row(artifact_frame, scaled_row, dt)
    _save_artifact_frame(new_frame, Path(artifact_path), artifact_key)

    new_reference = _advance_reference_returns(state.reference_daily_returns, usable)
    held_dict = {str(k): float(v) for k, v in scaled_row.items() if pd.notna(v)}

    from src.mhs.signal_state import SignalState as _SignalState

    new_state = _SignalState(
        schema_version=state.schema_version,
        params_digest=state.params_digest,
        flags_digest=state.flags_digest,
        frozen=state.frozen,
        last_decision_time=dt,
        held_target_row=held_dict,
        reference_daily_returns=new_reference,
    )
    save_signal_state(Path(state_path), new_state, artifact_key=artifact_key)

    return SignalRefreshReport(
        status="APPENDED", reason=None, decision_time=dt,
        n_symbols=int(scaled_row.count()), gross_exposure=float(scaled_row.abs().sum()),
        exposure_scale=float(exposure_scale), elapsed_seconds=time.perf_counter() - t0,
    )


def _assert_overlap_parity(artifact_frame: pd.DataFrame, target_weights: pd.DataFrame) -> None:
    overlap_idx = artifact_frame.index.intersection(target_weights.index)
    common_cols = artifact_frame.columns.intersection(target_weights.columns)
    for ts in overlap_idx:
        for col in common_cols:
            a_val = float(artifact_frame.loc[ts, col])
            b_val = float(target_weights.loc[ts, col])
            if pd.isna(a_val) and pd.isna(b_val):
                continue
            if pd.isna(a_val) or pd.isna(b_val):
                raise DataIntegrityError(f"overlap parity NaN mismatch symbol={col} time={ts}")
            if abs(a_val - b_val) > SIGNAL_OVERLAP_TOLERANCE:
                raise DataIntegrityError(
                    f"overlap parity failed symbol={col} time={ts} diff={abs(a_val - b_val)}"
                )


def _append_row(artifact_frame: pd.DataFrame | None, scaled_row: pd.Series, dt: pd.Timestamp) -> pd.DataFrame:
    new_row_df = pd.DataFrame([scaled_row], index=pd.DatetimeIndex([dt]))
    if artifact_frame is None or artifact_frame.empty:
        return new_row_df
    all_cols = sorted(set(artifact_frame.columns) | set(new_row_df.columns))
    combined = pd.concat([artifact_frame.reindex(columns=all_cols), new_row_df.reindex(columns=all_cols)])
    return combined.sort_index()


def _advance_reference_returns(existing: pd.Series, usable: pd.Series) -> pd.Series:
    if usable.empty:
        return existing.tail(SIGNAL_RETURN_TAIL_DAYS)
    last_idx = usable.index[-1]
    last_val = float(usable.iloc[-1])
    if last_idx in existing.index:
        return existing.tail(SIGNAL_RETURN_TAIL_DAYS)
    combined = pd.concat([existing, pd.Series([last_val], index=pd.DatetimeIndex([last_idx]))])
    return combined.sort_index().tail(SIGNAL_RETURN_TAIL_DAYS)


def _load_funding_by_symbol(root_str: str, data_root: str | None) -> dict[str, pd.Series]:
    """Best-effort funding load for the roster present under <root>/1h/.
    A missing/unreadable funding file for one symbol is not fatal --
    _build_fold_target_weights itself fails closed if the resulting funding
    coverage is insufficient (``no fold symbol has funding coverage``)."""
    import glob

    from src.market_data.storage.loaders import load_funding_rates

    search_root = root_str if root_str else "data/futures"
    pattern = os.path.join(search_root, "1h", "*.parquet")
    funding_by_symbol: dict[str, pd.Series] = {}
    for p in sorted(glob.glob(pattern)):
        sym = os.path.basename(p).removesuffix(".parquet")
        if data_root is not None:
            fp = Path(data_root) / "funding" / f"{sym}.parquet"
            if not fp.exists():
                fp = Path(data_root) / "futures" / "funding" / f"{sym}.parquet"
        else:
            from src.common.config import funding_path

            fp = funding_path(sym)
        if fp.exists():
            funding_by_symbol[sym] = load_funding_rates(str(fp))
    return funding_by_symbol


def _rolling_reference_returns(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    fold: AnchoredPurgedFold,
    request: Any,
    funding_by_symbol: dict[str, pd.Series],
    root_str: str,
) -> pd.Series:
    """I-REFERENCE-FIDELITY: replay the UNSCALED book with the exact bound/spec
    the backtest uses for its own reference series (evaluation.py:2311-2316),
    then drop the leading SIGNAL_REPLAY_WARMUP_DAYS rows (I-STATELESS-REPLAY)."""
    from src.application.research.mhs.evaluation import (
        _iter_mhs_execution_windows,
        _resolved_base_execution_spec,
    )
    from src.mhs.execution import replay_execution_windows

    spec = _resolved_base_execution_spec(request)
    windows = _iter_mhs_execution_windows(
        target_weights, signal_available_at, root_str, request.execution_timeframe,
        fold.validation_start, fold.validation_end, funding_by_symbol, request.mark_mode, spec,
    )
    replay_result = replay_execution_windows(windows, 1.0, "OHLCV_IMMEDIATE_TAKER", spec)
    ref_series = replay_result.ledger.equity.resample("1D").last().pct_change().dropna()
    if len(ref_series) <= SIGNAL_REPLAY_WARMUP_DAYS:
        return ref_series.iloc[0:0]
    return ref_series.iloc[SIGNAL_REPLAY_WARMUP_DAYS:]


def _resolve_exposure_scale(state: SignalState, usable: pd.Series) -> float:
    from src.application.research.mhs.scaling import _committee_capital_replay_scale, _exante_vol_target_scale

    if not usable.empty:
        warmup_mask = state.reference_daily_returns.index < usable.index[0]
        warmup = state.reference_daily_returns.loc[warmup_mask].sort_index()
        warmup = warmup if not warmup.empty else None
    else:
        warmup = state.reference_daily_returns if not state.reference_daily_returns.empty else None

    if str(state.frozen.pnl_vol_target_mode) == "constant_risk":
        from src.application.research.mhs.scaling import _constant_risk_scale

        base_scale = _constant_risk_scale(
            usable, target_vol=float(state.frozen.growth_budget_target_vol),
            cap=float(state.frozen.exposure_cap), warmup_returns=warmup,
        )
    else:
        base_scale = _exante_vol_target_scale(
            usable, target_vol=float(state.frozen.growth_budget_target_vol),
            cap=float(state.frozen.exposure_cap), warmup_returns=warmup,
        )
    committee_capital = bool(state.frozen.deployed_flags.get("committee_capital", False))
    committee_kelly_sizing = bool(state.frozen.deployed_flags.get("committee_kelly_sizing", False))
    scale_series = _committee_capital_replay_scale(
        base_scale, usable, committee_capital, committee_kelly_sizing,
        cap=float(state.frozen.exposure_cap),
    )
    exposure_scale = float(scale_series.iloc[-1]) if not scale_series.empty else 1.0
    return float(max(PNL_VOL_TARGET_SCALE_FLOOR, min(exposure_scale, float(state.frozen.exposure_cap))))
