"""Live signal step: realized reference and rolling compute."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.portfolio_state import default_portfolio_state_dir
from src.mhs.live_runtime import LiveRuntime
from src.mhs.live_strategy import LiveStrategyParams, snapshot_value

logger = logging.getLogger("LiveSignalStep")

try:
    from src.mhs.evaluation import _build_fold_target_weights  # noqa: F401
except Exception:  # noqa: BLE001,S110

    def _build_fold_target_weights(*_a: Any, **_k: Any) -> Any:  # type: ignore[misc]
        raise DataIntegrityError("missing _build_fold_target_weights")


def realized_daily_returns(portfolio_state_dir: Path, mode: str, *, bt_end: pd.Timestamp) -> pd.Series:
    p = Path(portfolio_state_dir)
    if not p.exists():
        return pd.Series(dtype="float64")
    shards = sorted(p.glob("*.parquet"))
    if not shards:
        return pd.Series(dtype="float64")
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: S112
            continue
    if not frames:
        return pd.Series(dtype="float64")
    try:
        combined = pd.concat(frames, ignore_index=True)
    except Exception:
        return pd.Series(dtype="float64")
    if combined.empty or "mode" not in combined.columns or "decision_time" not in combined.columns or "equity_usdt" not in combined.columns:
        return pd.Series(dtype="float64")
    # filter mode
    try:
        combined = combined[combined["mode"] == mode]
    except Exception:
        return pd.Series(dtype="float64")
    if combined.empty:
        return pd.Series(dtype="float64")
    # decision_time parsing
    try:
        combined["decision_time"] = pd.to_datetime(combined["decision_time"], utc=True, errors="coerce")
    except Exception:
        return pd.Series(dtype="float64")
    combined = combined.dropna(subset=["decision_time"])
    # equity finite and >0
    try:
        eq = pd.to_numeric(combined["equity_usdt"], errors="coerce")
    except Exception:
        return pd.Series(dtype="float64")
    mask = np.isfinite(eq.to_numpy(dtype="float64")) & (eq > 0)
    combined = combined[mask]
    if combined.empty:
        return pd.Series(dtype="float64")
    # bt_end filter strict >
    try:
        bt = pd.Timestamp(bt_end)
        if bt.tzinfo is None:  # noqa: SIM108
            bt = bt.tz_localize("UTC")
        else:
            bt = bt.tz_convert("UTC")
    except Exception:
        bt = pd.Timestamp(bt_end, tz="UTC")
    combined = combined[combined["decision_time"] > bt]
    if combined.empty:
        return pd.Series(dtype="float64")
    combined = combined.sort_values("decision_time")
    combined = combined.drop_duplicates("decision_time", keep="last")
    combined = combined.set_index("decision_time")
    combined = combined.sort_index()
    equity = combined["equity_usdt"]
    equity = pd.to_numeric(equity, errors="coerce")
    if len(equity) < 2:
        return pd.Series(dtype="float64")
    ret = equity.pct_change().dropna()
    ret = ret.rename("reference_daily_return")
    # ensure tz-aware UTC index
    try:
        ret.index = pd.DatetimeIndex(ret.index).tz_convert("UTC")
    except Exception:
        ret.index = pd.to_datetime(ret.index, utc=True)
    ret = ret.astype("float64")
    return ret


def _synthetic_fold(date: pd.Timestamp, params: LiveStrategyParams) -> Any:
    from src.mhs.evidence import AnchoredPurgedFold

    dt = pd.Timestamp(date).tz_convert("UTC").normalize() if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date).tz_localize("UTC").normalize()
    try:
        window_days = int(snapshot_value(params, "SIGNAL_PANEL_WINDOW_DAYS"))
    except DataIntegrityError:
        from src.mhs.params import SIGNAL_PANEL_WINDOW_DAYS as _def_wd  # noqa: N811

        window_days = int(_def_wd)
    try:
        warmup_hours = int(snapshot_value(params, "FOLD_PANEL_WARMUP_HOURS"))
    except DataIntegrityError:
        from src.mhs.params import FOLD_PANEL_WARMUP_HOURS as _def_wh  # noqa: N811

        warmup_hours = int(_def_wh)
    try:
        purge_hours = int(snapshot_value(params, "COMMITTEE_PURGE_HOURS"))
    except DataIntegrityError:
        from src.mhs.params import COMMITTEE_PURGE_HOURS as _def_ph  # noqa: N811

        purge_hours = int(_def_ph)
    vs = dt - pd.Timedelta(days=window_days) + pd.Timedelta(hours=warmup_hours)
    if vs >= dt:
        raise DataIntegrityError("SIGNAL_PANEL_WINDOW_DAYS too small")
    train_end = vs - pd.Timedelta(hours=purge_hours) - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=365)
    return AnchoredPurgedFold(
        train_start=train_start, train_end=train_end,
        validation_start=vs, validation_end=dt,
        forward_dependency_hours=24, purge_hours=purge_hours,
    )


def _load_funding_by_symbol(root_str: str) -> dict[str, pd.Series]:
    import glob
    import os

    from src.market_data.storage.loaders import load_funding_rates

    funding_by_symbol: dict[str, pd.Series] = {}
    search_root = root_str if root_str else "data/futures"
    pattern = os.path.join(search_root, "1h", "*.parquet") if os.path.isdir(os.path.join(search_root, "1h")) else os.path.join(search_root, "ohlcv", "1h", "*.parquet")
    if not glob.glob(pattern):
        from src.common.paths import FUTURES_DATA_DIR

        pattern = str(FUTURES_DATA_DIR / "ohlcv" / "1h" / "*.parquet")
    for p in sorted(glob.glob(pattern)):
        sym = os.path.basename(p).removesuffix(".parquet")
        try:
            from src.common.paths import funding_path

            fp = funding_path(sym)
            if fp.exists():
                funding_by_symbol[sym] = load_funding_rates(str(fp))
        except Exception:  # noqa: S112
            continue
    if not funding_by_symbol:
        try:
            import glob as _g

            fp_pattern = os.path.join(search_root, "funding", "*.parquet")
            if not _g.glob(fp_pattern):
                from src.common.paths import FUTURES_DATA_DIR as _fdd  # noqa: N811

                fp_pattern = str(_fdd / "funding" / "*.parquet")
            for fp in sorted(_g.glob(fp_pattern)):  # type: ignore[assignment]
                sym = os.path.basename(fp).removesuffix(".parquet")
                if sym not in funding_by_symbol:
                    try:
                        from src.market_data.storage.loaders import load_funding_rates as _lfr

                        funding_by_symbol[sym] = _lfr(fp)
                    except Exception:  # noqa: S112
                        continue
        except Exception:  # noqa: S110
            pass
    return funding_by_symbol


def compute_signal_row(
    params: LiveStrategyParams,
    runtime: LiveRuntime,
    data_root: str,
    date: pd.Timestamp,
    *,
    portfolio_state_dir: Path | None = None,
    mode: str = "shadow",
) -> tuple[pd.Series, pd.Series, float]:
    from src.mhs.params import PNL_VOL_TARGET_SCALE_FLOOR
    from src.mhs.scaling import (
        _committee_capital_replay_scale,
        _constant_risk_scale,
        _exante_vol_target_scale,
    )

    if not data_root:
        from src.common.paths import FUTURES_DATA_DIR

        data_root = str(FUTURES_DATA_DIR / "ohlcv")

    dt = pd.Timestamp(date).tz_convert("UTC").normalize() if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date).tz_localize("UTC").normalize()
    fold = _synthetic_fold(dt, params)
    from src.mhs.contracts import MhsDiagnosticRequest

    try:
        request = MhsDiagnosticRequest(**params.deployed_flags)
    except Exception as exc:
        raise DataIntegrityError(f"failed to reconstruct request: {exc}") from exc
    funding_by_symbol = _load_funding_by_symbol(data_root)

    seed_row = pd.Series(runtime.held_target_row, dtype="float64") if runtime.held_target_row else None
    target_weights, _signal_available_at, _minute_roster, grid_1h = _build_fold_target_weights(  # noqa: RUF059
        data_root,
        fold,
        request,
        funding_by_symbol,
        slow_horizon_override=int(params.slow_horizon_hours),
        committee_member_weights=dict(params.committee_member_weights),
        deadband_seed_row=seed_row,
        require_minute_roster=False,
    )
    if dt not in target_weights.index:
        del grid_1h
        gc.collect()
        raise DataIntegrityError(f"decision_time {dt} not in scored window")
    raw_row = target_weights.loc[dt]

    psd = portfolio_state_dir if portfolio_state_dir is not None else default_portfolio_state_dir()
    bt_end = pd.Timestamp(params.backtest_window[1])
    if bt_end.tzinfo is None:  # noqa: SIM108
        bt_end = bt_end.tz_localize("UTC")
    else:
        bt_end = bt_end.tz_convert("UTC")
    bt_end = bt_end.normalize()
    # warmup slice from frozen bootstrap anchor
    warmup_src = runtime.reference_daily_returns
    if not warmup_src.empty:
        if warmup_src.index.tz is None:
            warmup_src = warmup_src.copy()
            warmup_src.index = warmup_src.index.tz_localize("UTC")
        else:
            warmup_src = warmup_src.tz_convert("UTC")
        warmup_src = warmup_src.sort_index()
        warmup = warmup_src[warmup_src.index <= bt_end]
    else:
        warmup = pd.Series(dtype="float64")
        # ensure tz-aware type
        warmup.index = pd.DatetimeIndex([], tz="UTC")

    forward = realized_daily_returns(psd, mode, bt_end=bt_end)
    tv = float(params.growth_budget_target_vol)
    cap = float(params.exposure_cap)
    constant_risk = str(params.pnl_vol_target_mode) == "constant_risk"

    if forward.empty:
        if not warmup.empty:
            ref_series = warmup
        else:
            ref_series = pd.Series([0.0], index=pd.DatetimeIndex([bt_end], tz="UTC"), dtype="float64")
        warmup_arg = None
    else:
        ref_series = forward
        warmup_arg = warmup if not warmup.empty else None

    if constant_risk:
        base = _constant_risk_scale(ref_series, target_vol=tv, cap=cap, warmup_returns=warmup_arg)
    else:
        base = _exante_vol_target_scale(ref_series, target_vol=tv, cap=cap, warmup_returns=warmup_arg)
    committee_capital = bool(params.deployed_flags.get("committee_capital", False))
    committee_kelly = bool(params.deployed_flags.get("committee_kelly_sizing", False))
    scale_series = _committee_capital_replay_scale(base, ref_series, committee_capital, committee_kelly, cap=cap)
    scalar_raw = float(scale_series.iloc[-1]) if not scale_series.empty else 1.0
    scalar = float(max(PNL_VOL_TARGET_SCALE_FLOOR, min(scalar_raw, cap)))
    scaled_row = raw_row * scalar
    try:  # noqa: SIM105
        del grid_1h
    except NameError:
        pass
    gc.collect()
    return scaled_row, runtime.reference_daily_returns, scalar


def advance_to_date(
    params: LiveStrategyParams,
    runtime: LiveRuntime,
    weights_path: Path,
    data_root: str,
    target: pd.Timestamp,
    *,
    artifact_key: SecretStr | None = None,
    max_catchup_days: int = 30,
    portfolio_state_dir: Path | None = None,
    mode: str = "shadow",
) -> tuple[LiveRuntime, int, float]:
    from src.live.deployed_weights import append_weight_row, load_weights_frame

    target_dt = pd.Timestamp(target).tz_convert("UTC").normalize() if pd.Timestamp(target).tzinfo is not None else pd.Timestamp(target).tz_localize("UTC").normalize()
    last_dt = pd.Timestamp(runtime.last_decision_date).tz_convert("UTC").normalize() if pd.Timestamp(runtime.last_decision_date).tzinfo is not None else pd.Timestamp(runtime.last_decision_date).tz_localize("UTC").normalize()
    if target_dt <= last_dt:
        return runtime, 0, 1.0
    gap_days = (target_dt - last_dt).days
    if gap_days > max_catchup_days:
        scaled_row, new_reference, _scalar = compute_signal_row(
            params, runtime, data_root, target_dt, portfolio_state_dir=portfolio_state_dir, mode=mode
        )
        appended = append_weight_row(Path(weights_path), target_dt, scaled_row, artifact_key=artifact_key)
        updated = LiveRuntime(
            schema_version=runtime.schema_version,
            params_digest=runtime.params_digest,
            last_decision_date=target_dt,
            held_target_row={str(k): float(v) for k, v in scaled_row.items() if pd.notna(v)},
            reference_daily_returns=new_reference,
        )
        return updated, (1 if appended else 0), float(_scalar)

    cur_runtime = runtime
    rows_appended = 0
    last_scalar = 1.0
    cur = last_dt + pd.Timedelta(days=1)
    while cur <= target_dt:
        frame = load_weights_frame(Path(weights_path), artifact_key=artifact_key)
        if not frame.empty and cur in pd.DatetimeIndex(frame.index):
            cur_runtime = LiveRuntime(
                schema_version=cur_runtime.schema_version,
                params_digest=cur_runtime.params_digest,
                last_decision_date=cur,
                held_target_row=dict(cur_runtime.held_target_row),
                reference_daily_returns=cur_runtime.reference_daily_returns,
            )
            cur += pd.Timedelta(days=1)
            continue
        scaled_row, new_reference, _scalar = compute_signal_row(
            params, cur_runtime, data_root, cur, portfolio_state_dir=portfolio_state_dir, mode=mode
        )
        last_scalar = float(_scalar)
        appended = append_weight_row(Path(weights_path), cur, scaled_row, artifact_key=artifact_key)
        if appended:
            rows_appended += 1
        cur_runtime = LiveRuntime(
            schema_version=cur_runtime.schema_version,
            params_digest=cur_runtime.params_digest,
            last_decision_date=cur,
            held_target_row={str(k): float(v) for k, v in scaled_row.items() if pd.notna(v)},
            reference_daily_returns=new_reference,
        )
        cur += pd.Timedelta(days=1)
    return cur_runtime, rows_appended, last_scalar
