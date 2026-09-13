"""Live signal step: realized reference and rolling compute."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import SecretStr

import src.mhs.scaling as _scaling
from src.common.errors import DataIntegrityError
from src.live.portfolio_state import default_portfolio_state_dir
from src.mhs.live_runtime import LiveRuntime
from src.mhs.live_strategy import LiveStrategyParams
from src.mhs.scaling import compute_exposure_scale

logger = logging.getLogger("LiveSignalStep")

PANEL_HISTORY_REFERENCE_SYMBOL: str = "BTCUSDT"

try:
    from src.mhs.evaluation import _build_fold_target_weights  # noqa: F401
except Exception:  # noqa: BLE001,S110

    def _build_fold_target_weights(*_a: Any, **_k: Any) -> Any:  # type: ignore[misc]
        raise DataIntegrityError("missing _build_fold_target_weights")


def _assert_panel_history_available(
    data_root: str,
    panel_start: pd.Timestamp,
    reference_symbol: str = PANEL_HISTORY_REFERENCE_SYMBOL,
) -> None:
    path = Path(data_root) / "1h" / f"{reference_symbol}.parquet"
    try:
        df = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:
        raise DataIntegrityError(f"panel history unavailable: {path}: {exc}") from exc
    if df.empty or "timestamp" not in df.columns:
        raise DataIntegrityError(f"panel history unavailable: {path}")
    min_val = int(pd.to_numeric(df["timestamp"], errors="coerce").min())
    min_ts = pd.Timestamp(min_val, unit="ms", tz="UTC")
    ps = pd.Timestamp(panel_start)
    ps = ps.tz_localize("UTC") if ps.tzinfo is None else ps.tz_convert("UTC")
    if min_ts > ps:
        raise DataIntegrityError(f"panel history starts {min_ts} after panel_start {ps}")


def realized_equity(portfolio_state_dir: Path, mode: str, *, bt_end: pd.Timestamp) -> pd.Series:
    p = Path(portfolio_state_dir)
    if not p.exists():
        return pd.Series(dtype="float64")
    shards = sorted(p.glob("*.parquet"))
    if not shards:
        return pd.Series(dtype="float64")
    frames: list[pd.DataFrame] = []
    for shard in shards:
        df = pd.read_parquet(shard)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.Series(dtype="float64")
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty or "mode" not in combined.columns or "decision_time" not in combined.columns or "equity_usdt" not in combined.columns:
        return pd.Series(dtype="float64")
    # filter mode
    combined = combined[combined["mode"] == mode]
    if combined.empty:
        return pd.Series(dtype="float64")
    # decision_time parsing
    combined["decision_time"] = pd.to_datetime(combined["decision_time"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["decision_time"])
    # equity finite and >0
    eq = pd.to_numeric(combined["equity_usdt"], errors="coerce")
    mask = np.isfinite(eq.to_numpy(dtype="float64")) & (eq > 0)
    combined = combined[mask]
    if combined.empty:
        return pd.Series(dtype="float64")
    # bt_end filter strict >
    bt = pd.Timestamp(bt_end)
    if bt.tzinfo is None:  # noqa: SIM108
        bt = bt.tz_localize("UTC")
    else:
        bt = bt.tz_convert("UTC")
    combined = combined[combined["decision_time"] > bt]
    if combined.empty:
        return pd.Series(dtype="float64")
    combined = combined.sort_values("decision_time")
    combined = combined.drop_duplicates("decision_time", keep="last")
    combined = combined.set_index("decision_time")
    combined = combined.sort_index()
    equity = pd.to_numeric(combined["equity_usdt"], errors="coerce").astype("float64")
    equity.index = pd.DatetimeIndex(equity.index).tz_convert("UTC")
    equity = equity.sort_index()
    return equity


def descale_realized_returns(equity: pd.Series, applied_scale: pd.Series) -> pd.Series:
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    if equity is None or len(equity) < 2:
        return empty
    idx = pd.DatetimeIndex(equity.index).tz_convert("UTC")
    eq_vals = equity.to_numpy(dtype="float64")
    scale_index_set = set(pd.DatetimeIndex(applied_scale.index)) if applied_scale is not None and len(applied_scale) > 0 else set()
    # find first pair whose prev is in applied_scale
    start = None
    for i in range(1, len(idx)):
        if idx[i - 1] in scale_index_set:
            start = i
            break
    if start is None:
        return empty
    out_idx: list[pd.Timestamp] = []
    out_vals: list[float] = []
    for i in range(start, len(idx)):
        prev = idx[i - 1]
        cur = idx[i]
        if prev not in scale_index_set:
            raise DataIntegrityError(f"applied scale missing for {prev}")
        s = float(applied_scale.loc[prev])
        if not np.isfinite(s) or s <= 0:
            raise DataIntegrityError(f"applied scale non-positive/non-finite at {prev}: {s!r}")
        prev_eq = float(eq_vals[i - 1])
        cur_eq = float(eq_vals[i])
        val = (cur_eq / prev_eq - 1.0) / s
        out_idx.append(cur)
        out_vals.append(val)
    out = pd.Series(out_vals, index=pd.DatetimeIndex(out_idx, tz="UTC"), dtype="float64")
    out = out.sort_index()
    return out


def decision_mark_row(
    symbols: Iterable[str],
    date: pd.Timestamp,
    mark_path_fn: Callable[[str], Path],
) -> pd.Series:
    dt = pd.Timestamp(date)
    dt = dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")
    target_ms = int((dt - pd.Timedelta(hours=1)).value // 1_000_000)
    vals: dict[str, float] = {}
    for sym in symbols:
        path = Path(mark_path_fn(str(sym)))
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["timestamp", "close"])
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        cl = pd.to_numeric(df["close"], errors="coerce")
        hit = df.loc[ts == target_ms]
        if hit.empty:
            continue
        close_val = float(cl.loc[hit.index[0]])
        if not np.isfinite(close_val) or close_val <= 0:
            continue
        vals[str(sym)] = float(close_val)
    return pd.Series(vals, dtype="float64", name=dt)


def _synthetic_fold(date: pd.Timestamp, params: LiveStrategyParams) -> Any:
    from src.mhs.evidence import AnchoredPurgedFold

    dt = pd.Timestamp(date).tz_convert("UTC").normalize() if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date).tz_localize("UTC").normalize()
    window_days = int(params.policy.signal_window.panel_window_days)
    warmup_hours = int(params.policy.signal_window.fold_panel_warmup_hours)
    purge_hours = int(params.policy.signal_window.committee_purge_hours)
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
    applied_scale: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    if not data_root:
        from src.common.paths import FUTURES_DATA_DIR

        data_root = str(FUTURES_DATA_DIR / "ohlcv")

    dt = pd.Timestamp(date).tz_convert("UTC").normalize() if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date).tz_localize("UTC").normalize()
    _assert_panel_history_available(data_root, dt - pd.Timedelta(days=int(params.policy.signal_window.panel_window_days)))
    fold = _synthetic_fold(dt, params)
    request = params.policy.target_weights.to_request()  # TargetWeightPolicy.to_request seam
    funding_by_symbol = _load_funding_by_symbol(data_root)

    target_weights, _signal_available_at, _minute_roster, grid_1h = _build_fold_target_weights(  # noqa: RUF059
        data_root,
        fold,
        request,
        funding_by_symbol,
        slow_horizon_override=int(params.policy.slow_horizon_hours),
        committee_member_weights=dict(params.policy.committee_member_weights),
        require_minute_roster=False,
        panel_warmup_hours=int(params.policy.signal_window.fold_panel_warmup_hours),
        committee_oos_start=params.policy.signal_window.committee_oos_start,
        apply_rebalance_deadband=False,
    )
    if dt not in target_weights.index:
        del grid_1h
        gc.collect()
        raise DataIntegrityError(f"decision_time {dt} not in scored window")
    pre = target_weights.loc[dt]

    seed = pd.Series(runtime.held_target_row, dtype="float64") if runtime.held_target_row else None
    prescale_row = _scaling._apply_rebalance_deadband(pre.to_frame().T, seed_row=seed).iloc[0]

    psd = portfolio_state_dir if portfolio_state_dir is not None else default_portfolio_state_dir()
    bt_end = pd.Timestamp(params.backtest_window[1])
    if bt_end.tzinfo is None:  # noqa: SIM108
        bt_end = bt_end.tz_localize("UTC")
    else:
        bt_end = bt_end.tz_convert("UTC")
    bt_end = bt_end.normalize()
    warmup_src = runtime.reference_daily_returns
    if not warmup_src.empty:
        if warmup_src.index.tz is None:
            warmup_src = warmup_src.copy()
            warmup_src.index = warmup_src.index.tz_localize("UTC")
        elif str(warmup_src.index.tz) != "UTC":
            warmup_src = warmup_src.tz_convert("UTC")
        if not warmup_src.index.is_monotonic_increasing:
            warmup_src = warmup_src.sort_index()
        warmup = warmup_src[warmup_src.index <= bt_end]
    else:
        warmup = pd.Series(dtype="float64")
        warmup.index = pd.DatetimeIndex([], tz="UTC")

    _applied = pd.Series(dtype="float64") if applied_scale is None else applied_scale
    forward = descale_realized_returns(realized_equity(psd, mode, bt_end=bt_end), _applied)
    if not forward.empty and forward.index[-1] >= dt:
        raise DataIntegrityError(f"realized record must precede decision_time {dt}")
    placeholder = pd.Series([0.0], index=pd.DatetimeIndex([dt], tz="UTC"), dtype="float64")
    reference = placeholder if forward.empty else pd.concat([forward, placeholder])
    warmup_arg = warmup if not warmup.empty else None

    scale_series = compute_exposure_scale(reference, params.policy.sizing, warmup_returns=warmup_arg)
    scalar_raw = float(scale_series.loc[dt])
    cap = float(params.policy.sizing.exposure_cap)
    floor = float(params.policy.sizing.scale_floor)
    scalar = float(max(floor, min(scalar_raw, cap)))
    scaled_row = prescale_row * scalar
    try:  # noqa: SIM105
        del grid_1h
    except NameError:
        pass
    gc.collect()
    return scaled_row, prescale_row, scalar


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
    import src.market_data.services.futures_collection as _futures_collection
    from src.live.deployed_weights import (
        EXPOSURE_SCALE_COLUMN,
        EXPOSURE_SCALE_KEEP_ROWS,
        append_weight_row,
        decision_marks_path,
        exposure_scale_path,
        load_weights_frame,
    )

    target_dt = pd.Timestamp(target).tz_convert("UTC").normalize() if pd.Timestamp(target).tzinfo is not None else pd.Timestamp(target).tz_localize("UTC").normalize()
    last_dt = pd.Timestamp(runtime.last_decision_date).tz_convert("UTC").normalize() if pd.Timestamp(runtime.last_decision_date).tzinfo is not None else pd.Timestamp(runtime.last_decision_date).tz_localize("UTC").normalize()
    if target_dt <= last_dt:
        return runtime, 0, 1.0
    _scale_frame = load_weights_frame(exposure_scale_path(Path(weights_path)), artifact_key=artifact_key)
    if _scale_frame.empty or EXPOSURE_SCALE_COLUMN not in _scale_frame.columns:
        applied = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    else:
        applied = _scale_frame[EXPOSURE_SCALE_COLUMN].astype("float64").sort_index()
    gap_days = (target_dt - last_dt).days
    if gap_days > max_catchup_days:
        scaled_row, prescale_row, _scalar = compute_signal_row(
            params, runtime, data_root, target_dt, portfolio_state_dir=portfolio_state_dir, mode=mode, applied_scale=applied
        )
        # 스케일/마크를 weights 행보다 먼저 기록: weights 행이 있으면 스케일도 반드시 존재한다.
        append_weight_row(exposure_scale_path(Path(weights_path)), target_dt, pd.Series({EXPOSURE_SCALE_COLUMN: float(_scalar)}, dtype="float64"), artifact_key=artifact_key, keep_rows=EXPOSURE_SCALE_KEEP_ROWS)
        _syms = [str(s) for s in scaled_row.index if pd.notna(scaled_row[s]) and float(scaled_row[s]) != 0.0]
        _mark_fn = lambda s: _futures_collection._mark_price_path(s, "1h")  # noqa: E731
        append_weight_row(decision_marks_path(Path(weights_path)), target_dt, decision_mark_row(_syms, target_dt, _mark_fn), artifact_key=artifact_key)
        appended = append_weight_row(Path(weights_path), target_dt, scaled_row, artifact_key=artifact_key)
        updated = LiveRuntime(
            schema_version=runtime.schema_version,
            params_digest=runtime.params_digest,
            last_decision_date=target_dt,
            held_target_row={str(k): float(v) for k, v in prescale_row.items() if pd.notna(v)},
            reference_daily_returns=runtime.reference_daily_returns,
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
        scaled_row, prescale_row, _scalar = compute_signal_row(
            params, cur_runtime, data_root, cur, portfolio_state_dir=portfolio_state_dir, mode=mode, applied_scale=applied
        )
        last_scalar = float(_scalar)
        # 스케일/마크를 weights 행보다 먼저 기록: weights 행이 있으면 스케일도 반드시 존재한다.
        append_weight_row(exposure_scale_path(Path(weights_path)), cur, pd.Series({EXPOSURE_SCALE_COLUMN: float(_scalar)}, dtype="float64"), artifact_key=artifact_key, keep_rows=EXPOSURE_SCALE_KEEP_ROWS)
        _syms = [str(s) for s in scaled_row.index if pd.notna(scaled_row[s]) and float(scaled_row[s]) != 0.0]
        _mark_fn = lambda s: _futures_collection._mark_price_path(s, "1h")  # noqa: E731
        append_weight_row(decision_marks_path(Path(weights_path)), cur, decision_mark_row(_syms, cur, _mark_fn), artifact_key=artifact_key)
        appended = append_weight_row(Path(weights_path), cur, scaled_row, artifact_key=artifact_key)
        if appended:
            rows_appended += 1
        if applied.empty:
            applied = pd.Series([float(_scalar)], index=pd.DatetimeIndex([cur], tz="UTC"), dtype="float64")
        else:
            _new = pd.Series([float(_scalar)], index=pd.DatetimeIndex([cur], tz="UTC"), dtype="float64")
            applied = pd.concat([applied, _new]).sort_index()
        cur_runtime = LiveRuntime(
            schema_version=cur_runtime.schema_version,
            params_digest=cur_runtime.params_digest,
            last_decision_date=cur,
            held_target_row={str(k): float(v) for k, v in prescale_row.items() if pd.notna(v)},
            reference_daily_returns=cur_runtime.reference_daily_returns,
        )
        cur += pd.Timedelta(days=1)
    return cur_runtime, rows_appended, last_scalar
