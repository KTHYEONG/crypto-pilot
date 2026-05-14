from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import FUTURES_DATA_DIR
from src.core.optimization.opt_utils import compute_segment_merge_index
from src.domain.futures.data_loader import DataCollector
from src.domain.futures.ml_pipeline.pipeline_runner import _enrich_with_gp_features
from src.domain.futures.optimization.dashboard import REGIME_NAMES
from src.domain.futures.optimization.optimizer import inject_cs_momentum_ranks

_logger: logging.Logger = logging.getLogger("opt_data_utils")


def infer_regime_codes(df: pd.DataFrame) -> np.ndarray:
    """Infer HMM regime codes from probability columns."""
    n = len(df)

    def _float_col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(n, dtype=np.float64)
        # Force a writable array; some pandas-backed arrays can be read-only views.
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    bull = _float_col("hmm_prob_bull_calm")
    bull += _float_col("hmm_prob_bull_vol_up")
    bear = _float_col("hmm_prob_bear_trend")
    chop = _float_col("hmm_prob_chop")
    crisis = _float_col("hmm_prob_crisis")
    probs = np.column_stack(
        [
            np.nan_to_num(bull, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(bear, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(chop, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(crisis, nan=0.0, posinf=0.0, neginf=0.0),
        ]
    )
    return np.argmax(probs, axis=1).astype(np.int64, copy=False)


def compute_oos_regime_attribution(
    oos_port: dict[str, Any],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, Any]:
    """Compute performance attribution by market regime."""
    time_counts = np.zeros(len(REGIME_NAMES), dtype=np.int64)
    total_symbol_bars = 0
    for sym in symbols:
        smap = oos_data_maps.get(sym, {})
        df = smap.get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        o0 = int(smap.get(f"oos_start_idx_{tf}", 0))
        o0 = max(0, min(o0, len(df)))
        oos_df = df.iloc[o0:]
        if oos_df.empty:
            continue
        rc = infer_regime_codes(oos_df)
        time_counts += np.bincount(rc, minlength=len(REGIME_NAMES))
        total_symbol_bars += int(rc.size)

    trades_df = oos_port.get("trades_df", pd.DataFrame())
    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame()
    n_trades = len(trades_df)
    trade_codes = np.full(n_trades, -1, dtype=np.int64)

    aligned_master = oos_port.get("aligned_master_index")
    full_signal_dfs = oos_port.get("full_signal_dfs", {})
    if n_trades > 0 and isinstance(full_signal_dfs, dict):
        aligned_ns = np.array([], dtype=np.int64)
        if isinstance(aligned_master, pd.Series):
            aligned_ns = (
                pd.to_datetime(aligned_master, errors="coerce")
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64, copy=False)
            )
        elif aligned_master is not None:
            aligned_idx = pd.Index(np.asarray(aligned_master).ravel())
            aligned_ns = (
                pd.to_datetime(aligned_idx, errors="coerce")
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64, copy=False)
            )
        if aligned_ns.size > 0:
            nat_i64 = np.iinfo(np.int64).min
            sym_map: dict[str, pd.Series] = {}
            for sym in np.unique(trades_df["symbol"].astype(str).to_numpy()):
                sdf = full_signal_dfs.get(sym)
                if not isinstance(sdf, pd.DataFrame) or sdf.empty or "datetime" not in sdf.columns:
                    continue
                dt_ns = (
                    pd.to_datetime(sdf["datetime"], errors="coerce")
                    .to_numpy(dtype="datetime64[ns]")
                    .astype(np.int64, copy=False)
                )
                valid_dt = dt_ns != nat_i64
                if not valid_dt.any():
                    continue
                rc = infer_regime_codes(sdf)
                sr = pd.Series(rc[valid_dt], index=dt_ns[valid_dt], dtype=np.int64)
                sym_map[sym] = sr.groupby(level=0).last()

            entry_idx = pd.to_numeric(trades_df["entry_idx"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            sym_arr = trades_df["symbol"].astype(str).to_numpy()
            for sym, sr in sym_map.items():
                pos = np.where(sym_arr == sym)[0]
                if pos.size == 0:
                    continue
                e = entry_idx[pos]
                finite = np.isfinite(e)
                if not finite.any():
                    continue
                loc = e[finite].astype(np.int64)
                inb = (loc >= 0) & (loc < aligned_ns.size)
                if not inb.any():
                    continue
                valid_rows = pos[finite][inb]
                keys = aligned_ns[loc[inb]]
                mapped = sr.reindex(keys).to_numpy()
                ok = ~pd.isna(mapped)
                if ok.any():
                    trade_codes[valid_rows[ok]] = mapped[ok].astype(np.int64, copy=False)

    pnl = (
        pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if n_trades > 0
        else np.array([], dtype=np.float64)
    )
    side_arr = (
        trades_df["side"].astype(str).to_numpy() if n_trades > 0 else np.array([], dtype=object)
    )
    entry_idx = (
        pd.to_numeric(trades_df["entry_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        if n_trades > 0
        else np.array([], dtype=np.int64)
    )
    sym_arr = (
        trades_df["symbol"].astype(str).to_numpy() if n_trades > 0 else np.array([], dtype=object)
    )

    regime_metrics: dict[str, dict[str, float | int]] = {}
    for ridx, rname in enumerate(REGIME_NAMES):
        r_mask = trade_codes == ridx
        r_count = int(r_mask.sum())
        if r_count > 0:
            r_pnl = pnl[r_mask]
            gains = float(r_pnl[r_pnl > 0.0].sum())
            losses = abs(float(r_pnl[r_pnl < 0.0].sum()))
            if losses == 0.0:
                pf = 5.0 if gains > 0.0 else 1.0
            else:
                pf = gains / losses
            win_rate = float(np.mean(r_pnl > 0.0) * 100.0)
            avg_pnl = float(np.mean(r_pnl))
        else:
            win_rate = 0.0
            pf = 1.0
            avg_pnl = 0.0
        time_pct = float(100.0 * time_counts[ridx] / max(total_symbol_bars, 1))
        regime_metrics[rname] = {
            "time_pct": time_pct,
            "trade_count": r_count,
            "win_rate": win_rate,
            "profit_factor": float(pf),
            "avg_pnl": avg_pnl,
        }

    chop_idx = REGIME_NAMES.index("chop")
    chop_mask = trade_codes == chop_idx
    chop_trade_count = int(chop_mask.sum())
    chop_losses = abs(float(pnl[chop_mask & (pnl < 0.0)].sum())) if n_trades > 0 else 0.0
    total_losses = abs(float(pnl[pnl < 0.0].sum())) if n_trades > 0 else 0.0
    chop_loss_share = float(chop_losses / max(total_losses, 1e-12))
    chop_trade_share = float(chop_trade_count / max(n_trades, 1))

    flip_count = 0
    flip_pairs = 0
    if n_trades > 1 and chop_trade_count > 1:
        for sym in np.unique(sym_arr[chop_mask]):
            s_mask = (sym_arr == sym) & chop_mask
            if int(s_mask.sum()) < 2:
                continue
            ord_idx = np.argsort(entry_idx[s_mask], kind="mergesort")
            s_side = side_arr[s_mask][ord_idx]
            flip_count += int(np.sum(s_side[1:] != s_side[:-1]))
            flip_pairs += int(max(s_side.size - 1, 0))
    chop_flip_proxy = float(flip_count / max(flip_pairs, 1))

    return {
        "regime_metrics": regime_metrics,
        "chop_loss_share": chop_loss_share,
        "chop_trade_share": chop_trade_share,
        "chop_flip_proxy": chop_flip_proxy,
        "chop_flip_proxy_label": "side_switch_rate_within_chop_trades_by_symbol (proxy)",
        "trade_regime_coverage_pct": float(
            100.0 * np.mean(trade_codes >= 0) if n_trades > 0 else 0.0
        ),
    }


def assert_oos_gp_signal_alive(
    oos_data_maps: dict[str, dict[str, Any]], valid_symbols: list[str], tf: str
) -> None:
    """Verify that ML signals are not dead in the OOS period."""
    for sym in valid_symbols[: min(5, len(valid_symbols))]:
        df = oos_data_maps[sym][tf]
        if "ml_alpha_00" not in df.columns:
            raise RuntimeError(f"Pre-OOS: {sym} missing ml_alpha_00.")
        gp = df["ml_alpha_00"]
        if not pd.api.types.is_numeric_dtype(gp):
            raise RuntimeError(f"Pre-OOS: {sym} ml_alpha_00 non-numeric dtype={gp.dtype}")
        o0 = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
        oos_std = float(pd.to_numeric(gp.iloc[o0:], errors="coerce").std(ddof=0) or 0.0)
        if oos_std < 1e-6:
            raise RuntimeError(f"Pre-OOS: {sym} OOS ml_alpha_00 std={oos_std:.2e} (dead signal).")


def load_single_symbol_data(
    sym: str,
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
    target_tfs: list[str] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Load and enrich data for a single symbol across multiple timeframes."""
    try:
        temp_is: dict[str, Any] = {}
        temp_oos: dict[str, Any] = {}
        insufficient = False
        collector = DataCollector()

        tfs_to_load = set(target_tfs) if target_tfs else {tf, "1d", "1h", "4h"}

        # [Optimization] Pre-load funding and metrics data once to avoid redundant I/O per TF
        funding_df = None
        metrics_df = None
        if not skip_metrics:
            f_path = Path(FUTURES_DATA_DIR) / f"{sym.replace('/', '_')}_funding.parquet"
            if f_path.exists():
                try:
                    funding_df = pd.read_parquet(f_path)
                except Exception:
                    _logger.warning("Failed to load funding data for %s", sym)

            m_path = Path(FUTURES_DATA_DIR) / f"{sym.replace('/', '_')}_metrics.parquet"
            if m_path.exists():
                try:
                    metrics_df = pd.read_parquet(m_path)
                except Exception:
                    _logger.warning("Failed to load metrics data for %s", sym)

        for tf_l in tfs_to_load:
            raw_df = collector.collect_and_save(sym, tf_l, fetch_start, end)
            if raw_df is None or raw_df.empty:
                insufficient = True
                break

            if "datetime" not in raw_df.columns:
                raw_df = raw_df.reset_index()
                if "datetime" not in raw_df.columns and len(raw_df.columns) > 0:
                    raw_df = raw_df.rename(columns={str(raw_df.columns[0]): "datetime"})

            try:
                # [Optimization] Use localized merge logic to benefit from pre-loaded data
                df = raw_df.copy()
                if funding_df is not None and not funding_df.empty:
                    df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6
                    f_tmp = funding_df.copy()
                    f_tmp["timestamp"] = (
                        pd.to_datetime(f_tmp["timestamp"], unit="ms").astype("int64") // 10**6
                    )
                    exclude_fr = ["datetime", "symbol"]
                    cols_fr = [c for c in f_tmp.columns if c not in exclude_fr]
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        f_tmp[cols_fr].sort_values("timestamp"),
                        on="timestamp",
                        direction="backward",
                    )

                if metrics_df is not None and not metrics_df.empty:
                    if "timestamp" not in df.columns:
                        df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6
                    m_tmp = metrics_df.copy()
                    m_tmp["timestamp"] = pd.to_datetime(m_tmp["datetime"]).astype("int64") // 10**6
                    exclude_m = ["timestamp", "datetime", "create_time", "symbol"]
                    cols_m = [c for c in m_tmp.columns if c not in exclude_m]
                    df = pd.merge_asof(
                        df.sort_values("timestamp"),
                        m_tmp[["timestamp", *cols_m]].sort_values("timestamp"),
                        on="timestamp",
                        direction="backward",
                    )

                # Enrich with GP features
                if not skip_metrics:
                    df = _enrich_with_gp_features(df, tf=tf_l)
            except Exception as e:
                _logger.error("[%s] Merge/Enrich failed: %s", sym, e)
                insufficient = True
                break

            if df is None or df.empty or "datetime" not in df.columns:
                insufficient = True
                break

            df.reset_index(drop=True, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            is_start_dt = pd.Timestamp(start, tz="UTC")
            is_end_dt = pd.Timestamp(is_end, tz="UTC")

            is_mask = df["datetime"] < is_end_dt
            is_end_idx = int(is_mask.to_numpy().sum())

            # [Dynamic Quality Gate]
            min_bars_map = {"1h": 2000, "4h": 500, "1d": 300}
            min_bars_threshold = min_bars_map.get(tf_l, 300)

            if is_end_idx < min_bars_threshold:
                _logger.debug(
                    "[%s] %s history too short (%d < %d)",
                    sym,
                    tf_l,
                    is_end_idx,
                    min_bars_threshold,
                )
                insufficient = True
                break

            temp_is[tf_l] = df.iloc[:is_end_idx].copy()
            mask = temp_is[tf_l]["datetime"] >= is_start_dt
            temp_is[f"is_start_idx_{tf_l}"] = int(mask.to_numpy().argmax()) if mask.any() else 0
            temp_oos[tf_l] = df
            mask_oos = df["datetime"] >= is_end_dt
            idx_oos = int(mask_oos.to_numpy().argmax()) if mask_oos.any() else len(df)
            temp_oos[f"oos_start_idx_{tf_l}"] = idx_oos

        if insufficient:
            return sym, None, None, True

        temp_is[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_is[tf], temp_is["1d"])
        temp_oos[f"merge_idx_{tf}"] = compute_segment_merge_index(temp_oos[tf], temp_oos["1d"])
        return sym, temp_is, temp_oos, False
    except Exception as e:
        _logger.debug("[%s] Critical load failure: %s", sym, e)
        return sym, None, None, True


def load_futures_data_maps_for_symbols(
    symbols: list[str],
    tf: str,
    fetch_start: str,
    start: str,
    is_end: str,
    end: str,
    skip_metrics: bool = False,
    target_tfs: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Load data for multiple symbols in parallel and inject momentum ranks."""
    data_maps: dict[str, dict[str, Any]] = {}
    oos_data_maps: dict[str, dict[str, Any]] = {}
    valid_symbols: list[str] = []

    # [Fix] Filter out non-ASCII symbols before processing
    symbols = [s for s in symbols if all(ord(c) < 128 for c in s)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                load_single_symbol_data,
                sym,
                tf,
                fetch_start,
                start,
                is_end,
                end,
                skip_metrics,
                target_tfs,
            )
            for sym in symbols
        ]
        for f in concurrent.futures.as_completed(futures):
            sym, t_is, t_oos, insufficient = f.result()
            if not insufficient and t_is and t_oos:
                data_maps[sym], oos_data_maps[sym] = t_is, t_oos
                valid_symbols.append(sym)

    if len(valid_symbols) > 1:
        inject_cs_momentum_ranks(data_maps, valid_symbols, tf)
        inject_cs_momentum_ranks(oos_data_maps, valid_symbols, tf)

    return data_maps, oos_data_maps, valid_symbols
