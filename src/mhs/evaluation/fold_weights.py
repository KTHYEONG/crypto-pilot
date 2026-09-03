# mypy: ignore-errors
# ruff: noqa: F401, F821, I001, E402
from __future__ import annotations  # mypy: ignore-errors

import dataclasses
import os

import numpy as np
import pandas as pd

from src.mhs import research_go as _research_go
from src.mhs import scaling as _scaling
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.marks import (
    _fill_mark_parity_eligibility,
    _get_symbol_mark_frame,
    _pit_execution_mask,
)
from src.mhs.evidence import AnchoredPurgedFold
from src.mhs.execution import bar_funding_panel
from src.mhs.horizons import realized_vol
from src.mhs.books import inverse_realized_vol_tilt, phase_tranche_book, portfolio_rebalance_trigger, renormalize_within_mask, scale_book_to_target_gross
from src.mhs.funding import funding_carry_execution_book
from src.mhs.features import build_feature_books
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    COMMITTEE_TRANCHE_COUNT,
    FOLD_PANEL_WARMUP_HOURS,
    REBALANCE_TRACKING_ERROR_THRESHOLD,
)
from src.mhs.regime import beta_neutralize_weights, causal_market_beta, crash_regime_tilt_weights
from src.mhs.types import BOOK_BLEND_WEIGHTS, BOOK_SPECS, BookSpec, COMMITTEE_OOS_START, COMMITTEE_REGIME_ADAPTIVE_WINDOW, CRASH_REGIME_REFERENCE_SYMBOLS, FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS

from . import books, committee, folds, integrity, specs


def _build_fold_target_weights(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    slow_horizon_override: int | None = None,
    committee_member_weights: dict[str, float] | None = None,
    *,
    deadband_seed_row: pd.Series | None = None,
    require_minute_roster: bool = True,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str], pd.DatetimeIndex]:
    """Construct one fold's PIT decision targets with the quality calibration.

    Returns ``(target_weights, signal_available_at, minute_roster, grid_1h)``:
    the blend decision targets over the validation window, their ``+1h``
    signal-availability stamps, the minute-data roster, and the 1h feature
    grid. The 1h panel is sliced to ``[validation_start - warmup, validation_end]``
    (spec §3.1) so warm-up history feeds the 720-bar eligibility lookback and
    the 168h slow horizon without holding the full ``[train_start, validation_end]``
    panel. Signal quality (spec §3.2) applies EMA smoothing on each book, a
    volatility-regime cash scale, and the turnover deadband cap on the final
    blend targets. All objects are local to this builder and released when it
    returns, keeping per-fold peak memory bounded.

    ``deadband_seed_row`` (opt-in, default ``None`` reproduces every existing
    call byte-identically) threads through to ``_apply_rebalance_deadband`` so
    a live daily refresh over a rebuilt rolling window continues the deadband
    from an externally carried decision instead of resetting at this window's
    own first row (I-DEADBAND-CONTINUITY). Only defined under
    ``rebalance_filter='per_symbol_deadband'``; combining it with
    ``'portfolio_trigger'`` raises ``ValueError``.
    """
    ts = fold.train_start
    vs = fold.validation_start
    ve = fold.validation_end
    import src.mhs.evaluation as ev
    panel_start = max(ts, vs - pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS))
    _panel_columns = (
        ("close", "open", "quote_vol", "taker_buy_quote")
        if request.committee_capital
        else ("close", "open", "quote_vol")
    )
    panel = load_base_panel(
        root, "1h", _panel_columns, panel_start, ve,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    taker_buy_quote = panel["taker_buy_quote"] if request.committee_capital else None
    del panel
    grid_1h = close.index
    symbols = list(close.columns)
    funded = [
        s for s in symbols
        if s in funding_by_symbol and s not in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
    if not funded:
        raise RuntimeError("no fold symbol has funding coverage")
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    if taker_buy_quote is not None:
        taker_buy_quote = taker_buy_quote[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = bar_funding_panel(funding_window, grid_1h)
    del funding_window
    aligned_symbols = list(bar_funding.columns)
    if not aligned_symbols:
        raise RuntimeError("no fold symbol has causally aligned funding coverage")
    close = close[aligned_symbols]
    opens = opens[aligned_symbols]
    quote_vol = quote_vol[aligned_symbols]
    bar_funding = bar_funding[aligned_symbols]
    if taker_buy_quote is not None:
        taker_buy_quote = taker_buy_quote[aligned_symbols]

    eligible = liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    # Unlike run_mhs_horizon_diagnostic (which clears at entry), this function
    # is also called directly outside a diagnostic run (fork worker per fold,
    # or a unit test calling it standalone), so the process-level
    # _get_symbol_mark_frame cache is never guaranteed fresh for this root
    # otherwise -- a stale frame from a prior call against a different
    # data_root/mark fixture would silently leak in (lru_cache keys on
    # (symbol, timeframe) only, never on data_root).
    _get_symbol_mark_frame.cache_clear()
    eligible, _ = _fill_mark_parity_eligibility(close, eligible, request.fill_mark_parity_gate)
    log_close = np.log(close)
    if not request.committee_capital:
        del close
    fast = BOOK_SPECS["fast_reversal"]
    slow = (
        dataclasses.replace(BOOK_SPECS["slow_momentum"], horizon_hours=slow_horizon_override)
        if slow_horizon_override is not None
        else BOOK_SPECS["slow_momentum"]
    )
    fast_grid = pd.date_range(panel_start, ve, freq="6h", tz="UTC")
    slow_grid = pd.date_range(panel_start, ve, freq="24h", tz="UTC")
    fast_ema = specs._signal_ema_span(fast.band.sign, fast.horizon_hours, fast.step_hours)
    slow_ema = specs._signal_ema_span(slow.band.sign, slow.horizon_hours, slow.step_hours)
    w_fast = books._book_weights(log_close, eligible, fast, fast_grid, ema_span=fast_ema)
    execution_mask = _pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    if request.fast_book_mode == "horizon_ensemble":
        w_fast_execution = books._horizon_ensemble_execution_weights(
            log_close, eligible, execution_mask, fast, fast_grid,
            "horizon_ensemble", "raw", fast_ema,
        )
    else:
        w_fast_tilted = ev.inverse_realized_vol_tilt(
            w_fast, realized_vol(log_close, fast.horizon_hours).reindex(fast_grid),
        )
        w_fast_execution = ev.renormalize_within_mask(
            w_fast_tilted, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
        )
    w_slow_execution = books._horizon_ensemble_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        request.slow_book_mode, request.ensemble_signal, slow_ema,
    )
    # I-SINGLE-CONFIGURATION: one causal-beta computation shared by the legacy
    # slow-book neutralize and the committee execution book below.
    causal_beta = (
        causal_market_beta(
            log_close, eligible,
            CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS,
        )
        if request.beta_neutralize
        else None
    )
    if causal_beta is not None:
        w_slow_execution = beta_neutralize_weights(
            w_slow_execution,
            causal_beta.reindex(w_slow_execution.index),
            execution_mask.reindex(w_slow_execution.index).fillna(False),
            slow.min_symbols,
        )
    # The trend sleeve position rides the same 24h slow grid and must be
    # computed while `eligible` is still alive; it is released right after, so
    # only the tiny position Series survives (memory-order contract).
    trend_position = (
        folds._trend_sleeve_position(log_close, eligible, slow_grid)
        if (request.trend_sleeve and request.trend_sleeve_gross > 0.0)
        else None
    )
    del eligible
    if not request.committee_capital:
        del quote_vol
    del w_fast
    if request.fast_book_mode == "single_horizon":
        del w_fast_tilted
    w_slow_execution_1h = w_slow_execution.reindex(grid_1h).ffill().fillna(0.0)
    if request.crash_regime_tilt_alpha is not None:
        w_slow_execution_1h = ev.crash_regime_tilt_weights(
            w_slow_execution_1h, log_close,
            execution_mask.reindex(grid_1h).ffill().fillna(False),
            CRASH_REGIME_REFERENCE_SYMBOLS, slow.horizon_hours,
            request.crash_regime_tilt_alpha, min_symbols=slow.min_symbols,
        )
    if request.committee_capital:
        import src.mhs.evaluation as ev
        blend_1h = ev._committee_execution_book(
            close, quote_vol, taker_buy_quote, execution_mask, slow_grid, slow.min_symbols,
            COMMITTEE_TRANCHE_COUNT
            if (request.committee_tranche_smoothing or request.committee_regime_adaptive_tranche)
            else 1,
            regime_adaptive_window=(
                COMMITTEE_REGIME_ADAPTIVE_WINDOW
                if request.committee_regime_adaptive_tranche else None
            ),
            target_gross=_research_go._resolved_committee_target_gross(request),
            member_weights=committee_member_weights,
            carry_book=funding_carry_execution_book(bar_funding, execution_mask, FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS, slow_grid, COMMITTEE_TRANCHE_COUNT, slow.min_symbols) if request.funding_carry_sleeve else None, carry_weight=request.funding_carry_weight if request.funding_carry_sleeve else 0.0,
            members=_research_go._resolved_committee_members(request),
            coverage_cutoff=COMMITTEE_OOS_START,
            beta=causal_beta,
        ).reindex(grid_1h).fillna(0.0)
        del close, taker_buy_quote
    else:
        blend_1h = (
            BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_execution.reindex(grid_1h).ffill().fillna(0.0)
            + BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_execution_1h
        )
    del w_fast_execution, w_slow_execution, w_slow_execution_1h
    # Apply the additive sleeve before the regime cash-scale multiply and the
    # rebalance_filter branch so it inherits the same de-risking and turnover
    # gating the committee book already uses.
    if trend_position is not None:
        blend_1h = folds._apply_trend_sleeve(
            blend_1h, trend_position, execution_mask, request.trend_sleeve_gross,
        )
    _active_spec, active_grid = books._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    del _active_spec
    decision_grid = active_grid[(active_grid >= vs) & (active_grid <= ve)]
    target_weights = blend_1h.reindex(decision_grid)
    del blend_1h

    # The regime cash scale must read the traded execution roster, not the
    # full eligible universe: only the execution_mask symbols carry capital, so
    # their realized vol is the quantity that decides high-vol cash scaling.
    vol_mean = realized_vol(log_close, 48).where(execution_mask).reindex(decision_grid).mean(axis=1)
    regime_scale = _scaling._regime_cash_scale(vol_mean)
    if request.trend_efficiency_overlay:
        regime_scale = regime_scale.mul(
            _scaling._trend_efficiency_overlay_scale(log_close, execution_mask, fast.horizon_hours, decision_grid),
        )
    del execution_mask
    del log_close
    if request.rebalance_filter == "portfolio_trigger":
        if deadband_seed_row is not None:
            raise ValueError("deadband_seed_row requires rebalance_filter='per_symbol_deadband'")
        # Gate the unscaled book, then apply gross scale to preserve de-risking dynamics.
        target_weights = portfolio_rebalance_trigger(
            target_weights, REBALANCE_TRACKING_ERROR_THRESHOLD,
        ).mul(regime_scale, axis=0)
    else:
        target_weights = _scaling._apply_rebalance_deadband(
            target_weights.mul(regime_scale, axis=0), seed_row=deadband_seed_row,
        )

    if target_weights.empty:
        raise RuntimeError("fold decision grid is empty")
    execution_symbols = sorted(target_weights.columns[target_weights.ne(0.0).any(axis=0)])
    minute_roster = [
        s for s in execution_symbols
        if os.path.exists(os.path.join(root, request.execution_timeframe, f"{s}.parquet"))
    ]
    # 라이브 경로는 target weights만 emit하고 분단위 실행 리플레이를 하지 않으므로
    # minute roster 불변식은 백테스트(replay) 경로에서만 강제한다.
    if require_minute_roster and not minute_roster:
        raise RuntimeError("no fold decision symbol has minute execution data")
    signal_available_at = target_weights.index + pd.Timedelta(hours=1)
    return target_weights, signal_available_at, minute_roster, grid_1h