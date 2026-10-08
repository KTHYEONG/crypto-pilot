from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.marks import (
    _missing_execution_sources,
    _pit_execution_mask,
    clear_mhs_market_data_caches,
)
from src.core.panel import PanelQuarantine, liquid_half_eligibility, load_base_panel, slice_base_panel
from src.core.params import (
    PANEL_MIN_HISTORY_BARS,
    UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
    UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
)
from src.core.types import (
    BOOK_BLEND_WEIGHTS,
    BOOK_SPECS,
    COMMITTEE_REGIME_ADAPTIVE_WINDOW,
    CRASH_REGIME_REFERENCE_SYMBOLS,
    FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS,
)
from src.engine.execution import bar_funding_panel
from src.lab.mhs import research_go as _research_go
from src.lab.mhs import scaling as _scaling
from src.lab.mhs.contracts import MhsDiagnosticRequest
from src.lab.mhs.evidence import AnchoredPurgedFold
from src.lab.mhs.funding import funding_carry_execution_book
from src.lab.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    FOLD_PANEL_WARMUP_HOURS,
    REBALANCE_TRACKING_ERROR_THRESHOLD,
)
from src.lab.mhs.regime import beta_neutralize_weights, causal_market_beta, crash_regime_tilt_weights
from src.strategy.books import inverse_realized_vol_tilt, portfolio_rebalance_trigger, renormalize_within_mask
from src.strategy.features import FeatureAdmission
from src.strategy.horizons import realized_vol

from . import books, committee, folds, integrity, specs


def _resolve_effective_fold_window(
    fold: AnchoredPurgedFold,
    decision_start: pd.Timestamp | None,
    decision_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Effective decision window for one fold, defaulting to its validation span.

    Overrides exist for the train-only sizing reference; they must be tz-aware UTC,
    non-empty and must not start before ``fold.train_start`` (no pre-discovery
    borrowing). Raises ValueError with the existing messages otherwise.
    """
    ts = fold.train_start
    effective_start = fold.validation_start if decision_start is None else decision_start
    effective_end = fold.validation_end if decision_end is None else decision_end
    if not isinstance(effective_start, pd.Timestamp) or effective_start.tzinfo is None:
        raise ValueError("effective fold window bounds must be tz-aware UTC timestamps")
    if not isinstance(effective_end, pd.Timestamp) or effective_end.tzinfo is None:
        raise ValueError("effective fold window bounds must be tz-aware UTC timestamps")
    if str(effective_start.tzinfo) != "UTC" or str(effective_end.tzinfo) != "UTC":
        raise ValueError("effective fold window bounds must be UTC")
    if effective_start < ts or effective_start >= effective_end:
        raise ValueError("effective fold window is empty or precedes fold train_start")
    return effective_start, effective_end


def _rebalance_fold_weights(
    weights: pd.DataFrame,
    regime_scale: pd.Series,
    request: MhsDiagnosticRequest,
    apply_deadband: bool,
    seed_row: pd.Series | None,
) -> pd.DataFrame:
    """Apply turnover controls with the regime scale in the existing causal order."""
    if request.rebalance_filter == "portfolio_trigger":
        if seed_row is not None:
            raise ValueError("deadband_seed_row requires rebalance_filter='per_symbol_deadband'")
        return portfolio_rebalance_trigger(
            weights,
            REBALANCE_TRACKING_ERROR_THRESHOLD,
        ).mul(regime_scale, axis=0)
    scaled = weights.mul(regime_scale, axis=0)
    if apply_deadband is False:
        return scaled
    return _scaling._apply_rebalance_deadband(scaled, seed_row=seed_row)


def _validate_fold_committee_admission(
    request: MhsDiagnosticRequest, fold: AnchoredPurgedFold, committee_admission: FeatureAdmission | None
) -> None:
    if request.committee_capital:
        if committee_admission is None:
            raise integrity.CommitteeAdmissionIntegrityError(
                "committee_capital requires committee_admission (fold train_end boundary)"
            )
        if committee_admission.cutoff > fold.validation_start:
            raise integrity.CommitteeAdmissionIntegrityError(
                f"committee admission cutoff {committee_admission.cutoff} after validation_start {fold.validation_start}"
            )
        if committee_admission.cutoff != fold.train_end:
            raise integrity.CommitteeAdmissionIntegrityError(
                f"committee admission cutoff {committee_admission.cutoff} != fold train_end {fold.train_end}"
            )
    elif committee_admission is not None:
        raise ValueError("committee_admission requires committee_capital")


def _funding_aligned_fold_panels(
    close: pd.DataFrame,
    opens: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame | None,
    funding_by_symbol: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Restrict fold panels to symbols with observed, causally aligned funding.

    Returns:
        ``(close, opens, quote_vol, taker_buy_quote, bar_funding)`` on the aligned symbol set.
    Raises:
        RuntimeError: no symbol has funding coverage, or none survives causal alignment.
    """
    grid_1h = close.index
    symbols = list(close.columns)
    funded = [s for s in symbols if s in funding_by_symbol and s not in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS]
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
            (funding_by_symbol[s].index >= grid_1h[0]) & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
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
    return close, opens, quote_vol, taker_buy_quote, bar_funding


def _committee_fold_blend(
    close: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame | None,
    execution_mask: pd.DataFrame,
    bar_funding: pd.DataFrame,
    slow_grid: pd.DatetimeIndex,
    min_symbols: int,
    grid_1h: pd.DatetimeIndex,
    request: MhsDiagnosticRequest,
    committee_member_weights: dict[str, float] | None,
    causal_beta: pd.DataFrame | None,
    committee_admission: FeatureAdmission | None,
) -> pd.DataFrame:
    return (
        committee._committee_execution_book(
            close,
            quote_vol,
            taker_buy_quote,
            execution_mask,
            slow_grid,
            min_symbols,
            _research_go._resolved_committee_tranche_count(request),
            regime_adaptive_window=(
                COMMITTEE_REGIME_ADAPTIVE_WINDOW if request.committee_regime_adaptive_tranche else None
            ),
            target_gross=request.committee_target_gross,
            member_weights=committee_member_weights,
            carry_book=funding_carry_execution_book(
                bar_funding,
                execution_mask,
                FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS,
                slow_grid,
                request.committee_tranche_count,
                min_symbols,
            )
            if request.funding_carry_sleeve
            else None,
            carry_weight=request.funding_carry_weight if request.funding_carry_sleeve else 0.0,
            members=_research_go._resolved_committee_members(request),
            beta=causal_beta,
            admission=committee_admission,
        )
        .reindex(grid_1h)
        .fillna(0.0)
    )


def _fold_minute_roster(
    root: str,
    target_weights: pd.DataFrame,
    request: MhsDiagnosticRequest,
    vs: pd.Timestamp,
    ve: pd.Timestamp,
    require_minute_roster: bool,
) -> list[str]:
    execution_symbols = sorted(target_weights.columns[target_weights.ne(0.0).any(axis=0)])
    absent = set(_missing_execution_sources(root, execution_symbols, request.execution_timeframe))
    minute_roster = [s for s in execution_symbols if s not in absent]
    # 라이브 경로는 target weights만 emit하고 분단위 실행 리플레이를 하지 않으므로
    # minute roster 불변식은 백테스트(replay) 경로에서만 강제한다.
    if require_minute_roster and not minute_roster:
        raise RuntimeError("no fold decision symbol has minute execution data")
    if require_minute_roster:
        targeted_missing = sorted(
            s
            for s in execution_symbols
            if s in absent and bool((target_weights[s].notna() & target_weights[s].ne(0.0)).any())
        )
        if targeted_missing:
            raise DataIntegrityError(
                f"fold execution source missing for {len(targeted_missing)} targeted symbol(s) "
                f"[{', '.join(targeted_missing)}] timeframe={request.execution_timeframe} "
                f"root={root} decision_window=[{vs.isoformat()}, {ve.isoformat()}]"
            )
    return minute_roster


def _build_fold_target_weights(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    slow_horizon_override: int | None = None,
    committee_member_weights: dict[str, float] | None = None,
    *,
    decision_start: pd.Timestamp | None = None,
    decision_end: pd.Timestamp | None = None,
    deadband_seed_row: pd.Series | None = None,
    require_minute_roster: bool = True,
    base_panel: dict[str, pd.DataFrame] | None = None,
    panel_warmup_hours: int = FOLD_PANEL_WARMUP_HOURS,
    committee_admission: FeatureAdmission | None = None,
    apply_rebalance_deadband: bool = True,
    panel_quarantine: PanelQuarantine | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str], pd.DatetimeIndex]:
    """Build one fold's causal decision-grid target book from completed trade OHLCV and observed funding.

    Fold eligibility uses the same completed trade OHLCV and observed funding sources as
    top-level selection; historical mark availability cannot select fold members. Under
    committee capital the blended target is the committee execution book and the fast/slow
    momentum books are never built, so fast/slow-only knobs (fast_book_mode, slow_book_mode,
    ensemble_signal, crash_regime_tilt_alpha) cannot influence the result; beta_neutralize
    still applies through the committee book's causal beta.
    Under committee capital the member set is ``committee_admission`` -- the
    admission decided on rows strictly before ``fold.train_end``, the same
    boundary that fit ``committee_member_weights`` -- and no coverage audit runs
    on the fold panel, so no decision at or before T can depend on data after T
    (I-FOLD-ADMISSION-PIT). The same admission serves the validation and the
    train-reference windows.

    Raises:
        CommitteeAdmissionIntegrityError: committee capital without
            ``committee_admission``; ``committee_admission.cutoff`` after
            ``fold.validation_start`` (I-FOLD-ADMISSION-PIT); cutoff different
            from ``fold.train_end`` (I-COVERAGE-PIT); or an admission/weights
            mismatch surfaced by the committee book.
        ValueError: ``committee_admission`` given while committee capital is
            off (plus existing window/deadband errors).
        RuntimeError: ``require_minute_roster`` and no targeted symbol has an execution
            source ("no fold decision symbol has minute execution data", unchanged).
        DataIntegrityError: ``require_minute_roster`` and at least one symbol with a
            finite non-zero fold target lacks its execution source while others have
            one: "fold execution source missing for <n> targeted symbol(s) [<SYM>, ...]
            timeframe=<tf> root=<root> decision_window=[<vs iso>, <ve iso>]". Dropping the
            column instead would replay and certify a fold book that silently omits the
            symbol's P&L and costs.
    """
    ts = fold.train_start
    vs, ve = _resolve_effective_fold_window(fold, decision_start, decision_end)
    if apply_rebalance_deadband is False and request.rebalance_filter != "per_symbol_deadband":
        raise ValueError("apply_rebalance_deadband=False requires rebalance_filter='per_symbol_deadband'")
    _validate_fold_committee_admission(request, fold, committee_admission)
    panel_start = max(ts, vs - pd.Timedelta(hours=panel_warmup_hours))
    _panel_columns = (
        ("close", "open", "quote_vol", "taker_buy_quote")
        if request.committee_capital
        else ("close", "open", "quote_vol")
    )
    panel = (
        slice_base_panel(base_panel, panel_start, ve, min_bars=PANEL_MIN_HISTORY_BARS)
        if base_panel is not None
        else load_base_panel(
            root,
            "1h",
            _panel_columns,
            panel_start,
            ve,
            partition="dev",
            min_bars=PANEL_MIN_HISTORY_BARS,
            data_policy=request.data_policy,
            quarantine=panel_quarantine,
        )
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    taker_buy_quote = panel["taker_buy_quote"] if request.committee_capital else None
    del panel
    close, opens, quote_vol, taker_buy_quote, bar_funding = _funding_aligned_fold_panels(
        close, opens, quote_vol, taker_buy_quote, funding_by_symbol
    )
    grid_1h = close.index

    eligible = liquid_half_eligibility(
        quote_vol,
        lookback_bars=UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        min_history_bars=UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )
    clear_mhs_market_data_caches()
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
    if not request.committee_capital:
        fast_ema = specs._signal_ema_span(fast.band.sign, fast.horizon_hours, fast.step_hours)
        slow_ema = specs._signal_ema_span(slow.band.sign, slow.horizon_hours, slow.step_hours)
        w_fast = books._book_weights(log_close, eligible, fast, fast_grid, ema_span=fast_ema)
    execution_mask = _pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    if not request.committee_capital:
        if request.fast_book_mode == "horizon_ensemble":
            w_fast_execution = books._horizon_ensemble_execution_weights(
                log_close,
                eligible,
                execution_mask,
                fast,
                fast_grid,
                "horizon_ensemble",
                "raw",
                fast_ema,
            )
        else:
            w_fast_tilted = inverse_realized_vol_tilt(
                w_fast,
                realized_vol(log_close, fast.horizon_hours).reindex(fast_grid),
            )
            w_fast_execution = renormalize_within_mask(
                w_fast_tilted,
                execution_mask.reindex(w_fast.index).fillna(False),
                fast.min_symbols,
            )
        w_slow_execution = books._horizon_ensemble_execution_weights(
            log_close,
            eligible,
            execution_mask,
            slow,
            slow_grid,
            request.slow_book_mode,
            request.ensemble_signal,
            slow_ema,
        )
    # I-SINGLE-CONFIGURATION: one causal-beta computation shared by the legacy
    # slow-book neutralize and the committee execution book below.
    causal_beta = (
        causal_market_beta(
            log_close,
            eligible,
            CAUSAL_BETA_LOOKBACK_BARS,
            CAUSAL_BETA_MIN_PERIODS,
        )
        if request.beta_neutralize
        else None
    )
    if not request.committee_capital and causal_beta is not None:
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
        del quote_vol, w_fast
        if request.fast_book_mode == "single_horizon":
            del w_fast_tilted
        w_slow_execution_1h = w_slow_execution.reindex(grid_1h).ffill().fillna(0.0)
        if request.crash_regime_tilt_alpha is not None:
            w_slow_execution_1h = crash_regime_tilt_weights(
                w_slow_execution_1h,
                log_close,
                execution_mask.reindex(grid_1h).ffill().fillna(False),
                CRASH_REGIME_REFERENCE_SYMBOLS,
                slow.horizon_hours,
                request.crash_regime_tilt_alpha,
                min_symbols=slow.min_symbols,
            )
    if request.committee_capital:
        blend_1h = _committee_fold_blend(
            close, quote_vol, taker_buy_quote, execution_mask, bar_funding, slow_grid, slow.min_symbols,
            grid_1h, request, committee_member_weights, causal_beta, committee_admission,
        )
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
            blend_1h,
            trend_position,
            execution_mask,
            request.trend_sleeve_gross,
        )
    _active_spec, active_grid = books._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    del _active_spec
    decision_grid = active_grid[(active_grid >= vs) & (active_grid <= ve)]
    target_weights = blend_1h.reindex(decision_grid)
    del blend_1h

    # The regime cash scale must read the traded execution roster, not the
    # full eligible universe: only the execution_mask symbols carry capital, so
    # their realized vol is the quantity that decides high-vol cash scaling.
    regime_scale = (
        _scaling.regime_cash_scale_1h(
            log_close, execution_mask, grid_1h, fast.horizon_hours, request.trend_efficiency_overlay
        )
        .reindex(decision_grid)
        .fillna(1.0)
    )
    del execution_mask
    del log_close
    target_weights = _rebalance_fold_weights(
        target_weights,
        regime_scale,
        request,
        apply_rebalance_deadband,
        deadband_seed_row,
    )

    if target_weights.empty:
        raise RuntimeError("fold decision grid is empty")
    minute_roster = _fold_minute_roster(root, target_weights, request, vs, ve, require_minute_roster)
    signal_available_at = target_weights.index + pd.Timedelta(hours=1)
    return target_weights, signal_available_at, minute_roster, grid_1h
