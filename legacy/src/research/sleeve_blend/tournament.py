"""Pre-registered five-source strategy tournament for the ``core5_v1`` sleeve.

The tournament evaluates exactly the five frozen return sources — the two
existing controls (long-only Donchian and funding-signed directional Donchian)
plus the three existing-but-unadmitted technical families (Supertrend,
Parabolic SAR, and Keltner-channel breakout) — on the fixed production universe
with catalog-supplied parameters. A chronological discovery/qualification split
decides membership, weights, and leverage on discovery data alone; the
qualification window is evaluated once and can never change the selection. Each
candidate is first screened by ``compute_gate_feasibility`` and must then
independently pass the unchanged observation, dynamic-fold, and stress gates on
discovery evidence before it receives a nonzero blend weight. Everything else
is an auditable CASH/REJECTED outcome.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.baseline.backtest import (
    BacktestResult,
    run_backtest,
    run_directional_backtest,
)
from src.research.contracts import CostModel, StrategySpec
from src.research.evaluation.gate_feasibility import (
    GateFeasibility,
    compute_gate_feasibility,
)
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    equity_span_years,
)
from src.research.sleeve_blend.common import (
    _EMPTY_TRADE_COLUMNS,
    _common_index,
    _concat_sleeve_trades,
)
from src.research.sleeve_blend.contracts import (
    BlendUniverseSpec,
    CausalLeverageSpec,
    PortfolioBlendTournamentReport,
    PortfolioBlendTournamentRequest,
    TournamentCandidateEvidence,
)
from src.research.sleeve_blend.fixed import (
    apply_leverage_schedule,
    build_causal_leverage_schedule,
)
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.contracts import TechnicalCandidate

_logger = logging.getLogger("StrategyTournament")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0
_BARS_PER_YEAR = 2190

DONCHIAN_LONG_ONLY = "donchian_long_only_v1"
FUNDING_SIGNED_DIRECTIONAL = "funding_signed_directional_v1"
SUPERTREND = "technical_supertrend_long_v1"
PARABOLIC_SAR = "technical_parabolic_sar_long_v1"
KELTNER_BREAKOUT = "technical_keltner_channel_breakout_long_v1"

TOURNAMENT_RETURN_SOURCES: tuple[str, ...] = (
    DONCHIAN_LONG_ONLY,
    FUNDING_SIGNED_DIRECTIONAL,
    SUPERTREND,
    PARABOLIC_SAR,
    KELTNER_BREAKOUT,
)

_TOURNAMENT_TECHNICAL_CANDIDATES: dict[str, TechnicalCandidate] = {
    SUPERTREND: TechnicalCandidate(
        SUPERTREND, SUPERTREND, "supertrend", "LONG",
        {"period": 10, "mult": 3.0, "regime": 200}, 201,
    ),
    PARABOLIC_SAR: TechnicalCandidate(
        PARABOLIC_SAR, PARABOLIC_SAR, "parabolic_sar", "LONG",
        {"step": 0.02, "max_step": 0.2, "regime": 200}, 201,
    ),
    KELTNER_BREAKOUT: TechnicalCandidate(
        KELTNER_BREAKOUT, KELTNER_BREAKOUT, "keltner_channel_breakout", "LONG",
        {"period": 20, "mult": 2.0, "regime": 200}, 201,
    ),
}


def _schedule_hash(schedule: pd.Series) -> str:
    """Stable sha256 of a frozen per-bar leverage schedule."""
    payload = json.dumps(schedule.round(8).tolist(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_source(
    source: str,
    symbol: str,
    frame: pd.DataFrame,
    funding: pd.Series,
    costs: CostModel,
    signal_delay_bars: int,
) -> BacktestResult:
    """Run one frozen return source on one symbol under the given costs/delay."""
    if source == DONCHIAN_LONG_ONLY:
        return run_backtest(
            frame, StrategySpec(symbol=symbol), costs,
            signal_delay_bars=signal_delay_bars,
        )
    if source == FUNDING_SIGNED_DIRECTIONAL:
        return run_directional_backtest(
            frame, StrategySpec(symbol=symbol), costs, funding,
            signal_delay_bars=signal_delay_bars,
        )
    candidate = _TOURNAMENT_TECHNICAL_CANDIDATES[source]
    return run_technical_expert_backtest(
        frame, candidate, costs, funding, signal_delay_bars=signal_delay_bars,
    )


def _load_universe_data(
    universe: BlendUniverseSpec,
    start: str | None,
    end: str | pd.Timestamp | None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Load and fail-closed validate each symbol's 4h bars and funding window."""
    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in universe.symbols:
        df = load_ohlcv_4h(ohlcv_path(symbol, "1h"), start=start, end=end)
        if len(df) < 2:
            raise DataIntegrityError(f"no 4h bars for {symbol} in the window")
        funding = load_funding_rates(funding_path(symbol))
        bar_period = df.index[1] - df.index[0]
        window_end = df.index[-1] + bar_period
        funding = funding[
            (funding.index >= df.index[0]) & (funding.index < window_end)
        ]
        if len(funding) == 0:
            raise DataIntegrityError(f"no funding events for {symbol} in the window")
        data[symbol] = (df, funding)
    return data


def _blend_unit_equities(
    equities: dict[str, pd.Series],
    common: pd.DatetimeIndex,
) -> pd.Series:
    """Equal-capital-weight blend of unit-leverage equity curves on ``common``."""
    normalized = [
        eq.loc[common] / eq.loc[common].iloc[0] for eq in equities.values()
    ]
    return pd.Series(
        sum(normalized) / len(normalized), index=common, name="equity", dtype=np.float64,
    )


def _source_full_equity(
    source: str,
    data: dict[str, tuple[pd.DataFrame, pd.Series]],
    costs: CostModel,
    signal_delay_bars: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-weight blend of one source across the universe plus its trades."""
    results: dict[str, BacktestResult] = {}
    for symbol, (frame, funding) in data.items():
        results[symbol] = _run_source(
            source, symbol, frame, funding, costs, signal_delay_bars,
        )
    common = _common_index(results)
    equity = _blend_unit_equities(
        {symbol: res.equity for symbol, res in results.items()}, common,
    )
    return equity, _concat_sleeve_trades(results)


def _slice_trades(
    trades: pd.DataFrame,
    *,
    end_ts: pd.Timestamp,
    start_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Restrict closed trades to the window by wall-clock exit time."""
    if len(trades) == 0:
        return trades
    mask = trades["exit_time"] <= end_ts
    if start_ts is not None:
        mask &= trades["exit_time"] > start_ts
    return trades[mask].reset_index(drop=True)


def _feasibility_or_reason(
    equity: pd.Series,
    trades: pd.DataFrame,
) -> tuple[GateFeasibility | None, str | None]:
    """Return ``(feasibility, binding_constraint_or_reason)`` for discovery data.

    Returns ``(None, reason)`` when the discovery evidence cannot even be
    screened (too few marks, no variance, or a non-negative unit-leverage MDD),
    so an infeasible candidate is never run through the expensive qualification
    backtest and is never silently dropped.
    """
    if len(equity) < 2:
        return None, "insufficient_data"
    metrics = compute_metrics(equity, trades)
    returns = equity.pct_change().dropna()
    vol = float(returns.std() * np.sqrt(_BARS_PER_YEAR)) if len(returns) > 1 else 0.0
    try:
        years = equity_span_years(equity)
        feasibility = compute_gate_feasibility(
            metrics.sharpe, vol, metrics.mdd, years,
        )
    except ValueError as exc:
        return None, f"feasibility:{type(exc).__name__}:{exc}"
    if not feasibility.feasible:
        return None, f"feasibility:{feasibility.binding_constraint}"
    return feasibility, None


def _segment_gates(
    source: str,
    data: dict[str, tuple[pd.DataFrame, pd.Series]],
    costs: CostModel,
    equity: pd.Series,
    trades: pd.DataFrame,
) -> tuple[ReliabilityGateResult, FoldDistributionResult, ReliabilityGateResult]:
    """Run the unchanged observation/fold/stress gates on one segment.

    The stress gate re-runs the source under 1.5x fee, 2.0x slippage and one
    additional decision bar and slices the stressed ledger to the same segment
    index; it never re-fits anything and only composes existing evidence.
    """
    observation = compute_equity_reliability_gate(equity, len(trades))
    fold = compute_fold_distribution(
        BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame()),
    )
    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_equity, stress_full_trades = _source_full_equity(
        source, data, stressed_costs, 1,
    )
    stress_segment = stress_equity.loc[equity.index]
    stress_trades = _slice_trades(stress_full_trades, end_ts=equity.index[-1])
    stress = compute_equity_reliability_gate(
        stress_segment, len(stress_trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    return observation, fold, stress


def _reject_reason(
    observation: ReliabilityGateResult,
    fold: FoldDistributionResult,
    stress: ReliabilityGateResult,
) -> str:
    """Name the first discovery gate that failed (fail-closed, no silent drop)."""
    if observation.verdict != "PASS":
        return f"observation:{observation.verdict}"
    if not fold.gate_pass:
        return "fold:gate_pass=False"
    return f"stress:{stress.verdict}"


def run_strategy_tournament(
    request: PortfolioBlendTournamentRequest,
) -> PortfolioBlendTournamentReport:
    """Execute one sealed five-source tournament and return its immutable report.

    The two existing controls and the three technical sources are run on
    ``core5_v1`` at base and stressed costs. Each candidate's discovery window
    (bars up to ``request.discovery_end``) is feasibility-screened first; an
    admitted candidate must then independently PASS the unmodified observation,
    dynamic-fold, and stress gates on discovery evidence. The untouched
    qualification window is scored once for evidence and can never change the
    selected sources, weights, or leverage schedule. The admitted sources are
    equal-weight blended and a causal leverage schedule is built and applied
    identically to the base and stress ledgers.
    """
    data = _load_universe_data(request.universe, request.start, request.end)

    base_full: dict[str, pd.Series] = {}
    base_trades: dict[str, pd.DataFrame] = {}
    stress_full: dict[str, pd.Series] = {}
    stress_trades: dict[str, pd.DataFrame] = {}
    for source in TOURNAMENT_RETURN_SOURCES:
        base_full[source], base_trades[source] = _source_full_equity(
            source, data, request.costs, 0,
        )
        stressed_costs = CostModel(
            fee_rate=request.costs.fee_rate * _STRESS_FEE_MULT,
            slippage_rate=request.costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
        )
        stress_full[source], stress_trades[source] = _source_full_equity(
            source, data, stressed_costs, 1,
        )

    common = pd.DatetimeIndex(sorted(
        set.intersection(*(set(equity.index) for equity in base_full.values()))
    ))
    if len(common) < 2:
        raise DataIntegrityError(
            "tournament symbols share fewer than 2 common bars across all sources"
        )

    discovery_end = request.discovery_end
    qual_offset = pd.tseries.frequencies.to_offset(request.qualification_interval)
    qual_end = discovery_end + qual_offset
    if request.end is not None:
        end_ts = pd.Timestamp(request.end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        if end_ts < qual_end:
            qual_end = end_ts

    candidates: list[TournamentCandidateEvidence] = []
    selected: list[str] = []
    for source in TOURNAMENT_RETURN_SOURCES:
        equity = base_full[source]
        disc_mask = equity.index <= discovery_end
        disc_equity = equity.loc[disc_mask]
        disc_trades = _slice_trades(base_trades[source], end_ts=discovery_end)

        feasibility, reason = _feasibility_or_reason(disc_equity, disc_trades)
        if reason is not None:
            candidates.append(TournamentCandidateEvidence(
                return_source=source,
                feasibility=None,
                feasibility_binding=reason,
                discovery_observation=None,
                discovery_fold=None,
                discovery_stress=None,
                discovery_promotion=None,
                qualification_observation=None,
                qualification_fold=None,
                qualification_stress=None,
                qualification_promotion=None,
                admitted=False,
                rejected_reason=reason,
            ))
            continue
        assert feasibility is not None

        try:
            obs, fold, stress = _segment_gates(
                source, data, request.costs, disc_equity, disc_trades,
            )
        except ValueError as exc:
            candidates.append(TournamentCandidateEvidence(
                return_source=source,
                feasibility=feasibility,
                feasibility_binding=feasibility.binding_constraint,
                discovery_observation=None,
                discovery_fold=None,
                discovery_stress=None,
                discovery_promotion=None,
                qualification_observation=None,
                qualification_fold=None,
                qualification_stress=None,
                qualification_promotion=None,
                admitted=False,
                rejected_reason=f"gates:{type(exc).__name__}:{exc}",
            ))
            continue
        promotion = compose_promotion_verdict(obs, fold, stress, None)
        admitted = obs.verdict == "PASS" and fold.gate_pass and stress.verdict == "PASS"

        if not admitted:
            candidates.append(TournamentCandidateEvidence(
                return_source=source,
                feasibility=feasibility,
                feasibility_binding=feasibility.binding_constraint,
                discovery_observation=obs,
                discovery_fold=fold,
                discovery_stress=stress,
                discovery_promotion=promotion,
                qualification_observation=None,
                qualification_fold=None,
                qualification_stress=None,
                qualification_promotion=None,
                admitted=False,
                rejected_reason=_reject_reason(obs, fold, stress),
            ))
            continue

        qual_mask = (equity.index > discovery_end) & (equity.index <= qual_end)
        qual_equity = equity.loc[qual_mask]
        qual_trades = _slice_trades(
            base_trades[source], start_ts=discovery_end, end_ts=qual_end,
        )
        if len(qual_equity) < 2:
            qual_obs, qual_fold, qual_stress = None, None, None
        else:
            try:
                qual_obs, qual_fold, qual_stress = _segment_gates(
                    source, data, request.costs, qual_equity, qual_trades,
                )
            except ValueError:
                # Qualification is evidence-only and can never change selection;
                # un-computable windows are recorded as absent, never fabricated.
                qual_obs, qual_fold, qual_stress = None, None, None
        if qual_obs is None or qual_fold is None or qual_stress is None:
            qual_promotion = None
        else:
            qual_promotion = compose_promotion_verdict(qual_obs, qual_fold, qual_stress, None)

        candidates.append(TournamentCandidateEvidence(
            return_source=source,
            feasibility=feasibility,
            feasibility_binding=feasibility.binding_constraint,
            discovery_observation=obs,
            discovery_fold=fold,
            discovery_stress=stress,
            discovery_promotion=promotion,
            qualification_observation=qual_obs,
            qualification_fold=qual_fold,
            qualification_stress=qual_stress,
            qualification_promotion=qual_promotion,
            admitted=True,
            rejected_reason=None,
        ))
        selected.append(source)
        _logger.info(
            "[TOURNAMENT] source=%s admitted discovery_obs=%s fold=%s stress=%s",
            source, obs.verdict, fold.gate_pass, stress.verdict,
        )

    if selected:
        k = len(selected)
        blend_weights = tuple(1.0 / k for _ in selected)
        unit_blend = _blend_unit_equities(
            {source: base_full[source].loc[common] for source in selected}, common,
        )
        schedule = build_causal_leverage_schedule(unit_blend, CausalLeverageSpec())
        base_equity = apply_leverage_schedule(unit_blend, schedule, request.initial_equity)
        base_result = BacktestResult(
            equity=base_equity,
            trades=pd.concat(
                [base_trades[source] for source in selected], ignore_index=True,
            ),
            signals=pd.DataFrame(),
        )
        stress_unit_blend = _blend_unit_equities(
            {source: stress_full[source].loc[common] for source in selected}, common,
        )
        stress_equity = apply_leverage_schedule(
            stress_unit_blend, schedule, request.initial_equity,
        )
        stress_result = BacktestResult(
            equity=stress_equity,
            trades=pd.concat(
                [stress_trades[source] for source in selected], ignore_index=True,
            ),
            signals=pd.DataFrame(),
        )
    else:
        blend_weights = ()
        schedule = pd.Series(0.0, index=common, name="leverage", dtype=np.float64)
        empty_trades = pd.DataFrame(columns=list(_EMPTY_TRADE_COLUMNS))
        flat = pd.Series(
            np.full(len(common), request.initial_equity, dtype=np.float64),
            index=common, name="equity",
        )
        base_result = BacktestResult(
            equity=flat.copy(), trades=empty_trades.copy(), signals=pd.DataFrame(),
        )
        stress_result = BacktestResult(
            equity=flat.copy(), trades=empty_trades.copy(), signals=pd.DataFrame(),
        )

    qual_bars = common[(common > discovery_end) & (common <= qual_end)]
    qualification_start = qual_bars[0] if len(qual_bars) else discovery_end
    qualification_end = qual_bars[-1] if len(qual_bars) else None

    return PortfolioBlendTournamentReport(
        request=request,
        universe=request.universe,
        candidates=tuple(candidates),
        selected_return_sources=tuple(selected),
        blend_weights=blend_weights,
        leverage_schedule=schedule,
        schedule_hash=_schedule_hash(schedule),
        base_result=base_result,
        stress_result=stress_result,
        qualification_start=qualification_start,
        qualification_end=qualification_end,
    )


def _check_contract() -> None:
    """Executable assertions locking the tournament surface."""
    assert TOURNAMENT_RETURN_SOURCES == (
        "donchian_long_only_v1",
        "funding_signed_directional_v1",
        "technical_supertrend_long_v1",
        "technical_parabolic_sar_long_v1",
        "technical_keltner_channel_breakout_long_v1",
    )
    assert set(_TOURNAMENT_TECHNICAL_CANDIDATES) == {
        SUPERTREND, PARABOLIC_SAR, KELTNER_BREAKOUT,
    }
    assert BlendUniverseSpec().symbols == (
        "BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT",
    )


_check_contract()
