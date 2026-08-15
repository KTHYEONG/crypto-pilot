"""Quarterly rolling library admission application service.

Replays the sealed selection algorithm at every historical rebalance using only
its own ``observed_end`` snapshot, stitches only closed deployment quarters into
one master ledger, charges a single switch cost when the proposal changes, and
writes each decision to the append-only rebalance ledger plus the atomic
current-profile pointer. Warm-up observations feed indicators and router state
only; their returns never enter scoring or OOS PnL. Paper mode produces no
trading side effect; live execution requires explicit separate authorization.
"""

from __future__ import annotations  # noqa: I001

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from src.application.research.expert.admission import (
    _assemble_panel,
    _build_admission_context,
    _symbol_admission_worker,
)
from src.application.research.expert.admission import (
    _materialize_definitions as _materialize_universe_definitions,
)
from src.application.research.expert.admission_backtest import (
    _assemble_selected_panel,
    _master_result,
    _run_selected_tasks,
)
from src.application.research.expert.admission_backtest import (
    _materialize_definitions as _materialize_selected_definitions,
)
from src.application.research.expert.rebalance_ledger import (
    CURRENT_PROFILE_POINTER_PATH,
    REBALANCE_LEDGER_PATH,
    append_rebalance_record,
    rebalance_snapshot_key,
    write_current_profile,
)
from src.application.research.expert.window import resolve_common_technical_window
from src.common.errors import DataIntegrityError
from src.research.contracts import CostModel
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (  # noqa: I001
    FoldDistributionResult,
    ReliabilityGateConfig,
    compute_equal_duration_fold_distribution,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    derive_cost_multiple_hurdle_rate, equity_span_years,
)
from src.research.expert_portfolio.admission import (
    BoundedProposalSearchResult,
    enrich_proposal_diagnostics,
    evaluate_library_admission,
    pair_compatibility_matrix,
    pairwise_joint_negative_rates,
    pairwise_log_return_correlation,
    prefilter_admitted_by_family_symbol,
    priority_shortlist_family_unique_proposals,
)
from src.research.expert_portfolio.admission_reports import (
    LibraryAdmissionBacktestReport,
    LibraryAdmissionReport,
)
from src.research.expert_portfolio.admission_types import (
    AdmissionProposal,
    TechnicalLibraryAdmissionRequest,
    admission_proposal_id,
)
from src.research.expert_portfolio.backtest import run_expert_portfolio
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioSpec,
)
from src.research.expert_portfolio.rolling import (
    RebalanceWindow,
    RollingAdmissionConfig,
    RollingCandidateAuditRecord,
    RollingLibraryAdmissionReport,
    RollingSelectionRecord,
    build_rolling_rebalance_schedule,
    resolve_dynamic_shortlist_budget,
    select_rebalance_proposal,
)
from src.research.provenance.code_manifest import TECHNICAL_CODE_UNITS, compute_code_hash
from src.research.technical_experts.provenance import technical_data_hashes

_logger = logging.getLogger("RollingLibraryAdmission")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0
_FOLD_DURATION = "6MS"


@dataclass(frozen=True, slots=True)
class RollingLibraryAdmissionRequest:
    """Immutable request for one rolling library admission replay.

    ``profile`` is the frozen rolling candidate universe, ``as_of`` the frozen
    data snapshot (the only temporal source), ``config`` the quarterly policy,
    and ``mode`` is ``"paper"`` or ``"live"``. Live mode fails closed unless
    ``live_authorized``; ``require_complete_history`` rejects a request whose
    newest deployment quarter is still incomplete.
    """

    profile: TechnicalLibraryAdmissionRequest
    as_of: str | pd.Timestamp
    config: RollingAdmissionConfig = field(default_factory=RollingAdmissionConfig)
    mode: str = "paper"
    log_run: bool = False
    live_authorized: bool = False
    require_complete_history: bool = False

    def __post_init__(self) -> None:
        if self.config.symbols != self.profile.symbols:
            raise ValueError(
                f"rolling config symbols {self.config.symbols} must match the profile "
                f"symbols {self.profile.symbols}"
            )
        if self.mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {self.mode!r}")
        if self.mode == "live" and not self.live_authorized:
            raise RuntimeError(
                "live trading bridge requires explicit separate authorization"
            )
        pd.Timestamp(self.as_of, tz="UTC")


def _materialize_universe(
    profile: TechnicalLibraryAdmissionRequest,
    code_hash: str,
) -> tuple[ExpertDefinition, ...]:
    return _materialize_universe_definitions(profile, code_hash)


def _build_candidate_panel(
    profile: TechnicalLibraryAdmissionRequest,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int], tuple[ExpertDefinition, ...], str]:
    """Run the full frozen candidate universe once over a causal window."""
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    definitions = _materialize_universe(profile, code_hash)
    sources = tuple(sorted(profile.candidate_sources))
    evidence = {
        symbol: _symbol_admission_worker(symbol, sources, str(start), end)
        for symbol in profile.symbols
    }
    panel, trade_counts = _assemble_panel(evidence, definitions, profile.symbols)
    return panel, trade_counts, definitions, code_hash


def _slice_scored_panel(panel: pd.DataFrame, scored_start: pd.Timestamp) -> pd.DataFrame:
    """Slice the scored window and restore the single initial all-NaN row."""
    scored = panel.loc[scored_start:].copy()
    if len(scored) < 2:
        raise DataIntegrityError(
            f"scored panel has fewer than 2 bars from {scored_start}"
        )
    scored.iloc[0] = np.nan
    return scored


@dataclass(frozen=True, slots=True)
class WindowScenarioEvidence:
    """One window's immutable base/stress scenario evidence for rolling-v2.

    Every requested ``(symbol, source)`` is executed at most once per scenario
    and the resulting panels, component trade evidence, definitions, and causal
    context are reused verbatim by every shortlisted proposal; a proposal never
    loads market data or reruns a technical candidate. Candidate-runner counts
    and stage wall seconds are non-decision telemetry for the audit.
    """

    profile: TechnicalLibraryAdmissionRequest
    window: RebalanceWindow
    code_hash: str
    definitions: tuple[ExpertDefinition, ...]
    base_panel: pd.DataFrame
    base_trade_counts: dict[str, int]
    base_component_trades: pd.DataFrame
    base_context: pd.Series
    stress_panel: pd.DataFrame
    stress_component_trades: pd.DataFrame
    base_candidate_runner_calls: int
    stress_candidate_runner_calls: int
    base_wall_seconds: float
    stress_wall_seconds: float
    base_workers: int = 0
    stress_workers: int = 0


def build_window_scenario_evidence(
    profile: TechnicalLibraryAdmissionRequest,
    window: RebalanceWindow,
    config: RollingAdmissionConfig,
) -> WindowScenarioEvidence:
    """Run every candidate once per base/stress scenario and window.

    The base scenario (``config.base_delay_bars``, base costs) feeds both the
    candidate screen and every base master portfolio; the stress scenario keeps
    the existing one-bar delayed execution under the stress cost multipliers.
    Symbols are executed over one coarse process pool (``effective_worker_count``
    bounds the pool; ``max_workers == 1`` forces the identical sequential path),
    historical windows are never parallelized, and no nested process pool is
    created. A worker failure, a non-aligned panel, or missing scenario evidence
    fails closed with ``DataIntegrityError``; no partial evidence is returned.
    """
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    definitions = _materialize_universe_definitions(profile, code_hash)
    try:
        started = time.perf_counter()
        costs = CostModel()
        base_evidence, base_workers = _run_selected_tasks(
            definitions, str(window.load_start), window.observed_end, costs,
            config.base_delay_bars, profile.admission.max_workers,
        )
        base_panel, base_component_trades = _assemble_selected_panel(
            base_evidence, definitions,
        )
        base_trade_counts = {
            definition.expert_id: len(
                base_evidence[definition.symbols[0]][definition.return_source]["trades"],
            )
            for definition in definitions
        }
        base_context = _build_admission_context(
            profile.router, base_panel.index, window.load_start, window.observed_end,
        )
        base_wall_seconds = time.perf_counter() - started

        stress_started = time.perf_counter()
        stress_costs = CostModel(
            fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
            slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
        )
        stress_evidence, stress_workers = _run_selected_tasks(
            definitions, str(window.load_start), window.observed_end, stress_costs,
            1, profile.admission.max_workers,
        )
        stress_panel, stress_component_trades = _assemble_selected_panel(
            stress_evidence, definitions,
        )
        stress_wall_seconds = time.perf_counter() - stress_started
    except DataIntegrityError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(
            f"window evidence failed closed for {window.profile} at "
            f"{window.rebalance_start}: {exc}"
        ) from exc

    per_scenario = len(profile.symbols) * len(profile.candidate_sources)
    return WindowScenarioEvidence(
        profile=profile,
        window=window,
        code_hash=code_hash,
        definitions=definitions,
        base_panel=base_panel,
        base_trade_counts=base_trade_counts,
        base_component_trades=base_component_trades,
        base_context=base_context,
        stress_panel=stress_panel,
        stress_component_trades=stress_component_trades,
        base_candidate_runner_calls=per_scenario,
        stress_candidate_runner_calls=per_scenario,
        base_wall_seconds=round(base_wall_seconds, 6),
        stress_wall_seconds=round(stress_wall_seconds, 6),
        base_workers=base_workers,
        stress_workers=stress_workers,
    )


def _filter_component_trades(
    trades: pd.DataFrame,
    expert_ids: set[str],
) -> pd.DataFrame:
    """Slice the evidence component-trade frame down to one proposal's members."""
    if trades.empty or "expert_id" not in trades.columns:
        return trades
    return trades[trades["expert_id"].isin(expert_ids)]


def _run_proposal_from_evidence(
    proposal: AdmissionProposal,
    evidence: WindowScenarioEvidence,
    profile: TechnicalLibraryAdmissionRequest,
    window: RebalanceWindow,
    config: RollingAdmissionConfig,
) -> LibraryAdmissionBacktestReport:
    """Base/stress backtest for one shortlisted proposal from the shared evidence.

    The proposal only slices the evidence base/stress panels, the causal context,
    and the component trades it needs, then computes its master portfolios; it
    never loads market data or reruns a technical candidate. The rolling path
    always uses the per-symbol-winner router (hardcoded), and the observation
    gate's hurdle is derived from the proposal's measured annualized
    allocation-turnover cost scaled by ``config.hurdle_cost_multiple`` rather
    than a flat rate. ``wall_seconds`` captures the combined base+stress
    backtest wall-clock time.
    """
    started = time.perf_counter()
    expert_ids = tuple(proposal.expert_ids)
    definitions = tuple(
        definition
        for definition in evidence.definitions
        if definition.expert_id in expert_ids
    )
    if len(definitions) != len(expert_ids):
        raise DataIntegrityError(
            f"evidence is missing definitions for proposal {proposal.proposal_id}"
        )
    spec = ExpertPortfolioSpec(
        experts=definitions, router=profile.router, router_kind="per_symbol_winner_v2",
    )
    costs = CostModel()

    base_panel = _slice_scored_panel(evidence.base_panel, window.scored_start)[
        list(expert_ids)
    ]
    base_context = evidence.base_context.loc[base_panel.index]
    base = run_expert_portfolio(
        base_panel,
        spec,
        costs,
        initial_equity=config.initial_equity,
        decision_context=base_context,
    )
    base_result = _master_result(
        base, _filter_component_trades(evidence.base_component_trades, set(expert_ids)),
    )
    observation_metrics = compute_metrics(base_result.equity, base_result.trades)
    observation_gate = compute_equity_reliability_gate(
        base_result.equity, len(base_result.trades),
        config=dataclasses.replace(
            ReliabilityGateConfig(),
            hurdle_rate=derive_cost_multiple_hurdle_rate(base.allocation_cost.sum(), equity_span_years(base_result.equity), config.hurdle_cost_multiple),
        ),
    )
    observation_folds = compute_fold_distribution(base_result)

    stress_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_panel = _slice_scored_panel(evidence.stress_panel, window.scored_start)[
        list(expert_ids)
    ]
    stress = run_expert_portfolio(
        stress_panel,
        spec,
        stress_costs,
        initial_equity=config.initial_equity,
        fixed_weights=base.target_weights,
        signal_delay_bars=1,
    )
    stress_result = _master_result(
        stress,
        _filter_component_trades(evidence.stress_component_trades, set(expert_ids)),
    )
    stress_metrics = compute_metrics(stress_result.equity, stress_result.trades)
    stress_gate = compute_equity_reliability_gate(
        stress_result.equity,
        len(stress_result.trades),
        config=dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    stress_folds = compute_fold_distribution(stress_result)
    wall_seconds = time.perf_counter() - started
    promotion = compose_promotion_verdict(
        observation_gate, observation_folds, stress_gate, None,
    )
    symbols = sorted({definition.symbols[0] for definition in definitions})
    return LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=proposal.proposal_id,
        expert_ids=expert_ids,
        router=profile.router,
        window_start=str(base_panel.index[0]),
        window_end=str(base_panel.index[-1]),
        observation_metrics=observation_metrics,
        observation_gate=observation_gate,
        observation_folds=observation_folds,
        stress_metrics=stress_metrics,
        stress_gate=stress_gate,
        stress_folds=stress_folds,
        promotion=promotion,
        allocation_cost_total=float(base.allocation_cost.sum()),
        stress_allocation_cost_total=float(stress.allocation_cost.sum()),
        execution_workers=max(evidence.base_workers, evidence.stress_workers),
        wall_seconds=wall_seconds,
        code_hash=evidence.code_hash,
        data_hashes={symbol: technical_data_hashes(symbol) for symbol in symbols},
    )


def _run_proposal_backtest(
    expert_ids: tuple[str, ...],
    router: ContextualRouterSpec,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    initial_equity: float,
    max_workers: int | None,
    *,
    router_kind: Literal["global_winner_v1", "per_symbol_winner_v2"] = "global_winner_v1",
    base_delay_bars: int = 0,
    allow_same_symbol: bool = False,
) -> LibraryAdmissionBacktestReport:
    """In-memory base/stress backtest over an explicit window, no holdout seal."""
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    definitions = _materialize_selected_definitions(
        expert_ids, code_hash, allow_same_symbol=allow_same_symbol,
    )
    spec = ExpertPortfolioSpec(
        experts=definitions, router=router, router_kind=router_kind,
    )
    costs = CostModel()

    base_evidence, _ = _run_selected_tasks(
        definitions, str(start), end, costs, base_delay_bars, max_workers,
    )
    panel, component_trades = _assemble_selected_panel(base_evidence, definitions)
    context = _build_admission_context(router, panel.index, str(start), end)
    base = run_expert_portfolio(
        panel, spec, costs, initial_equity=initial_equity, decision_context=context,
    )
    base_result = _master_result(base, component_trades)
    observation_metrics = compute_metrics(base_result.equity, base_result.trades)
    observation_gate = compute_equity_reliability_gate(
        base_result.equity, len(base_result.trades),
    )
    observation_folds = compute_fold_distribution(base_result)

    stress_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_evidence, _ = _run_selected_tasks(
        definitions, str(start), end, stress_costs, 1, max_workers,
    )
    stress_panel, stress_trades = _assemble_selected_panel(stress_evidence, definitions)
    stress = run_expert_portfolio(
        stress_panel,
        spec,
        stress_costs,
        initial_equity=initial_equity,
        fixed_weights=base.target_weights,
        signal_delay_bars=1,
    )
    stress_result = _master_result(stress, stress_trades)
    stress_metrics = compute_metrics(stress_result.equity, stress_result.trades)
    stress_gate = compute_equity_reliability_gate(
        stress_result.equity,
        len(stress_result.trades),
        config=dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    stress_folds = compute_fold_distribution(stress_result)
    promotion = compose_promotion_verdict(
        observation_gate, observation_folds, stress_gate, None,
    )
    symbols = sorted({definition.symbols[0] for definition in definitions})
    return LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=admission_proposal_id(expert_ids),
        expert_ids=tuple(definition.expert_id for definition in definitions),
        router=router,
        window_start=str(panel.index[0]),
        window_end=str(panel.index[-1]),
        observation_metrics=observation_metrics,
        observation_gate=observation_gate,
        observation_folds=observation_folds,
        stress_metrics=stress_metrics,
        stress_gate=stress_gate,
        stress_folds=stress_folds,
        promotion=promotion,
        allocation_cost_total=float(base.allocation_cost.sum()),
        stress_allocation_cost_total=float(stress.allocation_cost.sum()),
        execution_workers=1,
        code_hash=code_hash,
        data_hashes={symbol: technical_data_hashes(symbol) for symbol in symbols},
    )


def _select_for_window(
    profile: TechnicalLibraryAdmissionRequest,
    window: RebalanceWindow,
    config: RollingAdmissionConfig,
    incumbent: AdmissionProposal | None,
) -> tuple[
    LibraryAdmissionReport,
    tuple[AdmissionProposal, ...],
    tuple[LibraryAdmissionBacktestReport, ...],
]:
    """Replay selection for one window using only data through ``observed_end``.

    Always runs the sole canonical priority family-unique best-first search over
    the shared base/stress evidence (per-symbol-winner routing is hardcoded).
    Backtesting is two-phase: first ``config.min_shortlist_budget`` proposals
    are generated and backtested as a probe; their measured wall-times
    extrapolate the window's effective shortlist budget via
    :func:`resolve_dynamic_shortlist_budget`. Only when that budget exceeds the
    probe size is the (cheap) best-first search re-run with the larger budget,
    and only the newly surfaced proposals are backtested — probe results are
    reused by proposal id rather than re-run. This makes the expansion driven
    purely by the measured budget, never capped by the probe generation size
    itself.
    """
    evidence = build_window_scenario_evidence(profile, window, config)
    scored_panel = _slice_scored_panel(evidence.base_panel, window.scored_start)
    scored_context = evidence.base_context.loc[scored_panel.index]
    selection = evaluate_library_admission(
        scored_panel,
        evidence.base_trade_counts,
        evidence.definitions,
        scored_context,
        profile.router,
        profile.admission,
        proposal_search="priority_family_unique_v3",
    )
    admitted_indexes = tuple(
        i for i, candidate in enumerate(selection.candidates) if candidate.admitted
    )
    if config.family_symbol_prefilter_top_k is not None:
        admitted_indexes, _dropped = prefilter_admitted_by_family_symbol(
            evidence.definitions, admitted_indexes, selection.candidates,
            config.family_symbol_prefilter_top_k,
        )
    admitted = tuple(evidence.definitions[i] for i in admitted_indexes)
    admitted_ids = {definition.expert_id for definition in admitted}
    correlation = np.empty((0, 0))
    joint_negative = np.empty((0, 0))
    compatibility = np.empty((0, 0), dtype=bool)
    search_result: BoundedProposalSearchResult | None = None
    probe_shortlist: list[AdmissionProposal] = []
    if admitted:
        completed = scored_panel[[d.expert_id for d in admitted]].to_numpy(
            dtype=np.float64,
        )[1:]
        correlation = pairwise_log_return_correlation(completed)
        joint_negative = pairwise_joint_negative_rates(completed)
        compatibility = pair_compatibility_matrix(
            completed, admitted, profile.admission, allow_same_symbol=True,
        )
        search_result = priority_shortlist_family_unique_proposals(
            admitted, correlation, joint_negative, compatibility,
            profile.admission, config.min_shortlist_budget,
        )
        probe_shortlist = (
            list(search_result.proposals)
            if search_result.generation_status == "COMPLETE"
            else []
        )

    def _backtest(proposal: AdmissionProposal) -> LibraryAdmissionBacktestReport:
        return dataclasses.replace(
            _run_proposal_from_evidence(proposal, evidence, profile, window, config),
            diversification_rank_key=proposal.rank_key(),
        )

    probe_reports = tuple(_backtest(proposal) for proposal in probe_shortlist)
    shortlist = list(probe_shortlist)
    training_reports = probe_reports
    generated_nodes = search_result.generated_nodes if search_result else 0

    if probe_reports:
        probe_wall_seconds = tuple(report.wall_seconds for report in probe_reports)
        effective_budget = resolve_dynamic_shortlist_budget(
            probe_wall_seconds, config.max_backtest_wall_seconds_per_window,
            config.min_shortlist_budget,
        )
        if effective_budget > len(probe_shortlist):
            expanded_result = priority_shortlist_family_unique_proposals(
                admitted, correlation, joint_negative, compatibility,
                profile.admission, effective_budget,
            )
            generated_nodes += expanded_result.generated_nodes
            if expanded_result.generation_status == "COMPLETE":
                expanded_shortlist = list(expanded_result.proposals)
                expanded_ids = {proposal.proposal_id for proposal in expanded_shortlist}
                kept_probe_reports = tuple(
                    report for report in probe_reports if report.proposal_id in expanded_ids
                )
                covered_ids = {report.proposal_id for report in kept_probe_reports}
                new_proposals = [
                    proposal for proposal in expanded_shortlist
                    if proposal.proposal_id not in covered_ids
                ]
                new_reports = tuple(_backtest(proposal) for proposal in new_proposals)
                shortlist = expanded_shortlist
                training_reports = kept_probe_reports + new_reports
                search_result = expanded_result

    if (
        search_result is not None
        and search_result.generation_status == "COMPLETE"
        and incumbent is not None
        and not any(
            proposal.proposal_id == incumbent.proposal_id for proposal in shortlist
        )
        and set(incumbent.expert_ids).issubset(admitted_ids)
    ):
        enriched_incumbent = enrich_proposal_diagnostics(
            incumbent, correlation, joint_negative, admitted,
        )
        shortlist.append(enriched_incumbent)
        training_reports = (*training_reports, _backtest(enriched_incumbent))

    selection = dataclasses.replace(
        selection,
        proposals=tuple(shortlist),
        generated_nodes=generated_nodes,
        generation_limit=(
            search_result.generation_limit
            if search_result else profile.admission.max_combinations
        ),
        generation_status=(
            search_result.generation_status
            if search_result else "NO_ADMITTED_CANDIDATES"
        ),
    )
    return selection, tuple(shortlist), training_reports


def _deployment_equity(
    selected: AdmissionProposal | None,
    profile: TechnicalLibraryAdmissionRequest,
    window: RebalanceWindow,
    config: RollingAdmissionConfig,
) -> pd.Series:
    """Normalized base ledger for one deployment quarter, starting at 1.0.

    CASH (``selected is None``) is a flat unit series. A selected proposal is
    backtested over the full warm-up window so router and indicator state carry
    into the quarter; only the deployment slice is returned.
    """
    if selected is None:
        index = pd.date_range(
            window.deploy_start, window.deploy_end, freq="4h", tz="UTC",
        )
        return pd.Series(1.0, index=index, name="equity", dtype="float64")
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    definitions = _materialize_selected_definitions(
        selected.expert_ids, code_hash, allow_same_symbol=True,
    )
    spec = ExpertPortfolioSpec(
        experts=definitions, router=profile.router, router_kind="per_symbol_winner_v2",
    )
    costs = CostModel()
    base_evidence, _ = _run_selected_tasks(
        definitions, str(window.load_start), window.deploy_end, costs,
        config.base_delay_bars, profile.admission.max_workers,
    )
    panel, _component_trades = _assemble_selected_panel(base_evidence, definitions)
    context = _build_admission_context(
        profile.router, panel.index, window.load_start, window.deploy_end,
    )
    base = run_expert_portfolio(
        panel, spec, costs, initial_equity=1.0, decision_context=context,
    )
    equity = base.backtest_result.equity
    segment = equity.loc[window.deploy_start:]
    if len(segment) < 1:
        raise DataIntegrityError(
            f"deployment segment is empty for {window.profile} at {window.deploy_start}"
        )
    return segment / segment.iloc[0]


def _stitch_segments(
    segments: list[pd.Series],
    proposal_ids: list[str | None],
    config: RollingAdmissionConfig,
) -> pd.Series:
    """Stitch closed deployment segments into one master ledger with switch costs.

    A switch charges ``switch_cost`` once at the first executable bar of the new
    deployment quarter; an unchanged proposal id never turns over.
    """
    if not segments:
        return pd.Series(dtype="float64")
    chunks: list[pd.Series] = []
    stitched_value = config.initial_equity
    previous_id: str | None = None
    for segment, proposal_id in zip(segments, proposal_ids, strict=True):
        segment_returns = segment.pct_change().fillna(0.0)
        if previous_id is not None and proposal_id != previous_id:
            segment_returns = segment_returns.copy()
            segment_returns.iloc[0] = (
                (1.0 + segment_returns.iloc[0]) * (1.0 - config.switch_cost) - 1.0
            )
        previous_id = proposal_id
        chunk_values = stitched_value * np.cumprod(
            1.0 + segment_returns.to_numpy(dtype=np.float64),
        )
        stitched_value = float(chunk_values[-1])
        chunks.append(
            pd.Series(chunk_values, index=segment.index, name="equity", dtype="float64"),
        )
    return pd.concat(chunks)


def _fold_summary(stitched: pd.Series) -> FoldDistributionResult:
    if (
        len(stitched) < 2
        or stitched.index[-1] < stitched.index[0] + pd.DateOffset(months=6)
    ):
        return FoldDistributionResult(
            n_folds=0, median_fold_cagr=0.0, worst_fold_cagr=0.0,
            median_fold_calmar=0.0, max_period_contribution=0.0, gate_pass=True,
        )
    return compute_equal_duration_fold_distribution(stitched, fold_duration=_FOLD_DURATION)


def _snapshot_inputs(
    request: RollingLibraryAdmissionRequest,
    window: RebalanceWindow,
    code_hash: str,
    data_hashes: dict[str, dict[str, str]],
) -> dict[str, object]:
    return {
        "profile": request.config.profile,
        "observed_end": str(window.observed_end),
        "code_hash": code_hash,
        "data_hashes": data_hashes,
        "candidate_sources": list(request.profile.candidate_sources),
        "router": dataclasses.asdict(request.profile.router),
        "admission": request.profile.admission.fingerprint(),
        "config": request.config.fingerprint(),
    }


def _run_one_rebalance(
    request: RollingLibraryAdmissionRequest,
    window: RebalanceWindow,
    incumbent: AdmissionProposal | None,
) -> tuple[
    RollingSelectionRecord,
    pd.Series | None,
    AdmissionProposal | None,
    RollingCandidateAuditRecord,
]:
    """Replay one rebalance and return its record, segment, incumbent, and audit."""
    selection, shortlist, training_reports = _select_for_window(
        request.profile, window, request.config, incumbent,
    )
    selected = select_rebalance_proposal(training_reports, incumbent)
    segment: pd.Series | None = None
    if window.status == "closed":
        segment = _deployment_equity(selected, request.profile, window, request.config)
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    data_hashes = {
        symbol: technical_data_hashes(symbol) for symbol in request.profile.symbols
    }
    snapshot_key = rebalance_snapshot_key(
        _snapshot_inputs(request, window, code_hash, data_hashes),
    )
    incumbent_kept = (
        incumbent is not None
        and selected is not None
        and selected.proposal_id == incumbent.proposal_id
    )
    audit = RollingCandidateAuditRecord.from_selection(
        window, selection, shortlist, training_reports, selected, snapshot_key,
    )
    audit = dataclasses.replace(
        audit,
        incumbent_kept=incumbent_kept,
        execution={
            "base_delay_bars": request.config.base_delay_bars,
            "stress_delay_bars": 1,
            "config_fingerprint": request.config.fingerprint(),
        },
    )
    record = RollingSelectionRecord(
        profile=request.config.profile,
        rebalance_start=str(window.rebalance_start),
        scored_start=str(window.scored_start),
        observed_end=str(window.observed_end),
        load_start=str(window.load_start),
        deploy_start=str(window.deploy_start),
        deploy_end=str(window.deploy_end),
        status=window.status,
        selection_status=selection.status,
        proposal_id=selected.proposal_id if selected is not None else None,
        expert_ids=selected.expert_ids if selected is not None else (),
        incumbent_kept=incumbent_kept,
        code_hash=code_hash,
        data_hashes=data_hashes,
        snapshot_key=snapshot_key,
    )
    return record, segment, selected, audit


def log_rolling_candidate_audit(audit: RollingCandidateAuditRecord) -> None:
    """Emit one window's candidate audit as structured DEBUG log lines.

    Replaces the old append-only ``rolling_candidate_audit.jsonl`` blob with
    flat ``[DATA]``/``[EVAL]`` tagged key=value lines following
    ``.claude/rules/logging.md``'s schema: window timing, admission aggregates,
    generation telemetry, the top-ranked shortlist summary (by
    ``selection_primary``, truncated to the top 3 with an explicit
    ``_truncated`` marker), and the final selected/CASH decision. The full
    candidates/proposals/training arrays are never logged verbatim.
    """
    window = audit.window
    selection = audit.selection
    _logger.debug(
        "[DATA] rebalance_start=%s status=%s scored_start=%s observed_end=%s",
        window.get("rebalance_start", ""), window.get("status", ""),
        window.get("scored_start", ""), window.get("observed_end", ""),
    )
    admitted = sum(1 for candidate in audit.candidates if candidate.get("admitted"))
    _logger.debug(
        "[DATA] candidates_total=%d candidates_admitted=%d coverage_states=%s coverage_sufficient=%s",
        len(audit.candidates), admitted,
        selection.get("covered_states", 0), selection.get("coverage_sufficient", False),
    )
    _logger.debug(
        "[EVAL] generation_status=%s generated_nodes=%d generation_limit=%d shortlist_size=%d",
        selection.get("generation_status", ""), selection.get("generated_nodes", 0),
        selection.get("generation_limit", 0), len(audit.shortlist),
    )
    ranked = sorted(
        audit.training,
        key=lambda entry: float(cast("float | int | str", entry.get("selection_primary", 0.0))),
    )
    max_shown = 3
    for i, entry in enumerate(ranked[:max_shown], start=1):
        _logger.debug(
            "[EVAL] top%d id=%s size=%d obs_verdict=%s stress_verdict=%s obs_lcb90=%.4f fold_pass=%s",
            i, entry.get("proposal_id", ""), entry.get("expert_count", 0),
            entry.get("observation_gate_verdict", ""), entry.get("stress_gate_verdict", ""),
            float(cast("float | int | str", entry.get("observation_lcb90", 0.0))),
            entry.get("observation_folds_pass", False),
        )
    if len(ranked) > max_shown:
        _logger.debug(
            "[EVAL] top%d_of_%d_truncated=%d", max_shown, len(ranked), len(ranked),
        )
    selected_id = audit.selected.get("proposal_id") if audit.selected else "CASH"
    _logger.debug(
        "[EVAL] selected=%s reason=%s", selected_id, audit.cash_reason or "",
    )


def run_rolling_library_admission(
    request: RollingLibraryAdmissionRequest,
    *,
    ledger_path: Path = REBALANCE_LEDGER_PATH,
    pointer_path: Path = CURRENT_PROFILE_POINTER_PATH,
) -> RollingLibraryAdmissionReport:
    """Replay every eligible rebalance through ``as_of`` and stitch closed quarters.

    Each historical window freezes its own ``observed_end`` snapshot, warm-up
    observations never enter scoring or OOS PnL, a switch is charged once at the
    following executable bar, and decisions are written append-only and
    idempotently. Every window also emits one structured candidate audit as
    DEBUG log lines via :func:`log_rolling_candidate_audit` (no audit file is
    written); the audit never touches the rebalance ledger, pointer, catalog, or
    trading state. A request for a complete historical verdict fails closed
    while the newest deployment quarter is still incomplete.
    """
    window = resolve_common_technical_window(
        request.profile.symbols, None, request.as_of,
    )
    windows = build_rolling_rebalance_schedule(
        window.common_start, request.as_of, request.config,
    )
    if not windows:
        _logger.info(
            "[EVAL] rolling status=NO_WINDOWS profile=%s as_of=%s common_start=%s",
            request.config.profile, request.as_of, window.common_start,
        )
        return RollingLibraryAdmissionReport(
            status="NO_WINDOWS",
            profile=request.config.profile,
            mode=request.mode,
            as_of=str(pd.Timestamp(request.as_of, tz="UTC")),
            common_start=str(window.common_start),
            common_end=str(window.common_end),
            windows=(),
            records=(),
            n_folds=0,
            median_fold_cagr=0.0,
            worst_fold_cagr=0.0,
            median_fold_calmar=0.0,
            max_period_contribution=0.0,
            fold_gate_pass=True,
            oos_start="",
            oos_end="",
            oos_return=0.0,
        )

    if request.require_complete_history and any(
        w.status == "live_or_partial" for w in windows
    ):
        raise ValueError(
            "request targets an incomplete historical segment: the latest "
            f"deployment quarter {windows[-1].deploy_start} is still open at "
            f"as_of {request.as_of}"
        )

    records: list[RollingSelectionRecord] = []
    completed_windows: list[RebalanceWindow] = []
    segments: list[pd.Series] = []
    deployment_proposal_ids: list[str | None] = []
    incumbent: AdmissionProposal | None = None
    failure: dict[str, object] | None = None
    status = "COMPLETE"
    for w in windows:
        try:
            record, segment, incumbent, audit = _run_one_rebalance(request, w, incumbent)
        except DataIntegrityError as exc:
            code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
            data_hashes = {
                symbol: technical_data_hashes(symbol)
                for symbol in request.profile.symbols
            }
            snapshot_key = rebalance_snapshot_key(
                _snapshot_inputs(request, w, code_hash, data_hashes),
            )
            log_rolling_candidate_audit(
                RollingCandidateAuditRecord.from_failure(w, snapshot_key, str(exc)),
            )
            _logger.warning(
                "[EVAL] rolling status=PARTIAL_FAILURE window=%s reason=%s",
                w.rebalance_start, exc,
            )
            failure = {
                "rebalance_start": str(w.rebalance_start),
                "reason": str(exc),
            }
            status = "PARTIAL_FAILURE"
            break
        stored = append_rebalance_record(record, ledger_path)
        records.append(stored)
        write_current_profile(stored, pointer_path)
        completed_windows.append(w)
        log_rolling_candidate_audit(audit)
        if w.status == "closed":
            segments.append(segment)
            deployment_proposal_ids.append(record.proposal_id)
        _logger.info(
            "[EVAL] rolling window=%s status=%s selection=%s proposal=%s",
            w.rebalance_start, w.status, record.selection_status, record.proposal_id,
        )

    stitched = _stitch_segments(segments, deployment_proposal_ids, request.config)
    folds = _fold_summary(stitched)
    oos_start = str(stitched.index[0]) if len(stitched) else ""
    oos_end = str(stitched.index[-1]) if len(stitched) else ""
    oos_return = float(stitched.iloc[-1] / stitched.iloc[0] - 1.0) if len(stitched) else 0.0
    report = RollingLibraryAdmissionReport(
        status=status,
        profile=request.config.profile,
        mode=request.mode,
        as_of=str(pd.Timestamp(request.as_of, tz="UTC")),
        common_start=str(window.common_start),
        common_end=str(window.common_end),
        windows=tuple(completed_windows),
        records=tuple(records),
        n_folds=folds.n_folds,
        median_fold_cagr=folds.median_fold_cagr,
        worst_fold_cagr=folds.worst_fold_cagr,
        median_fold_calmar=folds.median_fold_calmar,
        max_period_contribution=folds.max_period_contribution,
        fold_gate_pass=folds.gate_pass,
        oos_start=oos_start,
        oos_end=oos_end,
        oos_return=oos_return,
        failure=failure,
    )
    _logger.info(
        "[EVAL] rolling status=%s profile=%s windows=%d closed=%d folds=%d oos_return=%.4f",
        report.status, request.config.profile, len(completed_windows),
        report.closed_quarter_count, report.n_folds, report.oos_return,
    )
    return report


def _check_contract() -> None:
    """Executable assertions locking the rolling application surface."""
    assert run_rolling_library_admission.__name__ == "run_rolling_library_admission"
    assert build_window_scenario_evidence.__name__ == "build_window_scenario_evidence"
    assert log_rolling_candidate_audit.__name__ == "log_rolling_candidate_audit"


_check_contract()

__all__ = [
    "RollingLibraryAdmissionRequest",
    "WindowScenarioEvidence",
    "build_window_scenario_evidence",
    "log_rolling_candidate_audit",
    "run_rolling_library_admission",
]
