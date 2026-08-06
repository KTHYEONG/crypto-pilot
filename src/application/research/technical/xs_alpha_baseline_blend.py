"""Application-layer blend of ``xs_alpha_vol_weighted_v6`` with the frozen baseline.

Wires the two pure recombination primitives from
:mod:`src.research.technical_experts.xs_alpha_baseline_blend` onto the real
ledgers: v6's XS book is replayed with the existing
``build_xs_alpha_vol_weighted_weights`` + ``run_xs_composite_ledger`` path, the
frozen single-symbol Donchian baseline (``StrategySpec()`` defaults) is replayed
with the unchanged ``run_backtest``, both net-return legs are aligned on the
common bar grid, the sleeve weight is selected strictly on the discovery
window, the blend is applied over the full history, and every admission gate is
re-verified on the blended ledger. ``evaluate_xs_reliability`` is run on the
qualification(+holdout) stitched OOS window via the shared
``_oos_reliability_window`` helper -- the actual reliability gate this cycle is
aimed at, measured honestly rather than assumed from any diagnostic. Holdout
stays sealed unless ``unseal_holdout`` is set.

:func:`run_xs_alpha_baseline_blend_sized` is the sibling that adds the two
sizing levers: the sleeve weight is chosen by the worst-year-robust selector
and the gross leverage is chosen strictly from the discovery-window blended
net returns via ``solve_growth_optimal_risk`` (``use_drawdown_overlay=False``)
and applied as a pure linear scale -- the first configuration where the
reliability gate is reachable at all this project-cycle, reported exactly as
measured, never assumed to pass.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.research.technical.reliability_ledger import (
    persist_reliability_ledger_entry,
)
from src.application.research.technical.trend_screen import _load_symbol_data
from src.application.research.technical.xs_alpha_growth_sizing import (
    _realised_turnover,
    _sizing_payload,
)
from src.application.research.technical.xs_trend_screen import (
    XS_DISCOVERY_START,
    XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    _admission_payload,
    _bar_funding_series,
    _common_index,
    _fingerprint_without_self,
    _oos_reliability_window,
    _reliability_payload,
    _window_series,
)
from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import run_backtest
from src.research.contracts import CostModel, StrategySpec
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    block_size_search_hit_cap,
    compute_turnover_fold_upper_bound,
    derive_cost_multiple_hurdle_rate,
    derive_realized_weights_cost_total,
    equity_span_years,
)
from src.research.risk.growth_sizing import (
    GrowthSizingConfig,
    GrowthSizingResult,
    solve_growth_optimal_risk,
)
from src.research.sleeve_blend.tournament import (
    TOURNAMENT_RETURN_SOURCES,
    _run_source,
)
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsAdmissionResult,
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    XsReliabilityResult,
    build_xs_alpha_vol_weighted_weights,
    evaluate_xs_admission,
    evaluate_xs_reliability,
    run_xs_composite_ledger,
)
from src.research.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    QUALIFICATION_END,
    QUALIFICATION_START,
    TREND_SCREEN_CANDIDATES,
    TREND_SCREEN_SYMBOLS,
)
from src.research.technical_experts.xs_alpha_baseline_blend import (
    _blended_sharpe,
    _discovery_common,
    apply_fixed_gross_leverage,
    build_blended_ledger,
    compute_discovery_correlation,
    discovery_reliability_score,
    select_baseline_blend_weight,
    select_best_baseline_leg,
    select_robust_baseline_blend_weight,
)

# Pre-registered, frozen weight grid (v8): selection is argmax annualized
# Sharpe on the discovery window only. A coarse grid is deliberate -- the
# whole point is a small, bounded, deterministic search, not matrix inversion.
_DEFAULT_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

# Pre-registered, frozen gross-leverage grid (robust-blend sizing): 0.5 steps
# matching ``_XS_GROWTH_SIZING_CONFIG``'s spacing convention, denser than its
# ``(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)`` grid because the §2b passing band is
# only ~0.8x wide. ``reference_risk = 1.0`` makes grid values literal gross-
# leverage multiples, exactly the pure-linear scale §2b swept.
_DEFAULT_RISK_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

# Frozen comparison order for the baseline-leg selection: the tournament's
# five source-controlled candidates first (Donchian listed first so an exact
# blended-Sharpe tie reproduces today's status quo baseline choice -- the
# tie-break in ``select_best_baseline_leg`` keeps the earliest id), then every
# trend-screen catalog identity not already among them. Computed once at import
# time from the two frozen registries (no manual enumeration): 5 + 30 - 3 = 32
# distinct ids (revision 2 widening, answering "were only 5 tried").
_DEFAULT_CANDIDATE_ORDER: tuple[str, ...] = (
    TOURNAMENT_RETURN_SOURCES
    + tuple(
        candidate.candidate_id
        for candidate in TREND_SCREEN_CANDIDATES
        if candidate.candidate_id not in TOURNAMENT_RETURN_SOURCES
    )
)

# Candidate-id -> frozen trend-screen identity lookup for the non-tournament
# dispatch branch of ``run_xs_alpha_baseline_leg_selection``.
_TREND_SCREEN_CANDIDATES_BY_ID: dict[str, TechnicalCandidate] = {
    candidate.candidate_id: candidate for candidate in TREND_SCREEN_CANDIDATES
}

# Frozen persistence name (v8 of the XS alpha family), mirroring
# ``xs_growth_sizing_report_path``'s naming convention.
_BLEND_REPORT_NAME = "xs_alpha_baseline_blend_v8"

# Jointly-discovered blend parameters (ADR_20260805_xs_alpha_baseline_blend_joint_search):
# frozen from the one-time ``tools/research/xs_alpha_blend_joint_search.py`` run's printed
# ``best_params`` when ``plateau_passed=True``, else the v8_sized fallback ``(0.6, 1.0)``
# (honest negative-result rule -- never hand-pick a point to force a nicer number). The
# measured run reported ``plateau_passed=True`` (best_is_score=0.313846,
# plateau_neighbor_ratio=0.9100), so the printed best point is frozen here at full
# precision. They are referenced only by the CLI handler's ``set_defaults``:
# ``run_xs_alpha_baseline_blend_joint`` itself takes both as required keyword arguments with
# no defaults.
_JOINT_XS_ALPHA_WEIGHT: float = 0.2713516311918738
_JOINT_LEVERAGE_SCALE: float = 3.9768046145974894


@dataclass(frozen=True, slots=True)
class XsBaselineBlendReport:
    """Deterministic persisted outcome of one XS-alpha x baseline blend run."""

    profile: str
    blend_weight: float
    weight_grid: tuple[float, ...]
    discovery: XsAdmissionResult
    qualification: XsAdmissionResult
    holdout: XsAdmissionResult | None
    pre_blend_discovery: XsAdmissionResult
    pre_blend_qualification: XsAdmissionResult
    pre_blend_holdout: XsAdmissionResult | None
    baseline_discovery: XsAdmissionResult
    baseline_qualification: XsAdmissionResult
    baseline_holdout: XsAdmissionResult | None
    reliability: XsReliabilityResult | None = None
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "report_fingerprint", _fingerprint_without_self(self._body_payload()),
        )

    def _body_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "blend_weight": round(self.blend_weight, 8),
            "weight_grid": list(self.weight_grid),
            "discovery": _admission_payload(self.discovery),
            "qualification": _admission_payload(self.qualification),
            "holdout": (
                _admission_payload(self.holdout) if self.holdout is not None else None
            ),
            "pre_blend_discovery": _admission_payload(self.pre_blend_discovery),
            "pre_blend_qualification": _admission_payload(self.pre_blend_qualification),
            "pre_blend_holdout": (
                _admission_payload(self.pre_blend_holdout)
                if self.pre_blend_holdout is not None else None
            ),
            "baseline_discovery": _admission_payload(self.baseline_discovery),
            "baseline_qualification": _admission_payload(self.baseline_qualification),
            "baseline_holdout": (
                _admission_payload(self.baseline_holdout)
                if self.baseline_holdout is not None else None
            ),
            "reliability": (
                _reliability_payload(self.reliability)
                if self.reliability is not None else None
            ),
        }

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload = self._body_payload()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"

@dataclass(frozen=True, slots=True)
class XsBaselineBlendSizedReport:
    """Deterministic persisted outcome of one robust-blend + growth-sizing run.

    Same before/after/baseline admission structure as
    :class:`XsBaselineBlendReport` plus the growth-sizing result: the
    worst-year-robust blend weight, the discovery-only ``selected_risk``, and
    every gate re-verified on the final scaled ledger. The measured
    ``selected_risk`` and verdict are reported as-is -- never assumed to land
    inside the narrow §2b passing band.
    """

    profile: str
    blend_weight: float
    weight_grid: tuple[float, ...]
    sizing: GrowthSizingResult
    discovery: XsAdmissionResult
    qualification: XsAdmissionResult
    holdout: XsAdmissionResult | None
    pre_blend_discovery: XsAdmissionResult
    pre_blend_qualification: XsAdmissionResult
    pre_blend_holdout: XsAdmissionResult | None
    baseline_discovery: XsAdmissionResult
    baseline_qualification: XsAdmissionResult
    baseline_holdout: XsAdmissionResult | None
    reliability: XsReliabilityResult | None = None
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "report_fingerprint", _fingerprint_without_self(self._body_payload()),
        )

    def _body_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "blend_weight": round(self.blend_weight, 8),
            "weight_grid": list(self.weight_grid),
            "sizing": _sizing_payload(self.sizing),
            "discovery": _admission_payload(self.discovery),
            "qualification": _admission_payload(self.qualification),
            "holdout": (
                _admission_payload(self.holdout) if self.holdout is not None else None
            ),
            "pre_blend_discovery": _admission_payload(self.pre_blend_discovery),
            "pre_blend_qualification": _admission_payload(self.pre_blend_qualification),
            "pre_blend_holdout": (
                _admission_payload(self.pre_blend_holdout)
                if self.pre_blend_holdout is not None else None
            ),
            "baseline_discovery": _admission_payload(self.baseline_discovery),
            "baseline_qualification": _admission_payload(self.baseline_qualification),
            "baseline_holdout": (
                _admission_payload(self.baseline_holdout)
                if self.baseline_holdout is not None else None
            ),
            "reliability": (
                _reliability_payload(self.reliability)
                if self.reliability is not None else None
            ),
        }

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload = self._body_payload()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"

@dataclass(frozen=True, slots=True)
class XsBaselineBlendJointReport:
    """Deterministic persisted outcome of one joint-searched blend run.

    Same before/after/baseline admission structure as
    :class:`XsBaselineBlendSizedReport`, but the two free parameters
    (``xs_alpha_weight``, ``leverage_scale``) are explicit inputs instead of
    being selected inside the orchestrator: the joint discovery-only search
    (``tools/research/xs_alpha_blend_joint_search.py``) produces them once, the
    CLI handler freezes them as defaults, and this report measures the real
    gates at exactly that point -- reported as measured, never assumed to pass.
    """

    profile: str
    xs_alpha_weight: float
    leverage_scale: float
    discovery: XsAdmissionResult
    qualification: XsAdmissionResult
    holdout: XsAdmissionResult | None
    pre_blend_discovery: XsAdmissionResult
    pre_blend_qualification: XsAdmissionResult
    pre_blend_holdout: XsAdmissionResult | None
    baseline_discovery: XsAdmissionResult
    baseline_qualification: XsAdmissionResult
    baseline_holdout: XsAdmissionResult | None
    reliability: XsReliabilityResult | None = None
    qualification_turnover_neutral: XsAdmissionResult | None = None
    qualification_worst_fold_turnover: float | None = None
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "report_fingerprint", _fingerprint_without_self(self._body_payload()),
        )

    def _body_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "xs_alpha_weight": round(self.xs_alpha_weight, 8),
            "leverage_scale": round(self.leverage_scale, 8),
            "discovery": _admission_payload(self.discovery),
            "qualification": _admission_payload(self.qualification),
            "holdout": (
                _admission_payload(self.holdout) if self.holdout is not None else None
            ),
            "pre_blend_discovery": _admission_payload(self.pre_blend_discovery),
            "pre_blend_qualification": _admission_payload(self.pre_blend_qualification),
            "pre_blend_holdout": (
                _admission_payload(self.pre_blend_holdout)
                if self.pre_blend_holdout is not None else None
            ),
            "baseline_discovery": _admission_payload(self.baseline_discovery),
            "baseline_qualification": _admission_payload(self.baseline_qualification),
            "baseline_holdout": (
                _admission_payload(self.baseline_holdout)
                if self.baseline_holdout is not None else None
            ),
            "reliability": (
                _reliability_payload(self.reliability)
                if self.reliability is not None else None
            ),
            "qualification_turnover_neutral": (
                _admission_payload(self.qualification_turnover_neutral)
                if self.qualification_turnover_neutral is not None else None
            ),
            "qualification_worst_fold_turnover": (
                _round_finite(self.qualification_worst_fold_turnover)
                if self.qualification_worst_fold_turnover is not None else None
            ),
        }

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload = self._body_payload()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"


def _round_finite(value: float) -> float:
    """Round a diagnostic float to 8 decimals, coercing non-finite to 0.0.

    ``candidate_diagnostics`` values must persist as a strict JSON document --
    NaN/Infinity tokens are never written. The blended-Sharpe diagnostic is
    theoretically ``+inf`` for an exactly zero-variance blend (``_blended_sharpe``'s
    dominant case); in practice it is always finite, and this guard keeps the
    payload valid either way.
    """
    rounded = round(value, 8)
    return rounded if math.isfinite(rounded) else 0.0


@dataclass(frozen=True, slots=True)
class XsBaselineLegSelectionReport:
    """Deterministic persisted outcome of one baseline-leg comparison run.

    Same before/after admission and reliability structure as
    :class:`XsBaselineBlendReport` (profile, blend_weight, discovery/
    qualification/(holdout), pre-blend admission, reliability) plus the
    comparison outcome: the selected candidate id and one diagnostics entry
    per evaluated candidate carrying its discovery-window correlation, selected
    blend weight, and achieved blended Sharpe. Reported exactly as measured --
    the winner is never auto-promoted into any live entry point.
    """

    profile: str
    blend_weight: float
    weight_grid: tuple[float, ...]
    discovery: XsAdmissionResult
    qualification: XsAdmissionResult
    holdout: XsAdmissionResult | None
    pre_blend_discovery: XsAdmissionResult
    pre_blend_qualification: XsAdmissionResult
    pre_blend_holdout: XsAdmissionResult | None
    selected_candidate: str
    candidate_diagnostics: dict[str, dict[str, float]]
    reliability: XsReliabilityResult | None = None
    reliability_hurdle_neutral: XsReliabilityResult | None = None
    reliability_point_to_lcb90_ratio: float | None = None
    reliability_block_size_hit_cap: bool | None = None
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "report_fingerprint", _fingerprint_without_self(self._body_payload()),
        )

    def _body_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "blend_weight": round(self.blend_weight, 8),
            "weight_grid": list(self.weight_grid),
            "discovery": _admission_payload(self.discovery),
            "qualification": _admission_payload(self.qualification),
            "holdout": (
                _admission_payload(self.holdout) if self.holdout is not None else None
            ),
            "pre_blend_discovery": _admission_payload(self.pre_blend_discovery),
            "pre_blend_qualification": _admission_payload(self.pre_blend_qualification),
            "pre_blend_holdout": (
                _admission_payload(self.pre_blend_holdout)
                if self.pre_blend_holdout is not None else None
            ),
            "selected_candidate": self.selected_candidate,
            "candidate_diagnostics": {
                candidate_id: {
                    key: _round_finite(value) for key, value in diag.items()
                }
                for candidate_id, diag in self.candidate_diagnostics.items()
            },
            "reliability": (
                _reliability_payload(self.reliability)
                if self.reliability is not None else None
            ),
            "reliability_hurdle_neutral": (
                _reliability_payload(self.reliability_hurdle_neutral)
                if self.reliability_hurdle_neutral is not None else None
            ),
            "reliability_point_to_lcb90_ratio": self.reliability_point_to_lcb90_ratio,
            "reliability_block_size_hit_cap": self.reliability_block_size_hit_cap,
        }

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload = self._body_payload()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"


def _baseline_realized_position(
    trades: pd.DataFrame,
    frame_index: pd.DatetimeIndex,
    grid: pd.DatetimeIndex,
) -> pd.Series:
    """In/out position path of the frozen baseline, reindexed to the common grid.

    The baseline is single-position and directional, so its realized weight is
    1.0 while a trade is open and 0.0 while flat. This feeds
    ``count_closed_trades`` inside ``evaluate_xs_reliability`` as a coarse
    sleeve-rebalance proxy (semantically a coarser analog of the existing
    per-symbol trade count, not identical to it).
    """
    position = np.zeros(len(frame_index), dtype=np.float64)
    if len(trades) > 0:
        entry = trades["entry_bar"].to_numpy(dtype=np.int64)
        exit_bar = trades["exit_bar"].to_numpy(dtype=np.int64)
        for start, stop in zip(entry, exit_bar, strict=True):
            position[start : stop + 1] = 1.0
    return pd.Series(position, index=frame_index, name="baseline_position").reindex(grid)


def run_xs_alpha_baseline_blend(
    *,
    profile: str = XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    unseal_holdout: bool = False,
    weight_grid: tuple[float, ...] = _DEFAULT_WEIGHT_GRID,
) -> XsBaselineBlendReport:
    """Execute the v6 x baseline variance-reduction blend end to end.

    Loads v6's XS ledger and the frozen ``StrategySpec()`` Donchian baseline on
    the same bar grid, selects the sleeve weight strictly on the discovery
    window via :func:`select_baseline_blend_weight`, applies it over the full
    history via :func:`build_blended_ledger`, re-verifies ``evaluate_xs_admission``
    on the blended ledger (discovery/qualification, plus holdout only when
    ``unseal_holdout``), and evaluates ``evaluate_xs_reliability`` on the
    stitched OOS window. Pre-blend (v6 alone) and baseline-alone admission are
    recorded for an honest before/after comparison.
    """
    if profile != XS_VOL_WEIGHTED_ALPHA_PROFILE_ID:
        raise ValueError(
            f"baseline blending is restricted to '{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}'; "
            f"got '{profile}'"
        )
    end = resolve_evaluation_end(None, unseal_holdout=unseal_holdout)
    execution_spec = XsCompositeSpec()
    alpha_spec = XsAlphaCompositeSpec()

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        frame, funding, _fingerprint, _coverage = _load_symbol_data(
            symbol, XS_DISCOVERY_START, end,
        )
        data[symbol] = (frame, funding)

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        raise DataIntegrityError("xs baseline blend requires at least 2 common bars")

    opens = pd.DataFrame(
        {symbol: frame["open"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    bar_funding = pd.DataFrame(
        {
            symbol: _bar_funding_series(funding, frame.index).reindex(common)
            for symbol, (frame, funding) in data.items()
        },
    )
    closes = pd.DataFrame(
        {symbol: frame["close"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    taker = pd.DataFrame(
        {symbol: frame["taker_buy_ratio"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )

    opens_arr = opens.to_numpy(dtype=np.float64)
    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0
    benchmark = pd.Series(o2o.mean(axis=1), index=common, name="benchmark")

    weights = build_xs_alpha_vol_weighted_weights(
        closes, taker, bar_funding, opens, alpha_spec, execution_spec,
    )
    xs_equity, _xs_turnover = run_xs_composite_ledger(
        weights, opens, bar_funding, execution_spec,
    )
    xs_alpha_net = xs_equity.pct_change().fillna(0.0).rename("xs_alpha_net")

    btc_frame, btc_funding = data["BTCUSDT"]
    baseline_result = run_backtest(
        btc_frame, StrategySpec(), CostModel(), funding_rates=btc_funding,
    )
    baseline_equity = baseline_result.equity.reindex(common).rename("baseline_equity")
    baseline_net = baseline_equity.pct_change().fillna(0.0).rename("baseline_net")
    baseline_realized_weight = _baseline_realized_position(
        baseline_result.trades, btc_frame.index, common,
    )

    blend_weight = select_baseline_blend_weight(
        xs_alpha_net, baseline_net, XS_DISCOVERY_START, DISCOVERY_END, weight_grid,
    )

    xs_realized_weights = weights.shift(1 + execution_spec.execution_delay_bars).fillna(0.0)
    blended_equity, blended_weights = build_blended_ledger(
        xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight,
        blend_weight,
    )
    blended_turnover = _realised_turnover(blended_weights)
    admission_config = XsAdmissionConfig()

    discovery = evaluate_xs_admission(
        _window_series(blended_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(blended_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    qualification = evaluate_xs_admission(
        _window_series(blended_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(blended_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )

    holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    if unseal_holdout:
        post_cutoff = common[common > HOLDOUT_CUTOFF]
        if len(post_cutoff) >= 2:
            holdout_start = post_cutoff[0]
            holdout_end = post_cutoff[-1]
            holdout = evaluate_xs_admission(
                _window_series(blended_equity, holdout_start, holdout_end),
                _window_series(blended_turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                admission_config,
            )

    xs_turnover = _realised_turnover(xs_realized_weights)
    pre_blend_discovery = evaluate_xs_admission(
        _window_series(xs_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(xs_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    pre_blend_qualification = evaluate_xs_admission(
        _window_series(xs_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(xs_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    pre_blend_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        pre_blend_holdout = evaluate_xs_admission(
            _window_series(xs_equity, holdout_start, holdout_end),
            _window_series(xs_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    baseline_turnover = _realised_turnover(
        pd.DataFrame({"baseline": baseline_realized_weight}, index=common),
    )
    baseline_discovery = evaluate_xs_admission(
        _window_series(baseline_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(baseline_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    baseline_qualification = evaluate_xs_admission(
        _window_series(baseline_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(baseline_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    baseline_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        baseline_holdout = evaluate_xs_admission(
            _window_series(baseline_equity, holdout_start, holdout_end),
            _window_series(baseline_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    reliability: XsReliabilityResult | None = None
    oos_window = _oos_reliability_window(
        blended_equity, blended_weights, QUALIFICATION_START, QUALIFICATION_END,
        holdout_start, holdout_end,
    )
    if oos_window is not None:
        oos_equity, oos_weights = oos_window
        reliability = evaluate_xs_reliability(
            oos_equity,
            oos_weights,
            dataclasses.replace(
                ReliabilityGateConfig(),
                hurdle_rate=derive_cost_multiple_hurdle_rate(
                    derive_realized_weights_cost_total(
                        oos_weights, admission_config.round_trip_cost_rate
                    ),
                    equity_span_years(oos_equity),
                    2.0,
                ),
            ),
        )

    return XsBaselineBlendReport(
        profile=profile,
        blend_weight=blend_weight,
        weight_grid=weight_grid,
        discovery=discovery,
        qualification=qualification,
        holdout=holdout,
        pre_blend_discovery=pre_blend_discovery,
        pre_blend_qualification=pre_blend_qualification,
        pre_blend_holdout=pre_blend_holdout,
        baseline_discovery=baseline_discovery,
        baseline_qualification=baseline_qualification,
        baseline_holdout=baseline_holdout,
        reliability=reliability,
    )

def run_xs_alpha_baseline_blend_sized(
    *,
    unseal_holdout: bool = False,
    weight_grid: tuple[float, ...] = _DEFAULT_WEIGHT_GRID,
    risk_grid: tuple[float, ...] = _DEFAULT_RISK_GRID,
) -> XsBaselineBlendSizedReport:
    """Execute the robust-blend + growth-optimal-leverage pipeline end to end.

    Sibling to :func:`run_xs_alpha_baseline_blend` adding two levers: the
    sleeve weight is selected by the worst-year-robust criterion
    (:func:`select_robust_baseline_blend_weight`), and the gross leverage is
    selected strictly from the discovery-window blended net returns via
    ``solve_growth_optimal_risk`` (``use_drawdown_overlay=False`` -- the
    drawdown ladder is measured net-harmful to LCB90) then applied as a pure
    linear scale (:func:`apply_fixed_gross_leverage`). Every admission gate
    and the reliability gate are re-verified on the final scaled ledger; the
    measured ``selected_risk`` and verdict are reported exactly as measured.
    """
    end = resolve_evaluation_end(None, unseal_holdout=unseal_holdout)
    execution_spec = XsCompositeSpec()
    alpha_spec = XsAlphaCompositeSpec()

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        frame, funding, _fingerprint, _coverage = _load_symbol_data(
            symbol, XS_DISCOVERY_START, end,
        )
        data[symbol] = (frame, funding)

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        raise DataIntegrityError("xs baseline blend requires at least 2 common bars")

    opens = pd.DataFrame(
        {symbol: frame["open"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    bar_funding = pd.DataFrame(
        {
            symbol: _bar_funding_series(funding, frame.index).reindex(common)
            for symbol, (frame, funding) in data.items()
        },
    )
    closes = pd.DataFrame(
        {symbol: frame["close"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    taker = pd.DataFrame(
        {symbol: frame["taker_buy_ratio"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )

    opens_arr = opens.to_numpy(dtype=np.float64)
    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0
    benchmark = pd.Series(o2o.mean(axis=1), index=common, name="benchmark")

    weights = build_xs_alpha_vol_weighted_weights(
        closes, taker, bar_funding, opens, alpha_spec, execution_spec,
    )
    xs_equity, _xs_turnover = run_xs_composite_ledger(
        weights, opens, bar_funding, execution_spec,
    )
    xs_alpha_net = xs_equity.pct_change().fillna(0.0).rename("xs_alpha_net")

    btc_frame, btc_funding = data["BTCUSDT"]
    baseline_result = run_backtest(
        btc_frame, StrategySpec(), CostModel(), funding_rates=btc_funding,
    )
    baseline_equity = baseline_result.equity.reindex(common).rename("baseline_equity")
    baseline_net = baseline_equity.pct_change().fillna(0.0).rename("baseline_net")
    baseline_realized_weight = _baseline_realized_position(
        baseline_result.trades, btc_frame.index, common,
    )

    blend_weight = select_robust_baseline_blend_weight(
        xs_alpha_net, baseline_net, XS_DISCOVERY_START, DISCOVERY_END, weight_grid,
    )

    xs_realized_weights = weights.shift(1 + execution_spec.execution_delay_bars).fillna(0.0)
    blended_equity, blended_weights = build_blended_ledger(
        xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight,
        blend_weight,
    )
    blended_net = blended_equity.pct_change().fillna(0.0).rename("blended_net")

    discovery_blended_net = blended_net[
        (blended_net.index >= XS_DISCOVERY_START) & (blended_net.index <= DISCOVERY_END)
    ]
    sizing = solve_growth_optimal_risk(
        discovery_blended_net.to_numpy(dtype=np.float64),
        GrowthSizingConfig(
            risk_grid=risk_grid, reference_risk=1.0, max_drawdown=0.20,
        ),
        use_drawdown_overlay=False,
    )

    if sizing.selected_risk is None:
        scaled_equity = blended_equity
        scaled_weights = blended_weights
    else:
        scaled_net, scaled_weights = apply_fixed_gross_leverage(
            blended_net, blended_weights, sizing.selected_risk,
        )
        scaled_equity = pd.Series(
            float(blended_equity.iloc[0]) * np.cumprod(
                1.0 + scaled_net.to_numpy(dtype=np.float64),
            ),
            index=blended_equity.index,
            name="scaled_equity",
        )
    scaled_turnover = _realised_turnover(scaled_weights)

    admission_config = XsAdmissionConfig()
    discovery = evaluate_xs_admission(
        _window_series(scaled_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(scaled_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    qualification = evaluate_xs_admission(
        _window_series(scaled_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(scaled_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )

    holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    if unseal_holdout:
        post_cutoff = common[common > HOLDOUT_CUTOFF]
        if len(post_cutoff) >= 2:
            holdout_start = post_cutoff[0]
            holdout_end = post_cutoff[-1]
            holdout = evaluate_xs_admission(
                _window_series(scaled_equity, holdout_start, holdout_end),
                _window_series(scaled_turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                admission_config,
            )

    xs_turnover = _realised_turnover(xs_realized_weights)
    pre_blend_discovery = evaluate_xs_admission(
        _window_series(xs_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(xs_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    pre_blend_qualification = evaluate_xs_admission(
        _window_series(xs_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(xs_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    pre_blend_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        pre_blend_holdout = evaluate_xs_admission(
            _window_series(xs_equity, holdout_start, holdout_end),
            _window_series(xs_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    baseline_turnover = _realised_turnover(
        pd.DataFrame({"baseline": baseline_realized_weight}, index=common),
    )
    baseline_discovery = evaluate_xs_admission(
        _window_series(baseline_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(baseline_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    baseline_qualification = evaluate_xs_admission(
        _window_series(baseline_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(baseline_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    baseline_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        baseline_holdout = evaluate_xs_admission(
            _window_series(baseline_equity, holdout_start, holdout_end),
            _window_series(baseline_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    reliability: XsReliabilityResult | None = None
    oos_window = _oos_reliability_window(
        scaled_equity, scaled_weights, QUALIFICATION_START, QUALIFICATION_END,
        holdout_start, holdout_end,
    )
    if oos_window is not None:
        oos_equity, oos_weights = oos_window
        reliability = evaluate_xs_reliability(
            oos_equity,
            oos_weights,
            dataclasses.replace(
                ReliabilityGateConfig(),
                hurdle_rate=derive_cost_multiple_hurdle_rate(
                    derive_realized_weights_cost_total(
                        oos_weights, admission_config.round_trip_cost_rate
                    ),
                    equity_span_years(oos_equity),
                    2.0,
                ),
            ),
        )

    return XsBaselineBlendSizedReport(
        profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
        blend_weight=blend_weight,
        weight_grid=weight_grid,
        sizing=sizing,
        discovery=discovery,
        qualification=qualification,
        holdout=holdout,
        pre_blend_discovery=pre_blend_discovery,
        pre_blend_qualification=pre_blend_qualification,
        pre_blend_holdout=pre_blend_holdout,
        baseline_discovery=baseline_discovery,
        baseline_qualification=baseline_qualification,
        baseline_holdout=baseline_holdout,
        reliability=reliability,
    )

def run_xs_alpha_baseline_blend_joint(
    *,
    xs_alpha_weight: float,
    leverage_scale: float,
    unseal_holdout: bool = False,
) -> XsBaselineBlendJointReport:
    """Execute the joint-searched blend at an explicit weight and leverage.

    Third sibling in the blend family: the two free parameters discovered once
    by the joint search (``tools/research/xs_alpha_blend_joint_search.py``) are
    **required keyword arguments -- this function has no defaults**, so it stays
    callable/testable with arbitrary values (e.g. by a plateau re-check) without
    ``optuna`` installed and without the search having been run. Pipeline: v6's
    ledger and the frozen ``StrategySpec()``/``CostModel()`` baseline are replayed
    exactly as :func:`run_xs_alpha_baseline_blend` does; the blend weight is
    applied over the full history via :func:`build_blended_ledger`, the gross
    leverage is applied over the full history as a pure linear scale via
    :func:`apply_fixed_gross_leverage` (never the drawdown ladder, never
    ``solve_growth_optimal_risk``); and every admission gate plus the
    reliability gate are re-verified on the final scaled ledger -- the measured
    numbers are reported exactly as they land, never assumed to pass.
    """
    end = resolve_evaluation_end(None, unseal_holdout=unseal_holdout)
    execution_spec = XsCompositeSpec()
    alpha_spec = XsAlphaCompositeSpec()

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        frame, funding, _fingerprint, _coverage = _load_symbol_data(
            symbol, XS_DISCOVERY_START, end,
        )
        data[symbol] = (frame, funding)

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        raise DataIntegrityError("xs baseline blend requires at least 2 common bars")

    opens = pd.DataFrame(
        {symbol: frame["open"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    bar_funding = pd.DataFrame(
        {
            symbol: _bar_funding_series(funding, frame.index).reindex(common)
            for symbol, (frame, funding) in data.items()
        },
    )
    closes = pd.DataFrame(
        {symbol: frame["close"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    taker = pd.DataFrame(
        {symbol: frame["taker_buy_ratio"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )

    opens_arr = opens.to_numpy(dtype=np.float64)
    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0
    benchmark = pd.Series(o2o.mean(axis=1), index=common, name="benchmark")

    weights = build_xs_alpha_vol_weighted_weights(
        closes, taker, bar_funding, opens, alpha_spec, execution_spec,
    )
    xs_equity, _xs_turnover = run_xs_composite_ledger(
        weights, opens, bar_funding, execution_spec,
    )
    xs_alpha_net = xs_equity.pct_change().fillna(0.0).rename("xs_alpha_net")

    btc_frame, btc_funding = data["BTCUSDT"]
    baseline_result = run_backtest(
        btc_frame, StrategySpec(), CostModel(), funding_rates=btc_funding,
    )
    baseline_equity = baseline_result.equity.reindex(common).rename("baseline_equity")
    baseline_net = baseline_equity.pct_change().fillna(0.0).rename("baseline_net")
    baseline_realized_weight = _baseline_realized_position(
        baseline_result.trades, btc_frame.index, common,
    )

    xs_realized_weights = weights.shift(1 + execution_spec.execution_delay_bars).fillna(0.0)
    blended_equity, blended_weights = build_blended_ledger(
        xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight,
        xs_alpha_weight,
    )
    blended_net = blended_equity.pct_change().fillna(0.0).rename("blended_net")
    scaled_net, scaled_weights = apply_fixed_gross_leverage(
        blended_net, blended_weights, leverage_scale,
    )
    scaled_equity = pd.Series(
        float(blended_equity.iloc[0]) * np.cumprod(
            1.0 + scaled_net.to_numpy(dtype=np.float64),
        ),
        index=blended_equity.index,
        name="scaled_equity",
    )
    scaled_turnover = _realised_turnover(scaled_weights)

    admission_config = XsAdmissionConfig()
    discovery = evaluate_xs_admission(
        _window_series(scaled_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(scaled_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    qualification = evaluate_xs_admission(
        _window_series(scaled_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(scaled_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    scaled_turnover_qualification = _window_series(
        scaled_turnover, QUALIFICATION_START, QUALIFICATION_END,
    )
    qualification_turnover_neutral = evaluate_xs_admission(
        _window_series(scaled_equity, QUALIFICATION_START, QUALIFICATION_END),
        scaled_turnover_qualification,
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        dataclasses.replace(admission_config, turnover_max=math.inf),
    )
    qualification_worst_fold_turnover = compute_turnover_fold_upper_bound(
        scaled_turnover_qualification,
        bars_per_year=GrowthSizingConfig(_DEFAULT_RISK_GRID).bars_per_year,
    )

    holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    if unseal_holdout:
        post_cutoff = common[common > HOLDOUT_CUTOFF]
        if len(post_cutoff) >= 2:
            holdout_start = post_cutoff[0]
            holdout_end = post_cutoff[-1]
            holdout = evaluate_xs_admission(
                _window_series(scaled_equity, holdout_start, holdout_end),
                _window_series(scaled_turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                admission_config,
            )

    xs_turnover = _realised_turnover(xs_realized_weights)
    pre_blend_discovery = evaluate_xs_admission(
        _window_series(xs_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(xs_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    pre_blend_qualification = evaluate_xs_admission(
        _window_series(xs_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(xs_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    pre_blend_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        pre_blend_holdout = evaluate_xs_admission(
            _window_series(xs_equity, holdout_start, holdout_end),
            _window_series(xs_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    baseline_turnover = _realised_turnover(
        pd.DataFrame({"baseline": baseline_realized_weight}, index=common),
    )
    baseline_discovery = evaluate_xs_admission(
        _window_series(baseline_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(baseline_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    baseline_qualification = evaluate_xs_admission(
        _window_series(baseline_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(baseline_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    baseline_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        baseline_holdout = evaluate_xs_admission(
            _window_series(baseline_equity, holdout_start, holdout_end),
            _window_series(baseline_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    reliability: XsReliabilityResult | None = None
    oos_window = _oos_reliability_window(
        scaled_equity, scaled_weights, QUALIFICATION_START, QUALIFICATION_END,
        holdout_start, holdout_end,
    )
    if oos_window is not None:
        oos_equity, oos_weights = oos_window
        reliability = evaluate_xs_reliability(
            oos_equity,
            oos_weights,
            dataclasses.replace(
                ReliabilityGateConfig(),
                hurdle_rate=derive_cost_multiple_hurdle_rate(
                    derive_realized_weights_cost_total(
                        oos_weights, admission_config.round_trip_cost_rate
                    ),
                    equity_span_years(oos_equity),
                    2.0,
                ),
            ),
        )

    return XsBaselineBlendJointReport(
        profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
        xs_alpha_weight=xs_alpha_weight,
        leverage_scale=leverage_scale,
        discovery=discovery,
        qualification=qualification,
        holdout=holdout,
        pre_blend_discovery=pre_blend_discovery,
        pre_blend_qualification=pre_blend_qualification,
        pre_blend_holdout=pre_blend_holdout,
        baseline_discovery=baseline_discovery,
        baseline_qualification=baseline_qualification,
        baseline_holdout=baseline_holdout,
        reliability=reliability,
        qualification_turnover_neutral=qualification_turnover_neutral,
        qualification_worst_fold_turnover=qualification_worst_fold_turnover,
    )


def run_xs_alpha_baseline_leg_selection(
    *,
    unseal_holdout: bool = False,
    weight_grid: tuple[float, ...] = _DEFAULT_WEIGHT_GRID,
    candidate_order: tuple[str, ...] = _DEFAULT_CANDIDATE_ORDER,
) -> XsBaselineLegSelectionReport:
    """Compare the five frozen candidates as v6's baseline diversifier leg.

    Fourth sibling to :func:`run_xs_alpha_baseline_blend` -- a new, additive
    comparison tool, never a substitute. Loads v6's XS ledger exactly as the
    existing sibling does, replays every candidate on ``BTCUSDT`` only via the
    tournament's frozen ``_run_source`` dispatch (same ``StrategySpec()``/
    ``CostModel()``/catalog parameters, single-symbol scope matching the
    existing baseline), derives each leg's net returns and realized position
    via the existing candidate-agnostic ``_baseline_realized_position``, builds
    the per-candidate diagnostics (discovery-window correlation, selected blend
    weight, achieved blended Sharpe), picks the winner by
    :func:`select_best_baseline_leg`, applies the winning blend over the full
    history, and re-verifies discovery/qualification/(sealed-unless-unsealed)
    holdout admission plus the OOS reliability gate through the exact same
    calls the existing sibling makes. The winner is reported -- never
    auto-promoted into any live entry point.
    """
    end = resolve_evaluation_end(None, unseal_holdout=unseal_holdout)
    execution_spec = XsCompositeSpec()
    alpha_spec = XsAlphaCompositeSpec()

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        frame, funding, _fingerprint, _coverage = _load_symbol_data(
            symbol, XS_DISCOVERY_START, end,
        )
        data[symbol] = (frame, funding)

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        raise DataIntegrityError("xs baseline blend requires at least 2 common bars")

    opens = pd.DataFrame(
        {symbol: frame["open"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    bar_funding = pd.DataFrame(
        {
            symbol: _bar_funding_series(funding, frame.index).reindex(common)
            for symbol, (frame, funding) in data.items()
        },
    )
    closes = pd.DataFrame(
        {symbol: frame["close"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    taker = pd.DataFrame(
        {symbol: frame["taker_buy_ratio"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )

    opens_arr = opens.to_numpy(dtype=np.float64)
    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0
    benchmark = pd.Series(o2o.mean(axis=1), index=common, name="benchmark")

    weights = build_xs_alpha_vol_weighted_weights(
        closes, taker, bar_funding, opens, alpha_spec, execution_spec,
    )
    xs_equity, _xs_turnover = run_xs_composite_ledger(
        weights, opens, bar_funding, execution_spec,
    )
    xs_alpha_net = xs_equity.pct_change().fillna(0.0).rename("xs_alpha_net")

    btc_frame, btc_funding = data["BTCUSDT"]
    candidate_nets: dict[str, pd.Series] = {}
    candidate_realized_weights: dict[str, pd.Series] = {}
    for candidate_id in candidate_order:
        if candidate_id in TOURNAMENT_RETURN_SOURCES:
            candidate_result = _run_source(
                candidate_id, "BTCUSDT", btc_frame, btc_funding, CostModel(),
                signal_delay_bars=0,
            )
        else:
            candidate = _TREND_SCREEN_CANDIDATES_BY_ID[candidate_id]
            candidate_result = run_technical_expert_backtest(
                btc_frame, candidate, CostModel(), btc_funding,
                signal_delay_bars=0,
                stop_loss_mode="atr_multiple",
                stop_loss_value=2.0,
                atr_period=14,
            )
        candidate_equity = candidate_result.equity.reindex(common).rename(
            f"{candidate_id}_equity",
        )
        candidate_nets[candidate_id] = (
            candidate_equity.pct_change().fillna(0.0).rename(f"{candidate_id}_net")
        )
        candidate_realized_weights[candidate_id] = _baseline_realized_position(
            candidate_result.trades, btc_frame.index, common,
        )

    candidate_diagnostics: dict[str, dict[str, float]] = {}
    for candidate_id in candidate_order:
        net = candidate_nets[candidate_id]
        correlation = compute_discovery_correlation(
            xs_alpha_net, net, XS_DISCOVERY_START, DISCOVERY_END,
        )
        blend_weight = select_baseline_blend_weight(
            xs_alpha_net, net, XS_DISCOVERY_START, DISCOVERY_END, weight_grid,
        )
        disc_a, disc_b = _discovery_common(
            xs_alpha_net, net, XS_DISCOVERY_START, DISCOVERY_END,
        )
        blended_sharpe = _blended_sharpe(disc_a, disc_b, blend_weight)
        candidate_diagnostics[candidate_id] = {
            "correlation": correlation,
            "blend_weight": blend_weight,
            "blended_sharpe": blended_sharpe,
        }

    selected_candidate, blend_weight = select_best_baseline_leg(
        xs_alpha_net, candidate_nets, XS_DISCOVERY_START, DISCOVERY_END,
        candidate_order, weight_grid,
    )

    baseline_net = candidate_nets[selected_candidate]
    baseline_realized_weight = candidate_realized_weights[selected_candidate]

    xs_realized_weights = weights.shift(1 + execution_spec.execution_delay_bars).fillna(0.0)
    blended_equity, blended_weights = build_blended_ledger(
        xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight,
        blend_weight,
    )
    blended_turnover = _realised_turnover(blended_weights)
    admission_config = XsAdmissionConfig()

    discovery = evaluate_xs_admission(
        _window_series(blended_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(blended_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    qualification = evaluate_xs_admission(
        _window_series(blended_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(blended_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )

    holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    if unseal_holdout:
        post_cutoff = common[common > HOLDOUT_CUTOFF]
        if len(post_cutoff) >= 2:
            holdout_start = post_cutoff[0]
            holdout_end = post_cutoff[-1]
            holdout = evaluate_xs_admission(
                _window_series(blended_equity, holdout_start, holdout_end),
                _window_series(blended_turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                admission_config,
            )

    xs_turnover = _realised_turnover(xs_realized_weights)
    pre_blend_discovery = evaluate_xs_admission(
        _window_series(xs_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(xs_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    pre_blend_qualification = evaluate_xs_admission(
        _window_series(xs_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(xs_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )
    pre_blend_holdout: XsAdmissionResult | None = None
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        pre_blend_holdout = evaluate_xs_admission(
            _window_series(xs_equity, holdout_start, holdout_end),
            _window_series(xs_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    reliability: XsReliabilityResult | None = None
    reliability_hurdle_neutral: XsReliabilityResult | None = None
    reliability_point_to_lcb90_ratio: float | None = None
    reliability_block_size_hit_cap: bool | None = None
    oos_window = _oos_reliability_window(
        blended_equity, blended_weights, QUALIFICATION_START, QUALIFICATION_END,
        holdout_start, holdout_end,
    )
    if oos_window is not None:
        oos_equity, oos_weights = oos_window
        reliability = evaluate_xs_reliability(
            oos_equity,
            oos_weights,
            dataclasses.replace(
                ReliabilityGateConfig(),
                hurdle_rate=derive_cost_multiple_hurdle_rate(
                    derive_realized_weights_cost_total(
                        oos_weights, admission_config.round_trip_cost_rate
                    ),
                    equity_span_years(oos_equity),
                    2.0,
                ),
            ),
        )
        reliability_hurdle_neutral = evaluate_xs_reliability(
            oos_equity, oos_weights,
            dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
        )
        reliability_point_to_lcb90_ratio = (
            reliability.lcb.point_cagr / reliability.lcb.lcb90_cagr
            if reliability.lcb.lcb90_cagr != 0.0 else None
        )
        reliability_block_size_hit_cap = block_size_search_hit_cap(
            oos_equity.pct_change().dropna().to_numpy(dtype=np.float64),
        )

    return XsBaselineLegSelectionReport(
        profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
        blend_weight=blend_weight,
        weight_grid=weight_grid,
        discovery=discovery,
        qualification=qualification,
        holdout=holdout,
        pre_blend_discovery=pre_blend_discovery,
        pre_blend_qualification=pre_blend_qualification,
        pre_blend_holdout=pre_blend_holdout,
        reliability=reliability,
        reliability_hurdle_neutral=reliability_hurdle_neutral,
        reliability_point_to_lcb90_ratio=reliability_point_to_lcb90_ratio,
        reliability_block_size_hit_cap=reliability_block_size_hit_cap,
        selected_candidate=selected_candidate,
        candidate_diagnostics=candidate_diagnostics,
    )


def persist_xs_alpha_baseline_blend_report(report: XsBaselineBlendReport, path: Path) -> None:
    """Upsert into the consolidated pass/fail ledger, keyed by ``path.stem``."""
    persist_reliability_ledger_entry(path.stem, report.to_payload(), path.parent)


def xs_baseline_blend_report_path() -> Path:
    """Logical report key for the v8 blend report (ledger entry name, not a literal write target)."""
    return Path("docs/results") / f"{_BLEND_REPORT_NAME}.json"

def persist_xs_alpha_baseline_blend_sized_report(
    report: XsBaselineBlendSizedReport, path: Path,
) -> None:
    """Upsert into the consolidated pass/fail ledger, keyed by ``path.stem``."""
    persist_reliability_ledger_entry(path.stem, report.to_payload(), path.parent)


def xs_baseline_blend_sized_report_path() -> Path:
    """Logical report key for the robust-blend + sizing report (ledger entry name, not a literal write target)."""
    return Path("docs/results") / f"{_BLEND_REPORT_NAME}_sized.json"

def persist_xs_alpha_baseline_blend_joint_report(
    report: XsBaselineBlendJointReport, path: Path,
) -> None:
    """Upsert into the consolidated pass/fail ledger, keyed by ``path.stem``."""
    persist_reliability_ledger_entry(path.stem, report.to_payload(), path.parent)


def xs_baseline_blend_joint_report_path() -> Path:
    """Logical report key for the joint-searched blend report (ledger entry name, not a literal write target)."""
    return Path("docs/results") / f"{_BLEND_REPORT_NAME}_joint.json"


def persist_xs_alpha_baseline_leg_selection_report(
    report: XsBaselineLegSelectionReport, path: Path,
) -> None:
    """Upsert into the consolidated pass/fail ledger, keyed by ``path.stem``."""
    persist_reliability_ledger_entry(path.stem, report.to_payload(), path.parent)


def xs_baseline_leg_selection_report_path() -> Path:
    """Logical report key for the baseline-leg comparison report (ledger entry name, not a literal write target)."""
    return Path("docs/results") / "xs_alpha_baseline_leg_selection.json"


def _check_contract() -> None:
    """Executable assertions locking the baseline-blend entry-point surface."""
    from inspect import signature

    params = signature(run_xs_alpha_baseline_blend).parameters
    assert set(params) == {"profile", "unseal_holdout", "weight_grid"}
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
    sized_params = signature(run_xs_alpha_baseline_blend_sized).parameters
    assert set(sized_params) == {"unseal_holdout", "weight_grid", "risk_grid"}
    assert all(p.kind == p.KEYWORD_ONLY for p in sized_params.values())
    assert sized_params["risk_grid"].default == (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert XS_VOL_WEIGHTED_ALPHA_PROFILE_ID == "xs_alpha_vol_weighted_v6"
    assert _DEFAULT_WEIGHT_GRID == (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
    assert xs_baseline_blend_report_path().name == "xs_alpha_baseline_blend_v8.json"
    assert xs_baseline_blend_sized_report_path().name == (
        "xs_alpha_baseline_blend_v8_sized.json"
    )
    joint_params = signature(run_xs_alpha_baseline_blend_joint).parameters
    assert set(joint_params) == {"xs_alpha_weight", "leverage_scale", "unseal_holdout"}
    assert all(p.kind == p.KEYWORD_ONLY for p in joint_params.values())
    assert joint_params["xs_alpha_weight"].default is joint_params["xs_alpha_weight"].empty
    assert joint_params["leverage_scale"].default is joint_params["leverage_scale"].empty
    objective_params = signature(discovery_reliability_score).parameters
    assert set(objective_params) == {
        "xs_alpha_net", "xs_alpha_realized_weights", "baseline_net",
        "baseline_realized_weight", "discovery_start", "discovery_end",
        "xs_alpha_weight", "leverage_scale", "round_trip_cost_rate",
    }
    assert 0.0 <= _JOINT_XS_ALPHA_WEIGHT <= 1.0
    assert _JOINT_LEVERAGE_SCALE > 0.0
    assert xs_baseline_blend_joint_report_path().name == (
        "xs_alpha_baseline_blend_v8_joint.json"
    )
    leg_selection_params = signature(run_xs_alpha_baseline_leg_selection).parameters
    assert set(leg_selection_params) == {"unseal_holdout", "weight_grid", "candidate_order"}
    assert all(p.kind == p.KEYWORD_ONLY for p in leg_selection_params.values())
    assert _DEFAULT_CANDIDATE_ORDER[0] == "donchian_long_only_v1"
    assert xs_baseline_leg_selection_report_path().name == (
        "xs_alpha_baseline_leg_selection.json"
    )


_check_contract()
