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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.research.technical.trend_screen import _load_symbol_data
from src.application.research.technical.xs_alpha_growth_sizing import _realised_turnover
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
from src.research.evaluation.reliability import ReliabilityGateConfig
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
    TREND_SCREEN_SYMBOLS,
)
from src.research.technical_experts.xs_alpha_baseline_blend import (
    build_blended_ledger,
    select_baseline_blend_weight,
)

# Pre-registered, frozen weight grid (v8): selection is argmax annualized
# Sharpe on the discovery window only. A coarse grid is deliberate -- the
# whole point is a small, bounded, deterministic search, not matrix inversion.
_DEFAULT_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

# Frozen persistence name (v8 of the XS alpha family), mirroring
# ``xs_growth_sizing_report_path``'s naming convention.
_BLEND_REPORT_NAME = "xs_alpha_baseline_blend_v8"


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
            oos_equity, oos_weights, ReliabilityGateConfig(),
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


def persist_xs_alpha_baseline_blend_report(report: XsBaselineBlendReport, path: Path) -> None:
    """Write the byte-deterministic report payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")


def xs_baseline_blend_report_path() -> Path:
    """Default persistence location for the v8 blend report."""
    return Path("docs/results") / f"{_BLEND_REPORT_NAME}.json"


def _check_contract() -> None:
    """Executable assertions locking the baseline-blend entry-point surface."""
    from inspect import signature

    params = signature(run_xs_alpha_baseline_blend).parameters
    assert set(params) == {"profile", "unseal_holdout", "weight_grid"}
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
    assert XS_VOL_WEIGHTED_ALPHA_PROFILE_ID == "xs_alpha_vol_weighted_v6"
    assert _DEFAULT_WEIGHT_GRID == (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
    assert xs_baseline_blend_report_path().name == "xs_alpha_baseline_blend_v8.json"


_check_contract()
