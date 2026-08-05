"""Growth-optimal gross-leverage sizing overlay for admitted XS alpha profiles.

``xs_alpha_vol_weighted_v6`` is the strongest fully-admitted XS book; every
admission gate is scale-invariant, so the current unit-gross construction is a
size chosen by convention (``sum(abs(w)) == 1``), not by compounding. This
module wires the existing constraint-first growth-sizing primitives
(:mod:`src.research.risk.growth_sizing`) onto the profile's own realized
returns: the gross-leverage multiplier is selected strictly from the discovery
window, the drawdown-protected realised-risk overlay is applied over the full
history, and every admission gate is re-verified post-scaling. Only profiles
admitted end-to-end (v6 and v2) are accepted; sizing an alpha construction that
never proved a discovery/qualification edge is out of scope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.research.technical.trend_screen import _load_symbol_data
from src.application.research.technical.xs_trend_screen import (
    XS_ALPHA_PROFILE_ID,
    XS_DISCOVERY_START,
    XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    _admission_payload,
    _bar_funding_series,
    _common_index,
    _fingerprint_without_self,
    _window_series,
)
from src.common.errors import DataIntegrityError
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.risk.growth_sizing import (
    GrowthSizingConfig,
    GrowthSizingResult,
    compute_discovery_target_vol,
)
from src.research.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsAdmissionResult,
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    build_xs_alpha_vol_weighted_weights,
    build_xs_alpha_weights,
    evaluate_xs_admission,
    run_xs_composite_ledger,
    select_vol_target_window,
    size_xs_alpha_growth_optimal,
)
from src.research.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    QUALIFICATION_END,
    QUALIFICATION_START,
    TREND_SCREEN_SYMBOLS,
)

# Gross-leverage grid. `reference_risk = 1.0` reuses the existing unit-gross
# construction invariant (build_xs_neutral_weights guarantees pre-band
# sum(abs(w)) == 1) as the bookkeeping anchor, so grid values are literally
# gross-leverage multiples. The upper bound 6.0 is the turnover-gate-implied
# ceiling (v6's realized turnover is ~22.7-25.8x/year against the frozen
# 150.0/year gate). All other fields are the module's own frozen defaults.
_XS_GROWTH_SIZING_CONFIG = GrowthSizingConfig(
    risk_grid=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    reference_risk=1.0,
)

# Default vol-target window grid: v6's own already-frozen family inverse-vol
# tilt horizons (XsAlphaCompositeSpec().signal_windows). Passing a tuple makes
# run_xs_alpha_growth_sizing search the grid for the growth-optimal window
# (select_vol_target_window); an explicit int or None bypasses the search and
# reproduces the prior fixed-window / scalar-only behaviors byte-for-byte.


def _sizing_payload(result: GrowthSizingResult) -> dict[str, object]:
    return {
        "selected_risk": result.selected_risk,
        "median_log_growth": round(result.median_log_growth, 8),
        "mdd_breach_prob": round(result.mdd_breach_prob, 8),
        "ruin_prob": round(result.ruin_prob, 8),
        "feasible_risks": list(result.feasible_risks),
        "binding_constraint": result.binding_constraint,
        "block_size_used": result.block_size_used,
    }


@dataclass(frozen=True, slots=True)
class XsGrowthSizingReport:
    """Deterministic persisted outcome of one growth-optimal sizing run."""

    profile: str
    sizing: GrowthSizingResult
    discovery: XsAdmissionResult
    qualification: XsAdmissionResult
    holdout: XsAdmissionResult | None
    pre_scaling_discovery: XsAdmissionResult
    pre_scaling_qualification: XsAdmissionResult
    pre_scaling_holdout: XsAdmissionResult | None
    vol_target_window: int | None = None
    vol_target: float | None = None
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "report_fingerprint", _fingerprint_without_self(self._body_payload()),
        )

    def _body_payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "sizing": _sizing_payload(self.sizing),
            "discovery": _admission_payload(self.discovery),
            "qualification": _admission_payload(self.qualification),
            "holdout": _admission_payload(self.holdout) if self.holdout is not None else None,
            "pre_scaling_discovery": _admission_payload(self.pre_scaling_discovery),
            "pre_scaling_qualification": _admission_payload(self.pre_scaling_qualification),
            "pre_scaling_holdout": (
                _admission_payload(self.pre_scaling_holdout)
                if self.pre_scaling_holdout is not None
                else None
            ),
            "vol_target_window": self.vol_target_window,
            "vol_target": self.vol_target,
        }

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload = self._body_payload()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"


def _realised_turnover(realized_weights: pd.DataFrame) -> pd.Series:
    """Per-bar turnover of a realized-weight book, matching the ledger's sum-of-abs-changes."""
    values = realized_weights.to_numpy(dtype=np.float64)
    prev = np.zeros_like(values)
    prev[1:] = values[:-1]
    return pd.Series(
        np.abs(values - prev).sum(axis=1), index=realized_weights.index,
        name="turnover",
    )


def run_xs_alpha_growth_sizing(
    *,
    profile: str = XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    unseal_holdout: bool = False,
    vol_target_window: int | None | tuple[int, ...] = XsAlphaCompositeSpec().signal_windows,
) -> XsGrowthSizingReport:
    """Execute the growth-optimal gross-leverage sizing overlay for one profile.

    Only ``xs_alpha_vol_weighted_v6`` and ``xs_alpha_multihorizon_v2`` (the two
    profiles admitted end-to-end) are accepted. The base book is replayed under
    the frozen execution spec, pre-scaling admission is recorded, the overlay
    size is chosen strictly from discovery, and every admission gate is
    re-evaluated on the scaled equity/turnover -- a genuine post-scaling check,
    since turnover, breakeven cost, and realized MDD are not scale-invariant
    once the path-dependent drawdown ladder is in the loop. Holdout stays sealed
    unless ``unseal_holdout`` is set.

    The proactive vol-target overlay (``compute_discovery_target_vol`` +
    ``apply_vol_target_overlay`` in :mod:`src.research.risk.growth_sizing`) is
    enabled by default; when ``vol_target_window`` is a tuple (the frozen
    ``XsAlphaCompositeSpec().signal_windows`` default), the growth-optimal
    window is selected among it via :func:`select_vol_target_window`, and the
    resolved anchor is reported as ``vol_target``. Passing an explicit int
    bypasses the search (byte-for-byte reproduces ADR_20260805's fixed-``42``
    path for ``42``), and passing ``None`` disables vol-targeting entirely
    (byte-for-byte reproduces the ADR_20260805 original scalar-only path).
    """
    if profile not in (XS_ALPHA_PROFILE_ID, XS_VOL_WEIGHTED_ALPHA_PROFILE_ID):
        raise ValueError(
            f"growth sizing is restricted to profiles admitted end-to-end; "
            f"'{profile}' is not among '{XS_ALPHA_PROFILE_ID}', "
            f"'{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}'"
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
        raise DataIntegrityError("xs growth sizing requires at least 2 common bars")

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

    if profile == XS_VOL_WEIGHTED_ALPHA_PROFILE_ID:
        weights = build_xs_alpha_vol_weighted_weights(
            closes, taker, bar_funding, opens, alpha_spec, execution_spec,
        )
    else:
        weights = build_xs_alpha_weights(
            closes, taker, bar_funding, alpha_spec, execution_spec,
        )

    base_equity, base_turnover = run_xs_composite_ledger(
        weights, opens, bar_funding, execution_spec,
    )
    admission_config = XsAdmissionConfig()
    pre_scaling_discovery = evaluate_xs_admission(
        _window_series(base_equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(base_turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        admission_config,
    )
    pre_scaling_qualification = evaluate_xs_admission(
        _window_series(base_equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(base_turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        admission_config,
    )

    pre_scaling_holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    if unseal_holdout:
        post_cutoff = common[common > HOLDOUT_CUTOFF]
        if len(post_cutoff) >= 2:
            holdout_start = post_cutoff[0]
            holdout_end = post_cutoff[-1]
            pre_scaling_holdout = evaluate_xs_admission(
                _window_series(base_equity, holdout_start, holdout_end),
                _window_series(base_turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                admission_config,
            )

    if isinstance(vol_target_window, tuple):
        scaled_net, scaled_weights, sizing, resolved_window = select_vol_target_window(
            weights, opens, bar_funding, execution_spec,
            XS_DISCOVERY_START, DISCOVERY_END, _XS_GROWTH_SIZING_CONFIG,
            window_grid=vol_target_window,
        )
    else:
        resolved_window = vol_target_window
        scaled_net, scaled_weights, sizing = size_xs_alpha_growth_optimal(
            weights, opens, bar_funding, execution_spec,
            XS_DISCOVERY_START, DISCOVERY_END, _XS_GROWTH_SIZING_CONFIG,
            vol_target_window=vol_target_window,
        )
    if resolved_window is not None:
        discovery_net = base_equity.pct_change().dropna()
        discovery_net = discovery_net[
            (discovery_net.index >= XS_DISCOVERY_START) & (discovery_net.index <= DISCOVERY_END)
        ]
        vol_target = compute_discovery_target_vol(discovery_net, resolved_window)
    else:
        vol_target = None

    if sizing.selected_risk is None:
        scaled_equity = base_equity
        scaled_turnover = base_turnover
    else:
        scaled_equity = pd.Series(
            float(base_equity.iloc[0]) * np.cumprod(
                1.0 + scaled_net.reindex(base_equity.index).fillna(0.0).to_numpy(dtype=np.float64),
            ),
            index=base_equity.index,
            name="scaled_equity",
        )
        scaled_turnover = _realised_turnover(scaled_weights)

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
    if unseal_holdout and holdout_start is not None and holdout_end is not None:
        holdout = evaluate_xs_admission(
            _window_series(scaled_equity, holdout_start, holdout_end),
            _window_series(scaled_turnover, holdout_start, holdout_end),
            _window_series(benchmark, holdout_start, holdout_end),
            admission_config,
        )

    return XsGrowthSizingReport(
        profile=profile,
        sizing=sizing,
        discovery=discovery,
        qualification=qualification,
        holdout=holdout,
        pre_scaling_discovery=pre_scaling_discovery,
        pre_scaling_qualification=pre_scaling_qualification,
        pre_scaling_holdout=pre_scaling_holdout,
        vol_target_window=resolved_window,
        vol_target=vol_target,
    )


def persist_xs_growth_sizing_report(report: XsGrowthSizingReport, path: Path) -> None:
    """Write the byte-deterministic report payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")


def xs_growth_sizing_report_path(profile: str = XS_VOL_WEIGHTED_ALPHA_PROFILE_ID) -> Path:
    """Default persistence location for one growth-sized XS profile."""
    return Path("docs/results") / f"{profile}_growth_sized.json"


def _check_contract() -> None:
    """Executable assertions locking the growth-sizing entry-point surface."""
    from inspect import signature

    params = signature(run_xs_alpha_growth_sizing).parameters
    assert set(params) == {"profile", "unseal_holdout", "vol_target_window"}
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
    assert params["vol_target_window"].default == (42, 84, 168)
    assert params["vol_target_window"].default == XsAlphaCompositeSpec().signal_windows
    assert XS_ALPHA_PROFILE_ID == "xs_alpha_multihorizon_v2"
    assert XS_VOL_WEIGHTED_ALPHA_PROFILE_ID == "xs_alpha_vol_weighted_v6"
    assert _XS_GROWTH_SIZING_CONFIG.reference_risk == 1.0
    assert _XS_GROWTH_SIZING_CONFIG.risk_grid == (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert _XS_GROWTH_SIZING_CONFIG.max_drawdown == 0.20
    assert _XS_GROWTH_SIZING_CONFIG.horizon_years == 5.0
    assert _XS_GROWTH_SIZING_CONFIG.bars_per_year == 2190
    assert xs_growth_sizing_report_path(XS_VOL_WEIGHTED_ALPHA_PROFILE_ID).name == (
        "xs_alpha_vol_weighted_v6_growth_sized.json"
    )


_check_contract()
