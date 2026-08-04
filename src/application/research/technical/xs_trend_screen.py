"""Cross-sectional dollar-neutral composite trend screen profile.

``xs_neutral_composite_v1`` is the additive companion to the 450-cell
baseline-gate screen. It rebuilds the same 30 frozen identities as one
per-symbol composite score (mean over the 15 families of LONG target + SHORT
target), EWMA-smoothes and cross-sectionally demeans it into a unit-gross
dollar-neutral book with a no-trade band, and compounds it under the exact
production execution convention (t+1+delay open fills, turnover costs, funding
on the held book). Discovery and qualification are admitted by the scale-
invariant, structure-only gates of ``evaluate_xs_admission`` against the
equal-weight universe open-to-open return as the benchmark.

The screen reuses ``TREND_SCREEN_CANDIDATES``, ``TREND_SCREEN_SYMBOLS``, and the
shared sealed-holdout policy and never registers a production candidate: it is
research evidence only. Construction is fully causal (EWMA, cross-sectional
normalization, and the no-trade band all read only bars at or before the
current one), so loading data past the sealed cutoff never changes discovery
or qualification. The holdout window (every common bar strictly after
``HOLDOUT_CUTOFF``) is evaluated only when ``unseal_holdout`` is set and is
``None`` otherwise; this is intended as a single post-implementation
confirmation, not a re-tunable input.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.research.technical.trend_screen import _load_symbol_data
from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import _align_funding_rates
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.expert_portfolio.contextual_router import (
    build_causal_context_labels,
    state_labels,
)
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsAdmissionResult,
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    build_xs_alpha_family_weights,
    build_xs_alpha_weights,
    build_xs_neutral_weights,
    evaluate_xs_admission,
    run_xs_composite_ledger,
)
from src.research.technical_experts.signals import generate_signal_events
from src.research.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    QUALIFICATION_END,
    QUALIFICATION_START,
    TREND_SCREEN_CANDIDATES,
    TREND_SCREEN_FAMILIES,
    TREND_SCREEN_SYMBOLS,
)
from src.research.technical_experts.xs_contextual_router import (
    XsContextualAllocation,
    build_xs_causal_contextual_allocation,
    build_xs_context_market,
)

XS_NEUTRAL_PROFILE_ID = "xs_neutral_composite_v1"
XS_ALPHA_PROFILE_ID = "xs_alpha_multihorizon_v2"
XS_CONTEXTUAL_ALPHA_PROFILE_ID = "xs_alpha_contextual_v3"
# The four-symbol earliest-history gap is not recoverable before 2022-04-03.
# Keep the baseline catalog window unchanged, but start XS panel evaluation on
# the first timestamp with complete taker/quote fields across the universe.
XS_DISCOVERY_START = pd.Timestamp("2022-04-03", tz="UTC")

_ALPHA_FAMILY_ORDER = ("trend", "funding_contrarian", "taker_imbalance")

__all__ = [
    "XS_ALPHA_PROFILE_ID",
    "XS_CONTEXTUAL_ALPHA_PROFILE_ID",
    "XS_NEUTRAL_PROFILE_ID",
    "XsTrendScreenReport",
    "run_xs_trend_screen",
]


@dataclass(frozen=True, slots=True)
class XsTrendScreenReport:
    """Deterministic persisted outcome of one sealed XS composite profile."""

    profile: str
    universe: tuple[str, ...]
    spec: XsCompositeSpec
    discovery: XsAdmissionResult
    qualification: XsAdmissionResult
    symbols: dict[str, dict[str, object]]
    holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    alpha_spec: XsAlphaCompositeSpec | None = None
    stress_spec: XsCompositeSpec | None = None
    stress_discovery: XsAdmissionResult | None = None
    stress_qualification: XsAdmissionResult | None = None
    stress_holdout: XsAdmissionResult | None = None
    router_spec: dict[str, object] | None = None
    router_diagnostics: dict[str, object] | None = None
    family_admission: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload: dict[str, object] = {
            "profile": self.profile,
            "universe": list(self.universe),
            "discovery_start": XS_DISCOVERY_START.isoformat(),
            "discovery_end": DISCOVERY_END.isoformat(),
            "qualification_start": QUALIFICATION_START.isoformat(),
            "qualification_end": QUALIFICATION_END.isoformat(),
            "spec": {
                "halflife_bars": self.spec.halflife_bars,
                "no_trade_band": self.spec.no_trade_band,
                "execution_delay_bars": self.spec.execution_delay_bars,
                "fee_rate": self.spec.fee_rate,
                "slippage_rate": self.spec.slippage_rate,
                "round_trip_cost_rate": round(self.spec.round_trip_cost_rate(), 8),
            },
            "discovery": _admission_payload(self.discovery),
            "qualification": _admission_payload(self.qualification),
            "holdout_start": self.holdout_start.isoformat() if self.holdout_start is not None else None,
            "holdout_end": self.holdout_end.isoformat() if self.holdout_end is not None else None,
            "holdout": _admission_payload(self.holdout) if self.holdout is not None else None,
            "symbols": self.symbols,
        }
        if self.alpha_spec is not None:
            payload["alpha_spec"] = {
                "components": list(self.alpha_spec.components),
                "signal_windows": list(self.alpha_spec.signal_windows),
            }
        if self.stress_spec is not None:
            payload["stress_spec"] = {
                "halflife_bars": self.stress_spec.halflife_bars,
                "no_trade_band": self.stress_spec.no_trade_band,
                "execution_delay_bars": self.stress_spec.execution_delay_bars,
                "fee_rate": self.stress_spec.fee_rate,
                "slippage_rate": self.stress_spec.slippage_rate,
                "round_trip_cost_rate": round(self.stress_spec.round_trip_cost_rate(), 8),
            }
            stress_payload: dict[str, object] = {
                "discovery": (
                    _admission_payload(self.stress_discovery)
                    if self.stress_discovery is not None else None
                ),
                "qualification": (
                    _admission_payload(self.stress_qualification)
                    if self.stress_qualification is not None else None
                ),
            }
            if self.stress_holdout is not None:
                stress_payload["holdout"] = _admission_payload(self.stress_holdout)
            payload["stress"] = stress_payload
        if self.router_spec is not None:
            payload["router_spec"] = self.router_spec
        if self.router_diagnostics is not None:
            payload["router_diagnostics"] = self.router_diagnostics
        if self.family_admission is not None:
            payload["family_admission"] = self.family_admission
        payload["report_fingerprint"] = _fingerprint_without_self(payload)
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"


def _admission_payload(result: XsAdmissionResult) -> dict[str, object]:
    return {
        "admitted": result.admitted,
        "binding_constraint": result.binding_constraint,
        "sharpe": round(result.sharpe, 8),
        "beta": round(result.beta, 8),
        "cagr": round(result.cagr, 8),
        "mdd": round(result.mdd, 8),
        "t_stat": round(result.t_stat, 8),
        "annual_sharpe": {k: round(v, 8) for k, v in result.annual_sharpe.items()},
        "annualized_turnover": round(result.annualized_turnover, 8),
        "breakeven_cost": round(result.breakeven_cost, 8),
    }


def _fingerprint_without_self(payload: dict[str, object]) -> str:
    body = {k: v for k, v in payload.items() if k != "report_fingerprint"}
    encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _failed_result(constraint: str) -> XsAdmissionResult:
    return XsAdmissionResult(
        admitted=False,
        binding_constraint=constraint,
        sharpe=0.0,
        beta=0.0,
        cagr=0.0,
        mdd=0.0,
        t_stat=0.0,
        annual_sharpe={},
        annualized_turnover=0.0,
        breakeven_cost=0.0,
    )


def _candidate_by_family_side(family: str, side: str) -> TechnicalCandidate:
    for candidate in TREND_SCREEN_CANDIDATES:
        if candidate.family == family and candidate.side == side:
            return candidate
    raise ValueError(f"no {side} candidate for family '{family}'")


def _family_score(frame: pd.DataFrame, candidate: TechnicalCandidate) -> pd.Series:
    """Persistent signed target of one directional identity on the bar grid.

    Replicates the decision stream of ``run_technical_expert_backtest``: a long
    entry holds +1 until its long exit, a short entry holds -1 until its short
    exit, and the opposite side of the candidate is always flat.
    """
    events = generate_signal_events(frame, candidate)
    long_entry = events["long_entry"].to_numpy(dtype=bool)
    short_entry = events["short_entry"].to_numpy(dtype=bool)
    long_exit = events["long_exit"].to_numpy(dtype=bool)
    short_exit = events["short_exit"].to_numpy(dtype=bool)
    decisions = np.where(
        long_entry,
        1.0,
        np.where(
            short_entry,
            -1.0,
            np.where(long_exit | short_exit, 0.0, np.nan),
        ),
    )
    target = pd.Series(decisions, index=frame.index).ffill().fillna(0.0)
    return target.rename("target")


def _symbol_composite_score(frame: pd.DataFrame) -> pd.Series:
    """Mean over the 15 frozen families of (LONG target + SHORT target)."""
    total: np.ndarray | None = None
    for family in TREND_SCREEN_FAMILIES:
        long_target = _family_score(frame, _candidate_by_family_side(family, "LONG"))
        short_target = _family_score(frame, _candidate_by_family_side(family, "SHORT"))
        fam_score = (long_target + short_target).to_numpy(dtype=np.float64)
        total = fam_score if total is None else total + fam_score
    assert total is not None
    return pd.Series(total / len(TREND_SCREEN_FAMILIES), index=frame.index, name="composite")


def _common_index(indexes: list[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    common = reduce(pd.Index.intersection, indexes)
    return pd.DatetimeIndex(common)


def _bar_funding_series(funding: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(_align_funding_rates(funding, grid), index=grid, name="bar_funding")


def _window_series(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Inclusive ``[start, end]`` slice anchored to the mark before ``start``.

    Mirrors the screen convention: the boundary return into the window is
    included exactly once, so the first in-window return and the turnover into
    that bar are never dropped.
    """
    index = series.index
    prior = index[index < start]
    anchor = prior[-1] if len(prior) > 0 else None
    if anchor is None:
        return series[(index >= start) & (index <= end)]
    return series[(index >= anchor) & (index <= end)]


def _fail_closed_report(
    spec: XsCompositeSpec,
    constraint: str,
    *,
    profile: str = XS_NEUTRAL_PROFILE_ID,
    alpha_spec: XsAlphaCompositeSpec | None = None,
    stress_spec: XsCompositeSpec | None = None,
) -> XsTrendScreenReport:
    return XsTrendScreenReport(
        profile=profile,
        universe=TREND_SCREEN_SYMBOLS,
        spec=spec,
        discovery=_failed_result(constraint),
        qualification=_failed_result(constraint),
        symbols={},
        alpha_spec=alpha_spec,
        stress_spec=stress_spec,
    )


def _router_spec_payload(spec: ContextualRouterSpec | None) -> dict[str, object] | None:
    """Deterministic JSON-ready serialization of the frozen router spec."""
    if spec is None:
        return None
    return {
        "context_symbol": spec.context_symbol,
        "trend_lookback_bars": spec.trend_lookback_bars,
        "volatility_lookback_bars": spec.volatility_lookback_bars,
        "min_context_history_bars": spec.min_context_history_bars,
        "confidence": spec.confidence,
    }


def _selected_window(
    selected: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    """Inclusive ``[start, end]`` selection counts and fractions per sleeve."""
    rows = selected[(selected.index >= start) & (selected.index <= end)]
    keys = (*_ALPHA_FAMILY_ORDER, "CASH")
    counts = {key: int((rows == key).sum()) for key in keys}
    total = len(rows)
    fractions = {
        key: round(count / total, 8) if total else 0.0
        for key, count in counts.items()
    }
    return {"counts": counts, "fractions": fractions}


def _build_router_diagnostics(
    allocation: XsContextualAllocation,
    holdout_start: pd.Timestamp | None,
    holdout_end: pd.Timestamp | None,
) -> dict[str, object]:
    """Deterministic per-window and per-state router diagnostics."""
    selected = allocation.selected_sleeve
    windows: dict[str, object] = {
        "discovery": _selected_window(selected, XS_DISCOVERY_START, DISCOVERY_END),
        "qualification": _selected_window(selected, QUALIFICATION_START, QUALIFICATION_END),
    }
    if holdout_start is not None and holdout_end is not None:
        windows["holdout"] = _selected_window(selected, holdout_start, holdout_end)

    labels = allocation.decision_context
    n = len(labels)
    label_values = labels.to_numpy(dtype=object)
    states: dict[str, object] = {}
    for state in state_labels():
        state_rows = np.flatnonzero(label_values == state)
        completed = int((state_rows < n - 1).sum())
        last_lcb: dict[str, object] = {}
        for family in _ALPHA_FAMILY_ORDER:
            values = allocation.conditional_lcb[labels == state][family].dropna()
            last_lcb[family] = (
                round(float(values.iloc[-1]), 8) if not values.empty else None
            )
        states[state] = {"completed_samples": completed, "last_lcb": last_lcb}
    return {"windows": windows, "states": states}


def _build_family_admission(
    family_ledgers: dict[str, tuple[pd.Series, pd.Series]],
    benchmark: pd.Series,
) -> dict[str, object]:
    """Diagnostic-only standalone family admission, never the combined result."""
    out: dict[str, object] = {}
    for family, (equity, turnover) in family_ledgers.items():
        result = evaluate_xs_admission(equity, turnover, benchmark, XsAdmissionConfig())
        payload = _admission_payload(result)
        payload["diagnostic"] = True
        out[family] = payload
    return out


def run_xs_trend_screen(
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    unseal_holdout: bool = False,
    max_workers: int | None = None,
    profile: str = XS_NEUTRAL_PROFILE_ID,
) -> XsTrendScreenReport:
    """Execute one sealed XS composite screen profile.

    ``xs_neutral_composite_v1`` rebuilds the per-symbol composite score,
    EWMA-smoothed dollar-neutral book, and production execution ledger. The
    research-only ``xs_alpha_multihorizon_v2`` instead sources close, taker
    ratio, and causally aligned settled funding panels and builds the
    nine-component multi-horizon alpha book; its base and stress ledgers are
    both replayed from the same frozen target weights. Nothing is registered;
    ``end`` defaults to the sealed cutoff unless ``unseal_holdout`` is set.
    """
    if profile not in (
        XS_NEUTRAL_PROFILE_ID,
        XS_ALPHA_PROFILE_ID,
        XS_CONTEXTUAL_ALPHA_PROFILE_ID,
    ):
        raise ValueError(
            f"unknown xs screen profile '{profile}'; the source-controlled "
            f"profiles are '{XS_NEUTRAL_PROFILE_ID}', '{XS_ALPHA_PROFILE_ID}', "
            f"and '{XS_CONTEXTUAL_ALPHA_PROFILE_ID}'"
        )
    end = resolve_evaluation_end(end, unseal_holdout=unseal_holdout)
    requested_start = XS_DISCOVERY_START if start is None else pd.to_datetime(start, utc=True)
    effective_start = max(requested_start, XS_DISCOVERY_START)
    execution_spec = XsCompositeSpec()
    alpha_spec = (
        XsAlphaCompositeSpec()
        if profile in (XS_ALPHA_PROFILE_ID, XS_CONTEXTUAL_ALPHA_PROFILE_ID)
        else None
    )
    stress_spec = (
        dataclasses.replace(
            execution_spec,
            fee_rate=execution_spec.fee_rate * 1.5,
            slippage_rate=execution_spec.slippage_rate * 2.0,
            execution_delay_bars=execution_spec.execution_delay_bars + 1,
        )
        if alpha_spec is not None
        else None
    )

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    fingerprints: dict[str, dict[str, str]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        try:
            frame, funding, fingerprint, _coverage = _load_symbol_data(symbol, effective_start, end)
        except (DataIntegrityError, FileNotFoundError) as exc:
            return _fail_closed_report(
                execution_spec, f"symbol_unavailable:{symbol}:{type(exc).__name__}",
                profile=profile, alpha_spec=alpha_spec, stress_spec=stress_spec,
            )
        data[symbol] = (frame, funding)
        fingerprints[symbol] = fingerprint

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        return _fail_closed_report(
            execution_spec, "insufficient_common_grid",
            profile=profile, alpha_spec=alpha_spec, stress_spec=stress_spec,
        )

    opens = pd.DataFrame(
        {symbol: frame["open"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    bar_funding = pd.DataFrame(
        {
            symbol: _bar_funding_series(funding, frame.index).reindex(common)
            for symbol, (frame, funding) in data.items()
        },
    )

    opens_arr = opens.to_numpy(dtype=np.float64)
    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0
    benchmark = pd.Series(o2o.mean(axis=1), index=common, name="benchmark")

    score_frames: dict[str, pd.Series] | None = None
    router_spec: ContextualRouterSpec | None = None
    allocation: XsContextualAllocation | None = None
    family_ledgers: dict[str, tuple[pd.Series, pd.Series]] | None = None
    router_diagnostics: dict[str, object] | None = None
    family_admission: dict[str, object] | None = None
    if alpha_spec is not None:
        try:
            closes = pd.DataFrame(
                {
                    symbol: frame["close"].reindex(common)
                    for symbol, (frame, _funding) in data.items()
                },
            )
            taker = pd.DataFrame(
                {
                    symbol: frame["taker_buy_ratio"].reindex(common)
                    for symbol, (frame, _funding) in data.items()
                },
            )
            if profile == XS_CONTEXTUAL_ALPHA_PROFILE_ID:
                router_spec = ContextualRouterSpec(
                    context_symbol="XS_EQUAL_WEIGHT_MARKET",
                    trend_lookback_bars=42,
                    volatility_lookback_bars=42,
                    min_context_history_bars=168,
                    confidence=0.90,
                )
                family_weights = build_xs_alpha_family_weights(
                    closes, taker, bar_funding, alpha_spec, execution_spec,
                )
                family_ledgers = {}
                sleeve_returns: dict[str, pd.Series] = {}
                for family, family_w in family_weights.items():
                    equity, turnover = run_xs_composite_ledger(
                        family_w, opens, bar_funding, execution_spec,
                    )
                    family_ledgers[family] = (equity, turnover)
                    sleeve_returns[family] = equity.pct_change()
                sleeve_returns_frame = pd.DataFrame(sleeve_returns, index=common)
                market = build_xs_context_market(closes)
                labels = build_causal_context_labels(market, router_spec)
                allocation = build_xs_causal_contextual_allocation(
                    family_weights, sleeve_returns_frame, labels, router_spec,
                )
                weights = allocation.target_weights
            else:
                weights = build_xs_alpha_weights(
                    closes, taker, bar_funding, alpha_spec, execution_spec,
                )
        except (DataIntegrityError, ValueError) as exc:
            constraint = (
                f"contextual_router_invalid:{type(exc).__name__}"
                if profile == XS_CONTEXTUAL_ALPHA_PROFILE_ID
                else f"alpha_panel_invalid:{type(exc).__name__}"
            )
            return _fail_closed_report(
                execution_spec, constraint,
                profile=profile, alpha_spec=alpha_spec, stress_spec=stress_spec,
            )
    else:
        score_frames = {
            symbol: _symbol_composite_score(frame)
            for symbol, (frame, _funding) in data.items()
        }
        score = pd.DataFrame(
            {symbol: sf.reindex(common) for symbol, sf in score_frames.items()},
        )
        weights = build_xs_neutral_weights(
            score, execution_spec.halflife_bars, execution_spec.no_trade_band,
        )

    equity, turnover = run_xs_composite_ledger(weights, opens, bar_funding, execution_spec)

    discovery = evaluate_xs_admission(
        _window_series(equity, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(turnover, XS_DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
        XsAdmissionConfig(),
    )
    qualification = evaluate_xs_admission(
        _window_series(equity, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(turnover, QUALIFICATION_START, QUALIFICATION_END),
        _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
        XsAdmissionConfig(),
    )

    holdout: XsAdmissionResult | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    if unseal_holdout:
        post_cutoff = common[common > HOLDOUT_CUTOFF]
        if len(post_cutoff) >= 2:
            holdout_start, holdout_end = post_cutoff[0], post_cutoff[-1]
            holdout = evaluate_xs_admission(
                _window_series(equity, holdout_start, holdout_end),
                _window_series(turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                XsAdmissionConfig(),
            )

    stress_discovery: XsAdmissionResult | None = None
    stress_qualification: XsAdmissionResult | None = None
    stress_holdout: XsAdmissionResult | None = None
    if stress_spec is not None:
        stress_equity, stress_turnover = run_xs_composite_ledger(
            weights, opens, bar_funding, stress_spec,
        )
        stress_discovery = evaluate_xs_admission(
            _window_series(stress_equity, XS_DISCOVERY_START, DISCOVERY_END),
            _window_series(stress_turnover, XS_DISCOVERY_START, DISCOVERY_END),
            _window_series(benchmark, XS_DISCOVERY_START, DISCOVERY_END),
            XsAdmissionConfig(),
        )
        stress_qualification = evaluate_xs_admission(
            _window_series(stress_equity, QUALIFICATION_START, QUALIFICATION_END),
            _window_series(stress_turnover, QUALIFICATION_START, QUALIFICATION_END),
            _window_series(benchmark, QUALIFICATION_START, QUALIFICATION_END),
            XsAdmissionConfig(),
        )
        if unseal_holdout and holdout_start is not None:
            stress_holdout = evaluate_xs_admission(
                _window_series(stress_equity, holdout_start, holdout_end),
                _window_series(stress_turnover, holdout_start, holdout_end),
                _window_series(benchmark, holdout_start, holdout_end),
                XsAdmissionConfig(),
            )

    if router_spec is not None and allocation is not None and family_ledgers is not None:
        router_diagnostics = _build_router_diagnostics(
            allocation, holdout_start, holdout_end,
        )
        family_admission = _build_family_admission(family_ledgers, benchmark)

    symbols: dict[str, dict[str, object]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        if score_frames is not None:
            sf = score_frames[symbol]
            symbols[symbol] = {
                "composite_mean": round(float(sf.mean()), 8),
                "composite_std": round(float(sf.std()), 8),
                "fingerprint": fingerprints[symbol],
            }
        else:
            symbols[symbol] = {"fingerprint": fingerprints[symbol]}

    return XsTrendScreenReport(
        profile=profile,
        universe=TREND_SCREEN_SYMBOLS,
        spec=execution_spec,
        discovery=discovery,
        qualification=qualification,
        symbols=symbols,
        holdout=holdout,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        alpha_spec=alpha_spec,
        stress_spec=stress_spec,
        stress_discovery=stress_discovery,
        stress_qualification=stress_qualification,
        stress_holdout=stress_holdout,
        router_spec=_router_spec_payload(router_spec),
        router_diagnostics=router_diagnostics,
        family_admission=family_admission,
    )


def persist_xs_screen_report(report: XsTrendScreenReport, path: Path) -> None:
    """Write the byte-deterministic report payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")


def xs_screen_report_path(profile: str = XS_NEUTRAL_PROFILE_ID) -> Path:
    """Default persistence location for one source-controlled XS profile."""
    return Path("docs/results") / f"{profile}.json"


def _check_contract() -> None:
    """Executable assertions locking the XS screen entry-point surface."""
    from inspect import signature

    params = signature(run_xs_trend_screen).parameters
    assert set(params) >= {"start", "end", "unseal_holdout", "max_workers", "profile"}
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
    assert XS_NEUTRAL_PROFILE_ID == "xs_neutral_composite_v1"
    assert XS_ALPHA_PROFILE_ID == "xs_alpha_multihorizon_v2"
    assert XS_CONTEXTUAL_ALPHA_PROFILE_ID == "xs_alpha_contextual_v3"
    assert len(TREND_SCREEN_FAMILIES) == 15
    assert len(TREND_SCREEN_SYMBOLS) == 15


_check_contract()
