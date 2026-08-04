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

import hashlib
import json
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd

from src.application.research.technical.trend_screen import _load_symbol_data
from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import _align_funding_rates
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.cross_sectional import (
    XsAdmissionConfig,
    XsAdmissionResult,
    XsCompositeSpec,
    build_xs_neutral_weights,
    evaluate_xs_admission,
    run_xs_composite_ledger,
)
from src.research.technical_experts.signals import generate_signal_events
from src.research.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    DISCOVERY_START,
    QUALIFICATION_END,
    QUALIFICATION_START,
    TREND_SCREEN_CANDIDATES,
    TREND_SCREEN_FAMILIES,
    TREND_SCREEN_SYMBOLS,
)

XS_NEUTRAL_PROFILE_ID = "xs_neutral_composite_v1"

__all__ = [
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

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload: dict[str, object] = {
            "profile": self.profile,
            "universe": list(self.universe),
            "discovery_start": DISCOVERY_START.isoformat(),
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
) -> XsTrendScreenReport:
    return XsTrendScreenReport(
        profile=XS_NEUTRAL_PROFILE_ID,
        universe=TREND_SCREEN_SYMBOLS,
        spec=spec,
        discovery=_failed_result(constraint),
        qualification=_failed_result(constraint),
        symbols={},
    )


def run_xs_trend_screen(
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    unseal_holdout: bool = False,
    max_workers: int | None = None,
) -> XsTrendScreenReport:
    """Execute one sealed XS composite screen profile.

    Loads funding-complete 4h data once per symbol, builds the per-symbol
    composite score, constructs the EWMA-smoothed dollar-neutral unit-gross
    book with the frozen no-trade band, compounds it under the production
    execution convention, and admits discovery and qualification with the
    scale-invariant gates. Nothing is registered; ``end`` defaults to the
    sealed cutoff unless ``unseal_holdout`` is set.
    """
    end = resolve_evaluation_end(end, unseal_holdout=unseal_holdout)
    spec = XsCompositeSpec()

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    fingerprints: dict[str, dict[str, str]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        try:
            frame, funding, fingerprint, _coverage = _load_symbol_data(symbol, start, end)
        except (DataIntegrityError, FileNotFoundError) as exc:
            return _fail_closed_report(
                spec, f"symbol_unavailable:{symbol}:{type(exc).__name__}",
            )
        data[symbol] = (frame, funding)
        fingerprints[symbol] = fingerprint

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        return _fail_closed_report(spec, "insufficient_common_grid")

    score_frames = {
        symbol: _symbol_composite_score(frame) for symbol, (frame, _funding) in data.items()
    }
    score = pd.DataFrame(
        {symbol: sf.reindex(common) for symbol, sf in score_frames.items()},
    )
    weights = build_xs_neutral_weights(score, spec.halflife_bars, spec.no_trade_band)

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

    equity, turnover = run_xs_composite_ledger(weights, opens, bar_funding, spec)

    discovery = evaluate_xs_admission(
        _window_series(equity, DISCOVERY_START, DISCOVERY_END),
        _window_series(turnover, DISCOVERY_START, DISCOVERY_END),
        _window_series(benchmark, DISCOVERY_START, DISCOVERY_END),
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

    symbols: dict[str, dict[str, object]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        sf = score_frames[symbol]
        symbols[symbol] = {
            "composite_mean": round(float(sf.mean()), 8),
            "composite_std": round(float(sf.std()), 8),
            "fingerprint": fingerprints[symbol],
        }

    return XsTrendScreenReport(
        profile=XS_NEUTRAL_PROFILE_ID,
        universe=TREND_SCREEN_SYMBOLS,
        spec=spec,
        discovery=discovery,
        qualification=qualification,
        symbols=symbols,
        holdout=holdout,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )


def _check_contract() -> None:
    """Executable assertions locking the XS screen entry-point surface."""
    from inspect import signature

    params = signature(run_xs_trend_screen).parameters
    assert set(params) >= {"start", "end", "unseal_holdout", "max_workers"}
    assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
    assert XS_NEUTRAL_PROFILE_ID == "xs_neutral_composite_v1"
    assert len(TREND_SCREEN_FAMILIES) == 15
    assert len(TREND_SCREEN_SYMBOLS) == 15


_check_contract()
