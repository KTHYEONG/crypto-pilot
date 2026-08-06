"""Principled, measured universe-qualification filters for the XS alpha family.

Diagnosis-only candidate scan: coverage, funding presence, taker-buy-ratio
integrity, and recent liquidity are the only qualification inputs -- realized
returns are never consulted (the ``pit_universe.py`` invariant "Realized
returns are never consulted"). Nothing here gates production: the scan only
measures which already-collected symbols would qualify, so a later breadth
re-attempt on a different signal architecture never has to re-derive the
filter set.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

_TRAILING_DAILY_BARS = 180  # 30 days at 4h
_DAILY_BARS = 6  # 4h bars per day


@dataclass(frozen=True, slots=True)
class UniverseCandidateSpec:
    """Immutable, pre-registered scan parameters.

    ``min_coverage`` reuses the project's frozen coverage bar already
    established by :class:`PitUniverseSpec` (``min_bar_coverage=0.99``);
    ``seasoning_tolerance_days`` captures this scan's own measured tolerance
    between the requested ``discovery_start`` and a symbol's first bar.
    """

    min_coverage: float = 0.99
    seasoning_tolerance_days: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.min_coverage <= 1:
            raise ValueError(
                f"min_coverage must be in (0, 1], got {self.min_coverage}"
            )
        if self.seasoning_tolerance_days < 0:
            raise ValueError(
                f"seasoning_tolerance_days must be >= 0, got "
                f"{self.seasoning_tolerance_days}"
            )


@dataclass(frozen=True, slots=True)
class UniverseCandidateResult:
    """Per-symbol scan outcome. Covers archive metadata, never prices or returns."""

    symbol: str
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    coverage: float
    has_funding: bool
    taker_ratio_valid: bool
    avg_daily_quote_vol_recent: float
    qualifies: bool

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.first_bar.tzinfo is None or self.last_bar.tzinfo is None:
            raise ValueError("first_bar and last_bar must be tz-aware UTC")
        if self.last_bar < self.first_bar:
            raise ValueError("last_bar must not precede first_bar")
        if not 0 <= self.coverage <= 1:
            raise ValueError(f"coverage must be in [0, 1], got {self.coverage}")


def expected_4h_bar_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Number of 4h bars spanning ``[start, end]`` inclusive on an exact 4h grid."""
    if end < start:
        raise ValueError("end must not precede start")
    return int((end - start) // pd.Timedelta(hours=4)) + 1


def evaluate_universe_candidate(
    symbol: str,
    frame: pd.DataFrame,
    has_funding: bool,
    discovery_start: pd.Timestamp,
    end: pd.Timestamp,
    spec: UniverseCandidateSpec,
) -> UniverseCandidateResult:
    """Pure per-symbol evaluator; no I/O, no realized-return consultation.

    ``frame`` is one symbol's already-loaded 4h OHLCV restricted to
    ``[discovery_start, end]`` by the caller (mirrors ``_load_symbol_data``'s
    contract). ``coverage`` is measured against the full requested window
    (``expected_4h_bar_count``), not the frame's own span, so a late-IPO symbol
    is penalized for its missing early bars. ``taker_ratio_valid`` is the exact
    finite-and-in-``[0, 1]`` predicate of ``_validate_alpha_panels``, never a
    divergent copy. ``avg_daily_quote_vol_recent`` is the mean 4h quote volume
    over the trailing 180 bars (30 days) times 6, else ``0.0`` for shorter
    frames. ``qualifies`` is the AND of every filter. Fail-closed
    ``ValueError`` on an empty frame or a frame missing a required column --
    never a silent zero-fill.
    """
    required = ("close", "taker_buy_ratio", "quote_vol")
    if frame.empty:
        raise ValueError(f"frame for {symbol} is empty over [discovery_start, end]")
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(
            f"frame for {symbol} is missing required columns: {missing}"
        )

    coverage = len(frame) / expected_4h_bar_count(discovery_start, end)

    taker_values = frame["taker_buy_ratio"].to_numpy(dtype=np.float64)
    taker_ratio_valid = bool(
        np.isfinite(taker_values).all()
        and not (taker_values < 0.0).any()
        and not (taker_values > 1.0).any()
    )

    quote_vol = frame["quote_vol"].to_numpy(dtype=np.float64)
    if len(quote_vol) >= _TRAILING_DAILY_BARS:
        avg_daily_quote_vol_recent = (
            float(np.mean(quote_vol[-_TRAILING_DAILY_BARS:])) * _DAILY_BARS
        )
    else:
        avg_daily_quote_vol_recent = 0.0

    first_bar = frame.index[0]
    last_bar = frame.index[-1]
    seasoned = first_bar <= discovery_start + pd.Timedelta(
        days=spec.seasoning_tolerance_days
    )
    qualifies = bool(
        coverage >= spec.min_coverage
        and seasoned
        and has_funding
        and taker_ratio_valid
        and avg_daily_quote_vol_recent > 0
    )
    return UniverseCandidateResult(
        symbol=symbol,
        first_bar=first_bar,
        last_bar=last_bar,
        coverage=coverage,
        has_funding=has_funding,
        taker_ratio_valid=taker_ratio_valid,
        avg_daily_quote_vol_recent=avg_daily_quote_vol_recent,
        qualifies=qualifies,
    )


def _check_contract() -> None:
    """Executable assertions locking the candidate-scan contract surface."""
    spec = UniverseCandidateSpec()
    assert (spec.min_coverage, spec.seasoning_tolerance_days) == (0.99, 5)
    assert {f.name for f in fields(UniverseCandidateSpec)} == {
        "min_coverage", "seasoning_tolerance_days",
    }
    assert {f.name for f in fields(UniverseCandidateResult)} == {
        "symbol", "first_bar", "last_bar", "coverage", "has_funding",
        "taker_ratio_valid", "avg_daily_quote_vol_recent", "qualifies",
    }
    idx = pd.date_range("2022-04-01", periods=200, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": np.full(200, 100.0),
            "taker_buy_ratio": np.full(200, 0.5),
            "quote_vol": np.full(200, 1_000_000.0),
        },
        index=idx,
    )
    result = evaluate_universe_candidate(
        "TESTUSDT", frame, True, idx[0], idx[-1], UniverseCandidateSpec(),
    )
    assert result.qualifies is True
    bad = frame.copy()
    bad.loc[idx[5], "taker_buy_ratio"] = 1.5
    bad_result = evaluate_universe_candidate(
        "TESTUSDT", bad, True, idx[0], idx[-1], UniverseCandidateSpec(),
    )
    assert bad_result.taker_ratio_valid is False
    assert bad_result.qualifies is False


_check_contract()
