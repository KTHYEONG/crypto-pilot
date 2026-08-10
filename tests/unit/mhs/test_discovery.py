"""Discovery/qualification horizon-selection gate tests (spec §2, Part B)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.mhs.discovery import (
    _candidate_net_t,
    select_horizon_by_discovery_qualification,
)
from src.mhs.horizons import horizon_log_return

DISCOVERY_START = pd.Timestamp("2021-01-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
QUALIFICATION_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")

_SYMBOLS = ["S1", "S2", "S3", "S4"]
_SIGN = 1
_COST_BPS = 2.64
_PPY = 365.0 * 24


def _build_panel(
    seed: int,
    n_y1: int = 400,
    n_y2: int = 400,
    n_q: int = 400,
    align1: str = "h24",
    align2: str = "h48",
    alignq: str = "h48",
    k1: float = 2.0,
    k2: float = 0.2,
    kq: float = 1.0,
    phi: float = 0.0,
    q_phi: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Synthetic hourly panel whose open-to-open returns align with one
    candidate's log-return signal per window.

    ``log_close`` and ``opens`` are decoupled: the rank-weight signal is driven
    by ``log_close`` while the ledger's open-to-open returns are set to
    ``k * signal_target.shift(2)``, so the sign=+1 momentum book profits from a
    target signal with strength ``k``. ``phi`` negatively autocorrelates a
    segment's increments (mean reversion), which decorrelates the 24h and 48h
    signals inside that segment.
    """
    y1 = pd.date_range("2021-01-01", periods=n_y1, freq="1h", tz="UTC")
    y2 = pd.date_range("2022-01-01", periods=n_y2, freq="1h", tz="UTC")
    q = pd.date_range("2023-01-01", periods=n_q, freq="1h", tz="UTC")
    idx = y1.append(y2).append(q)
    rng = np.random.default_rng(seed)
    n = len(idx)
    drift = {"S1": -1.0, "S2": -0.5, "S3": 0.5, "S4": 1.0}
    n1, n2 = n_y1, n_y2
    incs: dict[str, np.ndarray] = {}
    for s in _SYMBOLS:
        u = rng.normal(drift[s] * 1e-4, 1e-3, n)
        u1, u2, u3 = u[:n1], u[n1 : n1 + n2], u[n1 + n2 :]
        for t in range(1, len(u2)):
            u2[t] = -phi * u2[t - 1] + (1 + phi) * u2[t]
        for t in range(1, len(u3)):
            u3[t] = -q_phi * u3[t - 1] + (1 + q_phi) * u3[t]
        incs[s] = np.concatenate([u1, u2, u3])
    log_close = pd.DataFrame(incs, index=idx).cumsum()

    sigs = {h: horizon_log_return(log_close, h) for h in (24, 48)}
    o2o = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
    for seg, tg, k in ((y1, align1, k1), (y2, align2, k2), (q, alignq, kq)):
        target = sigs[24 if tg == "h24" else 48]
        for s in _SYMBOLS:
            tgt = target[s].shift(2).to_numpy()
            o2o.loc[seg, s] = k * tgt[np.where(idx.isin(seg))[0]]
    o2o = o2o.fillna(0.0)
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(o2o.to_numpy(), axis=0)), index=idx, columns=_SYMBOLS,
    )
    bar_funding = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
    eligible = pd.DataFrame(True, index=idx, columns=_SYMBOLS)
    return log_close, opens, bar_funding, eligible, idx


def _score(log_close, opens, bar_funding, eligible, horizon: int, mask: np.ndarray, sign: int = _SIGN) -> float:
    return _candidate_net_t(
        log_close, eligible, opens, bar_funding, sign, horizon, mask,
        3, 1, _COST_BPS, _PPY,
    )


def _run(log_close, opens, bar_funding, eligible, idx, sign: int = _SIGN):
    return select_horizon_by_discovery_qualification(
        sign=sign, horizon_candidates=(24, 48), log_close=log_close,
        eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
        discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
        qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
    )


class TestDiscoveryQualificationGate:
    """SCENARIO_MHS_DISCOVERY_*: worst-year-robust selection, qualification
    single re-check, fail-closed behavior, and sign-consistent oriented scoring.

    SCENARIO_MHS_DISCOVERY_SIGN_REGRESSION_MOMENTUM_01: the three legacy
    scenarios below run the sign=+1 (momentum) direction and are regression
    anchors: sign=+1 must keep byte-identical behavior before/after the
    oriented-score fix (``min(sign * t) == min(t)``). The reversal-family
    scenarios (sign=-1) exercise the corrected worst-year semantics where
    "weakest evidence" is the yearly net_t closest to zero, not the most
    negative one.
    """

    def test_worst_year_robust_selection_beats_aggregate(self) -> None:
        """SCENARIO_MHS_DISCOVERY_WORST_YEAR_ROBUST_05: candidate 24 has a higher
        discovery aggregate net_t than candidate 48 but a worse single-year
        minimum, so the gate selects 48 (worst-year-robust), not 24 (which an
        aggregate criterion would have picked)."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        discovery_mask = np.asarray(idx.year <= 2022, dtype=bool)
        y1 = np.asarray(idx.year == 2021, dtype=bool)
        y2 = np.asarray(idx.year == 2022, dtype=bool)
        agg24 = _score(log_close, opens, bar_funding, eligible, 24, discovery_mask)
        agg48 = _score(log_close, opens, bar_funding, eligible, 48, discovery_mask)
        min24 = min(_score(log_close, opens, bar_funding, eligible, 24, y1),
                    _score(log_close, opens, bar_funding, eligible, 24, y2))
        min48 = min(_score(log_close, opens, bar_funding, eligible, 48, y1),
                    _score(log_close, opens, bar_funding, eligible, 48, y2))
        assert agg24 > agg48
        assert min24 < min48
        assert min48 >= 2.0

        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon == 48
        assert result.admitted is True
        assert dict(result.discovery_scores)[48] == pytest.approx(min48)

    def test_qualification_failure_does_not_reselect(self) -> None:
        """SCENARIO_MHS_DISCOVERY_QUALIFICATION_NO_RESELECT_06: discovery selects
        candidate 24 (best worst-year score), but 24 fails the qualification
        window |t|>=2.0 check while candidate 48 would pass -- the gate returns
        admitted=False with the same selected candidate recorded, never
        re-searching the grid on qualification."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            11, phi=0.0, k1=2.0, k2=0.2, kq=0.02, align1="h24", align2="h24",
        )
        y1 = np.asarray(idx.year == 2021, dtype=bool)
        y2 = np.asarray(idx.year == 2022, dtype=bool)
        q = np.asarray(idx.year == 2023, dtype=bool)
        min24 = min(_score(log_close, opens, bar_funding, eligible, 24, y1),
                    _score(log_close, opens, bar_funding, eligible, 24, y2))
        min48 = min(_score(log_close, opens, bar_funding, eligible, 48, y1),
                    _score(log_close, opens, bar_funding, eligible, 48, y2))
        q24 = _score(log_close, opens, bar_funding, eligible, 24, q)
        q48 = _score(log_close, opens, bar_funding, eligible, 48, q)
        assert min24 > min48
        assert min48 >= 2.0
        assert abs(q24) < 2.0 <= abs(q48)

        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon == 24
        assert result.admitted is False
        assert result.qualification_net_t == pytest.approx(q24)
        assert result.discovery_aggregate_net_t is not None

    def test_no_candidate_fails_closed(self) -> None:
        """SCENARIO_MHS_DISCOVERY_NO_CANDIDATE_FAILS_CLOSED_07: every candidate's
        discovery worst-year score is below the admission floor, so the gate
        returns selected_horizon=None and never evaluates qualification."""
        rng = np.random.default_rng(3)
        n = 800
        idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        incs = {s: rng.normal(0.0, 1e-3, n) for s in _SYMBOLS}
        log_close = pd.DataFrame(incs, index=idx).cumsum()
        o2o = pd.DataFrame(rng.normal(0.0, 1e-4, (n, 4)), index=idx, columns=_SYMBOLS)
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(o2o.to_numpy(), axis=0)), index=idx, columns=_SYMBOLS,
        )
        bar_funding = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
        eligible = pd.DataFrame(True, index=idx, columns=_SYMBOLS)
        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon is None
        assert result.admitted is False
        assert result.qualification_net_t is None
        assert result.qualification_sign_consistent is None

    def test_sign_minus_one_worst_year_is_closest_to_zero(self) -> None:
        """SCENARIO_MHS_DISCOVERY_SIGN_REVERSAL_WORST_YEAR_02: on a synthetic
        sign=-1 fixture whose discovery years are both negative for every
        candidate, the gate's reported worst-year score must be the yearly
        net_t CLOSEST TO ZERO (weakest evidence), not the most negative year.
        Reversal evidence is strongest when net_t is most negative, so plain
        min() picked the best year and ranked candidates backwards. Candidate
        48 has two moderate years while candidate 24 has one weak and one very
        strong year; the corrected oriented score ranks 48 (worst year about
        -30.6) stronger than 24 (worst year about -23.5)."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            3, phi=0.85, align1="h48", align2="h24",
        )
        y1 = np.asarray(idx.year == 2021, dtype=bool)
        y2 = np.asarray(idx.year == 2022, dtype=bool)
        sign = -1
        a1, a2 = (_score(log_close, opens, bar_funding, eligible, 24, m, sign) for m in (y1, y2))
        b1, b2 = (_score(log_close, opens, bar_funding, eligible, 48, m, sign) for m in (y1, y2))
        assert a1 < 0
        assert a2 < 0
        assert b1 < 0
        assert b2 < 0
        worst_a = max(a1, a2)
        worst_b = max(b1, b2)
        assert worst_a > min(a1, a2)
        assert worst_b > min(b1, b2)
        assert worst_b < worst_a

        result = _run(log_close, opens, bar_funding, eligible, idx, sign=sign)
        assert result.selected_horizon == 48
        assert dict(result.discovery_scores)[24] == pytest.approx(worst_a)
        assert dict(result.discovery_scores)[48] == pytest.approx(worst_b)

    def test_sign_minus_one_reversal_can_be_admitted(self) -> None:
        """SCENARIO_MHS_DISCOVERY_SIGN_REVERSAL_ADMITS_03: a synthetic sign=-1
        fixture whose every discovery-year net_t is strongly negative (worst
        year |t| >= 2.0) and whose qualification window is strongly negative
        with the same sign is admitted by the corrected gate. Before the fix
        the admission comparison ``best_score < 2.0`` was trivially true for
        any negative best_score, so a reversal family could never be admitted
        no matter how strong its (negative) evidence was."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, align1="h24", align2="h48",
        )
        result = _run(log_close, opens, bar_funding, eligible, idx, sign=-1)
        assert result.selected_horizon is not None
        assert result.admitted is True
        assert result.qualification_sign_consistent is True

    def test_sign_minus_one_all_nonfinite_years_never_selected(self) -> None:
        """SCENARIO_MHS_DISCOVERY_SIGN_NO_FINITE_YEARS_04: on a sign=-1 fixture
        where candidate 24's log-return signal is identically zero for the whole
        discovery window (a period-24 ``log_close`` makes the 24h diff exactly
        zero), every discovery-year net_t for that candidate is non-finite and
        it is assigned the worst possible ORIENTED score (``float("-inf")``,
        not ``sign * -inf`` which would flip to ``+inf`` under sign=-1). The
        finite-data candidate 36 is selected instead."""
        y1 = pd.date_range("2021-01-01", periods=400, freq="1h", tz="UTC")
        y2 = pd.date_range("2022-01-01", periods=400, freq="1h", tz="UTC")
        q = pd.date_range("2023-01-01", periods=400, freq="1h", tz="UTC")
        idx = y1.append(y2).append(q)
        n = len(idx)
        base = np.tile(np.arange(24, dtype=float) * 1e-3, (n // 24) + 1)[:n]
        phases = {"S1": 0, "S2": 6, "S3": 12, "S4": 18}
        log_close = pd.DataFrame(
            {s: np.roll(base, ph) for s, ph in phases.items()}, index=idx,
        )
        sig36 = horizon_log_return(log_close, 36)
        o2o = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
        for seg in (y1, y2, q):
            for s in _SYMBOLS:
                tgt = sig36[s].shift(2).to_numpy()
                o2o.loc[seg, s] = 0.5 * tgt[np.where(idx.isin(seg))[0]]
        o2o = o2o.fillna(0.0)
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(o2o.to_numpy(), axis=0)), index=idx, columns=_SYMBOLS,
        )
        bar_funding = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
        eligible = pd.DataFrame(True, index=idx, columns=_SYMBOLS)
        result = select_horizon_by_discovery_qualification(
            sign=-1, horizon_candidates=(24, 36), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
        )
        assert result.selected_horizon == 36
        assert math.isnan(dict(result.discovery_scores)[24])
        assert math.isfinite(dict(result.discovery_scores)[36])
