"""Discovery/qualification horizon-selection gate tests (spec §2, Part B)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import src.mhs.discovery as discovery
from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.discovery import (
    _candidate_net_t,
    fold_train_only_discovery_qualification,
    select_horizon_by_discovery_qualification,
)
from src.mhs.evaluation import AnchoredPurgedFold
from src.mhs.horizons import horizon_log_return, vol_normalized_horizon_signal

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


def _disagreement_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Momentum signal disagreement fixture (horizon=2).

    Cross-sectionally the raw ``horizon_log_return`` ranks NOISY above QUIET
    while the vol-normalized signal ranks QUIET above NOISY, so a sign=+1
    ``_horizon_weights`` built from the two signals must produce different
    weight books -- the fixture that proves the sign dispatch took effect.
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    log_close = pd.DataFrame(
        {
            "NOISY": [0.0, 0.5, -0.2, 0.4, 0.8, 0.7, 1.1, 0.9],
            "QUIET": [0.0, 0.01, 0.04, 0.06, 0.07, 0.10, 0.11, 0.14],
        },
        index=idx,
    )
    eligible = pd.DataFrame(True, index=idx, columns=log_close.columns)
    return log_close, eligible


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
    """Worst-year-robust selection, qualification single re-check, and fail-closed behavior."""

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

    def test_weight_reuse_builds_weights_once_per_candidate(self, monkeypatch) -> None:
        """SCENARIO_MHS_DISCOVERY_WEIGHT_REUSE_CALL_COUNT_09: after the weight
        hoisting refactor, ``_horizon_weights`` must be built once per candidate
        horizon (shared across every discovery year) plus exactly one more for
        the winning horizon's aggregate+qualification reuse -- never once per
        (horizon, year) pair. The admitted panel exercises the full path so the
        ``+1`` aggregate/qualification call is reachable."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        original = discovery._horizon_weights
        calls: list[int] = []

        def counting_wrapper(*args, **kwargs) -> pd.DataFrame:
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(discovery, "_horizon_weights", counting_wrapper)
        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.admitted is True
        assert len(calls) == len((24, 48)) + 1

    def test_build_candidate_weights_matches_direct_calls(self) -> None:
        """SCENARIO_MHS_HORIZON_SEARCH_EFF_01_BUILD_CANDIDATE_WEIGHTS:
        ``build_candidate_weights`` returns one weight book per candidate
        horizon, each byte-identical to a direct ``_horizon_weights`` call."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(24)
        cache = discovery.build_candidate_weights(log_close, eligible, 1, (24, 48), 3, 1)
        assert set(cache) == {24, 48}
        for h in (24, 48):
            pd.testing.assert_frame_equal(
                cache[h],
                discovery._horizon_weights(log_close, eligible, 1, h, 3, 1),
            )

    def test_cache_omitted_is_byte_identical(self) -> None:
        """SCENARIO_MHS_HORIZON_SEARCH_EFF_02_CACHE_OMITTED_IS_BYTE_IDENTICAL:
        with ``precomputed_candidate_weights`` omitted (the default None), the
        gate keeps producing the pre-change result on the worst-year-robust
        fixture (the ``test_worst_year_robust_selection_beats_aggregate``
        regression anchor): 48 selected, admitted, worst-year score reported."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        y1 = np.asarray(idx.year == 2021, dtype=bool)
        y2 = np.asarray(idx.year == 2022, dtype=bool)
        min48 = min(_score(log_close, opens, bar_funding, eligible, 48, y1),
                    _score(log_close, opens, bar_funding, eligible, 48, y2))
        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon == 48
        assert result.admitted is True
        assert dict(result.discovery_scores)[48] == pytest.approx(min48)

    def test_cache_supplied_skips_horizon_weights_calls(self, monkeypatch) -> None:
        """SCENARIO_MHS_HORIZON_SEARCH_EFF_03_CACHE_SUPPLIED_SKIPS_HORIZON_WEIGHTS_CALLS:
        with a complete ``precomputed_candidate_weights`` mapping supplied, the
        gate never calls ``_horizon_weights`` internally (call-count unchanged
        after the cache build) and returns a result identical in every field to
        the cache-free call on the admitted fixture."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        baseline = _run(log_close, opens, bar_funding, eligible, idx)
        assert baseline.admitted is True
        original = discovery._horizon_weights
        calls: list[int] = []

        def counting_wrapper(*args, **kwargs) -> pd.DataFrame:
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(discovery, "_horizon_weights", counting_wrapper)
        cache = discovery.build_candidate_weights(log_close, eligible, 1, (24, 48), 3, 1)
        n_build_calls = len(calls)
        assert n_build_calls == len((24, 48))
        result = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            precomputed_candidate_weights=cache,
        )
        assert len(calls) == n_build_calls
        assert result.selected_horizon == baseline.selected_horizon
        assert result.admitted == baseline.admitted
        assert result.discovery_scores == baseline.discovery_scores
        assert result.discovery_aggregate_net_t == baseline.discovery_aggregate_net_t
        assert result.qualification_net_t == baseline.qualification_net_t
        assert result.qualification_sign_consistent == baseline.qualification_sign_consistent

    def test_sign_plus_one_uses_vol_normalized_signal(self) -> None:
        """SCENARIO_DISCOVERY_MOMENTUM_USES_VOL_NORMALIZED: the sign=+1
        momentum discovery weights are built from the vol-normalized signal,
        not raw ``horizon_log_return``. On the disagreement fixture the two
        rank differently, so the assertion is a real (not accidental) check."""
        log_close, eligible = _disagreement_panel()
        weights = discovery._horizon_weights(log_close, eligible, 1, 2, 2, 1)
        expected = phase_tranche_book(
            rank_weight_book(vol_normalized_horizon_signal(log_close, 2), eligible, 1, 2),
            1,
        )
        raw_expected = phase_tranche_book(
            rank_weight_book(horizon_log_return(log_close, 2), eligible, 1, 2),
            1,
        )
        pd.testing.assert_frame_equal(weights, expected)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(weights, raw_expected)

    def test_sign_minus_one_keeps_raw_signal(self) -> None:
        """SCENARIO_DISCOVERY_REVERSAL_UNCHANGED: the sign=-1 reversal family
        keeps raw ``horizon_log_return`` weights exactly as before, even on a
        fixture where the vol-normalized variant would rank differently."""
        log_close, eligible = _disagreement_panel()
        weights = discovery._horizon_weights(log_close, eligible, -1, 2, 2, 1)
        expected = phase_tranche_book(
            rank_weight_book(horizon_log_return(log_close, 2), eligible, -1, 2),
            1,
        )
        pd.testing.assert_frame_equal(weights, expected)

    def test_yearly_net_t_matches_per_year_score(self) -> None:
        """SCENARIO_MHS_DISCOVERY_YEARLY_NET_T_EXPOSED_09: ``yearly_net_t``
        exposes the exact per-(horizon, year) net_t values the worst-year min
        was computed from -- one (year, net_t) entry per discovery year for
        every candidate, each matching an independent ``_score`` call, and the
        worst-year min derived from those entries equals ``discovery_scores``."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        y1 = np.asarray(idx.year == 2021, dtype=bool)
        y2 = np.asarray(idx.year == 2022, dtype=bool)
        result = _run(log_close, opens, bar_funding, eligible, idx)
        yearly = dict(result.yearly_net_t)
        assert set(yearly) == {24, 48}
        for h in (24, 48):
            years = dict(yearly[h])
            assert set(years) == {2021, 2022}
            assert years[2021] == pytest.approx(
                _score(log_close, opens, bar_funding, eligible, h, y1)
            )
            assert years[2022] == pytest.approx(
                _score(log_close, opens, bar_funding, eligible, h, y2)
            )
            worst = min(1 * t for t in years.values() if math.isfinite(t))
            assert dict(result.discovery_scores)[h] == pytest.approx(worst)

    def test_yearly_net_t_records_nonfinite_year(self) -> None:
        """SCENARIO_MHS_DISCOVERY_YEARLY_NET_T_NONFINITE_YEAR_10: on the
        all-non-finite-years fixture (candidate 24's signal is identically
        zero), ``yearly_net_t`` still carries one entry per discovery year for
        candidate 24 with NaN values -- the non-finite years are recorded
        transparently, not dropped, even though they never entered the
        worst-year min computation."""
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
        years_24 = dict(dict(result.yearly_net_t)[24])
        assert set(years_24) == {2021, 2022}
        assert all(math.isnan(v) for v in years_24.values())
        years_36 = dict(dict(result.yearly_net_t)[36])
        assert all(math.isfinite(v) for v in years_36.values())

    def test_yearly_net_t_field_does_not_change_existing_fields(self) -> None:
        """SCENARIO_MHS_DISCOVERY_YEARLY_NET_T_REGRESSION_11: adding
        ``yearly_net_t`` leaves every pre-existing field byte-identical to the
        cache-free baseline on the admitted worst-year-robust fixture."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon == 48
        assert result.admitted is True
        assert dict(result.discovery_scores)[48] == pytest.approx(
            min(
                _score(log_close, opens, bar_funding, eligible, 48,
                       np.asarray(idx.year == 2021, dtype=bool)),
                _score(log_close, opens, bar_funding, eligible, 48,
                       np.asarray(idx.year == 2022, dtype=bool)),
            )
        )
        assert result.discovery_aggregate_net_t is not None
        assert result.qualification_net_t is not None
        assert result.qualification_sign_consistent is True

class TestYearlyNetTDiagnostic:
    """SCENARIO_MHS_YEARLY_DIAGNOSTIC_COVERS_FULL_HISTORY_01: the report-only
    full-history diagnostic covers every requested calendar year -- including
    2024/2025, which the discovery-window-confined ``yearly_net_t`` field can
    never populate -- records zero-row/non-finite years as NaN rather than
    dropping or zeroing them, and fails closed on empty years / negative cost.
    """

    @staticmethod
    def _panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        idx = pd.date_range("2021-01-01", periods=5 * 8760, freq="1h", tz="UTC")
        rng = np.random.default_rng(7)
        base = np.tile(np.array([0.5, -0.5, 0.25, -0.25]), (len(idx), 1))
        weights = pd.DataFrame(base, index=idx, columns=_SYMBOLS)
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.standard_normal((len(idx), 4)) * 0.001, axis=0)),
            index=idx, columns=_SYMBOLS,
        )
        bar_funding = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
        return weights, opens, bar_funding

    def test_covers_full_history_with_finite_values(self) -> None:
        """A 2021-2025 hourly panel yields a finite net_t for every requested
        year (2024/2025 included), each matching an independent score call."""
        weights, opens, bar_funding = self._panel()
        out = discovery.yearly_net_t_diagnostic(
            weights, opens, bar_funding, (2021, 2022, 2023, 2024, 2025), _COST_BPS, _PPY,
        )
        assert set(out) == {2021, 2022, 2023, 2024, 2025}
        for year in (2021, 2022, 2023, 2024, 2025):
            assert math.isfinite(out[year])
        mask = discovery._year_mask(weights.index, 2021)
        assert out[2021] == pytest.approx(
            discovery._score_masked_net_t(
                weights, opens, bar_funding, mask, _COST_BPS, _PPY,
            )
        )

    def test_zero_row_year_returns_nan_not_dropped(self) -> None:
        """A panel covering only 2021 returns NaN -- never a KeyError, never a
        silently dropped key -- for the requested 2020/2022 years with no rows."""
        idx = pd.date_range("2021-01-01", periods=8760, freq="1h", tz="UTC")
        rng = np.random.default_rng(8)
        weights = pd.DataFrame(
            np.tile(np.array([0.5, -0.5, 0.25, -0.25]), (len(idx), 1)),
            index=idx, columns=_SYMBOLS,
        )
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.standard_normal((len(idx), 4)) * 0.001, axis=0)),
            index=idx, columns=_SYMBOLS,
        )
        bar_funding = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
        out = discovery.yearly_net_t_diagnostic(
            weights, opens, bar_funding, (2020, 2021, 2022), _COST_BPS, _PPY,
        )
        assert set(out) == {2020, 2021, 2022}
        assert math.isnan(out[2020])
        assert math.isfinite(out[2021])
        assert math.isnan(out[2022])

    def test_fails_closed_on_bad_inputs(self) -> None:
        """Empty years, negative cost, and non-positive periods per year all
        raise ValueError (the cost_response_curve validation convention)."""
        weights, opens, bar_funding = self._panel()
        with pytest.raises(ValueError, match="years must not be empty"):
            discovery.yearly_net_t_diagnostic(weights, opens, bar_funding, (), _COST_BPS, _PPY)
        with pytest.raises(ValueError, match="cost_bps must be >= 0"):
            discovery.yearly_net_t_diagnostic(weights, opens, bar_funding, (2021,), -1.0, _PPY)
        with pytest.raises(ValueError, match="periods_per_year must be > 0"):
            discovery.yearly_net_t_diagnostic(weights, opens, bar_funding, (2021,), _COST_BPS, 0.0)


class TestHacAdjustedNetTDiagnostic:
    """The Bartlett/HAC-adjusted prescreen t-stat is a pure, deterministic correction of the naive i.i.d. t-stat."""

    @staticmethod
    def _iid_series(seed: int = 3, n: int = 2000) -> np.ndarray:
        series = np.random.default_rng(seed).normal(0.0, 1.0, n)
        return series - series.mean()

    @staticmethod
    def _ar1_series(phi: float = 0.85, seed: int = 3, n: int = 2000) -> np.ndarray:
        # drifting increments so the level has a nonzero mean and the naive
        # t-stat is meaningful (a zero-mean AR(1) yields raw_t == 0 exactly,
        # making the adjusted-vs-raw comparison degenerate)
        u = np.random.default_rng(seed).normal(0.05, 1.0, n)
        series = np.zeros(n)
        for t in range(1, n):
            series[t] = phi * series[t - 1] + u[t]
        return series

    def test_hac_denom_iid_near_one(self) -> None:
        """SCENARIO_HAC_DENOM_IID_NEAR_ONE: a seeded i.i.d. zero-mean Gaussian
        series (n>=2000, max_lag=168) has no serial correlation, so the
        Bartlett long-run-variance ratio is close to 1.0 (within +-0.15) -- the
        adjustment never invents inflation where the data has none."""
        denom = discovery._bartlett_hac_denom(self._iid_series(), 168)
        assert denom == pytest.approx(1.0, abs=0.15)

    def test_hac_denom_zero_series_is_one(self) -> None:
        """Contract regression anchor: a zero series (zero variance) yields the
        no-adjustment denominator 1.0, never a division by zero."""
        assert discovery._bartlett_hac_denom(np.zeros(500), 168) == pytest.approx(1.0)

    def test_hac_denom_positive_autocorr_inflates(self) -> None:
        """SCENARIO_HAC_DENOM_POSITIVE_AUTOCORR_INFLATES: an AR(1) series with
        phi=0.85 (same seed/n as the i.i.d. case) has strong positive serial
        correlation, inflating the denominator materially above 1.0 (>2.0), and
        the adjusted |t| (raw_t / sqrt(denom)) is materially smaller in
        magnitude than the naive raw_t on the same series."""
        series = self._ar1_series()
        mean = series.mean()
        denom = discovery._bartlett_hac_denom(series - mean, 168)
        assert denom > 2.0
        raw_t = mean / series.std(ddof=1) * math.sqrt(len(series))
        assert abs(raw_t / math.sqrt(denom)) < abs(raw_t)

    def test_adjusted_net_t_iid_matches_raw(self) -> None:
        """SCENARIO_ADJUSTED_NET_T_IID_MATCHES_RAW: on a genuinely serially
        uncorrelated weights/opens panel (constant weights, i.i.d.-increment
        random-walk opens -- the same pattern as ``TestYearlyNetTDiagnostic._panel``)
        the adjusted net_t is within 10% relative tolerance of the raw net_t,
        proving the HAC adjustment is a no-op (denom~=1) on uncorrelated returns
        and introduces no mystery constant offset.

        Note: ``_build_panel(phi=0.0)`` is NOT i.i.d. -- its net series is an
        MA(23) rolling-signal process with Bartlett denom ~15.5 -- so this
        scenario uses a genuinely uncorrelated panel to exercise the intended
        no-op property."""
        idx = pd.date_range("2021-01-01", periods=4000, freq="1h", tz="UTC")
        rng = np.random.default_rng(7)
        weights = pd.DataFrame(
            np.tile(np.array([0.5, -0.5, 0.25, -0.25]), (len(idx), 1)),
            index=idx, columns=_SYMBOLS,
        )
        opens = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.standard_normal((len(idx), 4)) * 0.001, axis=0)),
            index=idx, columns=_SYMBOLS,
        )
        bar_funding = pd.DataFrame(0.0, index=idx, columns=_SYMBOLS)
        mask = np.ones(len(idx), dtype=bool)
        raw = discovery._score_masked_net_t(weights, opens, bar_funding, mask, _COST_BPS, _PPY)
        adj = discovery._score_masked_adjusted_net_t(
            weights, opens, bar_funding, mask, _COST_BPS, _PPY, max_lag_periods=24,
        )
        assert math.isfinite(raw)
        assert math.isfinite(adj)
        assert adj == pytest.approx(raw, rel=0.10)


class TestDiscoveryAdjustedOptIn:
    """The Bartlett/HAC-adjusted fields are opt-in-only diagnostics that never
    perturb the admission path."""

    def test_discovery_default_off_bit_identical(self) -> None:
        """With the flag omitted (default False) the gate returns the
        pre-change baseline on the admitted worst-year-robust fixture -- every
        raw field bit-identical -- and every new field sits at its empty/None
        default."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon == 48
        assert result.admitted is True
        assert result.discovery_scores == (
            (24, 16.43401479542989), (48, 25.517515748645305),
        )
        assert result.discovery_aggregate_net_t == 23.145363968299133
        assert result.qualification_net_t == 66.88810950782543
        assert result.qualification_sign_consistent is True
        assert result.yearly_net_t == (
            (24, ((2021, 39.39252393058957), (2022, 16.43401479542989))),
            (48, ((2021, 25.517515748645305), (2022, 37.86453888754656))),
        )
        assert result.yearly_adjusted_net_t == ()
        assert result.discovery_scores_adjusted == ()
        assert result.discovery_aggregate_adjusted_net_t is None
        assert result.qualification_adjusted_net_t is None
        assert result.adjusted_admitted is None

    def test_discovery_opt_in_preserves_raw_fields(self) -> None:
        """Identical inputs with the flag False vs True yield bit-identical raw
        admission-path fields, while the True call additionally populates the
        adjusted tables covering every candidate horizon."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        off = _run(log_close, opens, bar_funding, eligible, idx)
        on = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            compute_adjusted_net_t=True,
        )
        assert on.selected_horizon == off.selected_horizon
        assert on.admitted == off.admitted
        assert on.discovery_scores == off.discovery_scores
        assert on.discovery_aggregate_net_t == off.discovery_aggregate_net_t
        assert on.qualification_net_t == off.qualification_net_t
        assert on.qualification_sign_consistent == off.qualification_sign_consistent
        assert on.yearly_net_t == off.yearly_net_t
        assert set(dict(on.yearly_adjusted_net_t)) == {24, 48}
        assert set(dict(on.discovery_scores_adjusted)) == {24, 48}

    def test_adjusted_admitted_sign_invariant(self) -> None:
        """Dividing by sqrt(positive denom) never flips the sign of the
        qualification t-stat, and ``adjusted_admitted`` is True only when the
        raw sign is consistent AND the adjusted magnitude clears the admission
        floor."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        on = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            compute_adjusted_net_t=True,
        )
        assert on.selected_horizon is not None
        assert on.qualification_net_t is not None
        assert on.qualification_adjusted_net_t is not None
        assert math.isfinite(on.qualification_adjusted_net_t)
        assert math.copysign(1.0, on.qualification_adjusted_net_t) == math.copysign(
            1.0, on.qualification_net_t
        )
        assert on.qualification_sign_consistent is True
        assert abs(on.qualification_adjusted_net_t) >= 2.0
        assert on.adjusted_admitted is True


class TestRegimeScaledNetTDiagnostic:
    """The vol-regime cash-scale-adjusted fields are an opt-in-only
    diagnostic."""

    def test_regime_scale_constant_vol_is_one(self) -> None:
        """Constant vol maps to regime scale of 1.0."""
        idx = pd.date_range("2021-01-01", periods=800, freq="1h", tz="UTC")
        scale = discovery._discovery_regime_cash_scale(pd.Series(1.0, index=idx))
        assert (scale == 1.0).all()

    def test_regime_scale_vol_spike_clips_to_floor(self) -> None:
        """SCENARIO_REGIME_SCALE_VOL_SPIKE_CLIPS_TO_FLOOR: a sustained multi-
        week 10x vol spike drives the scale to the 0.5 floor inside the spike
        (never below, never above 1.0) and is byte-identical to the application
        layer's ``_regime_cash_scale`` on the same input -- the duplicated
        kernel mirrors ``evaluation.py`` exactly."""
        idx = pd.date_range("2021-01-01", periods=2000, freq="1h", tz="UTC")
        vol = pd.Series(1.0, index=idx)
        vol.iloc[800:1136] = 10.0
        scale = discovery._discovery_regime_cash_scale(vol)
        assert (scale >= 0.5).all()
        assert (scale <= 1.0).all()
        assert (scale.iloc[850:1100] == 0.5).all()

        from src.application.research.mhs import evaluation as ev

        pd.testing.assert_series_equal(
            scale,
            ev._regime_cash_scale(vol),
            check_names=False,
            check_freq=False,
        )

    def test_regime_scaled_net_t_scale_one_matches_raw(self) -> None:
        """SCENARIO_REGIME_SCALED_NET_T_SCALE_ONE_MATCHES_RAW: with
        ``regime_scale`` identically 1.0 the regime-scaled path degenerates
        exactly (rtol=1e-12) to ``_score_masked_net_t`` on the same inputs."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        weights = discovery._horizon_weights(log_close, eligible, 1, 24, 3, 1)
        mask = np.asarray(idx.year <= 2022, dtype=bool)
        raw = discovery._score_masked_net_t(
            weights, opens, bar_funding, mask, _COST_BPS, _PPY,
        )
        scale = pd.Series(1.0, index=idx)
        scaled = discovery._score_masked_regime_scaled_net_t(
            weights, scale, opens, bar_funding, mask, _COST_BPS, _PPY,
        )
        assert scaled == pytest.approx(raw, rel=1e-12)

    def test_discovery_default_off_regime_fields_empty(self) -> None:
        """SCENARIO_DISCOVERY_DEFAULT_OFF_REGIME_FIELDS_EMPTY: with the flag
        omitted (default False) the gate returns the pre-change baseline on the
        admitted worst-year-robust fixture -- every raw field bit-identical --
        and all five new regime fields sit at their empty/None defaults."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        result = _run(log_close, opens, bar_funding, eligible, idx)
        assert result.selected_horizon == 48
        assert result.admitted is True
        assert result.discovery_scores == (
            (24, 16.43401479542989), (48, 25.517515748645305),
        )
        assert result.discovery_aggregate_net_t == 23.145363968299133
        assert result.qualification_net_t == 66.88810950782543
        assert result.qualification_sign_consistent is True
        assert result.yearly_net_t == (
            (24, ((2021, 39.39252393058957), (2022, 16.43401479542989))),
            (48, ((2021, 25.517515748645305), (2022, 37.86453888754656))),
        )
        assert result.yearly_regime_scaled_net_t == ()
        assert result.discovery_scores_regime_scaled == ()
        assert result.discovery_aggregate_regime_scaled_net_t is None
        assert result.qualification_regime_scaled_net_t is None
        assert result.regime_scaled_admitted is None

    def test_discovery_opt_in_regime_preserves_raw_fields(self) -> None:
        """SCENARIO_DISCOVERY_OPT_IN_REGIME_PRESERVES_RAW_FIELDS: identical
        inputs with the flag False vs True yield bit-identical raw admission-path
        fields, while the True call additionally populates the regime-scaled
        tables covering every candidate horizon; the two opt-in diagnostics
        (HAC-adjusted and regime-scaled) are fully independent and composable."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        off = _run(log_close, opens, bar_funding, eligible, idx)
        on = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            compute_regime_scaled_net_t=True,
        )
        assert on.selected_horizon == off.selected_horizon
        assert on.admitted == off.admitted
        assert on.discovery_scores == off.discovery_scores
        assert on.discovery_aggregate_net_t == off.discovery_aggregate_net_t
        assert on.qualification_net_t == off.qualification_net_t
        assert on.qualification_sign_consistent == off.qualification_sign_consistent
        assert on.yearly_net_t == off.yearly_net_t
        assert on.yearly_regime_scaled_net_t != ()
        assert set(dict(on.yearly_regime_scaled_net_t)) == {24, 48}
        assert set(dict(on.discovery_scores_regime_scaled)) == {24, 48}

        adjusted_only = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            compute_adjusted_net_t=True,
        )
        assert adjusted_only.yearly_regime_scaled_net_t == ()
        assert adjusted_only.discovery_scores_regime_scaled == ()
        assert adjusted_only.yearly_adjusted_net_t != ()
        assert adjusted_only.adjusted_admitted is not None

        combo = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            compute_adjusted_net_t=True, compute_regime_scaled_net_t=True,
        )
        assert combo.yearly_regime_scaled_net_t != ()
        assert combo.yearly_adjusted_net_t != ()
        assert combo.adjusted_admitted == adjusted_only.adjusted_admitted
        assert combo.regime_scaled_admitted == on.regime_scaled_admitted

    def test_regime_scaled_admitted_sign_invariant(self) -> None:
        """SCENARIO_REGIME_SCALED_ADMITTED_SIGN_INVARIANT: ``regime_scaled_admitted``
        is True only when the RAW sign-consistency flag is True AND
        ``abs(qualification_regime_scaled_net_t) >= admission_t``, and it never
        influences ``admitted``/``selected_horizon``."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        on = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
            qualification_end=QUALIFICATION_END, min_symbols=3, tranche_count=1,
            compute_regime_scaled_net_t=True,
        )
        assert on.selected_horizon is not None
        assert on.qualification_regime_scaled_net_t is not None
        assert math.isfinite(on.qualification_regime_scaled_net_t)
        if on.regime_scaled_admitted:
            assert on.qualification_sign_consistent is True
            assert abs(on.qualification_regime_scaled_net_t) >= 2.0
        off = _run(log_close, opens, bar_funding, eligible, idx)
        assert on.selected_horizon == off.selected_horizon
        assert on.admitted == off.admitted


class TestFoldTrainOnlyDiscoveryQualification:
    """SCENARIO_MHS_FOLD_SAFE_HORIZON_01..03: the fold-scoped wrapper derives
    leak-free discovery/qualification bounds from one ``AnchoredPurgedFold`` and
    fails closed when the train window leaves no room for a disjoint split.
    """

    def test_insufficient_train_window_fails_closed_without_delegation(self, monkeypatch) -> None:
        """SCENARIO_MHS_FOLD_SAFE_HORIZON_01_INSUFFICIENT_TRAIN_WINDOW: a fold
        whose train window spans a single calendar year
        (``fold.train_end.year == fold.train_start.year``) leaves no room for a
        disjoint qualification split, so the wrapper returns the fail-closed
        result without calling into ``select_horizon_by_discovery_qualification``
        (verified by a call-count 0 monkeypatch)."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(24)
        fold = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-12-31", tz="UTC"),
            pd.Timestamp("2022-01-08", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            168, 168,
        )
        calls = {"n": 0}
        real = discovery.select_horizon_by_discovery_qualification

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(discovery, "select_horizon_by_discovery_qualification", counting)
        result = fold_train_only_discovery_qualification(
            1, (72, 96), log_close, eligible, opens, bar_funding, idx, fold,
        )
        assert calls["n"] == 0
        assert result.admitted is False
        assert result.selected_horizon is None
        assert result.discovery_scores == ()
        assert result.discovery_aggregate_net_t is None
        assert result.qualification_net_t is None
        assert result.qualification_sign_consistent is None

    def test_delegates_with_derived_bounds(self, monkeypatch) -> None:
        """SCENARIO_MHS_FOLD_SAFE_HORIZON_02_DELEGATES_WITH_DERIVED_BOUNDS: for
        a fold with train_start=2021-01-01 and train_end=2022-12-31 the wrapper
        calls the underlying selection with discovery_start=2021-01-01,
        discovery_end=2021-12-31 23:59:59.999999 and qualification_end=2022-12-31
        (captured via a forwarding monkeypatch), and the returned result is
        byte-identical to calling the selection directly with those bounds on
        the same panel."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        fold = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168, 168,
        )
        captured: dict = {}
        real = discovery.select_horizon_by_discovery_qualification

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(discovery, "select_horizon_by_discovery_qualification", spy)
        result = fold_train_only_discovery_qualification(
            1, (24, 48), log_close, eligible, opens, bar_funding, idx, fold,
            min_symbols=3, tranche_count=1,
        )
        assert captured["discovery_start"] == pd.Timestamp("2021-01-01", tz="UTC")
        assert captured["discovery_end"] == pd.Timestamp("2021-12-31 23:59:59.999999", tz="UTC")
        assert captured["qualification_end"] == pd.Timestamp("2022-12-31", tz="UTC")
        assert captured["sign"] == 1
        assert captured["horizon_candidates"] == (24, 48)
        assert captured["min_symbols"] == 3
        assert captured["tranche_count"] == 1
        expected = select_horizon_by_discovery_qualification(
            sign=1, horizon_candidates=(24, 48), log_close=log_close,
            eligible=eligible, opens=opens, bar_funding=bar_funding, grid_1h=idx,
            discovery_start=pd.Timestamp("2021-01-01", tz="UTC"),
            discovery_end=pd.Timestamp("2021-12-31 23:59:59.999999", tz="UTC"),
            qualification_end=pd.Timestamp("2022-12-31", tz="UTC"),
            min_symbols=3, tranche_count=1,
        )
        assert result == expected

    def test_fold_wrapper_forwards_cache_and_short_circuit_ignores_it(self, monkeypatch) -> None:
        """SCENARIO_MHS_HORIZON_SEARCH_EFF_04_FOLD_WRAPPER_FORWARDS_CACHE: the
        fold-scoped wrapper forwards the identical cache object into its
        ``select_horizon_by_discovery_qualification`` delegation (captured via a
        forwarding spy), the delegated result is byte-identical to the cache-free
        call, and the insufficient-train-window short-circuit returns without
        ever consulting the cache or calling into the selection."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(24)
        cache = discovery.build_candidate_weights(log_close, eligible, 1, (24, 48), 3, 1)
        fold = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168, 168,
        )
        captured: dict = {}
        real = discovery.select_horizon_by_discovery_qualification

        def spy(*args, **kwargs):
            captured["precomputed_candidate_weights"] = kwargs.get("precomputed_candidate_weights")
            return real(*args, **kwargs)

        monkeypatch.setattr(discovery, "select_horizon_by_discovery_qualification", spy)
        result = fold_train_only_discovery_qualification(
            1, (24, 48), log_close, eligible, opens, bar_funding, idx, fold,
            min_symbols=3, tranche_count=1, precomputed_candidate_weights=cache,
        )
        assert captured["precomputed_candidate_weights"] is cache
        expected = fold_train_only_discovery_qualification(
            1, (24, 48), log_close, eligible, opens, bar_funding, idx, fold,
            min_symbols=3, tranche_count=1,
        )
        assert result == expected

        short = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-12-31", tz="UTC"),
            pd.Timestamp("2022-01-08", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            168, 168,
        )
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(discovery, "select_horizon_by_discovery_qualification", counting)
        short_result = fold_train_only_discovery_qualification(
            1, (72, 96), log_close, eligible, opens, bar_funding, idx, short,
            precomputed_candidate_weights=cache,
        )
        assert calls["n"] == 0
        assert short_result.admitted is False
        assert short_result.selected_horizon is None
        assert short_result.discovery_scores == ()

    def test_nonfinite_discovery_year_fails_closed(self) -> None:
        """SCENARIO_MHS_FOLD_SAFE_HORIZON_03_NONFINITE_DISCOVERY_YEAR_FAILS_CLOSED:
        when every discovery-window yearly score is non-finite (here the
        discovery sub-window has < ``min_symbols`` eligible symbols, reproducing
        the measured 2021-coverage-gap case), the fold-scoped wrapper returns
        ``admitted=False``/``selected_horizon=None`` through the delegated
        fail-closed path -- no candidate can clear the admission floor."""
        log_close, opens, bar_funding, eligible, idx = _build_panel(24)
        eligible = eligible.copy()
        eligible.loc[idx.year == 2021, :] = False
        fold = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168, 168,
        )
        result = fold_train_only_discovery_qualification(
            1, (24, 48), log_close, eligible, opens, bar_funding, idx, fold,
            min_symbols=3, tranche_count=1,
        )
        assert result.admitted is False
        assert result.selected_horizon is None

    def test_fold_train_only_passthrough(self, monkeypatch) -> None:
        # compute_adjusted_net_t=True is forwarded into delegated selection.
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        fold = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168, 168,
        )
        captured: dict = {}
        real = discovery.select_horizon_by_discovery_qualification

        def spy(*args, **kwargs):
            captured["compute_adjusted_net_t"] = kwargs.get("compute_adjusted_net_t")
            return real(*args, **kwargs)

        monkeypatch.setattr(discovery, "select_horizon_by_discovery_qualification", spy)
        result = fold_train_only_discovery_qualification(
            1, (24, 48), log_close, eligible, opens, bar_funding, idx, fold,
            min_symbols=3, tranche_count=1, compute_adjusted_net_t=True,
        )
        assert captured["compute_adjusted_net_t"] is True
        assert result.yearly_adjusted_net_t != ()

        short = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-12-31", tz="UTC"),
            pd.Timestamp("2022-01-08", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            168, 168,
        )
        short_result = fold_train_only_discovery_qualification(
            1, (72, 96), log_close, eligible, opens, bar_funding, idx, short,
            min_symbols=3, tranche_count=1, compute_adjusted_net_t=True,
        )
        assert short_result.yearly_adjusted_net_t == ()
        assert short_result.discovery_scores_adjusted == ()
        assert short_result.discovery_aggregate_adjusted_net_t is None
        assert short_result.qualification_adjusted_net_t is None
        assert short_result.adjusted_admitted is None

    def test_fold_train_only_regime_passthrough(self, monkeypatch) -> None:
        # compute_regime_scaled_net_t=True is forwarded into delegated selection.
        log_close, opens, bar_funding, eligible, idx = _build_panel(
            24, phi=0.85, k1=2.0, k2=0.2, kq=1.0,
        )
        fold = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168, 168,
        )
        captured: dict = {}
        real = discovery.select_horizon_by_discovery_qualification

        def spy(*args, **kwargs):
            captured["compute_regime_scaled_net_t"] = kwargs.get("compute_regime_scaled_net_t")
            return real(*args, **kwargs)

        monkeypatch.setattr(discovery, "select_horizon_by_discovery_qualification", spy)
        result = fold_train_only_discovery_qualification(
            1, (24, 48), log_close, eligible, opens, bar_funding, idx, fold,
            min_symbols=3, tranche_count=1, compute_regime_scaled_net_t=True,
        )
        assert captured["compute_regime_scaled_net_t"] is True
        assert result.yearly_regime_scaled_net_t != ()

        short = AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-12-31", tz="UTC"),
            pd.Timestamp("2022-01-08", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            168, 168,
        )
        short_result = fold_train_only_discovery_qualification(
            1, (72, 96), log_close, eligible, opens, bar_funding, idx, short,
            min_symbols=3, tranche_count=1, compute_regime_scaled_net_t=True,
        )
        assert short_result.yearly_regime_scaled_net_t == ()
        assert short_result.discovery_scores_regime_scaled == ()
        assert short_result.discovery_aggregate_regime_scaled_net_t is None
        assert short_result.qualification_regime_scaled_net_t is None
        assert short_result.regime_scaled_admitted is None
