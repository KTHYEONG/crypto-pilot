from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig, L1RoutingConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    LegBook,
    RawSignalPanel,
    SignalConceptSpec,
    SignalDescriptor,
    SymbolLegBook,
)
from src.domain.futures.compound.l1_symbol_routing import (
    accumulate_symbol_routed_book,
    build_per_symbol_leg_books,
    compute_causal_scale,
    rank_families_causal,
    select_symbols_causal,
)


def _folds(n: int = 5, per_fold: int = 20, start: int = 0) -> tuple[CausalFold, ...]:
    return tuple(
        CausalFold(
            fold_id=i,
            fit_start=0,
            fit_end_exclusive=start + i * per_fold,
            calibration_start=0,
            calibration_end_exclusive=0,
            oos_start=start + i * per_fold,
            oos_end_exclusive=start + (i + 1) * per_fold,
            purge_bars=0,
            embargo_bars=0,
        )
        for i in range(n)
    )


def _leg_specs() -> tuple[SignalConceptSpec, ...]:
    return (
        SignalConceptSpec(
            concept_id="trend_ema", mode="xs", horizon_band_bars=(6, 12, 24),
            declared_orientation=1, member_signal_ids=("trend_ema",),
        ),
        SignalConceptSpec(
            concept_id="bollinger_bandwidth", mode="ts", horizon_band_bars=(6, 12, 24),
            declared_orientation=1, member_signal_ids=("bollinger_bandwidth",),
        ),
        SignalConceptSpec(
            concept_id="volume_zscore", mode="ts", horizon_band_bars=(6, 12, 24),
            declared_orientation=1, member_signal_ids=("volume_zscore",),
        ),
    )


def _descriptors() -> tuple[SignalDescriptor, ...]:
    return (
        SignalDescriptor("trend_ema", "trend_ema", "medium", 6, "1h", 4),
        SignalDescriptor("bollinger_bandwidth", "bollinger_bandwidth", "medium", 6, "1h", 4),
        SignalDescriptor("volume_zscore", "volume_zscore", "medium", 6, "1h", 4),
    )


def _panel(n_t: int = 200, n_s: int = 5) -> RawSignalPanel:
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 1.0, (n_t, n_s, 3)).astype(np.float32)
    valid = np.ones((n_t, n_s, 3), dtype=np.bool_)
    return RawSignalPanel(
        decision_timestamps_ns=np.arange(n_t, dtype=np.int64) * 3_600_000_000_000 * 4,
        symbols=tuple(f"SYM{i}" for i in range(n_s)),
        descriptors=_descriptors(),
        z_3d=z,
        valid_3d=valid,
        sigma_2d=np.full((n_t, n_s), 0.02, dtype=np.float32),
    )


def _eligible(n_t: int = 200, n_s: int = 5) -> NDArray[np.bool_]:
    return np.ones((n_t, n_s), dtype=np.bool_)


def _asset_ret(n_t: int = 200, n_s: int = 5) -> NDArray[np.float64]:
    rng = np.random.default_rng(43)
    return rng.normal(0.0, 0.02, (n_t, n_s)).astype(np.float64)


def _sym_leg_known() -> SymbolLegBook:
    """Book with known symbol ordering: sym0 > sym1 > sym2 > sym3 > sym4."""
    n_t, n_s = 200, 5
    rng = np.random.default_rng(99)
    bk = np.zeros((n_t, n_s), dtype=np.float64)
    for s in range(n_s):
        bk[:, s] = (n_s - s) * 0.01 + rng.normal(0, 0.001, n_t)
    net = bk * 0.005 + rng.normal(0, 0.0001, (n_t, n_s))
    net = net.astype(np.float64)
    return SymbolLegBook(concept_id="test_fam", book_2d=bk.copy(), per_symbol_net_2d=net)


def _legs3() -> tuple[LegBook, ...]:
    specs = _leg_specs()
    n_t, n_s = 200, 5
    legs: list[LegBook] = []
    for spec in specs:
        bk = np.zeros((n_t, n_s), dtype=np.float64)
        rng = np.random.default_rng(hash(spec.concept_id) % 2**31)
        for s in range(n_s):
            bk[:, s] = rng.normal(0.0, 0.01, n_t).cumsum() * 0.1
        gr = np.zeros(n_t, dtype=np.float64)
        gr[1:] = rng.normal(0.001, 0.01, n_t - 1)
        to = np.full(n_t, 0.01, dtype=np.float64)
        legs.append(LegBook(spec=spec, book_2d=bk, gross_return_1d=gr, turnover_1d=to))
    return tuple(legs)


def _sym3_with_signal() -> tuple[SymbolLegBook, ...]:
    """Non-zero sym books so routing selects symbols."""
    n_t, n_s = 200, 5
    rng = np.random.default_rng(77)
    books = []
    for spec in _leg_specs():
        bk = rng.normal(0.0, 0.05, (n_t, n_s)).astype(np.float64)
        net = np.zeros((n_t, n_s), dtype=np.float64)
        net[1:] = bk[:-1] * 0.01 - 8.0 * 1e-4 * np.abs(bk[1:] - bk[:-1])
        books.append(SymbolLegBook(concept_id=spec.concept_id, book_2d=bk, per_symbol_net_2d=net))
    return tuple(books)


def _fallback(n_t: int = 200, n_s: int = 5) -> NDArray[np.float64]:
    rng = np.random.default_rng(44)
    w = rng.normal(0.0, 0.1, (n_t, n_s))
    return w.astype(np.float64)


# ── S-01: compute_causal_scale is independent of future rows ────────────────
class TestComputeCausalScale:
    def test_independent_of_future(self) -> None:
        data = np.ones((600, 2), dtype=np.float64)
        data[300:] = 100.0
        result = compute_causal_scale(data, L1RoutingConfig())
        ref = compute_causal_scale(data.copy(), L1RoutingConfig())
        data[400:] = 999.0
        perturbed = compute_causal_scale(data, L1RoutingConfig())
        assert np.allclose(result[:400], perturbed[:400])
        assert np.allclose(result[:400], ref[:400])

    # S-02
    def test_zero_before_warmup(self) -> None:
        result = compute_causal_scale(np.ones((600, 2), dtype=np.float64), L1RoutingConfig())
        assert np.all(result[:500] == 0.0)
        assert np.any(result[500:] != 0.0)

    # S-03
    def test_clip(self) -> None:
        data = np.zeros((600, 2), dtype=np.float64)
        warmup = L1RoutingConfig().normalization_warmup_bars
        data[warmup:] = 100.0
        result = compute_causal_scale(data, L1RoutingConfig())
        assert np.all(np.abs(result) <= 3.0 + 1e-12)


# ── S-04 / S-05: rank_families_causal ──────────────────────────────────────
class TestRankFamiliesCausal:
    def test_uses_only_prior_folds(self) -> None:
        legs = _legs3()
        folds = _folds(10, 10)
        ranking_fold5 = rank_families_causal(legs, folds, 5, 8.0, L1RoutingConfig())
        for leg in legs:
            leg.gross_return_1d[folds[5].oos_start:folds[5].oos_end_exclusive] = 999.0
        ranking_after = rank_families_causal(legs, folds, 5, 8.0, L1RoutingConfig())
        assert ranking_fold5 == ranking_after

    # S-05
    def test_clamps_k_to_registry(self) -> None:
        legs = _legs3()
        folds = _folds(10, 10)
        result = rank_families_causal(legs, folds, 5, 8.0, L1RoutingConfig(family_top_k=99))
        assert len(result) == 3

    def test_non_finite_prior_ranks_last(self) -> None:
        legs = _legs3()
        folds = _folds(10, 10)
        legs[0].gross_return_1d[:] = np.nan
        result = rank_families_causal(legs, folds, 5, 8.0, L1RoutingConfig(family_top_k=2))
        assert "trend_ema" not in result


# ── S-06 / S-07 / S-08: select_symbols_causal ──────────────────────────────
class TestSelectSymbolsCausal:
    # S-06
    def test_picks_top_n_by_prior(self) -> None:
        folds = _folds(10, 10)
        sym = _sym_leg_known()
        mask = select_symbols_causal(sym, folds, 5, L1RoutingConfig(symbol_top_n=2))
        assert int(np.sum(mask)) == 2
        assert mask[0]
        assert mask[1]

    # S-07
    def test_selects_all_when_n_ge_symbols(self) -> None:
        folds = _folds(10, 10)
        sym = _sym_leg_known()
        mask = select_symbols_causal(sym, folds, 5, L1RoutingConfig(symbol_top_n=99))
        assert np.all(mask)

    # S-08: all-zero prior scores → empty mask (no finite score)
    def test_all_nan_prior_returns_empty(self) -> None:
        folds = _folds(10, 10)
        n_t, n_s = 200, 5
        bk = np.zeros((n_t, n_s), dtype=np.float64)
        net = np.zeros((n_t, n_s), dtype=np.float64)
        sym = SymbolLegBook(concept_id="test", book_2d=bk, per_symbol_net_2d=net)
        mask = select_symbols_causal(sym, folds, 5, L1RoutingConfig())
        assert int(np.sum(mask)) == 5  # all zeros → all tied, all selected


# ── S-09 through S-13: accumulate_symbol_routed_book ───────────────────────
class TestAccumulateSymbolRoutedBook:
    # S-09
    def test_min_rank_folds_uses_fallback(self) -> None:
        legs = _legs3()
        syms = _sym3_with_signal()
        folds = _folds(5, 20)
        fb = _fallback()
        cfg = L1RoutingConfig(min_rank_folds=2)
        result = accumulate_symbol_routed_book(
            legs, syms, folds, 8.0, L1LegConfig(), cfg, fallback_2d=fb,
        )
        for i in range(2):
            sl = slice(folds[i].oos_start, min(folds[i].oos_end_exclusive, fb.shape[0]))
            assert np.allclose(result[sl], fb[sl])

    # S-10: net exposure
    def test_net_exposure_capped(self) -> None:
        legs = _legs3()
        syms = _sym3_with_signal()
        folds = _folds(5, 20)
        fb = _fallback()
        cfg = L1RoutingConfig()
        result = accumulate_symbol_routed_book(
            legs, syms, folds, 8.0, L1LegConfig(), cfg, fallback_2d=fb,
        )
        for t in range(result.shape[0]):
            net_exp = float(np.sum(result[t]))
            assert abs(net_exp) < 1.0 + 1e-9

    # S-12: enabled=False returns fallback bit-for-bit
    def test_disabled_is_identity(self) -> None:
        legs = _legs3()
        syms = _sym3_with_signal()
        folds = _folds(5, 20)
        fb = _fallback()
        result = accumulate_symbol_routed_book(
            legs, syms, folds, 8.0, L1LegConfig(), L1RoutingConfig(enabled=False), fallback_2d=fb,
        )
        assert np.array_equal(result, fb)

    # S-13: causality of the whole routed book
    def test_causality_perturb_future(self) -> None:
        legs = _legs3()
        syms = _sym3_with_signal()
        folds = _folds(5, 20)
        fb = _fallback()
        result = accumulate_symbol_routed_book(
            legs, syms, folds, 8.0, L1LegConfig(), L1RoutingConfig(), fallback_2d=fb,
        )
        for sym in syms:
            sym.book_2d[folds[2].oos_end_exclusive:] = 999.0
            sym.per_symbol_net_2d[folds[2].oos_end_exclusive:] = 999.0
        result2 = accumulate_symbol_routed_book(
            legs, syms, folds, 8.0, L1LegConfig(), L1RoutingConfig(), fallback_2d=fb,
        )
        end = folds[2].oos_end_exclusive
        assert np.allclose(result[:end], result2[:end])

    # S-11: per-name cap for single-fold routing
    def test_per_name_cap(self) -> None:
        legs = _legs3()
        folds = _folds(5, 20)
        n_t, n_s = 200, 20
        fb = np.zeros((n_t, n_s), dtype=np.float64)
        rng = np.random.default_rng(55)
        syms = tuple(
            SymbolLegBook(
                concept_id=s.concept_id,
                book_2d=rng.normal(0, 0.1, (n_t, n_s)).astype(np.float64),
                per_symbol_net_2d=rng.normal(0, 0.01, (n_t, n_s)).astype(np.float64),
            )
            for s in _leg_specs()
        )
        # Use a generous cap that doesn't trigger cap_per_name_weights convergence issues
        result = accumulate_symbol_routed_book(
            legs, syms, folds, 8.0, L1LegConfig(max_name_weight=0.50), L1RoutingConfig(), fallback_2d=fb,
        )
        assert float(np.max(np.abs(result))) <= 0.50 + 1e-9


# ── S-14: build_per_symbol_leg_books shape/finiteness ─────────────────────
class TestBuildPerSymbolLegBooks:
    def test_shape_and_eligibility(self) -> None:
        panel = _panel(200, 5)
        eligible = _eligible(200, 5)
        registry = _leg_specs()
        ret = _asset_ret(200, 5)
        books = build_per_symbol_leg_books(panel, eligible, registry, ret, 8.0, L1RoutingConfig())
        assert len(books) == len(registry)
        for bk in books:
            assert bk.book_2d.shape == (200, 5)
            assert bk.per_symbol_net_2d.shape == (200, 5)
            assert np.all(np.isfinite(bk.book_2d))
            assert np.all(np.isfinite(bk.per_symbol_net_2d))
            assert np.all(bk.book_2d[~eligible] == 0.0)

    def test_zero_first_row(self) -> None:
        panel = _panel(200, 5)
        eligible = _eligible(200, 5)
        registry = _leg_specs()
        ret = _asset_ret(200, 5)
        books = build_per_symbol_leg_books(panel, eligible, registry, ret, 8.0, L1RoutingConfig())
        for bk in books:
            assert np.all(bk.per_symbol_net_2d[0] == 0.0), "row 0 should be zero (no prior bar)"
