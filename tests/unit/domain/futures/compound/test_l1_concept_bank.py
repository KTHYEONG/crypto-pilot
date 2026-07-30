from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import LegBook, RawSignalPanel, SignalConceptSpec, SignalDescriptor
from src.domain.futures.compound.l1_concept_bank import (
    build_concept_registry,
    build_leg_books,
    build_tranche_book,
    build_tranche_target,
)


def _make_panel(
    T: int = 10, S: int = 5, families: tuple[str, ...] = ("family_0", "family_1"),
) -> RawSignalPanel:
    rng = np.random.default_rng(42)
    n_sig = len(families)
    z = rng.standard_normal((T, S, n_sig)).astype(np.float32)
    valid = np.ones((T, S, n_sig), dtype=np.bool_)
    descs = tuple(
        SignalDescriptor(
            signal_id=f"{fam}:fast", family=fam, speed="fast",
            lookback_hours=24, native_timeframe="4h", declared_orientation=1,
        )
        for fam in families
    )
    ts = np.arange(T, dtype=np.int64) * 3600 * 4 * 10**9
    return RawSignalPanel(
        decision_timestamps_ns=ts, symbols=tuple(f"SYM{i}" for i in range(S)),
        descriptors=descs, z_3d=z, valid_3d=valid,
        sigma_2d=np.full((T, S), 0.01, dtype=np.float32),
    )


def _make_close(T: int = 10, S: int = 5) -> NDArray[np.float32]:
    rng = np.random.default_rng(42)
    close = np.full((T, S), 100.0, dtype=np.float32)
    for t in range(1, T):
        close[t] = close[t - 1] * (1.0 + rng.standard_normal(S).astype(np.float32) * 0.01)
    return close


class TestBuildTrancheBook:
    def test_build_tranche_book_happy_path(self) -> None:
        v = np.arange(1.0, 11.0).reshape(-1, 1)
        v = np.broadcast_to(v, (10, 3)).copy().astype(np.float64)
        book = build_tranche_book(v, 3)
        assert book.shape == (10, 3)
        assert np.allclose(book[0], v[0])
        assert np.allclose(book[1], (v[0] + v[1]) / 2)
        assert np.allclose(book[2], (v[0] + v[1] + v[2]) / 3)
        book1 = build_tranche_book(v, 1)
        assert np.allclose(book1, v)

    def test_build_tranche_book_turnover_bound(self) -> None:
        T, S = 100, 5
        rng = np.random.default_rng(42)
        v = rng.standard_normal((T, S)).astype(np.float64)
        v = v / np.maximum(np.sum(np.abs(v), axis=1, keepdims=True), 1e-12)
        h = 12
        book = build_tranche_book(v, h)
        turnover = np.sum(np.abs(np.diff(book, axis=0)), axis=1)
        max_turn = float(np.max(turnover[h:]))
        assert max_turn <= 2.0 / h + 1e-2


class TestBuildTrancheTarget:
    def test_build_tranche_target_mode_semantics(self) -> None:
        T, S = 5, 10
        rng = np.random.default_rng(42)
        z = rng.standard_normal((T, S)).astype(np.float32)
        eligible = np.ones((T, S), dtype=np.bool_)
        v_xs = build_tranche_target(z, eligible, "xs", min_cross_section=3)
        for t in range(T):
            assert abs(float(np.sum(v_xs[t]))) < 1e-10
            assert abs(float(np.sum(np.abs(v_xs[t]))) - 1.0) < 1e-10
        v_ts = build_tranche_target(z, eligible, "ts", min_cross_section=3)
        for t in range(T):
            assert abs(float(np.sum(np.abs(v_ts[t]))) - 1.0) < 1e-10
            assert np.array_equal(np.sign(v_ts[t]), np.sign(z[t].astype(np.float64)))

    def test_build_tranche_target_rejects_invalid_mode(self) -> None:
        z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        eligible = np.ones((1, 3), dtype=np.bool_)
        with pytest.raises(ValueError, match="mode must be 'xs' or 'ts'"):
            build_tranche_target(z, eligible, "diagonal", min_cross_section=3)

    def test_build_tranche_target_breadth_floor(self) -> None:
        T, S = 5, 10
        z = np.ones((T, S), dtype=np.float32)
        eligible = np.zeros((T, S), dtype=np.bool_)
        eligible[:, :2] = True
        v = build_tranche_target(z, eligible, "xs", min_cross_section=5)
        assert np.allclose(v, 0.0)


class TestBuildLegBooks:
    def test_build_leg_books_unavailable_feature_is_zero_not_raise(self) -> None:
        T, S = 10, 5
        panel = _make_panel(T, S, families=("family_0",))
        eligible = np.ones((T, S), dtype=np.bool_)
        close = _make_close(T, S)
        spec = SignalConceptSpec(
            concept_id="missing", mode="xs", horizon_band_bars=(6,),
            declared_orientation=1, member_signal_ids=("nonexistent",),
        )
        cfg = L1LegConfig(min_cross_section=2)
        legs = build_leg_books(panel, eligible, close, (spec,), cfg)
        assert len(legs) == 1
        assert np.allclose(legs[0].book_2d, 0.0)

    def test_build_leg_books_default_registry_produces_nonzero_signal(self) -> None:
        """Regression guard: member_signal_ids in the default registry are FAMILY
        names ("rsi"), not full signal_id ("rsi:fast"). A key mismatch here
        silently zeros every leg forever -- this must be caught by content,
        not shape, assertions."""
        T, S = 60, 8
        panel = _make_panel(T, S, families=("rsi", "trend_ema", "volume_zscore"))
        eligible = np.ones((T, S), dtype=np.bool_)
        close = _make_close(T, S)
        cfg = L1LegConfig(min_cross_section=2)
        registry = build_concept_registry(panel.descriptors, cfg)
        legs = build_leg_books(panel, eligible, close, registry, cfg)
        assert len(legs) == len(registry)
        trend_momentum_leg = next(leg for leg in legs if leg.spec.concept_id == "trend_momentum")
        assert np.any(trend_momentum_leg.book_2d != 0.0), (
            "trend_momentum book is all-zero even though rsi/trend_ema families "
            "are present in the panel -- member lookup is not matching by family"
        )
        vol_regime_leg = next(leg for leg in legs if leg.spec.concept_id == "vol_regime")
        assert np.any(vol_regime_leg.book_2d != 0.0), (
            "vol_regime book is all-zero even though volume_zscore family "
            "is present in the panel -- member lookup is not matching by family"
        )

    def test_build_leg_books_band_averages_full_horizon_band(self) -> None:
        """Regression guard: the book must be the average over every horizon in
        horizon_band_bars, not just horizon_band_bars[0]."""
        T, S = 60, 8
        panel = _make_panel(T, S, families=("family_0",))
        eligible = np.ones((T, S), dtype=np.bool_)
        close = _make_close(T, S)
        cfg = L1LegConfig(min_cross_section=2)
        spec_one_horizon = SignalConceptSpec(
            concept_id="c", mode="xs", horizon_band_bars=(6,),
            declared_orientation=1, member_signal_ids=("family_0",),
        )
        spec_full_band = SignalConceptSpec(
            concept_id="c", mode="xs", horizon_band_bars=(6, 12, 24),
            declared_orientation=1, member_signal_ids=("family_0",),
        )
        leg_one = build_leg_books(panel, eligible, close, (spec_one_horizon,), cfg)[0]
        leg_band = build_leg_books(panel, eligible, close, (spec_full_band,), cfg)[0]
        assert not np.allclose(leg_one.book_2d, leg_band.book_2d), (
            "single-horizon and full-band books are identical -- "
            "horizon_band_bars[1:] is being silently dropped"
        )


class TestBuildConceptRegistry:
    def test_build_concept_registry_matches_frozen_spec(self) -> None:
        cfg = L1LegConfig()
        registry = build_concept_registry((), cfg)
        assert len(registry) == 2
        ids = tuple(s.concept_id for s in registry)
        assert "trend_momentum" in ids
        assert "vol_regime" in ids
        tm = next(s for s in registry if s.concept_id == "trend_momentum")
        assert tm.mode == "xs"
        assert len(tm.member_signal_ids) == 10
        assert "rsi" in tm.member_signal_ids
        vr = next(s for s in registry if s.concept_id == "vol_regime")
        assert vr.mode == "ts"
        assert len(vr.member_signal_ids) == 3
        assert "volume_zscore" in vr.member_signal_ids


class TestL1LegPanel:
    """Contract-level coverage for L1LegPanel.__post_init__ -- this dataclass
    is the formal L1->L2 multi-leg interface; even though engine.py currently
    threads legs/weights through as loose values rather than assembling this
    object, its validation must be exercised directly."""

    @staticmethod
    def _valid_kwargs(T: int = 8, S: int = 3, K: int = 2):
        from src.domain.futures.compound.l1_leg_evaluation import evaluate_leg_alpha
        specs = tuple(
            SignalConceptSpec(
                concept_id=f"c{k}", member_signal_ids=("m",),
                mode="xs", horizon_band_bars=(6,), declared_orientation=1,
            )
            for k in range(K)
        )
        evidence = tuple(
            evaluate_leg_alpha(
                LegBook(
                    spec=specs[k],
                    book_2d=np.zeros((T, S), dtype=np.float64),
                    gross_return_1d=np.zeros(T, dtype=np.float64),
                    turnover_1d=np.zeros(T, dtype=np.float64),
                ),
                np.zeros(T, dtype=np.float64), (), 8.0, L1LegConfig(),
            )
            for k in range(K)
        )
        return {
            "decision_timestamps_ns": np.arange(T, dtype=np.int64),
            "symbols": tuple(f"S{i}" for i in range(S)),
            "leg_specs": specs,
            "books_3d": np.zeros((T, S, K), dtype=np.float32),
            "leg_weights_2d": np.zeros((T, K), dtype=np.float64),
            "combined_weights_2d": np.zeros((T, S), dtype=np.float64),
            "evidence": evidence,
            "admitted": False,
            "reasons": ("no_leg_evidence",),
        }

    def test_l1_leg_panel_valid_construction(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        panel = L1LegPanel(**self._valid_kwargs())
        assert panel.books_3d.shape == (8, 3, 2)
        assert panel.leg_weights_2d.shape == (8, 2)

    def test_l1_leg_panel_rejects_books_3d_shape_mismatch(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        kwargs = self._valid_kwargs()
        kwargs["books_3d"] = np.zeros((8, 3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="books_3d shape"):
            L1LegPanel(**kwargs)

    def test_l1_leg_panel_rejects_leg_weights_shape_mismatch(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        kwargs = self._valid_kwargs()
        kwargs["leg_weights_2d"] = np.zeros((8, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="leg_weights_2d shape"):
            L1LegPanel(**kwargs)

    def test_l1_leg_panel_rejects_combined_weights_shape_mismatch(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        kwargs = self._valid_kwargs()
        kwargs["combined_weights_2d"] = np.zeros((8, 9), dtype=np.float64)
        with pytest.raises(ValueError, match="combined_weights_2d shape"):
            L1LegPanel(**kwargs)

    def test_l1_leg_panel_rejects_evidence_length_mismatch(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        kwargs = self._valid_kwargs()
        kwargs["evidence"] = kwargs["evidence"][:1]
        with pytest.raises(ValueError, match="evidence length"):
            L1LegPanel(**kwargs)

    def test_l1_leg_panel_rejects_empty_symbols_or_specs(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        kwargs = self._valid_kwargs()
        kwargs["symbols"] = ()
        with pytest.raises(ValueError, match="symbols must be non-empty"):
            L1LegPanel(**kwargs)
        kwargs2 = self._valid_kwargs()
        kwargs2["leg_specs"] = ()
        with pytest.raises(ValueError, match="leg_specs must be non-empty"):
            L1LegPanel(**kwargs2)
        kwargs3 = self._valid_kwargs()
        kwargs3["decision_timestamps_ns"] = np.zeros((8, 1), dtype=np.int64)
        with pytest.raises(ValueError, match="decision_timestamps_ns must be 1-D"):
            L1LegPanel(**kwargs3)

    def test_l1_leg_panel_rejects_admitted_reasons_inconsistency(self) -> None:
        from src.domain.futures.compound.contracts import L1LegPanel
        kwargs = self._valid_kwargs()
        kwargs["admitted"] = True
        kwargs["reasons"] = ("should_be_empty",)
        with pytest.raises(ValueError, match="admitted panel must have empty reasons"):
            L1LegPanel(**kwargs)
        kwargs["admitted"] = False
        kwargs["reasons"] = ()
        with pytest.raises(ValueError, match="not-admitted panel must have at least one reason"):
            L1LegPanel(**kwargs)


class TestLegBookValidation:
    @staticmethod
    def _valid_kwargs(T: int = 5, S: int = 3) -> dict:
        spec = SignalConceptSpec(
            concept_id="t", member_signal_ids=("m",),
            mode="xs", horizon_band_bars=(6,), declared_orientation=1,
        )
        return {
            "spec": spec,
            "book_2d": np.zeros((T, S), dtype=np.float64),
            "gross_return_1d": np.zeros(T, dtype=np.float64),
            "turnover_1d": np.zeros(T, dtype=np.float64),
        }

    def test_leg_book_accepts_valid_input(self) -> None:
        book = LegBook(**self._valid_kwargs())
        assert book.book_2d.shape == (5, 3)

    def test_leg_book_rejects_non_2d_book(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["book_2d"] = np.zeros(5, dtype=np.float64)
        with pytest.raises(ValueError, match="book_2d must be 2-D"):
            LegBook(**kwargs)

    def test_leg_book_rejects_gross_return_length_mismatch(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["gross_return_1d"] = np.zeros(7, dtype=np.float64)
        with pytest.raises(ValueError, match="gross_return_1d"):
            LegBook(**kwargs)

    def test_leg_book_rejects_turnover_length_mismatch(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["turnover_1d"] = np.zeros(7, dtype=np.float64)
        with pytest.raises(ValueError, match="turnover_1d must be 1-D"):
            LegBook(**kwargs)

    def test_leg_book_rejects_non_finite_book(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["book_2d"] = np.full((5, 3), np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="book_2d must be finite"):
            LegBook(**kwargs)

    def test_leg_book_rejects_non_finite_gross_return(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["gross_return_1d"] = np.full(5, np.inf, dtype=np.float64)
        with pytest.raises(ValueError, match="gross_return_1d must be finite"):
            LegBook(**kwargs)

    def test_leg_book_rejects_non_finite_turnover(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["turnover_1d"] = np.full(5, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="turnover_1d must be finite"):
            LegBook(**kwargs)


class TestSignalConceptSpecValidation:
    @staticmethod
    def _valid_kwargs() -> dict:
        return {
            "concept_id": "t", "member_signal_ids": ("m",),
            "mode": "xs", "horizon_band_bars": (6,), "declared_orientation": 1,
        }

    @pytest.mark.parametrize(("field", "bad_value", "match"), [
        ("concept_id", "", "concept_id"),
        ("member_signal_ids", (), "member_signal_ids"),
        ("mode", "bad", "mode"),
        ("horizon_band_bars", (), "horizon_band_bars"),
        ("horizon_band_bars", (0,), "horizon_band_bars"),
        ("declared_orientation", 0, "declared_orientation"),
    ])
    def test_signal_concept_spec_rejects_invalid_fields(self, field: str, bad_value: object, match: str) -> None:
        kwargs = {**self._valid_kwargs(), field: bad_value}
        with pytest.raises(ValueError, match=match):
            SignalConceptSpec(**kwargs)
