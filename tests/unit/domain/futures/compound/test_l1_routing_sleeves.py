from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import DynamicCompoundingConfig
from src.domain.futures.compound.contracts import (
    CausalClusterFold,
    CausalFold,
    ClusterPanel,
    FamilyEdgeScreen,
    L1RoutingSleeve,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_regime_routing import (
    ExpertContribution,
    build_fold_expert_books,
    score_expert_returns,
)
from src.domain.futures.compound.l1_sleeves import build_family_routing_sleeves


def _panel(n: int = 100, n_syms: int = 4) -> RawSignalPanel:
    ts = np.arange(n, dtype=np.int64) * 14_400_000_000_000
    syms = tuple(f"S{i}" for i in range(n_syms))
    desc = (
        SignalDescriptor("xs_reversal:fast", "xs_reversal", "fast", 8, "4h", declared_orientation=1),
        SignalDescriptor("xs_reversal:medium", "xs_reversal", "medium", 24, "4h", declared_orientation=1),
    )
    rng = np.random.default_rng(42)
    z = rng.normal(0, 1, (n, n_syms, 2)).astype(np.float32)
    valid = np.ones((n, n_syms, 2), dtype=np.bool_)
    sigma = np.ones((n, n_syms), dtype=np.float32) * 0.02
    return RawSignalPanel(ts, syms, desc, z, valid, sigma)


def _bars(t: int = 100, n: int = 4) -> TimeframeBarCube:
    close = np.column_stack([np.linspace(100.0, 120.0 + i, t) for i in range(n)]).astype(np.float32)
    return TimeframeBarCube(
        "4h", np.arange(t, dtype=np.int64), tuple(f"S{i}" for i in range(n)),
        close * 0.9995, close * 1.001, close * 0.9985, close,
        np.ones((t, n), dtype=np.float32) * 1e6, np.ones((t, n), dtype=np.bool_),
    )


def _cluster_folds(n_syms: int = 4) -> tuple[CausalClusterFold, ...]:
    cp = ClusterPanel(
        symbols=tuple(f"S{i}" for i in range(n_syms)),
        cluster_labels=np.array([0, 0, 1, 1][:n_syms], dtype=np.int32),
        cluster_centroids=np.zeros((2, 4), dtype=np.float64),
        k_clusters=2,
    )
    return (
        CausalClusterFold(fold_id=0, fit_end_exclusive_4h=20, fit_end_time_ns=20 * 14_400_000_000_000, panel=cp, member_hash="h0"),
        CausalClusterFold(fold_id=1, fit_end_exclusive_4h=40, fit_end_time_ns=40 * 14_400_000_000_000, panel=cp, member_hash="h1"),
    )


def _folds() -> tuple[CausalFold, ...]:
    return (
        CausalFold(0, 0, 20, 0, 20, 20, 40, 2, 42),
        CausalFold(1, 0, 40, 0, 40, 40, 60, 2, 42),
    )


# ---- Scenario 1: L1RoutingSleeve validation ----

class TestL1RoutingSleeveValidation:

    def test_valid_sleeve_creates_ok(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        sleeve = L1RoutingSleeve("s1:f0:c0", "sig1", "fam1", 0, 0, mask, "hash1", 1)
        assert sleeve.sleeve_id == "s1:f0:c0"
        assert sleeve.declared_orientation == 1

    def test_empty_id_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match=r"sleeve_id.*must be non-empty"):
            L1RoutingSleeve("", "sig1", "fam1", 0, 0, mask, "hash1", 1)

    def test_empty_signal_id_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match=r"signal_id.*must be non-empty"):
            L1RoutingSleeve("s1", "", "fam1", 0, 0, mask, "hash1", 1)

    def test_empty_family_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match=r"family.*must be non-empty"):
            L1RoutingSleeve("s1", "sig1", "", 0, 0, mask, "hash1", 1)

    def test_negative_fold_id_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match=r"outer_fold_id.*>= 0"):
            L1RoutingSleeve("s1", "sig1", "fam1", -1, 0, mask, "hash1", 1)

    def test_negative_cluster_id_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match=r"cluster_id.*>= 0"):
            L1RoutingSleeve("s1", "sig1", "fam1", 0, -1, mask, "hash1", 1)

    def test_non_bool_mask_raises(self) -> None:
        mask = np.array([1, 1, 0, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="member_mask_1d must be 1-D bool"):
            L1RoutingSleeve("s1", "sig1", "fam1", 0, 0, mask, "hash1", 1)

    def test_single_member_raises(self) -> None:
        mask = np.array([True, False, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match="at least two members"):
            L1RoutingSleeve("s1", "sig1", "fam1", 0, 0, mask, "hash1", 1)

    def test_invalid_orientation_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match=r"declared_orientation.*-1 or 1"):
            L1RoutingSleeve("s1", "sig1", "fam1", 0, 0, mask, "hash1", 0)

    def test_empty_hash_raises(self) -> None:
        mask = np.array([True, True, False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match="member_hash must be non-empty"):
            L1RoutingSleeve("s1", "sig1", "fam1", 0, 0, mask, "", 1)


# ---- Scenario 2: build_family_routing_sleeves ----

class TestBuildFamilyRoutingSleeves:

    def test_admitted_ids_create_exact_fold_cluster_sleeves(self) -> None:
        panel = _panel(100, 4)
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=("xs_reversal",),
            admitted_signal_ids=("xs_reversal:fast", "xs_reversal:medium"),
        )
        sleeves = build_family_routing_sleeves(panel, screen, _cluster_folds(4), _folds())
        assert len(sleeves) == 8
        pairs = {(s.outer_fold_id, s.cluster_id) for s in sleeves}
        assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_orientation_and_member_hash_preserved(self) -> None:
        panel = _panel(100, 4)
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=("xs_reversal",),
            admitted_signal_ids=("xs_reversal:fast",),
        )
        sleeves = build_family_routing_sleeves(panel, screen, _cluster_folds(4), _folds())
        assert len(sleeves) == 4
        for s in sleeves:
            assert s.declared_orientation == 1
            assert s.member_hash in ("h0", "h1")

    def test_rejected_ids_create_no_sleeves(self) -> None:
        panel = _panel(100, 4)
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=(),
            admitted_signal_ids=(),
        )
        sleeves = build_family_routing_sleeves(panel, screen, _cluster_folds(4), _folds())
        assert sleeves == ()

    def test_singleton_clusters_skipped(self) -> None:
        panel = _panel(100, 4)
        cp_single = ClusterPanel(
            symbols=tuple(f"S{i}" for i in range(4)),
            cluster_labels=np.array([0, 0, 1, 2], dtype=np.int32),
            cluster_centroids=np.zeros((3, 4), dtype=np.float64),
            k_clusters=3,
        )
        cfs = (
            CausalClusterFold(fold_id=0, fit_end_exclusive_4h=20, fit_end_time_ns=20 * 14_400_000_000_000, panel=cp_single, member_hash="h0"),
        )
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=("xs_reversal",),
            admitted_signal_ids=("xs_reversal:fast",),
        )
        sleeves = build_family_routing_sleeves(panel, screen, cfs, _folds())
        folds_pairs = {(s.outer_fold_id, s.cluster_id) for s in sleeves}
        assert (0, 2) not in folds_pairs

    def test_unknown_admitted_id_raises(self) -> None:
        panel = _panel(100, 4)
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=("unknown",),
            admitted_signal_ids=("unknown:foo",),
        )
        with pytest.raises(ValueError, match=r"admitted signal id.*not found in panel"):
            build_family_routing_sleeves(panel, screen, _cluster_folds(4), _folds())

    def test_missing_fold_mapping_raises(self) -> None:
        panel = _panel(100, 4)
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=("xs_reversal",),
            admitted_signal_ids=("xs_reversal:fast",),
        )
        cfs = _cluster_folds(4)
        with pytest.raises(ValueError, match=r"fold.*from cluster_folds not found"):
            build_family_routing_sleeves(panel, screen, cfs, ())

    def test_orientation_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.domain.futures.compound.l1_sleeves as sleeves_module

        real_sleeve = sleeves_module.L1RoutingSleeve

        def corrupt_orientation(**kwargs: object) -> L1RoutingSleeve:
            kwargs["declared_orientation"] = -1
            return real_sleeve(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sleeves_module, "L1RoutingSleeve", corrupt_orientation)
        panel = _panel(100, 4)
        screen = FamilyEdgeScreen(
            records=(), n_effective_independent=1.0,
            admitted_families=("xs_reversal",),
            admitted_signal_ids=("xs_reversal:fast",),
        )
        with pytest.raises(ValueError, match=r"orientation mismatch"):
            build_family_routing_sleeves(panel, screen, _cluster_folds(4), _folds())


# ---- Scenario 3: build_fold_expert_books ----

class TestBuildFoldExpertBooks:

    def test_returns_allocator_weights_not_raw_z(self) -> None:
        panel = _panel(100, 4)
        bars = _bars(100, 4)
        contrib = (
            ExpertContribution("xs_reversal:fast", 0, 1, np.array([True, True, False, False], dtype=np.bool_), 0),
        )
        ac = DynamicCompoundingConfig()
        funding = np.zeros((400, 4), dtype=np.float32)
        weights = build_fold_expert_books(panel, contrib, bars, funding, ac, 8.0)
        assert weights.ndim == 3
        assert weights.shape[0] == 1
        assert weights.shape[1] == 100
        assert weights.shape[2] == 4
        assert np.all(np.isfinite(weights))

    def test_empty_input_returns_empty_3d(self) -> None:
        panel = _panel(100, 4)
        bars = _bars(100, 4)
        ac = DynamicCompoundingConfig()
        funding = np.zeros((400, 4), dtype=np.float32)
        weights = build_fold_expert_books(panel, (), bars, funding, ac, 8.0)
        assert weights.shape == (0, 100, 4)

    def test_weights_not_from_raw_z(self) -> None:
        panel = _panel(100, 4)
        bars = _bars(100, 4)
        contrib = (
            ExpertContribution("xs_reversal:fast", 0, 1, np.array([True, True, False, False], dtype=np.bool_), 0),
        )
        ac = DynamicCompoundingConfig()
        funding = np.zeros((400, 4), dtype=np.float32)
        weights = build_fold_expert_books(panel, contrib, bars, funding, ac, 8.0)
        raw_z = panel.z_3d[:, :, 0]
        assert not np.allclose(weights[0], raw_z)


# ---- Scenario 4: score_expert_returns (simple returns) ----

class TestScoreExpertReturns:

    def test_simple_return_timing_correct(self) -> None:
        t = 50
        n = 4
        weights = np.zeros((t, n), dtype=np.float64)
        weights[:-1, 0] = 1.0
        close = np.column_stack([np.linspace(100, 200, t) for _ in range(n)]).astype(np.float64)
        asset_return = np.zeros((t, n), dtype=np.float64)
        for i in range(1, t):
            asset_return[i] = close[i] / close[i - 1] - 1.0
        cost = np.ones((t, n), dtype=np.float32) * 8.0
        funding_4h = np.zeros((t, n), dtype=np.float64)
        gross, c, f, net = score_expert_returns(weights, asset_return, cost, funding_4h, 1, t - 1)
        for k, _t in enumerate(range(1, t - 1)):
            expected = float(np.dot(weights[_t], asset_return[_t + 1]))
            assert np.isclose(gross[k], expected)

    def test_gross_plus_cost_plus_funding_equals_net(self) -> None:
        t = 50
        n = 4
        weights = np.zeros((t, n), dtype=np.float64)
        weights[:, 0] = 0.5
        close = np.column_stack([np.linspace(100, 200, t) for _ in range(n)]).astype(np.float64)
        asset_return = np.zeros((t, n), dtype=np.float64)
        for i in range(1, t):
            asset_return[i] = close[i] / close[i - 1] - 1.0
        cost = np.full((t, n), 8.0, dtype=np.float32)
        funding_4h = np.full((t, n), 0.0001, dtype=np.float64)
        gross, c, f, net = score_expert_returns(weights, asset_return, cost, funding_4h, 1, t - 1)
        assert np.allclose(net, gross + c + f)

    def test_net_le_neg_one_raises(self) -> None:
        t = 20
        n = 2
        weights = np.zeros((t, n), dtype=np.float64)
        weights[5, 0] = 10.0
        asset_return = np.zeros((t, n), dtype=np.float64)
        asset_return[6, 0] = -0.20
        cost = np.zeros((t, n), dtype=np.float32)
        funding_4h = np.zeros((t, n), dtype=np.float64)
        with pytest.raises(ValueError, match=r"invalid_expert_return_domain|net return <= -1"):
            score_expert_returns(weights, asset_return, cost, funding_4h, 1, t - 1)

    def test_non_finite_components_raise(self) -> None:
        t = 20
        n = 2
        weights = np.zeros((t, n), dtype=np.float64)
        asset_return = np.full((t, n), np.nan, dtype=np.float64)
        cost = np.ones((t, n), dtype=np.float32) * 8.0
        funding_4h = np.zeros((t, n), dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite return components"):
            score_expert_returns(weights, asset_return, cost, funding_4h, 1, t - 1)


# ---- Scenario 5: integration - empty sleeves produce valid empty results ----

class TestEmptySleevesFlow:

    def test_empty_routing_sleeves_forward_to_handoff(self) -> None:
        from src.domain.futures.compound.l1_sleeves import build_exit_aware_handoff

        panel = _panel(100, 4)
        bars = _bars(100, 4)
        sleeves: tuple[L1RoutingSleeve, ...] = ()
        handoff_result = build_exit_aware_handoff(
            _fake_forecast(panel, bars), sleeves, bars, np.zeros(100, dtype=np.float64),
            unittest.mock.Mock(), folds=_folds(),
            weights_2d=np.zeros((100, 4), dtype=np.float64),
            cost_bps_4h=np.full((100, 4), 8.0, dtype=np.float32),
        )
        assert not handoff_result.evidence.admitted
        assert "no_admitted_sleeves" in handoff_result.evidence.reasons


def _fake_forecast(panel: RawSignalPanel, bars: TimeframeBarCube) -> object:
    from src.domain.futures.compound.contracts import CalibratedForecastPanel
    t, n = bars.close_2d.shape
    return CalibratedForecastPanel(
        decision_timestamps_ns=bars.timestamps_ns,
        symbols=panel.symbols,
        mu_2d=np.zeros((t, n), dtype=np.float32),
        se_2d=np.full((t, n), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((t, n, 0), dtype=np.float32),
        family_ids=(),
        admitted_signal_ids=(),
        fold_manifest_hash="",
    )


import unittest.mock
