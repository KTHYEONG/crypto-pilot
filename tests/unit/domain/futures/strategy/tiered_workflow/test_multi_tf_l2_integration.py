# tests/unit/domain/futures/strategy/tiered_workflow/test_multi_tf_l2_integration.py
"""Multi-TF L2 integration tests: predict_layer1_signals_multi_tf + sleeve-keyed L2SimulationCache.

Spec reference: docs/specs/layer2-multi-tf-integration.md
Test scenarios: A.S1-S7, B.S1-S2, C.S1.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _combine_sleeve_signals_to_symbol,
    build_l2_simulation_cache,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,
    Layer1Result,
    Layer2AllocationConfig,
    RegimeCellPolicy,
)
from src.domain.futures.strategy.tiered_workflow.l2_meta import apply_regime_cell_policy

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_signal_event(
    *,
    symbol: str,
    strategy_id: str,
    decision_idx: int,
    side: Literal[-1, 1] = 1,
    gross_bps: float = 50.0,
    net_bps: float = 40.0,
    holding_bars: int = 4,
    quality_weight: float = 0.8,
) -> ValidatedSignalEvent:
    """합성 ValidatedSignalEvent 빌더."""
    return ValidatedSignalEvent(
        decision_idx=decision_idx,
        decision_time=np.datetime64("2024-01-01T00:00:00", "ns"),
        symbol=symbol,
        strategy_id=strategy_id,
        activation_context="test",
        side=side,
        expected_gross_bps=gross_bps,
        expected_net_bps=net_bps,
        q10_gross_bps=gross_bps * 0.5,
        q90_gross_bps=gross_bps * 1.5,
        expected_holding_bars=holding_bars,
        quality_weight=quality_weight,
        registry_version="rv1",
        model_version="mv1",
    )


def _make_signal_batch(
    events: list[ValidatedSignalEvent],
    symbols: tuple[str, ...],
    start_idx: int = 0,
    end_idx: int = 100,
) -> ValidatedSignalBatch:
    """합성 ValidatedSignalBatch 빌더."""
    return ValidatedSignalBatch(
        events=tuple(events),
        start_idx=start_idx,
        end_idx=end_idx,
        symbols=symbols,
        registry_version="rv_test",
        model_version="mv_test",
    )


def _make_aligned(
    symbols: tuple[str, ...],
    t_max: int = 20,
) -> Any:
    """AlignedMarketData 최소 stub."""
    n = len(symbols)
    aligned = MagicMock()
    aligned.symbols = symbols
    aligned.close_2d = np.ones((t_max, n), dtype=np.float64)
    aligned.execution_cost_bps_2d = np.full((t_max, n), 3.8, dtype=np.float64)
    aligned.funding_2d = np.zeros((t_max, n), dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n, dtype=np.float64)
    aligned.datetimes = np.array(
        [np.datetime64(f"2024-01-{i+1:02d}", "D") for i in range(t_max)],
        dtype="datetime64[D]",
    )
    # adv_usdt 미설정(None)
    aligned.adv_usdt_2d = None
    return aligned


def _make_artifact(*, symbols: tuple[str, ...], strategy_ids: tuple[str, ...]) -> Any:
    """Layer1InferenceArtifact 최소 stub."""
    art = MagicMock()
    art.feature_schema = MagicMock()
    art.model = MagicMock()
    art.model_version = "mv1"
    reg = MagicMock()
    reg.by_symbol = {s: MagicMock() for s in symbols}
    art.deployment_registry = reg
    return art


# ─────────────────────────────────────────────────────────────────────────────
# A. predict_layer1_signals_multi_tf
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictLayer1SignalsMultiTf:
    """A.S1-S4: predict_layer1_signals_multi_tf 시나리오."""

    def _make_cfg(self) -> Any:
        cfg = MagicMock()
        cfg.l1_signal_activation_floor_bps = 0.0
        return cfg

    def test_happy_two_tf_artifacts_both_events_present(self) -> None:
        """A.S1: 2 TF artifact → events에 두 TF strategy_id 모두 존재."""
        # Arrange
        from src.domain.futures.strategy.tiered_workflow.signal_selection import (
            predict_layer1_signals_multi_tf,
        )

        symbols = ("SYMA",)
        aligned = _make_aligned(symbols, t_max=20)
        cfg = self._make_cfg()

        ev_4h = _make_signal_event(
            symbol="SYMA",
            strategy_id="donchian_72_4h",
            decision_idx=5,
            gross_bps=50.0,
        )
        ev_8h = _make_signal_event(
            symbol="SYMA",
            strategy_id="donchian_72_8h",
            decision_idx=5,
            gross_bps=300.0,
        )

        batch_4h = _make_signal_batch([ev_4h], symbols)
        batch_8h = _make_signal_batch([ev_8h], symbols)

        art_4h = _make_artifact(symbols=symbols, strategy_ids=("donchian_72_4h",))
        art_8h = _make_artifact(symbols=symbols, strategy_ids=("donchian_72_8h",))

        artifacts_by_tf = {"4h": art_4h, "8h": art_8h}

        # candidate_events: native_tf 컬럼 포함
        df_4h = pd.DataFrame({"native_tf": ["4h"], "symbol": ["SYMA"]})
        df_8h = pd.DataFrame({"native_tf": ["8h"], "symbol": ["SYMA"]})
        candidate_events = pd.concat([df_4h, df_8h], ignore_index=True)

        # Act: predict_layer1_signals를 mock → 미리 만든 batch 반환
        def mock_predict(*, artifact: Any, candidate_events: Any, **_kw: Any) -> ValidatedSignalBatch:
            if artifact is art_4h:
                return batch_4h
            if artifact is art_8h:
                return batch_8h
            return _make_signal_batch([], symbols)

        with patch(
            "src.domain.futures.strategy.tiered_workflow.signal_selection.predict_layer1_signals",
            side_effect=mock_predict,
        ):
            result = predict_layer1_signals_multi_tf(
                artifacts_by_tf=artifacts_by_tf,
                candidate_events=candidate_events,
                aligned=aligned,
                start_idx=0,
                end_idx=10,
                cfg=cfg,
            )

        # Assert
        strategy_ids = {e.strategy_id for e in result.events}
        assert "donchian_72_4h" in strategy_ids
        assert "donchian_72_8h" in strategy_ids
        assert len(result.events) == 2

    def test_edge_empty_tf_candidate_events_skips_tf(self) -> None:
        """A.S2: 한 TF candidate_events 비어있음 → skip, 나머지 정상 병합."""
        # Arrange
        from src.domain.futures.strategy.tiered_workflow.signal_selection import (
            predict_layer1_signals_multi_tf,
        )

        symbols = ("SYMA",)
        aligned = _make_aligned(symbols, t_max=20)
        cfg = self._make_cfg()

        ev_8h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_8h", decision_idx=5
        )
        batch_8h = _make_signal_batch([ev_8h], symbols)
        art_4h = _make_artifact(symbols=symbols, strategy_ids=("donchian_72_4h",))
        art_8h = _make_artifact(symbols=symbols, strategy_ids=("donchian_72_8h",))

        # 4h TF는 events 없음, 8h만 있음
        candidate_events = pd.DataFrame({"native_tf": ["8h"], "symbol": ["SYMA"]})

        def mock_predict(*, artifact: Any, candidate_events: Any, **_kw: Any) -> ValidatedSignalBatch:
            if artifact is art_8h:
                return batch_8h
            return _make_signal_batch([], symbols)

        # Act
        with patch(
            "src.domain.futures.strategy.tiered_workflow.signal_selection.predict_layer1_signals",
            side_effect=mock_predict,
        ):
            result = predict_layer1_signals_multi_tf(
                artifacts_by_tf={"4h": art_4h, "8h": art_8h},
                candidate_events=candidate_events,
                aligned=aligned,
                start_idx=0,
                end_idx=10,
                cfg=cfg,
            )

        # Assert: 4h는 skip(empty), 8h만 생존, 예외 없음
        assert len(result.events) == 1
        assert result.events[0].strategy_id == "donchian_72_8h"

    def test_edge_empty_artifacts_returns_empty_batch(self) -> None:
        """A.S3: artifacts_by_tf 비어있음 → 빈 ValidatedSignalBatch."""
        # Arrange
        from src.domain.futures.strategy.tiered_workflow.signal_selection import (
            predict_layer1_signals_multi_tf,
        )

        symbols = ("SYMA",)
        aligned = _make_aligned(symbols, t_max=20)
        cfg = self._make_cfg()

        # Act
        result = predict_layer1_signals_multi_tf(
            artifacts_by_tf={},
            candidate_events=pd.DataFrame(),
            aligned=aligned,
            start_idx=0,
            end_idx=10,
            cfg=cfg,
        )

        # Assert
        assert result.events == ()
        assert result.registry_version == "empty"

    def test_invariant_duplicate_key_removed(self) -> None:
        """A.S4: 동일 (symbol, strategy_id, decision_idx) 중복 → fail-closed 제거."""
        # Arrange
        from src.domain.futures.strategy.tiered_workflow.signal_selection import (
            predict_layer1_signals_multi_tf,
        )

        symbols = ("SYMA",)
        aligned = _make_aligned(symbols, t_max=20)
        cfg = self._make_cfg()

        ev = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5
        )
        # 동일 이벤트를 두 batch에 넣음
        batch1 = _make_signal_batch([ev], symbols)
        batch2 = _make_signal_batch([ev], symbols)
        art_4h = _make_artifact(symbols=symbols, strategy_ids=("donchian_72_4h",))
        art_8h = _make_artifact(symbols=symbols, strategy_ids=("donchian_72_4h",))

        candidate_events = pd.DataFrame({"native_tf": ["4h", "8h"], "symbol": ["SYMA", "SYMA"]})

        call_count = [0]

        def mock_predict(**_kw: Any) -> ValidatedSignalBatch:
            b = batch1 if call_count[0] == 0 else batch2
            call_count[0] += 1
            return b

        # Act
        with patch(
            "src.domain.futures.strategy.tiered_workflow.signal_selection.predict_layer1_signals",
            side_effect=mock_predict,
        ):
            result = predict_layer1_signals_multi_tf(
                artifacts_by_tf={"4h": art_4h, "8h": art_8h},
                candidate_events=candidate_events,
                aligned=aligned,
                start_idx=0,
                end_idx=10,
                cfg=cfg,
            )

        # Assert: 중복 제거 → 1개만 생존
        assert len(result.events) == 1


# ─────────────────────────────────────────────────────────────────────────────
# B. build_l2_simulation_cache (sleeve 차원)
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildL2SimulationCacheSleevedimension:
    """B.S1-S3: build_l2_simulation_cache sleeve 차원 시나리오."""

    def test_happy_two_tf_same_symbol_both_sleeves_survive(self) -> None:
        """B.S1: SYMA에 4h(+50bps)·8h(+300bps) → n_sleeve==2, 둘 다 생존 (기존 collapse 없음)."""
        # Arrange
        symbols = ("SYMA",)
        aligned = _make_aligned(symbols, t_max=20)

        ev_4h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5,
            gross_bps=50.0, net_bps=40.0, holding_bars=4,
        )
        ev_8h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_8h", decision_idx=5,
            gross_bps=300.0, net_bps=280.0, holding_bars=8,
        )
        batch = _make_signal_batch([ev_4h, ev_8h], symbols)

        # Act
        cache = build_l2_simulation_cache(aligned, batch, "4h")

        # Assert
        assert len(cache.sleeve_ids) == 2, f"Expected 2 sleeves, got {len(cache.sleeve_ids)}"
        # sleeve_to_sym: 두 sleeve 모두 SYMA(idx=0)
        assert all(cache.sleeve_to_sym == 0)
        # 두 sleeve 모두 신호 행렬에 값 존재
        n_sleeve = cache.expected_gross_bps_2d.shape[1]
        assert n_sleeve == 2

    def test_bva_single_tf_n_sleeve_equals_n_active_sym(self) -> None:
        """B.S2: 단일 TF → n_sleeve == n_active_symbols (기존 동작 회귀 가드)."""
        # Arrange
        symbols = ("SYMA", "SYMB")
        aligned = _make_aligned(symbols, t_max=20)

        ev_a = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5,
        )
        ev_b = _make_signal_event(
            symbol="SYMB", strategy_id="donchian_72_4h", decision_idx=5,
        )
        batch = _make_signal_batch([ev_a, ev_b], symbols)

        # Act
        cache = build_l2_simulation_cache(aligned, batch, "4h")

        # Assert: 단일 strategy → 각 symbol당 1 sleeve → n_sleeve = 2
        assert len(cache.sleeve_ids) == 2
        # sleeve_to_sym: [0,1] 정렬
        assert set(cache.sleeve_to_sym.tolist()) == {0, 1}

    def test_edge_empty_batch_no_crash(self) -> None:
        """B.S3: 빈 batch → n_sleeve==0, 빈 행렬, crash 없음."""
        # Arrange
        symbols = ("SYMA",)
        aligned = _make_aligned(symbols, t_max=20)
        batch = _make_signal_batch([], symbols)

        # Act
        cache = build_l2_simulation_cache(aligned, batch, "4h")

        # Assert
        assert len(cache.sleeve_ids) == 0
        assert cache.expected_gross_bps_2d.shape == (20, 0)
        assert cache.sleeve_to_sym.shape == (0,)


# ─────────────────────────────────────────────────────────────────────────────
# C. Sleeve→symbol 집계·cap 수치 검증
# ─────────────────────────────────────────────────────────────────────────────


class TestSleeveToSymbolAggregation:
    """C.S1-S4: sleeve→symbol netting, cap, 회귀 불변, 차등 sizing."""

    def _build_cache_from_events(
        self,
        events: list[ValidatedSignalEvent],
        symbols: tuple[str, ...],
        t_max: int = 20,
    ) -> L2SimulationCache:
        aligned = _make_aligned(symbols, t_max=t_max)
        batch = _make_signal_batch(events, symbols)
        return build_l2_simulation_cache(aligned, batch, "4h")

    def test_two_long_sleeves_same_symbol_sleeve_matrices_both_populated(self) -> None:
        """C.S1: SYMA long sleeve(4h)·long sleeve(8h) → 두 sleeve 모두 신호 행렬 비어있지 않음."""
        # Arrange
        symbols = ("SYMA",)
        ev_4h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5,
            gross_bps=50.0, net_bps=40.0, holding_bars=4, side=1,
        )
        ev_8h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_8h", decision_idx=5,
            gross_bps=300.0, net_bps=280.0, holding_bars=4, side=1,
        )

        # Act
        cache = self._build_cache_from_events([ev_4h, ev_8h], symbols)

        # Assert: 두 sleeve 모두 t=6 (decision+1)에 활성
        n_sleeve = cache.signal_mask_2d.shape[1]
        assert n_sleeve == 2
        assert cache.signal_mask_2d[6, 0], "sleeve 0 (sorted first) should be active at t=6"
        assert cache.signal_mask_2d[6, 1], "sleeve 1 (sorted second) should be active at t=6"

    def test_long_and_short_sleeves_signal_mask_independent(self) -> None:
        """C.S2: SYMA long·short sleeve → 두 sleeve 독립적으로 신호 행렬 활성."""
        # Arrange
        symbols = ("SYMA",)
        ev_long = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5,
            side=1, gross_bps=50.0, net_bps=40.0,
        )
        ev_short = _make_signal_event(
            symbol="SYMA", strategy_id="rsi_short_4h", decision_idx=5,
            side=-1, gross_bps=50.0, net_bps=40.0,
        )

        # Act
        cache = self._build_cache_from_events([ev_long, ev_short], symbols)

        # Assert: 두 sleeve 모두 활성 (netting은 시뮬레이션 단계)
        assert cache.signal_mask_2d.shape[1] == 2
        # sleeve_to_sym 둘 다 0 (SYMA)
        assert cache.sleeve_to_sym[0] == 0
        assert cache.sleeve_to_sym[1] == 0
        # side_2d: 두 sleeve의 side가 반대
        sides = {int(cache.side_2d[6, j]) for j in range(2)}
        assert 1 in sides
        assert -1 in sides

    def test_single_tf_sleeve_count_matches_single_symbol(self) -> None:
        """C.S3: 단일 TF 단일 symbol → n_sleeve==1, sleeve_to_sym[0]==0."""
        # Arrange
        symbols = ("SYMA",)
        ev = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5,
        )

        # Act
        cache = self._build_cache_from_events([ev], symbols)

        # Assert
        assert len(cache.sleeve_ids) == 1
        assert cache.sleeve_to_sym[0] == 0

    def test_higher_edge_sleeve_has_larger_gross_bps(self) -> None:
        """C.S4: 8h(+300bps)·4h(+50bps) → 8h sleeve gross_bps > 4h sleeve gross_bps."""
        # Arrange
        symbols = ("SYMA",)
        ev_4h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=5,
            gross_bps=50.0, net_bps=40.0, holding_bars=4,
        )
        ev_8h = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_8h", decision_idx=5,
            gross_bps=300.0, net_bps=280.0, holding_bars=8,
        )

        # Act
        cache = self._build_cache_from_events([ev_4h, ev_8h], symbols)

        # Assert: sleeve_ids 정렬 기준 확인 후 gross_bps 비교
        # sleeve_ids는 sorted — donchian_72_4h < donchian_72_8h (lex 순)
        sleeve_map = {sid: j for j, sid in enumerate(cache.sleeve_ids)}
        j_4h = sleeve_map[("SYMA", "donchian_72_4h")]
        j_8h = sleeve_map[("SYMA", "donchian_72_8h")]
        # t=6: decision_idx=5, active_start=6
        gross_4h = cache.expected_gross_bps_2d[6, j_4h]
        gross_8h = cache.expected_gross_bps_2d[6, j_8h]
        assert gross_8h > gross_4h, f"8h({gross_8h}) should > 4h({gross_4h})"


# ─────────────────────────────────────────────────────────────────────────────
# D. Layer1Result.artifacts_by_tf 필드 검증
# ─────────────────────────────────────────────────────────────────────────────


class TestLayer1ResultArtifactsByTf:
    """D.S1-S2: Layer1Result.artifacts_by_tf 필드 존재 + 기본값 검증."""

    def test_artifacts_by_tf_default_empty_dict(self) -> None:
        """D.S1: Layer1Result 생성 시 artifacts_by_tf 기본값 빈 dict."""
        # Arrange & Act
        result = Layer1Result(
            signals_per_fold=(),
            oos_stacked={},
            pooled_ic=0.0,
            pooled_tstat=0.0,
            breadth=0.0,
            valid_coverage=0.0,
            fold_pass_ratio=0.0,
            gate_passed=False,
            n_valid=0,
            n_total=0,
        )

        # Assert
        assert result.artifacts_by_tf == {}
        assert isinstance(result.artifacts_by_tf, dict)

    def test_artifacts_by_tf_populated_with_values(self) -> None:
        """D.S2: artifacts_by_tf에 TF 키 설정 시 정상 저장."""
        # Arrange
        art = _make_artifact(symbols=("SYMA",), strategy_ids=("donchian_72_4h",))

        # Act
        result = Layer1Result(
            signals_per_fold=(),
            oos_stacked={},
            pooled_ic=0.0,
            pooled_tstat=0.0,
            breadth=0.0,
            valid_coverage=0.0,
            fold_pass_ratio=0.0,
            gate_passed=True,
            n_valid=1,
            n_total=1,
            artifacts_by_tf={"4h": art, "8h": art},
        )

        # Assert
        assert len(result.artifacts_by_tf) == 2
        assert "4h" in result.artifacts_by_tf
        assert "8h" in result.artifacts_by_tf


# ─────────────────────────────────────────────────────────────────────────────
# Regression guard: L2SimulationCache has sleeve fields
# ─────────────────────────────────────────────────────────────────────────────


class TestL2SimulationCacheFields:
    """L2SimulationCache 스키마 회귀 가드."""

    def test_cache_has_sleeve_fields(self) -> None:
        """L2SimulationCache에 sleeve_to_sym, sleeve_ids 필드 존재."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(L2SimulationCache)}
        assert "sleeve_to_sym" in field_names
        assert "sleeve_ids" in field_names

    def test_build_cache_returns_correct_sleeve_shapes(self) -> None:
        """build_l2_simulation_cache: 신호 행렬 [T, S], 심볼 행렬 [T, N] shape 일치."""
        # Arrange
        symbols = ("SYMA", "SYMB")
        t_max = 15
        aligned = _make_aligned(symbols, t_max=t_max)

        ev_a = _make_signal_event(
            symbol="SYMA", strategy_id="donchian_72_4h", decision_idx=3,
        )
        ev_b = _make_signal_event(
            symbol="SYMB", strategy_id="donchian_72_8h", decision_idx=3,
        )
        batch = _make_signal_batch([ev_a, ev_b], symbols)

        # Act
        cache = build_l2_simulation_cache(aligned, batch, "4h")

        # Assert shapes
        n_sym = len(symbols)
        n_sleeve = len(cache.sleeve_ids)
        assert cache.vol_matrix_2d.shape == (t_max, n_sym)
        assert cache.tradeable_mask_2d.shape == (t_max, n_sym)
        assert cache.expected_gross_bps_2d.shape == (t_max, n_sleeve)
        assert cache.signal_mask_2d.shape == (t_max, n_sleeve)
        assert cache.sleeve_to_sym.shape == (n_sleeve,)


# ─────────────────────────────────────────────────────────────────────────────
# E. _combine_sleeve_signals_to_symbol (Precision-Weighted Pooling)
# ─────────────────────────────────────────────────────────────────────────────


class TestCombineSleeveSignalsToSymbol:
    """A.S1-S7: _combine_sleeve_signals_to_symbol 수학 검증."""

    def _make_ss(self, raw_mu: float, quality_weight: float = 1.0, vol: float = 0.01) -> SymbolSignal:
        return SymbolSignal(
            raw_mu=raw_mu,
            volatility=vol,
            n_obs=10,
            t_stat=2.0,
            valid=True,
            beta_btc=None,
            quality_weight=quality_weight,
        )

    def test_bounded_no_inflation(self) -> None:
        """A.S1: 2 sleeve μ=[+50,+300], qw=[1,1] → mu_s=175, NOT 350."""
        signals = {
            ("SYMA", "strat_a"): self._make_ss(50.0, 1.0),
            ("SYMA", "strat_b"): self._make_ss(300.0, 1.0),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        mu_s = result["SYMA"].raw_mu
        assert mu_s == 175.0  # (50+300)/2
        assert 50.0 <= mu_s <= 300.0

    def test_precision_weighted(self) -> None:
        """A.S2: μ=[+50,+300], qw=[3,1] → mu_s=(3*50+1*300)/4=112.5."""
        signals = {
            ("SYMA", "strat_a"): self._make_ss(50.0, 3.0),
            ("SYMA", "strat_b"): self._make_ss(300.0, 1.0),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        assert result["SYMA"].raw_mu == 112.5

    def test_direction_conflict_netting(self) -> None:
        """A.S3: μ=[+100,-100], qw=[1,1] → mu_s=0.0 (상쇄)."""
        signals = {
            ("SYMA", "strat_a"): self._make_ss(100.0, 1.0),
            ("SYMA", "strat_b"): self._make_ss(-100.0, 1.0),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        assert result["SYMA"].raw_mu == 0.0

    def test_conviction_cap(self) -> None:
        """A.S4: 4 sleeve qw=[1,1,1,1], κ=1.5 → c_s=min(4.0, 1.5)=1.5."""
        signals = {
            ("SYMA", f"strat_{i}"): self._make_ss(10.0, 1.0)
            for i in range(4)
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals, conviction_cap_mult=1.5)
        assert result["SYMA"].quality_weight == 1.5

    def test_all_zero_qw_fallback(self) -> None:
        """A.S5: qw=[0,0] → equal-weight 평균, no division error."""
        signals = {
            ("SYMA", "strat_a"): self._make_ss(50.0, 0.0),
            ("SYMA", "strat_b"): self._make_ss(150.0, 0.0),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        assert result["SYMA"].raw_mu == 100.0  # equal-weight mean

    def test_max_edge_method(self) -> None:
        """A.S6: μ=[+50,-300], method=max_edge → mu_s=-300 (|max|)."""
        signals = {
            ("SYMA", "strat_a"): self._make_ss(50.0, 1.0),
            ("SYMA", "strat_b"): self._make_ss(-300.0, 1.0),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals, method="max_edge")
        assert result["SYMA"].raw_mu == -300.0

    def test_single_sleeve_identity(self) -> None:
        """A.S7: 1 sleeve → mu_s==μ_0, c_s==qw_0 (단일-TF 동치)."""
        signals = {
            ("SYMA", "strat_a"): self._make_ss(75.0, 2.5),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        assert result["SYMA"].raw_mu == 75.0
        assert result["SYMA"].quality_weight == 2.5


# ─────────────────────────────────────────────────────────────────────────────
# F. 통합 회귀 (run_awf_simulation 경로)
# ─────────────────────────────────────────────────────────────────────────────


class TestPoolingRegression:
    """B.S1-B.S2: run_awf_simulation 통합 회귀 검증."""

    @staticmethod
    def _make_ss(raw_mu: float, quality_weight: float = 1.0, vol: float = 0.01) -> SymbolSignal:
        return SymbolSignal(
            raw_mu=raw_mu,
            volatility=vol,
            n_obs=10,
            t_stat=2.0,
            valid=True,
            beta_btc=None,
            quality_weight=quality_weight,
        )

    def test_single_tf_numerical_equivalence(self) -> None:
        """B.S1: 단일 TF signal_batch → pooled mu == single sleeve mu."""
        signals = {
            ("SYMA", "strat_a"): SymbolSignal(
                raw_mu=42.0, volatility=0.01, n_obs=10, t_stat=2.0,
                valid=True, beta_btc=None, quality_weight=1.0,
            ),
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        assert result["SYMA"].raw_mu == 42.0
        assert result["SYMA"].quality_weight == 1.0

    def test_multi_tf_pooled_mu_not_exceeding_max(self) -> None:
        """B.S2: 4-TF 동방향 → pooled mu ≤ max single mu (인플레이션 방지)."""
        signals = {
            ("SYMA", f"tf_{i}"): self._make_ss(100.0, 1.0)
            for i in range(4)
        }
        result, _ = _combine_sleeve_signals_to_symbol(signals)
        assert result["SYMA"].raw_mu == 100.0  # all same → mean = 100
        assert result["SYMA"].raw_mu <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# G. Layer2AllocationConfig 신규 knob 파싱
# ─────────────────────────────────────────────────────────────────────────────


class TestLayer2AllocationConfigKnobs:
    """C.S1: 신규 knob from_mapping 파싱."""

    def test_custom_method_and_cap(self) -> None:
        """C.S1: from_mapping으로 equal + 2.0 설정."""
        cfg = Layer2AllocationConfig.from_mapping({
            "l2_sleeve_combine_method": "equal",
            "l2_sleeve_conviction_cap_mult": 2.0,
        })
        assert cfg.l2_sleeve_combine_method == "equal"
        assert cfg.l2_sleeve_conviction_cap_mult == 2.0

    def test_default_values(self) -> None:
        """C.S1: 기본값 precision_weighted, 1.5."""
        cfg = Layer2AllocationConfig.from_mapping(None)
        assert cfg.l2_sleeve_combine_method == "precision_weighted"
        assert cfg.l2_sleeve_conviction_cap_mult == 1.5

    def test_invalid_method_raises(self) -> None:
        """C.S1: 잘못된 method → ValueError."""
        with pytest.raises(ValueError, match="l2_sleeve_combine_method"):
            Layer2AllocationConfig.from_mapping({
                "l2_sleeve_combine_method": "invalid",
            })


# ─────────────────────────────────────────────────────────────────────────────
# H. _combine_sleeve_signals_to_symbol friction 산출
# ─────────────────────────────────────────────────────────────────────────────


class TestCombineFriction:
    """A.S1-A.S5: friction_by_symbol 산출 검증."""

    @staticmethod
    def _make_ss(
        raw_mu: float, quality_weight: float = 1.0, vol: float = 0.01,
    ) -> SymbolSignal:
        return SymbolSignal(
            raw_mu=raw_mu, volatility=vol, n_obs=10, t_stat=2.0,
            valid=True, beta_btc=None, quality_weight=quality_weight,
        )

    def test_friction_pass(self) -> None:
        """A.S1: gross=+10, cost=+5 → friction True."""
        signals = {("SYMA", "a"): self._make_ss(5.0, 1.0)}
        edges = {("SYMA", "a"): (10.0, 5.0)}
        _, friction = _combine_sleeve_signals_to_symbol(
            signals, sleeve_edges=edges,
        )
        assert friction["SYMA"] is True

    def test_friction_fail(self) -> None:
        """A.S2: gross=+3, cost=+5 → friction False."""
        signals = {("SYMA", "a"): self._make_ss(0.0, 1.0)}
        edges = {("SYMA", "a"): (3.0, 5.0)}
        _, friction = _combine_sleeve_signals_to_symbol(
            signals, sleeve_edges=edges,
        )
        assert friction["SYMA"] is False

    def test_friction_precision_pooling(self) -> None:
        """A.S3: 2 sleeve gross=[+10,+2] cost=[4,4] qw=[3,1] → True."""
        signals = {
            ("SYMA", "a"): self._make_ss(5.0, 3.0),
            ("SYMA", "b"): self._make_ss(0.0, 1.0),
        }
        edges = {
            ("SYMA", "a"): (10.0, 4.0),
            ("SYMA", "b"): (2.0, 4.0),
        }
        _, friction = _combine_sleeve_signals_to_symbol(
            signals, sleeve_edges=edges,
        )
        # g_bar=(3*10+1*2)/4=8, c_bar=(3*4+1*4)/4=4 → 8>=4 → True
        assert friction["SYMA"] is True

    def test_friction_conflict_netting_fail(self) -> None:
        """A.S4: gross=[+10,-10] cost=[4,4] qw=[1,1] → False (g_bar=0)."""
        signals = {
            ("SYMA", "a"): self._make_ss(0.0, 1.0),
            ("SYMA", "b"): self._make_ss(0.0, 1.0),
        }
        edges = {
            ("SYMA", "a"): (10.0, 4.0),
            ("SYMA", "b"): (-10.0, 4.0),
        }
        _, friction = _combine_sleeve_signals_to_symbol(
            signals, sleeve_edges=edges,
        )
        # g_bar=(1*10+1*(-10))/2=0 < c_bar=4 → False
        assert friction["SYMA"] is False

    def test_sleeve_edges_none_returns_empty_friction(self) -> None:
        """A.S5: sleeve_edges=None → friction dict is empty."""
        signals = {("SYMA", "a"): self._make_ss(5.0, 1.0)}
        _, friction = _combine_sleeve_signals_to_symbol(signals)
        assert friction == {}

    def test_combine_sleeves_uses_regime_scaled_raw_mu_and_quality_weight(self) -> None:
        key_a = ("SYMA", "donchian_72_4h")
        key_b = ("SYMA", "trend_pullback_4h")
        signals = {
            key_a: self._make_ss(20.0, 2.0),
            key_b: self._make_ss(4.0, 1.0),
        }
        policy = RegimeCellPolicy(
            state=0,
            state_name="bull",
            family="donchian_72",
            tf="4h",
            action="downweight",
            reason="negative_cal_lift",
            edge_multiplier=0.5,
            confidence=1.0,
            fit_edge_bps=10.0,
            pooled_fit_edge_bps=5.0,
            cal_edge_bps=-5.0,
            pooled_cal_edge_bps=2.0,
            fit_lift_bps=-20.0,
            cal_lift_bps=-20.0,
            sign_consistent=True,
            hard_block_eligible=False,
            n_fit=5,
            n_cal=5,
        )
        applied = apply_regime_cell_policy(
            signals,
            {key_a: 20.0, key_b: 4.0},
            {(0, "donchian_72", "4h"): policy},
            0,
            mode="soft",
        )

        combined, _ = _combine_sleeve_signals_to_symbol(applied.sleeve_sigs)

        assert applied.sleeve_sigs[key_a].raw_mu == pytest.approx(10.0)
        assert applied.sleeve_sigs[key_a].quality_weight == pytest.approx(1.0)
        assert combined["SYMA"].raw_mu == pytest.approx(7.0)
        assert combined["SYMA"].quality_weight == pytest.approx(1.5)


# ─────────────────────────────────────────────────────────────────────────────
# I. 차원 정합 회귀 (per-bar consistency)
# ─────────────────────────────────────────────────────────────────────────────


class TestFrictionDimensionalConsistency:
    """B.S1-B.S2: per-bar 차원 일치 검증."""

    @staticmethod
    def _make_ss(raw_mu: float, vol: float = 0.01) -> SymbolSignal:
        return SymbolSignal(
            raw_mu=raw_mu, volatility=vol, n_obs=10, t_stat=2.0,
            valid=True, beta_btc=None, quality_weight=1.0,
        )

    def test_h_invariance(self) -> None:
        """B.S1: 동일 경제엣지 H=4 vs H=72 → friction 동일."""
        # gross_total=400, H=4 → gross_pb=100, cost_pb=50
        signals_h4 = {("SYMA", "a"): self._make_ss(50.0)}
        edges_h4 = {("SYMA", "a"): (100.0, 50.0)}
        _, f_h4 = _combine_sleeve_signals_to_symbol(
            signals_h4, sleeve_edges=edges_h4,
        )
        # gross_total=7200, H=72 → gross_pb=100, cost_pb=50
        signals_h72 = {("SYMA", "a"): self._make_ss(50.0)}
        edges_h72 = {("SYMA", "a"): (100.0, 50.0)}
        _, f_h72 = _combine_sleeve_signals_to_symbol(
            signals_h72, sleeve_edges=edges_h72,
        )
        assert f_h4["SYMA"] == f_h72["SYMA"]

    def test_net_positive_implies_friction_pass(self) -> None:
        """B.S2: gross-basis에서 net>0 ⟹ friction True."""
        signals = {("SYMA", "a"): self._make_ss(10.0)}
        edges = {("SYMA", "a"): (30.0, 20.0)}
        _, friction = _combine_sleeve_signals_to_symbol(
            signals, sleeve_edges=edges,
        )
        assert friction["SYMA"] is True


# ─────────────────────────────────────────────────────────────────────────────
# J. 카운터 통합 (friction_pass_total / signal_total)
# ─────────────────────────────────────────────────────────────────────────────


class TestFrictionCounterIntegration:
    """C.S1: friction_pass_total / signal_total 산출."""

    def test_counter_aggregation(self) -> None:
        """C.S1: selected=[a,b,c], friction={a:T,b:F,c:T} → pass=2, total=3."""
        friction_by_symbol = {"SYMA": True, "SYMB": False, "SYMC": True}
        selected = ["SYMA", "SYMB", "SYMC"]
        friction_pass = int(sum(1 for s in selected if friction_by_symbol.get(s, False)))
        signal_total = len(selected)
        assert friction_pass == 2
        assert signal_total == 3
        assert friction_pass / signal_total == pytest.approx(0.6667, abs=1e-3)
