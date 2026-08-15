"""signal_batch_convert 벡터화 등가성·gate·경계·relaxed 모드 검증."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
    ValidatedSignalBatch,
)

# ─── Shared builders ─────────────────────────────────────────────────────────


def _make_evidence(symbol: str, strategy_id: str, activation_context: str, qw: float = 0.75) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(symbol=symbol, strategy_id=strategy_id, activation_context=activation_context),
        mean_gross_bps=5.0,
        mean_incremental_bps=3.0,
        p_value=0.04,
        q_value=0.08,
        positive_fold_ratio=0.7,
        n_obs=200,
        effective_n=180.0,
        n_folds=4,
        quality_weight=qw,
        hard_eligible=True,
    )


def _make_registry(entries: list[tuple[str, str, str, float]]) -> QualifiedSignalRegistry:
    by_symbol: dict[str, list[SymbolStrategyEvidence]] = {}
    for sym, stid, actx, qw in entries:
        ev = _make_evidence(sym, stid, actx, qw)
        by_symbol.setdefault(sym, []).append(ev)
    return QualifiedSignalRegistry(
        by_symbol={k: tuple(v) for k, v in by_symbol.items()},
        ready_symbols=tuple(by_symbol.keys()),
        trade_scope_count=len(by_symbol),
        registry_version="v1",
    )


def _make_model_output(
    symbols: list[str],
    strategy_ids: list[str],
    activation_contexts: list[str],
    entry_idxs: list[int],
    sides: list[int],
    gross_bps: list[float],
    n_bars: int = 200,
) -> Any:
    n = len(symbols)
    events = pd.DataFrame(
        {
            "symbol": symbols,
            "strategy_id": strategy_ids,
            "activation_context": activation_contexts,
            "entry_idx": entry_idxs,
            "side": sides,
            "family": ["trend"] * n,
            "variant": ["v1"] * n,
            "expected_holding_bars": [12] * n,
            "cost_floor_bps": [7.5] * n,
        }
    )
    gross = np.array(gross_bps, dtype=np.float64)
    return SimpleNamespace(
        events=events,
        expected_gross_bps=gross,
        expected_net_bps=gross - 5.0,
        q10_gross_bps=gross - 3.0,
        q10_net_bps=gross - 8.0,
        q90_gross_bps=gross + 10.0,
        q90_net_bps=gross + 5.0,
        _has_explicit_expected_gross_bps=True,
    )


def _make_datetimes(n: int) -> Any:
    return np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n)],
        dtype="datetime64[ns]",
    )


def _call(
    model_output: Any,
    registry: QualifiedSignalRegistry,
    n_bars: int = 200,
    floor: float = 0.0,
    cfg: Any = None,
    native_tf: str = "",
) -> ValidatedSignalBatch:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import _candidate_output_to_signal_batch

    datetimes = _make_datetimes(n_bars)
    return _candidate_output_to_signal_batch(
        model_output=model_output,
        registry=registry,
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        model_version="v1",
        activation_floor_bps=floor,
        cfg=cfg,
        native_tf=native_tf,
    )


# ─── Scenario 1: registry filter (등가성 기능 검증) ──────────────────────────


def test_signal_batch_returns_only_registry_matched_events() -> None:
    # Arrange — 3 events, registry에 2개만 등록
    model_output = _make_model_output(
        symbols=["BTCUSDT", "BTCUSDT", "ETHUSDT"],
        strategy_ids=["trend:v1", "trend:v1", "mom:v2"],
        activation_contexts=["C1", "C2", "C1"],
        entry_idxs=[10, 20, 30],
        sides=[1, 1, -1],
        gross_bps=[15.0, 12.0, 8.0],
    )
    registry = _make_registry(
        [
            ("BTCUSDT", "trend:v1", "C1", 0.80),  # C1만 등록, C2 미등록
            ("ETHUSDT", "mom:v2", "C1", 0.60),
        ]
    )

    # Act
    result = _call(model_output, registry)

    # Assert
    assert len(result.events) == 2
    syms = {e.symbol for e in result.events}
    actxs = {e.activation_context for e in result.events}
    assert syms == {"BTCUSDT", "ETHUSDT"}
    assert "C2" not in actxs  # C2는 registry 미등록 → 필터


def test_signal_batch_field_values_correct() -> None:
    # Arrange
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["trend:v1"],
        activation_contexts=["C1"],
        entry_idxs=[50],
        sides=[1],
        gross_bps=[20.0],
    )
    registry = _make_registry([("BTCUSDT", "trend:v1", "C1", 0.55)])

    # Act
    result = _call(model_output, registry)

    # Assert
    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.symbol == "BTCUSDT"
    assert ev.strategy_id == "trend:v1"
    assert ev.activation_context == "C1"
    assert ev.decision_idx == 49  # entry_idx=50 → decision_idx=49
    assert ev.side == 1
    assert ev.expected_gross_bps == pytest.approx(20.0)
    assert ev.expected_net_bps == pytest.approx(15.0)
    assert ev.quality_weight == pytest.approx(0.55)
    assert ev.registry_version == "v1"
    assert ev.model_version == "v1"


# ─── Scenario 2: empty frame early return ────────────────────────────────────


def test_signal_batch_empty_frame_returns_empty_batch() -> None:
    # Arrange
    empty_out = SimpleNamespace(
        events=pd.DataFrame(),
        expected_gross_bps=np.zeros(0),
        expected_net_bps=np.zeros(0),
        q10_gross_bps=np.zeros(0),
        q10_net_bps=np.zeros(0),
        q90_gross_bps=np.zeros(0),
        q90_net_bps=np.zeros(0),
        _has_explicit_expected_gross_bps=True,
    )
    registry = _make_registry([("BTCUSDT", "trend:v1", "C1", 0.8)])

    # Act
    result = _call(empty_out, registry)

    # Assert
    assert result.events == ()
    assert result.start_idx == 0
    assert result.end_idx == 0


# ─── Scenario 3: decision_idx boundary guards ────────────────────────────────


def test_signal_batch_excludes_entry_idx_zero_and_out_of_bounds() -> None:
    # Arrange — entry_idx=0 → dec=-1(exclude), entry_idx=201 → dec=200≥200(exclude), entry_idx=100 → ok
    model_output = _make_model_output(
        symbols=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        strategy_ids=["trend:v1", "trend:v1", "trend:v1"],
        activation_contexts=["C1", "C1", "C1"],
        entry_idxs=[0, 201, 100],
        sides=[1, 1, 1],
        gross_bps=[20.0, 20.0, 20.0],
    )
    registry = _make_registry([("BTCUSDT", "trend:v1", "C1", 0.8)])

    # Act — n_bars=200 → valid index range [0,199]
    result = _call(model_output, registry, n_bars=200)

    # Assert — 오직 entry_idx=100(dec=99)만 통과
    assert len(result.events) == 1
    assert result.events[0].decision_idx == 99


# ─── Scenario 4: has_explicit_gross=False → empty events ─────────────────────


def test_signal_batch_no_gross_returns_empty_events() -> None:
    # Arrange
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["trend:v1"],
        activation_contexts=["C1"],
        entry_idxs=[10],
        sides=[1],
        gross_bps=[20.0],
    )
    model_output._has_explicit_expected_gross_bps = False
    registry = _make_registry([("BTCUSDT", "trend:v1", "C1", 0.8)])

    # Act
    result = _call(model_output, registry)

    # Assert
    assert result.events == ()


# ─── Scenario 5: activation_floor_bps threshold gate ─────────────────────────


def test_signal_batch_floor_filters_low_gross() -> None:
    # Arrange — 2 events: gross=5bps(below floor), gross=20bps(above)
    model_output = _make_model_output(
        symbols=["BTCUSDT", "ETHUSDT"],
        strategy_ids=["trend:v1", "trend:v1"],
        activation_contexts=["C1", "C1"],
        entry_idxs=[10, 20],
        sides=[1, 1],
        gross_bps=[5.0, 20.0],
    )
    registry = _make_registry(
        [
            ("BTCUSDT", "trend:v1", "C1", 0.8),
            ("ETHUSDT", "trend:v1", "C1", 0.7),
        ]
    )

    # Act — floor=10.0 → 5.0 는 제외
    result = _call(model_output, registry, floor=10.0)

    # Assert
    assert len(result.events) == 1
    assert result.events[0].symbol == "ETHUSDT"
    assert result.events[0].expected_gross_bps == pytest.approx(20.0)


# ─── Scenario 6: activation_match_regime=False → relaxed key (sym+strat) ─────


def test_signal_batch_relaxed_mode_collapses_activation_context() -> None:
    # Arrange — registry에 actx="all" 등록(relaxed build 표준), frame에는 다양한 actx
    model_output = _make_model_output(
        symbols=["BTCUSDT", "BTCUSDT"],
        strategy_ids=["trend:v1", "trend:v1"],
        activation_contexts=["C_trend", "C_rev"],  # 여러 actx
        entry_idxs=[10, 20],
        sides=[1, 1],
        gross_bps=[15.0, 12.0],
    )
    # relaxed mode: source_keys_relaxed = {(symbol, strategy_id)} — actx 무관
    registry = _make_registry(
        [
            ("BTCUSDT", "trend:v1", "all", 0.70),  # registry actx="all"
        ]
    )

    # Act — cfg.l1_activation_match_regime=False
    cfg = SimpleNamespace(l1_activation_match_regime=False)
    result = _call(model_output, registry, cfg=cfg)

    # Assert — sym+strat 일치하므로 2개 모두 통과 (actx collapse→"all")
    assert len(result.events) == 2
    for ev in result.events:
        assert ev.activation_context == "all"


# ─── Scenario 7: side=-1 (short) 처리 ───────────────────────────────────────


def test_signal_batch_short_side_preserved() -> None:
    # Arrange
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["rev:v1"],
        activation_contexts=["C1"],
        entry_idxs=[10],
        sides=[-1],
        gross_bps=[18.0],
    )
    registry = _make_registry([("BTCUSDT", "rev:v1", "C1", 0.6)])

    # Act
    result = _call(model_output, registry)

    # Assert
    assert len(result.events) == 1
    assert result.events[0].side == -1


# ─── Scenario 8: quality_weight lookup — first-match semantics ───────────────


def test_signal_batch_quality_weight_first_match() -> None:
    # Arrange — 같은 key가 registry에 중복이 있는 경우 첫 번째 qw 사용
    # (정상 운영에서 중복 없지만 방어 검증)
    ev1 = _make_evidence("BTCUSDT", "trend:v1", "C1", qw=0.90)
    ev2 = _make_evidence("BTCUSDT", "trend:v1", "C1", qw=0.10)  # 같은 key, 다른 qw
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev1, ev2)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="v1",
    )
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["trend:v1"],
        activation_contexts=["C1"],
        entry_idxs=[10],
        sides=[1],
        gross_bps=[15.0],
    )

    # Act
    result = _call(model_output, registry)

    # Assert — 첫 번째 ev의 qw(0.90) 사용
    assert len(result.events) == 1
    assert result.events[0].quality_weight == pytest.approx(0.90)


# ─── Fix B: TF-suffix-aware key matching (L2-CRISIS-WIRING-AND-TF-SIGNAL-LOSS-FIX) ──


def test_strip_tf_suffix_exact_match_only() -> None:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import _strip_tf_suffix

    assert _strip_tf_suffix("trend_donchian:donchian_72_12h", "12h") == "trend_donchian:donchian_72"
    assert _strip_tf_suffix("trend_donchian:donchian_72", "12h") == "trend_donchian:donchian_72"
    assert _strip_tf_suffix("trend_donchian:donchian_72", "") == "trend_donchian:donchian_72"
    # not a substring match: "...472" does not end with "_4h"
    assert _strip_tf_suffix("weird_variant_472", "4h") == "weird_variant_472"


def test_strip_tf_suffix_series_matches_scalar_elementwise() -> None:
    import pandas as pd
    from src.domain.futures.strategy.tiered_workflow.signal_selection import _strip_tf_suffix_series

    values = pd.Series(["strat_a_4h", "strat_b", "strat_c_12h", "strat_d_472"])
    result = _strip_tf_suffix_series(values, "12h")
    expected = pd.Series(["strat_a_4h", "strat_b", "strat_c", "strat_d_472"])
    pd.testing.assert_series_equal(result, expected)

    # empty native_tf → no-op
    result_noop = _strip_tf_suffix_series(values, "")
    pd.testing.assert_series_equal(result_noop, values)


def test_candidate_output_to_signal_batch_recovers_suffix_mismatched_htf_events() -> None:
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["trend_donchian:donchian_72_12h"],
        activation_contexts=["all"],
        entry_idxs=[10],
        sides=[1],
        gross_bps=[15.0],
    )
    registry = _make_registry([("BTCUSDT", "trend_donchian:donchian_72", "all", 0.80)])

    result = _call(model_output, registry, native_tf="12h")

    assert len(result.events) == 1
    assert result.events[0].strategy_id == "trend_donchian:donchian_72_12h"
    assert result.events[0].quality_weight == pytest.approx(0.80)


def test_candidate_output_to_signal_batch_native_tf_4h_no_regression() -> None:
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["trend_donchian:donchian_72_4h"],
        activation_contexts=["all"],
        entry_idxs=[10],
        sides=[1],
        gross_bps=[15.0],
    )
    registry = _make_registry([("BTCUSDT", "trend_donchian:donchian_72_4h", "all", 0.80)])

    result = _call(model_output, registry, native_tf="4h")

    assert len(result.events) == 1
    assert result.events[0].quality_weight == pytest.approx(0.80)


def test_candidate_output_to_signal_batch_genuine_nonmembership_still_excluded() -> None:
    model_output = _make_model_output(
        symbols=["BTCUSDT"],
        strategy_ids=["no_such_strat:variant_xyz_12h"],
        activation_contexts=["all"],
        entry_idxs=[10],
        sides=[1],
        gross_bps=[15.0],
    )
    registry = _make_registry([("BTCUSDT", "different_strat:variant_abc", "all", 0.80)])

    result = _call(model_output, registry, native_tf="12h")

    assert len(result.events) == 0
