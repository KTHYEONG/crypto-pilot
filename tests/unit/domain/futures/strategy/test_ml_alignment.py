from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
from src.domain.futures.strategy.candidate_portfolio import (
    _resolve_breakeven_floor,
    _selection_component_frame,
    select_candidate_events_for_portfolio,
)
from src.domain.futures.strategy.common import alignment
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _frame(n: int = 6) -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": np.linspace(100.0, 106.0, n),
            "high": np.linspace(101.0, 107.0, n),
            "low": np.linspace(99.0, 105.0, n),
            "close": np.linspace(100.5, 106.5, n),
            "volume": np.linspace(1000.0, 1200.0, n),
        }
    )


def test_align_data_maps_funding_defaults_to_zero_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    out = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert out.funding_2d.shape == (len(frame), 1)
    assert np.allclose(out.funding_2d[:, 0], 0.0)
    assert np.all(out.active_mask[:, 0])
    assert np.all(out.warm_mask[:, 0])
    assert not np.any(out.entry_block_mask[:, 0])
    assert not np.any(out.kill_mask[:, 0])


def test_align_data_maps_missing_ohlcv_column_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame().drop(columns=["volume"])
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    with pytest.raises(ValueError, match="missing required column: volume"):
        alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")


def test_align_data_maps_uses_first_finite_metadata_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().assign(
        cluster_id=[3.0, np.nan, 9.0, 9.0, 9.0, 9.0],
        beta_vs_market=[1.2, np.nan, 2.8, 2.8, 2.8, 2.8],
        cluster_size=[2.0, np.nan, 6.0, 6.0, 6.0, 6.0],
        anchor_cluster_member=[1.0, np.nan, 0.0, 0.0, 0.0, 0.0],
    )
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    out = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert out.cluster_id_1d is not None
    assert out.beta_vs_market_1d is not None
    assert out.cluster_size_1d is not None
    assert out.anchor_cluster_1d is not None
    assert out.cluster_id_1d.tolist() == [3.0]
    assert np.allclose(out.beta_vs_market_1d, np.array([1.2], dtype=np.float32))
    assert out.cluster_size_1d.tolist() == [2.0]
    assert out.anchor_cluster_1d.tolist() == [1.0]


def test_align_data_maps_taker_buy_and_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().assign(
        taker_buy_base=np.linspace(100.0, 200.0, 6),
        trades=np.linspace(10.0, 60.0, 6),
    )
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    out = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert out.taker_buy_2d is not None
    assert out.trades_2d is not None
    assert out.taker_buy_2d.shape == (len(frame), 1)
    assert out.trades_2d.shape == (len(frame), 1)
    assert np.allclose(out.taker_buy_2d[:, 0], frame["taker_buy_base"].to_numpy())
    assert np.allclose(out.trades_2d[:, 0], frame["trades"].to_numpy())


def test_align_data_maps_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any
    frame = _frame()
    data_maps = {"BTCUSDT": {"4h": frame}}
    
    call_count = 0
    
    def fake_compute(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}}
        
    monkeypatch.setattr(alignment, "compute_multi_alignment_info", fake_compute)
    
    # 캐시 비우기
    alignment._ALIGNED_DATA_MAPS_CACHE.clear()
    
    # 1. 첫 실행 -> 캐시 미스
    out1 = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert call_count == 1
    
    # 2. 두 번째 실행 -> 캐시 히트
    out2 = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert call_count == 1
    assert out1 is out2
    
    # 3. 다른 인자 -> 캐시 미스
    _ = alignment.align_data_maps(data_maps, ["BTCUSDT", "ETHUSDT"], "4h")
    assert call_count == 2


# ---------------------------------------------------------------------------
# Selection utility mode contract tests
# ---------------------------------------------------------------------------


def _candidate_frame(n: int = 10) -> pd.DataFrame:
    """Synthetic candidate events with large downside_drag to stress EU modes."""
    rng = np.random.default_rng(42)
    dt = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": dt,
            "symbol": [f"SYM{i % 3}" for i in range(n)],
            "p_pass": rng.uniform(0.45, 0.70, n),
            "mu_net_decision_bps": rng.uniform(5.0, 25.0, n),
            # q10 is MAE-based (path-risk proxy, magnitude ~200bps) — much larger than mu
            "q10_net_bps": rng.uniform(-300.0, -150.0, n),
            "turnover_proxy": np.ones(n),
            "cost_floor_bps": np.full(n, 7.5),
            "edge_after_hurdle_bps": rng.uniform(-10.0, 20.0, n),
            "sl_thr_bps": np.full(n, 250.0),
        }
    )


def test_selection_utility_mode_direct_excludes_downside_drag() -> None:
    # Arrange
    cfg = CandidateStrategyConfig(selection_utility_mode="expected_edge_direct")
    events = _candidate_frame()

    # Act
    frame = _selection_component_frame(
        events=events, cfg=cfg
    )

    # Assert — expected_utility_bps must equal mu_net_decision_bps exactly
    expected = pd.to_numeric(events["mu_net_decision_bps"], errors="coerce")
    actual = pd.to_numeric(frame["expected_utility_bps"], errors="coerce")
    pd.testing.assert_series_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_names=False)


def test_selection_utility_mode_additive_preserves_legacy() -> None:
    # Arrange
    cfg = CandidateStrategyConfig(selection_utility_mode="additive_drag")
    events = _candidate_frame()

    # Act
    frame = _selection_component_frame(
        events=events, cfg=cfg
    )

    # Assert — additive_drag formula: mu - downside_penalty * |q10| - turnover_penalty
    mu = events["mu_net_decision_bps"].to_numpy(dtype=np.float64)
    q10 = np.clip(events["q10_net_bps"].to_numpy(dtype=np.float64), None, 0.0)
    expected_eu = mu - float(cfg.downside_penalty) * np.abs(q10) - float(cfg.turnover_penalty)
    actual = frame["expected_utility_bps"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(actual, expected_eu, rtol=1e-9)


def test_fold_adaptive_breakeven_floor_uses_cost_quantile() -> None:
    # Arrange
    cfg = CandidateStrategyConfig(
        breakeven_floor_mode="fold_adaptive",
        breakeven_floor_cost_quantile=0.50,
        min_net_floor_cost_fraction=0.50,
    )
    events = _candidate_frame()
    # Override cost_floor_bps with a known distribution
    events["cost_floor_bps"] = np.array([5.0, 10.0, 15.0, 7.0, 8.0, 12.0, 6.0, 9.0, 11.0, 14.0], dtype=np.float64)

    # Act
    floor = _resolve_breakeven_floor(events, cfg)

    # Assert — median(cost_floor_bps) * min_net_floor_cost_fraction
    expected = float(np.median(events["cost_floor_bps"].to_numpy())) * 0.50
    assert floor == pytest.approx(expected, rel=1e-9)


def test_expected_edge_direct_selects_when_additive_blocks() -> None:
    # Arrange — large q10 makes additive EU deeply negative; direct uses mu only
    cfg_additive = CandidateStrategyConfig(
        selection_utility_mode="additive_drag",
        selection_min_expected_utility_bps=0.0,
        min_net_floor_cost_fraction=0.0,
    )
    cfg_direct = CandidateStrategyConfig(
        selection_utility_mode="expected_edge_direct",
        selection_min_expected_utility_bps=0.0,
        min_net_floor_cost_fraction=0.0,
    )
    events = _candidate_frame()
    # Force a scenario where additive drag is extreme (q10 = -500bps)
    events["q10_net_bps"] = -500.0

    # Act
    frame_add = _selection_component_frame(
        events=events, cfg=cfg_additive
    )
    frame_dir = _selection_component_frame(
        events=events, cfg=cfg_direct
    )
    eligible_additive = int((frame_add["expected_utility_bps"] >= 0.0).sum())
    eligible_direct = int((frame_dir["expected_utility_bps"] >= 0.0).sum())

    # Assert — additive mode blocks all (EU << 0), direct mode passes all (mu > 0)
    assert eligible_additive == 0, f"expected 0 eligible in additive mode, got {eligible_additive}"
    assert eligible_direct == len(events), f"expected all eligible in direct mode, got {eligible_direct}"


def test_production_topk_sorts_by_mu_in_direct_mode() -> None:
    """Production topk must rank by mu_net (expected_utility_bps) when mode=expected_edge_direct."""
    # Arrange — candidates with deliberately inverted mu vs additive utility ordering
    # Candidate A: high mu (+30), extreme q10 (-500) → additive EU = 0.6*30 - 0.4*500 - 0.5 ≈ -182, mu=30
    # Candidate B: low mu (+5),  moderate q10 (-20) → additive EU = 0.6*5 - 0.4*20 - 0.5 ≈ -5.5, mu=5
    # Additive would pick B (less negative). Direct mode should pick A (higher mu).
    rng = np.random.default_rng(0)
    dt = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")] * 2)
    events = pd.DataFrame(
        {
            "datetime": dt,
            "symbol": ["SYM0", "SYM1"],
            "family": ["f", "f"],
            "variant": ["v", "v"],
            "side": [1, 1],
            "p_pass": [0.60, 0.60],
            "mu_net_decision_bps": [30.0, 5.0],   # A >> B
            "q10_net_bps": [-500.0, -20.0],        # A much worse than B
            "q90_net_bps": [60.0, 10.0],
            "turnover_proxy": [1.0, 1.0],
            "cost_floor_bps": [7.5, 7.5],
            "edge_after_hurdle_bps": [30.0, 5.0],
            "sl_thr_bps": [250.0, 250.0],
            "raw_score": rng.uniform(0.5, 1.0, 2),
            "score_z": rng.uniform(0, 1, 2),
            "entry_idx": [100, 100],
            "expected_holding_bars": [6, 6],
            "min_holding_bars": [1, 1],
            "stop_atr_mult": [1.5, 1.5],
            "take_profit_atr_mult": [2.0, 2.0],
            "side_flipped": [False, False],
        }
    )
    # utility_score reflects additive formula (as produced by edge model)
    additive_a = 0.60 * 30.0 - (1 - 0.60) * 500.0 - 0.5  # ≈ -182
    additive_b = 0.60 * 5.0 - (1 - 0.60) * 20.0 - 0.5    # ≈ -5.5
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.60, 0.60]),
        mu_gross_bps=np.array([30.0, 5.0]),
        mu_net_decision_bps=np.array([30.0, 5.0]),
        q10_net_bps=np.array([-500.0, -20.0]),
        q90_net_bps=np.array([60.0, 10.0]),
        utility_score=np.array([additive_a, additive_b]),  # additive order: B > A
        selection_thresholds={},
    )
    cfg_direct = CandidateStrategyConfig(
        selection_utility_mode="expected_edge_direct",
        selection_min_expected_utility_bps=-1000.0,  # open floor
        min_net_floor_cost_fraction=0.0,
        catastrophic_shortfall_bps=1000.0,           # don't veto on q10
        selection_top_quantile=0.50,                 # keep 1 of 2
    )

    # Act
    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg_direct)

    # Assert — candidate A (mu=30) must be selected, not B (mu=5)
    assert len(selected) == 1, f"expected 1 selected, got {len(selected)}"
    assert float(selected["mu_net_decision_bps"].iloc[0]) == pytest.approx(30.0), (
        f"wrong candidate selected: mu={selected['mu_net_decision_bps'].iloc[0]}"
    )
