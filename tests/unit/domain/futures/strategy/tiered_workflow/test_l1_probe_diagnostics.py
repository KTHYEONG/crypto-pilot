from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    ProbeBreadthDiagnostics,
    _format_probe_diag,
    _l1_probe_diag_enabled,
    compute_probe_breadth_diagnostics,
)

VOL = np.ones((200, 200), dtype=np.float64)
SYM_TO_IDX = {f"s{i}": i for i in range(200)}


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock()
    cfg.l1_probe_top_k = 3
    cfg.l1_min_cross_section = 1
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _merged(
    exp: list[float],
    real: list[float],
    decision_idx: list[int],
    symbols: list[str],
    qw: list[float] | None = None,
    side: list[int] | None = None,
) -> pd.DataFrame:
    n = len(exp)
    df = pd.DataFrame({
        "decision_idx": decision_idx,
        "symbol": symbols,
        "strategy_id": ["s"] * n,
        "expected_gross_bps": np.asarray(exp, dtype=float),
        "realized_side_adjusted_gross_bps": np.asarray(real, dtype=float),
        "quality_weight": np.ones(n, dtype=float) if qw is None else np.asarray(qw, dtype=float),
    })
    if side is not None:
        df["side"] = np.asarray(side, dtype=int)
    return df


# ─── Scenario 1: Selection-inflation detection ────────────────────────────

def test_compute_probe_diag_detects_selection_inflation() -> None:
    exp = [100.0, 90.0, 80.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    real = [100.0, 90.0, 80.0, -50.0, -40.0, -30.0, -20.0, -10.0, -5.0, -1.0]
    decision_idx = [0] * 10
    symbols = [f"s{i}" for i in range(10)]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )

    assert diag is not None
    assert diag.probe_gross_by_k[3] > 0.0
    assert diag.probe_gross_by_k[-1] < diag.probe_gross_by_k[3]
    assert diag.avg_breadth_per_decision == 10.0
    assert diag.n_events == 10
    assert diag.n_decisions == 1


# ─── Scenario 2: Gross → net reversal by rt_cost ──────────────────────────

def test_compute_probe_diag_net_below_gross_by_rtcost() -> None:
    exp = [4.0, 4.0, 4.0]
    real = [4.0, 4.0, 4.0]
    decision_idx = [0, 0, 0]
    symbols = ["s0", "s1", "s2"]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg(expected_cost_bps=6.0)

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )

    assert diag is not None
    assert diag.probe_gross_by_k[3] == pytest.approx(4.0, abs=1e-9)
    assert diag.probe_net_by_k[3] == pytest.approx(-2.0, abs=1e-9)
    assert diag.rt_cost_bps == 6.0


# ─── Scenario 3: Rank-IC absence (random) ─────────────────────────────────

def test_compute_probe_diag_zero_rank_ic_when_no_signal() -> None:
    rng = np.random.RandomState(42)
    n = 200
    exp = rng.uniform(-10, 10, n).tolist()
    real = rng.uniform(-10, 10, n).tolist()
    decision_idx = [0] * n
    symbols = [f"s{i}" for i in range(n)]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )

    assert diag is not None
    assert abs(diag.rank_ic_all) < 0.2
    assert abs(diag.rank_ic_tstat) < 2.0


# ─── Scenario 4: Rank-IC positive (strong signal) ─────────────────────────

def test_compute_probe_diag_positive_rank_ic_when_aligned() -> None:
    rng = np.random.RandomState(42)
    n = 100
    exp_arr = np.linspace(-10.0, 10.0, n)
    real_arr = exp_arr + rng.normal(0, 1.0, n)
    exp = exp_arr.tolist()
    real = real_arr.tolist()
    decision_idx = [0] * n
    symbols = [f"s{i}" for i in range(n)]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )

    assert diag is not None
    assert diag.rank_ic_all > 0.5
    assert diag.rank_ic_tstat > 2.0


# ─── Scenario 5: Edge cases ───────────────────────────────────────────────

def test_compute_probe_diag_empty_returns_none() -> None:
    merged = pd.DataFrame()
    cfg = _make_cfg()
    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )
    assert diag is None


def test_compute_probe_diag_single_event_graceful() -> None:
    merged = _merged([5.0], [3.0], [0], ["s0"])
    cfg = _make_cfg()
    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )
    assert diag is not None
    assert diag.rank_ic_all == 0.0
    assert diag.rank_ic_tstat == 0.0
    assert diag.n_events == 1
    assert diag.n_decisions == 1
    assert 3 in diag.probe_gross_by_k
    assert -1 in diag.probe_gross_by_k


# ─── Scenario 6: Env gate ─────────────────────────────────────────────────

class TestL1ProbeDiagEnvGate:
    def test_disabled_by_default(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("L1_PROBE_DIAG", raising=False)
            assert not _l1_probe_diag_enabled()

    def test_disabled_with_zero(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("L1_PROBE_DIAG", "0")
            assert not _l1_probe_diag_enabled()

    def test_disabled_with_false(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("L1_PROBE_DIAG", "False")
            assert not _l1_probe_diag_enabled()

    def test_enabled_with_one(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("L1_PROBE_DIAG", "1")
            assert _l1_probe_diag_enabled()

    def test_enabled_with_true(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("L1_PROBE_DIAG", "true")
            assert _l1_probe_diag_enabled()


# ─── Helper: format probe diag ────────────────────────────────────────────

def test_format_probe_diag_output() -> None:
    diag = ProbeBreadthDiagnostics(
        fold_id=5,
        n_events=100,
        n_decisions=10,
        avg_breadth_per_decision=10.0,
        probe_gross_by_k={3: 90.0, 10: 50.0, 20: 30.0, -1: 11.4},
        probe_net_by_k={3: 84.0, 10: 44.0, 20: 24.0, -1: 5.4},
        rank_ic_all=0.35,
        rank_ic_tstat=3.72,
        realized_mean_all=11.4,
        realized_median_all=5.0,
        realized_pos_fraction_all=0.6,
        rt_cost_bps=6.0,
    )
    formatted = _format_probe_diag(diag)
    assert "[L1-PROBE-DIAG]" not in formatted
    assert "fold=5" in formatted
    assert "rank_ic=0.3500" in formatted
    assert "gross_k3=90.00" in formatted
    assert "net_k3=84.00" in formatted
    assert "rt_cost=6.0" in formatted


# ─── Regime decomposition (market regime code_1d) ─────────────────────────

def test_compute_probe_diag_regime_breakdown_by_code() -> None:
    # decision_idx 0,1 -> bull(0); 2,3 -> crisis(2)
    exp = [10.0, 12.0, 8.0, 9.0]
    real = [100.0, 80.0, -50.0, -30.0]
    decision_idx = [0, 1, 2, 3]
    symbols = ["s0", "s1", "s2", "s3"]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg(expected_cost_bps=5.0)
    regime_code = np.array([0, 0, 2, 2], dtype=np.int8)  # bull,bull,crisis,crisis

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0, regime_code_1d=regime_code,
    )

    assert diag is not None
    assert set(diag.regime_breakdown.keys()) == {"bull", "crisis"}
    bull_n, _bull_gross, bull_net, bull_pos, _bull_ic = diag.regime_breakdown["bull"]
    crisis_n, _c_gross, crisis_net, crisis_pos, _c_ic = diag.regime_breakdown["crisis"]
    assert bull_n == 2
    assert crisis_n == 2
    assert bull_net == pytest.approx(90.0 - 5.0)   # mean(100,80)-rt
    assert crisis_net == pytest.approx(-40.0 - 5.0)  # mean(-50,-30)-rt
    assert bull_pos == pytest.approx(1.0)
    assert crisis_pos == pytest.approx(0.0)


def test_compute_probe_diag_no_regime_when_code_none() -> None:
    merged = _merged([10.0, 8.0], [50.0, -20.0], [0, 1], ["s0", "s1"])
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0, regime_code_1d=None,
    )

    assert diag is not None
    assert diag.regime_breakdown == {}


# ─── Residual-alpha decomposition ─────────────────────────────────────────

def test_compute_probe_diag_residual_separates_beta_and_alpha() -> None:
    # 단일 bar, 4 이벤트. exp가 real과 동조 → residual_ic 양(+), selection_alpha 양(+).
    exp = [40.0, 30.0, 20.0, 10.0]
    real = [100.0, 80.0, 20.0, 0.0]   # per_bar_mean=50
    decision_idx = [0, 0, 0, 0]
    symbols = ["s0", "s1", "s2", "s3"]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )

    assert diag is not None
    assert diag.beta_edge_bps == pytest.approx(50.0)          # 횡단면 평균
    # top-3 by exp=[40,30,20] -> real[100,80,20] mean=66.67 - 50
    assert diag.selection_alpha_bps == pytest.approx(66.6667 - 50.0, abs=1e-2)
    assert diag.residual_ic > 0.9                              # exp가 잔차 순위 예측
    assert diag.n_residual_events == 4


def test_compute_probe_diag_residual_zero_when_single_event_bars() -> None:
    # 모든 bar가 단일 이벤트 → 횡단면 없음 → residual 정의 불가(0).
    merged = _merged([10.0, 20.0, 30.0], [50.0, -20.0, 40.0], [0, 1, 2], ["s0", "s1", "s2"])
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0,
    )

    assert diag is not None
    assert diag.n_residual_events == 0
    assert diag.beta_edge_bps == pytest.approx(0.0)
    assert diag.selection_alpha_bps == pytest.approx(0.0)
    assert diag.residual_ic == pytest.approx(0.0)


# ─── Phase-1: Bear-side directionality — regime_side_split ────────────────

def test_compute_probe_diag_bear_net_long_bias_detected() -> None:
    exp = [10.0, 8.0, 6.0, 5.0]
    real = [-10.0, -8.0, -6.0, 12.0]
    decision_idx = [0, 1, 2, 3]
    symbols = ["s0", "s1", "s2", "s3"]
    side = [1, 1, 1, -1]
    merged = _merged(exp, real, decision_idx, symbols, side=side)
    cfg = _make_cfg()
    regime_code = np.array([1, 1, 1, 1], dtype=np.int8)

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0, regime_code_1d=regime_code,
    )

    assert diag is not None
    bear = diag.regime_side_split["bear"]
    long_frac, long_mean, short_mean, n_long, n_short = bear
    assert long_frac == pytest.approx(0.75)
    assert long_mean == pytest.approx(-8.0)
    assert short_mean == pytest.approx(12.0)
    assert n_long == 3
    assert n_short == 1
    assert long_frac >= 0.65
    assert long_mean < 0.0 <= short_mean


def test_compute_probe_diag_side_split_defaults_long_when_missing() -> None:
    exp = [10.0, 8.0, 6.0]
    real = [-5.0, -3.0, 12.0]
    decision_idx = [0, 1, 2]
    symbols = ["s0", "s1", "s2"]
    merged = _merged(exp, real, decision_idx, symbols)
    cfg = _make_cfg()
    regime_code = np.array([1, 1, 1], dtype=np.int8)

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0, regime_code_1d=regime_code,
    )

    assert diag is not None
    bear = diag.regime_side_split["bear"]
    long_frac, _long_mean, short_mean, _n_long, n_short = bear
    assert long_frac == pytest.approx(1.0)
    assert n_short == 0
    assert short_mean == pytest.approx(0.0)


def test_compute_probe_diag_side_split_all_short_no_zero_div() -> None:
    exp = [5.0, 3.0]
    real = [5.0, 3.0]
    decision_idx = [0, 1]
    symbols = ["s0", "s1"]
    side = [-1, -1]
    merged = _merged(exp, real, decision_idx, symbols, side=side)
    cfg = _make_cfg()
    regime_code = np.array([1, 1], dtype=np.int8)

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0, regime_code_1d=regime_code,
    )

    assert diag is not None
    bear = diag.regime_side_split["bear"]
    long_frac, long_mean, short_mean, n_long, n_short = bear
    assert long_frac == pytest.approx(0.0)
    assert long_mean == pytest.approx(0.0)
    assert short_mean == pytest.approx(4.0)
    assert n_long == 0
    assert n_short == 2


def test_compute_probe_diag_side_split_empty_when_no_regime() -> None:
    merged = _merged([10.0, 8.0], [5.0, -3.0], [0, 1], ["s0", "s1"], side=[1, -1])
    cfg = _make_cfg()

    diag = compute_probe_breadth_diagnostics(
        merged=merged, volatility_2d=VOL, symbol_to_idx=SYM_TO_IDX,
        cfg=cfg, fold_id=0, seed=0, regime_code_1d=None,
    )

    assert diag is not None
    assert diag.regime_side_split == {}


def test_format_probe_diag_renders_side_split() -> None:
    diag = ProbeBreadthDiagnostics(
        fold_id=0,
        n_events=58,
        n_decisions=5,
        avg_breadth_per_decision=11.6,
        probe_gross_by_k={3: 30.0, -1: 5.0},
        probe_net_by_k={3: 24.0, -1: -1.0},
        rank_ic_all=0.15,
        rank_ic_tstat=1.15,
        realized_mean_all=5.0,
        realized_median_all=2.0,
        realized_pos_fraction_all=0.55,
        rt_cost_bps=6.0,
        regime_breakdown={"bear": (58, 5.0, -1.0, 0.55, 0.15)},
        regime_side_split={"bear": (0.78, -1.1, 0.3, 45, 13)},
    )
    formatted = _format_probe_diag(diag)
    assert "SIDE[bear]=long78%/lr-1.1/sr+0.3/nl45/ns13" in formatted
