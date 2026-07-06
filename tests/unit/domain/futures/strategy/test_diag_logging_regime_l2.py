"""Regime→L2 bucket routing 진단 로깅 테스트.

Target:
    - market_regime.py: Step F — [REGIME-DIST] DEBUG log
    - l2_meta.py: Step E — [L2-BUCKET-STATS] / [L2-BUCKET-EDGE-FIT] DEBUG logs
    - awf_sim.py: Steps B/A/C/D — [L2-REGIME-OCC], [L2-BUCKET-MAP], [L2-BUCKET-FILTER], [L2-BUCKET-DROP] DEBUG logs

Scenarios:
    1. compute_market_regime_context → DEBUG 로그에 regime 분포가 정확히 출력되는가
    2. filter_sleeves_by_bucket(empty_bucket_edges) → 모든 sleeve 제거
    3. filter_sleeves_by_bucket(empty_sleeves) → 빈 dict 반환
    4. filter_sleeves_by_bucket(known_edge) → edge>floor인 sleeve만 통과
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.market_regime import compute_market_regime_context
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    RegimeDebugDiagnostics,
    RegimeGranularityDebugStat,
    RegimeRoutingDiagnostics,
)
from src.domain.futures.strategy.tiered_workflow.l2_meta import filter_sleeves_by_bucket
from src.domain.futures.strategy.tiered_workflow.regime_debug import (
    replace_selected_regime_debug_diagnostics,
)

# Reuse the existing test helper from test_market_regime.py
from tests.unit.domain.futures.strategy.test_market_regime import _make_aligned

# ---------------------------------------------------------------------------
# Scenario 3: Regime Occupancy — compute_market_regime_context DEBUG log
# ---------------------------------------------------------------------------


def test_compute_market_regime_context_emits_regime_dist_debug_log(caplog: pytest.LogCaptureFixture) -> None:
    """compute_market_regime_context가 [REGIME-DIST] DEBUG 로그를 방출하는지 검증."""
    aligned = _make_aligned()

    with caplog.at_level(logging.DEBUG, logger="src.domain.futures.strategy.market_regime"):
        ctx = compute_market_regime_context(aligned=aligned)

    assert ctx.code_1d is not None
    assert ctx.code_1d.shape[0] == aligned.close_2d.shape[0]

    # [REGIME-DIST] 로그가 존재하는지 확인
    regime_dist_records = [r for r in caplog.records if "[REGIME-DIST]" in r.getMessage()]
    assert len(regime_dist_records) >= 1, "[REGIME-DIST] DEBUG log not found"

    msg = regime_dist_records[0].getMessage()
    assert "total_bars=" in msg
    assert "bull_quiet" in msg or "bear_quiet" in msg or "transition" in msg


def test_compute_market_regime_context_emits_debug_only_when_enabled(caplog: pytest.LogCaptureFixture) -> None:
    """DEBUG 레벨이 아닐 때 [REGIME-DIST] 로그가 방출되지 않는지 검증."""
    aligned = _make_aligned()

    with caplog.at_level(logging.INFO, logger="src.domain.futures.strategy.market_regime"):
        compute_market_regime_context(aligned=aligned)

    regime_dist_records = [r for r in caplog.records if "[REGIME-DIST]" in r.getMessage()]
    assert len(regime_dist_records) == 0


def test_compute_market_regime_context_regime_code_range() -> None:
    """모든 regime code가 유효 범위(0-5) 내에 있는지 검증."""
    aligned = _make_aligned()
    ctx = compute_market_regime_context(aligned=aligned)

    unique_codes = np.unique(ctx.code_1d)
    for code in unique_codes:
        assert 0 <= int(code) <= 5, f"Invalid regime code: {code}"


# ---------------------------------------------------------------------------
# Scenario 2: Bucket Filter Zero-Pass — empty bucket_edges
# ---------------------------------------------------------------------------


def _dummy_sleeve_sigs() -> dict[tuple[str, str], SymbolSignal]:
    """Minimal sleeve sigs dict for filter tests."""
    sig = SymbolSignal(
        raw_mu=0.5,
        volatility=0.2,
        n_obs=1,
        t_stat=0.0,
        valid=True,
        beta_btc=None,
        quality_weight=1.0,
    )
    return {
        ("BTCUSDT", "familyA_tf_4h"): sig,
        ("ETHUSDT", "familyB_tf_1h"): sig,
    }


def test_filter_sleeves_by_bucket_empty_edges_returns_empty() -> None:
    """bucket_edges가 비어있으면 모든 sleeve가 제거되어야 한다."""
    sigs = _dummy_sleeve_sigs()
    bucket_edges: dict[tuple[int, str, str], float] = {}

    result = filter_sleeves_by_bucket(sigs, bucket_edges, regime_now=0, edge_floor_bps=100.0)

    assert len(result) == 0


def test_filter_sleeves_by_bucket_with_known_edges_passes_only_above_floor() -> None:
    """edge_floor_bps 이상인 bucket의 sleeve만 통과해야 한다.

    참고: _parse_meta_group_ids("familyA_tf_4h") → ("familyA_tf", "4h")
    """
    sigs = _dummy_sleeve_sigs()
    bucket_edges = {
        (0, "familyA_tf", "4h"): 150.0,  # above floor
        (0, "familyB_tf", "1h"): 50.0,  # below floor
    }

    result = filter_sleeves_by_bucket(sigs, bucket_edges, regime_now=0, edge_floor_bps=100.0)

    assert ("BTCUSDT", "familyA_tf_4h") in result
    assert ("ETHUSDT", "familyB_tf_1h") not in result


def test_filter_sleeves_by_bucket_edge_matches_floor_boundary() -> None:
    """edge == floor_bps인 경우 통과하지 않아야 한다 (> not >=)."""
    sigs = _dummy_sleeve_sigs()
    bucket_edges = {
        (0, "familyA_tf", "4h"): 100.0,  # exactly floor
    }

    result = filter_sleeves_by_bucket(sigs, bucket_edges, regime_now=0, edge_floor_bps=100.0)

    assert len(result) == 0


# ---------------------------------------------------------------------------
# Scenario 4: Edge Case — Empty Sleeve Pool
# ---------------------------------------------------------------------------


def test_filter_sleeves_by_bucket_empty_sleeves_returns_empty() -> None:
    """sleeve_sigs가 비어있으면 즉시 빈 dict를 반환해야 한다."""
    result = filter_sleeves_by_bucket({}, {}, regime_now=0, edge_floor_bps=100.0)
    assert result == {}


def test_filter_sleeves_by_bucket_empty_sleeves_nonempty_edges_returns_empty() -> None:
    """sleeve_sigs는 비었지만 bucket_edges가 있어도 빈 dict를 반환해야 한다."""
    bucket_edges = {(0, "familyA", "4h"): 200.0}
    result = filter_sleeves_by_bucket({}, bucket_edges, regime_now=0, edge_floor_bps=100.0)
    assert result == {}


# ---------------------------------------------------------------------------
# Bucket Unobserved Key — test edge=0 fallback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step G: Bucket hit ratio tracking
# ---------------------------------------------------------------------------


def test_bucket_hit_ratio_active_same_as_hit() -> None:
    """n_active=10, n_hit=10 → hit_pct=100%"""
    _active, _hit = 10, 10
    _pct = _hit / max(_active, 1) * 100.0
    assert _pct == 100.0


def test_bucket_hit_ratio_no_hits() -> None:
    """n_active=5, n_hit=0 → hit_pct=0%"""
    _active, _hit = 5, 0
    _pct = _hit / max(_active, 1) * 100.0
    assert _pct == 0.0


def test_bucket_hit_ratio_zero_active() -> None:
    """n_active=0 → skip (division by zero guard)"""
    _active, _hit = 0, 0
    _pct = _hit / max(_active, 1) * 100.0
    assert _pct == 0.0


# ---------------------------------------------------------------------------
# Step H: Regime shift JS divergence
# ---------------------------------------------------------------------------


def test_js_divergence_identical_distributions() -> None:
    """동일 분포 → JS divergence ≈ 0.0"""
    _freq = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=np.float64)
    _m = (_freq + _freq) / 2.0
    _js = 0.0
    for _p, _q in zip(_freq, _m, strict=True):
        if _p > 0:
            _js += _p * np.log2(_p / _q) if _q > 0 else 0.0
    for _p, _q in zip(_freq, _m, strict=True):
        if _p > 0:
            _js += _p * np.log2(_p / _q) if _q > 0 else 0.0
    _js /= 2.0
    assert abs(_js) < 1e-10


def test_js_divergence_shifted_distribution() -> None:
    """fit=90% transition vs OOS=90% bull_quiet → JS divergence > 0.15"""
    _fit_freq = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.1], dtype=np.float64)
    _oos_freq = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float64)
    _m = (_fit_freq + _oos_freq) / 2.0
    _js = 0.0
    for _p, _q in zip(_fit_freq, _m, strict=True):
        if _p > 0:
            _js += _p * np.log2(_p / _q) if _q > 0 else 0.0
    for _p, _q in zip(_oos_freq, _m, strict=True):
        if _p > 0:
            _js += _p * np.log2(_p / _q) if _q > 0 else 0.0
    _js /= 2.0
    assert _js > 0.15, f"JS divergence too low: {_js:.4f}"


# ---------------------------------------------------------------------------
# Step J: OOS vs Fit bucket edge comparison
# ---------------------------------------------------------------------------


def test_bucket_oos_compute_rmse_mae_bias() -> None:
    """fit=(50,20,-10) oos=(55,5,-8) → RMSE/MAE/bias/corr"""
    _fit_vals = np.array([50.0, 20.0, -10.0], dtype=np.float64)
    _oos_vals = np.array([55.0, 5.0, -8.0], dtype=np.float64)
    _errors = _oos_vals - _fit_vals
    _rmse = float(np.sqrt(np.mean(_errors**2)))
    _mae = float(np.mean(np.abs(_errors)))
    _bias = float(np.mean(_errors))
    _corr = float(np.corrcoef(_fit_vals, _oos_vals)[0, 1])

    assert abs(_rmse - 9.20) < 0.1, f"RMSE={_rmse:.2f}"
    assert abs(_mae - 7.33) < 0.1, f"MAE={_mae:.2f}"
    assert abs(_bias + 2.667) < 0.1, f"Bias={_bias:.2f}"
    assert _corr > 0.90, f"Corr={_corr:.4f}"


def test_bucket_oos_no_common_buckets() -> None:
    """fit_regime=0 only vs OOS_regime=4 only → n_common=0, skip"""
    _fit_edges = {(0, "famA", "4h"): 50.0}
    _oos_edges: dict[tuple[int, str, str], float] = {(4, "famA", "4h"): 30.0}
    _common = set(_fit_edges) & set(_oos_edges)
    assert len(_common) == 0


def test_bucket_oos_fewer_than_3_common_returns_zero_corr() -> None:
    """n_common < 3 → correlation = 0.0"""
    _fit_vals = np.array([50.0, 20.0], dtype=np.float64)
    _oos_vals = np.array([55.0, 5.0], dtype=np.float64)
    _corr = float(np.corrcoef(_fit_vals, _oos_vals)[0, 1]) if len(_fit_vals) >= 3 else 0.0
    assert _corr == 0.0


def test_bucket_oos_identifies_underover_fit() -> None:
    """sort by (oos-fit): top = underfit, bottom = overfit"""
    _common = {(0, "famA", "4h"), (0, "famB", "4h"), (0, "famC", "4h")}
    _fit_edges = {(0, "famA", "4h"): 50.0, (0, "famB", "4h"): 20.0, (0, "famC", "4h"): -10.0}
    _oos_edges = {(0, "famA", "4h"): 55.0, (0, "famB", "4h"): 5.0, (0, "famC", "4h"): -8.0}

    _underfit = sorted(_common, key=lambda _bk: _oos_edges[_bk] - _fit_edges[_bk], reverse=True)[:5]
    _overfit = sorted(_common, key=lambda _bk: _oos_edges[_bk] - _fit_edges[_bk])[:5]

    # underfit = oos >> fit (positive surplus)
    assert _underfit[0] == (0, "famA", "4h")  # surplus = +5
    # overfit = fit >> oos (negative surplus)
    assert _overfit[0] == (0, "famB", "4h")  # deficit = -15


def test_bucket_oos_edge_case_under3() -> None:
    """n=2 → corr=0.0, 하지만 rmse/mae/bias는 정상 계산"""
    _fit_vals = np.array([50.0, 20.0], dtype=np.float64)
    _oos_vals = np.array([55.0, 5.0], dtype=np.float64)
    _errors = _oos_vals - _fit_vals
    _rmse = float(np.sqrt(np.mean(_errors**2)))
    _mae = float(np.mean(np.abs(_errors)))
    _bias = float(np.mean(_errors))
    _corr = 0.0  # < 3 buckets
    assert abs(_rmse - 11.18) < 0.1
    assert abs(_mae - 10.0) < 0.1
    assert _bias == -5.0
    assert _corr == 0.0


def test_l2_regime_log_reports_effective_state_count() -> None:
    """[REGIME] 로그 계약이 3-state 표형식 요약을 노출하는지 검증."""
    source = Path("src/execution/opt_main_futures.py").read_text(encoding="utf-8")

    assert "[REGIME]" in source
    assert "metric        | value" in source
    assert "compression   | %s" in source
    assert "states        | 3" in source
    assert "distribution  | %s" in source
    assert "policy_mode   | %s" in source
    assert "policy_source | fit/cal" in source
    assert "oos_debug     | evaluation only" in source
    assert "🟢 stable" in source or "🟠 unstable" in source
    assert "[REGIME] DEBUG" in source
    assert "global_reliable" in source
    assert "n_downweight" in source
    assert "n_hard_block_eligible" in source
    assert "sign_consistency_ratio" in source
    assert "hard_block_enabled" in source
    assert "mean_cal_lift_bps" in source
    assert "raw_states=6" not in source
    assert "proof_failed path=%s effective_states=%d" in source


def test_awf_sim_consumes_regime_routing_diagnostics_for_debug() -> None:
    """AWF debug path가 cache.regime_routing_diagnostics를 직접 소비하는지 검증."""
    source = Path("src/domain/futures/strategy/tiered_workflow/awf_sim.py").read_text(encoding="utf-8")

    assert "cache.regime_routing_diagnostics" in source
    assert "[L2-REGIME-DIAG]" in source
    assert "[L2-REGIME-POLICY]" in source
    assert "source=fit/cal" in source
    assert "OOS DEBUG = evaluation only" in source
    assert "active_state_names" in source
    assert "n_hard_block_eligible" in source
    assert "sign_consistency_ratio" in source
    assert "hard_block_enabled" in source


def test_regime_debug_log_tables_are_declared_in_source() -> None:
    """DEBUG 표형식 출력 계약이 source 상에 노출되는지 검증."""
    opt_source = Path("src/execution/opt_main_futures.py").read_text(encoding="utf-8")
    awf_source = Path("src/domain/futures/strategy/tiered_workflow/awf_sim.py").read_text(encoding="utf-8")

    assert "[REGIME-DEBUG-GRANULARITY]" in opt_source
    assert "[REGIME-DEBUG-CELLS]" in opt_source
    assert "compression_loss_bps" in opt_source
    assert "[REGIME] DEBUG" in opt_source
    assert "mean_confidence" in opt_source
    assert "n_hard_block_eligible" in opt_source
    assert "sign_consistency_ratio" in opt_source
    assert "hard_block_enabled" in opt_source
    assert "[REGIME-DEBUG-SELECTED]" in awf_source
    assert "state  | bars | realized_return_bps" in awf_source


def test_replace_selected_regime_debug_diagnostics_uses_realized_bar_returns() -> None:
    routing_diag = RegimeRoutingDiagnostics(
        active_state_count=3,
        active_state_names=("bull", "bear", "crisis"),
        compression_enabled=True,
        proof_passed=False,
        conditioning_path="pooled_fallback",
        mean_lift_bps=0.0,
        n_eff=0.0,
        nw_tstat=0.0,
        deflated_sharpe=0.0,
        fold_pass_ratio=0.0,
        n_folds_evaluated=0,
        bucket_hit_pct_by_fold=(),
        js_divergence_by_fold=(),
        debug_diagnostics=RegimeDebugDiagnostics(
            granularity_stats=(
                RegimeGranularityDebugStat(
                    label="effective_3",
                    state_count=3,
                    proof_passed=False,
                    conditioning_path="pooled_fallback",
                    mean_lift_bps=0.0,
                    nw_tstat=0.0,
                    fold_pass_ratio=0.0,
                    n_folds_evaluated=0,
                    bucket_hit_pct_mean=0.0,
                    oos_cell_ic=0.0,
                    oos_cell_rmse_bps=0.0,
                    oos_cell_bias_bps=0.0,
                ),
            ),
            top_positive_cells=(),
            top_negative_cells=(),
            worst_error_cells=(),
            compression_loss_bps=0.0,
            selected_regime_return_bps=(999.0, 999.0, 999.0),
            selected_regime_bar_count=(9, 9, 9),
        ),
    )

    updated = replace_selected_regime_debug_diagnostics(
        routing_diag=routing_diag,
        selected_return_sum_bps=np.array([30.0, -20.0, 0.0], dtype=np.float64),
        selected_bar_count=np.array([2, 4, 0], dtype=np.int64),
    )

    assert updated.debug_diagnostics is not None
    assert updated.debug_diagnostics.selected_regime_return_bps == pytest.approx((15.0, -5.0, 0.0))
    assert updated.debug_diagnostics.selected_regime_bar_count == (2, 4, 0)


def test_filter_sleeves_by_bucket_when_current_regime_bucket_missing_returns_empty() -> None:
    """현재 regime이 bucket_edges에 없으면 edge=0 → 모든 sleeve 제거."""
    sigs = _dummy_sleeve_sigs()
    bucket_edges = {
        (0, "familyA_tf", "4h"): 200.0,  # regime=0만 있음
        (0, "familyB_tf", "1h"): 150.0,
    }

    # regime_now=1 (bull_volatile) — no bucket edges
    result = filter_sleeves_by_bucket(sigs, bucket_edges, regime_now=1, edge_floor_bps=100.0)

    assert len(result) == 0
