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

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.market_regime import compute_market_regime_context
from src.domain.futures.strategy.tiered_workflow.l2_meta import filter_sleeves_by_bucket

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
        raw_mu=0.5, volatility=0.2, n_obs=1, t_stat=0.0,
        valid=True, beta_btc=None, quality_weight=1.0,
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
        (0, "familyA_tf", "4h"): 150.0,   # above floor
        (0, "familyB_tf", "1h"): 50.0,    # below floor
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

def test_filter_sleeves_by_bucket_unobserved_regime_returns_empty() -> None:
    """현재 regime이 bucket_edges에 없으면 edge=0 → 모든 sleeve 제거."""
    sigs = _dummy_sleeve_sigs()
    bucket_edges = {
        (0, "familyA_tf", "4h"): 200.0,   # regime=0만 있음
        (0, "familyB_tf", "1h"): 150.0,
    }

    # regime_now=1 (bull_volatile) — no bucket edges
    result = filter_sleeves_by_bucket(sigs, bucket_edges, regime_now=1, edge_floor_bps=100.0)

    assert len(result) == 0
