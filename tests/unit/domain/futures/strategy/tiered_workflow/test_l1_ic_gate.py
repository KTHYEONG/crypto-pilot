from __future__ import annotations

from dataclasses import replace

from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.signal_selection import evaluate_layer1_readiness


def _fold(
    fold_id: int,
    rank_ic_all: float = 0.0,
    rank_ic_tstat: float = 0.0,
    probe_bps: float = 0.0,
    probe_lcb_bps: float = 0.0,
    probe_series_bps: tuple[float, ...] = (),
    passed: bool = True,
) -> Layer1FoldReadiness:
    return Layer1FoldReadiness(
        fold_id=fold_id,
        registry_source_end_idx=10,
        outer_oos_start_idx=11,
        outer_oos_end_idx=20,
        ready_symbols=("BTC", "ETH", "SOL"),
        matched_event_count=50,
        unmatched_event_count=0,
        realized_match_ratio=1.0,
        unique_decision_count=10,
        prediction_unique_count=5,
        opportunity_ic=None,
        opportunity_ic_tstat=0.0,
        probe_bps=probe_bps,
        probe_lcb_bps=probe_lcb_bps,
        probe_series_bps=probe_series_bps,
        effective_symbol_count=3.0,
        passed=passed,
        blockers=(),
        rank_ic_all=rank_ic_all,
        rank_ic_tstat=rank_ic_tstat,
    )


def _base_cfg() -> CandidateStrategyConfig:
    return CandidateStrategyConfig(
        l1_sym_count_mode="effective_n",
        l1_min_effective_sym_n=3.0,
        l1_min_fold_ratio=0.50,
        l1_min_realized_match_ratio=0.90,
        l1_probe_lcb_pooled=True,
        l1_min_probe_bps=0.0,
    )


# ─── S1: Happy path — strong IC across all folds → PASS ──────────────────────


def test_s1_happy_strong_ic_passes() -> None:
    """Fold별 IC=[0.12,0.10,0.11,0.09], t 모두 > 2, sign_ratio=1.0 → gate PASS."""
    fold_reports = (
        _fold(0, rank_ic_all=0.12, rank_ic_tstat=3.5, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(1, rank_ic_all=0.10, rank_ic_tstat=2.8, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(2, rank_ic_all=0.11, rank_ic_tstat=3.0, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(3, rank_ic_all=0.09, rank_ic_tstat=2.5, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
    )
    cfg = _base_cfg()
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )
    assert report.passed is True
    assert len(report.blockers) == 0


# ─── S2: Zero-IC block — IC near zero with inconsistent sign → BLOCK ─────────


def test_s2_zero_ic_is_monitoring_only_no_block() -> None:
    """IC≈0, IC hard gate 보류 → gate는 다른 메트릭으로만 PASS/BLOCK."""
    fold_reports = (
        _fold(0, rank_ic_all=-0.10, rank_ic_tstat=0.5, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(1, rank_ic_all=0.01, rank_ic_tstat=0.3, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(2, rank_ic_all=0.00, rank_ic_tstat=0.1, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(3, rank_ic_all=-0.02, rank_ic_tstat=-0.4, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
    )
    cfg = _base_cfg()
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )
    # IC hard gate 제거 — IC는 DEBUG 모니터링만, gate는 다른 메트릭으로 통과
    assert report.passed is True
    assert len(report.blockers) == 0
    # IC check 자체가 check_specs에 없어야 함
    assert not any(c.key.startswith("ic_") for c in report.checks)


# ─── S3: Single-fold luck — IC hard gate 없으므로 PASS (DEBUG monitor only)


def test_s3_single_fold_luck_passes_no_ic_gate() -> None:
    """1개 fold만 t=8, 나머지≈0 but IC hard gate 없음 → PASS."""
    fold_reports = (
        _fold(0, rank_ic_all=0.15, rank_ic_tstat=8.0, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(1, rank_ic_all=-0.01, rank_ic_tstat=-0.2, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(2, rank_ic_all=-0.02, rank_ic_tstat=-0.5, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(3, rank_ic_all=0.03, rank_ic_tstat=0.8, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
    )
    cfg = _base_cfg()
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )
    assert report.passed is True
    assert not any(c.key.startswith("ic_") for c in report.checks)


# ─── S4: probe_metric=breadth — probe_lcb uses gross_all path (probe values unaffected)


def test_s4_probe_metric_breadth_uses_all_symbols() -> None:
    """probe_metric=breadth: probe_lcb computed from full breadth, not top-k."""
    fold_reports = (
        _fold(0, rank_ic_all=0.12, rank_ic_tstat=3.5, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(1, rank_ic_all=0.10, rank_ic_tstat=2.8, probe_bps=50.0, probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
    )
    cfg = replace(_base_cfg(), l1_probe_metric="breadth")
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )
    assert report.passed is True
    assert len(report.blockers) == 0


# ─── S5: Edge case — empty fold reports / IC NaN → graceful finite guard


def test_s5_empty_fold_reports_graceful() -> None:
    """fold reports가 빈 tuple → BLOCK, not crash."""
    cfg = _base_cfg()
    report = evaluate_layer1_readiness(
        fold_reports=(), fold_cov=0.0, trade_scope_count=0, cfg=cfg, seed=42,
    )
    assert report.passed is False
    assert len(report.checks) > 0


def test_s5_nan_ic_graceful() -> None:
    """IC가 NaN이어도 gate는 정상 동작 (IC hard gate 없음)."""
    nan = float("nan")
    fold_reports = (
        _fold(0, rank_ic_all=nan, rank_ic_tstat=nan, probe_bps=50.0,
              probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
        _fold(1, rank_ic_all=nan, rank_ic_tstat=nan, probe_bps=50.0,
              probe_lcb_bps=20.0, probe_series_bps=(50.0,)),
    )
    cfg = _base_cfg()
    report = evaluate_layer1_readiness(
        fold_reports=fold_reports, fold_cov=1.0, trade_scope_count=57, cfg=cfg, seed=42,
    )
    # IC hard gate 제거 — NaN IC가 gate decision을 망가뜨리지 않음
    assert not any(c.key.startswith("ic_") for c in report.checks)
