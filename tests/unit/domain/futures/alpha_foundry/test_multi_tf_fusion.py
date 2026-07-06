from __future__ import annotations

import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.multi_tf_fusion import (
    fuse_multi_timeframe_evidence,
)

_COLS = [
    "run_id", "timeframe", "family", "variant", "recipe_id", "archetype",
    "n_events", "effective_n", "mean_net_bps", "nw_tstat", "block_lcb_bps",
    "rank_ic", "incremental_rank_ic", "cost_drag_ratio", "turnover_per_year",
    "compute_cost_score", "gate_passed", "reject_reasons", "bucket_key",
    "bucket_rank", "selected_for_l1", "redundant_with", "bucket_eff_test_count",
    "global_eff_test_count", "bootstrap_lcb_bps", "bootstrap_agree", "created_at_ms",
]


def _row(
    tf: str,
    mean_net_bps: float,
    *,
    family: str = "trend_ma",
    variant: str = "ema_18_108",
    gate_passed: bool = True,
    reject_reasons: str = "",
) -> dict[str, object]:
    base: dict[str, object] = dict.fromkeys(_COLS, 0)
    base.update({
        "run_id": "r1",
        "timeframe": tf,
        "family": family,
        "variant": variant,
        "recipe_id": f"{family}:{variant}:{tf}",
        "archetype": "trend",
        "mean_net_bps": mean_net_bps,
        "block_lcb_bps": mean_net_bps,
        "gate_passed": gate_passed,
        "reject_reasons": reject_reasons,
        "bootstrap_agree": True,
    })
    return base


class TestFuseMultiTimeframeEvidence:
    # Scenario 1.5: 4개 TF 전부 존재, 4h 기준 3/4가 동일 부호 -> corroborated
    def test_corroborated_when_majority_sign_agrees(self) -> None:
        evidence_by_tf = {
            "4h": pd.DataFrame([_row("4h", 18.8)]),
            "6h": pd.DataFrame([_row("6h", 12.1)]),
            "8h": pd.DataFrame([_row("8h", 9.5)]),
            "12h": pd.DataFrame([_row("12h", -3.0)]),
        }
        results = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        assert len(results) == 4
        native_4h = next(r for r in results if r.native_timeframe == "4h")
        assert native_4h.tf_coverage_count == 3
        assert native_4h.corroboration_tier == "corroborated"
        assert native_4h.sign_agreement_ratio == pytest.approx(2 / 3)
        assert native_4h.fused_conviction_score == pytest.approx(18.8 * 1.15)

    # Scenario 2.7: 단일 TF만 데이터 존재
    def test_insufficient_coverage_when_only_one_tf(self) -> None:
        evidence_by_tf = {"4h": pd.DataFrame([_row("4h", 18.8)])}
        results = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        assert len(results) == 1
        assert results[0].tf_coverage_count == 0
        assert results[0].corroboration_tier == "insufficient_coverage"
        assert results[0].sign_agreement_ratio == 0.0
        assert results[0].fused_conviction_score == pytest.approx(18.8)

    # Scenario 2.8: 타임프레임 신호 반대부호(아티팩트 의심)
    def test_contradicted_when_majority_sign_disagrees(self) -> None:
        evidence_by_tf = {
            "4h": pd.DataFrame([_row("4h", 18.8)]),
            "6h": pd.DataFrame([_row("6h", -12.1)]),
            "8h": pd.DataFrame([_row("8h", -9.5)]),
            "12h": pd.DataFrame([_row("12h", -3.0)]),
        }
        results = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        native_4h = next(r for r in results if r.native_timeframe == "4h")
        assert native_4h.corroboration_tier == "contradicted"
        assert native_4h.fused_conviction_score == pytest.approx(-18.8)
        assert native_4h.fused_conviction_score < 0

    # 커버리지되나 min_events 미달인 TF는 covered에서 제외
    def test_excludes_insufficient_events_rows_from_coverage(self) -> None:
        evidence_by_tf = {
            "4h": pd.DataFrame([_row("4h", 18.8)]),
            "6h": pd.DataFrame([_row(
                "6h", 0.0, gate_passed=False, reject_reasons="insufficient_events",
            )]),
        }
        results = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        native_4h = next(r for r in results if r.native_timeframe == "4h")
        assert native_4h.tf_coverage_count == 0
        assert native_4h.corroboration_tier == "insufficient_coverage"

    # Scenario 3.2: 동일 (family,variant,timeframe) 중복 행
    def test_raises_on_duplicate_rows(self) -> None:
        evidence_by_tf = {
            "4h": pd.DataFrame([_row("4h", 18.8), _row("4h", 20.0)]),
        }
        with pytest.raises(ValueError, match="duplicate"):
            fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)

    def test_empty_input_returns_empty(self) -> None:
        assert fuse_multi_timeframe_evidence(evidence_by_tf={}) == ()

    def test_multiple_family_variant_groups_independent(self) -> None:
        evidence_by_tf = {
            "4h": pd.DataFrame([
                _row("4h", 18.8, family="trend_ma", variant="ema_18_108"),
                _row("4h", 5.0, family="xs_carry", variant="xs_carry_96"),
            ]),
            "6h": pd.DataFrame([
                _row("6h", 12.1, family="trend_ma", variant="ema_18_108"),
            ]),
        }
        results = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        assert len(results) == 3
        families = {r.family for r in results}
        assert families == {"trend_ma", "xs_carry"}

    # Regression: 실데이터 검증에서 발견 — synthetic recipe의 variant는 관례상
    # '_{tf}' 접미사가 붙어(e.g. "tpc_50_200_8h") family+variant 그룹핑이
    # TF마다 달라져 corroboration이 항상 insufficient_coverage로 오판정됐음.
    def test_groups_across_tf_suffixed_variants(self) -> None:
        evidence_by_tf = {
            "4h": pd.DataFrame([_row("4h", 18.8, family="dual_momentum", variant="dm_24_96_4h")]),
            "6h": pd.DataFrame([_row("6h", 12.1, family="dual_momentum", variant="dm_24_96_6h")]),
            "8h": pd.DataFrame([_row("8h", 9.5, family="dual_momentum", variant="dm_24_96_8h")]),
        }
        results = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        assert len(results) == 3
        native_4h = next(r for r in results if r.native_timeframe == "4h")
        assert native_4h.variant == "dm_24_96"
        assert native_4h.tf_coverage_count == 2
        assert native_4h.corroboration_tier == "corroborated"
