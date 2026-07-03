from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.family_admission import (
    MIN_CORR_OVERLAP_BARS,
    compute_trend_sleeve_corr,
    evaluate_family_admission,
)


@pytest.fixture
def promotions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "AUSDT"],
        "family": ["residual_reversion"] * 4 + ["trend_ma"],
        "variant": ["v1", "v1", "v2", "v1", "ma"],
        "lcb_bps": [12.0, 8.0, 3.0, -1.0, 50.0],
    })


class TestComputeTrendSleeveCorr:
    def test_perfect_corr_and_insufficient_overlap(self) -> None:
        n_overlap = MIN_CORR_OVERLAP_BARS + 20  # 50 bars of overlap
        base = np.linspace(0, 10, n_overlap)
        rows: list[dict[str, object]] = []

        for sym, values in [
            ("AUSDT", base),          # identical → r ≈ +1.0
            ("BUSDT", -base),         # inverse → r ≈ -1.0
        ]:
            for i, v in enumerate(values):
                rows.append({
                    "symbol": sym,
                    "family": "residual_reversion",
                    "decision_idx": i,
                    "realized_side_adjusted_gross_bps": v,
                })
                rows.append({
                    "symbol": sym,
                    "family": "trend_ma",
                    "decision_idx": i,
                    "realized_side_adjusted_gross_bps": v if sym == "AUSDT" else -v,
                })

        sym = "CUSDT"
        rows.extend({
            "symbol": sym,
            "family": "residual_reversion",
            "decision_idx": i,
            "realized_side_adjusted_gross_bps": 1.0,
        } for i in range(20))
        rows.extend({
            "symbol": sym,
            "family": "trend_ma",
            "decision_idx": i,
            "realized_side_adjusted_gross_bps": 1.0,
        } for i in range(20, 40))

        df = pd.DataFrame(rows)
        result = compute_trend_sleeve_corr(
            df,
            candidate_family="residual_reversion",
            trend_families=("trend_ma",),
        )

        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-10)


class TestEvaluateFamilyAdmission:
    def test_three_symbols_positive_lcb_admitted(
        self, promotions_df: pd.DataFrame,
    ) -> None:
        verdict = evaluate_family_admission(
            promotions_df, "residual_reversion",
            trend_sleeve_corr=0.2,
        )
        assert verdict.admitted is True
        assert verdict.n_promoted_symbols == 3
        assert verdict.min_lcb_bps == pytest.approx(3.0)
        assert verdict.family == "residual_reversion"
        assert verdict.trend_sleeve_corr == 0.2

    def test_below_min_symbols_rejected(
        self, promotions_df: pd.DataFrame,
    ) -> None:
        filtered = promotions_df[
            promotions_df["symbol"] != "CUSDT"
        ].reset_index(drop=True)
        verdict = evaluate_family_admission(
            filtered, "residual_reversion",
            trend_sleeve_corr=0.2,
        )
        assert verdict.admitted is False
        assert "n_symbols_lt_3" in verdict.reasons

    def test_high_trend_corr_rejected(
        self, promotions_df: pd.DataFrame,
    ) -> None:
        verdict = evaluate_family_admission(
            promotions_df, "residual_reversion",
            trend_sleeve_corr=0.7,
        )
        assert verdict.admitted is False
        assert "trend_corr_gt_max" in verdict.reasons

    def test_missing_corr_skips_gate(
        self, promotions_df: pd.DataFrame,
    ) -> None:
        verdict = evaluate_family_admission(
            promotions_df, "residual_reversion",
            trend_sleeve_corr=None,
        )
        assert verdict.admitted is True
        assert "trend_corr_unavailable" in verdict.reasons
        assert verdict.n_promoted_symbols == 3

    def test_unknown_family_rejected(
        self, promotions_df: pd.DataFrame,
    ) -> None:
        verdict = evaluate_family_admission(
            promotions_df, "nonexistent",
            trend_sleeve_corr=0.2,
        )
        assert verdict.admitted is False
        assert verdict.n_promoted_symbols == 0
        assert verdict.min_lcb_bps != verdict.min_lcb_bps  # nan

    def test_zero_lcb_excluded(
        self, promotions_df: pd.DataFrame,
    ) -> None:
        extra = pd.DataFrame({
            "symbol": ["ZUSDT"],
            "family": ["residual_reversion"],
            "variant": ["v1"],
            "lcb_bps": [0.0],
        })
        df = pd.concat([promotions_df, extra], ignore_index=True)
        verdict = evaluate_family_admission(
            df, "residual_reversion",
            trend_sleeve_corr=0.2,
        )
        assert verdict.n_promoted_symbols == 3
        assert verdict.min_lcb_bps == pytest.approx(3.0)
