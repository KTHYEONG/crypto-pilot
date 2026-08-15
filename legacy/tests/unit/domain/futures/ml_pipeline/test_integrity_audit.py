from __future__ import annotations

import pandas as pd

from src.domain.futures.data_loader import summarize_dataframe_integrity
from src.domain.futures.legacy.ml_pipeline import pipeline_runner
from src.domain.futures.optimization import opt_data_utils


def test_summarize_dataframe_integrity_detects_basic_issues() -> None:
    dt = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z"], utc=True
    )
    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": [1.0, 0.0, 1.0],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
            "x": [1.0, float("inf"), None],
        }
    )
    out = summarize_dataframe_integrity(df, timeframe="1h")
    assert out["duplicate_dt"] >= 1.0
    assert out["inf_count"] >= 1.0
    assert out["nonpositive_price_count"] >= 1.0


def test_feature_group_coverage_has_expected_groups() -> None:
    df = pd.DataFrame(
        {
            "ret_1": [0.1, None],
            "funding_rate": [0.01, 0.0],
            "sum_open_interest": [100.0, 100.0],
            "global_lsr_z_24h": [0.0, 0.0],
            "taker_buy_ratio": [0.2, 0.3],
            "macro_trend_24h": [1.0, 1.0],
            "hmm_prob_crisis": [0.1, 0.2],
        }
    )
    cov = opt_data_utils._feature_group_coverage(df)
    assert set(cov.keys()) == {
        "price",
        "funding",
        "oi",
        "lsr",
        "taker_orderflow",
        "macro",
        "hmm_derived",
    }
    assert cov["funding"]["col_count"] >= 1.0
    assert cov["hmm_derived"]["non_null_coverage"] > 0.0


def test_build_integrity_summary_collects_stage_rows() -> None:
    panel_idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"), ["BTCUSDT"]],
        names=["datetime", "symbol"],
    )
    panel_df = pd.DataFrame({"hmm_prob_crisis": [None, 0.2], "target": [0.0, 0.1]}, index=panel_idx)
    maps = {
        "BTCUSDT": {
            "integrity_audit": [{"stage": "raw", "timeframe": "1h", "nan_pct": 0.1, "inf_count": 0.0, "zero_ratio": 0.1, "duplicate_dt": 0.0, "gap_count": 0.0, "nonpositive_price_count": 0.0}],
            "feature_group_coverage": {"hmm_derived": {"col_count": 1.0, "non_null_coverage": 0.5, "non_zero_coverage": 0.5}},
        }
    }
    summary = pipeline_runner._build_integrity_summary(maps, panel_df, "1h", panel_fillna_cols=["hmm_prob_crisis"])
    assert "panel" in summary
    assert "stages" in summary
    assert "feature_group_coverage" in summary
    assert float(summary["panel_pre_fillna_nan_pct"]) > 0.0
