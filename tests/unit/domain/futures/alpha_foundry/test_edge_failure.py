from __future__ import annotations

import pandas as pd

from src.domain.futures.alpha_foundry.edge_failure import classify_edge_failure_rows


class TestClassifyEdgeFailureRows:
    def test_cost_dominated_attribution(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [4.0],
                "mean_cost_bps": [11.0],
                "cost_drag_ratio": [2.75],
                "nw_tstat": [0.5],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [-7.0],
                "gross_lcb_bps": [-3.0],
            }
        )
        result = classify_edge_failure_rows(df)
        assert result["failure_axis"].iloc[0] == "cost_dominated"

    def test_weak_gross_edge(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [1.0],
                "mean_cost_bps": [1.0],
                "cost_drag_ratio": [1.0],
                "nw_tstat": [0.5],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [0.0],
                "gross_lcb_bps": [0.0],
            }
        )
        result = classify_edge_failure_rows(df, min_gross_lcb_bps=2.0)
        assert result["failure_axis"].iloc[0] == "weak_gross_edge"

    def test_statistically_unstable(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [10.0],
                "mean_cost_bps": [3.0],
                "cost_drag_ratio": [0.3],
                "nw_tstat": [0.8],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [5.0],
                "gross_lcb_bps": [7.0],
            }
        )
        result = classify_edge_failure_rows(df, weak_tstat_abs=1.25)
        assert result["failure_axis"].iloc[0] == "statistically_unstable"

    def test_insufficient_sample(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [10.0],
                "mean_cost_bps": [3.0],
                "cost_drag_ratio": [0.3],
                "nw_tstat": [2.0],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [5.0],
                "gross_lcb_bps": [7.0],
                "effective_n": [5.0],
            }
        )
        result = classify_edge_failure_rows(df)
        assert result["failure_axis"].iloc[0] == "insufficient_sample"

    def test_turnover_dominated(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [10.0],
                "mean_cost_bps": [3.0],
                "cost_drag_ratio": [0.3],
                "nw_tstat": [2.0],
                "turnover_per_year": [400.0],
                "net_lcb_bps": [5.0],
                "gross_lcb_bps": [7.0],
            }
        )
        result = classify_edge_failure_rows(df, high_turnover_per_year=180.0)
        assert result["failure_axis"].iloc[0] == "turnover_dominated"

    def test_unknown_when_no_failure_detected(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [20.0],
                "mean_cost_bps": [1.0],
                "cost_drag_ratio": [0.05],
                "nw_tstat": [3.0],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [15.0],
                "gross_lcb_bps": [18.0],
            }
        )
        result = classify_edge_failure_rows(df)
        assert result["failure_axis"].iloc[0] == "unknown"

    def test_returns_copy_not_mutated(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [4.0],
                "mean_cost_bps": [11.0],
                "cost_drag_ratio": [2.75],
                "nw_tstat": [0.5],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [-7.0],
                "gross_lcb_bps": [-3.0],
            }
        )
        result = classify_edge_failure_rows(df)
        assert "failure_axis" in result.columns
        assert "failure_axes" in result.columns
        assert "failure_diagnostic" in result.columns

    def test_missing_optional_columns_default_to_zero(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [4.0],
                "mean_cost_bps": [11.0],
                "cost_drag_ratio": [2.75],
                "nw_tstat": [0.5],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [-7.0],
                "gross_lcb_bps": [-3.0],
            }
        )
        result = classify_edge_failure_rows(df)
        assert result["failure_axis"].iloc[0] == "cost_dominated"

    def test_cost_dominated_without_weak_gross_edge(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "mean_gross_bps": [10.0],
                "mean_cost_bps": [8.0],
                "cost_drag_ratio": [0.8],
                "nw_tstat": [2.0],
                "turnover_per_year": [50.0],
                "net_lcb_bps": [2.0],
                "gross_lcb_bps": [5.0],
            }
        )
        result = classify_edge_failure_rows(df)
        assert result["failure_axis"].iloc[0] == "cost_dominated"

    def test_empty_dataframe_returns_columns(self) -> None:
        df = pd.DataFrame(
            {
                "recipe_id": [],
                "mean_gross_bps": [],
                "mean_cost_bps": [],
                "cost_drag_ratio": [],
                "nw_tstat": [],
                "turnover_per_year": [],
                "net_lcb_bps": [],
                "gross_lcb_bps": [],
            }
        )
        result = classify_edge_failure_rows(df)
        assert list(result.columns) == [
            "recipe_id",
            "mean_gross_bps",
            "mean_cost_bps",
            "cost_drag_ratio",
            "nw_tstat",
            "turnover_per_year",
            "net_lcb_bps",
            "gross_lcb_bps",
            "failure_axis",
            "failure_axes",
            "failure_diagnostic",
        ]
