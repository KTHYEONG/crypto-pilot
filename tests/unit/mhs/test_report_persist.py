def test_fold_summary_persists_scalar_sizing_reference_lineage() -> None:
    from src.mhs.report.persist import _fold_summary

    fold = type(
        "Fold",
        (),
        {
            "fold_index": 1,
            "validation_start": "2022-01-01",
            "validation_end": "2022-03-31",
            "primary_valid": True,
            "primary_autocorr_sharpe": 1.0,
            "primary_naive_sharpe": 1.0,
            "primary_net_ann": 0.1,
            "primary_geometric_cagr": 0.1,
            "primary_max_drawdown": -0.1,
            "stress_naive_sharpe": 0.5,
            "failures": (),
            "book_structure": {
                "sizing_reference_start": "2021-02-08T00:00:00+00:00",
                "sizing_reference_end": "2021-12-30T00:00:00+00:00",
                "sizing_reference_daily_rows": 325.0,
            },
        },
    )()

    summary = _fold_summary(fold)

    assert summary["sizing_reference_start"] == "2021-02-08T00:00:00+00:00"
    assert summary["sizing_reference_end"] == "2021-12-30T00:00:00+00:00"
    assert summary["sizing_reference_daily_rows"] == 325.0
