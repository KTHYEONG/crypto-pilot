"""Post-book diagnostic gating and deployment-readiness invariants."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

import src.mhs.evaluation.concurrency as concurrency
from src.common.errors import DataIntegrityError
from src.mhs import statistics
from src.mhs.evaluation.windows import _book_outcome
from src.mhs.horizons import horizon_log_return
from tests.fixtures.mhs_requests import research_baseline
from tests.unit.mhs.test_evaluation_appresearch import _build_book_outcome_args


def test_concurrency_module_present() -> None:
    assert concurrency.__name__ == "src.mhs.evaluation.concurrency"
    assert callable(concurrency._run_books_concurrent)


def test_concurrency_post_book_base_panel_injection() -> None:
    assert "base_panel" in inspect.signature(concurrency._run_post_book_concurrently).parameters


def test_post_diag_deploy_readiness_independent_of_report_only_flags(mhs_market, monkeypatch) -> None:
    args = _build_book_outcome_args(mhs_market)
    blend, _ = _book_outcome(**args)
    monkeypatch.setattr(statistics, "_BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(statistics, "_BOOTSTRAP_MEAN_BLOCK", 24)
    opens = args["opens"]
    signal = horizon_log_return(np.log(opens), 48)
    kwargs = {
        "blend_report": blend, "root": args["root"],
        "execution_symbols": list(opens.columns),
        "minute_grid": pd.date_range(args["start"], args["end"], freq="3min"),
        "eligible": opens.notna(), "opens": opens,
        "bar_funding": args["bar_funding"], "grid_1h": args["grid_1h"],
        "fast": args["spec"],
    }
    off = concurrency._run_post_diag_deploy(
        **kwargs, signal_48h=pd.DataFrame(), request=research_baseline(),
    )
    on = concurrency._run_post_diag_deploy(
        **kwargs, signal_48h=signal,
        request=research_baseline(placebo_diagnostic=True, bootstrap_ci_diagnostic=True),
    )
    assert off[0] is None
    assert off[1] is None
    net = blend.primary.ledger.equity.resample("1h").last().dropna().pct_change().dropna()
    assert on[0] == statistics._bootstrap_ci(net, 20, 24, statistics._BOOTSTRAP_SEED)
    assert on[1] == statistics._placebo_sharpe_percentile(
        signal, opens.notna(), opens, args["bar_funding"], args["grid_1h"],
        args["spec"], blend.primary_naive_sharpe, 500, statistics._BOOTSTRAP_SEED,
    )
    assert off[2:] == on[2:]
    assert off[4] is not None

    kwargs["blend_report"] = dataclasses.replace(blend, primary_naive_sharpe=None)
    assert concurrency._run_post_diag_deploy(
        **kwargs, signal_48h=pd.DataFrame(), request=research_baseline(),
    )[1] is None
    with pytest.raises(DataIntegrityError, match="naive Sharpe for the placebo"):
        concurrency._run_post_diag_deploy(
            **kwargs, signal_48h=signal, request=research_baseline(placebo_diagnostic=True),
        )
