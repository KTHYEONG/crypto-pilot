from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.application.expert_portfolio_evaluation import run_expert_portfolio_evaluation
from src.research.baseline.backtest import BacktestResult
from src.research.expert_portfolio.backtest import ExpertPortfolioBacktestResult
from src.research.expert_portfolio.catalog import ExpertLibraryBlueprint, ExpertLibraryCatalog
from src.research.expert_portfolio.contracts import ExpertDefinition
from src.research.expert_portfolio.contracts import ExpertPortfolioEvaluationRequest
from src.research.provenance.ledger import load_evaluation_runs, load_events
from src.research.provenance.registration import register_expert_library
from tests.fixtures.catalog import write_blueprint_files

_APPLICATION_MODULE = "src.application.expert_portfolio_evaluation"


def _concentrated_equity() -> pd.Series:
    idx = pd.date_range("2023-01-01", "2025-12-31", freq="D", tz="UTC")
    values = np.full(len(idx), 10_000.0)
    values[idx.year == 2024] = 20_000.0
    values[idx.year >= 2025] = 20_000.0
    return pd.Series(values, index=idx, name="equity", dtype=np.float64)


def _synthetic_result(equity: pd.Series, panel: pd.DataFrame) -> ExpertPortfolioBacktestResult:
    return ExpertPortfolioBacktestResult(
        backtest_result=BacktestResult(
            equity=equity,
            trades=pd.DataFrame(columns=["entry_bar"]),
            signals=pd.DataFrame(),
        ),
        target_weights=pd.DataFrame(
            {"e1": 0.0, "e2": 0.0, "CASH": 1.0}, index=equity.index,
        ),
        allocation_cost=pd.Series(0.0, index=equity.index, name="allocation_cost"),
        component_returns=panel,
    )


def test_expert_evaluation_links_active_registration(tmp_path: Path, monkeypatch) -> None:
    # PL-EXPERT-003: a registered library evaluates to one evaluation event
    # linked to its registration, and the filtered run comparison excludes the
    # registration itself.
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    code_units, data_files = write_blueprint_files(lib_dir)
    blueprint = ExpertLibraryBlueprint(
        library_id="integration_library",
        experts=(
            ExpertDefinition(
                "e1", "cointegration_residual", "pair_residual", ("AUSDT",), "run_backtest", "abc",
            ),
            ExpertDefinition(
                "e2", "cointegration_residual", "pair_residual", ("BUSDT",), "run_backtest", "def",
            ),
        ),
        supported_runners=frozenset({"run_backtest"}),
        code_units=code_units,
        data_files=data_files,
        observation_end="2025-12-31",
    )
    catalog = ExpertLibraryCatalog(blueprints={"integration_library": blueprint})
    ledger = tmp_path / "runs.jsonl"

    registration = register_expert_library(
        "integration_library", catalog=catalog, ledger_path=ledger,
    )

    equity = _concentrated_equity()
    panel = pd.DataFrame(
        {"e1": [0.001] * len(equity), "e2": [0.001] * len(equity)}, index=equity.index,
    )
    synthetic = _synthetic_result(equity, panel)

    def fake_ohlcv(path, start=None, end=None) -> pd.DataFrame:
        return pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=equity.index,
        )

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.load_ohlcv_4h", fake_ohlcv)
    monkeypatch.setattr(
        "src.research.expert_portfolio.runners.run_backtest",
        lambda df, spec, costs, signal_delay_bars=0: BacktestResult(
            equity=equity, trades=pd.DataFrame(), signals=pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.run_expert_portfolio",
        lambda component_returns, library, costs, *, initial_equity=10_000.0,
        fixed_weights=None, signal_delay_bars=0: synthetic,
    )

    report = run_expert_portfolio_evaluation(
        ExpertPortfolioEvaluationRequest(library_id="integration_library"),
        catalog=catalog,
        ledger_path=ledger,
    )
    assert report.status == "PASS"
    assert report.record is not None
    assert report.record["parent_registration_id"] == registration.registration_id

    events = load_events(ledger)
    evaluations = [e for e in events if e.record_type == "evaluation"]
    assert len(evaluations) == 1
    assert evaluations[0].payload["parent_registration_id"] == registration.registration_id

    runs = load_evaluation_runs(ledger)
    assert len(runs) == 1
    assert set(runs["record_type"]) == {"evaluation"}
    assert runs.loc[0, "parent_registration_id"] == registration.registration_id
