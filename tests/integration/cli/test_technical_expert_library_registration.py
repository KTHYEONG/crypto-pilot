from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.expert_portfolio.catalog import (
    build_technical_price_v1_blueprint,
    default_catalog,
)
from src.research.expert_portfolio.models import ExpertDefinition
from src.research.provenance.ledger import (
    append_event,
    build_evaluation_event,
    load_events,
)
from src.research.technical_experts.catalog import TECHNICAL_CANDIDATES
from tests.fixtures.catalog import write_blueprint_files


def _technical_definition(candidate, symbol: str) -> ExpertDefinition:
    return ExpertDefinition(
        expert_id=f"{candidate.return_source}:{symbol}",
        return_source=candidate.return_source,
        family=candidate.family,
        symbols=(symbol,),
        runner="run_technical_expert",
        code_hash="c" * 64,
    )


def _candidate(return_source: str):
    return next(
        c for c in TECHNICAL_CANDIDATES if c.return_source == return_source
    )


def _fake_technical_evaluation(ledger: Path, return_source: str, status: str) -> None:
    """Append one immutable technical_expert evaluation event to a temp ledger."""
    event = build_evaluation_event(
        workflow="technical_expert",
        ts="2026-08-01T00:00:00+00:00",
        git_sha="abc",
        git_dirty=False,
        metrics={},
        reliability={},
        promotion={
            "status": status,
            "observation_verdict": "PASS",
            "fold_gate_pass": True,
            "stress_verdict": "PASS",
            "holdout_verdict": "PASS" if status == "HOLDOUT_PASS" else None,
        },
        kind="technical_expert",
        symbol="BTCUSDT",
        candidate_id=return_source,
        return_source=return_source,
    )
    append_event(event, ledger_path=ledger)


def test_components_are_admission_diagnostics_only_composite_ledger_promotes(tmp_path: Path) -> None:
    # TE-07 / LAE-09: a blueprint for technical_price_v1 is a pre-registration
    # admission result, not a promotion gate: individual candidate results are
    # admission diagnostics, and only the registered composite master ledger
    # determines promotion. A failed, same-family duplicate, or unregistered
    # component cannot be registered, and the default catalog stays empty
    # without source-controlled admission evidence.
    code_units, data_files = write_blueprint_files(tmp_path)
    macd_long = _candidate("technical_macd_histogram_regime_long_v1")
    macd_short = _candidate("technical_macd_histogram_regime_short_v1")
    rsi_long = _candidate("technical_rsi_trend_pullback_long_v1")

    approved = (
        _technical_definition(macd_long, "BTCUSDT"),
        _technical_definition(rsi_long, "ETHUSDT"),
    )
    blueprint = build_technical_price_v1_blueprint(
        approved, code_units=code_units, data_files=data_files,
    )
    assert blueprint.library_id == "technical_price_v1"
    assert all(e.runner == "run_technical_expert" for e in blueprint.experts)
    assert len(blueprint.experts) == 2

    # Same-family candidates are substitutes, never independent experts.
    with pytest.raises(ValueError, match="per family"):
        build_technical_price_v1_blueprint(
            (
                _technical_definition(macd_long, "BTCUSDT"),
                _technical_definition(macd_short, "ETHUSDT"),
            ),
            code_units=code_units,
            data_files=data_files,
        )

    # One component per underlying symbol.
    with pytest.raises(ValueError, match="per symbol"):
        build_technical_price_v1_blueprint(
            (
                _technical_definition(macd_long, "BTCUSDT"),
                _technical_definition(rsi_long, "BTCUSDT"),
            ),
            code_units=code_units,
            data_files=data_files,
        )

    # An unregistered / rejected return source cannot be re-labelled.
    bogus = ExpertDefinition(
        "bogus", "technical_naive_bollinger_mean_reversion_long_v1",
        "bollinger_mean_reversion", ("BTCUSDT",), "run_technical_expert", "x" * 64,
    )
    with pytest.raises(ValueError, match="unknown or retired"):
        build_technical_price_v1_blueprint(
            (bogus,), code_units=code_units, data_files=data_files,
        )

    # A failed candidate is absent from the approved set selected from the ledger.
    ledger = tmp_path / "runs.jsonl"
    _fake_technical_evaluation(
        ledger, "technical_macd_histogram_regime_long_v1", "HOLDOUT_PASS",
    )
    _fake_technical_evaluation(ledger, "technical_mfi_trend_pullback_long_v1", "REJECTED")
    events = load_events(ledger)
    passed_sources = {
        event.payload["return_source"]
        for event in events
        if event.record_type == "evaluation"
        and event.payload.get("promotion", {}).get("status") == "HOLDOUT_PASS"
    }
    assert "technical_macd_histogram_regime_long_v1" in passed_sources
    assert "technical_mfi_trend_pullback_long_v1" not in passed_sources

    # No recorded HOLDOUT_PASS evidence exists in the real catalog path yet.
    assert default_catalog().blueprints == {}


def test_contextual_library_promotion_uses_only_composite_master_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ECR-06: a contextual library is promoted only from its composite
    # master-ledger observation/fold/stress/holdout evidence. Individual expert
    # reports remain recorded admission diagnostics and can never act as
    # independent promotion gates, so a failing fold on the routed master ledger
    # keeps the library REJECTED even though per-expert evidence is HOLDOUT_PASS.
    from src.application.research.expert.evaluation import run_expert_portfolio_evaluation
    from src.research.baseline.backtest import BacktestResult
    from src.research.expert_portfolio.backtest import ExpertPortfolioBacktestResult
    from src.research.expert_portfolio.models import (
        ContextualRouterSpec,
        ExpertPortfolioEvaluationRequest,
        ExpertPortfolioSpec,
    )
    from src.research.expert_portfolio.registry import RegisteredExpertLibrary
    from src.research.provenance.registration import RegistrationRecord

    macd_long = _candidate("technical_macd_histogram_regime_long_v1")
    rsi_long = _candidate("technical_rsi_trend_pullback_long_v1")
    library = ExpertPortfolioSpec(
        experts=(
            _technical_definition(macd_long, "BTCUSDT"),
            _technical_definition(rsi_long, "ETHUSDT"),
        ),
        router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
    )
    # Record per-expert admission diagnostics as HOLDOUT_PASS; these must not
    # gate promotion by themselves.
    ledger = tmp_path / "runs.jsonl"
    _fake_technical_evaluation(
        ledger, "technical_macd_histogram_regime_long_v1", "HOLDOUT_PASS",
    )
    _fake_technical_evaluation(
        ledger, "technical_rsi_trend_pullback_long_v1", "HOLDOUT_PASS",
    )

    idx = pd.date_range("2023-01-01", "2025-12-31", freq="D", tz="UTC")
    equity = pd.Series(np.full(len(idx), 10_000.0), index=idx, name="equity")
    equity.iloc[idx.year == 2024] = 20_000.0  # concentrated growth -> fold failure
    equity.iloc[idx.year >= 2025] = 20_000.0
    panel = pd.DataFrame(
        {e.expert_id: [0.001] * len(idx) for e in library.experts}, index=idx,
    )
    trades = pd.DataFrame({
        "expert_id": [library.experts[0].expert_id] * 40,
        "symbol": ["BTCUSDT"] * 40,
        "entry_bar": np.arange(40),
        "exit_bar": np.arange(40) + 1,
        "entry_time": idx[:40],
        "exit_time": idx[1:41],
        "entry_price": [100.0] * 40,
        "exit_price": [101.0] * 40,
        "qty": [1.0] * 40,
        "reason": ["channel"] * 40,
        "pnl": [10.0] * 40,
        "return_pct": [0.01] * 40,
        "funding_pnl": [0.0] * 40,
    })
    result = ExpertPortfolioBacktestResult(
        backtest_result=BacktestResult(
            equity=equity, trades=pd.DataFrame(), signals=pd.DataFrame(),
        ),
        target_weights=pd.DataFrame(
            {e.expert_id: 0.0 for e in library.experts} | {"CASH": 1.0}, index=idx,
        ),
        allocation_cost=pd.Series(0.0, index=idx),
        component_returns=panel,
    )
    context = pd.Series(["up_low_vol"] * len(idx), index=idx)
    used_contexts: list[object] = []

    def fake_resolve(_library_id, *, catalog=None, ledger_path=None):
        return RegisteredExpertLibrary(
            library_id="lib-ctx",
            registration_id="reg-1",
            spec=library,
            registration=RegistrationRecord(
                registration_id="reg-1",
                library_id="lib-ctx",
                status="ACTIVE",
                fingerprint={"experts": []},
                registered_at="2026-01-01T00:00:00+00:00",
                record={"registration_id": "reg-1"},
            ),
        )

    def fake_run(
        component_returns, spec, costs, *, initial_equity=10_000.0,
        fixed_weights=None, signal_delay_bars=0, decision_context=None,
    ):
        if fixed_weights is None:
            used_contexts.append(decision_context)
        return result

    monkeypatch.setattr(
        "src.application.research.expert.evaluation.resolve_registered_library",
        fake_resolve,
    )
    monkeypatch.setattr(
        "src.application.research.expert.evaluation.build_component_panel",
        lambda spec, start, end, costs, *, signal_delay_bars=0: (panel, trades),
    )
    monkeypatch.setattr(
        "src.application.research.expert.evaluation.build_library_decision_context",
        lambda spec, index, start, end: context,
    )
    monkeypatch.setattr(
        "src.application.research.expert.evaluation.run_expert_portfolio", fake_run,
    )

    report = run_expert_portfolio_evaluation(
        ExpertPortfolioEvaluationRequest(library_id="lib-ctx", log_run=False),
    )
    assert used_contexts == [context]
    assert report.fold_distribution.gate_pass is False
    assert report.promotion.status == "REJECTED"
