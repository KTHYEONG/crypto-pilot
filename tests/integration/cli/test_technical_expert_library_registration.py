from __future__ import annotations

from pathlib import Path

import pytest

from src.research.expert_portfolio.catalog import (
    build_technical_price_v1_blueprint,
    default_catalog,
)
from src.research.expert_portfolio.contracts import ExpertDefinition
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


def test_only_holdout_pass_components_are_registerable(tmp_path: Path) -> None:
    # TE-07: a blueprint for technical_price_v1 admits only holdout-pass
    # components, one per family and one per symbol; a failed, same-family
    # duplicate, or unregistered component cannot be registered, and the
    # default catalog stays empty without recorded HOLDOUT_PASS evidence.
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
