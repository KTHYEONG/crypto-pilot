from __future__ import annotations

import pytest

from src.research.expert_portfolio.contracts import (
    ExpertDefinition,
    ExpertPortfolioEvaluationRequest,
    ExpertPortfolioSpec,
    lcb_z_score,
)


def _expert(
    expert_id: str = "pair_residual_v1",
    return_source: str = "cointegration_residual",
    family: str = "pair_residual",
    symbols: tuple[str, ...] = ("A", "B"),
    runner: str = "run_pair_residual",
    code_hash: str = "abc",
) -> ExpertDefinition:
    return ExpertDefinition(
        expert_id, return_source, family, symbols, runner, code_hash,
    )


def test_expert_definition_rejects_incomplete_identity() -> None:
    # EP-01: blank identifiers, duplicate symbols, or a missing source/family/code
    # hash must fail closed; a definition is never silently coerced.
    with pytest.raises(ValueError, match="expert_id"):
        _expert(expert_id="")
    with pytest.raises(ValueError, match="return_source"):
        _expert(return_source="")
    with pytest.raises(ValueError, match="family"):
        _expert(family="")
    with pytest.raises(ValueError, match="symbols"):
        _expert(symbols=())
    with pytest.raises(ValueError, match="duplicates"):
        _expert(symbols=("A", "A"))
    with pytest.raises(ValueError, match="runner"):
        _expert(runner="")
    with pytest.raises(ValueError, match="code_hash"):
        _expert(code_hash="")


def test_expert_definition_preserves_exact_identity() -> None:
    expert = _expert()
    assert expert.expert_id == "pair_residual_v1"
    assert expert.return_source == "cointegration_residual"
    assert expert.family == "pair_residual"
    assert expert.symbols == ("A", "B")
    assert expert.runner == "run_pair_residual"
    assert expert.code_hash == "abc"
    with pytest.raises(AttributeError):
        expert.expert_id = "mutated"  # type: ignore[misc]


def test_expert_portfolio_spec_requires_nonempty_unique_experts() -> None:
    with pytest.raises(ValueError, match="at least one expert"):
        ExpertPortfolioSpec(experts=())
    with pytest.raises(ValueError, match="unique"):
        ExpertPortfolioSpec(experts=(_expert("a"), _expert("a", symbols=("C", "D"))))


def test_expert_portfolio_spec_validates_constraint_bounds() -> None:
    kwargs = {"experts": (_expert(), _expert("b", symbols=("C", "D")))}
    with pytest.raises(ValueError, match="gross_exposure"):
        ExpertPortfolioSpec(gross_exposure=0.0, **kwargs)
    with pytest.raises(ValueError, match="gross_exposure"):
        ExpertPortfolioSpec(gross_exposure=1.5, **kwargs)
    with pytest.raises(ValueError, match="family_exposure_limit"):
        ExpertPortfolioSpec(family_exposure_limit=0.0, **kwargs)
    with pytest.raises(ValueError, match="symbol_exposure_limit"):
        ExpertPortfolioSpec(symbol_exposure_limit=2.0, **kwargs)
    with pytest.raises(ValueError, match="min_history_bars"):
        ExpertPortfolioSpec(min_history_bars=0, **kwargs)
    with pytest.raises(ValueError, match="confidence"):
        ExpertPortfolioSpec(confidence=0.70, **kwargs)


def test_expert_portfolio_spec_defaults_are_frozen() -> None:
    spec = ExpertPortfolioSpec(experts=(_expert(), _expert("b", symbols=("C", "D"))))
    assert spec.gross_exposure == 1.0
    assert spec.family_exposure_limit == 1.0
    assert spec.symbol_exposure_limit == 1.0
    assert spec.min_history_bars == 30
    assert spec.confidence == 0.90


def test_fingerprint_locks_definitions_and_config() -> None:
    spec = ExpertPortfolioSpec(experts=(_expert(),))
    fp = spec.fingerprint()
    assert [e["expert_id"] for e in fp["experts"]] == ["pair_residual_v1"]
    assert fp["gross_exposure"] == 1.0

    changed = ExpertPortfolioSpec(experts=(_expert(code_hash="different"),))
    assert changed.fingerprint() != fp


def test_lcb_z_score_rejects_unsupported_confidence() -> None:
    assert lcb_z_score(0.90) > 1.0
    with pytest.raises(ValueError, match="confidence"):
        lcb_z_score(0.70)


def test_evaluation_request_requires_library_id() -> None:
    request = ExpertPortfolioEvaluationRequest(library_id="lib-a")
    assert request.library_id == "lib-a"
    assert request.log_run is True
    assert request.unseal_holdout is False
    with pytest.raises(ValueError, match="library_id"):
        ExpertPortfolioEvaluationRequest(library_id="")
