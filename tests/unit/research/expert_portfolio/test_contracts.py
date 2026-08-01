from __future__ import annotations

import pytest

from src.research.expert_portfolio.contracts import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioEvaluationRequest,
    ExpertPortfolioSpec,
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionRequest,
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


def test_contextual_router_spec_validates_inputs() -> None:
    # The pre-registered router rejects an empty context symbol and any
    # non-positive bar count before routing can run, and only the supported
    # LCB confidence levels are accepted.
    with pytest.raises(ValueError, match="context_symbol"):
        ContextualRouterSpec("", 1, 1, 1)
    with pytest.raises(ValueError, match="trend_lookback_bars"):
        ContextualRouterSpec("BTCUSDT", 0, 1, 1)
    with pytest.raises(ValueError, match="volatility_lookback_bars"):
        ContextualRouterSpec("BTCUSDT", 1, 0, 1)
    with pytest.raises(ValueError, match="min_context_history_bars"):
        ContextualRouterSpec("BTCUSDT", 1, 1, 0)
    with pytest.raises(ValueError, match="confidence"):
        ContextualRouterSpec("BTCUSDT", 1, 1, 1, confidence=0.70)


def test_contextual_router_spec_preserves_identity_and_fingerprint() -> None:
    router = ContextualRouterSpec("BTCUSDT", 60, 20, 30)
    assert router.context_symbol == "BTCUSDT"
    assert router.trend_lookback_bars == 60
    assert router.volatility_lookback_bars == 20
    assert router.min_context_history_bars == 30
    assert router.confidence == 0.90
    with pytest.raises(AttributeError):
        router.context_symbol = "ETHUSDT"  # type: ignore[misc]

    routed = ExpertPortfolioSpec(experts=(_expert(),), router=router)
    assert routed.fingerprint()["router"]["context_symbol"] == "BTCUSDT"
    assert ExpertPortfolioSpec(experts=(_expert(),)).fingerprint()["router"] is None


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


def _admission_config(**overrides: object) -> LibraryAdmissionConfig:
    base: dict[str, object] = {
        "min_experts": 2,
        "max_experts": 4,
        "min_closed_trades": 1,
        "min_active_return_bars": 1,
        "max_abs_pairwise_log_return_correlation": 0.8,
        "max_joint_negative_return_rate": 0.5,
        "min_context_covered_states": 1,
        "max_combinations": 100,
    }
    base.update(overrides)
    return LibraryAdmissionConfig(**base)  # type: ignore[arg-type]


def test_library_admission_config_preserves_identity_and_telemetry_exclusion() -> None:
    # LAE-01: the admission config is immutable and max_workers is execution
    # telemetry that never participates in the admission fingerprint.
    config = _admission_config(max_workers=1)
    assert config.max_workers == 1
    with pytest.raises(AttributeError):
        config.min_experts = 3  # type: ignore[misc]
    assert "max_workers" not in config.fingerprint()
    assert config.fingerprint() == _admission_config().fingerprint()


def test_library_admission_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_experts"):
        _admission_config(min_experts=0)
    with pytest.raises(ValueError, match="max_experts"):
        _admission_config(min_experts=3, max_experts=2)
    with pytest.raises(ValueError, match="min_closed_trades"):
        _admission_config(min_closed_trades=-1)
    with pytest.raises(ValueError, match="min_active_return_bars"):
        _admission_config(min_active_return_bars=-1)
    with pytest.raises(ValueError, match="max_abs_pairwise"):
        _admission_config(max_abs_pairwise_log_return_correlation=1.5)
    with pytest.raises(ValueError, match="max_joint_negative"):
        _admission_config(max_joint_negative_return_rate=-0.1)
    with pytest.raises(ValueError, match="min_context_covered_states"):
        _admission_config(min_context_covered_states=7)
    with pytest.raises(ValueError, match="max_combinations"):
        _admission_config(max_combinations=0)
    with pytest.raises(ValueError, match="max_workers"):
        _admission_config(max_workers=0)


def test_library_admission_request_preserves_identity() -> None:
    request = TechnicalLibraryAdmissionRequest(
        candidate_sources=("technical_macd_histogram_regime_long_v1",),
        symbols=("BTCUSDT",),
        router=ContextualRouterSpec("BTCUSDT", 60, 20, 30),
        admission=_admission_config(max_experts=1, min_experts=1, max_combinations=1),
    )
    assert request.symbols == ("BTCUSDT",)
    assert request.admission.max_experts == 1
    with pytest.raises(AttributeError):
        request.symbols = ("ETHUSDT",)  # type: ignore[misc]


def test_library_admission_request_rejects_invalid_universe_and_sealed_end() -> None:
    # LAE-01: duplicate sources/symbols, an unknown candidate source, empty
    # universes, and any end past the sealed holdout cutoff fail closed.
    router = ContextualRouterSpec("BTCUSDT", 60, 20, 30)
    base = _admission_config(max_experts=1, min_experts=1, max_combinations=1)
    with pytest.raises(ValueError, match="duplicates"):
        TechnicalLibraryAdmissionRequest(
            ("technical_macd_histogram_regime_long_v1",) * 2,
            ("BTCUSDT",), router, base,
        )
    with pytest.raises(ValueError, match="duplicates"):
        TechnicalLibraryAdmissionRequest(
            ("technical_macd_histogram_regime_long_v1",),
            ("BTCUSDT", "BTCUSDT"), router, base,
        )
    with pytest.raises(ValueError, match="candidate_sources must not be empty"):
        TechnicalLibraryAdmissionRequest((), ("BTCUSDT",), router, base)
    with pytest.raises(ValueError, match="symbols must not be empty"):
        TechnicalLibraryAdmissionRequest(
            ("technical_macd_histogram_regime_long_v1",), (), router, base,
        )
    with pytest.raises(ValueError, match="unknown or retired"):
        TechnicalLibraryAdmissionRequest(
            ("technical_naive_nope_long_v1",), ("BTCUSDT",), router, base,
        )
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        TechnicalLibraryAdmissionRequest(
            ("technical_macd_histogram_regime_long_v1",),
            ("BTCUSDT",), router, base, end="2026-06-01",
        )
