from __future__ import annotations

import pandas as pd
import pytest

from src.research.expert_portfolio.admission_types import (
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionPipelineRequest,
    TechnicalLibraryAdmissionRequest,
    resolve_library_admission_profile,
    technical_5symbol_2022_v1_profile,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
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


def test_frozen_profile_carries_the_exact_spec_dates_and_limits() -> None:
    # LAP-03: the frozen first profile freezes the exact universe, dates, router,
    # activity, pair screen, sizes, and budget stated in the specification.
    profile = technical_5symbol_2022_v1_profile()
    assert profile.candidate_sources == (
        "technical_ema_alignment_long_v1",
        "technical_ema_alignment_short_v1",
        "technical_macd_histogram_regime_long_v1",
        "technical_macd_histogram_regime_short_v1",
        "technical_adx_di_regime_long_v1",
        "technical_adx_di_regime_short_v1",
        "technical_ichimoku_cloud_long_v1",
        "technical_ichimoku_cloud_short_v1",
        "technical_bb_squeeze_breakout_long_v1",
        "technical_bb_squeeze_breakout_short_v1",
        "technical_rsi_trend_pullback_long_v1",
        "technical_rsi_trend_pullback_short_v1",
        "technical_stochastic_trend_pullback_long_v1",
        "technical_stochastic_trend_pullback_short_v1",
        "technical_cci_trend_pullback_long_v1",
        "technical_cci_trend_pullback_short_v1",
        "technical_mfi_trend_pullback_long_v1",
        "technical_mfi_trend_pullback_short_v1",
    )
    assert len(profile.candidate_sources) == 18
    assert profile.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    assert profile.start == "2022-04-01 00:00"
    assert str(pd.Timestamp(profile.end, tz="UTC")) == "2024-12-31 20:00:00+00:00"
    assert profile.router.context_symbol == "BTCUSDT"
    assert profile.router.trend_lookback_bars == 48
    assert profile.router.volatility_lookback_bars == 48
    assert profile.router.min_context_history_bars == 96
    assert profile.router.confidence == 0.90
    admission = profile.admission
    assert admission.min_experts == 2
    assert admission.max_experts == 5
    assert admission.min_closed_trades == 20
    assert admission.min_active_return_bars == 200
    assert admission.max_abs_pairwise_log_return_correlation == 0.50
    assert admission.max_joint_negative_return_rate == 0.15
    assert admission.min_context_covered_states == 6
    assert admission.max_combinations == 1_000_000


def test_library_admission_pipeline_request_requires_temporal_separation() -> None:
    # LAP-03: an OOS start not strictly later than selection.end, a non-positive
    # budget, and an evaluation end past the sealed cutoff all fail closed.
    selection = technical_5symbol_2022_v1_profile()
    with pytest.raises(ValueError, match="evaluation_start"):
        TechnicalLibraryAdmissionPipelineRequest(
            selection=selection,
            evaluation_start="2024-12-31 20:00",
            evaluation_end="2025-12-31 20:00",
            max_backtest_proposals=24,
        )
    with pytest.raises(ValueError, match="max_backtest_proposals"):
        TechnicalLibraryAdmissionPipelineRequest(
            selection=selection,
            evaluation_start="2025-01-01",
            evaluation_end="2025-12-31 20:00",
            max_backtest_proposals=0,
        )
    with pytest.raises(ValueError, match="initial_equity"):
        TechnicalLibraryAdmissionPipelineRequest(
            selection=selection,
            evaluation_start="2025-01-01",
            evaluation_end="2025-12-31 20:00",
            max_backtest_proposals=24,
            initial_equity=0,
        )
    with pytest.raises(ValueError, match="evaluation_end"):
        TechnicalLibraryAdmissionPipelineRequest(
            selection=selection,
            evaluation_start="2025-01-01",
            evaluation_end="2024-12-31 20:00",
            max_backtest_proposals=24,
        )
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        TechnicalLibraryAdmissionPipelineRequest(
            selection=selection,
            evaluation_start="2025-01-01",
            evaluation_end="2026-06-01",
            max_backtest_proposals=24,
        )


def test_pipeline_request_round_trips_the_frozen_profile_identity() -> None:
    selection = technical_5symbol_2022_v1_profile()
    request = TechnicalLibraryAdmissionPipelineRequest(
        selection=selection,
        evaluation_start="2025-01-01",
        evaluation_end="2025-12-31 20:00",
        max_backtest_proposals=24,
    )
    assert request.selection is selection
    assert request.max_backtest_proposals == 24
    assert request.initial_equity == 10_000.0
    assert resolve_library_admission_profile("technical-5symbol-2022-v1") == selection
    with pytest.raises(ValueError, match="unknown library admission profile"):
        resolve_library_admission_profile("technical-4symbol-2021-v0")


def test_rolling_profile_freezes_the_same_universe_without_fixed_dates() -> None:
    # RLA-CLI: the rolling profile reuses the frozen 18-source five-symbol
    # universe and limits, carries no fixed selection dates, and resolves only by
    # its exact canonical name.
    from src.research.expert_portfolio.admission_types import (
        ROLLING_LIBRARY_ADMISSION_PROFILES,
        resolve_rolling_library_admission_profile,
        technical_5symbol_rolling_profile,
    )

    profile = technical_5symbol_rolling_profile()
    static = technical_5symbol_2022_v1_profile()
    assert profile.candidate_sources == static.candidate_sources
    assert profile.symbols == static.symbols
    assert profile.router == static.router
    assert profile.admission == static.admission
    assert profile.start is None
    assert profile.end is None
    assert resolve_rolling_library_admission_profile("technical-5symbol-rolling") == profile
    assert "technical-5symbol-rolling" in ROLLING_LIBRARY_ADMISSION_PROFILES
    with pytest.raises(ValueError, match="unknown rolling library admission profile"):
        resolve_rolling_library_admission_profile("technical-4symbol-rolling-v0")


def test_contracts_facade_preserves_canonical_object_identity() -> None:
    """The compatibility facade re-exports the canonical module objects."""
    from src.research.expert_portfolio import admission_reports, admission_types
    from src.research.expert_portfolio import contracts as facade
    from src.research.expert_portfolio import models

    assert facade.ExpertDefinition is models.ExpertDefinition
    assert facade.ContextualRouterSpec is models.ContextualRouterSpec
    assert facade.ExpertPortfolioSpec is models.ExpertPortfolioSpec
    assert facade.ExpertPortfolioEvaluationRequest is models.ExpertPortfolioEvaluationRequest
    assert facade.lcb_z_score is models.lcb_z_score
    assert facade.LibraryAdmissionConfig is admission_types.LibraryAdmissionConfig
    assert facade.TechnicalLibraryAdmissionRequest is admission_types.TechnicalLibraryAdmissionRequest
    assert facade.TechnicalLibraryAdmissionBacktestRequest is (
        admission_types.TechnicalLibraryAdmissionBacktestRequest
    )
    assert facade.admission_proposal_id is admission_types.admission_proposal_id
    assert facade.expert_ids_from_admission_proposal_id is (
        admission_types.expert_ids_from_admission_proposal_id
    )
    assert facade.CandidateAdmissionResult is admission_types.CandidateAdmissionResult
    assert facade.AdmissionProposal is admission_types.AdmissionProposal
    assert facade.LibraryAdmissionReport is admission_reports.LibraryAdmissionReport
    assert facade.LibraryAdmissionBacktestReport is (
        admission_reports.LibraryAdmissionBacktestReport
    )
