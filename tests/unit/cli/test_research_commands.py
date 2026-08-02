from __future__ import annotations

import pytest

from src.cli.commands.research import expert_portfolio as expert_portfolio_cli
from src.cli.main import build_root_parser
from src.research.contracts import (
    BaselineEvaluationRequest,
    CashCarryEvaluationRequest,
    PortfolioEvaluationRequest,
    SleeveBlendEvaluationRequest,
    TechnicalExpertEvaluationRequest,
)
from src.research.expert_portfolio.models import ExpertPortfolioEvaluationRequest


def test_expert_portfolio_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.expert_portfolio.evaluation.run_expert_portfolio_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert-portfolio",
        "--library-id", "pair_residual_v1",
        "--start", "2022-04-01",
        "--end", "2025-12-31",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [ExpertPortfolioEvaluationRequest(
        library_id="pair_residual_v1",
        start="2022-04-01",
        end="2025-12-31",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_expert_portfolio_cli_defaults_keep_holdout_sealed(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.expert_portfolio.evaluation.run_expert_portfolio_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert-portfolio", "--library-id", "pair_residual_v1", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [ExpertPortfolioEvaluationRequest(
        library_id="pair_residual_v1",
        start=None,
        end=None,
        initial_equity=10_000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_expert_portfolio_cli_requires_library_id() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "expert-portfolio", "--no-log-run"])


def test_library_admission_pipeline_cli_parses_and_dispatches(monkeypatch) -> None:
    # LAP-CLI: the frozen profile leaf requires only a profile name and dispatches
    # one pipeline request with the OOS defaults frozen by the specification.
    from src.research.expert_portfolio.admission_reports import LibraryAdmissionPipelineReport

    captured: list[object] = []

    def _fake_pipeline(request) -> LibraryAdmissionPipelineReport:
        captured.append(request)
        return LibraryAdmissionPipelineReport(
            status="COMPLETE",
            profile="technical-5symbol-2022-v1",
            requested_start=None,
            common_start="2022-04-01 00:00:00+00:00",
            effective_start="2022-05-04 12:00:00+00:00",
            selection_end="2024-12-31 20:00:00+00:00",
            evaluation_start="2025-01-01",
            evaluation_end="2025-12-31 20:00",
            structural_combinations=1,
            pair_compatible_count=0,
            shortlist=(),
        )

    monkeypatch.setattr(
        expert_portfolio_cli.admission_pipeline_module,
        "run_technical_library_admission_pipeline",
        _fake_pipeline,
    )
    args = build_root_parser().parse_args([
        "research", "run", "library-admission-pipeline",
        "--profile", "technical-5symbol-2022-v1",
    ])
    args.handler(args)
    assert len(captured) == 1
    request = captured[0]
    assert request.selection.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    assert request.evaluation_start == "2025-01-01"
    assert request.evaluation_end == "2025-12-31 20:00"
    assert request.max_backtest_proposals == 24
    assert request.initial_equity == 10_000.0


def test_rolling_library_admission_cli_parses_and_dispatches(monkeypatch) -> None:
    # RLA-CLI: the rolling leaf resolves the due R from as_of and dispatches one
    # paper-mode request without requiring candidate or date selection.
    from src.research.expert_portfolio.rolling import RollingLibraryAdmissionReport

    captured: list[object] = []

    def _fake_rolling(request) -> RollingLibraryAdmissionReport:
        captured.append(request)
        return RollingLibraryAdmissionReport(
            status="COMPLETE",
            profile="technical-5symbol-rolling-v1",
            mode="paper",
            as_of="2026-07-07 20:00:00+00:00",
            common_start="2022-04-01 00:00:00+00:00",
            common_end="2026-07-07 20:00:00+00:00",
            windows=(),
            records=(),
            n_folds=4,
            median_fold_cagr=0.02,
            worst_fold_cagr=0.0,
            median_fold_calmar=0.5,
            max_period_contribution=0.2,
            fold_gate_pass=True,
            oos_start="2024-07-01 00:00:00+00:00",
            oos_end="2026-06-30 20:00:00+00:00",
            oos_return=0.10,
        )

    monkeypatch.setattr(
        expert_portfolio_cli.rolling_admission_module,
        "run_rolling_library_admission",
        _fake_rolling,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert-portfolio-rolling",
        "--profile", "technical-5symbol-rolling-v1",
        "--as-of", "2026-07-07 20:00:00+00:00",
        "--mode", "paper",
    ])
    args.handler(args)
    assert len(captured) == 1
    request = captured[0]
    assert request.profile.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    assert str(request.as_of) == "2026-07-07 20:00:00+00:00"
    assert request.mode == "paper"
    assert request.config.profile == "technical-5symbol-rolling-v1"


def test_technical_expert_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[TechnicalExpertEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.technical_experts.evaluation.run_technical_expert_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "technical-expert",
        "--candidate-id", "technical_macd_histogram_regime_long_v1",
        "--symbol", "BTCUSDT",
        "--start", "2022-04-01",
        "--end", "2025-12-31",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [TechnicalExpertEvaluationRequest(
        candidate_id="technical_macd_histogram_regime_long_v1",
        symbol="BTCUSDT",
        start="2022-04-01",
        end="2025-12-31",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_technical_expert_cli_accepts_candidate_and_symbol_only() -> None:
    # TE-CLI: the frozen screen takes only a candidate and symbol; any
    # indicator/threshold flag is rejected at the parser boundary.
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([
            "research", "run", "technical-expert",
            "--candidate-id", "technical_macd_histogram_regime_long_v1",
            "--rsi-period", "14",
        ])


def test_technical_expert_cli_requires_candidate_id() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "technical-expert", "--symbol", "BTCUSDT"])


def test_baseline_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[BaselineEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.baseline.evaluation.run_baseline_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "baseline", "--symbol", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [BaselineEvaluationRequest(
        symbol="ETHUSDT", log_run=False, unseal_holdout=False,
    )]


def test_portfolio_cli_parses_symbols_and_dispatches(monkeypatch) -> None:
    calls: list[PortfolioEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.portfolio.evaluation.run_portfolio_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "--symbols", "BTCUSDT", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [PortfolioEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT"), log_run=False,
    )]


def test_cash_carry_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[CashCarryEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.cash_carry.evaluation.run_cash_carry_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "cash-carry", "--symbol", "BTCUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [CashCarryEvaluationRequest(symbol="BTCUSDT", log_run=False)]


def test_sleeve_blend_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.sleeve_blend.evaluation.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "sleeve-blend",
        "--symbols", "BTCUSDT", "ETHUSDT",
        "--mdd-budget-fraction", "0.80",
        "--start", "2022-04-01",
        "--end", "2025-01-01",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT"),
        mdd_budget_fraction=0.80,
        candidate_kind="fixed_long_only_v1",
        start="2022-04-01",
        end="2025-01-01",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_sleeve_blend_cli_directional_candidate_kind(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.sleeve_blend.evaluation.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "sleeve-blend",
        "--candidate-kind", "funding_signed_directional_v1", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        mdd_budget_fraction=0.85,
        candidate_kind="funding_signed_directional_v1",
        log_run=False,
    )]
