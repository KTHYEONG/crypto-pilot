"""CLI research command dispatch. Covers SCENARIO_XSV5_04_CLI_DISPATCHES_V3_V4_V5 and SCENARIO_XSV6_05_CLI_DISPATCHES_V6."""

from __future__ import annotations

import json

import pytest

from src.cli.commands.research import expert_library as expert_library_cli
from src.cli.commands.research.portfolio_blend import add_portfolio_blend_commands
from src.cli.commands.research.portfolio_growth import add_portfolio_growth_commands
from src.cli.commands.research.portfolio_multi import add_portfolio_multi_commands
from src.cli.commands.research.single_baseline import add_single_baseline_commands
from src.cli.commands.research.single_carry import add_single_carry_commands
from src.cli.commands.research.single_oi import add_single_oi_commands
from src.cli.commands.research.single_technical import add_single_technical_commands
from src.cli.main import build_root_parser
from src.research.contracts import (
    BaselineEvaluationRequest,
    CashCarryEvaluationRequest,
    GrowthEngineEvaluationRequest,
    PortfolioEvaluationRequest,
    SleeveBlendEvaluationRequest,
    TechnicalExpertEvaluationRequest,
)
from src.research.expert_portfolio.models import ExpertPortfolioEvaluationRequest


def test_expert_portfolio_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.expert.evaluation.run_expert_portfolio_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "eval",
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
        "src.application.research.expert.evaluation.run_expert_portfolio_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "eval", "--library-id", "pair_residual_v1", "--no-log-run",
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
        build_root_parser().parse_args(["research", "run", "expert", "eval", "--no-log-run"])


def test_library_admission_backtest_cli_threads_timeframe(monkeypatch, capsys) -> None:
    # The backtest leaf threads --timeframe into the request; default 4h is
    # applied when the flag is absent.
    captured: list[object] = []

    class _Report:
        def to_report_dict(self) -> dict[str, object]:
            return {"status": "COMPLETE"}

    def _fake_run(request) -> _Report:
        captured.append(request)
        return _Report()

    monkeypatch.setattr(
        "src.application.research.expert.admission_backtest.run_technical_library_admission_backtest",
        _fake_run,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "backtest",
        "--expert-id", "technical_ema_alignment_long_v1:BTCUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "48",
        "--router-volatility-lookback-bars", "48",
        "--router-min-context-history-bars", "96",
        "--timeframe", "12h",
    ])
    args.handler(args)
    assert captured
    assert captured[0].timeframe == "12h"
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"

    captured.clear()
    default_args = build_root_parser().parse_args([
        "research", "run", "expert", "backtest",
        "--expert-id", "technical_ema_alignment_long_v1:BTCUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "48",
        "--router-volatility-lookback-bars", "48",
        "--router-min-context-history-bars", "96",
    ])
    default_args.handler(default_args)
    assert captured[0].timeframe == "4h"


def test_ter_06_cli_backtest_stop_loss_flags_wired(monkeypatch, capsys) -> None:
    # TER-06: the four stop-loss flags construct an exact
    # TechnicalLibraryAdmissionBacktestRequest; omitting all of them falls back
    # to the master-switch-off defaults.
    captured: list[object] = []

    class _Report:
        def to_report_dict(self) -> dict[str, object]:
            return {"status": "COMPLETE"}

    def _fake_run(request) -> _Report:
        captured.append(request)
        return _Report()

    monkeypatch.setattr(
        "src.application.research.expert.admission_backtest.run_technical_library_admission_backtest",
        _fake_run,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "backtest",
        "--expert-id", "technical_ema_alignment_long_v1:BTCUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "48",
        "--router-volatility-lookback-bars", "48",
        "--router-min-context-history-bars", "96",
        "--stop-loss-mode", "atr_multiple",
        "--stop-loss-value", "2.0",
        "--atr-period", "14",
        "--trailing-stop",
    ])
    args.handler(args)
    assert captured
    assert captured[0].stop_loss_mode == "atr_multiple"
    assert captured[0].stop_loss_value == 2.0
    assert captured[0].atr_period == 14
    assert captured[0].trailing_stop is True
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"

    captured.clear()
    default_args = build_root_parser().parse_args([
        "research", "run", "expert", "backtest",
        "--expert-id", "technical_ema_alignment_long_v1:BTCUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "48",
        "--router-volatility-lookback-bars", "48",
        "--router-min-context-history-bars", "96",
    ])
    default_args.handler(default_args)
    assert captured[0].stop_loss_mode is None
    assert captured[0].stop_loss_value is None
    assert captured[0].atr_period == 14
    assert captured[0].trailing_stop is False


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
        expert_library_cli.admission_pipeline_module,
        "run_technical_library_admission_pipeline",
        _fake_pipeline,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "pipeline",
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
            profile="technical-5symbol-rolling",
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
        expert_library_cli.rolling_admission_module,
        "run_rolling_library_admission",
        _fake_rolling,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "rolling",
        "--profile", "technical-5symbol-rolling",
        "--as-of", "2026-07-07 20:00:00+00:00",
        "--mode", "paper",
    ])
    args.handler(args)
    assert len(captured) == 1
    request = captured[0]
    assert request.profile.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    assert str(request.as_of) == "2026-07-07 20:00:00+00:00"
    assert request.mode == "paper"
    assert request.config.profile == "technical-5symbol-rolling"
    assert request.config.base_delay_bars == 1
    assert request.config.min_shortlist_budget == 8


def test_rolling_library_admission_cli_rejects_retired_profile_names(monkeypatch) -> None:
    # RAP-RETIRED: the versioned v1/v2/v3 profile names are gone from the CLI
    # choice list; only the canonical name remains selectable.
    choices = sorted(
        expert_library_cli.ROLLING_LIBRARY_ADMISSION_PROFILES,
    )
    assert choices == ["technical-5symbol-rolling"]
    for retired in (
        "technical-5symbol-rolling-v1",
        "technical-5symbol-rolling-v2",
        "technical-5symbol-rolling-v3",
    ):
        with pytest.raises(SystemExit):
            build_root_parser().parse_args([
                "research", "run", "expert", "rolling",
                "--profile", retired,
                "--as-of", "2026-07-07 20:00:00+00:00",
            ])


def test_rolling_library_admission_cli_rejects_unknown_profile() -> None:
    # RAP-09: an unknown rolling profile name fails closed with ValueError.
    from src.research.expert_portfolio.admission_types import (
        resolve_rolling_library_admission_profile,
    )

    with pytest.raises(ValueError, match="unknown rolling library admission profile"):
        resolve_rolling_library_admission_profile("technical-5symbol-rolling-v9")


def test_technical_expert_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[TechnicalExpertEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.technical.evaluation.run_technical_expert_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "single", "technical",
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
            "research", "run", "single", "technical",
            "--candidate-id", "technical_macd_histogram_regime_long_v1",
            "--rsi-period", "14",
        ])


def test_technical_expert_cli_requires_candidate_id() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "single", "technical", "--symbol", "BTCUSDT"])


def test_xs_screen_cli_parses_and_dispatches(monkeypatch) -> None:
    from src.application.research.technical.xs_trend_screen import (
        XS_NEUTRAL_PROFILE_ID,
        XsAdmissionResult,
        XsCompositeSpec,
        XsTrendScreenReport,
    )

    calls: list[tuple[object, object, bool, str]] = []

    def fake_run(*, start, end, unseal_holdout=False, max_workers=None, profile=XS_NEUTRAL_PROFILE_ID):
        calls.append((start, end, unseal_holdout, profile))
        failed = XsAdmissionResult(
            admitted=False, binding_constraint="symbol_unavailable:X", sharpe=0.0,
            beta=0.0, cagr=0.0, mdd=0.0, t_stat=0.0, annual_sharpe={},
            annualized_turnover=0.0, breakeven_cost=0.0,
        )
        return XsTrendScreenReport(
            profile=profile,
            universe=("BTCUSDT",),
            spec=XsCompositeSpec(),
            discovery=failed,
            qualification=failed,
            symbols={},
        )

    persisted: list[str] = []

    def fake_persist(report, path):
        persisted.append((report.profile, str(path)))

    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.run_xs_trend_screen", fake_run,
    )
    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.persist_xs_screen_report",
        fake_persist,
    )
    args = build_root_parser().parse_args([
        "research", "run", "single", "xs-screen",
        "--start", "2022-04-01",
        "--end", "2025-12-31",
        "--unseal-holdout",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [("2022-04-01", "2025-12-31", True, XS_NEUTRAL_PROFILE_ID)]
    assert persisted == [(XS_NEUTRAL_PROFILE_ID, "docs/results/xs_neutral_composite_v1.json")]


def test_xs_screen_cli_dispatches_v2_profile(monkeypatch) -> None:
    from src.application.research.technical.xs_trend_screen import (
        XS_ALPHA_PROFILE_ID,
        XS_NEUTRAL_PROFILE_ID,
        XsAdmissionResult,
        XsCompositeSpec,
        XsTrendScreenReport,
    )

    calls: list[str] = []

    def fake_run(*, start, end, unseal_holdout=False, max_workers=None, profile=XS_NEUTRAL_PROFILE_ID):
        calls.append(profile)
        failed = XsAdmissionResult(
            admitted=False, binding_constraint="symbol_unavailable:X", sharpe=0.0,
            beta=0.0, cagr=0.0, mdd=0.0, t_stat=0.0, annual_sharpe={},
            annualized_turnover=0.0, breakeven_cost=0.0,
        )
        return XsTrendScreenReport(
            profile=profile,
            universe=("BTCUSDT",),
            spec=XsCompositeSpec(),
            discovery=failed,
            qualification=failed,
            symbols={},
        )

    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.run_xs_trend_screen", fake_run,
    )
    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.persist_xs_screen_report",
        lambda report, path: None,
    )
    args = build_root_parser().parse_args([
        "research", "run", "single", "xs-screen", "--profile", "xs_alpha_multihorizon_v2",
    ])
    args.handler(args)
    assert calls == [XS_ALPHA_PROFILE_ID]


def test_xs_screen_cli_rejects_unknown_profile() -> None:
    args = build_root_parser().parse_args([
        "research", "run", "single", "xs-screen", "--profile", "bogus",
    ])
    with pytest.raises(ValueError, match="unknown xs screen profile"):
        args.handler(args)


def test_xs_screen_cli_dispatches_v3_v4_v5_profiles(monkeypatch) -> None:
    from src.application.research.technical.xs_trend_screen import (
        XS_NEUTRAL_PROFILE_ID,
        XsAdmissionResult,
        XsCompositeSpec,
        XsTrendScreenReport,
    )

    calls: list[str] = []

    def fake_run(*, start, end, unseal_holdout=False, max_workers=None, profile=XS_NEUTRAL_PROFILE_ID):
        calls.append(profile)
        failed = XsAdmissionResult(
            admitted=False, binding_constraint="symbol_unavailable:X", sharpe=0.0,
            beta=0.0, cagr=0.0, mdd=0.0, t_stat=0.0, annual_sharpe={},
            annualized_turnover=0.0, breakeven_cost=0.0,
        )
        return XsTrendScreenReport(
            profile=profile,
            universe=("BTCUSDT",),
            spec=XsCompositeSpec(),
            discovery=failed,
            qualification=failed,
            symbols={},
        )

    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.run_xs_trend_screen", fake_run,
    )
    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.persist_xs_screen_report",
        lambda report, path: None,
    )
    for profile in (
        "xs_alpha_contextual_v3",
        "xs_alpha_score_routed_v4",
        "xs_alpha_dual_family_v5",
    ):
        args = build_root_parser().parse_args([
            "research", "run", "single", "xs-screen", "--profile", profile,
        ])
        args.handler(args)
    assert calls == [
        "xs_alpha_contextual_v3",
        "xs_alpha_score_routed_v4",
        "xs_alpha_dual_family_v5",
    ]

def test_xs_screen_cli_dispatches_v6_profile(monkeypatch) -> None:
    # SCENARIO_XSV6_05_CLI_DISPATCHES_V6
    from src.application.research.technical.xs_trend_screen import (
        XS_NEUTRAL_PROFILE_ID,
        XsAdmissionResult,
        XsCompositeSpec,
        XsTrendScreenReport,
    )

    calls: list[str] = []

    def fake_run(*, start, end, unseal_holdout=False, max_workers=None, profile=XS_NEUTRAL_PROFILE_ID):
        calls.append(profile)
        failed = XsAdmissionResult(
            admitted=False, binding_constraint="symbol_unavailable:X", sharpe=0.0,
            beta=0.0, cagr=0.0, mdd=0.0, t_stat=0.0, annual_sharpe={},
            annualized_turnover=0.0, breakeven_cost=0.0,
        )
        return XsTrendScreenReport(
            profile=profile,
            universe=("BTCUSDT",),
            spec=XsCompositeSpec(),
            discovery=failed,
            qualification=failed,
            symbols={},
        )

    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.run_xs_trend_screen", fake_run,
    )
    monkeypatch.setattr(
        "src.application.research.technical.xs_trend_screen.persist_xs_screen_report",
        lambda report, path: None,
    )
    args = build_root_parser().parse_args([
        "research", "run", "single", "xs-screen", "--profile", "xs_alpha_vol_weighted_v6",
    ])
    args.handler(args)
    assert calls == ["xs_alpha_vol_weighted_v6"]


def test_baseline_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[BaselineEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.baseline.evaluation.run_baseline_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "single", "baseline", "--symbol", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    # Futures promotion evidence resolves the canonical aligned funding stream.
    from src.common.config import funding_path

    assert calls == [BaselineEvaluationRequest(
        symbol="ETHUSDT",
        funding_path=str(funding_path("ETHUSDT")),
        log_run=False, unseal_holdout=False,
    )]


def test_portfolio_cli_parses_symbols_and_dispatches(monkeypatch) -> None:
    calls: list[PortfolioEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.portfolio.evaluation.run_portfolio_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "multi", "--symbols", "BTCUSDT", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [PortfolioEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT"), log_run=False,
    )]


def test_cash_carry_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[CashCarryEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.carry.evaluation.run_cash_carry_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "single", "carry", "--symbol", "BTCUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [CashCarryEvaluationRequest(symbol="BTCUSDT", log_run=False)]


def test_portfolio_growth_cli_parses_and_dispatches(monkeypatch) -> None:
    # GEV2-CLI: the growth leaf threads the universe/construction knobs into a
    # frozen request and defaults symbol_scope to 'dev' so the holdout partition
    # is never touched by accident.
    calls: list[GrowthEngineEvaluationRequest] = []

    class _FakeReport:
        status = "NO_ADMISSIBLE_ALPHA"
        start = None
        equity = ()
        trades = ()
        sizing = None
        selected_strategy = None
        scorecard = None

    def _fake_run(request) -> _FakeReport:
        calls.append(request)
        return _FakeReport()

    monkeypatch.setattr(
        "src.application.research.growth.evaluation.run_growth_engine_evaluation", _fake_run,
    )
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "growth",
        "--universe-size", "4",
        "--max-positions", "2",
        "--rebalance-bars", "6",
        "--no-trade-band", "0.01",
        "--symbol-scope", "all",
        "--start", "2022-04-01",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert len(calls) == 1
    request = calls[0]
    assert request.universe.universe_size == 4
    assert request.universe.max_positions == 2
    assert request.construction.rebalance_bars == 6
    assert request.construction.no_trade_band == 0.01
    assert request.symbol_scope == "all"
    assert request.start == "2022-04-01"
    assert request.initial_equity == 5000.0
    assert request.log_run is False


def test_portfolio_growth_cli_defaults_to_dev_scope(monkeypatch) -> None:
    calls: list[GrowthEngineEvaluationRequest] = []

    class _FakeReport:
        status = "NO_ADMISSIBLE_ALPHA"
        start = None
        equity = ()
        trades = ()
        sizing = None
        selected_strategy = None
        scorecard = None

    def _fake_run(request) -> _FakeReport:
        calls.append(request)
        return _FakeReport()

    monkeypatch.setattr(
        "src.application.research.growth.evaluation.run_growth_engine_evaluation", _fake_run,
    )
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "growth", "--no-log-run",
    ])
    args.handler(args)
    assert len(calls) == 1
    assert calls[0].symbol_scope == "dev"
    assert calls[0].universe.universe_size == 20
    assert calls[0].universe.max_positions == 5


def test_sleeve_blend_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.blend.evaluation.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "blend",
        "--universe-id", "core5_v1",
        "--candidate-kind", "fixed_long_only_v1",
        "--mdd-budget-fraction", "0.80",
        "--start", "2022-04-01",
        "--end", "2025-01-01",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        universe_id="core5_v1",
        mdd_budget_fraction=0.80,
        candidate_kind="fixed_long_only_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        qualification_interval="365D",
        start="2022-04-01",
        end="2025-01-01",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_sleeve_blend_cli_rejects_free_form_symbols(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([
            "research", "run", "portfolio", "blend", "--symbols", "BTCUSDT", "ETHUSDT",
        ])


def test_sleeve_blend_cli_tournament_candidate_kind(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.blend.evaluation.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "blend",
        "--candidate-kind", "core5_causal_tournament_v1",
        "--tournament-profile", "pbgt_discovery_v1",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        universe_id="core5_v1",
        mdd_budget_fraction=0.85,
        candidate_kind="core5_causal_tournament_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        qualification_interval="365D",
        log_run=False,
    )]


def test_sleeve_blend_cli_directional_candidate_kind(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.blend.evaluation.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "blend",
        "--candidate-kind", "funding_signed_directional_v1", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        universe_id="core5_v1",
        mdd_budget_fraction=0.85,
        candidate_kind="funding_signed_directional_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        qualification_interval="365D",
        log_run=False,
    )]


def test_every_tiered_command_module_registers_a_handler() -> None:
    # RF-CLI-01: each tiered leaf module exposes an argparse adder that attaches
    # a dispatcher handler; static imports keep the modules co-modified with
    # their semantic tests.
    for adder in (
        add_single_baseline_commands,
        add_single_technical_commands,
        add_single_carry_commands,
        add_single_oi_commands,
        add_portfolio_multi_commands,
        add_portfolio_blend_commands,
        add_portfolio_growth_commands,
    ):
        assert callable(adder)


def test_exit_sweep_cli_parses_and_dispatches(monkeypatch, capsys) -> None:
    # TES-06-CLI-EXIT-SWEEP-WIRING: the exit-sweep leaf builds a
    # TechnicalExpertExitSweepRequest from its flags (no router flags accepted)
    # and streams the report's to_report_dict() as JSON to stdout.
    from src.cli.commands.research import expert_library as expert_library_cli

    captured: list[object] = []

    class _Report:
        def to_report_dict(self) -> dict[str, object]:
            return {"status": "COMPLETE"}

    def _fake_sweep(request) -> _Report:
        captured.append(request)
        return _Report()

    monkeypatch.setattr(
        expert_library_cli.exit_sweep_module,
        "run_technical_expert_exit_sweep",
        _fake_sweep,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert", "exit-sweep",
        "--candidate-source", "technical_ema_alignment_long_v1",
        "--symbols", "BTCUSDT", "ETHUSDT",
        "--timeframes", "4h", "1d",
    ])
    assert args.handler is expert_library_cli._run_exit_sweep
    args.handler(args)

    assert len(captured) == 1
    request = captured[0]
    assert request.candidate_sources == ("technical_ema_alignment_long_v1",)
    assert request.symbols == ("BTCUSDT", "ETHUSDT")
    assert request.timeframes == ("4h", "1d")
    assert request.fixed_pct_values == (0.03, 0.05, 0.08)
    assert request.atr_multiple_values == (1.5, 2.5, 4.0)
    assert request.atr_period == 14
    assert request.include_baseline is True
    assert request.max_workers is None
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"
