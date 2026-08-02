from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.application.research.expert_portfolio import (
    admission_backtest as abt,
)
from src.application.research.expert_portfolio import rolling_admission as ra
from src.research.baseline.backtest import BacktestResult
from src.research.expert_portfolio.admission_types import (
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionRequest,
)
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.expert_portfolio.rolling import (
    RollingAdmissionConfig,
    build_rolling_rebalance_schedule,
)

_SOURCES = (
    "technical_macd_histogram_regime_long_v1",
    "technical_rsi_trend_pullback_long_v1",
)


def _request(symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> TechnicalLibraryAdmissionRequest:
    return TechnicalLibraryAdmissionRequest(
        candidate_sources=_SOURCES,
        symbols=symbols,
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        admission=LibraryAdmissionConfig(
            min_experts=2,
            max_experts=4,
            min_closed_trades=1,
            min_active_return_bars=1,
            max_abs_pairwise_log_return_correlation=1.0,
            max_joint_negative_return_rate=1.0,
            min_context_covered_states=1,
            max_combinations=1000,
            max_workers=1,
        ),
    )


def _config(symbols: tuple[str, ...]) -> RollingAdmissionConfig:
    return RollingAdmissionConfig(
        profile="technical-5symbol-rolling-v2",
        symbols=symbols,
        router_kind="per_symbol_winner_v2",
        proposal_search="bounded_family_unique_v2",
        base_delay_bars=1,
    )


def _window(config: RollingAdmissionConfig) -> object:
    return build_rolling_rebalance_schedule(
        pd.Timestamp("2022-04-01", tz="UTC"),
        pd.Timestamp("2024-10-01", tz="UTC"),
        config,
    )[0]


def _utc(ts) -> pd.Timestamp:
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _fake_load_technical_market_data(symbol, start, end):
    index = pd.date_range(_utc(start), _utc(end), freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.ones(len(index)),
            "high": np.ones(len(index)),
            "low": np.ones(len(index)),
            "close": np.ones(len(index)),
            "volume": np.ones(len(index)),
        },
        index=index,
    )
    funding = pd.Series(0.0, index=index)
    return frame, funding


def _fake_technical_backtest(frame, candidate, costs, funding, *, signal_delay_bars):
    index = frame.index
    n = len(index)
    # Deterministic per-source returns keyed by the bar timestamp, so the cached
    # (load-window) and uncached (scored-window) panels are bitwise equal.
    seed = sum(ord(ch) for ch in candidate.return_source) * 97 + 1
    base = (seed % 5 - 2) * 0.004
    noise = np.array(
        [np.random.default_rng(seed + int(ts)).normal(0.0, 0.002) for ts in index.asi8],
    )
    returns = np.full(n, base) + noise
    returns[0] = np.nan
    equity = pd.Series(
        (1.0 + np.nan_to_num(returns)).cumprod(), index=index, name="equity",
    )
    trades = pd.DataFrame(
        {"entry_bar": [0, 5], "exit_bar": [3, 8], "pnl": [1.0, 1.0], "return_pct": [0.01, 0.01]},
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


def _fake_context(router, index, start, end) -> pd.Series:
    return pd.Series(["up_low_vol"] * len(index), index=index)


def _patch_evidence_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls: dict[str, int] = {"n": 0}

    def _counting_backtest(frame, candidate, costs, funding, *, signal_delay_bars):
        calls["n"] += 1
        return _fake_technical_backtest(
            frame, candidate, costs, funding, signal_delay_bars=signal_delay_bars,
        )

    monkeypatch.setattr(abt, "_load_technical_market_data", _fake_load_technical_market_data)
    monkeypatch.setattr(abt, "run_technical_expert_backtest", _counting_backtest)
    monkeypatch.setattr(ra, "_build_admission_context", _fake_context)
    monkeypatch.setattr(ra, "technical_data_hashes", lambda symbol: {symbol: "hash"})
    return calls


def _assert_metrics_close(a, b) -> None:
    """Compare master-equity-derived metrics up to fixture float noise."""
    for field_name in ("cagr", "mdd", "sharpe", "sortino", "calmar", "expectancy",
                       "win_rate", "payoff_ratio", "exposure", "turnover"):
        assert getattr(a, field_name) == pytest.approx(getattr(b, field_name), rel=1e-9)
    assert a.trade_count == b.trade_count
    assert a.profit_factor == b.profit_factor
    assert a.trades_per_year == b.trades_per_year


class TestWindowScenarioEvidence:
    def test_one_cached_scenario_evaluation_per_symbol_source_is_reused(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # RAP-03: increasing the shortlist length never increases candidate-runner
        # calls; each (symbol, source) runs exactly once per scenario.
        calls = _patch_evidence_io(monkeypatch)
        request = _request()
        config = _config(request.symbols)
        window = _window(config)
        per_scenario = len(request.symbols) * len(request.candidate_sources)

        evidence = ra.build_window_scenario_evidence(request, window, config)
        assert evidence.base_candidate_runner_calls == per_scenario
        assert evidence.stress_candidate_runner_calls == per_scenario
        assert calls["n"] == 2 * per_scenario
        assert evidence.base_panel.shape[0] > 0
        assert list(evidence.base_trade_counts) == [
            definition.expert_id for definition in evidence.definitions
        ]

        calls["n"] = 0
        ra._select_for_window_v2(
            request, window, dataclasses.replace(config, shortlist_budget=2), None,
        )
        small_budget_runs = calls["n"]
        calls["n"] = 0
        ra._select_for_window_v2(
            request, window, dataclasses.replace(config, shortlist_budget=20), None,
        )
        large_budget_runs = calls["n"]
        # more proposals -> exactly one evidence build per run, no extra candidates
        assert small_budget_runs == 2 * per_scenario
        assert large_budget_runs == 2 * per_scenario

    def test_cached_and_uncached_fixture_paths_are_identical(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # RAP-04: the evidence-cached v2 path and the uncached per-proposal path
        # produce identical base/stress master metrics, gates, and final winner.
        _patch_evidence_io(monkeypatch)
        request = _request()
        config = _config(request.symbols)
        window = _window(config)

        selection, shortlist, cached_reports = ra._select_for_window_v2(
            request, window, config, None,
        )
        assert selection.status == "COMPLETE"
        assert shortlist
        assert selection.generated_nodes > 0
        assert selection.generation_limit == request.admission.max_combinations
        assert selection.generation_status == "COMPLETE"

        uncached_reports = []
        for proposal in shortlist:
            report = ra._run_proposal_backtest(
                proposal.expert_ids,
                request.router,
                window.scored_start,
                window.observed_end,
                config.initial_equity,
                request.admission.max_workers,
                router_kind="per_symbol_winner_v2",
                base_delay_bars=1,
                allow_same_symbol=True,
            )
            uncached_reports.append(
                dataclasses.replace(
                    report, diversification_rank_key=proposal.rank_key(),
                ),
            )

        assert len(cached_reports) == len(uncached_reports)
        for cached, uncached in zip(cached_reports, uncached_reports, strict=True):
            assert cached.proposal_id == uncached.proposal_id
            # gate verdicts, fold passes, and the promotion outcome are exact
            assert cached.observation_gate.verdict == uncached.observation_gate.verdict
            assert cached.stress_gate.verdict == uncached.stress_gate.verdict
            assert cached.observation_folds.gate_pass == uncached.observation_folds.gate_pass
            assert cached.stress_folds.gate_pass == uncached.stress_folds.gate_pass
            assert cached.promotion.status == uncached.promotion.status
            # master-equity-derived metrics match to float precision (the two
            # fixture paths reconstruct pct_change over different warm-up spans)
            _assert_metrics_close(cached.observation_metrics, uncached.observation_metrics)
            _assert_metrics_close(cached.stress_metrics, uncached.stress_metrics)
            assert cached.observation_gate.lcb90_cagr == pytest.approx(
                uncached.observation_gate.lcb90_cagr, rel=1e-9,
            )
            assert cached.stress_gate.lcb90_cagr == pytest.approx(
                uncached.stress_gate.lcb90_cagr, rel=1e-9,
            )
            assert cached.allocation_cost_total == pytest.approx(
                uncached.allocation_cost_total, rel=1e-9,
            )
            assert cached.stress_allocation_cost_total == pytest.approx(
                uncached.stress_allocation_cost_total, rel=1e-9,
            )

        cached_selected = ra.select_rebalance_proposal(cached_reports, None)
        uncached_selected = ra.select_rebalance_proposal(uncached_reports, None)
        assert (cached_selected is None) == (uncached_selected is None)
        if cached_selected is not None:
            assert cached_selected.proposal_id == uncached_selected.proposal_id

    def test_missing_scenario_evidence_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # RAP-03: a worker that raises fails closed with DataIntegrityError and
        # never yields partial evidence.
        _patch_evidence_io(monkeypatch)
        request = _request()
        config = _config(request.symbols)
        window = _window(config)

        def _broken(frame, candidate, costs, funding, *, signal_delay_bars):
            raise RuntimeError("market data unavailable")

        monkeypatch.setattr(abt, "run_technical_expert_backtest", _broken)
        from src.common.errors import DataIntegrityError

        with pytest.raises(DataIntegrityError, match="market data unavailable"):
            ra.build_window_scenario_evidence(request, window, config)
