from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.growth import evaluation as growth_evaluation_module
from src.cli.main import main as cli_main

_SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


def _write_fake_ohlcv_files(root: Path) -> None:
    """Write complete 1h parquet archives with near-flat, noisy prices.

    The price surface is deliberately nearly flat: the pre-registered momentum
    signal cannot produce a significant out-of-sample t-stat on it, so the
    falsification verdict deterministically fails and the engine must hold CASH.
    """
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    n = len(hourly)
    rng = np.random.default_rng(0)
    directory = root / "futures" / "ohlcv" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    for i, symbol in enumerate(_SYMBOLS):
        noise = rng.normal(0.0, 0.0001, n)
        price = 100.0 * (1.0 + noise)
        df = pd.DataFrame({
            "timestamp": _ms(hourly),
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": 100.0,
            "quote_vol": 1000.0 * (1.0 + i),
        })
        df.to_parquet(directory / f"{symbol}.parquet")


@pytest.fixture
def fake_growth_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _write_fake_ohlcv_files(tmp_path)

    def _fake_ohlcv_path(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / "1h" / f"{symbol}.parquet"

    def _fake_funding_path(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol}.parquet"

    monkeypatch.setattr(growth_evaluation_module, "ohlcv_path", _fake_ohlcv_path)
    monkeypatch.setattr(growth_evaluation_module, "funding_path", _fake_funding_path)
    return tmp_path


# Rolling PASS fixture: every symbol's published taker-imbalance ratio oscillates
# around the neutral 0.5 on a slow, stationary sinusoid (phases spread evenly),
# and the per-hour drift is aligned with ``ratio - 0.5``.  The top/bottom book
# therefore rotates across segments -- genuinely generating closed trades -- while
# the signal remains strong and stationary in every 12-month discovery window, so
# ``taker_imbalance_v1`` wins the causal router deterministically and every real
# gate (observation, equal-duration fold, stress, symbol holdout) passes.
_TAKER_SYMBOLS = (
    "BTCUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "ADAUSDT",
    "LINKUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "APTUSDT", "SUIUSDT",
    "OPUSDT", "UNIUSDT", "ETHUSDT", "BNBUSDT", "AVAXUSDT", "NEARUSDT",
)
# UNI/ETH/BNB/AVAX/NEAR land in the symbol holdout partition at dev_fraction=0.60;
# their book is smaller, so a stronger per-hour drift lets the frozen finalist
# retain >= 50% of its development edge (the unchanged symbol-holdout retention gate).
_HOLDOUT_SYMBOLS = frozenset({"UNIUSDT", "ETHUSDT", "BNBUSDT", "AVAXUSDT", "NEARUSDT"})
_HOLDOUT_DRIFT_BOOST = 3.5
_TAKER_AMPLITUDE = 0.15
_TAKER_PERIOD_HOURS = 3000.0
_TAKER_DRIFT_RATE = 5e-4
_TAKER_NOISE = 2e-4


def _write_taker_sorted_ohlcv_files(
    root: Path,
    *,
    holdout_boost: float = _HOLDOUT_DRIFT_BOOST,
    drift_rate: float = _TAKER_DRIFT_RATE,
) -> None:
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    n = len(hourly)
    rng = np.random.default_rng(7)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    timestamp = (hourly - epoch) // pd.Timedelta("1ms")
    t_hours = np.arange(n, dtype=np.float64)
    phases = np.radians(np.linspace(0.0, 360.0, len(_TAKER_SYMBOLS), endpoint=False))
    directory = root / "futures" / "ohlcv" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    for i, symbol in enumerate(_TAKER_SYMBOLS):
        ratio = 0.5 + _TAKER_AMPLITUDE * np.sin(
            2.0 * np.pi * t_hours / _TAKER_PERIOD_HOURS + phases[i],
        )
        ratio = np.clip(ratio, 0.02, 0.98)
        boost = holdout_boost if symbol in _HOLDOUT_SYMBOLS else 1.0
        drift_hourly = drift_rate * boost * (ratio - 0.5)
        eps = rng.normal(0.0, _TAKER_NOISE, n)
        price = 100.0
        prices = np.empty(n)
        for t in range(n):
            price = price * (1.0 + drift_hourly[t] + eps[t])
            prices[t] = price
        quote_vol = 1000.0 * (1.0 + i)
        df = pd.DataFrame({
            "timestamp": timestamp,
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices,
            "volume": 100.0,
            "quote_vol": quote_vol,
            "taker_buy_quote": quote_vol * ratio,
        })
        df.to_parquet(directory / f"{symbol}.parquet")


@pytest.fixture
def fake_growth_env_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _write_taker_sorted_ohlcv_files(tmp_path)

    def _fake_ohlcv_path(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / "1h" / f"{symbol}.parquet"

    def _fake_funding_path(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol}.parquet"

    monkeypatch.setattr(growth_evaluation_module, "ohlcv_path", _fake_ohlcv_path)
    monkeypatch.setattr(growth_evaluation_module, "funding_path", _fake_funding_path)
    return tmp_path


@pytest.fixture
def fake_growth_env_holdout_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    # The development symbols carry the full rotating signal, but the symbol
    # holdout symbols are drift-free (holdout_boost=0), so the frozen finalist
    # retains none of its development edge out of the dev partition and the
    # symbol-holdout retention gate must hold CASH.
    _write_taker_sorted_ohlcv_files(tmp_path, holdout_boost=0.0)

    def _fake_ohlcv_path(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / "1h" / f"{symbol}.parquet"

    def _fake_funding_path(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol}.parquet"

    monkeypatch.setattr(growth_evaluation_module, "ohlcv_path", _fake_ohlcv_path)
    monkeypatch.setattr(growth_evaluation_module, "funding_path", _fake_funding_path)
    return tmp_path


@pytest.mark.slow
class TestGrowthEngineCli:
    # GEV2-13-NO-ALPHA-IS-CASH
    def test_no_alpha_is_flat_cash_and_cli_exits_cleanly(
        self,
        fake_growth_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        captured: list[growth_evaluation_module.GrowthEngineReport] = []
        original = growth_evaluation_module.run_growth_engine_evaluation

        def _spy(request):
            report = original(request)
            captured.append(report)
            return report

        monkeypatch.setattr(
            growth_evaluation_module, "run_growth_engine_evaluation", _spy,
        )

        with caplog.at_level(logging.INFO):
            cli_main([
                "research", "run", "portfolio", "growth",
                "--universe-size", "3",
                "--max-positions", "3",
                "--start", "2020-01-01",
                "--no-log-run",
            ])

        assert len(captured) == 1
        report = captured[0]
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.start == pd.Timestamp("2021-01-01", tz="UTC")
        assert len(report.trades) == 0
        assert len(report.equity) > 0
        assert float(report.equity.iloc[0]) == 10_000.0
        assert np.allclose(report.equity.to_numpy(), 10_000.0)
        assert report.promotion is None
        assert report.record is None
        assert report.falsification is not None
        assert report.falsification.passed is False
        assert report.selected_strategy is not None
        assert report.scorecard is not None
        assert report.scorecard.family_size == 12
        assert len(report.scorecard.entries) == 12

        assert "status=NO_ADMISSIBLE_ALPHA" in caplog.text

    # GSD-07: no passing finalist yields flat CASH plus a deterministic
    # candidate scorecard in the frozen registry order.
    def test_no_passing_finalist_publishes_deterministic_scorecard(
        self,
        fake_growth_env: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report = growth_evaluation_module.run_growth_engine_evaluation(
            growth_evaluation_module.GrowthEngineEvaluationRequest(
                universe=growth_evaluation_module.PitUniverseSpec(
                    universe_size=3, max_positions=3,
                ),
                construction=growth_evaluation_module.NetConstructionSpec(),
                start="2020-01-01",
                symbol_scope="dev",
                log_run=False,
            )
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.scorecard is not None
        # A family can clear the discovery plateau by chance on flat noise, but
        # the discovery-only lower-confidence admission still holds the sleeve
        # flat, so the engine remains CASH with an empty promotion result.
        assert report.scorecard.reason == "no_admitted_sleeve"
        order = [
            (entry.strategy_id, entry.parameter) for entry in report.scorecard.entries
        ]
        assert order == [
            ("funding_contrarian_v1", 42),
            ("funding_contrarian_v1", 84),
            ("funding_contrarian_v1", 168),
            ("taker_imbalance_v1", 42),
            ("taker_imbalance_v1", 84),
            ("taker_imbalance_v1", 168),
            ("vol_adjusted_trend_v1", 42),
            ("vol_adjusted_trend_v1", 84),
            ("vol_adjusted_trend_v1", 180),
            ("donchian_channel_position_v1", 42),
            ("donchian_channel_position_v1", 84),
            ("donchian_channel_position_v1", 168),
        ]
        funding_entries = [e for e in report.scorecard.entries if e.strategy_id == "funding_contrarian_v1"]
        assert all(e.status == "DATA_INVALID" for e in funding_entries)
        assert all(e.dev_discovery_score is None for e in funding_entries)
        price_entries = [e for e in report.scorecard.entries if e.strategy_id != "funding_contrarian_v1"]
        assert all(e.status == "SCREENED" for e in price_entries)
        assert report.promotion is None
        assert all(e.dev_discovery_score is not None for e in price_entries)

    def test_run_logs_growth_engine_status(self, fake_growth_env: Path) -> None:
        # The CLI must exit 0 (no exception) and reach the growth handler.
        cli_main([
            "research", "run", "portfolio", "growth",
            "--universe-size", "3",
            "--max-positions", "3",
            "--start", "2020-01-01",
            "--no-log-run",
        ])

    # GSD-06: family selection is dev-discovery only, passes multiplicity size
    # 12, and inspects the holdout only for the dev-selected finalist.  With a
    # stable cross-sectional taker-imbalance signal the dev finalist passes
    # every unchanged gate and the engine promotes it.
    def test_dev_discovery_selection_passes_and_promotes_finalist(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # GPR-05-REAL-PROMOTION-EVIDENCE: capture the promotion evidence that
        # compose_promotion_verdict actually receives; it must be measured
        # values (real reliability gates), never fabricated zero-valued PASSes.
        captured: dict[str, object] = {}
        original = growth_evaluation_module.compose_promotion_verdict

        def _spy(observation, folds, stress, holdout):
            captured["observation"] = observation
            captured["folds"] = folds
            captured["stress"] = stress
            captured["holdout"] = holdout
            return original(observation, folds, stress, holdout)

        monkeypatch.setattr(
            growth_evaluation_module, "compose_promotion_verdict", _spy,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            growth_evaluation_module.GrowthEngineEvaluationRequest(
                universe=growth_evaluation_module.PitUniverseSpec(
                    universe_size=18, max_positions=6, dev_fraction=0.60,
                ),
                construction=growth_evaluation_module.NetConstructionSpec(
                    rebalance_bars=3, no_trade_band=0.0,
                ),
                start="2020-01-01",
                symbol_scope="dev",
                log_run=False,
            )
        )
        assert report.status == "PASS"

        observation = captured["observation"]
        assert observation.verdict == "PASS"
        assert observation.trade_count >= 30
        assert observation.lcb90_cagr > 0.0
        assert captured["folds"].n_folds >= 3
        assert captured["folds"].gate_pass is True
        assert captured["stress"].verdict == "PASS"
        assert captured["stress"].trade_count >= 30
        assert captured["holdout"] is not None
        assert captured["holdout"].verdict == "PASS"
        assert captured["holdout"].trade_count >= 30
        assert report.selected_strategy == "taker_imbalance_v1"
        assert report.promotion is not None
        assert report.promotion.observation_verdict == "PASS"
        assert report.promotion.fold_gate_pass is True
        assert report.promotion.stress_verdict == "PASS"
        assert report.promotion.holdout_verdict == "PASS"
        assert report.falsification is not None
        assert report.falsification.passed is True
        assert report.falsification.binding_constraint == "none"
        assert report.sizing is not None
        assert report.sizing.selected_risk is not None
        assert len(report.trades) == 0
        assert len(report.equity) > 0
        assert report.equity.dropna().iloc[-1] > report.equity.iloc[0]

        # The scorecard is complete (exactly twelve variants), deterministic,
        # and only the selected family was admitted by the discovery plateau.
        assert report.scorecard is not None
        assert report.scorecard.family_size == 12
        assert report.scorecard.reason is None
        assert report.scorecard.selected_strategy_id == "taker_imbalance_v1"
        assert report.scorecard.selected_parameter == 42
        assert len(report.scorecard.entries) == 12
        taker_entries = [
            e for e in report.scorecard.entries if e.strategy_id == "taker_imbalance_v1"
        ]
        assert all(e.family_passed for e in taker_entries)
        assert all(e.status == "SCREENED" for e in taker_entries)

        # Family-size 12 is threaded into the unchanged Bonferroni multiplicity
        # correction used by the falsification verdict.
        from src.research.evaluation.falsification import multiplicity_adjusted_t_floor
        assert abs(
            report.falsification.required_t_floor
            - multiplicity_adjusted_t_floor(12, 2.0)
        ) < 1e-9

    def _pass_request(self) -> growth_evaluation_module.GrowthEngineEvaluationRequest:
        return growth_evaluation_module.GrowthEngineEvaluationRequest(
            universe=growth_evaluation_module.PitUniverseSpec(
                universe_size=18, max_positions=6, dev_fraction=0.60,
            ),
            construction=growth_evaluation_module.NetConstructionSpec(
                rebalance_bars=3, no_trade_band=0.0,
            ),
            start="2020-01-01",
            symbol_scope="dev",
            log_run=False,
        )

    @staticmethod
    def _fast_reliability_call(original, equity, closed_trade_count, config):
        """Keep negative-path evidence real while using the minimum valid bootstrap budget."""
        effective_config = dataclasses.replace(
            config or growth_evaluation_module.ReliabilityGateConfig(),
            n_bootstrap=100,
        )
        return original(equity, closed_trade_count, effective_config)

    # GPR-06-STRESS-AND-HOLDOUT-CASH
    def test_observation_gate_failure_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module.compute_equity_reliability_gate
        final_calls = {"count": 0}

        def _spy(equity, closed_trade_count, config=None):
            result = self._fast_reliability_call(
                original, equity, closed_trade_count, config,
            )
            # The first default 3000-bootstrap call after routing is the
            # stitched deployment observation gate.  Preserve its measured
            # statistics and force only its verdict to cover the fail-closed
            # observation branch; stress and holdout remain real evaluations.
            if config is None or (
                config.hurdle_rate != 0.0 and config.n_bootstrap == 3000
            ):
                final_calls["count"] += 1
                if final_calls["count"] == 1:
                    result = dataclasses.replace(result, verdict="FAIL")
            return result

        monkeypatch.setattr(
            growth_evaluation_module, "compute_equity_reliability_gate", _spy,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.falsification is not None
        assert report.falsification.passed is True
        assert report.scorecard is not None
        assert report.scorecard.reason == "observation"

    # GPR-06-STRESS-AND-HOLDOUT-CASH
    def test_stress_gate_failure_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module.compute_equity_reliability_gate

        def _spy(equity, closed_trade_count, config=None):
            result = self._fast_reliability_call(
                original, equity, closed_trade_count, config,
            )
            # The stress stream is the only reliability call with hurdle_rate=0;
            # force it to FAIL so the fail-closed composition is exercised while
            # observation/fold/holdout evidence stays genuinely measured.
            if config is not None and config.hurdle_rate == 0.0:
                result = dataclasses.replace(result, verdict="FAIL")
            return result

        monkeypatch.setattr(
            growth_evaluation_module, "compute_equity_reliability_gate", _spy,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.falsification is not None
        assert report.falsification.passed is True
        assert report.scorecard is not None
        assert report.scorecard.reason == "stress"

    # GPR-06-STRESS-AND-HOLDOUT-CASH
    def test_symbol_holdout_failure_holds_cash(
        self,
        fake_growth_env_holdout_dead: Path,
    ) -> None:
        report = growth_evaluation_module.run_growth_engine_evaluation(
            growth_evaluation_module.GrowthEngineEvaluationRequest(
                universe=growth_evaluation_module.PitUniverseSpec(
                    universe_size=18, max_positions=6, dev_fraction=0.60,
                ),
                construction=growth_evaluation_module.NetConstructionSpec(
                    rebalance_bars=3, no_trade_band=0.0,
                ),
                start="2020-01-01",
                symbol_scope="dev",
                log_run=False,
            )
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.falsification is not None
        assert report.falsification.passed is False
        assert report.falsification.binding_constraint == "symbol_holdout"
        assert report.scorecard is not None
        assert report.scorecard.reason == "symbol_holdout"

    # GPR-06-STRESS-AND-HOLDOUT-CASH
    def test_fold_gate_failure_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_reliability = growth_evaluation_module.compute_equity_reliability_gate

        def _spy(equity, config=None, fold_duration="6MS"):
            raise ValueError("synthetic fold-evidence failure")

        def _fast_reliability(equity, closed_trade_count, config=None):
            return self._fast_reliability_call(
                original_reliability, equity, closed_trade_count, config,
            )

        monkeypatch.setattr(
            growth_evaluation_module, "compute_equal_duration_fold_distribution", _spy,
        )
        monkeypatch.setattr(
            growth_evaluation_module, "compute_equity_reliability_gate", _fast_reliability,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.falsification is not None
        assert report.falsification.binding_constraint == "fold_concentration"
        assert report.scorecard is not None
        assert report.scorecard.reason == "fold_concentration"

    def test_infeasible_discovery_sizing_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module._segment_sizing
        original_reliability = growth_evaluation_module.compute_equity_reliability_gate

        def _spy(discovery_stream, sizing_config):
            return dataclasses.replace(
                original(discovery_stream, sizing_config),
                selected_risk=None,
                binding_constraint="infeasible",
            )

        monkeypatch.setattr(growth_evaluation_module, "_segment_sizing", _spy)
        monkeypatch.setattr(
            growth_evaluation_module,
            "compute_equity_reliability_gate",
            lambda equity, closed_trade_count, config=None: self._fast_reliability_call(
                original_reliability, equity, closed_trade_count, config,
            ),
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.scorecard is not None
        assert report.scorecard.reason == "infeasible_risk"

    def test_no_passing_family_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            growth_evaluation_module,
            "_screen_discovery_candidates",
            lambda *args, **kwargs: ((), ()),
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_passing_family"

    def test_insufficient_segment_bars_hold_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.research.portfolio.growth_router import GrowthSegment

        def _segments(dates):
            future = dates[-1] + pd.DateOffset(months=1)
            return [GrowthSegment((future,), (future,))]

        monkeypatch.setattr(growth_evaluation_module, "build_rolling_segments", _segments)
        monkeypatch.setattr(growth_evaluation_module, "enough_deployment_folds", lambda _: True)
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_passing_family"

    def test_missing_stress_returns_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module._compute_stream

        def _spy(weights, forward_returns, construction, forward_funding=None):
            stream = original(weights, forward_returns, construction, forward_funding)
            if construction.costs.fee_rate > self._pass_request().construction.costs.fee_rate:
                return dataclasses.replace(stream, net=stream.net * float("nan"))
            return stream

        monkeypatch.setattr(growth_evaluation_module, "_compute_stream", _spy)
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_stress_returns"

    def test_missing_deployment_returns_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module._sleeve_net_stream

        def _spy(screen, fwd, construction, settled_funding, grid):
            stream = original(screen, fwd, construction, settled_funding, grid)
            active_bars = int((screen.weights.abs().sum(axis=1) > 0.0).sum())
            if active_bars < 1000:
                return dataclasses.replace(stream, net=stream.net * float("nan"))
            return stream

        monkeypatch.setattr(growth_evaluation_module, "_sleeve_net_stream", _spy)
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_deployment_returns"

    def test_finalist_data_invalid_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module.screen_growth_strategy_weights

        def _spy(strategy_id, parameter, schedule, *args):
            screen = original(strategy_id, parameter, schedule, *args)
            if len(schedule) <= 3:
                return dataclasses.replace(screen, status="DATA_INVALID")
            return screen

        monkeypatch.setattr(
            growth_evaluation_module, "screen_growth_strategy_weights", _spy,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert report.scorecard is not None
        assert report.scorecard.reason == "finalist_data_invalid"

    def test_missing_context_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            growth_evaluation_module, "_segment_context_state", lambda *args: None,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_context"

    def test_missing_context_sleeve_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(growth_evaluation_module, "causal_router", lambda scores: None)
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_context_sleeve"

    def test_missing_selected_parameter_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module._discovery_sleeve_evidence

        def _spy(*args):
            evidence = original(*args)
            assert evidence is not None
            return dataclasses.replace(
                evidence,
                family=dataclasses.replace(evidence.family, chosen_parameter=None),
            )

        monkeypatch.setattr(growth_evaluation_module, "_discovery_sleeve_evidence", _spy)
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_chosen_parameter"

    # GPR-06-STRESS-AND-HOLDOUT-CASH
    def test_holdout_reliability_gate_failure_holds_cash(
        self,
        fake_growth_env_pass: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = growth_evaluation_module.compute_equity_reliability_gate
        final_calls = {"count": 0}

        def _spy(equity, closed_trade_count, config=None):
            result = self._fast_reliability_call(
                original, equity, closed_trade_count, config,
            )
            # The final observation and symbol-holdout gates use the default
            # 3000-bootstrap config (the router uses 500, stress uses hurdle 0).
            # The second such call is the symbol-holdout reliability gate; force
            # it to FAIL so the fail-closed composition is exercised.
            if config is None or (
                config.hurdle_rate != 0.0 and config.n_bootstrap == 3000
            ):
                final_calls["count"] += 1
                if final_calls["count"] == 2:
                    result = dataclasses.replace(result, verdict="FAIL")
            return result

        monkeypatch.setattr(
            growth_evaluation_module, "compute_equity_reliability_gate", _spy,
        )
        report = growth_evaluation_module.run_growth_engine_evaluation(
            self._pass_request(),
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.promotion is None
        assert len(report.trades) == 0
        assert report.falsification is not None
        assert report.falsification.passed is True
        assert report.scorecard is not None
        assert report.scorecard.reason == "symbol_holdout"
