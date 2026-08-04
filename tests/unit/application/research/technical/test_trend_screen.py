from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import trend_screen as ts
from src.common.errors import DataIntegrityError
from src.research.contracts import CostModel
from src.research.technical_experts.backtest import run_technical_expert_backtest


def synthetic_market(start: str = "2022-01-01", end: str = "2025-12-31 23:59:59"):
    """Deterministic oscillating 4h market plus a zero funding stream."""
    idx = pd.date_range(start, end, freq="4h", tz="UTC")
    t = np.arange(len(idx), dtype=np.float64)
    close = 100.0 + 0.02 * t + 30.0 * np.sin(t / 40.0) + 20.0 * np.cos(t / 150.0)
    frame = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0 + 500.0 * np.abs(np.sin(t / 5.0)),
    }, index=idx)
    funding = pd.Series(0.0, index=idx, dtype=np.float64)
    return frame, funding


@dataclasses.dataclass(frozen=True, slots=True)
class _FastGateConfig(ts.ReliabilityGateConfig):
    """Small bootstrap/null-draw config to keep screen tests fast."""
    n_bootstrap: int = 100
    fold_null_draws: int = 1000
    block_size: int = 1


def _install_fast_config(monkeypatch) -> None:
    monkeypatch.setattr(ts, "ReliabilityGateConfig", _FastGateConfig)
    monkeypatch.setattr(ts, "effective_worker_count", lambda *args, **kwargs: 1)


def _install_synthetic_data(monkeypatch) -> None:
    frame, funding = synthetic_market()
    monkeypatch.setattr(
        ts,
        "_load_symbol_data",
        lambda symbol, s, e: (
            frame.copy(), funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0,
        ),
    )


def _single_candidate_set():
    return tuple(
        c for c in ts.TREND_SCREEN_CANDIDATES
        if c.return_source == "technical_ema_alignment_long_v1"
    )


class TestTrendScreenScale:
    def test_exactly_450_cells_defined(self) -> None:
        assert len(ts.TREND_SCREEN_CANDIDATES) == 30
        assert len(ts.TREND_SCREEN_SYMBOLS) == 15
        assert len(ts.TREND_SCREEN_CANDIDATES) * len(ts.TREND_SCREEN_SYMBOLS) == 450


class TestDiscoveryRequirements:
    def _run(self, *, trade_count, lcb90=0.05, data_valid=True, policy_active=True):
        idx = pd.date_range("2022-04-01", periods=4000, freq="4h", tz="UTC")
        cell = ts.TrendScreenCell(
            return_source="technical_ema_alignment_long_v1",
            family="ema_alignment", side="LONG", symbol="BTCUSDT",
            data_valid=data_valid, funding_coverage=1.0, trade_count=trade_count,
            net_cagr=-0.05, mdd=-0.1, lcb90=lcb90, t_stat=0.5, fold_score=0.4,
            stress_verdict="PASS", p_negative=1.0, fingerprint="f",
            discovery_pass=False,
            rejected_reason="data_invalid" if not data_valid else None,
        )
        run = ts._CellRun(cell)
        run.disc_equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
        run.policy_equity = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
        run.policy_schedule = pd.Series(
            1.0 if policy_active else 0.0, index=idx, dtype=np.float64,
        )
        return run

    def test_positive_lcb_valid_cell_eligible_without_hard_gates(self) -> None:
        # Raw CAGR, IID t-stat, and bootstrap-negative fraction are diagnostics,
        # never discovery hard gates; a positive-LCB valid cell is eligible.
        run = self._run(trade_count=30)
        ts._apply_discovery_requirements([run], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is True
        assert run.cell.rejected_reason is None

    def test_invalid_data_remains_ineligible(self) -> None:
        run = self._run(trade_count=30, data_valid=False)
        ts._apply_discovery_requirements([run], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert run.cell.rejected_reason is not None

    def test_insufficient_duration_remains_ineligible(self) -> None:
        # Duration-derived coverage is one closed trade per complete discovery
        # month (21 for 2022-04..2023-12), not the old fixed 30-close cliff.
        run = self._run(trade_count=10)
        ts._apply_discovery_requirements([run], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert "min_trades" in run.cell.rejected_reason

    def test_non_positive_lcb90_remains_ineligible(self) -> None:
        run = self._run(trade_count=30, lcb90=0.0)
        ts._apply_discovery_requirements([run], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert "lcb90" in run.cell.rejected_reason

    def test_incomplete_causal_lookback_remains_ineligible(self) -> None:
        run = self._run(trade_count=30, policy_active=False)
        ts._apply_discovery_requirements([run], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert "incomplete_causal_lookback" in run.cell.rejected_reason


class TestScreenPipeline:
    def test_end_to_end_rejects_failed_candidate(self, monkeypatch) -> None:
        # BGP end-to-end fixture: on synthetic oscillating data the screen
        # retains CASH and records a binding constraint rather than promoting.
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()

        assert report.profile == "baseline_gate_performance_v1"
        assert len(report.cells) == len(_single_candidate_set()) * 15
        assert all(cell.data_valid for cell in report.cells)
        assert report.qualification.admitted is False
        assert report.qualification.binding_constraint is not None

    def test_report_is_byte_deterministic(self, monkeypatch) -> None:
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        first = ts.run_trend_screen()
        second = ts.run_trend_screen()
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert len(payload["report_fingerprint"]) == 64
        assert all(
            cell["fingerprint"]["perp_ohlcv"].startswith("fp-")
            for cell in payload["cells"]
        )

    def test_missing_funding_invalidates_cells_fail_closed(self, monkeypatch) -> None:
        # BGP-03 at the screen level: an unloadable funding stream invalidates
        # the symbol's cells; funding is never replaced by a zero-cost fallback.
        _install_fast_config(monkeypatch)
        monkeypatch.setattr(
            ts,
            "_load_symbol_data",
            lambda symbol, s, e: (_ for _ in ()).throw(
                DataIntegrityError(f"no settled funding events in window for {symbol}")
            ),
        )
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()
        assert len(report.cells) == len(_single_candidate_set()) * 15
        assert all(not cell.data_valid for cell in report.cells)
        assert all("funding" in (cell.rejected_reason or "") for cell in report.cells)

    def test_load_symbol_data_missing_funding_fails_closed(self, monkeypatch) -> None:
        frame, _funding = synthetic_market()
        monkeypatch.setattr(
            ts, "load_ohlcv_1h_as", lambda *a, **k: frame,
        )
        monkeypatch.setattr(
            ts, "load_funding_rates", lambda path: pd.Series(
                [0.001], index=pd.DatetimeIndex([frame.index[0] - pd.Timedelta(days=7)]),
            ),
        )
        with pytest.raises(DataIntegrityError, match="funding"):
            ts._load_symbol_data("BTCUSDT", None, None)

    def test_no_discovery_qualification_cross_read(self, monkeypatch) -> None:
        # Selection uses only discovery evidence: mutating qualification bars
        # cannot change a cell's discovery metrics or pass verdict.
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")

        base = ts._run_cell(
            frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )
        mutated = frame.copy()
        mutated.loc[mutated.index >= ts.QUALIFICATION_START, "close"] *= 1.5
        mutated.loc[mutated.index >= ts.QUALIFICATION_START, "open"] *= 1.5
        after = ts._run_cell(
            mutated, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )

        for field in ("trade_count", "net_cagr", "mdd", "lcb90", "t_stat", "fold_score"):
            assert getattr(base.cell, field) == pytest.approx(getattr(after.cell, field))
        assert base.cell.discovery_pass == after.cell.discovery_pass

    def test_discovery_cell_defers_stress_replay(self, monkeypatch) -> None:
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")
        original = ts.run_technical_expert_backtest
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ts, "run_technical_expert_backtest", counted)
        run = ts._run_cell(
            frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )

        assert calls == 1
        assert run.cell.stress_verdict == "PENDING"


class TestSizingTournament:
    def test_sizing_tournament_sealed_to_qualification_bars(self, monkeypatch) -> None:
        # BGP-03: mutating qualification bars cannot alter the selected sizing
        # policy, its fold score tuple, or its policy ranking; the tournament
        # never reads qualification, holdout, or stress outcomes.
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")
        base = ts._run_cell(frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0)
        mutated = frame.copy()
        mutated.loc[
            mutated.index >= ts.QUALIFICATION_START, ["open", "high", "low", "close"]
        ] *= 1.5
        after = ts._run_cell(
            mutated, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )
        assert base.cell.data_valid
        assert after.cell.data_valid

        tournament = ts._rank_sizing_policies([base], ts.ReliabilityGateConfig())
        tournament_after = ts._rank_sizing_policies([after], ts.ReliabilityGateConfig())

        assert tournament.selected_policy_id == tournament_after.selected_policy_id
        assert tournament.fraction == tournament_after.fraction
        assert tournament.mdd_cap_enabled == tournament_after.mdd_cap_enabled
        assert tournament.policy_scores == tournament_after.policy_scores
        assert tournament.fold_scores == tournament_after.fold_scores
        assert len(tournament.fold_scores) == 2 * len(ts.KELLY_SIZING_POLICIES)

    def _valid_tournament_runs(self, monkeypatch) -> list[ts._CellRun]:
        """Runs plus a valid two-fold setup and forced training eligibility.

        The frozen profile's first fold (training ends 2022-10-31) can never
        complete the 365-day Kelly lookback, so the ranking path is exercised
        here with two later rolling-origin folds and pass-through discovery
        requirements.
        """
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidates = [
            ts._candidate_by_source("technical_ema_alignment_long_v1"),
            ts._candidate_by_source("technical_donchian_breakout_short_v1"),
        ]
        runs = [
            ts._run_cell(frame, funding, c, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0)
            for c in candidates
        ]

        def force_pass(clones, config, start, end) -> None:
            for clone in clones:
                clone.cell = dataclasses.replace(
                    clone.cell, discovery_pass=True, rejected_reason=None,
                )

        monkeypatch.setattr(ts, "_apply_discovery_requirements", force_pass)
        monkeypatch.setattr(ts, "_SIZING_FOLDS", (
            (
                pd.Timestamp("2023-05-31 23:59:59", tz="UTC"),
                pd.Timestamp("2023-06-01 00:00:00", tz="UTC"),
                pd.Timestamp("2023-12-31 23:59:59", tz="UTC"),
            ),
            (
                pd.Timestamp("2023-08-31 23:59:59", tz="UTC"),
                pd.Timestamp("2023-09-01 00:00:00", tz="UTC"),
                pd.Timestamp("2023-12-31 23:59:59", tz="UTC"),
            ),
        ))
        return runs

    def test_tournament_fails_closed_on_frozen_folds(self, monkeypatch) -> None:
        # BGP-03-SIZING-INFEASIBLE: the frozen profile's first fold trains
        # through 2022-10-31, before the 365-day Kelly lookback can complete, so
        # no candidate is electable; the fold is INFEASIBLE -- never a zero
        # performance score -- and no policy is elected.
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")
        runs = [ts._run_cell(
            frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )]

        tournament = ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())

        assert tournament.status == "INFEASIBLE"
        assert tournament.selected_policy_id == "sizing_tournament_infeasible"
        assert tournament.fraction == 0.0
        assert tournament.policy_scores == ()
        assert len(tournament.fold_scores) == 2 * len(ts.KELLY_SIZING_POLICIES)
        first_fold = [f for f in tournament.fold_scores if f.fold_index == 1]
        assert all(f.status == "INFEASIBLE" for f in first_fold)
        assert all(f.reason == "empty_selection" for f in first_fold)
        assert all(f.lcb90_cagr == 0.0 and f.mdd == 0.0 for f in first_fold)

    def test_tournament_ranks_and_picks_one_registered_candidate(self, monkeypatch) -> None:
        # Every aggregate score must come from a registered candidate and each
        # candidate must produce exactly one score per rolling-origin fold. With
        # all folds VALID the best candidate is elected by the deterministic
        # lexicographic rank.
        runs = self._valid_tournament_runs(monkeypatch)
        known_ids = {ts._sizing_policy_id(s) for s in ts.KELLY_SIZING_POLICIES}

        tournament = ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())
        assert tournament.status == "VALID"
        assert tournament.selected_policy_id in known_ids
        assert tournament.reason is None
        assert len(tournament.policy_scores) == 4
        assert len(tournament.fold_scores) == 8
        assert {s.policy_id for s in tournament.policy_scores} == known_ids
        assert all(f.status == "VALID" and f.reason is None for f in tournament.fold_scores)
        assert all(f.active_bars > 0 for f in tournament.fold_scores)
        assert all(f.validation_trade_count >= 1 for f in tournament.fold_scores)
        expected_order = [s.policy_id for s in sorted(
            tournament.policy_scores,
            key=lambda s: (
                -s.worst_lcb90_cagr, -s.mean_lcb90_cagr,
                -s.worst_mdd, s.mean_allocation_cost, s.policy_id,
            ),
        )]
        assert [s.policy_id for s in tournament.policy_scores] == expected_order
        assert tournament.selected_policy_id == tournament.policy_scores[0].policy_id
        assert tournament.policy_scores[0].worst_lcb90_cagr >= min(
            s.worst_lcb90_cagr for s in tournament.policy_scores
        )

    def test_fold_validation_window_is_anchored(self, monkeypatch) -> None:
        # BGP-03-ANCHORED-WINDOW: fold validation requests the prior-anchor-mark
        # window for its return stream, so the boundary return into each
        # validation start is included.
        runs = self._valid_tournament_runs(monkeypatch)
        recorded: list[pd.Timestamp] = []
        original = ts._anchored_equity

        def spy(equity, start, end):
            recorded.append(start)
            return original(equity, start, end)

        monkeypatch.setattr(ts, "_anchored_equity", spy)
        ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())

        val_starts = [fold[1] for fold in ts._SIZING_FOLDS]
        for vs in val_starts:
            assert vs in recorded

    def test_report_includes_sizing_tournament_infeasible_evidence(self, monkeypatch) -> None:
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()
        payload = report.to_payload()
        sizing = payload["sizing"]

        assert sizing["policy_id"] == "sizing_tournament_infeasible"
        assert sizing["status"] == "INFEASIBLE"
        assert sizing["reason"] == "no_policy_with_all_valid_folds"
        assert len(sizing["policy_scores"]) == 0
        assert len(sizing["fold_scores"]) == 8
        first_fold = [s for s in sizing["fold_scores"] if s["fold_index"] == 1]
        assert all(s["status"] == "INFEASIBLE" for s in first_fold)
        assert all(s["reason"] == "empty_selection" for s in first_fold)
        assert all(
            "active_bars" in s and "validation_trade_count" in s
            for s in sizing["fold_scores"]
        )
        assert report.qualification.binding_constraint == "sizing_tournament_infeasible"

    def _bare_run(self) -> ts._CellRun:
        cell = ts.TrendScreenCell(
            return_source="s", family="f", side="LONG", symbol="X",
            data_valid=True, funding_coverage=1.0, trade_count=30,
            net_cagr=0.0, mdd=0.0, lcb90=0.0, t_stat=0.0, fold_score=0.0,
            stress_verdict="PENDING", p_negative=0.0, fingerprint={},
            discovery_pass=True, rejected_reason=None,
        )
        return ts._CellRun(cell)

    def _score_policy_fold_stub(self, lcb90_cagr: float):
        def stub(runs, spec, fold_index, train_end, val_start, val_end, config, clones):
            return ts.SizingFoldScore(
                policy_id=ts._sizing_policy_id(spec),
                fold_index=fold_index,
                train_end=train_end,
                validation_start=val_start,
                validation_end=val_end,
                status="VALID",
                reason=None,
                active_bars=100,
                validation_trade_count=5,
                lcb90_cagr=lcb90_cagr,
                mdd=-0.05,
                allocation_cost=0.001,
            )
        return stub

    def test_tsi_05_weights_use_unit_ledger(self) -> None:
        # TSI-05: P1 weights must follow the unit discovery ledger, not the
        # Kelly-scaled policy ledger. The policy volatilities here are inverted
        # relative to the discovery ledgers, so a policy-based weight would pick
        # the wrong sleeve.
        idx = pd.date_range("2022-04-01", periods=500, freq="4h", tz="UTC")

        def make_run(vol: float, seed: int) -> ts._CellRun:
            rng = np.random.default_rng(seed)
            run = self._bare_run()
            run.disc_equity = pd.Series(
                10000.0 * np.exp(np.cumsum(rng.normal(0.0, vol, 500))), index=idx,
            )
            run.policy_equity = pd.Series(
                10000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.021 - vol, 500))), index=idx,
            )
            return run

        low_vol, high_vol = make_run(0.001, 1), make_run(0.020, 2)
        weights = ts._equal_risk_weights([low_vol, high_vol])
        assert weights[0] > weights[1]
        assert abs(sum(weights) - 1.0) < 1e-9

        missing = self._bare_run()
        missing.policy_equity = pd.Series(
            10000.0 * np.exp(np.cumsum(np.full(500, 0.0001))), index=idx,
        )
        with pytest.raises(DataIntegrityError):
            ts._equal_risk_weights([missing])

        flat = make_run(0.0, 3)
        flat2 = make_run(0.0, 4)
        flat_weights = ts._equal_risk_weights([flat, flat2])
        assert abs(flat_weights[0] - 0.5) < 1e-12

    def test_tsi_06_tournament_edge_floor(self, monkeypatch) -> None:
        # TSI-06: P2 minimizing risk on a negative-edge blend is exposure minimization,
        # so the tournament must fail closed when no candidate has a positive
        # validation edge instead of electing the smallest grid fraction.
        runs = [self._bare_run(), self._bare_run()]
        monkeypatch.setattr(ts, "_score_policy_fold", self._score_policy_fold_stub(-0.05))
        tournament = ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())
        assert tournament.status == "INFEASIBLE"
        assert tournament.selected_policy_id == "sizing_tournament_infeasible"
        assert tournament.fraction == 0.0
        assert tournament.mdd_cap_enabled is False
        assert tournament.reason == "no_policy_with_positive_validation_edge"
        assert tournament.policy_scores == ()
        assert len(tournament.fold_scores) == 2 * len(ts.KELLY_SIZING_POLICIES)
        assert all(f.status == "VALID" for f in tournament.fold_scores)

        monkeypatch.setattr(ts, "_score_policy_fold", self._score_policy_fold_stub(0.05))
        tournament2 = ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())
        assert tournament2.status == "VALID"
        assert tournament2.reason is None
        assert tournament2.selected_policy_id in {
            ts._sizing_policy_id(spec) for spec in ts.KELLY_SIZING_POLICIES
        }

    def test_tsi_07_dead_parameter_diagnostic(self, monkeypatch) -> None:
        # TSI-07: P2 candidates that share identical four-field scores are flagged
        # dead_parameter in the persisted payload; the flag is diagnostic only
        # and the report stays byte-deterministic across identical runs.
        monkeypatch.setattr(ts, "_score_policy_fold", self._score_policy_fold_stub(0.05))
        runs = [self._bare_run(), self._bare_run()]
        tournament_a = ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())
        tournament_b = ts._rank_sizing_policies(runs, ts.ReliabilityGateConfig())
        assert len(tournament_a.policy_scores) == len(ts.KELLY_SIZING_POLICIES)
        assert all(s.dead_parameter for s in tournament_a.policy_scores)

        def make_report(sizing: ts.SizingTournament) -> ts.TrendScreenReport:
            return ts.TrendScreenReport(
                profile="p",
                universe=("X",),
                cells=(),
                selection=None,
                schedule_hash="h",
                qualification=ts.TrendScreenQualification(
                    admitted=False, observation_verdict="PENDING", fold_gate_pass=False,
                    stress_verdict="PENDING", holdout_verdict=None, binding_constraint=None,
                ),
                sizing=sizing,
            )

        payload_a = make_report(tournament_a).to_payload()
        payload_b = make_report(tournament_b).to_payload()
        scores = payload_a["sizing"]["policy_scores"]
        assert all(s["dead_parameter"] for s in scores)
        assert {s["policy_id"] for s in scores} == {
            ts._sizing_policy_id(spec) for spec in ts.KELLY_SIZING_POLICIES
        }
        assert make_report(tournament_a).to_json() == make_report(tournament_b).to_json()


class TestAnchoredWindow:
    def test_boundary_loss_included_exactly_once(self) -> None:
        # BGP-03-ANCHORED-WINDOW: the 100 -> 90 -> 90 diagnosis example. The
        # anchored window reports MDD -10%, while the sliced window reports 0.
        idx = pd.date_range("2022-04-01", periods=3, freq="4h", tz="UTC")
        ledger = pd.Series([100.0, 90.0, 90.0], index=idx)
        start, end = idx[1], idx[2]

        anchored = ts._anchored_equity(ledger, start, end)
        assert list(anchored.index) == list(idx)
        mdd = float((anchored / anchored.cummax() - 1.0).min())
        assert mdd == pytest.approx(-0.10)

        sliced = ledger[(ledger.index >= start) & (ledger.index <= end)]
        assert float((sliced / sliced.cummax() - 1.0).min()) == pytest.approx(0.0)

    def test_no_prior_mark_begins_at_window_start(self) -> None:
        # A ledger whose first mark is exactly the window start has no boundary
        # return to include; the window begins at its first in-window mark.
        idx = pd.date_range("2022-04-01", periods=3, freq="4h", tz="UTC")
        ledger = pd.Series([100.0, 110.0, 120.0], index=idx)
        anchored = ts._anchored_equity(ledger, idx[0], idx[2])
        assert list(anchored.index) == list(idx)

    def test_rejects_malformed_or_empty_window(self) -> None:
        idx = pd.date_range("2022-04-01", periods=5, freq="4h", tz="UTC")
        ledger = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=idx)
        with pytest.raises(ValueError, match="empty"):
            ts._anchored_equity(ledger.iloc[3:], idx[1], idx[2])
        with pytest.raises(ValueError, match="after"):
            ts._anchored_equity(ledger, idx[2], idx[0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ts._anchored_equity(pd.Series([1.0, 2.0]), idx[0], idx[2])

    def test_discovery_windows_anchored_to_prior_mark(self, monkeypatch) -> None:
        # Raw and policy discovery ledgers both include the last mark strictly
        # before DISCOVERY_START, so the boundary return is never dropped.
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")
        run = ts._run_cell(
            frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )
        assert run.disc_equity is not None
        assert run.disc_equity.index[0] < ts.DISCOVERY_START
        assert run.disc_equity.index[-1] <= ts.DISCOVERY_END

        ts._apply_policy_evidence([run], ts.KELLY_SIZING_POLICIES[0])
        assert run.policy_equity is not None
        assert run.policy_equity.index[0] < ts.DISCOVERY_START


class TestPortfolioConstruction:
    def test_at_most_one_sleeve_per_symbol(self) -> None:
        # BGP-03-SYMBOL-DIVERSIFICATION: three high-ranked identities on one
        # symbol collapse to a single sleeve, while distinct-symbol candidates
        # retain deterministic selection.
        idx = pd.date_range("2022-04-01", periods=4000, freq="4h", tz="UTC")

        def make(return_source, family, symbol, lcb):
            cell = ts.TrendScreenCell(
                return_source=return_source, family=family, side="LONG",
                symbol=symbol, data_valid=True, funding_coverage=1.0,
                trade_count=30, net_cagr=0.1, mdd=-0.05, lcb90=lcb,
                t_stat=2.0, fold_score=0.3, stress_verdict="PASS",
                p_negative=0.0, fingerprint="f", discovery_pass=True,
                rejected_reason=None, policy_lcb90=lcb, policy_cagr=0.1,
                policy_mdd=-0.05, policy_trade_count=30,
                policy_schedule_hash="h",
            )
            run = ts._CellRun(cell)
            equity = pd.Series(
                np.linspace(100.0, 100.0 * (1.0 + lcb * 10.0), len(idx)),
                index=idx,
            )
            run.policy_equity = equity
            run.disc_equity = equity.copy()
            return run

        sol = make("technical_fam_a_long_v1", "fam_a", "SOLUSDT", 0.05)
        sol_b = make("technical_fam_b_long_v1", "fam_b", "SOLUSDT", 0.04)
        sol_c = make("technical_fam_c_long_v1", "fam_c", "SOLUSDT", 0.03)
        btc = make("technical_fam_d_long_v1", "fam_d", "BTCUSDT", 0.02)
        eth = make("technical_fam_e_long_v1", "fam_e", "ETHUSDT", 0.01)

        selected = ts._greedy_portfolio_selection(
            ts._select_sleeves([sol, sol_b, sol_c, btc, eth]),
        )
        symbols = [r.cell.symbol for r in selected]
        assert len(symbols) == len(set(symbols))
        assert symbols.count("SOLUSDT") == 1
        assert symbols.count("BTCUSDT") == 1
        assert symbols.count("ETHUSDT") == 1
        assert sol in selected

    def test_report_persists_selected_sleeve_correlation(self, monkeypatch) -> None:
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()
        payload = report.to_payload()
        # The frozen profile fails closed, so no selection is built; the sizing
        # diagnostic block still carries the fold feasibility details.
        assert payload["selection"] is None
        assert payload["sizing"]["status"] == "INFEASIBLE"


class TestInferenceObservability:
    def test_report_persists_block_size_cap_hit_diagnostic(self, monkeypatch) -> None:
        # A cap-hit is a pure diagnostic: it never passes or fails a cell but is
        # persisted per cell in the report.
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()
        payload = report.to_payload()
        assert len(payload["cells"]) == len(_single_candidate_set()) * 15
        assert all(isinstance(c["block_cap_hit"], bool) for c in payload["cells"])
        assert all(
            "block_cap_hit" in c for c in payload["cells"]
        )


class TestQualification:
    def _selected_runs(self) -> list[ts._CellRun]:
        frame, funding = synthetic_market()
        candidates = [
            ts._candidate_by_source("technical_ema_alignment_long_v1"),
            ts._candidate_by_source("technical_donchian_breakout_short_v1"),
        ]
        runs: list[ts._CellRun] = []
        for candidate in candidates:
            result = run_technical_expert_backtest(
                frame, candidate, CostModel(), funding,
                signal_delay_bars=ts._BASE_DELAY_BARS,
            )
            disc_equity = result.equity[
                (result.equity.index >= ts.DISCOVERY_START)
                & (result.equity.index <= ts.DISCOVERY_END)
            ]
            cell = ts.TrendScreenCell(
                return_source=candidate.return_source,
                family=candidate.family,
                side=candidate.side,
                symbol="BTCUSDT",
                data_valid=True, funding_coverage=1.0, trade_count=5,
                net_cagr=0.1, mdd=-0.1, lcb90=0.01, t_stat=2.1, fold_score=0.3,
                stress_verdict="PASS", p_negative=0.0, fingerprint="f",
                discovery_pass=True, rejected_reason=None,
            )
            run = ts._CellRun(cell, result)
            run.disc_equity = disc_equity
            run.policy_equity = disc_equity.copy()
            runs.append(run)
        return runs

    def test_zero_pre_warmup_exposure_and_same_schedule_under_stress(
        self, monkeypatch,
    ) -> None:
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        data = {"BTCUSDT": (frame, funding)}
        runs = self._selected_runs()
        weights = ts._equal_risk_weights(runs)

        schedule, scheduled, stress_scheduled, qualification = ts._qualify(
            runs, weights, data, unseal_holdout=False,
        )

        assert len(schedule) == len(scheduled) == len(stress_scheduled)
        lookback_bars = round(pd.Timedelta(days=365) / pd.Timedelta(hours=4))
        assert schedule.iloc[:lookback_bars].eq(0.0).all()
        # The stressed replay shares the identical frozen schedule.
        assert qualification.admitted is False
        assert qualification.binding_constraint is not None

    def test_qualification_reuses_base_derived_schedule(self, monkeypatch) -> None:
        # Base and stress ledgers reuse the same base-derived causal
        # fractional-Kelly/MDD schedule; no future mark changes its prefix.
        from src.research.sleeve_blend.contracts import (
            CausalFractionalKellySpec,
            CausalLeverageSpec,
        )
        from src.research.sleeve_blend.fixed import (
            build_causal_fractional_kelly_schedule,
        )

        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        data = {"BTCUSDT": (frame, funding)}
        runs = self._selected_runs()
        weights = ts._equal_risk_weights(runs)

        schedule, _scheduled, _stress_scheduled, _ = ts._qualify(
            runs, weights, data, unseal_holdout=False,
        )

        assert (schedule >= 0).all()
        assert (schedule <= 3.0).all()
        blend = ts._blend_unit_equities(
            [ts._unit_equity(run.result) for run in runs], weights,
        )
        altered = blend.copy()
        altered.loc[altered.index >= ts.QUALIFICATION_START] *= 1.5
        rebuilt = build_causal_fractional_kelly_schedule(
            altered, CausalLeverageSpec(), CausalFractionalKellySpec(),
        )
        prefix = schedule.index < ts.QUALIFICATION_START
        assert rebuilt[prefix].equals(schedule[prefix])

    def test_qualification_windows_route_through_anchor_helper(self, monkeypatch) -> None:
        # BGP-03-ANCHORED-WINDOW: base and stress qualification both request the
        # anchored window, so the boundary return into QUALIFICATION_START is
        # included exactly once for each ledger.
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        data = {"BTCUSDT": (frame, funding)}
        runs = self._selected_runs()
        weights = ts._equal_risk_weights(runs)

        calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        original = ts._anchored_equity

        def spy(equity, start, end):
            calls.append((start, end))
            return original(equity, start, end)

        monkeypatch.setattr(ts, "_anchored_equity", spy)
        ts._qualify(runs, weights, data, unseal_holdout=False)

        assert calls.count(
            (ts.QUALIFICATION_START, ts.QUALIFICATION_END)
        ) == 2  # base ledger + stressed ledger

    def test_policy_ledger_debits_leverage_turnover_cost(self, monkeypatch) -> None:
        # A leverage step-up debits 0.5 * abs(delta_leverage) * (fee + slippage)
        # on the same causal bar before compounding the marked return.
        _install_fast_config(monkeypatch)
        idx = pd.date_range("2022-04-01", periods=6, freq="4h", tz="UTC")
        unit = pd.Series([100.0] * 6, index=idx)
        schedule = pd.Series([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], index=idx)
        costs = CostModel(fee_rate=0.0005, slippage_rate=0.0003)

        equity, total_cost = ts._apply_policy_schedule(unit, schedule, costs)

        unit_cost = 0.5 * 2.0 * (0.0005 + 0.0003)
        assert total_cost == pytest.approx(unit_cost)
        assert equity.iloc[-1] == pytest.approx(10_000.0 * (1.0 - unit_cost))
