"""Contract scenarios XSC-06, XSV3-04, XSV3-05, XSV4-06, SCENARIO_XSV5_02_PROFILE_DISPATCHES_DUAL_FAMILY_PIPELINE, and SCENARIO_XSV5_03_UNKNOWN_PROFILE_STILL_FAILS_CLOSED for the XS screen orchestration.

XSC-06-SCREEN-DETERMINISTIC-AND-SEALED, XSV3-04-FINAL-LEDGER-COSTS,
XSV3-05-FAIL-CLOSED, XSV4-06-PROFILE-DISPATCH, XSV5-02-PROFILE-DISPATCH,
XSV5-03-UNKNOWN-PROFILE-FAIL-CLOSED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import xs_trend_screen as xs
from src.common.errors import DataIntegrityError
from src.research.technical_experts.catalog import TECHNICAL_CANDIDATES


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


def _install_synthetic_data(monkeypatch) -> None:
    """Stub the data loader and speed up the score builder.

    The composite-score math is covered by the cross_sectional unit tests; here
    a deterministic per-symbol oscillation drives the orchestration so the two
    runs must agree byte-for-byte without the heavy 450-identity replay. The
    per-symbol salt keeps the cross-sectional alpha panels non-degenerate.
    """
    def fake_load(symbol: str, start, end) -> tuple[pd.DataFrame, pd.Series, dict[str, str], float]:
        frame, funding = synthetic_market()
        frame = frame.copy()
        salt = float(sum(ord(c) for c in symbol))
        frame["close"] = frame["close"] * (1.0 + 0.02 * (salt % 7))
        frame["open"] = frame["open"] * (1.0 + 0.02 * (salt % 7))
        frame["taker_buy_ratio"] = (
            0.5 + 0.03 * np.sin(np.arange(len(frame)) / 9.0 + salt)
        )
        frame.attrs["symbol"] = symbol
        return frame, funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0

    def fast_score(frame: pd.DataFrame) -> pd.Series:
        salt = sum(ord(c) for c in str(frame.attrs["symbol"]))
        values = np.sin(np.arange(len(frame)) / 7.0 + salt) + np.cos(np.arange(len(frame)) / 29.0)
        return pd.Series(values, index=frame.index, name="composite")

    monkeypatch.setattr(xs, "_load_symbol_data", fake_load)
    monkeypatch.setattr(xs, "_symbol_composite_score", fast_score)


class TestXsScreenOrchestration:
    def test_xsc_06_payload_is_byte_deterministic(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)

        first = xs.run_xs_trend_screen()
        second = xs.run_xs_trend_screen()

        assert first.profile == "xs_neutral_composite_v1"
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert len(payload["report_fingerprint"]) == 64
        assert set(payload["universe"]) == set(xs.TREND_SCREEN_SYMBOLS)
        assert payload["spec"]["round_trip_cost_rate"] == 0.0008
        assert set(payload["discovery"]) >= {"admitted", "binding_constraint", "sharpe", "beta"}
        assert set(payload["qualification"]) >= {"admitted", "binding_constraint", "sharpe", "beta"}
        for symbol, stats in payload["symbols"].items():
            assert stats["fingerprint"]["perp_ohlcv"] == f"fp-{symbol}"

    def test_xsc_06_routes_end_through_resolve_evaluation_end(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        calls: list[tuple[object, bool]] = []
        original = xs.resolve_evaluation_end

        def spy(end, *, unseal_holdout):
            calls.append((end, unseal_holdout))
            return original(end, unseal_holdout=unseal_holdout)

        monkeypatch.setattr(xs, "resolve_evaluation_end", spy)
        xs.run_xs_trend_screen()
        xs.run_xs_trend_screen(unseal_holdout=True)

        assert calls[0] == (None, False)
        assert calls[1] == (None, True)

    def test_xsc_06_explicit_end_past_cutoff_raises(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        with pytest.raises(RuntimeError, match="Holdout sealed"):
            xs.run_xs_trend_screen(end="2026-06-01")

    def test_xsc_06_registers_nothing_in_production_catalog(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        before = len(TECHNICAL_CANDIDATES)
        xs.run_xs_trend_screen()
        assert len(TECHNICAL_CANDIDATES) == before == 18

    def test_xsc_06_unavailable_symbol_fails_closed_named(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)

        def failing_load(symbol: str, start, end):
            if symbol == "NEARUSDT":
                raise DataIntegrityError("no settled funding events in window")
            frame, funding = synthetic_market()
            frame = frame.copy()
            frame.attrs["symbol"] = symbol
            return frame, funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0

        monkeypatch.setattr(xs, "_load_symbol_data", failing_load)
        report = xs.run_xs_trend_screen()

        assert report.qualification.admitted is False
        assert report.qualification.binding_constraint is not None
        assert "symbol_unavailable:NEARUSDT" in report.qualification.binding_constraint
        assert report.discovery.admitted is False
        assert report.discovery.binding_constraint == report.qualification.binding_constraint

    def test_xsc_06_insufficient_common_grid_fails_closed(self, monkeypatch) -> None:
        def tiny_load(symbol: str, start, end) -> tuple[pd.DataFrame, pd.Series, dict[str, str], float]:
            idx = pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC")
            frame = pd.DataFrame({
                "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0],
                "volume": [1000.0],
            }, index=idx)
            funding = pd.Series(0.0, index=idx)
            return frame, funding, {"perp_ohlcv": f"fp-{symbol}"}, 1.0

        monkeypatch.setattr(xs, "_load_symbol_data", tiny_load)
        report = xs.run_xs_trend_screen()

        assert report.qualification.admitted is False
        assert "insufficient_common_grid" in report.qualification.binding_constraint

    def test_xsc_06_window_slice_anchors_to_prior_mark(self) -> None:
        # The evaluation window includes the mark strictly before its start, so
        # the boundary return into discovery/qualification is never dropped.
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        sliced = xs._window_series(series, idx[2], idx[4])
        assert list(sliced.index) == list(idx[1:5])
        assert list(sliced.values) == [2.0, 3.0, 4.0, 5.0]

    def test_xsc_06_window_slice_no_prior_mark_starts_at_window(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        sliced = xs._window_series(series, idx[0], idx[2])
        assert list(sliced.index) == list(idx[:3])

    def test_xsc_06_composite_score_is_family_mean_within_unit_range(self) -> None:
        # The per-symbol composite is the mean of 15 {-1,0,1} family scores, so
        # it stays inside [-1, +1] and is non-trivial on a trending market.
        idx = pd.date_range("2022-01-01", periods=800, freq="4h", tz="UTC")
        t = np.arange(len(idx), dtype=np.float64)
        close = 100.0 + 30.0 * np.sin(t / 20.0)
        frame = pd.DataFrame({
            "open": close - 0.2, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 1000.0,
        }, index=idx)
        score = xs._symbol_composite_score(frame)
        assert score.index.equals(frame.index)
        assert float(score.min()) >= -1.0
        assert float(score.max()) <= 1.0
        assert float(score.std()) > 0.0

    def test_xsc_06_unknown_family_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="no LONG candidate"):
            xs._candidate_by_family_side("not_a_family", "LONG")


class TestAlphaProfileOrchestration:
    def test_xsa_03_unknown_profile_fails_closed(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        with pytest.raises(ValueError, match="unknown xs screen profile"):
            xs.run_xs_trend_screen(profile="not_a_profile")

    def test_xsa_03_v1_json_stays_unchanged(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        default = xs.run_xs_trend_screen()
        explicit = xs.run_xs_trend_screen(profile=xs.XS_NEUTRAL_PROFILE_ID)
        assert default.profile == "xs_neutral_composite_v1"
        assert default.to_json() == explicit.to_json()
        payload = default.to_payload()
        assert "alpha_spec" not in payload
        assert "stress" not in payload
        assert "stress_spec" not in payload

    def test_xsa_03_v2_payload_has_frozen_alpha_and_stress_records(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        first = xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID)
        second = xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID)

        assert first.profile == "xs_alpha_multihorizon_v2"
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert len(payload["report_fingerprint"]) == 64
        assert payload["alpha_spec"]["signal_windows"] == [42, 84, 168]
        assert payload["alpha_spec"]["components"] == [
            "trend", "funding_contrarian", "taker_imbalance",
        ]
        assert payload["stress_spec"]["execution_delay_bars"] == 2
        assert payload["stress_spec"]["fee_rate"] == pytest.approx(0.00075)
        assert payload["stress_spec"]["slippage_rate"] == pytest.approx(0.0006)
        assert payload["stress"]["qualification"]["admitted"] is not None
        assert payload["stress"]["discovery"]["admitted"] is not None
        for stats in payload["symbols"].values():
            assert stats["fingerprint"]["perp_ohlcv"].startswith("fp-")

    def test_xsa_03_stress_reuses_base_target_weights_verbatim(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        captured: list[pd.DataFrame] = []
        original = xs.run_xs_composite_ledger

        def spy(weights, opens, bar_funding, spec):
            captured.append(weights.copy())
            return original(weights, opens, bar_funding, spec)

        monkeypatch.setattr(xs, "run_xs_composite_ledger", spy)
        xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID)

        assert len(captured) == 2
        assert captured[0].equals(captured[1])
        assert captured[0].index.equals(captured[1].index)

    def test_xsa_03_v2_research_only_registers_nothing(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        before = len(TECHNICAL_CANDIDATES)
        xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID)
        assert len(TECHNICAL_CANDIDATES) == before

    def test_xsa_03_alpha_panel_invalid_fails_closed_named(self, monkeypatch) -> None:
        def nan_taker_load(symbol: str, start, end):
            frame, funding = synthetic_market()
            frame = frame.copy()
            frame["taker_buy_ratio"] = 0.5
            frame.loc[frame.index[100], "taker_buy_ratio"] = 1.5
            frame.attrs["symbol"] = symbol
            return frame, funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0

        monkeypatch.setattr(xs, "_load_symbol_data", nan_taker_load)
        report = xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID)
        assert report.qualification.admitted is False
        assert "alpha_panel_invalid" in report.qualification.binding_constraint
        assert report.alpha_spec is not None

    def test_xsa_03_stress_holdout_replayed_when_unsealed(self, monkeypatch) -> None:
        idx = pd.date_range("2022-01-01", "2026-07-07 20:00:00", freq="4h", tz="UTC")
        t = np.arange(len(idx), dtype=np.float64)

        def extended_load(symbol: str, start, end):
            salt = float(sum(ord(c) for c in symbol))
            close = 100.0 + 0.02 * t + 30.0 * np.sin(t / 40.0 + salt) + 20.0 * np.cos(t / 150.0)
            frame = pd.DataFrame({
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
                "taker_buy_ratio": 0.5 + 0.03 * np.sin(t / 9.0 + salt),
            }, index=idx)
            funding = pd.Series(0.0, index=idx, dtype=np.float64)
            frame.attrs["symbol"] = symbol
            return frame, funding, {"perp_ohlcv": f"fp-{symbol}"}, 1.0

        monkeypatch.setattr(xs, "_load_symbol_data", extended_load)
        report = xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID, unseal_holdout=True)
        assert report.holdout is not None
        assert report.holdout_start is not None
        assert report.holdout_start >= xs.HOLDOUT_CUTOFF
        payload = report.to_payload()
        assert payload["stress"]["holdout"]["admitted"] is not None

    def test_xsa_03_persistence_is_byte_deterministic(self, monkeypatch, tmp_path) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs.run_xs_trend_screen(profile=xs.XS_ALPHA_PROFILE_ID)
        path = tmp_path / "report.json"
        xs.persist_xs_screen_report(report, path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == report.to_json()
        assert (
            xs.xs_screen_report_path(xs.XS_ALPHA_PROFILE_ID).name
            == "xs_alpha_multihorizon_v2.json"
        )
        assert (
            xs.xs_screen_report_path(xs.XS_NEUTRAL_PROFILE_ID).name
            == "xs_neutral_composite_v1.json"
        )


class TestContextualProfileOrchestration:
    def test_xsv3_04_final_target_matrix_replayed_verbatim_for_base_and_stress(
        self, monkeypatch,
    ) -> None:
        _install_synthetic_data(monkeypatch)
        captured: list[pd.DataFrame] = []
        original = xs.run_xs_composite_ledger

        def spy(weights, opens, bar_funding, spec):
            captured.append(weights.copy())
            return original(weights, opens, bar_funding, spec)

        monkeypatch.setattr(xs, "run_xs_composite_ledger", spy)
        report = xs.run_xs_trend_screen(profile=xs.XS_CONTEXTUAL_ALPHA_PROFILE_ID)

        assert report.profile == "xs_alpha_contextual_v3"
        assert len(captured) == 5
        assert captured[-2].equals(captured[-1])
        assert captured[-1].index.equals(captured[-2].index)

    def test_xsv3_04_report_carries_router_spec_and_diagnostics(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs.run_xs_trend_screen(profile=xs.XS_CONTEXTUAL_ALPHA_PROFILE_ID)
        payload = report.to_payload()
        assert payload["router_spec"]["context_symbol"] == "XS_EQUAL_WEIGHT_MARKET"
        assert payload["router_spec"]["min_context_history_bars"] == 168
        assert set(payload["router_diagnostics"]) == {"windows", "states"}
        windows = payload["router_diagnostics"]["windows"]
        assert set(windows["discovery"]["counts"]) == {
            "trend", "funding_contrarian", "taker_imbalance", "CASH",
        }
        assert set(windows["qualification"]["counts"]) == {
            "trend", "funding_contrarian", "taker_imbalance", "CASH",
        }
        for info in payload["router_diagnostics"]["states"].values():
            assert "completed_samples" in info
            assert set(info["last_lcb"]) == {
                "trend", "funding_contrarian", "taker_imbalance",
            }
        for family in ("trend", "funding_contrarian", "taker_imbalance"):
            assert payload["family_admission"][family]["diagnostic"] is True
        assert len(payload["report_fingerprint"]) == 64

    def test_xsv3_04_holdout_window_in_diagnostics_when_unsealed(self, monkeypatch) -> None:
        idx = pd.date_range("2022-01-01", "2026-07-07 20:00:00", freq="4h", tz="UTC")
        t = np.arange(len(idx), dtype=np.float64)

        def extended_load(symbol: str, start, end):
            salt = float(sum(ord(c) for c in symbol))
            close = 100.0 + 0.02 * t + 30.0 * np.sin(t / 40.0 + salt) + 20.0 * np.cos(t / 150.0)
            frame = pd.DataFrame({
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
                "taker_buy_ratio": 0.5 + 0.03 * np.sin(t / 9.0 + salt),
            }, index=idx)
            funding = pd.Series(0.0, index=idx, dtype=np.float64)
            frame.attrs["symbol"] = symbol
            return frame, funding, {"perp_ohlcv": f"fp-{symbol}"}, 1.0

        monkeypatch.setattr(xs, "_load_symbol_data", extended_load)
        report = xs.run_xs_trend_screen(
            profile=xs.XS_CONTEXTUAL_ALPHA_PROFILE_ID, unseal_holdout=True,
        )
        assert report.holdout is not None
        payload = report.to_payload()
        windows = payload["router_diagnostics"]["windows"]
        assert "holdout" in windows
        assert set(windows["holdout"]["counts"]) == {
            "trend", "funding_contrarian", "taker_imbalance", "CASH",
        }
        assert payload["stress"]["holdout"]["admitted"] is not None

    def test_xsv3_05_bad_router_data_fails_closed_named(self, monkeypatch) -> None:
        def nan_taker_load(symbol: str, start, end):
            frame, funding = synthetic_market()
            frame = frame.copy()
            frame["taker_buy_ratio"] = 0.5
            frame.loc[frame.index[100], "taker_buy_ratio"] = 1.5
            frame.attrs["symbol"] = symbol
            return frame, funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0

        monkeypatch.setattr(xs, "_load_symbol_data", nan_taker_load)
        report = xs.run_xs_trend_screen(profile=xs.XS_CONTEXTUAL_ALPHA_PROFILE_ID)
        assert report.discovery.admitted is False
        assert report.qualification.admitted is False
        assert "contextual_router_invalid" in report.qualification.binding_constraint
        assert report.alpha_spec is not None


class TestScoreRoutedProfileOrchestration:
    def test_xsv4_06_profile_dispatches_score_routed_pipeline(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs.run_xs_trend_screen(profile=xs.XS_SCORE_ROUTED_ALPHA_PROFILE_ID)
        assert report.profile == "xs_alpha_score_routed_v4"
        payload = report.to_payload()
        assert payload["router_spec"]["context_symbol"] == "XS_EQUAL_WEIGHT_MARKET"
        assert payload["router_spec"]["min_context_history_bars"] == 168
        assert set(payload["router_diagnostics"]) == {"windows", "states"}
        windows = payload["router_diagnostics"]["windows"]
        assert set(windows["discovery"]["counts"]) == {
            "trend", "funding_contrarian", "taker_imbalance", "CASH",
        }
        assert set(windows["qualification"]["counts"]) == {
            "trend", "funding_contrarian", "taker_imbalance", "CASH",
        }
        for family in ("trend", "funding_contrarian", "taker_imbalance"):
            assert payload["family_admission"][family]["diagnostic"] is True
        assert payload["stress"]["qualification"]["admitted"] is not None
        assert len(payload["report_fingerprint"]) == 64

    def test_xsv4_06_weights_from_neutral_weights_on_combined_score(
        self, monkeypatch,
    ) -> None:
        _install_synthetic_data(monkeypatch)
        combined: dict[str, pd.DataFrame] = {}
        original_build = xs.build_xs_causal_score_selection

        def spy_build(sleeve_scores, sleeve_returns, decision_context, router_spec):
            allocation = original_build(
                sleeve_scores, sleeve_returns, decision_context, router_spec,
            )
            combined["score"] = allocation.combined_score.copy()
            return allocation

        monkeypatch.setattr(xs, "build_xs_causal_score_selection", spy_build)
        weighted: dict[str, pd.DataFrame] = {}
        original_weights = xs.build_xs_neutral_weights

        def spy_weights(score, halflife, band):
            weighted["score"] = score.copy()
            return original_weights(score, halflife, band)

        monkeypatch.setattr(xs, "build_xs_neutral_weights", spy_weights)
        report = xs.run_xs_trend_screen(profile=xs.XS_SCORE_ROUTED_ALPHA_PROFILE_ID)
        assert report.router_spec is not None
        assert weighted["score"].equals(combined["score"])


class TestDualFamilyProfileOrchestration:
    def test_xsv5_02_profile_dispatches_dual_family_pipeline(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs.run_xs_trend_screen(profile=xs.XS_DUAL_FAMILY_ALPHA_PROFILE_ID)
        assert report.profile == "xs_alpha_dual_family_v5"
        assert report.alpha_spec is not None
        assert report.router_spec is None
        assert report.router_diagnostics is None
        assert report.family_admission is None
        assert report.holdout is None
        payload = report.to_payload()
        assert payload["stress"]["qualification"]["admitted"] is not None
        assert len(payload["report_fingerprint"]) == 64

    def test_xsv5_02_holdout_stays_sealed_by_default(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs.run_xs_trend_screen(profile=xs.XS_DUAL_FAMILY_ALPHA_PROFILE_ID)
        assert report.holdout is None
        assert report.holdout_start is None
        assert report.holdout_end is None

    def test_xsv5_03_unknown_profile_still_fails_closed(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        with pytest.raises(ValueError, match="unknown xs screen profile") as excinfo:
            xs.run_xs_trend_screen(profile="not_a_profile")
        message = str(excinfo.value)
        assert "xs_neutral_composite_v1" in message
        assert "xs_alpha_multihorizon_v2" in message
        assert "xs_alpha_contextual_v3" in message
        assert "xs_alpha_score_routed_v4" in message
        assert "xs_alpha_dual_family_v5" in message
