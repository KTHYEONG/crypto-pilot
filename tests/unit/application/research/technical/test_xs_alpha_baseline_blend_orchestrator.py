"""Contract scenario XABB-05 for the XS alpha x baseline blend orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import xs_alpha_baseline_blend as xs_blend
from src.application.research.technical.xs_trend_screen import (
    XS_DISCOVERY_START,
    XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
)
from src.common.errors import DataIntegrityError
from src.research.risk.growth_sizing import GrowthSizingResult
from src.research.technical_experts.cross_sectional import (
    XsAdmissionResult,
    XsReliabilityResult,
)
from src.research.technical_experts.trend_screen_catalog import DISCOVERY_END


def _synthetic_market(start: str = "2022-01-01", end: str = "2025-12-31 23:59:59"):
    idx = pd.date_range(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="4h",
    )
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


def _install_synthetic_data(monkeypatch, *, perturb_after: pd.Timestamp | None = None) -> None:
    """Stub the data loader for the blend orchestrator with a deterministic market.

    When ``perturb_after`` is set, every bar at or after that timestamp gets a
    time-varying price drift and a flat high taker ratio -- a deterministic,
    strongly different qualification-window market (discovery stays identical).
    """
    def fake_load(symbol: str, start, end) -> tuple[pd.DataFrame, pd.Series, dict[str, str], float]:
        # The orchestrator passes the sealed cutoff when holdout is sealed and
        # None when it is unsealed; slice the synthetic market accordingly.
        data_end = "2026-07-07 20:00:00" if end is None else str(pd.Timestamp(end))
        frame, funding = _synthetic_market(end=data_end)
        frame = frame.copy()
        salt = float(sum(ord(c) for c in symbol))
        frame["taker_buy_ratio"] = 0.5 + 0.03 * np.sin(np.arange(len(frame)) / 9.0 + salt)
        if perturb_after is not None:
            late = frame.index >= perturb_after
            n = int(late.sum())
            frame.loc[late, ["open", "high", "low", "close"]] *= (
                1.0 + 0.01 * np.arange(n)
            )[:, None]
            frame.loc[late, "taker_buy_ratio"] = 0.95
        frame["close"] = frame["close"] * (1.0 + 0.02 * (salt % 7))
        frame["open"] = frame["open"] * (1.0 + 0.02 * (salt % 7))
        frame.attrs["symbol"] = symbol
        return frame, funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0

    monkeypatch.setattr(xs_blend, "_load_symbol_data", fake_load)


class TestBaselineBlendOrchestration:
    def test_xabb_05_orchestrator_loads_blends_and_verifies_gates(self, monkeypatch) -> None:
        # XABB-05: v6 + frozen baseline are loaded, the blend weight is selected
        # from discovery, admission is re-verified on the blended ledger, and
        # reliability is evaluated on the stitched OOS window.
        _install_synthetic_data(monkeypatch)
        report = xs_blend.run_xs_alpha_baseline_blend()

        assert report.profile == XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
        assert 0.0 <= report.blend_weight <= 1.0
        assert isinstance(report.discovery, XsAdmissionResult)
        assert isinstance(report.qualification, XsAdmissionResult)
        assert isinstance(report.pre_blend_discovery, XsAdmissionResult)
        assert isinstance(report.pre_blend_qualification, XsAdmissionResult)
        assert isinstance(report.baseline_discovery, XsAdmissionResult)
        assert isinstance(report.baseline_qualification, XsAdmissionResult)
        assert isinstance(report.reliability, XsReliabilityResult)
        assert len(report.report_fingerprint) == 64

    def test_xabb_05_holdout_stays_sealed_by_default(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs_blend.run_xs_alpha_baseline_blend()
        assert report.holdout is None
        assert report.pre_blend_holdout is None
        assert report.baseline_holdout is None

    def test_xabb_05_holdout_replayed_only_when_unsealed(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs_blend.run_xs_alpha_baseline_blend(unseal_holdout=True)
        assert report.holdout is not None
        assert report.pre_blend_holdout is not None
        assert report.baseline_holdout is not None

    def test_xabb_05_rejects_non_v6_profile(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        with pytest.raises(ValueError, match="restricted"):
            xs_blend.run_xs_alpha_baseline_blend(profile="xs_alpha_positioning_v7")

    def test_xabb_05_fails_closed_on_short_common_grid(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        short_grid = pd.DatetimeIndex(
            [pd.Timestamp("2022-01-01", tz="UTC")],
        )
        monkeypatch.setattr(xs_blend, "_common_index", lambda indexes: short_grid)
        with pytest.raises(DataIntegrityError, match="at least 2 common bars"):
            xs_blend.run_xs_alpha_baseline_blend()

    def test_xabb_05_payload_is_byte_deterministic(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        first = xs_blend.run_xs_alpha_baseline_blend()
        second = xs_blend.run_xs_alpha_baseline_blend()
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert set(payload) == {
            "profile", "blend_weight", "weight_grid",
            "discovery", "qualification", "holdout",
            "pre_blend_discovery", "pre_blend_qualification", "pre_blend_holdout",
            "baseline_discovery", "baseline_qualification", "baseline_holdout",
            "reliability", "report_fingerprint",
        }
        assert payload["weight_grid"] == list(xs_blend._DEFAULT_WEIGHT_GRID)

    def test_xabb_05_persistence_is_byte_deterministic(self, monkeypatch, tmp_path) -> None:
        _install_synthetic_data(monkeypatch)
        report = xs_blend.run_xs_alpha_baseline_blend()
        path = tmp_path / "blend.json"
        xs_blend.persist_xs_alpha_baseline_blend_report(report, path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == report.to_json()
        assert xs_blend.xs_baseline_blend_report_path().name == "xs_alpha_baseline_blend_v8.json"

def _feasible_sizing(monkeypatch, selected_risk: float = 3.0) -> None:
    """Make the discovery-only leverage selection feasible (bootstrap math covered elsewhere)."""
    def fake_solve(unit_returns, config, *, use_drawdown_overlay: bool) -> GrowthSizingResult:
        return GrowthSizingResult(
            selected_risk=selected_risk, median_log_growth=0.5,
            mdd_breach_prob=0.01, ruin_prob=0.0, feasible_risks=(selected_risk,),
            binding_constraint="none", block_size_used=10,
        )

    monkeypatch.setattr(xs_blend, "solve_growth_optimal_risk", fake_solve)


class TestBaselineBlendSizedOrchestration:
    def test_xabrs_sized_entry_point_signature(self) -> None:
        # Contract python_assertion: keyword-only surface with frozen risk grid.
        from inspect import signature

        params = signature(xs_blend.run_xs_alpha_baseline_blend_sized).parameters
        assert set(params) == {"unseal_holdout", "weight_grid", "risk_grid"}
        assert all(p.kind == p.KEYWORD_ONLY for p in params.values())
        assert params["risk_grid"].default == (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

    def test_xabrs_sized_orchestrator_runs_and_carries_sizing(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=3.0)
        report = xs_blend.run_xs_alpha_baseline_blend_sized()

        assert report.profile == XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
        assert isinstance(report.sizing, GrowthSizingResult)
        assert report.sizing.selected_risk == 3.0
        assert isinstance(report.discovery, XsAdmissionResult)
        assert isinstance(report.qualification, XsAdmissionResult)
        assert isinstance(report.reliability, XsReliabilityResult)
        assert len(report.report_fingerprint) == 64

    def test_xabrs_04_leverage_selected_from_discovery_only(self, monkeypatch) -> None:
        # XABRS-04: solve_growth_optimal_risk receives ONLY discovery-window
        # blended net returns; appending different qualification-window data
        # must not change the selected risk.
        expected_disc_bars = len(pd.date_range(
            XS_DISCOVERY_START, DISCOVERY_END, freq="4h", tz="UTC",
        ))
        captured: dict[str, np.ndarray] = {}

        def recording_solve(unit_returns, config, *, use_drawdown_overlay: bool) -> GrowthSizingResult:
            captured["input"] = np.asarray(unit_returns, dtype=np.float64)
            return GrowthSizingResult(
                selected_risk=3.0, median_log_growth=0.5, mdd_breach_prob=0.01,
                ruin_prob=0.0, feasible_risks=(3.0,), binding_constraint="none",
                block_size_used=10,
            )

        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(xs_blend, "solve_growth_optimal_risk", recording_solve)
        first = xs_blend.run_xs_alpha_baseline_blend_sized()
        first_input = captured["input"]

        captured.clear()
        _install_synthetic_data(
            monkeypatch, perturb_after=DISCOVERY_END + pd.Timedelta(hours=4),
        )
        second = xs_blend.run_xs_alpha_baseline_blend_sized()
        second_input = captured["input"]

        assert first_input.size == expected_disc_bars
        assert second_input.size == expected_disc_bars
        assert np.array_equal(first_input, second_input)
        assert first.sizing.selected_risk == second.sizing.selected_risk

    def test_xabrs_05_infeasible_fails_closed_to_unit_scale(self, monkeypatch) -> None:
        # XABRS-05: selected_risk=None fails closed to the unscaled robust
        # blend (byte-identical gates to a feasible scale=1.0 run), reports
        # binding_constraint='infeasible', and never raises or picks an
        # arbitrary leverage.
        def infeasible_solve(unit_returns, config, *, use_drawdown_overlay: bool) -> GrowthSizingResult:
            return GrowthSizingResult(
                None, 0.0, 0.0, 0.0, (), "infeasible", 10,
            )

        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(xs_blend, "solve_growth_optimal_risk", infeasible_solve)
        report_infeasible = xs_blend.run_xs_alpha_baseline_blend_sized()
        assert report_infeasible.sizing.selected_risk is None
        assert report_infeasible.sizing.binding_constraint == "infeasible"

        _install_synthetic_data(monkeypatch)

        def unit_solve(unit_returns, config, *, use_drawdown_overlay: bool) -> GrowthSizingResult:
            return GrowthSizingResult(
                selected_risk=1.0, median_log_growth=0.5, mdd_breach_prob=0.01,
                ruin_prob=0.0, feasible_risks=(1.0,), binding_constraint="none",
                block_size_used=10,
            )

        monkeypatch.setattr(xs_blend, "solve_growth_optimal_risk", unit_solve)
        report_unit = xs_blend.run_xs_alpha_baseline_blend_sized()
        assert report_unit.sizing.selected_risk == 1.0
        assert report_infeasible.discovery == report_unit.discovery
        assert report_infeasible.qualification == report_unit.qualification
        assert report_infeasible.reliability == report_unit.reliability

    def test_xabrs_sized_holdout_stays_sealed_by_default(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch)
        report = xs_blend.run_xs_alpha_baseline_blend_sized()
        assert report.holdout is None
        assert report.pre_blend_holdout is None
        assert report.baseline_holdout is None

    def test_xabrs_sized_payload_is_byte_deterministic(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=2.5)
        first = xs_blend.run_xs_alpha_baseline_blend_sized()
        second = xs_blend.run_xs_alpha_baseline_blend_sized()
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert set(payload) == {
            "profile", "blend_weight", "weight_grid", "sizing",
            "discovery", "qualification", "holdout",
            "pre_blend_discovery", "pre_blend_qualification", "pre_blend_holdout",
            "baseline_discovery", "baseline_qualification", "baseline_holdout",
            "reliability", "report_fingerprint",
        }
        assert payload["weight_grid"] == list(xs_blend._DEFAULT_WEIGHT_GRID)
        assert payload["sizing"]["selected_risk"] == 2.5

    def test_xabrs_sized_persistence_is_byte_deterministic(self, monkeypatch, tmp_path) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=2.0)
        report = xs_blend.run_xs_alpha_baseline_blend_sized()
        path = tmp_path / "sized.json"
        xs_blend.persist_xs_alpha_baseline_blend_sized_report(report, path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == report.to_json()
        assert (
            xs_blend.xs_baseline_blend_sized_report_path().name
            == "xs_alpha_baseline_blend_v8_sized.json"
        )

    def test_xabrs_sized_holdout_replayed_only_when_unsealed(self, monkeypatch) -> None:
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

        monkeypatch.setattr(xs_blend, "_load_symbol_data", extended_load)
        _feasible_sizing(monkeypatch, selected_risk=2.0)
        report = xs_blend.run_xs_alpha_baseline_blend_sized(unseal_holdout=True)
        assert report.holdout is not None
        assert report.pre_blend_holdout is not None
        assert report.baseline_holdout is not None

    def test_xabrs_sized_fails_closed_on_short_common_grid(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        short_grid = pd.DatetimeIndex(
            [pd.Timestamp("2022-01-01", tz="UTC")],
        )
        monkeypatch.setattr(xs_blend, "_common_index", lambda indexes: short_grid)
        with pytest.raises(DataIntegrityError, match="at least 2 common bars"):
            xs_blend.run_xs_alpha_baseline_blend_sized()
