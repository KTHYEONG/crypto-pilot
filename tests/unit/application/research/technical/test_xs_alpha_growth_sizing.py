"""Contract scenarios SCENARIO_XSV6SIZE_03, SCENARIO_XSV6SIZE_04, and
SCENARIO_XSV6SIZE_05 for the growth-optimal sizing overlay orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import xs_alpha_growth_sizing as xs_growth
from src.application.research.technical.xs_trend_screen import (
    XS_ALPHA_PROFILE_ID,
    XS_CONTEXTUAL_ALPHA_PROFILE_ID,
    XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
)
from src.research.risk.growth_sizing import GrowthSizingResult
from src.research.technical_experts import cross_sectional as cs


def _synthetic_market(start: str = "2022-01-01", end: str = "2025-12-31 23:59:59"):
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
    """Stub the data loader for the growth-sizing module with a deterministic market."""
    def fake_load(symbol: str, start, end) -> tuple[pd.DataFrame, pd.Series, dict[str, str], float]:
        frame, funding = _synthetic_market()
        frame = frame.copy()
        salt = float(sum(ord(c) for c in symbol))
        frame["close"] = frame["close"] * (1.0 + 0.02 * (salt % 7))
        frame["open"] = frame["open"] * (1.0 + 0.02 * (salt % 7))
        frame["taker_buy_ratio"] = 0.5 + 0.03 * np.sin(np.arange(len(frame)) / 9.0 + salt)
        frame.attrs["symbol"] = symbol
        return frame, funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0

    monkeypatch.setattr(xs_growth, "_load_symbol_data", fake_load)


def _feasible_sizing(monkeypatch, selected_risk: float = 2.0) -> None:
    """Make the sizing bootstrap feasible quickly (the bootstrap math is covered elsewhere)."""
    def fake_solve(unit_returns, config) -> GrowthSizingResult:
        return GrowthSizingResult(
            selected_risk=selected_risk, median_log_growth=0.5,
            mdd_breach_prob=0.01, ruin_prob=0.0, feasible_risks=(selected_risk,),
            binding_constraint="none", block_size_used=10,
        )

    monkeypatch.setattr(cs, "solve_growth_optimal_risk", fake_solve)


class TestGrowthSizingOrchestration:
    def test_xsv6size_03_orchestrator_rejects_unadmitted_profile(self, monkeypatch) -> None:
        # SCENARIO_XSV6SIZE_03_ORCHESTRATOR_REJECTS_UNADMITTED_PROFILE
        _install_synthetic_data(monkeypatch)
        with pytest.raises(ValueError, match="admitted end-to-end"):
            xs_growth.run_xs_alpha_growth_sizing(profile=XS_CONTEXTUAL_ALPHA_PROFILE_ID)

    def test_xsv6size_04_report_carries_pre_and_post_scaling_gates(self, monkeypatch) -> None:
        # SCENARIO_XSV6SIZE_04_REPORT_CARRIES_PRE_AND_POST_SCALING_GATES
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=2.0)
        report = xs_growth.run_xs_alpha_growth_sizing(profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID)

        assert report.profile == XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
        assert isinstance(report.sizing, GrowthSizingResult)
        assert report.sizing.selected_risk == 2.0
        assert report.pre_scaling_discovery is not None
        assert report.pre_scaling_qualification is not None
        assert report.discovery is not None
        assert report.qualification is not None
        assert report.pre_scaling_discovery is not report.discovery
        assert report.pre_scaling_qualification is not report.qualification
        assert report.discovery != report.pre_scaling_discovery
        assert report.qualification != report.pre_scaling_qualification
        assert len(report.report_fingerprint) == 64

    def test_xsv6size_05_holdout_stays_sealed_by_default(self, monkeypatch) -> None:
        # SCENARIO_XSV6SIZE_05_HOLDOUT_STAYS_SEALED_BY_DEFAULT
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=1.0)
        report = xs_growth.run_xs_alpha_growth_sizing(profile=XS_ALPHA_PROFILE_ID)
        assert report.holdout is None
        assert report.pre_scaling_holdout is None

    def test_xsv6size_infeasible_sizing_fails_closed_at_orchestration(self, monkeypatch) -> None:
        # The overlay never deploys: post-scaling admission is re-verified on the
        # unchanged base book, so the gates match the pre-scaling evidence.
        _install_synthetic_data(monkeypatch)

        def infeasible_solve(unit_returns, config) -> GrowthSizingResult:
            return GrowthSizingResult(
                selected_risk=None, median_log_growth=0.0,
                mdd_breach_prob=0.0, ruin_prob=0.0, feasible_risks=(),
                binding_constraint="infeasible", block_size_used=10,
            )

        monkeypatch.setattr(cs, "solve_growth_optimal_risk", infeasible_solve)
        report = xs_growth.run_xs_alpha_growth_sizing(profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID)
        assert report.sizing.selected_risk is None
        assert report.discovery == report.pre_scaling_discovery
        assert report.qualification == report.pre_scaling_qualification
        assert report.holdout is None

    def test_xsv6size_payload_is_byte_deterministic(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=1.0)
        first = xs_growth.run_xs_alpha_growth_sizing(profile=XS_ALPHA_PROFILE_ID)
        second = xs_growth.run_xs_alpha_growth_sizing(profile=XS_ALPHA_PROFILE_ID)
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert set(payload) == {
            "profile", "sizing", "discovery", "qualification", "holdout",
            "pre_scaling_discovery", "pre_scaling_qualification",
            "pre_scaling_holdout", "vol_target_window", "vol_target",
            "report_fingerprint",
        }
        assert payload["sizing"]["selected_risk"] == 1.0

    def test_xsv6size_holdout_replayed_only_when_unsealed(self, monkeypatch) -> None:
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

        monkeypatch.setattr(xs_growth, "_load_symbol_data", extended_load)
        _feasible_sizing(monkeypatch, selected_risk=1.0)
        report = xs_growth.run_xs_alpha_growth_sizing(
            profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID, unseal_holdout=True,
        )
        assert report.holdout is not None
        assert report.pre_scaling_holdout is not None
        assert report.holdout is not report.pre_scaling_holdout

    def test_xsv6size_persistence_is_byte_deterministic(self, monkeypatch, tmp_path) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=1.0)
        report = xs_growth.run_xs_alpha_growth_sizing(profile=XS_ALPHA_PROFILE_ID)
        path = tmp_path / "growth_sized.json"
        xs_growth.persist_xs_growth_sizing_report(report, path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == report.to_json()
        assert (
            xs_growth.xs_growth_sizing_report_path(XS_VOL_WEIGHTED_ALPHA_PROFILE_ID).name
            == "xs_alpha_vol_weighted_v6_growth_sized.json"
        )

    # SCENARIO_VOLTARGET_08_ORCHESTRATOR_DEFAULT_ENABLED_42
    def test_voltarget_08_orchestrator_defaults_to_42(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=1.0)
        report = xs_growth.run_xs_alpha_growth_sizing(profile=XS_ALPHA_PROFILE_ID)
        assert report.vol_target_window == 42
        assert report.vol_target is not None
        assert np.isfinite(report.vol_target)

    # SCENARIO_VOLTARGET_09_ORCHESTRATOR_OPT_OUT_REPRODUCES_ORIGINAL
    def test_voltarget_09_opt_out_reproduces_original(self, monkeypatch) -> None:
        _install_synthetic_data(monkeypatch)
        _feasible_sizing(monkeypatch, selected_risk=2.0)
        report = xs_growth.run_xs_alpha_growth_sizing(
            profile=XS_VOL_WEIGHTED_ALPHA_PROFILE_ID, vol_target_window=None,
        )
        assert report.vol_target_window is None
        assert report.vol_target is None
        assert report.sizing.selected_risk == 2.0
