"""Contract scenario XABB-05 for the XS alpha x baseline blend orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import xs_alpha_baseline_blend as xs_blend
from src.application.research.technical.xs_trend_screen import (
    XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
)
from src.common.errors import DataIntegrityError
from src.research.technical_experts.cross_sectional import (
    XsAdmissionResult,
    XsReliabilityResult,
)


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


def _install_synthetic_data(monkeypatch) -> None:
    """Stub the data loader for the blend orchestrator with a deterministic market."""
    def fake_load(symbol: str, start, end) -> tuple[pd.DataFrame, pd.Series, dict[str, str], float]:
        # The orchestrator passes the sealed cutoff when holdout is sealed and
        # None when it is unsealed; slice the synthetic market accordingly.
        data_end = "2026-07-07 20:00:00" if end is None else str(pd.Timestamp(end))
        frame, funding = _synthetic_market(end=data_end)
        frame = frame.copy()
        salt = float(sum(ord(c) for c in symbol))
        frame["close"] = frame["close"] * (1.0 + 0.02 * (salt % 7))
        frame["open"] = frame["open"] * (1.0 + 0.02 * (salt % 7))
        frame["taker_buy_ratio"] = 0.5 + 0.03 * np.sin(np.arange(len(frame)) / 9.0 + salt)
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
