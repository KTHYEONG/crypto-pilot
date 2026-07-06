from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    XsFactorSpreadDiagnostics,
    _format_xs_spread_diag,
    _l1_xs_spread_diag_enabled,
    compute_xs_factor_spread_diagnostics,
)


def _xs_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["strategy_id"] = df["family"] + ":" + df["variant"]
    return df


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock()
    cfg.expected_cost_bps = 0.0
    cfg.l1_bootstrap_block_bars = 6
    cfg.l1_bootstrap_samples = 200
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _positive_factor_rows(
    n_bars: int,
    family: str,
    variant: str,
    archetype: str = "xs_alpha",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b in range(n_bars):
        rows.append(
            {
                "decision_idx": b,
                "symbol": "A",
                "family": family,
                "variant": variant,
                "archetype": archetype,
                "side": 1,
                "score_z": 1.0,
                "realized_side_adjusted_gross_bps": 40.0,
            }
        )
        rows.append(
            {
                "decision_idx": b,
                "symbol": "B",
                "family": family,
                "variant": variant,
                "archetype": archetype,
                "side": -1,
                "score_z": -1.0,
                "realized_side_adjusted_gross_bps": 35.0,
            }
        )
    return rows


def _null_factor_rows(n_bars: int, family: str, variant: str, archetype: str = "xs_alpha") -> list[dict[str, object]]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for b in range(n_bars):
        noise = float(rng.uniform(-50, 50))
        rows.append(
            {
                "decision_idx": b,
                "symbol": "A",
                "family": family,
                "variant": variant,
                "archetype": archetype,
                "side": 1,
                "score_z": 0.5,
                "realized_side_adjusted_gross_bps": noise,
            }
        )
        rows.append(
            {
                "decision_idx": b,
                "symbol": "B",
                "family": family,
                "variant": variant,
                "archetype": archetype,
                "side": -1,
                "score_z": -0.5,
                "realized_side_adjusted_gross_bps": -noise,
            }
        )
    return rows


class TestXsSpreadDiagEnvGate:
    def test_disabled_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("L1_XS_SPREAD_DIAG", raising=False)
        assert _l1_xs_spread_diag_enabled() is False

    def test_disabled_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_XS_SPREAD_DIAG", "")
        assert _l1_xs_spread_diag_enabled() is False

    def test_disabled_when_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_XS_SPREAD_DIAG", "0")
        assert _l1_xs_spread_diag_enabled() is False

    def test_disabled_when_false_lower(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_XS_SPREAD_DIAG", "false")
        assert _l1_xs_spread_diag_enabled() is False

    def test_disabled_when_false_capital(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_XS_SPREAD_DIAG", "False")
        assert _l1_xs_spread_diag_enabled() is False

    def test_enabled_when_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_XS_SPREAD_DIAG", "1")
        assert _l1_xs_spread_diag_enabled() is True

    def test_enabled_when_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_XS_SPREAD_DIAG", "true")
        assert _l1_xs_spread_diag_enabled() is True


class TestXsSpreadDiagCompute:
    def test_separates_positive_and_null_factor(self) -> None:
        rows = _positive_factor_rows(12, "xs_carry", "xs_carry_96")
        rows += _null_factor_rows(12, "xs_flow", "xs_flow_24")
        frame = _xs_frame(rows)
        cfg = _make_cfg()
        result = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )
        assert result is not None
        pos = result.by_factor.get("xs_carry:xs_carry_96")
        assert pos is not None
        _, _, pos_mean, _, pos_sharpe, pos_lcb, _, _, _, _ = pos
        assert pos_mean > 0
        assert pos_sharpe > 1.0
        assert pos_lcb > 0
        null = result.by_factor.get("xs_flow:xs_flow_24")
        assert null is not None
        _, _, null_mean, _, null_sharpe, _, _, _, _, _ = null
        assert null_mean == pytest.approx(0, abs=5)
        assert null_sharpe == pytest.approx(0, abs=0.5)

    def test_excludes_short_history_factor(self) -> None:
        rows = _positive_factor_rows(12, "xs_carry", "xs_carry_96")
        rows += _positive_factor_rows(5, "xs_flow", "xs_flow_24")
        frame = _xs_frame(rows)
        cfg = _make_cfg(min_bars=8)
        result = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )
        assert result is not None
        assert "xs_carry:xs_carry_96" in result.by_factor
        assert "xs_flow:xs_flow_24" not in result.by_factor

    def test_returns_none_when_no_xs_events(self) -> None:
        rows = _positive_factor_rows(12, "trend", "trend_ma", archetype="trend")
        rows += _null_factor_rows(12, "trend", "trend_revert", archetype="trend")
        frame = _xs_frame(rows)
        cfg = _make_cfg()
        result = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )
        assert result is None

    def test_family_fallback_when_archetype_missing(self) -> None:
        rows: list[dict[str, object]] = []
        for b in range(12):
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "A",
                    "family": "xs_momentum",
                    "variant": "xs_momentum_48",
                    "side": 1,
                    "score_z": 1.0,
                    "realized_side_adjusted_gross_bps": 40.0,
                }
            )
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "B",
                    "family": "xs_momentum",
                    "variant": "xs_momentum_48",
                    "side": -1,
                    "score_z": -1.0,
                    "realized_side_adjusted_gross_bps": 35.0,
                }
            )
        frame = _xs_frame(rows)
        if "archetype" in frame.columns:
            frame = frame.drop(columns=["archetype"])
        cfg = _make_cfg()
        result = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )
        assert result is not None
        assert len(result.by_factor) > 0

    def test_rank_ic_positive_when_score_predicts_realized(self) -> None:
        rows: list[dict[str, object]] = []
        for b in range(12):
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "A",
                    "family": "xs_carry",
                    "variant": "xs_carry_96",
                    "archetype": "xs_alpha",
                    "side": 1,
                    "score_z": 2.0,
                    "realized_side_adjusted_gross_bps": 50.0,
                }
            )
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "B",
                    "family": "xs_carry",
                    "variant": "xs_carry_96",
                    "archetype": "xs_alpha",
                    "side": -1,
                    "score_z": -2.0,
                    "realized_side_adjusted_gross_bps": 30.0,
                }
            )
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "C",
                    "family": "xs_carry",
                    "variant": "xs_carry_96",
                    "archetype": "xs_alpha",
                    "side": 1,
                    "score_z": -0.5,
                    "realized_side_adjusted_gross_bps": 10.0,
                }
            )
        frame = _xs_frame(rows)
        cfg = _make_cfg()
        result = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )
        assert result is not None
        _, _, _, _, _, _, ic, ict, _, _ = result.by_factor["xs_carry:xs_carry_96"]
        assert ic > 0
        assert ict > 0

    def test_constant_series_no_zero_division(self) -> None:
        rows: list[dict[str, object]] = []
        for b in range(12):
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "A",
                    "family": "xs_carry",
                    "variant": "xs_carry_96",
                    "archetype": "xs_alpha",
                    "side": 1,
                    "score_z": 1.0,
                    "realized_side_adjusted_gross_bps": 30.0,
                }
            )
            rows.append(
                {
                    "decision_idx": b,
                    "symbol": "B",
                    "family": "xs_carry",
                    "variant": "xs_carry_96",
                    "archetype": "xs_alpha",
                    "side": -1,
                    "score_z": -1.0,
                    "realized_side_adjusted_gross_bps": 30.0,
                }
            )
        frame = _xs_frame(rows)
        cfg = _make_cfg()
        result = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )
        assert result is not None
        _, _, _, std, sharpe, _, _, _, _, _ = result.by_factor["xs_carry:xs_carry_96"]
        assert std == 0.0
        assert np.isfinite(sharpe)

    def test_lcb_deterministic_under_seed(self) -> None:
        rows = _positive_factor_rows(12, "xs_carry", "xs_carry_96")
        frame = _xs_frame(rows)
        cfg = _make_cfg()
        r1 = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=123,
        )
        r2 = compute_xs_factor_spread_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=123,
        )
        assert r1 is not None
        assert r2 is not None
        lcb1 = r1.by_factor["xs_carry:xs_carry_96"][5]
        lcb2 = r2.by_factor["xs_carry:xs_carry_96"][5]
        assert lcb1 == lcb2


class TestFormatXsSpreadDiag:
    def test_tokens_present(self) -> None:
        diag = XsFactorSpreadDiagnostics(
            fold_id=0,
            by_factor={
                "xs_carry:xs_carry_96": (12, 24, 37.5, 2.0, 18.75, 35.0, 0.5, 2.1, 0.5, 0.95),
            },
        )
        out = _format_xs_spread_diag(diag)
        assert "XS[xs_carry:xs_carry_96]=" in out
        assert "sh" in out
        assert "lcb" in out
        assert "ic" in out
        assert "n" in out
