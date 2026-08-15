from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    FamilyRegimeDiagnostics,
    _format_family_regime_diag,
    _l1_family_regime_diag_enabled,
    compute_family_regime_edge_diagnostics,
)


def _family_regime_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock()
    cfg.l1_bootstrap_block_bars = 6
    cfg.l1_bootstrap_samples = 200
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _family_regime_rows(
    n_bars: int,
    family: str,
    regime_code: int,
    gross_bps: float,
) -> list[dict[str, object]]:
    rows = [
        {
            "decision_idx": b,
            "symbol": "A",
            "family": family,
            "entry_regime_code": regime_code,
            "side": 1,
            "score_z": 1.0,
            "realized_side_adjusted_gross_bps": gross_bps,
        }
        for b in range(n_bars)
    ]
    return rows


class TestFamilyRegimeDiagEnvGate:
    def test_disabled_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("L1_FAMILY_REGIME_DIAG", raising=False)
        assert _l1_family_regime_diag_enabled() is False

    def test_disabled_when_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_FAMILY_REGIME_DIAG", "0")
        assert _l1_family_regime_diag_enabled() is False

    def test_enabled_when_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L1_FAMILY_REGIME_DIAG", "1")
        assert _l1_family_regime_diag_enabled() is True


class TestFamilyRegimeDiagCompute:
    def test_compute_family_regime_edge_diagnostics_success(self) -> None:
        rows = _family_regime_rows(30, "residual_reversion", 0, 50.0)
        rows += _family_regime_rows(30, "trend_donchian", 1, 30.0)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )

        assert result is not None
        cell_a = result.by_family_regime[("residual_reversion", 0)]
        n_bars_a, n_events_a, mean_a, _std_a, _sharpe_a, _lcb_a, _ic_a = cell_a
        assert n_bars_a == 30
        assert n_events_a == 30
        assert mean_a == pytest.approx(50.0)
        cell_b = result.by_family_regime[("trend_donchian", 1)]
        _n_bars_b, _n_events_b, mean_b, *_ = cell_b
        assert mean_b == pytest.approx(30.0)

    def test_excludes_cell_below_min_bars(self) -> None:
        rows = _family_regime_rows(30, "residual_reversion", 0, 50.0)
        rows += _family_regime_rows(5, "dual_momentum", 2, 40.0)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
            min_bars=8,
        )

        assert result is not None
        assert ("residual_reversion", 0) in result.by_family_regime
        assert ("dual_momentum", 2) not in result.by_family_regime

    def test_returns_none_when_realized_event_results_empty(self) -> None:
        frame = pd.DataFrame()
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )

        assert result is None

    def test_returns_none_when_entry_regime_code_column_missing(self) -> None:
        rows = _family_regime_rows(30, "residual_reversion", 0, 50.0)
        frame = _family_regime_frame(rows).drop(columns=["entry_regime_code"])
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )

        assert result is None

    def test_lcb_deterministic_under_seed(self) -> None:
        rows = _family_regime_rows(30, "residual_reversion", 0, 50.0)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        r1 = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=123,
        )
        r2 = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=123,
        )

        assert r1 is not None
        assert r2 is not None
        lcb1 = r1.by_family_regime[("residual_reversion", 0)][5]
        lcb2 = r2.by_family_regime[("residual_reversion", 0)][5]
        assert lcb1 == lcb2

    def test_constant_series_no_zero_division(self) -> None:
        rows = _family_regime_rows(12, "residual_reversion", 0, 30.0)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )

        assert result is not None
        _n_bars, _n_events, _mean, std, sharpe, _lcb, _ic = result.by_family_regime[("residual_reversion", 0)]
        assert std == 0.0
        assert np.isfinite(sharpe)


class TestFormatFamilyRegimeDiag:
    def test_tokens_present(self) -> None:
        diag = FamilyRegimeDiagnostics(
            fold_id=0,
            by_family_regime={
                ("residual_reversion", 1): (30, 30, 50.0, 2.0, 25.0, 45.0, 0.3),
            },
        )

        out = _format_family_regime_diag(diag)

        assert "residual_reversion@R1=" in out
        assert "gross" in out
        assert "sh" in out
        assert "lcb" in out
        assert "ic" in out


def _family_regime_rows_with_side(
    n_bars: int,
    family: str,
    regime_code: int,
    gross_bps: float,
    side: int,
) -> list[dict[str, object]]:
    rows = [
        {
            "decision_idx": b,
            "symbol": "A",
            "family": family,
            "entry_regime_code": regime_code,
            "side": side,
            "score_z": 1.0,
            "realized_side_adjusted_gross_bps": gross_bps,
        }
        for b in range(n_bars)
    ]
    return rows


class TestFamilyRegimeDiagSplitSide:
    def test_split_side_adds_side_keyed_dict(self) -> None:
        rows = _family_regime_rows_with_side(30, "trend_ma", 0, 50.0, 1)
        rows += _family_regime_rows_with_side(30, "trend_ma", 0, -20.0, -1)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
            split_side=True,
        )

        assert result is not None
        assert result.by_family_regime_side is not None
        long_cell = result.by_family_regime_side[("trend_ma", 0, "long")]
        short_cell = result.by_family_regime_side[("trend_ma", 0, "short")]
        assert long_cell[2] == pytest.approx(50.0)
        assert short_cell[2] == pytest.approx(-20.0)
        # Backward-compat: unsplit dict unaffected
        assert result.by_family_regime[("trend_ma", 0)][2] == pytest.approx(15.0)

    def test_default_split_side_none(self) -> None:
        rows = _family_regime_rows(30, "residual_reversion", 0, 50.0)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
        )

        assert result is not None
        assert result.by_family_regime_side is None

    def test_split_side_missing_column_graceful(self) -> None:
        rows = _family_regime_rows(30, "residual_reversion", 0, 50.0)
        frame = _family_regime_frame(rows).drop(columns=["side"])
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
            split_side=True,
        )

        assert result is not None
        assert result.by_family_regime_side is None

    def test_split_side_zero_side_excluded(self) -> None:
        rows = _family_regime_rows_with_side(30, "trend_ma", 0, 50.0, 1)
        rows += _family_regime_rows_with_side(30, "trend_ma", 0, 0.0, 0)
        frame = _family_regime_frame(rows)
        cfg = _make_cfg()

        result = compute_family_regime_edge_diagnostics(
            realized_event_results=frame,
            cfg=cfg,
            fold_id=0,
            seed=0,
            split_side=True,
        )

        assert result is not None
        assert result.by_family_regime_side is not None
        assert ("trend_ma", 0, "long") in result.by_family_regime_side
        assert ("trend_ma", 0, "short") not in result.by_family_regime_side
        assert result.by_family_regime_side[("trend_ma", 0, "long")][0] == 30
