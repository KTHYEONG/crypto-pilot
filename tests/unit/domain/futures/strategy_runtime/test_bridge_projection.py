from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid


def _make_panel(
    score_2d: np.ndarray,
    valid_2d: np.ndarray,
    side_2d: np.ndarray,
    to_2d: np.ndarray,
    datetimes: np.ndarray,
    symbols: tuple[str, ...] = ("SYM0",),
    family: str = "test",
    variant: str = "test_v1",
    holding_bars: int = 3,
) -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={},
        datetimes=datetimes,
        symbols=symbols,
        signed_score_2d=score_2d,
        side_hint_2d=side_2d,
        expected_holding_bars=holding_bars,
        min_holding_bars=1,
        stop_atr_mult=50.0,
        take_profit_atr_mult=50.0,
        turnover_proxy_2d=to_2d,
        valid_mask_2d=valid_2d,
        metadata={},
        archetype="trend",
        allowed_regimes=("bull_quiet", "bear_quiet", "transition"),
        exit_policies=(),
    )


def test_ltf_mode_last_backward_compat() -> None:
    """Scenario 1: ltf_mode='last' matches prior searchsorted behavior."""
    panel_dt = np.array(
        ["2026-01-01T00", "2026-01-01T01", "2026-01-01T02", "2026-01-01T03"],
        dtype="datetime64[ns]",
    )
    base_dt = np.array(["2026-01-01T04"], dtype="datetime64[ns]")
    score = np.array([[0.5], [0.3], [-0.2], [0.8]], dtype=np.float64)

    panel = _make_panel(
        score_2d=score,
        valid_2d=np.ones((4, 1), dtype=bool),
        side_2d=np.array([[1], [1], [-1], [1]], dtype=np.int8),
        to_2d=np.full((4, 1), 0.1, dtype=np.float64),
        datetimes=panel_dt,
    )

    result = _project_panel_to_base_grid(panel, base_dt, "1h", "4h", ltf_mode="last")

    assert result.signed_score_2d[0, 0] == 0.8
    assert result.valid_mask_2d[0, 0]
    assert result.side_hint_2d[0, 0] == 1
    assert result.turnover_proxy_2d[0, 0] == pytest.approx(0.1, abs=1e-6)


def test_ltf_mode_mean_aggregates_all_bars() -> None:
    """Scenario 2: ltf_mode='mean' uses mean(score), any(valid), mode(side), mean(to).

    Window semantics match label='right', closed='right': bar at boundary
    (00:00) belongs to previous window; bars (01:00, 02:00, 03:00]
    are aggregated.
    """
    panel_dt = np.array(
        ["2026-01-01T00", "2026-01-01T01", "2026-01-01T02", "2026-01-01T03"],
        dtype="datetime64[ns]",
    )
    base_dt = np.array(["2026-01-01T04"], dtype="datetime64[ns]")
    score = np.array([[0.5], [0.3], [-0.2], [0.8]], dtype=np.float64)

    panel = _make_panel(
        score_2d=score,
        valid_2d=np.ones((4, 1), dtype=bool),
        side_2d=np.array([[1], [1], [-1], [1]], dtype=np.int8),
        to_2d=np.full((4, 1), 0.1, dtype=np.float64),
        datetimes=panel_dt,
    )

    result = _project_panel_to_base_grid(panel, base_dt, "1h", "4h", ltf_mode="mean")

    # Bars at indices [1,2,3]: scores [0.3, -0.2, 0.8] → mean 0.30
    assert result.signed_score_2d[0, 0] == pytest.approx(0.30, abs=1e-6)
    assert result.valid_mask_2d[0, 0]
    # side_hint mode of [1, -1, 1] = 1
    assert result.side_hint_2d[0, 0] == 1
    assert result.turnover_proxy_2d[0, 0] == pytest.approx(0.1, abs=1e-6)


def test_ltf_mode_mean_multiple_windows() -> None:
    """Scenario 3: each base window gets its own correct aggregation."""
    hours = [f"2026-01-01T{h:02d}" for h in [0, 2, 4, 6, 8, 10, 12, 14]]
    panel_dt = np.array(hours, dtype="datetime64[ns]")
    base_dt = np.array(
        ["2026-01-01T04", "2026-01-01T08", "2026-01-01T12", "2026-01-01T16"],
        dtype="datetime64[ns]",
    )
    score = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]], dtype=np.float64)

    panel = _make_panel(
        score_2d=score,
        valid_2d=np.ones((8, 1), dtype=bool),
        side_2d=np.full((8, 1), 1, dtype=np.int8),
        to_2d=np.full((8, 1), 0.1, dtype=np.float64),
        datetimes=panel_dt,
    )

    result = _project_panel_to_base_grid(panel, base_dt, "2h", "4h", ltf_mode="mean")

    # Window boundaries: (prev_boundary, base_bar] with closed="right" semantics.
    # Bar 0 (00:00) is at the start boundary of window 0 → excluded.
    # Window 0: bars [1,2] → mean([1.0, 2.0]) = 1.5
    assert result.signed_score_2d[0, 0] == pytest.approx(1.5, abs=1e-6)
    # Window 1: bars [3,4] → mean([3.0, 4.0]) = 3.5
    assert result.signed_score_2d[1, 0] == pytest.approx(3.5, abs=1e-6)
    # Window 2: bars [5,6] → mean([5.0, 6.0]) = 5.5
    assert result.signed_score_2d[2, 0] == pytest.approx(5.5, abs=1e-6)
    # Window 3: bar  [7]   → 7.0
    assert result.signed_score_2d[3, 0] == pytest.approx(7.0, abs=1e-6)


def test_ltf_mode_mean_empty_window() -> None:
    """Scenario 4: window with no panel bars → zeros, invalid."""
    panel_dt = np.array(["2026-01-01T06", "2026-01-01T07"], dtype="datetime64[ns]")
    base_dt = np.array(["2026-01-01T04"], dtype="datetime64[ns]")
    score = np.array([[1.0], [2.0]], dtype=np.float64)

    panel = _make_panel(
        score_2d=score,
        valid_2d=np.ones((2, 1), dtype=bool),
        side_2d=np.full((2, 1), 1, dtype=np.int8),
        to_2d=np.full((2, 1), 0.1, dtype=np.float64),
        datetimes=panel_dt,
    )

    result = _project_panel_to_base_grid(panel, base_dt, "1h", "4h", ltf_mode="mean")

    assert result.signed_score_2d[0, 0] == 0.0
    assert not result.valid_mask_2d[0, 0]
    assert result.side_hint_2d[0, 0] == 0


def test_ltf_mode_invalid_raises() -> None:
    """Scenario 5: unknown ltf_mode raises ValueError."""
    panel_dt = np.array(["2026-01-01T00", "2026-01-01T01"], dtype="datetime64[ns]")
    base_dt = np.array(["2026-01-01T04"], dtype="datetime64[ns]")

    panel = _make_panel(
        score_2d=np.zeros((2, 1), dtype=np.float64),
        valid_2d=np.ones((2, 1), dtype=bool),
        side_2d=np.full((2, 1), 1, dtype=np.int8),
        to_2d=np.full((2, 1), 0.1, dtype=np.float64),
        datetimes=panel_dt,
    )

    with pytest.raises(ValueError, match="Unknown ltf_mode"):
        _project_panel_to_base_grid(panel, base_dt, "1h", "4h", ltf_mode="invalid")


def test_ltf_mode_ignored_for_htf() -> None:
    """Scenario 6: HTF branch ignores ltf_mode."""
    panel_dt = np.array(["2026-01-01T12", "2026-01-02T00"], dtype="datetime64[ns]")
    base_dt = np.array(
        ["2026-01-01T04", "2026-01-01T08", "2026-01-01T12", "2026-01-01T16", "2026-01-01T20", "2026-01-02T00"],
        dtype="datetime64[ns]",
    )
    score = np.array([[10.0], [20.0]], dtype=np.float64)

    panel = _make_panel(
        score_2d=score,
        valid_2d=np.ones((2, 1), dtype=bool),
        side_2d=np.full((2, 1), 1, dtype=np.int8),
        to_2d=np.full((2, 1), 0.1, dtype=np.float64),
        datetimes=panel_dt,
    )

    result_l = _project_panel_to_base_grid(panel, base_dt, "12h", "4h", ltf_mode="last")
    result_m = _project_panel_to_base_grid(panel, base_dt, "12h", "4h", ltf_mode="mean")

    np.testing.assert_array_equal(result_l.signed_score_2d, result_m.signed_score_2d)
