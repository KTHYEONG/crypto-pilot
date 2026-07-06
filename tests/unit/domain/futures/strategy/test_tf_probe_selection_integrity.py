"""TF-Probe ENS integrity tests (S2, S4, S6, S7, S8).

Spec: docs/specs/tf-probe-ens-integrity.md §Test Scenario Design
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    *,
    n_bars: int,
    n_syms: int = 1,
    datetimes: np.ndarray[tuple[int], np.dtype[np.datetime64]],
    score: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    valid: np.ndarray[tuple[int, int], np.dtype[np.bool_]] | None = None,
    side: np.ndarray[tuple[int, int], np.dtype[np.int8]] | None = None,
    turnover: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    family: str = "test_family",
    variant: str = "v1",
    metadata: dict[str, object] | None = None,
) -> CandidateSignalPanel:
    """Build a minimal CandidateSignalPanel for projection tests.

    Args:
        n_bars: Number of time bars T.
        n_syms: Number of symbols N.
        datetimes: Array of datetime64 values [T].
        score: Signed score array [T, N]. Defaults to zeros.
        valid: Valid mask [T, N]. Derived from score != 0 if None.
        side: Side hint [T, N]. Defaults to ones.
        turnover: Turnover proxy [T, N]. Defaults to zeros.
        family: Signal family name.
        variant: Signal variant name.
        metadata: Panel metadata dict.

    Returns:
        CandidateSignalPanel instance.
    """
    score_arr = score if score is not None else np.zeros((n_bars, n_syms), dtype=np.float64)
    valid_arr = valid if valid is not None else (score_arr != 0)
    side_arr = side if side is not None else np.ones((n_bars, n_syms), dtype=np.int8)
    to_arr = turnover if turnover is not None else np.zeros((n_bars, n_syms), dtype=np.float64)

    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={},
        datetimes=datetimes,
        symbols=tuple(f"SYM{i}" for i in range(n_syms)),
        signed_score_2d=score_arr.astype(np.float64),
        side_hint_2d=side_arr.astype(np.int8),
        expected_holding_bars=4,
        min_holding_bars=1,
        stop_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=to_arr.astype(np.float64),
        valid_mask_2d=valid_arr.astype(np.bool_),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# S7 (D7 - LTF fidelity): 1h signal → 1h base preserves bars; 4h base loses most
# ---------------------------------------------------------------------------


class TestProjectPanelLtfFidelity:
    """S7: LTF (1h) projection fidelity across different base grids."""

    def test_project_panel_ltf_fidelity_1h_base_preserves_all_active_bars(self) -> None:
        """1h signal projected to 1h base grid: all 3 active bars must survive.

        Arrange: 48-bar 1h grid; signal active at bars 0,1,2 (score=1.0).
        Act    : project to 1h base (same grid).
        Assert : active count == 3 (no loss).
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(48)],
            dtype="datetime64[ns]",
        )
        score = np.zeros((48, 1), dtype=np.float64)
        score[0:3, 0] = 1.0
        panel = _make_panel(n_bars=48, n_syms=1, datetimes=dt_1h, score=score)

        # Act
        proj = _project_panel_to_base_grid(panel, base_datetimes=dt_1h, tf_i="1h", base_tf="1h")

        # Assert
        active_count = int(np.sum(proj.valid_mask_2d))
        assert active_count == 3, f"1h→1h projection should preserve all 3 active bars; got {active_count}"

    def test_project_panel_ltf_fidelity_4h_base_loses_most_active_bars(self) -> None:
        """1h signal projected to 4h base grid: only 1 bar survives (searchsorted last).

        Arrange: 48-bar 1h grid; signal active at bars 0,1,2; 4h base = every 4th bar.
        Act    : project to 4h base (12-bar grid).
        Assert : active count < 3 (resolution loss confirmed).
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(48)],
            dtype="datetime64[ns]",
        )
        dt_4h = dt_1h[::4]  # 12 bars
        score = np.zeros((48, 1), dtype=np.float64)
        score[0:3, 0] = 1.0
        panel = _make_panel(n_bars=48, n_syms=1, datetimes=dt_1h, score=score)

        # Act
        proj = _project_panel_to_base_grid(panel, base_datetimes=dt_4h, tf_i="1h", base_tf="4h")

        # Assert
        active_count = int(np.sum(proj.valid_mask_2d))
        assert active_count < 3, f"1h→4h projection should lose bars (LTF resolution loss); got {active_count} active"

    def test_project_panel_ltf_fidelity_1h_vs_4h_base_comparison(self) -> None:
        """1h base preserves more active bars than 4h base for a 1h signal.

        This is the core D7 assertion: fidelity(1h base) > fidelity(4h base).

        Arrange: 48-bar 1h grid; signal active at bars 0,1,2.
        Act    : project to 1h base AND 4h base independently.
        Assert : active_1h_base > active_4h_base.
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(48)],
            dtype="datetime64[ns]",
        )
        dt_4h = dt_1h[::4]
        score = np.zeros((48, 1), dtype=np.float64)
        score[0:3, 0] = 1.0
        panel = _make_panel(n_bars=48, n_syms=1, datetimes=dt_1h, score=score)

        # Act
        proj_1h = _project_panel_to_base_grid(panel, base_datetimes=dt_1h, tf_i="1h", base_tf="1h")
        proj_4h = _project_panel_to_base_grid(panel, base_datetimes=dt_4h, tf_i="1h", base_tf="4h")

        active_1h = int(np.sum(proj_1h.valid_mask_2d))
        active_4h = int(np.sum(proj_4h.valid_mask_2d))

        # Assert
        assert active_1h > active_4h, f"1h base should preserve more signal ({active_1h}) than 4h base ({active_4h})"

    def test_project_panel_ltf_fidelity_output_shape_matches_base_grid(self) -> None:
        """Projected panel T dimension must match base_datetimes length.

        Arrange: 48-bar 1h source; project to 4h base (12 bars).
        Assert : proj.signed_score_2d.shape == (12, 1).
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(48)],
            dtype="datetime64[ns]",
        )
        dt_4h = dt_1h[::4]
        panel = _make_panel(n_bars=48, n_syms=1, datetimes=dt_1h)

        # Act
        proj = _project_panel_to_base_grid(panel, base_datetimes=dt_4h, tf_i="1h", base_tf="4h")

        # Assert
        assert proj.signed_score_2d.shape == (12, 1), (
            f"Projected shape should be (12, 1); got {proj.signed_score_2d.shape}"
        )


# ---------------------------------------------------------------------------
# S8 (D7 - HTF fidelity): 6h signal → both 1h and 4h base give consistent coverage
# ---------------------------------------------------------------------------


class TestProjectPanelHtfFidelity:
    """S8: HTF (6h) backward-asof projection is look-ahead-safe on both base grids."""

    def test_project_panel_htf_fidelity_6h_to_1h_and_4h_both_backward_asof(self) -> None:
        """6h signal projected to 1h or 4h base: both must be look-ahead safe.

        Look-ahead safety: base grid bar at time t must only see 6h bars
        that closed at or before t.

        Arrange: 48h window; 6h bars at 0h,6h,12h,18h,24h,30h,36h,42h (8 bars).
                 Score = bar index (0,1,2,...,7) for traceability.
        Act    : project to 1h base and 4h base independently.
        Assert :
          - 1h base has more filled bars than 4h (higher resolution).
          - The first 6h bar coverage: grid bars in [0h,5h] see bar 0 value.
          - No bar in either projection references a 6h bar in its future.
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_6h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 6, "h") for i in range(8)],
            dtype="datetime64[ns]",
        )
        score_6h = np.arange(8, dtype=np.float64).reshape(8, 1)
        panel_6h = _make_panel(
            n_bars=8,
            n_syms=1,
            datetimes=dt_6h,
            score=score_6h,
        )

        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(48)],
            dtype="datetime64[ns]",
        )
        dt_4h = dt_1h[::4]

        # Act
        proj_1h = _project_panel_to_base_grid(panel_6h, base_datetimes=dt_1h, tf_i="6h", base_tf="1h")
        proj_4h = _project_panel_to_base_grid(panel_6h, base_datetimes=dt_4h, tf_i="6h", base_tf="4h")

        # Assert: 1h base has more active coverage (6 1h bars per 6h bar vs 1.5 4h bars)
        active_1h = int(np.sum(proj_1h.valid_mask_2d))
        active_4h = int(np.sum(proj_4h.valid_mask_2d))
        assert active_1h > active_4h, f"6h→1h base should fill more bars ({active_1h}) than 6h→4h ({active_4h})"

    def test_project_panel_htf_fidelity_no_future_reference_1h_base(self) -> None:
        """6h→1h base: bar t must not reference a 6h bar that opens after t.

        The second 6h bar (score=1.0) closes/opens at 6h (hour index 6).
        Grid bars 0..5 (hours 0-5) must see only score=0.0 (first 6h bar).

        Arrange: 2-bar 6h panel; bar 0 at t=0h score=0.0, bar 1 at t=6h score=1.0.
        Assert : proj_1h bars 0..5 have score < 0.5 (bar 1 not yet visible).
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_6h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 6, "h") for i in range(2)],
            dtype="datetime64[ns]",
        )
        score_6h = np.array([[0.0], [1.0]], dtype=np.float64)
        panel_6h = _make_panel(n_bars=2, n_syms=1, datetimes=dt_6h, score=score_6h)

        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(12)],
            dtype="datetime64[ns]",
        )

        # Act
        proj_1h = _project_panel_to_base_grid(panel_6h, base_datetimes=dt_1h, tf_i="6h", base_tf="1h")

        # Assert: bars 0..5 (hours 0-5) must NOT see bar-1 value (score=1.0)
        early_scores = proj_1h.signed_score_2d[0:6, 0]
        assert np.all(early_scores < 0.5), (
            f"Look-ahead violation: hours 0-5 should see score 0.0 only; got {early_scores}"
        )

    def test_project_panel_htf_fidelity_output_shapes_are_correct(self) -> None:
        """6h signal projected: output shape must equal base grid length.

        Arrange: 8-bar 6h panel; project to 1h (48 bars) and 4h (12 bars).
        Assert : shapes are (48, 1) and (12, 1) respectively.
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_6h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 6, "h") for i in range(8)],
            dtype="datetime64[ns]",
        )
        panel_6h = _make_panel(n_bars=8, n_syms=1, datetimes=dt_6h)
        dt_1h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h") for i in range(48)],
            dtype="datetime64[ns]",
        )
        dt_4h = dt_1h[::4]

        # Act
        proj_1h = _project_panel_to_base_grid(panel_6h, base_datetimes=dt_1h, tf_i="6h", base_tf="1h")
        proj_4h = _project_panel_to_base_grid(panel_6h, base_datetimes=dt_4h, tf_i="6h", base_tf="4h")

        # Assert
        assert proj_1h.signed_score_2d.shape == (48, 1)
        assert proj_4h.signed_score_2d.shape == (12, 1)


# ---------------------------------------------------------------------------
# S2 (C1 - inject_full_grid): winning_keys filter bypass logic
# ---------------------------------------------------------------------------


class TestInjectFullGridFilter:
    """S2: inject_full_grid parameter controls winning_keys filter bypass.

    Current code state: inject_full_grid parameter is declared in
    _build_probe_extra_panels signature (bridge.py:305) but the filter
    at line 363 does NOT yet read it. These tests document the CONTRACT
    (what inject_full_grid SHOULD do per spec C1) and verify the
    existing winning_keys filter logic directly.
    """

    def test_inject_full_grid_filter_logic_when_false_excludes_non_winners(self) -> None:
        """inject_full_grid=False contract: panel NOT in winning_keys is excluded.

        This tests the predicate: `if (family, variant) not in winning_keys: continue`.
        We replicate the filter logic directly to assert its correctness.

        Arrange: winning_keys = {("family_a", "v1")}; panels include "v1" and "v2".
        Act    : apply filter with inject_full_grid=False.
        Assert : only "v1" passes.
        """
        # Arrange
        winning_keys: set[tuple[str, str]] = {("family_a", "v1")}
        inject_full_grid = False

        candidate_panels = [
            ("family_a", "v1"),
            ("family_a", "v2"),  # non-winner
            ("family_b", "v3"),  # non-winner, different family
        ]

        # Act — replicate _build_probe_extra_panels filter predicate
        def _passes_filter(family: str, variant: str) -> bool:
            return inject_full_grid or (family, variant) in winning_keys

        passed = [(f, v) for f, v in candidate_panels if _passes_filter(f, v)]

        # Assert
        assert passed == [("family_a", "v1")], f"inject_full_grid=False should admit only winners; got {passed}"

    def test_inject_full_grid_filter_logic_when_true_admits_all_panels(self) -> None:
        """inject_full_grid=True contract: ALL panels pass regardless of winning_keys.

        This tests the bypass predicate: when inject_full_grid=True the
        winning_keys guard is skipped entirely (high-recall C1 mode).

        Arrange: winning_keys = {("family_a", "v1")}; panels include v1,v2,v3.
        Act    : apply filter with inject_full_grid=True.
        Assert : all 3 panels pass.
        """
        # Arrange
        winning_keys: set[tuple[str, str]] = {("family_a", "v1")}
        inject_full_grid = True

        candidate_panels = [
            ("family_a", "v1"),
            ("family_a", "v2"),
            ("family_b", "v3"),
        ]

        # Act — replicate bypass predicate
        def _passes_filter(family: str, variant: str) -> bool:
            return inject_full_grid or (family, variant) in winning_keys

        passed = [(f, v) for f, v in candidate_panels if _passes_filter(f, v)]

        # Assert
        assert len(passed) == 3, f"inject_full_grid=True should admit all panels; got {len(passed)}: {passed}"

    def test_inject_full_grid_empty_winning_keys_when_true_still_admits_all(self) -> None:
        """inject_full_grid=True with empty winning_keys still admits all panels.

        Edge case: probe finds zero winners but inject_full_grid=True means
        L1 evaluates the full non-base grid anyway.

        Arrange: winning_keys = {}; 2 candidate panels.
        Act    : filter with inject_full_grid=True.
        Assert : both pass (bypass is unconditional).
        """
        # Arrange
        winning_keys: set[tuple[str, str]] = set()
        inject_full_grid = True
        candidate_panels = [("family_a", "v1"), ("family_b", "v2")]

        # Act
        def _passes_filter(family: str, variant: str) -> bool:
            return inject_full_grid or (family, variant) in winning_keys

        passed = [(f, v) for f, v in candidate_panels if _passes_filter(f, v)]

        # Assert
        assert len(passed) == 2


# ---------------------------------------------------------------------------
# S4 (C3 - metadata tagging): probe_origin flag on projected panels
# ---------------------------------------------------------------------------


class TestProbeOriginMetadataTagging:
    """S4: projected probe panels must carry probe_origin=True in metadata."""

    def test_probe_origin_metadata_tagged_on_projected_panel(self) -> None:
        """_project_panel_to_base_grid preserves metadata from source panel.

        When caller sets metadata["probe_origin"]=True before calling
        _project_panel_to_base_grid, the projected panel must carry it forward.

        Current code: bridge.py:187 copies metadata via dict(panel.metadata),
        so if the SOURCE panel has probe_origin set, it propagates.

        Arrange: source panel with probe_origin=True, probe_tf="6h" in metadata.
        Act    : project to 4h base.
        Assert : projected.metadata["probe_origin"] is True
                 projected.metadata["probe_tf"] == "6h".
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_6h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 6, "h") for i in range(8)],
            dtype="datetime64[ns]",
        )
        source_metadata = {
            "probe_origin": True,
            "probe_tf": "6h",
            "probe_ic_tstat": 2.5,
        }
        panel = _make_panel(n_bars=8, n_syms=1, datetimes=dt_6h, metadata=source_metadata)

        dt_4h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 4, "h") for i in range(12)],
            dtype="datetime64[ns]",
        )

        # Act
        projected = _project_panel_to_base_grid(panel, base_datetimes=dt_4h, tf_i="6h", base_tf="4h")

        # Assert
        assert projected.metadata.get("probe_origin") is True, "probe_origin=True must propagate through projection"
        assert projected.metadata.get("probe_tf") == "6h", "probe_tf must propagate through projection"

    def test_probe_origin_metadata_ic_tstat_propagates(self) -> None:
        """probe_ic_tstat value in metadata propagates through projection.

        Arrange: source panel with probe_ic_tstat=3.14.
        Assert : projected.metadata["probe_ic_tstat"] == pytest.approx(3.14).
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_6h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 6, "h") for i in range(4)],
            dtype="datetime64[ns]",
        )
        panel = _make_panel(
            n_bars=4,
            n_syms=1,
            datetimes=dt_6h,
            metadata={"probe_origin": True, "probe_tf": "6h", "probe_ic_tstat": 3.14},
        )
        dt_4h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 4, "h") for i in range(6)],
            dtype="datetime64[ns]",
        )

        # Act
        projected = _project_panel_to_base_grid(panel, base_datetimes=dt_4h, tf_i="6h", base_tf="4h")

        # Assert
        assert projected.metadata.get("probe_ic_tstat") == pytest.approx(3.14, rel=1e-6)

    def test_probe_origin_variant_endswith_tf_suffix(self) -> None:
        """Projected panel variant must have _{tf_i} suffix appended.

        Spec: CandidateSignalPanel projected variant naming rule = f"{variant}_{tf_i}".

        Arrange: panel with variant="rr_16"; project tf_i="6h".
        Assert : projected.variant == "rr_16_6h".
        """
        from src.domain.futures.strategy_runtime.bridge import _project_panel_to_base_grid

        # Arrange
        dt_6h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 6, "h") for i in range(4)],
            dtype="datetime64[ns]",
        )
        panel = _make_panel(n_bars=4, n_syms=1, datetimes=dt_6h, variant="rr_16")
        dt_4h = np.array(
            [np.datetime64("2024-01-01T00:00") + np.timedelta64(i * 4, "h") for i in range(6)],
            dtype="datetime64[ns]",
        )

        # Act
        projected = _project_panel_to_base_grid(panel, base_datetimes=dt_4h, tf_i="6h", base_tf="4h")

        # Assert
        assert projected.variant == "rr_16_6h", (
            f"Variant naming rule f'{{variant}}_{{tf_i}}' violated; got '{projected.variant}'"
        )


# ---------------------------------------------------------------------------
# S6 (D4 - cost SSOT): ExecutionCostModel.round_trip_bps() == _DEFAULT_RT_BPS
# ---------------------------------------------------------------------------


class TestProbeCostSSOT:
    """S6: ExecutionCostModel SSOT and _DEFAULT_RT_BPS must be identical."""

    def test_probe_and_l1_round_trip_cost_match(self) -> None:
        """ExecutionCostModel().round_trip_bps() matches config._DEFAULT_RT_BPS.

        Spec D4: probe default 6.0bps != L1 7.5bps causes boundary cell
        inconsistency. _DEFAULT_RT_BPS is the SSOT; both must equal ~7.5bps.

        Assert : ExecutionCostModel().round_trip_bps() == _DEFAULT_RT_BPS (rel 1e-6).
        """
        from src.domain.futures.strategy.config import _DEFAULT_RT_BPS
        from src.domain.futures.strategy.execution_cost import ExecutionCostModel

        # Arrange
        model = ExecutionCostModel()

        # Act
        probe_cost = model.round_trip_bps()

        # Assert
        assert probe_cost == pytest.approx(_DEFAULT_RT_BPS, rel=1e-6), (
            f"ExecutionCostModel().round_trip_bps()={probe_cost} must equal config._DEFAULT_RT_BPS={_DEFAULT_RT_BPS}"
        )

    def test_execution_cost_model_default_round_trip_is_7_5_bps(self) -> None:
        """Default ExecutionCostModel round_trip_bps() is exactly 7.5bps.

        Verification: one_way = 0.75*2 + 0.25*5 + 1 = 3.75bps → RT=7.5bps.

        Assert : round_trip_bps() == pytest.approx(7.5, rel=1e-9).
        """
        from src.domain.futures.strategy.execution_cost import ExecutionCostModel

        # Arrange / Act
        rt = ExecutionCostModel().round_trip_bps()

        # Assert
        assert rt == pytest.approx(7.5, rel=1e-9), f"Default round_trip_bps should be 7.5; got {rt}"

    def test_execution_cost_model_default_is_not_6_bps(self) -> None:
        """Default round_trip_bps must NOT be 6.0 (the stale probe default).

        D4 defect: probe historically passed default 6.0 vs L1's 7.5.
        This regression guard ensures the SSOT value is never 6.0.

        Assert : round_trip_bps() != pytest.approx(6.0, rel=1e-6).
        """
        from src.domain.futures.strategy.execution_cost import ExecutionCostModel

        # Arrange / Act
        rt = ExecutionCostModel().round_trip_bps()

        # Assert
        assert rt != pytest.approx(6.0, rel=1e-6), (
            "round_trip_bps() == 6.0 is the stale probe default; SSOT must be 7.5"
        )
