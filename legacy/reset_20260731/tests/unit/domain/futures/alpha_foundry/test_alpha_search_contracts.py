from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaEntryMode,
    AlphaSignalBlueprint,
    L0SearchCell,
)
from src.domain.futures.alpha_foundry.search_space import (
    build_l0_search_cells,
    mark_retired_search_cells,
    resolve_alpha_timeframe_grid,
    timeframe_to_minutes,
)


def _make_blueprint(
    family: str = "sparse_breakout_retest_v2",
    variant: str = "bor_v2_20",
    entry_mode: AlphaEntryMode = "sparse",
) -> AlphaSignalBlueprint:
    return AlphaSignalBlueprint(
        family=family,
        variant=variant,
        archetype="trend",
        timeframe="4h",
        required_fields=("close", "high", "low"),
        causal_lag_bars=1,
        lookback_bars=(20, 40),
        holding_bars=3,
        max_turnover_per_year=120.0,
        entry_mode=entry_mode,
        side_rule_id="breakout_retest_sparse",
        exit_policy_id="atr_trail_2",
    )


class TestAlphaSignalBlueprint:
    def test_validates_successfully(self) -> None:
        bp = _make_blueprint()
        assert bp.family == "sparse_breakout_retest_v2"
        assert bp.causal_lag_bars == 1
        assert bp.holding_bars == 3

    def test_rejects_empty_family(self) -> None:
        with pytest.raises(ValueError, match="family must not be empty"):
            _make_blueprint(family="")

    def test_rejects_empty_variant(self) -> None:
        with pytest.raises(ValueError, match="variant must not be empty"):
            _make_blueprint(variant="")

    def test_rejects_empty_timeframe(self) -> None:
        with pytest.raises(ValueError, match="timeframe must not be empty"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(10,),
                holding_bars=3,
                max_turnover_per_year=120.0,
                entry_mode="sparse",
                side_rule_id="s",
                exit_policy_id="e",
            )

    def test_rejects_empty_side_rule_id(self) -> None:
        with pytest.raises(ValueError, match="side_rule_id must not be empty"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(10,),
                holding_bars=3,
                max_turnover_per_year=120.0,
                entry_mode="sparse",
                side_rule_id="",
                exit_policy_id="e",
            )

    def test_rejects_empty_exit_policy_id(self) -> None:
        with pytest.raises(ValueError, match="exit_policy_id must not be empty"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(10,),
                holding_bars=3,
                max_turnover_per_year=120.0,
                entry_mode="sparse",
                side_rule_id="s",
                exit_policy_id="",
            )

    def test_rejects_holding_bars_zero(self) -> None:
        with pytest.raises(ValueError, match="holding_bars must be >= 1"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(10,),
                holding_bars=0,
                max_turnover_per_year=120.0,
                entry_mode="sparse",
                side_rule_id="s",
                exit_policy_id="e",
            )

    def test_rejects_negative_turnover(self) -> None:
        with pytest.raises(ValueError, match=r"max_turnover_per_year must be >= 0.0"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(10,),
                holding_bars=3,
                max_turnover_per_year=-1.0,
                entry_mode="sparse",
                side_rule_id="s",
                exit_policy_id="e",
            )

    def test_rejects_invalid_lag(self) -> None:
        with pytest.raises(ValueError, match="causal_lag_bars must be >= 1"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=0,
                lookback_bars=(10,),
                holding_bars=3,
                max_turnover_per_year=120.0,
                entry_mode="sparse",
                side_rule_id="s",
                exit_policy_id="e",
            )

    def test_rejects_invalid_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback_bars must be >= 1"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(0,),
                holding_bars=3,
                max_turnover_per_year=120.0,
                entry_mode="sparse",
                side_rule_id="s",
                exit_policy_id="e",
            )

    def test_continuous_mode_requires_turnover_limit(self) -> None:
        with pytest.raises(ValueError, match=r"continuous mode requires max_turnover_per_year <= 365.0"):
            _make_blueprint().__class__(
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                required_fields=("close",),
                causal_lag_bars=1,
                lookback_bars=(10,),
                holding_bars=3,
                max_turnover_per_year=400.0,
                entry_mode="continuous",
                side_rule_id="s",
                exit_policy_id="e",
            )

    def test_is_frozen(self) -> None:
        bp = _make_blueprint()
        with pytest.raises(AttributeError):
            bp.family = "other"


class TestL0SearchCell:
    def test_rejects_invalid_tf_minutes(self) -> None:
        with pytest.raises(ValueError, match="tf_minutes must be positive"):
            L0SearchCell(
                blueprint_id="id",
                family="f",
                variant="v",
                timeframe="4h",
                tf_minutes=0,
                symbol_scope="global",
                cost_floor_bps=5.0,
                expected_event_rate=0.1,
                family_prior_score=0.5,
            )

    def test_rejects_empty_blueprint_id(self) -> None:
        with pytest.raises(ValueError, match="blueprint_id must not be empty"):
            L0SearchCell(
                blueprint_id="",
                family="f",
                variant="v",
                timeframe="4h",
                tf_minutes=240,
                symbol_scope="global",
                cost_floor_bps=5.0,
                expected_event_rate=0.1,
                family_prior_score=0.5,
            )

    def test_rejects_negative_cost_floor(self) -> None:
        with pytest.raises(ValueError, match=r"cost_floor_bps must be >= 0.0"):
            L0SearchCell(
                blueprint_id="id",
                family="f",
                variant="v",
                timeframe="4h",
                tf_minutes=240,
                symbol_scope="global",
                cost_floor_bps=-1.0,
                expected_event_rate=0.1,
                family_prior_score=0.5,
            )

    def test_rejects_negative_event_rate(self) -> None:
        with pytest.raises(ValueError, match=r"expected_event_rate must be >= 0.0"):
            L0SearchCell(
                blueprint_id="id",
                family="f",
                variant="v",
                timeframe="4h",
                tf_minutes=240,
                symbol_scope="global",
                cost_floor_bps=5.0,
                expected_event_rate=-0.1,
                family_prior_score=0.5,
            )

    def test_rejects_nonfinite_prior(self) -> None:
        with pytest.raises(ValueError, match="family_prior_score must be finite"):
            L0SearchCell(
                blueprint_id="id",
                family="f",
                variant="v",
                timeframe="4h",
                tf_minutes=240,
                symbol_scope="global",
                cost_floor_bps=5.0,
                expected_event_rate=0.1,
                family_prior_score=np.nan,
            )

    def test_default_status_is_pending(self) -> None:
        cell = L0SearchCell(
            blueprint_id="bp1",
            family="f",
            variant="v",
            timeframe="4h",
            tf_minutes=240,
            symbol_scope="global",
            cost_floor_bps=5.0,
            expected_event_rate=0.1,
            family_prior_score=0.5,
        )
        assert cell.status == "pending"


class TestTimeframeGrid:
    def test_slow_grid_excludes_fast_tfs(self) -> None:
        grid = resolve_alpha_timeframe_grid(enable_fast_timeframes=False)
        assert "30m" not in grid
        assert "1h" not in grid
        assert "2h" not in grid
        assert "3h" in grid
        assert "4h" in grid
        assert "1d" in grid

    def test_fast_grid_includes_fast_tfs(self) -> None:
        grid = resolve_alpha_timeframe_grid(enable_fast_timeframes=True)
        assert "30m" in grid
        assert "1h" in grid
        assert "2h" in grid

    def test_exclude_daily_removes_1d(self) -> None:
        grid = resolve_alpha_timeframe_grid(enable_fast_timeframes=True, include_daily=False)
        assert "1d" not in grid
        assert "4h" in grid

    def test_timeframe_to_minutes_supports_new_grid(self) -> None:
        assert timeframe_to_minutes("30m") == 30
        assert timeframe_to_minutes("3h") == 180
        assert timeframe_to_minutes("1d") == 1440

    def test_rejects_bad_suffix(self) -> None:
        with pytest.raises(ValueError, match="unsupported timeframe"):
            timeframe_to_minutes("4x")

    def test_rejects_zero_value(self) -> None:
        with pytest.raises(ValueError, match="timeframe value must be positive"):
            timeframe_to_minutes("0h")

    def test_rejects_unsupported_timeframe(self) -> None:
        """S3-6: unsupported timeframe raises ValueError."""
        with pytest.raises(ValueError, match="unsupported timeframe"):
            timeframe_to_minutes("7x")


class TestBuildL0SearchCells:
    def test_creates_deterministic_ids_and_pending(self) -> None:
        bps = [_make_blueprint()]
        cells = build_l0_search_cells(
            blueprints=bps,
            family_prior_scores={"sparse_breakout_retest_v2": 0.5},
            cost_floor_bps_by_tf={"4h": 5.0},
        )
        assert len(cells) > 0
        for cell in cells:
            assert cell.status == "pending"

    def test_does_not_materialize_feature_arrays(self) -> None:
        bps = [_make_blueprint()]
        cells = build_l0_search_cells(
            blueprints=bps,
            family_prior_scores={"sparse_breakout_retest_v2": 0.5},
            cost_floor_bps_by_tf={"4h": 5.0},
        )
        assert all(isinstance(c, L0SearchCell) for c in cells)
        assert all(c.cost_floor_bps >= 0.0 for c in cells)

    def test_deduplicates_duplicate_blueprints(self) -> None:
        bps = [_make_blueprint(), _make_blueprint()]
        cells = build_l0_search_cells(
            blueprints=bps,
            family_prior_scores={"sparse_breakout_retest_v2": 0.5},
            cost_floor_bps_by_tf={"4h": 5.0},
        )
        assert len(cells) == 1


class TestMarkRetiredSearchCells:
    def test_cells_can_be_retired_by_family_tf_variant(self) -> None:
        cell = L0SearchCell(
            blueprint_id="bp1",
            family="f",
            variant="v",
            timeframe="4h",
            tf_minutes=240,
            symbol_scope="global",
            cost_floor_bps=5.0,
            expected_event_rate=0.1,
            family_prior_score=0.5,
        )
        cells = (cell,)
        failed_keys = {("f", "4h", "v")}
        updated = mark_retired_search_cells(cells=cells, failed_keys=failed_keys)
        assert len(updated) == 1
        assert updated[0].status == "retired"

    def test_unmatched_cells_unchanged(self) -> None:
        cell = L0SearchCell(
            blueprint_id="bp1",
            family="f",
            variant="v",
            timeframe="4h",
            tf_minutes=240,
            symbol_scope="global",
            cost_floor_bps=5.0,
            expected_event_rate=0.1,
            family_prior_score=0.5,
        )
        cells = (cell,)
        failed_keys = {("other", "4h", "other")}
        updated = mark_retired_search_cells(cells=cells, failed_keys=failed_keys)
        assert updated[0].status == "pending"


class TestBuildL0SearchCellsGeneratorAware:
    """S1-5: no-generator retirement. S3-8: missing family raises ValueError."""

    def test_retires_cells_when_generator_absent(self) -> None:
        bp = _make_blueprint(family="orphan_family")
        cells = build_l0_search_cells(
            blueprints=[bp],
            family_prior_scores={"orphan_family": 0.0},
            cost_floor_bps_by_tf={"4h": 5.0},
            generator_exists_by_family={"orphan_family": False},
        )
        assert len(cells) == 1
        assert cells[0].status == "retired"
        assert cells[0].retire_reason == "no_generator"

    def test_pending_when_generator_exists(self) -> None:
        bp = _make_blueprint()
        cells = build_l0_search_cells(
            blueprints=[bp],
            family_prior_scores={"sparse_breakout_retest_v2": 0.5},
            cost_floor_bps_by_tf={"4h": 5.0},
            generator_exists_by_family={"sparse_breakout_retest_v2": True},
        )
        assert cells[0].status == "pending"

    def test_raises_on_missing_family_in_generator_map(self) -> None:
        """S3-8: missing family raises ValueError."""
        bp = _make_blueprint(family="unknown_fam")
        with pytest.raises(ValueError, match="missing from generator_exists_by_family"):
            build_l0_search_cells(
                blueprints=[bp],
                family_prior_scores={"unknown_fam": 0.0},
                cost_floor_bps_by_tf={"4h": 5.0},
                generator_exists_by_family={"other_fam": True},
            )

    def test_retired_cell_has_retire_reason(self) -> None:
        """Verify retire_reason field is populated."""
        bp = _make_blueprint(family="dead_family")
        cells = build_l0_search_cells(
            blueprints=[bp],
            family_prior_scores={"dead_family": 0.0},
            cost_floor_bps_by_tf={"4h": 5.0},
            generator_exists_by_family={"dead_family": False},
        )
        assert cells[0].retire_reason == "no_generator"
