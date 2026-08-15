from __future__ import annotations

import pytest

from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
    resolve_config_by_tf,
)


def test_resolve_config_by_tf_same_as_base_returns_anchor() -> None:
    # Arrange
    anchor_4h = 180

    # Act
    result = resolve_config_by_tf(anchor_4h=anchor_4h, tf="4h")

    # Assert
    assert result == 180


def test_resolve_config_by_tf_scales_to_faster_timeframe() -> None:
    # Arrange
    anchor_4h = 180

    # Act
    result = resolve_config_by_tf(anchor_4h=anchor_4h, tf="2h")

    # Assert
    assert result == 360


def test_resolve_config_by_tf_scales_to_slower_timeframe() -> None:
    # Arrange
    anchor_4h = 180

    # Act
    result = resolve_config_by_tf(anchor_4h=anchor_4h, tf="1d")

    # Assert
    assert result == 30


@pytest.mark.parametrize("tf", ["2h", "4h", "6h", "8h", "12h", "1d"])
def test_resolve_config_by_tf_covers_full_l1_tfs_grid_without_dict_entries(tf: str) -> None:
    # Arrange
    anchor_4h = 42

    # Act
    result = resolve_config_by_tf(anchor_4h=anchor_4h, tf=tf)

    # Assert -- must resolve for every l1_tfs entry even though no _BY_TF dict lists tf explicitly
    assert result >= 1


def test_min_universe_size_for_evidence_configured() -> None:
    # Arrange / Act
    value = OPT_FUTURES_CONFIG["MIN_UNIVERSE_SIZE_FOR_EVIDENCE"]

    # Assert
    assert value == 50


def test_membership_warmup_days_configured() -> None:
    # Arrange / Act
    value = OPT_FUTURES_CONFIG["MEMBERSHIP_WARMUP_DAYS"]

    # Assert
    assert value == 42


def test_tf_probe_is_disabled_by_default() -> None:
    assert OPT_FUTURES_CONFIG["ENABLE_TF_PROBE"] is False
