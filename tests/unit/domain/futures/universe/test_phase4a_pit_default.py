"""Tests for Phase 4-A: k_in cap on PITUniverseConfig and PIT as default engine."""

import pytest

from src.domain.futures.universe.config import PITUniverseConfig, UniverseConfig


def test_pit_universe_config_k_in_defaults_to_0() -> None:
    """PITUniverseConfig.k_in must default to 0 (capacity-coverage path)."""
    # Arrange / Act
    cfg = PITUniverseConfig()

    # Assert
    assert cfg.k_in == 0
    assert cfg.capacity_coverage_target == pytest.approx(0.90)
    assert cfg.k_max == 150  # Phase 3: breadth-maximizing backstop


def test_universe_config_engine_defaults_to_pit() -> None:
    """UniverseConfig.universe_engine must default to 'pit'."""
    # Arrange / Act
    cfg = UniverseConfig()

    # Assert
    assert cfg.universe_engine == "pit"


def test_universe_config_k_in_zero_means_capacity_coverage_path() -> None:
    """k_in=0 selects the capacity-coverage path, not uncapped pass-through."""
    # Arrange / Act
    cfg = UniverseConfig(pit_config=PITUniverseConfig(k_in=0))

    # Assert: k_in=0 triggers capacity-coverage; coverage target and k_max are present
    assert cfg.pit_config.k_in == 0
    assert cfg.pit_config.capacity_coverage_target == pytest.approx(0.90)
    assert cfg.pit_config.k_max == 150  # Phase 3: breadth-maximizing backstop


# ---------------------------------------------------------------------------
# Scenario 4 — capacity-coverage prefix logic (unit-level, pure arithmetic)
# ---------------------------------------------------------------------------

def _apply_coverage_prefix(
    caps: list[float],
    target: float,
    k_max: int,
) -> int:
    """Mirror of pipeline.py coverage-prefix logic; returns selected count."""
    total = sum(caps)
    if total <= 0.0:
        return min(len(caps), k_max)
    csum = 0.0
    n = len(caps)
    prefix = n
    for i, c in enumerate(caps):
        csum += c
        if csum >= target * total:
            prefix = i + 1
            break
    return min(prefix, k_max)


def test_capacity_coverage_prefix_selects_2_symbols() -> None:
    """Scenario 4 happy path: cumsum crosses 90%x200=180 at index 1 -> 2 symbols."""
    # Arrange
    caps = [100.0, 80.0, 15.0, 4.0, 1.0]  # sorted desc, total=200
    target = 0.90
    k_max = 100

    # Act
    selected = _apply_coverage_prefix(caps, target, k_max)

    # Assert: cumsum[0]=100, cumsum[1]=180 >= 180 → prefix=2
    assert selected == 2


def test_capacity_coverage_prefix_fail_open_when_total_zero() -> None:
    """total<=0 → fail-open: select all eligible up to k_max."""
    # Arrange
    caps = [0.0, 0.0, 0.0]
    target = 0.90
    k_max = 100

    # Act
    selected = _apply_coverage_prefix(caps, target, k_max)

    # Assert: fail-open returns min(len, k_max) = 3
    assert selected == 3


def test_capacity_coverage_prefix_clipped_by_k_max() -> None:
    """When natural prefix exceeds k_max, result is clipped to k_max."""
    # Arrange: 10 symbols each with equal capacity → prefix would be all 10, but k_max=5
    caps = [10.0] * 10  # total=100; 90%x100=90; prefix=9
    target = 0.90
    k_max = 5

    # Act
    selected = _apply_coverage_prefix(caps, target, k_max)

    # Assert: min(9, 5) = 5
    assert selected == 5


def test_pit_universe_config_validator_rejects_invalid_coverage_target() -> None:
    """capacity_coverage_target outside (0, 1] must raise ValueError."""
    with pytest.raises(ValueError, match="capacity_coverage_target"):
        PITUniverseConfig(capacity_coverage_target=0.0)

    with pytest.raises(ValueError, match="capacity_coverage_target"):
        PITUniverseConfig(capacity_coverage_target=1.1)


def test_pit_universe_config_validator_rejects_k_max_below_1() -> None:
    """k_max < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="k_max"):
        PITUniverseConfig(k_max=0)
