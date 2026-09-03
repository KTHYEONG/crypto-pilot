from __future__ import annotations

import os

import pytest

import src.common.settings as settings

from src.common.settings import HARDWARE_MAX_WORKERS


def test_hardware_max_workers_defaults_from_cpu_and_is_at_least_one() -> None:
    assert HARDWARE_MAX_WORKERS >= 1


def test_environment_override_can_only_reduce_the_cap(monkeypatch) -> None:
    monkeypatch.setenv("HARDWARE_MAX_WORKERS", "1")
    assert settings._default_hardware_max_workers() == 1
    monkeypatch.delenv("HARDWARE_MAX_WORKERS")
    assert settings._default_hardware_max_workers() == (os.cpu_count() or 1)


def test_environment_override_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("HARDWARE_MAX_WORKERS", "0")
    with pytest.raises(ValueError, match=">= 1"):
        settings._default_hardware_max_workers()
    monkeypatch.setenv("HARDWARE_MAX_WORKERS", "abc")
    with pytest.raises(ValueError, match="positive integer"):
        settings._default_hardware_max_workers()


def test_effective_worker_count_bounds_requested_and_hardware() -> None:
    assert settings.effective_worker_count(3, requested=None, hardware_cap=8) == 3
    assert settings.effective_worker_count(3, requested=2, hardware_cap=8) == 2
    assert settings.effective_worker_count(3, requested=1, hardware_cap=8) == 1
    assert settings.effective_worker_count(5, requested=10, hardware_cap=2) == 2
    assert settings.effective_worker_count(5, requested=None, hardware_cap=8) == 5
    with pytest.raises(ValueError, match="distinct_symbol_count"):
        settings.effective_worker_count(0)
    with pytest.raises(ValueError, match="requested"):
        settings.effective_worker_count(3, requested=0)
