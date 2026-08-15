"""Tests for mark price preparation in backtest inputs."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.backtest.preparation import prepare_backtest_inputs


def test_prepare_backtest_inputs_mark_price_success() -> None:
    """Test successful mark price alignment and shape validation."""
    n_bars, n_syms = 100, 3
    exec_open_1m = np.random.default_rng(42).normal(100.0, 1.0, (n_bars, n_syms))
    aligned_data = {"exec_open_1m": exec_open_1m}
    params = {"FUTURES_EXECUTION_MODE": "intrabar_1m"}

    mark_price_1m_raw = np.random.default_rng(42).normal(100.0, 1.0, (n_bars, n_syms))
    prepared = prepare_backtest_inputs(
        aligned_data=aligned_data,
        params=params,
        mark_price_1m_raw=mark_price_1m_raw,
    )

    assert prepared.mark_price_1m is not None
    np.testing.assert_array_equal(prepared.mark_price_1m, mark_price_1m_raw)
    assert prepared.execution_mode == "intrabar_1m"


def test_prepare_backtest_inputs_mark_price_shape_mismatch() -> None:
    """Test that ValueError is raised when mark_price_1m_raw shape mismatches."""
    n_bars, n_syms = 100, 3
    exec_open_1m = np.random.default_rng(42).normal(100.0, 1.0, (n_bars, n_syms))
    aligned_data = {"exec_open_1m": exec_open_1m}
    params = {"FUTURES_EXECUTION_MODE": "intrabar_1m"}

    # Mismatch shape: (n_bars, n_syms - 1)
    mark_price_1m_raw = np.random.default_rng(42).normal(100.0, 1.0, (n_bars, n_syms - 1))

    with pytest.raises(ValueError, match="shape"):
        prepare_backtest_inputs(
            aligned_data=aligned_data,
            params=params,
            mark_price_1m_raw=mark_price_1m_raw,
        )


def test_prepare_backtest_inputs_mark_price_none_fallback() -> None:
    """Test successful preparation when mark_price_1m_raw is None."""
    n_bars, n_syms = 100, 3
    exec_open_1m = np.random.default_rng(42).normal(100.0, 1.0, (n_bars, n_syms))
    aligned_data = {"exec_open_1m": exec_open_1m}
    params = {"FUTURES_EXECUTION_MODE": "intrabar_1m"}

    prepared = prepare_backtest_inputs(
        aligned_data=aligned_data,
        params=params,
        mark_price_1m_raw=None,
    )

    assert prepared.mark_price_1m is None
    assert prepared.execution_mode == "intrabar_1m"
