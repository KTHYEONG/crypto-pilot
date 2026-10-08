"""Observable-day boundaries of shared account liquidity inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.liquidity import causal_adv_sigma


def test_liquidity_excludes_current_and_future_daily_bars() -> None:
    index = pd.date_range("2021-01-01", periods=35, tz="UTC")
    quote = pd.DataFrame({"A": np.arange(1.0, 36.0)}, index=index)
    close = pd.DataFrame({"A": np.exp(np.arange(35) ** 2 / 100.0)}, index=index)
    adv, sigma = causal_adv_sigma(quote, close)
    assert np.isnan(adv.iloc[0, 0])
    assert np.isnan(sigma.iloc[2, 0])
    assert adv.iloc[30, 0] == 15.5
    assert adv.iloc[31, 0] == 16.5
    assert sigma.iloc[3, 0] == pytest.approx(0.02 / np.sqrt(2.0))
    cutoff = index[31]
    quote.loc[cutoff:] *= 1000
    close.loc[cutoff:] *= 1000
    changed_adv, changed_sigma = causal_adv_sigma(quote, close)
    pd.testing.assert_frame_equal(changed_adv.loc[:cutoff], adv.loc[:cutoff])
    pd.testing.assert_frame_equal(changed_sigma.loc[:cutoff], sigma.loc[:cutoff])
