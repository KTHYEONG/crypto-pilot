"""Causal liquidity inputs shared by the account replay and the live daemon.

``causal_adv_sigma`` lives here instead of ``account_sources`` so the live
daemon can size exposure without importing the research replay assembly
(``FrozenSourceContext`` pulls in the frozen research run and its evaluation
machinery).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_adv_sigma(
    daily_quote_volume: pd.DataFrame, daily_close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal ADV and daily sigma on the daily grid.

    ADV is the 30-day median daily quote volume shifted one day and sigma is the
    21-day std of daily log returns shifted one day, so neither includes the day
    it is labelled on.
    """
    adv = daily_quote_volume.rolling(30, min_periods=1).median().shift(1)
    closes = daily_close
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(closes / closes.shift(1))
    daily_sigma = log_returns.rolling(21, min_periods=1).std().shift(1)
    return adv, daily_sigma
