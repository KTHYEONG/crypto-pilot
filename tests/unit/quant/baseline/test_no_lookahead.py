from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.quant.contracts import CostModel, StrategySpec
from src.market_data.storage.loaders import load_ohlcv_4h
from src.quant.baseline.backtest import run_backtest

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


@pytest.fixture
def spec() -> StrategySpec:
    return StrategySpec()


@pytest.fixture
def costs() -> CostModel:
    return CostModel()


class TestNoLookahead:
    def test_future_perturbation(self, spec: StrategySpec, costs: CostModel) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31")
        cut = pd.Timestamp("2024-06-30", tz="UTC")
        base_result = run_backtest(df, spec, costs)
        base_equity = base_result.equity

        rng = np.random.default_rng(0)
        perturbed = df.copy()
        mask = perturbed.index > cut
        n_perturb = mask.sum()
        factors = rng.uniform(0.5, 1.5, size=(n_perturb, 4))
        for i, col in enumerate(["open", "high", "low", "close"]):
            perturbed.loc[mask, col] = perturbed.loc[mask, col].values * factors[:, i]
        perturbed["high"] = perturbed[["open", "high", "low", "close"]].max(axis=1)
        perturbed["low"] = perturbed[["open", "high", "low", "close"]].min(axis=1)

        perturbed_result = run_backtest(perturbed, spec, costs)
        perturbed_equity = perturbed_result.equity

        assert base_equity[base_equity.index <= cut].equals(
            perturbed_equity[perturbed_equity.index <= cut]
        ), "pre-cut equity must be bit-identical"

    def test_no_negative_shift(self) -> None:
        # A negative shift building a forward-return *evaluation label*
        # (assigned to a `fwd`/`fwd_ret`-named variable) is not lookahead bias:
        # the label is consumed only by post-hoc scoring, never fed back into
        # a causal decision. The scan still fails closed on any other
        # negative shift, which would leak future data into a live signal.
        forward_label_pattern = re.compile(
            r"shift\(-\(forward_bars \+ 1\)\)|\bfwd_ret\s*=.*\.shift\(-1\)"
        )
        src = Path("src")
        for pyfile in src.rglob("*.py"):
            text = pyfile.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if "shift(-" in line and not forward_label_pattern.search(line):
                    pytest.fail(f"shift(-1) found in {pyfile}:{i}: {line.strip()}")
