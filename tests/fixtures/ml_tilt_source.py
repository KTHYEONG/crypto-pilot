"""Hermetic synthetic census and test-scale ML tilt values for strategy ML scenarios."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.resources import MhsMemoryBudget
from src.core.types import ExecutionSpec
from src.engine.backtest_evidence import StrategyReportPeriod
from src.engine.strategy_backtest import LoadedStrategySource, StrategyBacktestRequest, _strategy_source_fingerprint
from src.strategy.store import EXAMPLE_STORE_ROOT, MlTiltValues, ml_tilt_values
from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

SYMBOLS = tuple(f"S{i:02d}" for i in range(30))
SOURCE_START = pd.Timestamp("2020-08-01", tz="UTC")
SOURCE_END = pd.Timestamp("2021-04-10", tz="UTC")
EVAL_START = pd.Timestamp("2021-03-05", tz="UTC")
EVAL_END = pd.Timestamp("2021-03-20", tz="UTC")
ANCHOR = pd.Timestamp("2021-03-01", tz="UTC")
ROOT = Path("root")


def ml_test_values() -> MlTiltValues:
    base = ml_tilt_values(EXAMPLE_STORE_ROOT)
    assert base is not None
    return dataclasses.replace(
        base, tilt=0.25, horizon_days=1, label_clip=3.0, vol_floor_quantile=0.1, smooth_halflife_days=2.0,
        retrain_days=91, purge_days=1, schedule_anchor=ANCHOR, feature_breadth=20, num_leaves=7,
        min_child_samples=5, learning_rate=0.1, n_trees=20, subsample=1.0, colsample_bytree=1.0,
        reg_lambda=0.0, seed=3, min_train_rows=20,
    )


def request(**overrides: object) -> StrategyBacktestRequest:
    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar")
    stress = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0, decision_anchor="submit_bar")
    params: dict[str, object] = {
        "source_start": SOURCE_START,
        "evaluation_start": EVAL_START,
        "evaluation_end": EVAL_END,
        "strategy": FLOW_MOM_TOP20_ACCOUNT_UNIT,
        "initial_equity": 100000.0,
        "base_spec": base,
        "stress_spec": stress,
        "report_periods": (
            StrategyReportPeriod(label="P1", start=EVAL_START, end=EVAL_END),
        ),
        "data_root": ROOT,
    }
    params.update(overrides)
    return StrategyBacktestRequest(**params)  # type: ignore[arg-type]


def synthetic_source(req: StrategyBacktestRequest, seed: int = 11) -> LoadedStrategySource:
    """Hourly random-walk census with momentum-relevant volume; funding is flat and unbroken."""
    idx = pd.date_range(SOURCE_START, SOURCE_END, freq="h", inclusive="left")
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.01, (len(idx), len(SYMBOLS))), axis=0)) * 100.0,
        index=idx, columns=list(SYMBOLS), dtype="float64",
    )
    quote = pd.DataFrame(rng.uniform(1e6, 1e7, (len(idx), len(SYMBOLS))), index=idx, columns=list(SYMBOLS))
    taker = quote * rng.uniform(0.3, 0.7, (len(idx), len(SYMBOLS)))
    panels = {"close": close, "quote_vol": quote, "taker_buy_quote": taker}
    completed = (idx + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[ns]")
    available = pd.DataFrame(
        np.tile(completed[:, None], (1, len(SYMBOLS))), index=idx, columns=list(SYMBOLS),
    ).apply(lambda col: pd.to_datetime(col).dt.tz_localize("UTC"))
    funding_index = pd.date_range(SOURCE_START, SOURCE_END + pd.Timedelta(days=30), freq="8h", tz="UTC")
    funding = {sym: pd.Series(0.0001, index=funding_index, dtype="float64") for sym in SYMBOLS}
    return LoadedStrategySource(
        fingerprint=_strategy_source_fingerprint(req, str(ROOT)),
        daily_close=close.resample("1D").last().astype("float64"),
        daily_quote_volume=quote.resample("1D").sum(min_count=1).astype("float64"),
        hourly_panels=panels,
        hourly_available_at=available,
        census=SYMBOLS,
        funding_by_symbol=funding,
        funding_failures={},
        root=str(ROOT),
        budget=MhsMemoryBudget(),
    )
