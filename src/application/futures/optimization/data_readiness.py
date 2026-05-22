from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.domain.futures.optimization.opt_data_utils import filter_symbols_by_data_sufficiency


@dataclass(slots=True, frozen=True)
class DataWindowContract:
    """Data-window contract used by sufficiency checks."""

    fetch_start: date
    is_start: date
    oos_start: date
    end: date
    tf: str
    warmup_bars: int
    require_exec_1m: bool


@dataclass(slots=True, frozen=True)
class DataReadinessResult:
    """Data sufficiency evaluation output with filtered maps."""

    kept_symbols: tuple[str, ...]
    filtered_is_maps: dict[str, dict[str, Any]]
    filtered_oos_maps: dict[str, dict[str, Any]]
    report: pd.DataFrame
    contract: DataWindowContract


def evaluate_data_readiness(
    *,
    tf: str,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    fetch_start: date,
    is_start: date,
    oos_start: date,
    end: date,
    require_exec_1m: bool,
) -> DataReadinessResult:
    kept, filtered_is, filtered_oos, report_df, warmup_bars = filter_symbols_by_data_sufficiency(
        tf=tf,
        data_maps=data_maps,
        oos_data_maps=oos_data_maps,
        valid_symbols=valid_symbols,
        fetch_start=fetch_start.isoformat(),
        is_start=is_start.isoformat(),
        oos_start=oos_start.isoformat(),
        oos_end=end.isoformat(),
        require_exec_1m=require_exec_1m,
    )
    contract = DataWindowContract(
        fetch_start=fetch_start,
        is_start=is_start,
        oos_start=oos_start,
        end=end,
        tf=tf,
        warmup_bars=warmup_bars,
        require_exec_1m=require_exec_1m,
    )
    return DataReadinessResult(
        kept_symbols=tuple(kept),
        filtered_is_maps=filtered_is,
        filtered_oos_maps=filtered_oos,
        report=report_df,
        contract=contract,
    )
