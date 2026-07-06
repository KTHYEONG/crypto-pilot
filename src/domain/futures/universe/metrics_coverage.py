"""OI/LSR metrics 캐시 커버리지 진단 (Phase 1b, measure-first).

docs/specs/l1-nontrend-diversification-measure-first.md C1 참조.
xs_oi_skew/positioning_unwind/lsr_oi_regime_filter family는 OI/LSR metrics
parquet(FUTURES_DATA_DIR/{symbol}_metrics.parquet)가 물질화돼 있어야 이벤트가
생성된다. 이 모듈은 실행 전 심볼별 non-null 커버리지를 측정해 해당 family
트랙을 진행할지(track_go) 판정하는 순수 진단 함수만 제공한다 — 게이트 로직에는
관여하지 않는다.

Time Complexity: O(S) parquet reads where S = len(symbols), each O(N) for N rows.
Space Complexity: O(N) per symbol frame (not retained across symbols).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.settings import FUTURES_DATA_DIR

MIN_METRICS_COVERAGE_RATIO: float = 0.70
MIN_ELIGIBLE_SYMBOLS: int = 3

_OI_COLUMN = "sum_open_interest"
_LSR_COLUMN = "long_short_ratio"


@dataclass(frozen=True)
class MetricsCoverageEntry:
    symbol: str
    n_rows: int
    oi_non_null_ratio: float
    lsr_non_null_ratio: float
    first_valid_ts: pd.Timestamp | None
    eligible: bool


@dataclass(frozen=True)
class MetricsCoverageReport:
    entries: tuple[MetricsCoverageEntry, ...]
    n_eligible: int
    track_go: bool


def _empty_entry(symbol: str) -> MetricsCoverageEntry:
    return MetricsCoverageEntry(
        symbol=symbol,
        n_rows=0,
        oi_non_null_ratio=0.0,
        lsr_non_null_ratio=0.0,
        first_valid_ts=None,
        eligible=False,
    )


def _compute_entry(
    symbol: str,
    *,
    data_dir: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    min_coverage_ratio: float,
) -> MetricsCoverageEntry:
    path = data_dir / f"{symbol}_metrics.parquet"
    if not path.exists():
        return _empty_entry(symbol)

    try:
        df = pd.read_parquet(path)
    except Exception:
        return _empty_entry(symbol)

    if df.empty or "datetime" not in df.columns:
        return _empty_entry(symbol)

    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.assign(datetime=dt).dropna(subset=["datetime"])
    if start is not None:
        df = df[df["datetime"] >= start]
    if end is not None:
        df = df[df["datetime"] <= end]

    n_rows = len(df)
    if n_rows == 0:
        return _empty_entry(symbol)

    oi_ratio = float(df[_OI_COLUMN].notna().mean()) if _OI_COLUMN in df.columns else 0.0
    lsr_ratio = float(df[_LSR_COLUMN].notna().mean()) if _LSR_COLUMN in df.columns else 0.0
    first_valid_ts: pd.Timestamp | None = None
    if _OI_COLUMN in df.columns:
        valid_idx = df[_OI_COLUMN].first_valid_index()
        if valid_idx is not None:
            first_valid_ts = df.loc[valid_idx, "datetime"]

    eligible = min(oi_ratio, lsr_ratio) >= min_coverage_ratio
    return MetricsCoverageEntry(
        symbol=symbol,
        n_rows=n_rows,
        oi_non_null_ratio=oi_ratio,
        lsr_non_null_ratio=lsr_ratio,
        first_valid_ts=first_valid_ts,
        eligible=eligible,
    )


def compute_metrics_coverage_report(
    symbols: tuple[str, ...],
    *,
    data_dir: Path | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    min_coverage_ratio: float = MIN_METRICS_COVERAGE_RATIO,
    min_eligible_symbols: int = MIN_ELIGIBLE_SYMBOLS,
) -> MetricsCoverageReport:
    """심볼별 OI/LSR metrics 커버리지를 측정하고 트랙 진행 여부를 판정한다."""
    resolved_dir = data_dir if data_dir is not None else FUTURES_DATA_DIR
    entries = tuple(
        _compute_entry(
            symbol,
            data_dir=resolved_dir,
            start=start,
            end=end,
            min_coverage_ratio=min_coverage_ratio,
        )
        for symbol in symbols
    )
    n_eligible = sum(1 for e in entries if e.eligible)
    return MetricsCoverageReport(
        entries=entries,
        n_eligible=n_eligible,
        track_go=n_eligible >= min_eligible_symbols,
    )
