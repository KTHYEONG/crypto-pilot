from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.market_data.public_archive.service import plan_collection

_SYMBOL = "SOLUSDT"
_PREFIX = "data/futures/um/daily/metrics/SOLUSDT/SOLUSDT-metrics-"


class _Lister:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.calls: list[str] = []

    def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str]:
        del timeout
        self.calls.append(prefix)
        return [k for k in self.keys if k.startswith(prefix)]


def _keys(days: list[date]) -> list[str]:
    stray = [f"{_PREFIX}1993.zip", f"{_PREFIX}2024-01-unsort.zip", f"{_PREFIX}2024-01-15.zip.CHECKSUM"]
    return [f"{_PREFIX}{d.isoformat()}.zip" for d in days] + stray


@pytest.mark.parametrize(
    ("start", "end", "mask"),
    [
        (date(2024, 1, 15), date(2024, 1, 16), None),
        (date(2023, 6, 1), date(2024, 1, 16), None),
        (date(2024, 1, 15), date(2024, 1, 16), {"metrics": {_SYMBOL: {date(2024, 1, 15), date(2024, 1, 16)}}}),
        (date(2023, 6, 1), date(2024, 1, 16), {"metrics": {_SYMBOL: {date(2024, 1, 15), date(2024, 1, 16)}}}),
    ],
)
def test_non_daily_keys_are_skipped_not_fatal(tmp_path: Path, start: date, end: date, mask: object) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    lister = _Lister(_keys(days))
    plans = plan_collection(
        lister, root=tmp_path, datasets=["metrics"], symbols={"metrics": [_SYMBOL]},
        start=start, end=end, allowed=mask,  # type: ignore[arg-type]
    )
    assert plans is not None
    assert [p.days for p in plans] == [tuple(days)]
