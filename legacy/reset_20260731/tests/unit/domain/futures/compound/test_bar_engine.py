
from src.domain.futures.compound.bar_engine import (
    aggregate_timeframe_bars,
    build_multi_timeframe_bars,
)


def test_bar_engine_importable() -> None:
    assert aggregate_timeframe_bars is not None
    assert build_multi_timeframe_bars is not None
