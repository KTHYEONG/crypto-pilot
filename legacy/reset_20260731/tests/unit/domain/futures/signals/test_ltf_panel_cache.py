from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.signals.ltf_alpha import (
    build_ltf_native_alpha_panels_streaming,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData


@pytest.fixture
def clear_opt_config() -> None:
    key = "LTF_PANEL_CACHE_ENABLED"
    if key in OPT_FUTURES_CONFIG:
        del OPT_FUTURES_CONFIG[key]


def _minimal_aligned(n_bars: int = 100, n_sym: int = 5) -> AlignedMarketData:
    import pandas as pd

    base_dt = pd.date_range("2026-01-01", periods=n_bars, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    ones = np.ones((n_bars, n_sym), dtype=np.float64)
    mask = np.ones((n_bars, n_sym), dtype=bool)
    return AlignedMarketData(
        datetimes=base_dt,
        symbols=tuple(f"SYM{i}" for i in range(n_sym)),
        open_2d=ones * 100.0,
        high_2d=ones * 105.0,
        low_2d=ones * 95.0,
        close_2d=ones * 100.0,
        volume_2d=ones * 1000.0,
        funding_2d=np.zeros((n_bars, n_sym), dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=mask,
        kill_mask=~mask,
    )


def _minimal_plan(sym_list: tuple[str, ...]) -> object:
    return type("_Plan", (), {"skip_reason": None, "symbols": sym_list, "max_workers": 1})()


def test_ltf_panel_cache_hit_on_second_call(tmp_path, mocker):
    cache_dir = str(tmp_path / "ltf_cache")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"LTF_PANEL_CACHE_ENABLED": True, "LTF_PANEL_CACHE_DIR": cache_dir},
        clear=False,
    )
    spy = mocker.patch(
        "src.domain.futures.signals.ltf_alpha._process_streaming_symbol",
        side_effect=lambda **kw: None,
    )
    aligned = _minimal_aligned(n_bars=48, n_sym=3)
    plan = _minimal_plan(aligned.symbols)

    def load_frame(_sym: str) -> None:
        return None

    result1 = build_ltf_native_alpha_panels_streaming(
        aligned=aligned, plan=plan, load_frame=load_frame, budget=None
    )
    result2 = build_ltf_native_alpha_panels_streaming(
        aligned=aligned, plan=plan, load_frame=load_frame, budget=None
    )
    assert spy.call_count == len(aligned.symbols)
    assert isinstance(result1, tuple)
    assert isinstance(result2, tuple)


def test_ltf_panel_cache_miss_on_aligned_shape_change(tmp_path, mocker):
    cache_dir = str(tmp_path / "ltf_cache")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"LTF_PANEL_CACHE_ENABLED": True, "LTF_PANEL_CACHE_DIR": cache_dir},
        clear=False,
    )
    spy = mocker.patch(
        "src.domain.futures.signals.ltf_alpha._process_streaming_symbol",
        side_effect=lambda **kw: None,
    )
    aligned_a = _minimal_aligned(n_bars=100, n_sym=5)
    aligned_b = _minimal_aligned(n_bars=50, n_sym=3)
    plan_a = _minimal_plan(aligned_a.symbols)
    plan_b = _minimal_plan(aligned_b.symbols)
    def load_frame(_sym: str) -> None:
        return None

    build_ltf_native_alpha_panels_streaming(
        aligned=aligned_a, plan=plan_a, load_frame=load_frame, budget=None
    )
    build_ltf_native_alpha_panels_streaming(
        aligned=aligned_b, plan=plan_b, load_frame=load_frame, budget=None
    )
    assert spy.call_count == len(aligned_a.symbols) + len(aligned_b.symbols)


def test_cache_corrupt_file_triggers_recompute(tmp_path, mocker, caplog):
    cache_dir = str(tmp_path / "ltf_cache")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"LTF_PANEL_CACHE_ENABLED": True, "LTF_PANEL_CACHE_DIR": cache_dir},
        clear=False,
    )
    spy = mocker.patch(
        "src.domain.futures.signals.ltf_alpha._process_streaming_symbol",
        side_effect=lambda **kw: None,
    )
    aligned = _minimal_aligned(n_bars=48, n_sym=2)
    plan = _minimal_plan(aligned.symbols)
    def load_frame(_sym: str) -> None:
        return None

    result1 = build_ltf_native_alpha_panels_streaming(
        aligned=aligned, plan=plan, load_frame=load_frame, budget=None
    )
    count_after_first = spy.call_count
    n_cached_files = len(list(tmp_path.rglob("*.pkl")))
    assert n_cached_files >= 1

    for pkl in tmp_path.rglob("*.pkl"):
        pkl.write_bytes(b"CORRUPT")

    spy.call_count = count_after_first
    result2 = build_ltf_native_alpha_panels_streaming(
        aligned=aligned, plan=plan, load_frame=load_frame, budget=None
    )
    assert spy.call_count > count_after_first


def test_cache_disabled_skips_disk_io(tmp_path, mocker):
    cache_dir = str(tmp_path / "ltf_cache")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"LTF_PANEL_CACHE_ENABLED": False, "LTF_PANEL_CACHE_DIR": cache_dir},
        clear=False,
    )
    spy = mocker.patch(
        "src.domain.futures.signals.ltf_alpha._process_streaming_symbol",
        side_effect=lambda **kw: None,
    )
    aligned = _minimal_aligned(n_bars=48, n_sym=2)
    plan = _minimal_plan(aligned.symbols)
    def load_frame(_sym: str) -> None:
        return None

    build_ltf_native_alpha_panels_streaming(
        aligned=aligned, plan=plan, load_frame=load_frame, budget=None
    )
    build_ltf_native_alpha_panels_streaming(
        aligned=aligned, plan=plan, load_frame=load_frame, budget=None
    )
    assert spy.call_count == len(aligned.symbols) * 2
