"""MHS 파이프라인 단위 테스트 전용 fixture."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

import src.mhs.marks as marks
import src.market_data.services.futures_collection as fc
from src.mhs.marks import clear_mhs_market_data_caches
from tests.unit.mhs.test_evaluation_appresearch import (
    _write_3m_cache,
    _write_mhs_market,
)


from types import SimpleNamespace

import psutil
import pytest

# fork-admission 게이트(assert_fork_admission/plan_worker_count)는 실측
# psutil.virtual_memory()를 참조한다. 기본값을 넉넉하게 고정해 xdist 동시
# 워커의 메모리 경합에 따라 게이트가 우연히 발동하는 것을 막는다(테스트
# 로직이 아니라 동시 실행 중인 다른 워커의 부하에 결과가 좌우되는 플레이키를
# 방지). RAM 가드 자체를 검증하는 테스트는 자신의 monkeypatch로 이 기본값을
# 이후에 덮어써 정상적으로 오버라이드한다.
_AMPLE_MEMORY = SimpleNamespace(total=64 * 2**30, available=60 * 2**30)


@pytest.fixture(autouse=True)
def _mhs_ample_virtual_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _AMPLE_MEMORY)


@pytest.fixture(scope="module")
def _mhs_shared_roots(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("mhs_shared_base")
    cache: dict[str, tuple[Path, pd.Timestamp]] = {}

    def _strip_mark_files(root: Path) -> None:
        mark_root = root / "markPriceKlines"
        if mark_root.exists():
            shutil.rmtree(mark_root, ignore_errors=True)

    def _get(key: str) -> tuple[Path, pd.Timestamp]:
        if key not in cache:
            if key == "long":
                root = base / "market_long"
                end = _write_mhs_market(root, n_hours=26304, with_minute=False)
            elif key == "default":
                root = base / "market"
                end = _write_mhs_market(root)
                _write_3m_cache(root)
            elif key == "btc":
                root = base / "market_btc"
                end = _write_mhs_market(root, include_btc=True)
                _write_3m_cache(root)
            elif key == "fund":
                root = base / "market_funding_vary"
                end = _write_mhs_market(root, funding_cross_sectional=True)
                _write_3m_cache(root)
            elif key == "tbq":
                root = base / "market_tbq"
                end = _write_mhs_market(root, include_taker_buy_quote=True)
                _write_3m_cache(root)
            else:
                raise KeyError(key)
            _strip_mark_files(root)
            cache[key] = (root, end)
        return cache[key]

    return _get

@pytest.fixture
def mhs_market_long(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots("long")
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # Retained loaders are stateless and read the lake directly; the shared
    # invalidation entry point keeps runs isolated when fixtures redirect
    # funding/mark roots between tests.
    clear_mhs_market_data_caches()
    return root, end

@pytest.fixture
def mhs_market(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots("default")
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # Retained loaders are stateless and read the lake directly; the shared
    # invalidation entry point keeps runs isolated when fixtures redirect
    # funding/mark roots between tests.
    clear_mhs_market_data_caches()
    return root, end

@pytest.fixture
def mhs_market_with_btc(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots("btc")
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # Retained loaders are stateless and read the lake directly; the shared
    # invalidation entry point keeps runs isolated when fixtures redirect
    # funding/mark roots between tests.
    clear_mhs_market_data_caches()
    return root, end

@pytest.fixture
def mhs_market_funding_vary(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots("fund")
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # Retained loaders are stateless and read the lake directly; the shared
    # invalidation entry point keeps runs isolated when fixtures redirect
    # funding/mark roots between tests.
    clear_mhs_market_data_caches()
    return root, end

@pytest.fixture
def mhs_market_with_taker_buy_quote(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots("tbq")
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # Retained loaders are stateless and read the lake directly; the shared
    # invalidation entry point keeps runs isolated when fixtures redirect
    # funding/mark roots between tests.
    clear_mhs_market_data_caches()
    return root, end
