"""MHS 파이프라인 단위 테스트 전용 fixture."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

import src.application.research.mhs.marks as marks
import src.market_data.services.futures_collection as fc
from src.application.research.mhs import evaluation as ev
from tests.unit.application.research.mhs.test_evaluation import (
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
def _mhs_shared_roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, pd.Timestamp]]:
    base = tmp_path_factory.mktemp("mhs_shared_base")
    roots: dict[str, tuple[Path, pd.Timestamp]] = {}

    root_long = base / "market_long"
    end_long = _write_mhs_market(root_long, n_hours=26304, with_minute=False)
    roots["long"] = (root_long, end_long)

    root_default = base / "market"
    end_default = _write_mhs_market(root_default)
    _write_3m_cache(root_default)
    roots["default"] = (root_default, end_default)

    root_btc = base / "market_btc"
    end_btc = _write_mhs_market(root_btc, include_btc=True)
    _write_3m_cache(root_btc)
    roots["btc"] = (root_btc, end_btc)

    root_fund = base / "market_funding_vary"
    end_fund = _write_mhs_market(root_fund, funding_cross_sectional=True)
    _write_3m_cache(root_fund)
    roots["fund"] = (root_fund, end_fund)

    root_tbq = base / "market_tbq"
    end_tbq = _write_mhs_market(root_tbq, include_taker_buy_quote=True)
    _write_3m_cache(root_tbq)
    roots["tbq"] = (root_tbq, end_tbq)

    return roots

@pytest.fixture
def mhs_market_long(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots["long"]
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; the module-scoped _mhs_shared_roots fixture
    # reuses the same symbol names across five distinct roots, so a test that
    # already populated the cache from a different root would otherwise leak
    # stale mark data into this one.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end

@pytest.fixture
def mhs_market(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots["default"]
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; the module-scoped _mhs_shared_roots fixture
    # reuses the same symbol names across five distinct roots, so a test that
    # already populated the cache from a different root would otherwise leak
    # stale mark data into this one.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end

@pytest.fixture
def mhs_market_with_btc(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots["btc"]
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; the module-scoped _mhs_shared_roots fixture
    # reuses the same symbol names across five distinct roots, so a test that
    # already populated the cache from a different root would otherwise leak
    # stale mark data into this one.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end

@pytest.fixture
def mhs_market_funding_vary(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots["fund"]
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; the module-scoped _mhs_shared_roots fixture
    # reuses the same symbol names across five distinct roots, so a test that
    # already populated the cache from a different root would otherwise leak
    # stale mark data into this one.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end

@pytest.fixture
def mhs_market_with_taker_buy_quote(_mhs_shared_roots, monkeypatch):
    root, end = _mhs_shared_roots["tbq"]
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; the module-scoped _mhs_shared_roots fixture
    # reuses the same symbol names across five distinct roots, so a test that
    # already populated the cache from a different root would otherwise leak
    # stale mark data into this one.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end
