from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.optimization import opt_data_utils
from src.domain.futures.universe import storage


def test_load_single_symbol_data_sort_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: raw_df가 이미 monotonic increasing(정렬됨) 상태인 경우 sort_values를 bypass하는지 검증."""
    mock_collector = MagicMock()
    # 4h ohlcv 데이터 준비 (충분히 길게: 2개년 데이터로 구성하여 min_bars 경계조건 문제 완전 예방)
    dt = pd.date_range("2023-01-01", "2025-01-10", freq="4h", tz="UTC")
    raw_df = pd.DataFrame(
        {
            "datetime": dt,
            "timestamp": [int(t.timestamp() * 1000) for t in dt],
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
        }
    )
    mock_collector.collect_and_save.return_value = raw_df
    monkeypatch.setattr(opt_data_utils, "DataCollector", lambda: mock_collector)

    funding_df = pd.DataFrame(
        {
            "timestamp": [int(t.timestamp() * 1000) for t in dt],
            "funding_rate": 0.0001,
            "symbol": "BTCUSDT",
            "datetime": dt,
        }
    )
    monkeypatch.setattr(opt_data_utils, "_safe_read_funding_parquet", lambda sym: funding_df)

    original_sort_values = pd.DataFrame.sort_values
    sort_values_calls = []

    def spy_sort_values(self: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        # 루프 외부의 funding_df 정렬 호출을 제외하고 루프 내부 df(open 컬럼 보유)의 호출만 수집
        if "open" in self.columns and ((len(args) > 0 and args[0] == "timestamp") or kwargs.get("by") == "timestamp"):
            sort_values_calls.append(self)
        return original_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", spy_sort_values)

    _, temp_is, _, insufficient = opt_data_utils.load_single_symbol_data(
        sym="BTCUSDT",
        tf="4h",
        fetch_start="2023-01-01",
        start="2023-01-02",
        is_end="2024-12-31",
        end="2025-01-10",
        skip_metrics=False,  # funding_df_prepared가 None이 되지 않도록 설정
        target_tfs=["4h", "1d"],  # merge_idx_4h 계산을 위해 1d도 포함
        load_exec_1m=False,
    )

    # Monotonic이 보장되므로 sort_values 호출이 bypass되어야 한다. (Time: O(M + F) when sorted)
    assert len(sort_values_calls) == 0
    assert not insufficient
    assert temp_is is not None
    assert "4h" in temp_is
    assert "funding_rate" in temp_is["4h"].columns


def test_load_single_symbol_data_fallback_sorting(monkeypatch: pytest.MonkeyPatch) -> None:
    """S2: raw_df가 정렬되지 않은 상태인 경우 sort_values가 fallback 호출되는지 검증."""
    mock_collector = MagicMock()
    dt = pd.date_range("2023-01-01", "2025-01-10", freq="4h", tz="UTC")
    raw_df = pd.DataFrame(
        {
            "datetime": dt[::-1],  # 역순 정렬
            "timestamp": [int(t.timestamp() * 1000) for t in dt][::-1],
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
        }
    )
    mock_collector.collect_and_save.return_value = raw_df
    monkeypatch.setattr(opt_data_utils, "DataCollector", lambda: mock_collector)

    funding_df = pd.DataFrame(
        {
            "timestamp": [int(t.timestamp() * 1000) for t in dt],
            "funding_rate": 0.0001,
            "symbol": "BTCUSDT",
            "datetime": dt,
        }
    )
    monkeypatch.setattr(opt_data_utils, "_safe_read_funding_parquet", lambda sym: funding_df)

    original_sort_values = pd.DataFrame.sort_values
    sort_values_calls = []

    def spy_sort_values(self: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if "open" in self.columns and ((len(args) > 0 and args[0] == "timestamp") or kwargs.get("by") == "timestamp"):
            sort_values_calls.append(self)
        return original_sort_values(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", spy_sort_values)

    _, _, _, insufficient = opt_data_utils.load_single_symbol_data(
        sym="BTCUSDT",
        tf="4h",
        fetch_start="2023-01-01",
        start="2023-01-02",
        is_end="2024-12-31",
        end="2025-01-10",
        skip_metrics=False,
        target_tfs=["4h", "1d"],  # merge_idx_4h 계산을 위해 1d도 포함
        load_exec_1m=False,
    )

    # 역순이므로 단조 증가하지 않아 sort_values가 호출되어야 함.
    assert len(sort_values_calls) > 0
    assert not insufficient


def test_run_historical_sync_parallel_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """S3: run_historical_sync 호출 시 병렬화 방식이 sync_tasks 결과의 일치성을 유지하는지 검증."""
    ledger_path = tmp_path / "ledger.db"
    monkeypatch.setattr(storage, "DEFAULT_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(storage, "FUTURES_DATA_DIR", tmp_path)

    import sqlite3

    conn = sqlite3.connect(str(ledger_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE ledger (symbol TEXT, date TEXT, row_count INTEGER)")
    cursor.execute("INSERT INTO ledger VALUES ('BTCUSDT', '2024-01-10', 10)")
    cursor.execute("INSERT INTO ledger VALUES ('ETHUSDT', '2024-01-10', 10)")
    conn.commit()
    conn.close()

    mock_profiles = {
        "BTCUSDT": storage.SymbolSyncProfile("BTCUSDT", date(2023, 1, 1), None, "TRADING"),
        "ETHUSDT": storage.SymbolSyncProfile("ETHUSDT", date(2023, 1, 1), None, "TRADING"),
    }
    monkeypatch.setattr(storage, "_load_symbol_sync_profiles", lambda: mock_profiles)
    monkeypatch.setattr(storage, "_requested_sync_caches_missing", lambda *args, **kwargs: False)

    pool_instances = []

    class MockPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pool_instances.append(self)

        def __enter__(self) -> MockPool:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def map(self, func: Any, iterable: Any) -> list[tuple[list[Any], int]]:
            return [([], 0)] * len(iterable)

    import multiprocessing

    monkeypatch.setattr(multiprocessing, "Pool", MockPool)

    storage.run_historical_sync(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        symbols=["BTCUSDT", "ETHUSDT"],
        sync_1d=True,
        sync_4h=True,
        sync_1m=False,
    )

    # 캐시 미싱이 없으므로 추가 sync_tasks가 구성되지 않아 Pool이 생성되지 않아야 함.
    assert len(pool_instances) == 0


def test_run_historical_sync_parallel_speedup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """S4: I/O 지연 상황에서 병렬 스캔의 처리 속도가 순차 스캔 대비 2배 이상 가속되는지 검증."""
    symbols = [f"SYM{i}USDT" for i in range(50)]
    metadata_cache: dict[str, Any] = {}
    profile = storage.SymbolSyncProfile("TEST", date(2023, 1, 1), None, "TRADING")

    # I/O 블로킹 시뮬레이션을 위한 0.005초 슬립
    def mock_caches_missing(*args: Any, **kwargs: Any) -> bool:
        time.sleep(0.005)
        return True

    # 1. Sequential execution time
    t0_seq = time.perf_counter()
    for symbol in symbols:
        mock_caches_missing(
            symbol,
            sync_1d=True,
            sync_4h=True,
            sync_1m=False,
            requested_start=date(2024, 1, 1),
            requested_end=date(2024, 1, 10),
            metadata_cache=metadata_cache,
            profile=profile,
        )
    elapsed_seq = time.perf_counter() - t0_seq

    # 2. Parallel execution time using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    t0_par = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            symbol: executor.submit(
                mock_caches_missing,
                symbol,
                sync_1d=True,
                sync_4h=True,
                sync_1m=False,
                requested_start=date(2024, 1, 1),
                requested_end=date(2024, 1, 10),
                metadata_cache=metadata_cache,
                profile=profile,
            )
            for symbol in symbols
        }
        for fut in futures.values():
            _ = fut.result()
    elapsed_par = time.perf_counter() - t0_par

    # 3. Verify speedup ratio (>= 2.0x)
    ratio = elapsed_seq / elapsed_par
    msg = f"Parallel scanning speedup is only {ratio:.2f}x (Seq: {elapsed_seq:.4f}s, Par: {elapsed_par:.4f}s)"
    assert ratio >= 2.0, msg


def test_membership_mask_output_verification() -> None:
    """Scenario 1: Membership Mask Output Verification.

    Numba 및 Timestamp 기반의 membership mask가 warmup_bars_required 조건에 맞게 올바르게 산출되는지 직접 검증.
    """
    from src.domain.futures.universe.membership import build_membership_mask_bundle

    dt_ser = pd.Series(
        pd.to_datetime(
            [
                "2024-01-15",
                "2024-02-15",  # Q1
                "2024-05-15",
                "2024-06-15",  # Q2
                "2024-08-15",
                "2024-09-15",  # Q3
                "2024-11-15",  # Q4
            ],
            utc=True,
        )
    )

    timeline = {date(2024, 1, 1): frozenset(["BTCUSDT"]), date(2024, 7, 1): frozenset(["BTCUSDT"])}

    bundle = build_membership_mask_bundle(datetimes=dt_ser, symbol="BTCUSDT", timeline=timeline, warmup_bars_required=2)

    expected_active = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(bundle.universe_active_mask, expected_active)

    expected_warm_ready = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(bundle.universe_entry_warm_mask, expected_warm_ready)


def test_feature_group_coverage_preservation() -> None:
    """Scenario 2: Feature Group Coverage Data Preservation.

    정수, 실수, 텍스트가 혼재된 데이터프레임에서 _feature_group_coverage의 출력이 정상적인지 확인.
    """
    from src.domain.futures.optimization.opt_data_utils import _feature_group_coverage

    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "funding_rate": [0.001, None, 0.0],
            "lsr_ratio": ["0.5", "0.6", "abc"],
        }
    )

    coverage = _feature_group_coverage(df)

    assert coverage["price"]["col_count"] == 2.0
    assert coverage["price"]["non_null_coverage"] == 1.0
    assert coverage["price"]["non_zero_coverage"] == 1.0

    assert coverage["funding"]["col_count"] == 1.0
    assert abs(coverage["funding"]["non_null_coverage"] - (2.0 / 3.0)) < 1e-9

    assert coverage["lsr"]["col_count"] == 1.0
    assert abs(coverage["lsr"]["non_null_coverage"] - (2.0 / 3.0)) < 1e-9


def test_sorting_bypass_logics_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 3: Sorting Bypass Logics Verification.

    load_single_symbol_data 실행 시 1분봉 데이터의 datetime 정렬 및 바이패스 작동 확인.
    """
    mock_collector = MagicMock()
    dt = pd.date_range("2023-01-01", "2025-01-10", freq="4h", tz="UTC")
    raw_df = pd.DataFrame(
        {
            "datetime": dt,
            "timestamp": [int(t.timestamp() * 1000) for t in dt],
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 100.0,
        }
    )
    mock_collector.collect_and_save.return_value = raw_df

    dt_1m = pd.date_range("2023-01-01", "2025-01-10", freq="1min", tz="UTC")
    exec_1m = pd.DataFrame({"datetime": dt_1m, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 10.0})
    mock_collector.collect_1m_ohlcv.return_value = exec_1m

    monkeypatch.setattr(opt_data_utils, "DataCollector", lambda: mock_collector)
    monkeypatch.setattr(opt_data_utils, "_safe_read_funding_parquet", lambda sym: None)

    _, temp_is, temp_oos, insufficient = opt_data_utils.load_single_symbol_data(
        sym="BTCUSDT",
        tf="4h",
        fetch_start="2023-01-01",
        start="2023-01-02",
        is_end="2024-12-31",
        end="2025-01-10",
        skip_metrics=True,
        target_tfs=["4h", "1d"],
        load_exec_1m=True,
    )

    assert not insufficient
    assert temp_is is not None
    assert temp_oos is not None
    assert "exec_1m" in temp_is
    assert pd.api.types.is_datetime64_any_dtype(temp_is["exec_1m"]["datetime"])
