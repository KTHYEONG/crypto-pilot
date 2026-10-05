"""MHS 파이프라인 통합 테스트 전용 fixture."""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import pandas as pd
from collections.abc import Iterator
from pathlib import Path

import src.mhs.marks as marks

from src.mhs import statistics
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
from tests.integration.mhs._report_cache import (
    SYNTHETIC_DEFAULT_PROFILE,
    CachedDiagnostic,
    DiagnosticReportCache,
    DiagnosticRunSpec,
    report_cache_group_violations,
)


def _load_horizon_diagnostic_helpers():
    from tests.integration.mhs.test_mhs_horizon_diagnostic import (
        DEV_SYMBOLS,
        START,
        _write_mhs_market,
    )

    return DEV_SYMBOLS, START, _write_mhs_market


from types import SimpleNamespace
from unittest.mock import patch

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


@pytest.fixture(scope="module", autouse=True)
def _mhs_module_ample_virtual_memory() -> None:
    """Keep module-scoped diagnostic fixtures independent of host RAM load."""
    with patch("src.mhs.parallel.psutil.virtual_memory", return_value=_AMPLE_MEMORY):
        yield


@pytest.fixture(scope="module")
def synthetic_market(tmp_path_factory) -> tuple[Path, pd.Timestamp]:
    import src.market_data.services.futures_collection as fc

    dev_symbols, _, write_market = _load_horizon_diagnostic_helpers()
    root = tmp_path_factory.mktemp("mhs_market")
    end = write_market(root, dev_symbols)
    originals = {
        "funding_path": marks.funding_path,
        "mark_price_path": fc._mark_price_path,
        "_BOOTSTRAP_REPLICATES": statistics._BOOTSTRAP_REPLICATES,
        "_BOOTSTRAP_MEAN_BLOCK": statistics._BOOTSTRAP_MEAN_BLOCK,
        "_BOOTSTRAP_SEED": statistics._BOOTSTRAP_SEED,
    }
    marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    statistics._BOOTSTRAP_REPLICATES = 20
    statistics._BOOTSTRAP_MEAN_BLOCK = 24
    statistics._BOOTSTRAP_SEED = 20260807
    yield root, end
    for name, value in originals.items():
        if name == "mark_price_path":
            fc._mark_price_path = value
        elif name == "funding_path":
            marks.funding_path = value
        else:
            setattr(statistics, name, value)

@pytest.fixture(scope="session")
def mhs_report_cache() -> Iterator[DiagnosticReportCache]:
    cache = DiagnosticReportCache()
    yield cache
    cache.clear()


@pytest.fixture
def canonical_report_run(mhs_report_cache, synthetic_market, request) -> Iterator[CachedDiagnostic]:
    _, start, _ = _load_horizon_diagnostic_helpers()
    root, end = synthetic_market
    spec = DiagnosticRunSpec(
        research_baseline(start=str(start), end=str(end), data_root=str(root),
                          execution_timeframe="3m", log_run=False),
        SYNTHETIC_DEFAULT_PROFILE,
    )
    with mhs_report_cache.lease(spec, consumer=request.node.nodeid) as entry:
        yield entry


@pytest.fixture
def report(canonical_report_run):
    """Projection of the canonical run; shares one execution via the report cache."""
    return canonical_report_run.report


@pytest.fixture(scope="module")
def touch_report(synthetic_market):
    _, start, _ = _load_horizon_diagnostic_helpers()
    root, end = synthetic_market
    return run_mhs_horizon_diagnostic(
        research_baseline(
            start=str(start), end=str(end), data_root=str(root),
            execution_timeframe="3m", log_run=False, touch_diagnostic=True,
        ),
    )

@pytest.fixture
def annualization_report(canonical_report_run):
    """Projection of the canonical run on the 3m execution grid."""
    return canonical_report_run.report


@pytest.fixture
def calibrated_report(canonical_report_run):
    """Projection of the canonical run with wiring captures."""
    return canonical_report_run.report, canonical_report_run.observations.calibration_captures()


def pytest_collection_modifyitems(config, items):
    violations = report_cache_group_violations(items, Path(__file__).parent)
    if violations:
        raise pytest.UsageError("MHS report-cache xdist groups:\n" + "\n".join(violations))
