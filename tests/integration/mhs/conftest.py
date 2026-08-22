"""MHS 파이프라인 통합 테스트 전용 fixture."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

import src.application.research.mhs.marks as marks
from src.application.research.mhs import evaluation as ev
import inspect

from src.application.research.mhs import scaling
from src.application.research.mhs import statistics
from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs.evaluation import run_mhs_horizon_diagnostic
from tests.integration.mhs.test_mhs_horizon_diagnostic import (
    DEV_SYMBOLS,
    START,
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
def synthetic_market(tmp_path_factory) -> tuple[Path, pd.Timestamp]:
    import src.market_data.services.futures_collection as fc

    root = tmp_path_factory.mktemp("mhs_market")
    end = _write_mhs_market(root, DEV_SYMBOLS)
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

@pytest.fixture(scope="module")
def report(synthetic_market):
    root, end = synthetic_market
    return run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(start=str(START), end=str(end), data_root=str(root), execution_timeframe="1m", log_run=False),
    )

@pytest.fixture(scope="module")
def touch_report(synthetic_market):
    root, end = synthetic_market
    return run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(
            start=str(START), end=str(end), data_root=str(root),
            execution_timeframe="1m", log_run=False, touch_diagnostic=True,
        ),
    )

@pytest.fixture(scope="module")
def annualization_report(synthetic_market):
    """SCENARIO_MHS_ANNUALIZATION_04: a full diagnostic on the default 5m
    execution grid, used to prove the corrected hourly-grid annualization of
    every real-execution-ledger headline metric against the pre-fix formulas
    applied to the same raw ledger.

    Module-scoped sibling fixtures (``fold_market``/``late_market_report``)
    re-point ``fc._mark_price_path``/``marks.funding_path`` at their own roots and
    only restore them at module teardown, so this fixture re-asserts the
    synthetic-market paths itself before running the diagnostic.
    """
    import src.market_data.services.futures_collection as fc

    root, end = synthetic_market
    originals = {
        "funding_path": marks.funding_path,
        "mark_price_path": fc._mark_price_path,
    }
    marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    try:
        return run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                execution_timeframe="5m", log_run=False,
            ),
        )
    finally:
        marks.funding_path = originals["funding_path"]
        fc._mark_price_path = originals["mark_price_path"]

@pytest.fixture(scope="module")
def calibrated_report(synthetic_market):
    """Full diagnostic run with spies on the R1/R3 wiring seams: records the
    ``ema_span`` passed to every ``_book_weights`` call and which callers invoke
    the regime cash scale and turnover deadband.

    The top-level book replays and anchored folds now run in forked workers
    (Phase 3, P10/P14), so the capture stores use ``multiprocessing.Manager``
    proxies: a fork child's writes are reflected in the parent's proxy instead
    of mutating a copy-on-write shadow.
    """
    from multiprocessing import Manager

    root, end = synthetic_market
    mgr = Manager()
    captured = mgr.dict({
        "ema_spans": mgr.dict(),
        "regime_callers": mgr.list(),
        "deadband_callers": mgr.list(),
    })
    real_book_weights = ev._book_weights
    real_regime = scaling._regime_cash_scale
    real_deadband = scaling._apply_rebalance_deadband

    def _book_weights(log_close, eligible, spec, step_grid, ema_span=None):
        spans = captured["ema_spans"].get(spec.band.name)
        if spans is None:
            spans = mgr.list()
            captured["ema_spans"][spec.band.name] = spans
        spans.append(ema_span)
        return real_book_weights(log_close, eligible, spec, step_grid, ema_span=ema_span)

    def _regime(*args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        captured["regime_callers"].append(caller)
        return real_regime(*args, **kwargs)

    def _deadband(*args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        captured["deadband_callers"].append(caller)
        return real_deadband(*args, **kwargs)

    ev._book_weights = _book_weights
    scaling._regime_cash_scale = _regime
    scaling._apply_rebalance_deadband = _deadband
    try:
        report = run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                execution_timeframe="1m", log_run=False,
            ),
        )
    finally:
        ev._book_weights = real_book_weights
        scaling._regime_cash_scale = real_regime
        scaling._apply_rebalance_deadband = real_deadband
    yield report, captured
    mgr.shutdown()
