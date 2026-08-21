"""Golden identity tests for the MHS horizon diagnostic report.

SCENARIO_ANALYSIS_ARCHITECTURE_04: Running the full pipeline on the synthetic
market and serializing the report yields a payload equal to the captured golden
under assert_report_identical with renames={}.  Equality is exact: every float
matches by repr(), no tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.golden.compare import GOLDEN_MATRIX_NAMES, assert_report_identical

GOLDEN_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "golden" / "mhs_report_golden.json"

START = pd.Timestamp("2021-01-01", tz="UTC")
N_HOURS = 2000
DEV_SYMBOLS = [
    sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
    if True  # placeholder; will filter by partition at runtime
][:8]


def _write_mhs_market(
    root: Path,
    symbols: list[str],
    n_hours: int = N_HOURS,
) -> pd.Timestamp:
    """Replicate the synthetic market from test_mhs_horizon_diagnostic.py."""
    hourly = pd.date_range(START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    hour_dir = root / "1h"
    minute_dir = root / "1m"
    five_dir = root / "5m"
    funding_dir = root / "funding"
    mark_dir = root / "markPriceKlines" / "1h"
    for d in (hour_dir, minute_dir, five_dir, funding_dir, mark_dir):
        d.mkdir(parents=True, exist_ok=True)

    n = len(hourly)
    minute_idx = pd.date_range(START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    n_min = len(minute_idx)

    for i, sym in enumerate(symbols):
        sym_n = n
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, sym_n)))
        pd.DataFrame(
            {
                "timestamp": epoch,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "quote_vol": [1000.0] * sym_n,
            },
        ).to_parquet(hour_dir / f"{sym}.parquet")

        sym_n_min = n_min
        minute_noise = rng.normal(0.0, 0.0003, sym_n_min)
        hourly_level = np.repeat(prices, 60)[:sym_n_min]
        minute_prices = hourly_level * np.exp(minute_noise)
        pd.DataFrame(
            {
                "timestamp": minute_epoch,
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
                "quote_vol": [1000.0] * sym_n_min,
            },
        ).to_parquet(minute_dir / f"{sym}.parquet")

        five_frame = pd.DataFrame(
            {
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
            },
            index=minute_idx,
        ).resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"},
        ).dropna()
        pd.DataFrame(
            {
                "timestamp": (five_frame.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
                "open": five_frame["open"].to_numpy(),
                "high": five_frame["high"].to_numpy(),
                "low": five_frame["low"].to_numpy(),
                "close": five_frame["close"].to_numpy(),
                "quote_vol": np.full(len(five_frame), 5000.0),
            },
        ).to_parquet(five_dir / f"{sym}.parquet")

        pd.DataFrame(
            {
                "timestamp": epoch,
                "funding_rate": [0.00005] * sym_n,
                "datetime": hourly,
            },
        ).to_parquet(funding_dir / f"{sym}.parquet")

        mark_hourly = (
            pd.Series(minute_prices, index=minute_idx)
            .resample("1h")
            .last()
            .reindex(hourly)
            .to_numpy()
        )
        pd.DataFrame(
            {
                "timestamp": epoch,
                "open": mark_hourly,
                "high": mark_hourly,
                "low": mark_hourly,
                "close": mark_hourly,
                "datetime": hourly,
            },
        ).to_parquet(mark_dir / f"{sym}.parquet")
    return end


@pytest.fixture(scope="module")
def golden_payload():
    """Load the captured golden report."""
    if not GOLDEN_PATH.exists():
        pytest.skip("Golden not captured yet; run tools/capture_golden.py")
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def synthetic_market(tmp_path_factory):
    import src.market_data.services.futures_collection as fc
    import src.application.research.mhs.marks as marks
    import src.application.research.mhs.statistics as statistics
    from src.research.universe.pit_universe import symbol_partition

    symbols = [
        sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(sym) == "dev"
    ][:8]

    root = tmp_path_factory.mktemp("mhs_golden_market")
    end = _write_mhs_market(root, symbols)
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


def test_golden_identity(synthetic_market, golden_payload):
    """SCENARIO_ANALYSIS_ARCHITECTURE_04: Full pipeline on synthetic market
    yields a payload equal to the golden under assert_report_identical.
    """
    from src.application.research.mhs.contracts import MhsDiagnosticRequest
    from src.application.research.mhs.evaluation import run_mhs_horizon_diagnostic

    root, end = synthetic_market
    report = run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(
            start=str(START), end=str(end), data_root=str(root),
            execution_timeframe="1m", log_run=False,
        )
    )
    actual_payload = report.to_payload()
    # Exclude non-deterministic fields: wall-time measurements and resource stats
    _NON_DETERMINISTIC = frozenset({
        "elapsed_seconds", "run_elapsed_seconds", "resource_measurements",
    })
    assert_report_identical(golden_payload, actual_payload, renames={}, exclude=_NON_DETERMINISTIC)


# One opt-in flag flipped per golden (baseline = all defaults).  Exactly mirrors
# capture_matrix.py.  The identity test for non-baseline goldens requires a
# market written with ``include_taker_buy_quote=True`` (see capture_matrix.py).
MATRIX_OVERRIDES: dict[str, dict[str, object]] = {
    "baseline": {},
    "committee": {"committee_capital": True},
    "discovery": {"discovery_gate": True},
    "trend_sleeve": {"trend_sleeve": True, "trend_sleeve_gross": 0.15},
    "fold_safe": {"fold_safe_horizon_selection": True},
}

_GOLDEN_PATHS = {
    name: GOLDEN_PATH.parent / f"mhs_report_golden_{name}.json"
    for name in GOLDEN_MATRIX_NAMES
}


@pytest.fixture(scope="module")
def matrix_golden_payload(request):
    """Load the named golden; skip if not yet captured."""
    name = request.param
    path = _GOLDEN_PATHS[name]
    if not path.exists():
        pytest.skip(f"Golden not captured yet: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def matrix_market(request, tmp_path_factory):
    """Synthetic market for a named golden.

    Non-baseline goldens need the ``taker_buy_quote`` column, so those markets
    are written via ``test_evaluation._write_mhs_market(include_taker_buy_quote=True)``.
    """
    import src.application.research.mhs.marks as marks
    import src.application.research.mhs.statistics as statistics
    from src.market_data.services.futures_collection import _mark_price_path as _fc_mark_price_path
    from tests.unit.application.research.mhs.test_evaluation import (
        _START as _TE_START,
        _write_mhs_market as _write_taker_market,
    )

    name = request.param
    # Every matrix golden (baseline included) was captured by capture_matrix.py
    # using test_evaluation._write_mhs_market(n_hours=2700, ...) -- matching that
    # exact writer/param set here (not test_golden_identity's own _write_mhs_market,
    # which defaults to N_HOURS=2000 and a different symbol/date layout) is required
    # for the baseline matrix golden to be reproducible.
    root = tmp_path_factory.mktemp(f"mhs_golden_market_{name}")
    end = _write_taker_market(root, n_hours=2700, include_taker_buy_quote=(name != "baseline"))
    start = _TE_START

    originals = {
        "funding_path": marks.funding_path,
        "_mark_price_path": _fc_mark_price_path,
        "_BOOTSTRAP_REPLICATES": statistics._BOOTSTRAP_REPLICATES,
        "_BOOTSTRAP_MEAN_BLOCK": statistics._BOOTSTRAP_MEAN_BLOCK,
        "_BOOTSTRAP_SEED": statistics._BOOTSTRAP_SEED,
    }
    marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    # test_evaluation writes mark frames under markPriceKlines/1h
    import src.market_data.services.futures_collection as fc

    def _mp(symbol, timeframe):
        return root / "markPriceKlines" / timeframe / f"{symbol}.parquet"

    fc._mark_price_path = _mp
    statistics._BOOTSTRAP_REPLICATES = 20
    statistics._BOOTSTRAP_MEAN_BLOCK = 24
    statistics._BOOTSTRAP_SEED = 20260807
    yield root, end, start
    marks.funding_path = originals["funding_path"]
    fc._mark_price_path = originals["_mark_price_path"]
    for nm, val in originals.items():
        if nm == "funding_path" or nm == "_mark_price_path":
            continue
        setattr(statistics, nm, val)


@pytest.mark.parametrize(
    ("name", "matrix_market", "matrix_golden_payload"),
    [(n, n, n) for n in GOLDEN_MATRIX_NAMES],
    ids=GOLDEN_MATRIX_NAMES,
    indirect=["matrix_market", "matrix_golden_payload"],
)
def test_golden_identity_matrix(name, matrix_market, matrix_golden_payload):
    """SCENARIO_MHS_STAGE_DECOMP_00_MATRIX: each named golden matches the decomposed pipeline."""
    from src.application.research.mhs.contracts import MhsDiagnosticRequest
    from src.application.research.mhs.evaluation import run_mhs_horizon_diagnostic

    root, end, start = matrix_market
    request = MhsDiagnosticRequest(
        start=str(start), end=str(end), data_root=str(root),
        execution_timeframe="1m", log_run=False,
        **MATRIX_OVERRIDES[name],
    )
    report = run_mhs_horizon_diagnostic(request)
    actual_payload = report.to_payload()
    _NON_DETERMINISTIC = frozenset({
        "elapsed_seconds", "run_elapsed_seconds", "resource_measurements",
    })
    assert_report_identical(
        matrix_golden_payload, actual_payload, renames={}, exclude=_NON_DETERMINISTIC
    )


def test_run_mhs_diagnostic_entry_point_matches_golden(synthetic_market, golden_payload):
    """The CLI's actual entry point (MhsRunConfig -> run_mhs_diagnostic) is
    bit-exact against the same golden test_golden_identity validates via the
    MhsDiagnosticRequest -> run_mhs_horizon_diagnostic entry point.

    Exercises the full six-stage decomposition end to end through
    src/mhs/pipeline/orchestrator.py -> runner.py -> stages/{panel,selection,
    book,committee,replay,fold}.py -> stages/assemble.py:
    SCENARIO_MHS_STAGE_DECOMP_01_ASSEMBLE
    SCENARIO_MHS_STAGE_DECOMP_02_FOLDS
    SCENARIO_MHS_STAGE_DECOMP_03_REPLAY
    SCENARIO_MHS_STAGE_DECOMP_04_PANEL_SELECTION
    SCENARIO_MHS_STAGE_DECOMP_05_BOOKS_COMMITTEE
    SCENARIO_MHS_STAGE_DECOMP_06_ORCHESTRATOR
    """
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    root, end = synthetic_market
    config = MhsRunConfig(
        start=str(START), end=str(end), data_root=str(root),
        execution_timeframe="1m", log_run=False,
        committee_capital=False, committee_regime_adaptive_tranche=False,
        funding_carry_sleeve=False, committee_target_gross=None,
        pnl_vol_target_mode="median_relative",
    )
    report = run_mhs_diagnostic(config)
    actual_payload = report.to_payload()
    _NON_DETERMINISTIC = frozenset({"elapsed_seconds", "run_elapsed_seconds", "resource_measurements"})
    assert_report_identical(golden_payload, actual_payload, renames={}, exclude=_NON_DETERMINISTIC)
