"""Realistic-execution primary swap (mhs_realistic_execution_primary_swap).

Verifies the Research-GO primary anchoring change: both the top-level book
replay and the anchored-fold replay feed their ``primary``/``strict`` report
slots from the ``OHLCV_IMMEDIATE_TAKER`` bound (cost-stressed x3 for the
``stress`` slot), the former ``OHLCV_STRICT_PROXY`` bound is demoted to an
informational ``patient_reference`` diagnostic on ``MhsBookReport`` only, and
the top-level ``fill_source`` metadata reports the new primary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
from src.application.research.mhs import evaluation as ev
from src.mhs.contracts import ExecutionSpec
from src.mhs.evaluation import DeploymentReadinessResult
from src.mhs.execution import strategy_aware_execution_replay
from src.research.universe.pit_universe import symbol_partition

_START = pd.Timestamp("2021-01-01", tz="UTC")

_FOLD = ev.AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)

_DEV_SYMBOLS = [
    s for s in (
        "MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT",
    )
    if symbol_partition(s) == "dev"
]


def _write_mhs_market(root: Path) -> pd.Timestamp:
    symbols = _DEV_SYMBOLS
    n_hours = 2700
    hourly = pd.date_range(_START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    hdir = root / "1h"
    mdir = root / "1m"
    fdir = root / "funding"
    mkdir = root / "markPriceKlines" / "1h"
    for d in (hdir, mdir, fdir, mkdir):
        d.mkdir(parents=True, exist_ok=True)
    minute_idx = pd.date_range(_START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    for i, sym in enumerate(symbols):
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n_hours)))
        pd.DataFrame(
            {"timestamp": epoch, "open": prices, "high": prices * 1.001,
             "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * n_hours},
        ).to_parquet(hdir / f"{sym}.parquet")
        # 1m fills track the same underlying instrument as the 1h close within
        # a tight intra-hour noise band (mirrors real exchange data); an
        # unrelated random walk would diverge unboundedly from `prices` over
        # the fixture's window and spuriously trip fill_mark_parity_mask
        # (I1). Draws the identical rng.normal(..., len(minute_idx)) shape as
        # before so `prices`'s own draws stay at the same rng-stream
        # position -- only this local formula changes.
        minute_noise = rng.normal(0.0, 0.0003, len(minute_idx))
        hourly_level = np.repeat(prices, 60)[: len(minute_idx)]
        mp = hourly_level * np.exp(minute_noise)
        pd.DataFrame(
            {"timestamp": minute_epoch, "open": mp, "high": mp * 1.0005,
             "low": mp * 0.9995, "close": mp, "quote_vol": [1000.0] * len(minute_idx)},
        ).to_parquet(mdir / f"{sym}.parquet")
        pd.DataFrame(
            {"timestamp": epoch, "funding_rate": [0.00005] * n_hours, "datetime": hourly},
        ).to_parquet(fdir / f"{sym}.parquet")
        mark = pd.Series(mp, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
        pd.DataFrame(
            {"timestamp": epoch, "open": mark, "high": mark, "low": mark, "close": mark, "datetime": hourly},
        ).to_parquet(mkdir / f"{sym}.parquet")
    return end


@pytest.fixture
def mhs_market(tmp_path, monkeypatch):
    root = tmp_path / "market"
    end = _write_mhs_market(root)
    monkeypatch.setattr(ev, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; a prior test in the same process/worker using
    # a different root with an overlapping symbol name would otherwise leak
    # stale mark data into this fixture's replay.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end


def _build_book_outcome_args(mhs_market) -> dict[str, object]:
    root, end = mhs_market
    symbols = _DEV_SYMBOLS[:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = ev.MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    bar_funding = bar_funding[aligned]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned}
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    phase = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = ev.renormalize_within_mask(
        w_fast, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    return {
        "name": "fast_reversal",
        "spec": fast,
        "n_symbols": len(aligned),
        "step_grid": fast_grid,
        "weights_step": w_fast,
        "grid_1h": grid_1h,
        "opens": opens,
        "bar_funding": bar_funding,
        "phase": phase,
        "root": str(root),
        "request": request,
        "funding_by_symbol": funding_by_symbol,
        "start": _START,
        "end": end,
        "event_window_bars": fast.horizon_hours,
        "initial_equity": 1.0,
        "replay_weights_step": w_fast_execution,
    }


def test_stress_cost_execution_spec_triples_cost_fields() -> None:
    """SCENARIO_MHS_REALISTIC_EXECUTION_STRESS_SPEC_TRIPLED_01."""
    base = ExecutionSpec()
    spec = ev._stress_cost_execution_spec()
    assert spec.maker_fee_bps == base.maker_fee_bps * ev.MHS_STRESS_COST_MULTIPLIER
    assert spec.taker_fee_bps == base.taker_fee_bps * ev.MHS_STRESS_COST_MULTIPLIER
    assert spec.taker_slippage_bps == base.taker_slippage_bps * ev.MHS_STRESS_COST_MULTIPLIER
    assert spec.passive_timeout_minutes == base.passive_timeout_minutes
    assert spec == ExecutionSpec(maker_fee_bps=6.0, taker_fee_bps=15.0, taker_slippage_bps=9.0)


def test_toplevel_book_primary_is_immediate_taker(mhs_market) -> None:
    """SCENARIO_MHS_REALISTIC_EXECUTION_TOPLEVEL_PRIMARY_IS_IMMEDIATE_TAKER_02."""
    report, _ = ev._book_outcome(**_build_book_outcome_args(mhs_market))
    assert report.primary is not None
    assert report.stress is not None
    assert report.primary.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert report.primary.ledger.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert report.stress.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert report.stress.ledger.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert report.patient_reference is not None
    assert report.patient_reference.fill_source == "OHLCV_STRICT_PROXY"
    assert report.patient_reference.ledger.fill_source == "OHLCV_STRICT_PROXY"
    assert report.patient_reference_naive_sharpe is not None
    assert report.stress.ledger.fee_charge.sum() > report.primary.ledger.fee_charge.sum()


def test_fold_primary_is_immediate_taker(mhs_market) -> None:
    """SCENARIO_MHS_REALISTIC_EXECUTION_FOLD_PRIMARY_IS_IMMEDIATE_TAKER_03."""
    root, end = mhs_market
    symbols = _DEV_SYMBOLS[:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = ev.MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    fold_report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    if fold_report.strict is None or fold_report.stress is None:
        assert fold_report.failures, "fold must either complete or report typed failures"
        return
    assert fold_report.strict.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert fold_report.strict.ledger.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert fold_report.stress.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert fold_report.stress.ledger.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert ev.MHS_GO_PRIMARY_SHARPE_FLOOR == 0.6
    assert ev.MHS_GO_REASON_PRIMARY_SHARPE == "PRIMARY_AUTOCORR_SHARPE_BELOW_0_6"
    assert ev.MHS_GO_REASON_STRESS_SHARPE == "STRESS_SHARPE_NOT_POSITIVE"


def test_report_fill_source_is_immediate_taker(mhs_market, monkeypatch) -> None:
    """SCENARIO_MHS_REALISTIC_EXECUTION_FILL_SOURCE_METADATA_04."""
    root, end = mhs_market
    blend_report, _ = ev._book_outcome(**_build_book_outcome_args(mhs_market))
    assert blend_report.primary is not None
    deployment = DeploymentReadinessResult(
        geometric_cagr=0.0, max_drawdown=0.0, calmar=0.0, expected_shortfall=0.0,
        worst_1d=0.0, worst_7d=0.0, worst_event=0.0, time_under_water_bars=0,
        recovery_bars=None, probability_final_wealth_below_initial=0.0,
        probability_mdd_over_20pct=0.0, probability_mdd_over_30pct=0.0,
        leverage_ruin_probabilities={}, concentration={},
        participation_warnings={}, research_go_eligible=False,
        execution_go_eligible=False, pilot_go_eligible=False, scale_go_eligible=False,
    )
    monkeypatch.setattr(
        ev, "_run_books_concurrent",
        lambda *a, **k: (blend_report, blend_report, blend_report, {}),
    )
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently",
        lambda *a, **k: (None, None, {}, {}, (), deployment),
    )
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    request = ev.MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert report.mark_source == blend_report.primary.ledger.mark_source


def _small_replay(bound: str) -> ev.StrategyExecutionReplayResult:
    idx = pd.date_range("2021-01-01 12:01", periods=4000, freq="1min", tz="UTC")
    px = pd.DataFrame({"A": [100.0] * len(idx)}, index=idx)
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
    signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
    return strategy_aware_execution_replay(
        target, signal_at, px, px, px, px,
        pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0, bound, ExecutionSpec(),
    )


def test_compact_payload_strips_patient_reference(tmp_path) -> None:
    """The informational patient_reference must not bloat the compact JSON:
    like primary/stress it is replaced by a row-count artifact ref, while the
    scalar patient_reference_naive_sharpe stays in the summary."""
    replay = _small_replay("OHLCV_IMMEDIATE_TAKER")
    patient = _small_replay("OHLCV_STRICT_PROXY")
    book = ev.MhsBookReport(
        name="fast_reversal", band="FAST", horizon_hours=24, step_hours=6,
        tranche_count=1, n_symbols=1,
        phase=ev.PhaseDiagnosticResult(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),
        prescreen={}, tail=ev.TailSensitivityResult(0.0, 0.0, {}, 1, 0, 0.0, 0.0, 0.0, 0.0),
        primary=replay, stress=replay,
        primary_autocorr_sharpe=0.1, primary_naive_sharpe=0.1, primary_net_ann=0.01,
        primary_geometric_cagr=0.01, primary_max_drawdown=-0.01,
        primary_annualized_turnover=1.0, stress_naive_sharpe=0.1,
        patient_reference=patient, patient_reference_naive_sharpe=-0.7,
    )
    report = ev.MhsHorizonDiagnosticReport(
        feature="multi_horizon_market_state", status="COMPLETE", start="2021-01-01",
        end="2021-01-04", resolved_end="2021-01-04", partition="dev",
        execution_tiers_bps=(2.5, 5.0), books={"fast_reversal": book}, blend=None,
        blend_target_gross=0.0, blend_cash_fraction=0.0, eligible_symbols=1,
        trials_attempted=1, deflated_sharpe_ratio=None, xs_rank_ic={},
        date_clustered_regression={}, horizon_diagnostics={}, bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=DeploymentReadinessResult(
            0.01, -0.01, 1.0, -0.01, -0.01, -0.01, -0.01, 0, None, 0.5, 0.0, 0.0, {}, {},
            {}, False, False, False, False,
        ),
        synthetic_stress={}, participation_warnings={}, termination_counts={},
        unsupported_assumptions=(), anchored_folds=(), folds=(),
        research_go=ev.MhsResearchGoResult(False, (), 0, 0),
        fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="MARK_PRICE",
        execution_timeframe="1m", execution_universe_size=1,
        execution_symbols=("A",), run_elapsed_seconds=0.1,
    )
    out = tmp_path / "mhs_report.json"
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.COMPACT)
    payload = json.loads(out.read_text())
    book_payload = payload["books"]["fast_reversal"]
    assert book_payload["patient_reference_naive_sharpe"] == -0.7
    ref = book_payload["patient_reference"]
    assert set(ref) == {"fills", "units", "notional_weights", "ledger", "times"}
    assert all(set(v) == {"row_count"} for v in ref.values())
    assert ref["ledger"]["row_count"] == len(patient.ledger.equity)
    assert "fast_reversal_patient_reference" in payload["replay_ids"]
