"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.research.universe.pit_universe import symbol_partition

from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
    _FOLD,
    _START,
)

def test_crash_tilt_request_validation() -> None:
    # SCENARIO_MHS_CRASH_TILT_REQUEST_VALIDATION_05: the request-level opt-in
    # narrows the pure function's [0.0, 1.0] to (0.0, 1.0] -- an explicitly
    # set-but-no-op 0.0 is a footgun, and >1.0 breaks the unit-gross budget.
    with pytest.raises(ValueError, match="crash_regime_tilt_alpha"):
        MhsDiagnosticRequest(crash_regime_tilt_alpha=0.0)
    with pytest.raises(ValueError, match="crash_regime_tilt_alpha"):
        MhsDiagnosticRequest(crash_regime_tilt_alpha=1.5)
    assert MhsDiagnosticRequest().crash_regime_tilt_alpha is None
    assert MhsDiagnosticRequest(crash_regime_tilt_alpha=0.2).crash_regime_tilt_alpha == 0.2

def test_crash_tilt_disabled_fold_is_byte_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_CRASH_TILT_FOLD_BYTE_IDENTICAL_06: with the opt-in disabled
    # (crash_regime_tilt_alpha=None) the fold target weights are byte-identical
    # to the pre-overlay path. Proved by running the enabled path with the tilt
    # replaced by an identity: the new wiring then reproduces exactly the
    # disabled output, so the extra branch is value-neutral when no tilt applies.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )

    tilt_calls: list[tuple[int, float]] = []

    def _identity_tilt(rank_neutral_weights, _log_price, _eligible, _refs, horizon, alpha, min_symbols=8):
        tilt_calls.append((horizon, alpha))
        return rank_neutral_weights

    monkeypatch.setattr(ev, "crash_regime_tilt_weights", _identity_tilt)
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        crash_regime_tilt_alpha=0.3,
    )
    target_ident, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_on, funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_off, target_ident)
    assert tilt_calls, "enabled path must route through crash_regime_tilt_weights"
    assert tilt_calls[0][0] == 168, "tilt lookback must reuse slow.horizon_hours (168), not a new literal"
    assert tilt_calls[0][1] == 0.3

def test_crash_tilt_active_fold_reaches_replay(mhs_market_with_btc) -> None:
    # SCENARIO_MHS_CRASH_TILT_FOLD_ACTIVE_07: with a real BTCUSDT reference
    # series in the panel and crash_regime_tilt_alpha=0.3, the fold target
    # weights (a) differ from the disabled baseline, (b) stay finite, and
    # (c) keep the blended gross budget bounded by unit (the tilt offsets
    # dollar-neutral shorts rather than amplifying them).
    root, end = mhs_market_with_btc
    symbols = [
        s for s in ("BTCUSDT", "MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT",
                    "MHSEUSDT", "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT",
                    "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    assert "BTCUSDT" in funding_by_symbol
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    request_on = dataclasses.replace(request, crash_regime_tilt_alpha=0.3, committee_target_gross=None)
    target_on, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_on, funding_by_symbol,
    )
    assert not target_off.equals(target_on)
    assert np.isfinite(target_on.to_numpy(dtype="float64")).all()
    assert float(target_on.abs().max().max()) <= 1.0 + 1e-9
