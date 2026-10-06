"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.fold_weights as fold_weights


def test_fold_weights_module_present() -> None:
    assert fold_weights.__name__ == "src.mhs.evaluation.fold_weights"
    assert callable(fold_weights._build_fold_target_weights)


def test_slice_base_panel_in_memory_identity() -> None:
    import pandas as pd
    import numpy as np
    from src.mhs.panel import slice_base_panel

    dates = pd.date_range("2021-01-01", periods=3000, freq="1h", tz="UTC")
    close_df = pd.DataFrame({
        "SYM1": np.linspace(100.0, 200.0, len(dates)),
        "SYM2": np.linspace(50.0, 60.0, len(dates)),
        "SYM_SHORT": [10.0] * 500 + [np.nan] * (len(dates) - 500),
    }, index=dates)
    open_df = close_df * 0.99
    base_panel = {"close": close_df, "open": open_df}

    start = dates[100]
    end = dates[2500]
    sliced = slice_base_panel(base_panel, start, end, min_bars=2000)

    assert "SYM1" in sliced["close"].columns
    assert "SYM2" in sliced["close"].columns
    assert "SYM_SHORT" not in sliced["close"].columns
    assert sliced["close"].index[0] == start
    assert sliced["close"].index[-1] == end
    np.testing.assert_allclose(
        sliced["close"]["SYM1"].to_numpy(),
        close_df.loc[start:end, "SYM1"].to_numpy(),
    )


def test_slice_base_panel_validation_errors() -> None:
    import pytest
    import pandas as pd
    from src.mhs.panel import slice_base_panel

    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = pd.Timestamp("2021-02-01", tz="UTC")

    with pytest.raises(ValueError, match="base_panel must be non-empty"):
        slice_base_panel({}, start, end)

    close_df = pd.DataFrame({"SYM1": [1.0] * 10}, index=pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC"))
    with pytest.raises(ValueError, match="no symbol survived the panel filters"):
        slice_base_panel({"close": close_df}, start, end, min_bars=100)


# --- auto appended from contract: signal_input_quarantine ---


def test_build_fold_target_weights_threads_panel_quarantine_to_loader(monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.mhs.evaluation.fold_weights as fold_weights
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.evidence import AnchoredPurgedFold
    from src.mhs.panel import PanelQuarantine

    class _StopError(Exception):
        pass

    captured: dict[str, object] = {}

    def _fake_loader(*args, **kwargs):
        captured.update(kwargs)
        raise _StopError

    monkeypatch.setattr(fold_weights, "load_base_panel", _fake_loader)
    request = MhsDiagnosticRequest()
    dt = pd.Timestamp("2026-09-05", tz="UTC")
    fold = AnchoredPurgedFold(
        train_start=dt - pd.Timedelta(days=500), train_end=dt - pd.Timedelta(days=400),
        validation_start=dt - pd.Timedelta(days=30), validation_end=dt,
        forward_dependency_hours=24, purge_hours=24,
    )
    quarantine = PanelQuarantine(protected=frozenset({"BTCUSDT"}))

    with pytest.raises(_StopError):
        fold_weights._build_fold_target_weights("root", fold, request, {}, require_minute_roster=False, panel_quarantine=quarantine)

    assert captured["quarantine"] is quarantine


def test_build_fold_target_weights_threads_request_data_policy_to_loader(monkeypatch) -> None:

    import pandas as pd
    import pytest

    import src.mhs.evaluation.fold_weights as fold_weights
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.evidence import AnchoredPurgedFold

    class _StopError(Exception):
        pass

    captured: dict[str, object] = {}

    def _fake_loader(*args, **kwargs):
        captured.update(kwargs)
        raise _StopError

    monkeypatch.setattr(fold_weights, "load_base_panel", _fake_loader)
    request = MhsDiagnosticRequest(data_policy="zombie_mask_v1")
    dt = pd.Timestamp("2026-09-05", tz="UTC")
    fold = AnchoredPurgedFold(
        train_start=dt - pd.Timedelta(days=500), train_end=dt - pd.Timedelta(days=400),
        validation_start=dt - pd.Timedelta(days=30), validation_end=dt,
        forward_dependency_hours=24, purge_hours=24,
    )

    with pytest.raises(_StopError):
        fold_weights._build_fold_target_weights("root", fold, request, {}, require_minute_roster=False)

    assert captured["data_policy"] == "zombie_mask_v1"


def _fold() -> object:
    import pandas as pd
    from src.mhs.evidence import AnchoredPurgedFold

    return AnchoredPurgedFold(
        train_start=pd.Timestamp("2021-01-01", tz="UTC"),
        train_end=pd.Timestamp("2021-04-01", tz="UTC"),
        validation_start=pd.Timestamp("2021-05-01", tz="UTC"),
        validation_end=pd.Timestamp("2021-08-01", tz="UTC"),
        forward_dependency_hours=24,
        purge_hours=24,
    )


def test_effective_window_defaults_to_validation_span() -> None:
    from src.mhs.evaluation.fold_weights import _resolve_effective_fold_window

    fold = _fold()
    assert _resolve_effective_fold_window(fold, None, None) == (fold.validation_start, fold.validation_end)


def test_effective_window_rejects_naive_bounds() -> None:
    import pandas as pd
    import pytest
    from src.mhs.evaluation.fold_weights import _resolve_effective_fold_window

    fold = _fold()
    with pytest.raises(ValueError, match="tz-aware UTC"):
        _resolve_effective_fold_window(fold, pd.Timestamp("2021-05-01"), None)
    with pytest.raises(ValueError, match="tz-aware UTC"):
        _resolve_effective_fold_window(fold, None, pd.Timestamp("2021-06-01"))


def test_effective_window_rejects_non_utc_zone() -> None:
    import pandas as pd
    import pytest
    from src.mhs.evaluation.fold_weights import _resolve_effective_fold_window

    fold = _fold()
    start = pd.Timestamp("2021-05-01", tz="Asia/Seoul")
    end = pd.Timestamp("2021-06-01", tz="Asia/Seoul")
    with pytest.raises(ValueError, match="must be UTC"):
        _resolve_effective_fold_window(fold, start, end)


def test_effective_window_rejects_pre_train_start_and_empty_span() -> None:
    import pandas as pd
    import pytest
    from src.mhs.evaluation.fold_weights import _resolve_effective_fold_window

    fold = _fold()
    with pytest.raises(ValueError, match="empty or precedes fold train_start"):
        _resolve_effective_fold_window(fold, pd.Timestamp("2020-12-01", tz="UTC"), None)
    same = pd.Timestamp("2021-05-01", tz="UTC")
    with pytest.raises(ValueError, match="empty or precedes fold train_start"):
        _resolve_effective_fold_window(fold, same, same)


def _committee_fold_panel():
    import numpy as np
    import pandas as pd

    from src.mhs.evidence import AnchoredPurgedFold

    idx = pd.date_range("2021-01-01", periods=2000, freq="1h", tz="UTC")
    cols = [f"SYM{i}USDT" for i in range(9)] + ["BTCUSDT"]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.1, (2000, 10)), axis=0), index=idx, columns=cols,
    )
    quote_vol = pd.DataFrame(1e6, index=idx, columns=cols)
    base_panel = {
        "close": close,
        "open": close.copy(),
        "quote_vol": quote_vol,
        "taker_buy_quote": quote_vol * 0.55,
    }
    fold = AnchoredPurgedFold(idx[0], idx[100], idx[800], idx[1800], 24, 24)
    funding = {c: pd.Series(0.0, index=idx) for c in cols}
    return base_panel, fold, funding


def _run_fold_target(request, base_panel, fold, funding):
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    return _build_fold_target_weights(
        "root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24,
    )


def test_committee_fold_book_invariant_to_fast_slow_only_knobs(monkeypatch) -> None:
    import dataclasses

    import pandas as pd

    from tests.fixtures.mhs_requests import research_baseline

    def _unexpected_momentum_computation(*args, **kwargs):
        raise AssertionError("committee folds must skip unused momentum computations")

    for name in ("_book_weights", "_horizon_ensemble_execution_weights"):
        monkeypatch.setattr(fold_weights.books, name, _unexpected_momentum_computation)
    monkeypatch.setattr(fold_weights.specs, "_signal_ema_span", _unexpected_momentum_computation)
    for name in (
        "inverse_realized_vol_tilt", "renormalize_within_mask",
        "beta_neutralize_weights", "crash_regime_tilt_weights",
    ):
        monkeypatch.setattr(fold_weights, name, _unexpected_momentum_computation)

    base_panel, fold, funding = _committee_fold_panel()
    base_request = research_baseline(committee_capital=True)
    base = _run_fold_target(base_request, base_panel, fold, funding)
    assert base[0].ne(0.0).any().any()
    for knob in (
        {"fast_book_mode": "horizon_ensemble"},
        {"slow_book_mode": "horizon_ensemble"},
        {"ensemble_signal": "vol_normalized"},
        {"crash_regime_tilt_alpha": 0.5},
    ):
        variant = _run_fold_target(
            dataclasses.replace(base_request, **knob), base_panel, fold, funding,
        )
        pd.testing.assert_frame_equal(variant[0], base[0], check_exact=True)
        pd.testing.assert_index_equal(variant[1], base[1], exact=True)
        assert variant[2] == base[2]
        pd.testing.assert_index_equal(variant[3], base[3], exact=True)

    neutralized = _run_fold_target(
        dataclasses.replace(base_request, beta_neutralize=True), base_panel, fold, funding,
    )
    assert not base[0].equals(neutralized[0])


def test_committee_fold_book_still_honors_beta_neutralization() -> None:
    import dataclasses

    from tests.fixtures.mhs_requests import research_baseline

    base_panel, fold, funding = _committee_fold_panel()
    base_request = research_baseline(committee_capital=True)
    base = _run_fold_target(base_request, base_panel, fold, funding)
    neutralized = _run_fold_target(
        dataclasses.replace(base_request, beta_neutralize=True), base_panel, fold, funding,
    )
    assert not base[0].equals(neutralized[0])


def test_non_committee_fold_book_still_uses_fast_slow_knobs(monkeypatch) -> None:
    import dataclasses

    import src.mhs.evaluation.books as books_mod
    from src.mhs.types import BOOK_SPECS
    from tests.fixtures.mhs_requests import research_baseline

    base_panel, fold, funding = _committee_fold_panel()
    base_request = research_baseline()
    base = _run_fold_target(base_request, base_panel, fold, funding)
    for knob in (
        {"slow_book_mode": "horizon_ensemble"},
        {"beta_neutralize": True},
        {"crash_regime_tilt_alpha": 0.5},
    ):
        variant = _run_fold_target(
            dataclasses.replace(base_request, **knob),
            base_panel, fold, funding,
        )
        assert not base[0].equals(variant[0]), knob

    real = books_mod._horizon_ensemble_execution_weights
    seen_specs: list = []

    def _spy(log_close, eligible, execution_mask, spec, grid, *args, **kwargs):
        seen_specs.append(spec)
        return real(log_close, eligible, execution_mask, spec, grid, *args, **kwargs)

    monkeypatch.setattr(books_mod, "_horizon_ensemble_execution_weights", _spy)
    _run_fold_target(
        dataclasses.replace(base_request, fast_book_mode="horizon_ensemble"),
        base_panel, fold, funding,
    )
    assert BOOK_SPECS["fast_reversal"] in seen_specs


def test_non_committee_fold_preserves_execution_order(monkeypatch) -> None:
    from tests.fixtures.mhs_requests import research_baseline

    calls = []

    def _record(module, name, label):
        original = getattr(module, name)

        def _spy(*args, **kwargs):
            calls.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, _spy)

    _record(fold_weights.books, "_book_weights", "fast_book")
    _record(fold_weights, "_pit_execution_mask", "execution_mask")
    _record(fold_weights, "inverse_realized_vol_tilt", "fast_tilt")
    _record(fold_weights, "renormalize_within_mask", "fast_execution")
    _record(fold_weights.books, "_horizon_ensemble_execution_weights", "slow_execution")
    _record(fold_weights, "causal_market_beta", "causal_beta")
    _record(fold_weights, "beta_neutralize_weights", "slow_neutralize")
    _record(fold_weights.folds, "_trend_sleeve_position", "trend_position")
    _record(fold_weights, "crash_regime_tilt_weights", "crash_tilt")

    base_panel, fold, funding = _committee_fold_panel()
    request = research_baseline(
        beta_neutralize=True, trend_sleeve=True, trend_sleeve_gross=0.1,
        crash_regime_tilt_alpha=0.5,
    )
    result = _run_fold_target(request, base_panel, fold, funding)
    assert result[0].ne(0.0).any().any()
    assert calls == [
        "fast_book", "execution_mask", "fast_tilt", "fast_execution",
        "slow_execution", "causal_beta", "slow_neutralize", "trend_position", "crash_tilt",
    ]
