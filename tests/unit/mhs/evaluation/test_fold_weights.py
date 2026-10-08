"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.fold_weights as fold_weights


def test_rebalance_trigger_holds_unscaled_book_before_regime_scaling():
    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest

    dates = pd.date_range("2027-01-01", periods=3, freq="h", tz="UTC")
    weights = pd.DataFrame({"LONG": [0.5, 0.51, 0.75], "SHORT": [-0.5, -0.51, -0.75]}, index=dates)
    scale = pd.Series([1.0, 0.1, 0.2], index=dates)
    result = fold_weights._rebalance_fold_weights(
        weights, scale, MhsDiagnosticRequest(rebalance_filter="portfolio_trigger"), True, None,
    )
    expected = pd.DataFrame({"LONG": [0.5, 0.05, 0.15], "SHORT": [-0.5, -0.05, -0.15]}, index=dates)
    pd.testing.assert_frame_equal(result, expected)
    assert result.sum(axis=1).eq(0).all()


def test_rebalance_without_deadband_preserves_scaled_targets():
    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest

    dates = pd.date_range("2027-01-01", periods=2, freq="h", tz="UTC")
    weights = pd.DataFrame({"LONG": [0.5, 0.51], "SHORT": [-0.5, -0.51]}, index=dates)
    scale = pd.Series([1.0, 0.5], index=dates)
    result = fold_weights._rebalance_fold_weights(
        weights, scale, MhsDiagnosticRequest(rebalance_filter="per_symbol_deadband"), False, weights.iloc[0],
    )
    expected = pd.DataFrame({"LONG": [0.5, 0.255], "SHORT": [-0.5, -0.255]}, index=dates)
    pd.testing.assert_frame_equal(result, expected)


def test_portfolio_trigger_rejects_per_symbol_seed():
    import pandas as pd
    import pytest

    from src.mhs.contracts import MhsDiagnosticRequest

    dates = pd.date_range("2027-01-01", periods=1, freq="h", tz="UTC")
    weights = pd.DataFrame({"BTCUSDT": [1.0]}, index=dates)
    with pytest.raises(ValueError, match="deadband_seed_row requires"):
        fold_weights._rebalance_fold_weights(
            weights, pd.Series([1.0], index=dates), MhsDiagnosticRequest(rebalance_filter="portfolio_trigger"),
            True, weights.iloc[0],
        )


def test_fold_weights_module_present() -> None:
    assert fold_weights.__name__ == "src.mhs.evaluation.fold_weights"
    assert callable(fold_weights._build_fold_target_weights)


def test_slice_base_panel_in_memory_identity() -> None:
    import pandas as pd
    import numpy as np
    from src.core.panel import slice_base_panel

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
    from src.core.panel import slice_base_panel

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
    from src.core.panel import PanelQuarantine

    class _StopError(Exception):
        pass

    captured: dict[str, object] = {}

    def _fake_loader(*args, **kwargs):
        captured.update(kwargs)
        raise _StopError

    monkeypatch.setattr(fold_weights, "load_base_panel", _fake_loader)
    request = MhsDiagnosticRequest()
    from src.mhs import research_go as _research_go
    from src.strategy.features import FeatureAdmission as _FeatureAdmission

    dt = pd.Timestamp("2026-09-05", tz="UTC")
    _admission = (
        _FeatureAdmission(
            dt - pd.Timedelta(days=400), _research_go._resolved_committee_members(request),
        )
        if request.committee_capital
        else None
    )
    fold = AnchoredPurgedFold(
        train_start=dt - pd.Timedelta(days=500), train_end=dt - pd.Timedelta(days=400),
        validation_start=dt - pd.Timedelta(days=30), validation_end=dt,
        forward_dependency_hours=24, purge_hours=24,
    )
    quarantine = PanelQuarantine(protected=frozenset({"BTCUSDT"}))

    with pytest.raises(_StopError):
        fold_weights._build_fold_target_weights("root", fold, request, {}, require_minute_roster=False, panel_quarantine=quarantine, committee_admission=_admission)

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
    from src.mhs import research_go as _research_go_dp
    from src.strategy.features import FeatureAdmission as _FeatureAdmissionDp

    dt = pd.Timestamp("2026-09-05", tz="UTC")
    _admission_dp = (
        _FeatureAdmissionDp(
            dt - pd.Timedelta(days=400), _research_go_dp._resolved_committee_members(request),
        )
        if request.committee_capital
        else None
    )
    fold = AnchoredPurgedFold(
        train_start=dt - pd.Timedelta(days=500), train_end=dt - pd.Timedelta(days=400),
        validation_start=dt - pd.Timedelta(days=30), validation_end=dt,
        forward_dependency_hours=24, purge_hours=24,
    )

    with pytest.raises(_StopError):
        fold_weights._build_fold_target_weights("root", fold, request, {}, require_minute_roster=False, committee_admission=_admission_dp)

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
    from src.mhs import research_go as _research_go
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights
    from src.strategy.features import FeatureAdmission

    admission = (
        FeatureAdmission(fold.train_end, _research_go._resolved_committee_members(request))
        if request.committee_capital
        else None
    )
    return _build_fold_target_weights(
        "root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24,
        committee_admission=admission,
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
    from src.core.types import BOOK_SPECS
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


# --- I-FOLD-ADMISSION-PIT: train-end-frozen committee fold admission ---------
# Fixture per docs/specs/28_committee_fold_admission_pit_spec.md: in-memory
# full-history synthetic panel (40 symbols, t(4) returns, taker_buy_quote),
# long enough for train + purge + >=600h validation + FOLD_PANEL_WARMUP_HOURS.
# Admission comes from production code on the full-history panels; targets are
# built with base_panel=<full panel> and require_minute_roster=False.
# T is the midpoint of the validation window floored to the day.

_PIT_N_ROWS = 3600
_PIT_TRAIN_END_POS = 2000
_PIT_VALIDATION_START_POS = 2100
_PIT_VALIDATION_END_POS = 2820  # 720h validation window
_PIT_SYMBOLS = tuple(f"SYM{i:02d}" for i in range(40))


def _pit_market(seed: int = 0):
    import numpy as np
    import pandas as pd

    from src.mhs.evidence import AnchoredPurgedFold
    from tests.fixtures.mhs_requests import research_baseline

    idx = pd.date_range("2021-01-01", periods=_PIT_N_ROWS, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    rets = rng.standard_t(df=4, size=(_PIT_N_ROWS, len(_PIT_SYMBOLS))) * 0.005
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list(_PIT_SYMBOLS),
    )
    quote_vol = pd.DataFrame(
        1e6 * (1.0 + 0.05 * rng.standard_normal((_PIT_N_ROWS, len(_PIT_SYMBOLS)))),
        index=idx, columns=list(_PIT_SYMBOLS),
    ).clip(lower=1e5)
    taker_buy_quote = quote_vol * 0.55
    panel = {
        "close": close,
        "open": close.copy(),
        "quote_vol": quote_vol,
        "taker_buy_quote": taker_buy_quote,
    }
    request = research_baseline(committee_capital=True, committee_member_set="flow_momentum")
    fold = AnchoredPurgedFold(
        idx[0],
        idx[_PIT_TRAIN_END_POS],
        idx[_PIT_VALIDATION_START_POS],
        idx[_PIT_VALIDATION_END_POS],
        24,
        24,
    )
    funding = {s: pd.Series(0.0, index=idx) for s in _PIT_SYMBOLS}
    midpoint = fold.validation_start + (fold.validation_end - fold.validation_start) / 2
    decision_t = midpoint.floor("D")
    return panel, fold, funding, request, decision_t


def _pit_admission(panel, fold, request):
    import pandas as pd

    from src.mhs import research_go as _research_go
    from src.mhs.evaluation.committee import _committee_boundary_admission_and_weights
    from src.core.marks import _pit_execution_mask
    from src.core.panel import liquid_half_eligibility
    from src.core.params import (
        UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )

    eligible = liquid_half_eligibility(
        panel["quote_vol"],
        lookback_bars=UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        min_history_bars=UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )
    execution_mask = _pit_execution_mask(
        panel["quote_vol"], eligible, request.execution_universe_size,
    )
    decision_grid = pd.date_range(
        panel["close"].index[0], panel["close"].index[-1], freq="24h", tz="UTC",
    )
    admission_by_label, _ = _committee_boundary_admission_and_weights(
        panel["close"], panel["quote_vol"], panel["taker_buy_quote"],
        execution_mask, decision_grid, 8, {"fold": fold.train_end},
        members=_research_go._resolved_committee_members(request),
        evidence_weighting=False,
    )
    return admission_by_label["fold"]


def _pit_targets(panel, fold, funding, request, admission, **kwargs):
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    return _build_fold_target_weights(
        "root", fold, request, funding,
        base_panel=panel, require_minute_roster=False,
        committee_admission=admission, **kwargs,
    )


def test_committee_fold_requires_admission() -> None:
    import pytest

    from src.mhs.evaluation.fold_weights import _build_fold_target_weights
    from src.mhs.evaluation.integrity import CommitteeAdmissionIntegrityError

    _, fold, funding, request, _ = _pit_market()
    # Fails closed before any panel I/O: root does not exist and there is no
    # base_panel, so a load error would surface first without the guard.
    with pytest.raises(CommitteeAdmissionIntegrityError):
        _build_fold_target_weights(
            "/nonexistent/panel/root", fold, request, funding,
            base_panel=None, require_minute_roster=False, committee_admission=None,
        )


def test_admission_cutoff_after_validation_start_fails_closed() -> None:
    import pandas as pd
    import pytest

    from src.mhs.evaluation.integrity import CommitteeAdmissionIntegrityError
    from src.strategy.features import FeatureAdmission

    panel, fold, funding, request, _ = _pit_market()
    admission = _pit_admission(panel, fold, request)
    late = FeatureAdmission(
        fold.validation_start + pd.Timedelta(hours=1), admission.admitted,
    )
    with pytest.raises(CommitteeAdmissionIntegrityError):
        _pit_targets(panel, fold, funding, request, late)


def test_admission_cutoff_must_equal_train_end() -> None:
    import pandas as pd
    import pytest

    from src.mhs.evaluation.integrity import CommitteeAdmissionIntegrityError
    from src.strategy.features import FeatureAdmission

    panel, fold, funding, request, _ = _pit_market()
    admission = _pit_admission(panel, fold, request)
    # PIT (before validation_start) but a different boundary than the one that
    # fit the member weights: still rejected (I-COVERAGE-PIT).
    other_boundary = FeatureAdmission(
        fold.train_end - pd.Timedelta(hours=24), admission.admitted,
    )
    with pytest.raises(CommitteeAdmissionIntegrityError):
        _pit_targets(panel, fold, funding, request, other_boundary)
    # cutoff == train_end succeeds.
    targets, *_ = _pit_targets(panel, fold, funding, request, admission)
    assert not targets.empty
    assert targets.ne(0.0).any().any()


def test_admission_without_committee_capital_rejected() -> None:
    import pytest

    from tests.fixtures.mhs_requests import research_baseline

    panel, fold, funding, _, _ = _pit_market()
    plain_request = research_baseline()
    assert not plain_request.committee_capital
    committee_request_panel, _, _, committee_request, _ = _pit_market()
    admission = _pit_admission(committee_request_panel, fold, committee_request)
    with pytest.raises(ValueError, match="committee_admission"):
        _pit_targets(panel, fold, funding, plain_request, admission)


def test_fold_targets_invariant_to_post_t_source_outage() -> None:
    import numpy as np
    import pandas as pd

    panel, fold, funding, request, decision_t = _pit_market()
    admission = _pit_admission(panel, fold, request)
    base_targets, *_ = _pit_targets(panel, fold, funding, request, admission)

    perturbed = {name: frame.copy() for name, frame in panel.items()}
    post_t = perturbed["taker_buy_quote"].index > decision_t
    perturbed["taker_buy_quote"].loc[post_t, list(_PIT_SYMBOLS[:20])] = np.nan

    perturbed_admission = _pit_admission(perturbed, fold, request)
    assert perturbed_admission == admission
    perturbed_targets, *_ = _pit_targets(
        perturbed, fold, funding, request, perturbed_admission,
    )

    base_pre = base_targets[base_targets.index <= decision_t]
    perturbed_pre = perturbed_targets[perturbed_targets.index <= decision_t]
    assert len(base_pre) > 0
    assert base_pre.ne(0.0).any().any()
    pd.testing.assert_frame_equal(perturbed_pre, base_pre, check_exact=True)

    base_post = base_targets[base_targets.index > decision_t]
    perturbed_post = perturbed_targets[perturbed_targets.index > decision_t]
    assert len(base_post) > 0
    assert not perturbed_post.equals(base_post)


def test_fold_targets_invariant_to_truncation_at_t() -> None:
    import pandas as pd

    panel, fold, funding, request, decision_t = _pit_market()
    admission = _pit_admission(panel, fold, request)
    full_targets, *_ = _pit_targets(panel, fold, funding, request, admission)
    truncated_targets, *_ = _pit_targets(
        panel, fold, funding, request, admission, decision_end=decision_t,
    )
    pd.testing.assert_frame_equal(
        truncated_targets, full_targets[full_targets.index <= decision_t],
        check_exact=True,
    )


def test_fold_targets_invariant_to_post_t_value_corruption() -> None:
    import numpy as np
    import pandas as pd

    panel, fold, funding, request, decision_t = _pit_market()
    admission = _pit_admission(panel, fold, request)
    base_targets, *_ = _pit_targets(panel, fold, funding, request, admission)

    rng = np.random.default_rng(1234)
    corrupted = {name: frame.copy() for name, frame in panel.items()}
    for name, frame in corrupted.items():
        post_t = frame.index > decision_t
        noise = np.exp(0.01 * rng.standard_normal((int(post_t.sum()), frame.shape[1])))
        corrupted[name].loc[post_t] = frame.loc[post_t] * noise

    corrupted_admission = _pit_admission(corrupted, fold, request)
    assert corrupted_admission == admission
    corrupted_targets, *_ = _pit_targets(
        corrupted, fold, funding, request, corrupted_admission,
    )
    pd.testing.assert_frame_equal(
        corrupted_targets[corrupted_targets.index <= decision_t],
        base_targets[base_targets.index <= decision_t],
        check_exact=True,
    )


def _execution_source_fold_market():
    import numpy as np
    import pandas as pd

    from src.mhs.evidence import AnchoredPurgedFold

    idx = pd.date_range("2021-01-01", periods=2000, freq="1h", tz="UTC")
    cols = [f"SYM{i}USDT" for i in range(10)]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.1, (2000, 10)), axis=0), index=idx, columns=cols,
    )
    quote_vol = pd.DataFrame(1e6, index=idx, columns=cols)
    base_panel = {"close": close, "open": close.copy(), "quote_vol": quote_vol}
    fold = AnchoredPurgedFold(idx[0], idx[100], idx[800], idx[1800], 24, 24)
    funding = {c: pd.Series(0.0, index=idx) for c in cols}
    return base_panel, fold, funding


def _first_targeted_symbol(target_weights):
    for col in target_weights.columns:
        series = target_weights[col]
        if bool((series.notna() & series.ne(0.0)).any()):
            return col
    raise AssertionError("fixture must target at least one symbol")


def _fold_request():
    from tests.fixtures.mhs_requests import research_baseline

    return research_baseline(execution_timeframe="3m")


def test_fold_targeted_symbol_without_execution_source_fails_closed(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    base_panel, fold, funding = _execution_source_fold_market()
    request = _fold_request()
    probe, *_ = _build_fold_target_weights(
        "root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24,
    )
    victim = _first_targeted_symbol(probe)
    execution_symbols = sorted(probe.columns[probe.ne(0.0).any(axis=0)])
    assert victim in execution_symbols
    three = tmp_path / "3m"
    three.mkdir(parents=True)
    for sym in execution_symbols:
        if sym != victim:
            (three / f"{sym}.parquet").touch()

    with pytest.raises(DataIntegrityError, match=r"fold execution source missing for 1 targeted symbol") as excinfo:
        _build_fold_target_weights(
            str(tmp_path), fold, request, funding,
            base_panel=base_panel, require_minute_roster=True, panel_warmup_hours=24,
        )
    exc = excinfo.value
    assert str(exc).startswith("fold execution source missing for 1 targeted symbol(s)")
    assert victim in str(exc)
    assert "decision_window=" in str(exc)
    assert fold.validation_start.isoformat() in str(exc)


def test_fold_missing_source_maps_to_execution_gap_reason(tmp_path, monkeypatch) -> None:
    from src.mhs.evaluation import folds

    base_panel, fold, funding = _execution_source_fold_market()
    request = _fold_request()
    monkeypatch.setattr(fold_weights, "load_base_panel", lambda *_a, **_k: base_panel)
    targets, *_ = fold_weights._build_fold_target_weights(
        str(tmp_path), fold, request, funding, require_minute_roster=False,
    )
    victim = _first_targeted_symbol(targets)
    three = tmp_path / "3m"
    three.mkdir()
    for symbol in targets.columns:
        if symbol != victim:
            (three / f"{symbol}.parquet").touch()

    def _unexpected_reference(*_a, **_k):
        raise AssertionError("missing validation source must fail before train replay")

    monkeypatch.setattr(folds, "_fold_train_reference_returns", _unexpected_reference)
    report = folds._run_anchored_fold(str(tmp_path), fold, request, funding, 1.0, 0)
    assert report.strict is None
    assert report.stress is None
    assert report.primary_valid is False
    assert report.failures == ("RELEVANT_EXECUTION_DATA_GAP",)


def test_fold_without_any_execution_source_keeps_incomplete_outcome(tmp_path) -> None:
    import pytest

    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    base_panel, fold, funding = _execution_source_fold_market()
    with pytest.raises(RuntimeError, match="no fold decision symbol has minute execution data"):
        _build_fold_target_weights(
            str(tmp_path), fold, _fold_request(), funding,
            base_panel=base_panel, require_minute_roster=True, panel_warmup_hours=24,
        )


def test_fold_builder_without_minute_roster_requirement_is_unchanged(tmp_path) -> None:
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    base_panel, fold, funding = _execution_source_fold_market()
    request = _fold_request()
    probe, *_ = _build_fold_target_weights(
        "root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24,
    )
    victim = _first_targeted_symbol(probe)
    execution_symbols = sorted(probe.columns[probe.ne(0.0).any(axis=0)])
    three = tmp_path / "3m"
    three.mkdir(parents=True)
    for sym in execution_symbols:
        if sym != victim:
            (three / f"{sym}.parquet").touch()

    target_weights, _signals, minute_roster, _grid = _build_fold_target_weights(
        str(tmp_path), fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24,
    )
    assert victim not in minute_roster
    assert minute_roster == [s for s in execution_symbols if s != victim]


def test_fold_with_all_execution_sources_present_is_unchanged(tmp_path) -> None:
    import pandas as pd

    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    base_panel, fold, funding = _execution_source_fold_market()
    request = _fold_request()
    reference, ref_signals, _ref_roster, _ref_grid = _build_fold_target_weights(
        "root", fold, request, funding,
        base_panel=base_panel, require_minute_roster=False, panel_warmup_hours=24,
    )
    execution_symbols = sorted(reference.columns[reference.ne(0.0).any(axis=0)])
    three = tmp_path / "3m"
    three.mkdir(parents=True)
    for sym in execution_symbols:
        (three / f"{sym}.parquet").touch()

    target_weights, signals, minute_roster, _grid = _build_fold_target_weights(
        str(tmp_path), fold, request, funding,
        base_panel=base_panel, require_minute_roster=True, panel_warmup_hours=24,
    )
    assert minute_roster == execution_symbols
    pd.testing.assert_frame_equal(target_weights, reference, check_exact=True)
    pd.testing.assert_index_equal(signals, ref_signals, exact=True)


def test_shared_train_reference_falls_back_on_missing_source(monkeypatch) -> None:
    import pandas as pd
    import pytest

    import src.mhs.evaluation.folds as folds_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.evidence import AnchoredPurgedFold

    base_panel, fold, funding = _execution_source_fold_market()
    request = _fold_request()
    other = AnchoredPurgedFold(
        fold.train_start, fold.train_start + pd.Timedelta(hours=900),
        fold.train_start + pd.Timedelta(hours=1000), fold.train_start + pd.Timedelta(hours=1900), 24, 24,
    )

    def _boom(*_args, **_kwargs):
        raise DataIntegrityError("fold execution source missing for 1 targeted symbol(s) [SYM0USDT]")

    monkeypatch.setattr(folds_mod, "_fold_reference_targets", _boom)
    assert folds_mod._build_shared_train_reference(
        "root", ((0, fold), (1, other)), request, funding, 1.0, None, None,
    ) is None
    with pytest.raises(ValueError, match="at least two folds"):
        folds_mod._build_shared_train_reference("root", ((0, fold),), request, funding, 1.0, None, None)


def test_fold_executes_boundary_admission_verbatim() -> None:
    import pytest

    from src.mhs.evaluation.integrity import CommitteeAdmissionIntegrityError
    from src.strategy.features import FeatureAdmission

    panel, fold, funding, request, _ = _pit_market()
    admission = _pit_admission(panel, fold, request)
    # The fold panel (912h warmup) is shorter than the 1006-row warmup of
    # xs_idio_mom_336h, so the old in-window audit always dropped it; the
    # train-end-frozen boundary admission keeps it.
    assert "xs_idio_mom_336h" in admission.admitted
    base_targets, *_ = _pit_targets(panel, fold, funding, request, admission)

    reduced = FeatureAdmission(
        admission.cutoff,
        tuple(name for name in admission.admitted if name != "xs_idio_mom_336h"),
    )
    assert len(reduced.admitted) == len(admission.admitted) - 1
    reduced_targets, *_ = _pit_targets(panel, fold, funding, request, reduced)
    assert not reduced_targets.equals(base_targets)

    mismatched_weights = dict.fromkeys(reduced.admitted, 1.0)
    assert set(mismatched_weights) != set(admission.admitted)
    with pytest.raises(CommitteeAdmissionIntegrityError):
        _pit_targets(
            panel, fold, funding, request, admission,
            committee_member_weights=mismatched_weights,
        )
