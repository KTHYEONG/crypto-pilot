"""SCENARIO_SIGNAL_03-08/11: incremental signal refresh forward scorer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.application.research.mhs.evaluation as evaluation_mod
import src.mhs.signal_refresh as signal_refresh_mod
from src.common.errors import DataIntegrityError
from src.mhs.params import (
    COMMITTEE_OOS_START,
    FOLD_PANEL_WARMUP_HOURS,
    PNL_VOL_TARGET_BURN_IN_DAYS,
    PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
    SIGNAL_PANEL_WINDOW_DAYS,
    SIGNAL_REPLAY_WARMUP_DAYS,
    SIGNAL_RETURN_TAIL_DAYS,
)
from src.mhs.signal_refresh import refresh_signal_row
from src.mhs.signal_state import (
    FrozenSignalParams,
    SignalState,
    compute_flags_digest,
    compute_params_digest,
    save_signal_state,
)

D0 = pd.Timestamp("2026-08-24 00:00Z")
D1 = pd.Timestamp("2026-08-25 00:00Z")

_FLAGS = {
    "committee_capital": True,
    "committee_evidence_weighting": True,
    "committee_kelly_sizing": True,
    "committee_member_set": "flow_momentum",
    "committee_regime_adaptive_tranche": True,
    "committee_tranche_smoothing": False,
    "committee_target_gross": 0.92,
    "beta_neutralize": False,
    "trend_sleeve": False,
    "trend_sleeve_gross": 0.0,
    "trend_efficiency_overlay": False,
    "rebalance_filter": "per_symbol_deadband",
    "fast_book_mode": "single_horizon",
    "slow_book_mode": "single_horizon",
    "ensemble_signal": "raw",
    "execution_universe_size": 60,
    "funding_carry_sleeve": True,
    "funding_carry_weight": 0.3,
    "pnl_vol_target_mode": "growth_budget",
    "growth_envelope": "growth_extreme_budgeted",
    "exposure_scale_two_sided": True,
    "exposure_drawdown_brake": False,
    "fill_mark_parity_gate": True,
}

_ADMITTED = ("flow_imb_720h", "flow_imb_168h", "xs_mom_336h", "xs_idio_mom_336h", "mom3_skew_168h")


def _state(**overrides) -> SignalState:
    frozen = FrozenSignalParams(
        slow_horizon_hours=168,
        committee_member_weights=dict.fromkeys(_ADMITTED, 0.2),
        admitted_members=_ADMITTED,
        growth_budget_target_vol=0.4,
        exposure_cap=3.0,
        growth_envelope="growth_extreme_budgeted",
        execution_universe_size=60,
        pnl_vol_target_mode="growth_budget",
        deployed_flags=dict(_FLAGS),
    )
    defaults = {
        "schema_version": 1,
        "params_digest": compute_params_digest(),
        "flags_digest": compute_flags_digest(_FLAGS),
        "frozen": frozen,
        "last_decision_time": D0,
        "held_target_row": {"AAAUSDT": 0.02, "BUSDT": -0.02},
        "reference_daily_returns": pd.Series(
            [0.001, -0.0005],
            index=pd.DatetimeIndex([D0 - pd.Timedelta(days=2), D0 - pd.Timedelta(days=1)]),
        ),
    }
    defaults.update(overrides)
    return SignalState(**defaults)


def _seed_files(tmp_path: Path, state: SignalState, artifact_rows: pd.DataFrame | None):
    state_path = tmp_path / "signal_state.json"
    save_signal_state(state_path, state)
    artifact_path = tmp_path / "deployed_target_weights.parquet"
    if artifact_rows is not None:
        artifact_rows.to_parquet(artifact_path, index=True)
    return state_path, artifact_path


def _patch_no_replay(monkeypatch, *, exposure_scale: float = 1.0) -> None:
    monkeypatch.setattr(signal_refresh_mod, "_load_funding_by_symbol", lambda *a, **k: {})
    monkeypatch.setattr(signal_refresh_mod, "_rolling_reference_returns", lambda *a, **k: pd.Series(dtype="float64"))
    monkeypatch.setattr(signal_refresh_mod, "_resolve_exposure_scale", lambda *a, **k: exposure_scale)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_SCENARIO_SIGNAL_03_REFRESH_APPENDS_EXACTLY_ONE_ROW(tmp_path, monkeypatch) -> None:
    _patch_no_replay(monkeypatch)

    published = pd.DataFrame({"AAAUSDT": [0.02], "BUSDT": [-0.02]}, index=pd.DatetimeIndex([D0]))
    state_path, artifact_path = _seed_files(tmp_path, _state(), published)

    def fake_builder(root, fold, request, funding, *, slow_horizon_override, committee_member_weights, deadband_seed_row):
        idx = pd.DatetimeIndex([D0, D1])
        frame = pd.DataFrame({"AAAUSDT": [0.02, 0.025], "BUSDT": [-0.02, -0.015]}, index=idx)
        return frame, idx + pd.Timedelta(hours=1), [], idx

    monkeypatch.setattr(evaluation_mod, "_build_fold_target_weights", fake_builder)

    report = refresh_signal_row(state_path, artifact_path, D1)
    assert report.status == "APPENDED"

    result = pd.read_parquet(artifact_path)
    assert list(result.index) == [D0, D1]
    assert result.loc[D0, "AAAUSDT"] == pytest.approx(0.02)
    assert result.loc[D1, "AAAUSDT"] == pytest.approx(0.025)


def test_SCENARIO_SIGNAL_04_SAME_DAY_RERUN_IS_NOOP(tmp_path, monkeypatch) -> None:
    _patch_no_replay(monkeypatch)
    published = pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([D0]))
    state_path, artifact_path = _seed_files(tmp_path, _state(), published)

    def fake_builder(root, fold, request, funding, *, slow_horizon_override, committee_member_weights, deadband_seed_row):
        idx = pd.DatetimeIndex([D0, D1])
        frame = pd.DataFrame({"AAAUSDT": [0.02, 0.03]}, index=idx)
        return frame, idx + pd.Timedelta(hours=1), [], idx

    monkeypatch.setattr(evaluation_mod, "_build_fold_target_weights", fake_builder)

    first = refresh_signal_row(state_path, artifact_path, D1)
    assert first.status == "APPENDED"
    artifact_sha_1 = _sha256(artifact_path)
    state_sha_1 = _sha256(state_path)

    second = refresh_signal_row(state_path, artifact_path, D1)
    assert second.status == "NOOP"
    assert _sha256(artifact_path) == artifact_sha_1
    assert _sha256(state_path) == state_sha_1

    third = refresh_signal_row(state_path, artifact_path, D0)
    assert third.status == "NOOP"
    assert _sha256(artifact_path) == artifact_sha_1
    assert _sha256(state_path) == state_sha_1


def test_SCENARIO_SIGNAL_05_OVERLAP_DIVERGENCE_FAILS_CLOSED(tmp_path, monkeypatch) -> None:
    _patch_no_replay(monkeypatch)
    published = pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([D0]))
    state_path, artifact_path = _seed_files(tmp_path, _state(), published)
    artifact_sha_before = _sha256(artifact_path)
    state_sha_before = _sha256(state_path)

    def fake_builder_diverges(root, fold, request, funding, *, slow_horizon_override, committee_member_weights, deadband_seed_row):
        idx = pd.DatetimeIndex([D0, D1])
        frame = pd.DataFrame({"AAAUSDT": [0.02 + 1e-3, 0.03]}, index=idx)
        return frame, idx + pd.Timedelta(hours=1), [], idx

    monkeypatch.setattr(evaluation_mod, "_build_fold_target_weights", fake_builder_diverges)
    with pytest.raises(DataIntegrityError, match="AAAUSDT"):
        refresh_signal_row(state_path, artifact_path, D1)
    assert _sha256(artifact_path) == artifact_sha_before
    assert _sha256(state_path) == state_sha_before

    def fake_builder_tiny_diff(root, fold, request, funding, *, slow_horizon_override, committee_member_weights, deadband_seed_row):
        idx = pd.DatetimeIndex([D0, D1])
        frame = pd.DataFrame({"AAAUSDT": [0.02 + 1e-12, 0.03]}, index=idx)
        return frame, idx + pd.Timedelta(hours=1), [], idx

    monkeypatch.setattr(evaluation_mod, "_build_fold_target_weights", fake_builder_tiny_diff)
    report = refresh_signal_row(state_path, artifact_path, D1)
    assert report.status == "APPENDED"


def test_SCENARIO_SIGNAL_06_MEMBER_SET_DRIFT_FAILS_CLOSED(tmp_path, monkeypatch) -> None:
    _patch_no_replay(monkeypatch)

    def fake_builder(root, fold, request, funding, *, slow_horizon_override, committee_member_weights, deadband_seed_row):
        idx = pd.DatetimeIndex([D1])
        frame = pd.DataFrame({"AAAUSDT": [0.02]}, index=idx)
        return frame, idx + pd.Timedelta(hours=1), [], idx

    monkeypatch.setattr(evaluation_mod, "_build_fold_target_weights", fake_builder)

    # A decision_time whose synthetic fold's validation_start lands at/before
    # COMMITTEE_OOS_START breaks the coverage-cutoff precondition the pinned
    # admission guarantee depends on -- must fail closed rather than silently
    # trust admission.
    unsafe_decision_time = COMMITTEE_OOS_START + pd.Timedelta(hours=1)
    state_path, artifact_path = _seed_files(tmp_path, _state(last_decision_time=unsafe_decision_time - pd.Timedelta(days=1)), None)
    with pytest.raises(DataIntegrityError, match="member-parity precondition"):
        refresh_signal_row(state_path, artifact_path, unsafe_decision_time)

    empty_admitted_state = _state(
        frozen=FrozenSignalParams(
            slow_horizon_hours=168, committee_member_weights={}, admitted_members=(),
            growth_budget_target_vol=0.4, exposure_cap=3.0, growth_envelope="growth_extreme_budgeted",
            execution_universe_size=60, pnl_vol_target_mode="growth_budget", deployed_flags=dict(_FLAGS),
        ),
    )
    state_path2, artifact_path2 = _seed_files(tmp_path, empty_admitted_state, None)
    with pytest.raises(DataIntegrityError, match="admitted_members"):
        refresh_signal_row(state_path2, artifact_path2, D1)

    # Direct premise check: an empty pre-cutoff slice trivially admits every spec.
    from src.mhs.features import feature_coverage_audit

    empty_idx = pd.DatetimeIndex([], tz="UTC")
    empty_feature = pd.DataFrame(index=empty_idx, columns=["AAAUSDT"], dtype=float)
    empty_mask = pd.DataFrame(index=empty_idx, columns=["AAAUSDT"], dtype=bool)
    coverage = feature_coverage_audit(empty_feature, empty_mask)
    assert coverage == {}
    assert not any(cov < 1.0 for cov in coverage.values())


def test_SCENARIO_SIGNAL_07_DAILY_PATH_IS_BOUNDED_AND_DISCOVERY_FREE(tmp_path, monkeypatch) -> None:
    _patch_no_replay(monkeypatch)
    captured: dict[str, object] = {}

    def fake_builder(root, fold, request, funding, *, slow_horizon_override, committee_member_weights, deadband_seed_row):
        captured["fold"] = fold
        idx = pd.DatetimeIndex([D1])
        frame = pd.DataFrame({"AAAUSDT": [0.02]}, index=idx)
        return frame, idx + pd.Timedelta(hours=1), [], idx

    monkeypatch.setattr(evaluation_mod, "_build_fold_target_weights", fake_builder)
    state_path, artifact_path = _seed_files(tmp_path, _state(), None)
    report = refresh_signal_row(state_path, artifact_path, D1)
    assert report.status == "APPENDED"

    fold = captured["fold"]
    panel_start = fold.validation_start - pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS)
    assert panel_start == D1 - pd.Timedelta(days=SIGNAL_PANEL_WINDOW_DAYS)

    # I-NO-DISCOVERY: no reference to the sealed-holdout machinery anywhere in
    # the module source, and the fold/selection/discovery stage modules are
    # not among this module's own module-level imports.
    source = Path(signal_refresh_mod.__file__).read_text(encoding="utf-8")
    assert "resolve_evaluation_end(" not in source
    assert "final_oos_2026h1" not in source
    module_names = {
        getattr(v, "__module__", "") for v in vars(signal_refresh_mod).values() if hasattr(v, "__module__")
    }
    assert not any("pipeline.stages.fold" in m or "pipeline.stages.selection" in m or m == "src.mhs.discovery" for m in module_names)


def test_SCENARIO_SIGNAL_08_EXPOSURE_SCALE_FROM_UNSCALED_REPLAY(monkeypatch) -> None:
    from src.application.research.mhs.contracts import MhsDiagnosticRequest
    from src.application.research.mhs.evaluation import _resolved_base_execution_spec
    from src.mhs.evidence import AnchoredPurgedFold

    request = MhsDiagnosticRequest(**_FLAGS)
    fold = AnchoredPurgedFold(
        train_start=pd.Timestamp("2024-01-01", tz="UTC"),
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_start=pd.Timestamp("2026-04-01", tz="UTC"),
        validation_end=D1,
        forward_dependency_hours=24, purge_hours=720,
    )
    idx = pd.date_range(fold.validation_start, fold.validation_end, freq="1D", tz="UTC")
    target_weights = pd.DataFrame({"AAAUSDT": [0.02] * len(idx)}, index=idx)
    signal_available_at = idx + pd.Timedelta(hours=1)

    captured: dict[str, object] = {}

    class _FakeLedger:
        def __init__(self, equity: pd.Series) -> None:
            self.equity = equity

    class _FakeReplayResult:
        def __init__(self, equity: pd.Series) -> None:
            self.ledger = _FakeLedger(equity)

    rng = np.random.default_rng(0)
    daily_returns = rng.normal(0.0, 0.01, size=len(idx) + 1)
    equity = pd.Series(np.cumprod(1.0 + daily_returns), index=pd.date_range(idx[0] - pd.Timedelta(days=1), idx[-1], freq="1D", tz="UTC"))

    def fake_replay(windows, initial_equity, bound, spec):
        captured["initial_equity"] = initial_equity
        captured["bound"] = bound
        captured["spec"] = spec
        return _FakeReplayResult(equity)

    monkeypatch.setattr(evaluation_mod, "_iter_mhs_execution_windows", lambda *a, **k: iter([]))
    monkeypatch.setattr("src.mhs.execution.replay_execution_windows", fake_replay)

    usable = signal_refresh_mod._rolling_reference_returns(target_weights, signal_available_at, fold, request, {}, "")

    assert captured["bound"] == "OHLCV_IMMEDIATE_TAKER"
    assert captured["initial_equity"] == 1.0
    assert captured["spec"] == _resolved_base_execution_spec(request)

    full_ref = equity.resample("1D").last().pct_change().dropna()
    assert len(usable) == max(len(full_ref) - SIGNAL_REPLAY_WARMUP_DAYS, 0)
    pd.testing.assert_series_equal(usable, full_ref.iloc[SIGNAL_REPLAY_WARMUP_DAYS:])

    # exposure scale reproduces the independent composition to 1e-12 and clamps.
    from src.application.research.mhs.scaling import _committee_capital_replay_scale, _exante_vol_target_scale

    state = _state()
    scale = signal_refresh_mod._resolve_exposure_scale(state, usable)
    expected_base = _exante_vol_target_scale(
        usable, target_vol=state.frozen.growth_budget_target_vol, cap=state.frozen.exposure_cap,
        warmup_returns=state.reference_daily_returns if not usable.empty and state.reference_daily_returns.index[-1] < usable.index[0] else None,
    )
    expected_series = _committee_capital_replay_scale(
        expected_base, usable, True, True, cap=state.frozen.exposure_cap,
    )
    expected = float(expected_series.iloc[-1]) if not expected_series.empty else 1.0
    from src.mhs.params import PNL_VOL_TARGET_SCALE_FLOOR

    expected = max(PNL_VOL_TARGET_SCALE_FLOOR, min(expected, state.frozen.exposure_cap))
    assert scale == pytest.approx(expected, abs=1e-12)
    assert PNL_VOL_TARGET_SCALE_FLOOR <= scale <= state.frozen.exposure_cap

    doubled = usable * 2.0
    scale_doubled = signal_refresh_mod._resolve_exposure_scale(state, doubled)
    assert scale_doubled < scale

    # I-STATELESS-REPLAY: repeated calls over the same window are deterministic.
    scale_again = signal_refresh_mod._resolve_exposure_scale(state, usable)
    assert scale_again == scale


def test_SCENARIO_SIGNAL_11_WINDOW_CONSTANTS_SATISFY_DERIVED_BOUNDS() -> None:
    assert max(720, FOLD_PANEL_WARMUP_HOURS) <= SIGNAL_PANEL_WINDOW_DAYS * 24
    assert SIGNAL_PANEL_WINDOW_DAYS * 24 > 2000
    assert 0 < SIGNAL_REPLAY_WARMUP_DAYS < SIGNAL_PANEL_WINDOW_DAYS
    assert SIGNAL_PANEL_WINDOW_DAYS - SIGNAL_REPLAY_WARMUP_DAYS >= 4 * PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS
    assert SIGNAL_RETURN_TAIL_DAYS > PNL_VOL_TARGET_BURN_IN_DAYS


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_SIGNAL_03_REFRESH_APPENDS_EXACTLY_ONE_ROW",
    "SCENARIO_SIGNAL_04_SAME_DAY_RERUN_IS_NOOP",
    "SCENARIO_SIGNAL_05_OVERLAP_DIVERGENCE_FAILS_CLOSED",
    "SCENARIO_SIGNAL_06_MEMBER_SET_DRIFT_FAILS_CLOSED",
    "SCENARIO_SIGNAL_07_DAILY_PATH_IS_BOUNDED_AND_DISCOVERY_FREE",
    "SCENARIO_SIGNAL_08_EXPOSURE_SCALE_FROM_UNSCALED_REPLAY",
    "SCENARIO_SIGNAL_11_WINDOW_CONSTANTS_SATISFY_DERIVED_BOUNDS",
)
