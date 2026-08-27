"""SCENARIO_SIGNAL_01/02: signal state round-trip, sealing, params/flags binding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.errors import ArtifactSealError
from src.mhs.signal_state import (
    FrozenSignalParams,
    SignalState,
    assert_state_binding,
    compute_flags_digest,
    compute_params_digest,
    load_signal_state,
    save_signal_state,
)

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")

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


def _state(**overrides) -> SignalState:
    frozen = FrozenSignalParams(
        slow_horizon_hours=168,
        committee_member_weights={"flow_imb_720h": 0.2, "flow_imb_168h": 0.2, "xs_mom_336h": 0.2, "xs_idio_mom_336h": 0.2, "mom3_skew_168h": 0.2},
        admitted_members=("flow_imb_720h", "flow_imb_168h", "xs_mom_336h", "xs_idio_mom_336h", "mom3_skew_168h"),
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
        "flags_digest": "0" * 16,
        "frozen": frozen,
        "last_decision_time": DECISION_TIME,
        "held_target_row": {"AAAUSDT": 0.02, "BUSDT": -0.02},
        "reference_daily_returns": pd.Series(
            [0.001, -0.002, 0.0005],
            index=pd.DatetimeIndex(
                [DECISION_TIME - pd.Timedelta(days=2), DECISION_TIME - pd.Timedelta(days=1), DECISION_TIME]
            ),
        ),
    }
    defaults.update(overrides)
    return SignalState(**defaults)


def test_SCENARIO_SIGNAL_01_STATE_ROUND_TRIPS_SEALED_AND_PLAINTEXT(tmp_path: Path) -> None:
    state = _state()
    path = tmp_path / "signal_state.json"
    save_signal_state(path, state)
    loaded = load_signal_state(path)
    assert loaded.frozen == state.frozen
    assert loaded.last_decision_time == state.last_decision_time
    assert loaded.last_decision_time.tzinfo is not None
    assert loaded.held_target_row == state.held_target_row
    pd.testing.assert_series_equal(loaded.reference_daily_returns, state.reference_daily_returns)

    key = SecretStr("QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE=")
    sealed_path = save_signal_state(tmp_path / "sealed.json", state, artifact_key=key)
    assert str(sealed_path).endswith(".enc")
    raw_bytes = sealed_path.read_bytes()
    assert raw_bytes.startswith(b"CPSEAL01")
    with pytest.raises(ArtifactSealError):
        load_signal_state(sealed_path)
    reloaded = load_signal_state(sealed_path, artifact_key=key)
    assert reloaded.frozen == state.frozen

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_signal_state(corrupt)

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_signal_state(bad_schema)


def test_SCENARIO_SIGNAL_02_STATE_BINDING_REJECTS_DRIFTED_CONSTANTS(monkeypatch) -> None:
    state = _state(flags_digest=compute_flags_digest(_FLAGS))
    assert_state_binding(state, _FLAGS)

    import src.mhs.params as params_mod

    monkeypatch.setattr(params_mod, "REBALANCE_DEADBAND_POSITION_FRACTION", 0.99)
    with pytest.raises(DataIntegrityError, match="params_digest"):
        assert_state_binding(state, _FLAGS)
    monkeypatch.undo()

    drifted_flags = dict(_FLAGS)
    drifted_flags["committee_target_gross"] = 0.80
    with pytest.raises(DataIntegrityError, match="flags_digest"):
        assert_state_binding(state, drifted_flags)

    unbound_drift_flags = dict(_FLAGS)
    unbound_drift_flags["log_run"] = not unbound_drift_flags.get("log_run", True)
    assert_state_binding(state, unbound_drift_flags)  # unbound flag never affects the digest

    d1 = compute_params_digest()
    d2 = compute_params_digest()
    assert d1 == d2
    assert len(d1) == 16
    assert d1 == d1.lower()


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_SIGNAL_01_STATE_ROUND_TRIPS_SEALED_AND_PLAINTEXT",
    "SCENARIO_SIGNAL_02_STATE_BINDING_REJECTS_DRIFTED_CONSTANTS",
)
