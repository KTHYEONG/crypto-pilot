from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.futures.run_contracts import FuturesRunConfig, RunPolicyError
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    resolve_corroboration_evidence_for_target,
)
from src.domain.futures.strategy.event_grid_contracts import (
    EventGridContractError,
    MissingNativeTfEventsError,
    normalize_native_l1_events,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    _resolve_labeled_events_for_tf,
)


def _run_config(*, mode: str = "gate", phase: str = "l1") -> FuturesRunConfig:
    return FuturesRunConfig(
        timeframe="4h",
        date="2026-05-01",
        trials=1,
        phase=phase,  # type: ignore[arg-type]
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
        seed=42,
        l0_runtime=AlphaFoundryRuntimeConfig(mode=mode),  # type: ignore[arg-type]
    )


def test_l1_rejects_inactive_l0_runtime() -> None:
    with pytest.raises(RunPolicyError, match=r"requires l0_runtime.mode='gate'"):
        _run_config(mode="off")


def test_active_l0_runtime_ok() -> None:
    cfg = _run_config(mode="gate")
    assert cfg.l0_runtime.mode == "gate"


def test_l2_no_l0_runtime_check_skipped() -> None:
    cfg = _run_config(mode="off", phase="l2")
    assert cfg.l0_runtime.mode == "off"


def test_native_event_grid_rejects_cross_tf_index() -> None:
    native_dt = np.array(
        ["2024-01-01T00:00", "2024-01-01T12:00", "2024-01-02T00:00"],
        dtype="datetime64[ns]",
    )
    events = pd.DataFrame(
        {
            "event_id": [7],
            "datetime": [pd.Timestamp("2024-01-01T00:00Z")],
            "entry_idx": [12],
        }
    )
    with pytest.raises(EventGridContractError, match="event_id=7"):
        normalize_native_l1_events(
            events=events, native_datetimes=native_dt, timeframe="12h", required_horizon_bars=1,
        )


def test_native_event_grid_happy_path() -> None:
    native_dt = np.array(
        ["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00", "2024-01-01T18:00"],
        dtype="datetime64[ns]",
    )
    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "datetime": [
                pd.Timestamp("2024-01-01T00:00Z"),
                pd.Timestamp("2024-01-01T05:00Z"),
                pd.Timestamp("2024-01-01T10:00Z"),
            ],
            "entry_idx": [1, 1, 2],
        }
    )
    result = normalize_native_l1_events(
        events=events, native_datetimes=native_dt, timeframe="1h", required_horizon_bars=1,
    )
    assert result.audit.eligible_count == 3
    assert result.audit.mismatch_count == 0
    assert result.audit.terminal_maturity_count == 0
    assert len(result.eligible_events) == 3


def test_terminal_maturity_removed() -> None:
    native_dt = np.array(
        ["2024-01-01T00:00", "2024-01-01T12:00", "2024-01-02T00:00"],
        dtype="datetime64[ns]",
    )
    events = pd.DataFrame(
        {
            "event_id": [1, 2],
            "datetime": [
                pd.Timestamp("2024-01-01T00:00Z"),
                pd.Timestamp("2024-01-01T18:00Z"),
            ],
            "entry_idx": [1, 2],
        }
    )
    result = normalize_native_l1_events(
        events=events, native_datetimes=native_dt, timeframe="12h", required_horizon_bars=1,
    )
    assert result.audit.eligible_count == 1
    assert result.audit.terminal_maturity_count == 1
    assert result.audit.mismatch_count == 0
    assert len(result.eligible_events) == 1


def _events(entry_idx: list[int]) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=len(entry_idx), freq="2h", tz="UTC")
    return pd.DataFrame(
        {
            "event_id": list(range(100, 100 + len(entry_idx))),
            "datetime": timestamps,
            "entry_idx": entry_idx,
            "native_tf": ["2h"] * len(entry_idx),
            "symbol": ["BTCUSDT"] * len(entry_idx),
            "strategy_id": ["r1"] * len(entry_idx),
        }
    )


def test_terminal_events_are_audited_not_silently_dropped() -> None:
    grid = pd.date_range("2026-01-01", periods=8, freq="2h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    events = _events([1, 2, 3, 4, 5, 6, 7, 8])

    result = normalize_native_l1_events(
        events=events,
        native_datetimes=grid,
        timeframe="2h",
        required_horizon_bars=2,
    )

    assert result.audit.status == "terminal_excluded"
    assert result.audit.terminal_maturity_count == 3
    assert result.audit.first_terminal_event_id == 105
    assert result.audit.last_terminal_event_id == 107
    assert result.eligible_events["event_id"].tolist() == [100, 101, 102, 103, 104]


def test_non_terminal_index_mismatch_fails_closed() -> None:
    grid = pd.date_range("2026-01-01", periods=8, freq="2h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    events = _events([0, 2])

    with pytest.raises(EventGridContractError, match="timeframe=2h event_id=100"):
        normalize_native_l1_events(
            events=events,
            native_datetimes=grid,
            timeframe="2h",
            required_horizon_bars=1,
        )


def test_missing_native_frame_never_falls_back_to_pooled() -> None:
    pooled = pd.DataFrame({"native_tf": ["1h"], "entry_idx": [1]})
    with pytest.raises(MissingNativeTfEventsError, match="tf=6h"):
        _resolve_labeled_events_for_tf("6h", pooled, {"1h": pooled}, require_native=True)


def test_legitimate_empty_native_key_no_error() -> None:
    empty = pd.DataFrame({"entry_idx": pd.Series(dtype=np.int64)})
    result = _resolve_labeled_events_for_tf("6h", empty, {"6h": empty}, require_native=True)
    assert len(result) == 0


def test_adding_1h_does_not_change_existing_target_fusion_input() -> None:
    def frame(tf: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timeframe": [tf],
                "family": ["trend_ma"],
                "variant": [f"ema_{tf}"],
                "recipe_id": [f"trend_ma:{tf}"],
                "reject_reasons": [""],
                "mean_net_bps": [10.0],
                "block_lcb_bps": [8.0],
            }
        )
    control = {tf: frame(tf) for tf in ("2h", "4h", "6h", "8h", "12h", "1d")}
    treatment = {"1h": frame("1h"), **control}
    refs = ("2h", "4h", "6h", "8h", "12h", "1d")
    assert resolve_corroboration_evidence_for_target(
        target_tf="6h", evidence_by_tf=control, reference_tfs=refs
    ).keys() == resolve_corroboration_evidence_for_target(
        target_tf="6h", evidence_by_tf=treatment, reference_tfs=refs
    ).keys()


def test_resolve_corroboration_missing_target_raises() -> None:
    with pytest.raises(ValueError, match="missing native evidence"):
        resolve_corroboration_evidence_for_target(
            target_tf="6h",
            evidence_by_tf={"2h": pd.DataFrame()},
            reference_tfs=("2h",),
        )
