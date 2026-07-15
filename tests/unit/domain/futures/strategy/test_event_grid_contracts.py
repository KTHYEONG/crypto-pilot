from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.application.futures.optimization.strategy_service import (
    ActiveL0RuntimeContractError,
    require_active_l0_runtime,
)
from src.application.futures.run_contracts import FuturesRunConfig
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    resolve_corroboration_evidence_for_target,
)
from src.domain.futures.strategy.event_grid_contracts import (
    EventGridContractError,
    MissingNativeTfEventsError,
    validate_native_event_grid,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    _resolve_labeled_events_for_tf,
)


def _run_config(*, mode: str = "gate") -> FuturesRunConfig:
    return FuturesRunConfig(
        timeframe="4h",
        date="2026-05-01",
        trials=1,
        phase="l1",
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
        seed=42,
        l0_runtime=AlphaFoundryRuntimeConfig(mode=mode),  # type: ignore[arg-type]
    )


def test_l1_rejects_inactive_l0_runtime() -> None:
    with pytest.raises(ActiveL0RuntimeContractError, match=r"requires l0_runtime.mode='gate'"):
        require_active_l0_runtime(_run_config(mode="off"))


def test_active_l0_runtime_ok() -> None:
    cfg = _run_config(mode="gate")
    runtime = require_active_l0_runtime(cfg)
    assert runtime.mode == "gate"


def test_l2_no_l0_runtime_check_skipped() -> None:
    cfg = replace(_run_config(mode="off"), phase="l2")
    runtime = require_active_l0_runtime(cfg)
    assert runtime.mode == "off"


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
        validate_native_event_grid(events=events, native_datetimes=native_dt, timeframe="12h")


def test_native_event_grid_happy_path() -> None:
    native_dt = np.array(
        ["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00"],
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
    result, audit = validate_native_event_grid(events=events, native_datetimes=native_dt, timeframe="1h")
    assert audit.valid_count == 3
    assert audit.mismatch_count == 0
    assert audit.terminal_maturity_count == 0
    assert len(result) == 3


def test_terminal_maturity_removed() -> None:
    native_dt = np.array(
        ["2024-01-01T00:00", "2024-01-01T12:00"],
        dtype="datetime64[ns]",
    )
    events = pd.DataFrame(
        {
            "event_id": [1, 2],
            "datetime": [
                pd.Timestamp("2024-01-01T00:00Z"),
                pd.Timestamp("2024-01-02T00:00Z"),
            ],
            "entry_idx": [1, 2],
        }
    )
    result, audit = validate_native_event_grid(events=events, native_datetimes=native_dt, timeframe="12h")
    assert audit.valid_count == 1
    assert audit.terminal_maturity_count == 1
    assert audit.mismatch_count == 0
    assert len(result) == 1


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
