"""Regression tests for L1 evidence fields in signal contracts."""
from __future__ import annotations

import numpy as np

from src.domain.futures.signals.contracts import ValidatedSignalEvent


def test_validated_signal_event_l1_evidence_defaults_are_backward_compatible() -> None:
    event = ValidatedSignalEvent(
        symbol="BTCUSDT",
        strategy_id="momentum",
        decision_idx=0,
        decision_time=np.datetime64("2026-01-01"),
        expected_gross_bps=1.0,
        expected_net_bps=0.5,
        q10_gross_bps=-1.0,
        q90_gross_bps=2.0,
        expected_holding_bars=1,
        side=1,
        quality_weight=1.0,
        activation_context="test",
        registry_version="test",
        model_version="test",
        l1_lcb_net_bps=0.0,
        l1_breakeven_bps=0.0,
    )
    assert event.l1_lcb_net_bps == 0.0
    assert event.l1_breakeven_bps == 0.0
