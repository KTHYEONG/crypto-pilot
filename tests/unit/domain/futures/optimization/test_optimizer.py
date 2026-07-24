from __future__ import annotations

import pandas as pd

from src.domain.futures.optimization.optimizer import compute_multi_alignment_info


def test_compute_multi_alignment_info_returns_common_causal_window() -> None:
    frame = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=220, freq="h")})
    result = compute_multi_alignment_info({"BTCUSDT": {"4h": frame}}, ["BTCUSDT"], "4h", 0)
    assert result is not None
    assert result["eff_ref_len"] == 220


def test_compute_multi_alignment_info_rejects_short_panels() -> None:
    frame = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=10, freq="h")})
    assert compute_multi_alignment_info({"BTCUSDT": {"4h": frame}}, ["BTCUSDT"], "4h", 0) is None
