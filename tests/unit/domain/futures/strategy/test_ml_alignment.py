from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.common import alignment


def _frame(n: int = 6) -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": np.linspace(100.0, 106.0, n),
            "high": np.linspace(101.0, 107.0, n),
            "low": np.linspace(99.0, 105.0, n),
            "close": np.linspace(100.5, 106.5, n),
            "volume": np.linspace(1000.0, 1200.0, n),
        }
    )


def test_align_data_maps_funding_defaults_to_zero_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    out = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert out.funding_2d.shape == (len(frame), 1)
    assert np.allclose(out.funding_2d[:, 0], 0.0)
    assert np.all(out.active_mask[:, 0])
    assert np.all(out.warm_mask[:, 0])
    assert not np.any(out.entry_block_mask[:, 0])
    assert not np.any(out.kill_mask[:, 0])


def test_align_data_maps_missing_ohlcv_column_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame().drop(columns=["volume"])
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    with pytest.raises(ValueError, match="missing required column: volume"):
        alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")


def test_align_data_maps_uses_first_finite_metadata_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().assign(
        cluster_id=[3.0, np.nan, 9.0, 9.0, 9.0, 9.0],
        beta_vs_market=[1.2, np.nan, 2.8, 2.8, 2.8, 2.8],
        cluster_size=[2.0, np.nan, 6.0, 6.0, 6.0, 6.0],
        anchor_cluster_member=[1.0, np.nan, 0.0, 0.0, 0.0, 0.0],
    )
    data_maps = {"BTCUSDT": {"4h": frame}}
    monkeypatch.setattr(
        alignment,
        "compute_multi_alignment_info",
        lambda *_args, **_kwargs: {"eff_ref_len": len(frame), "alignment_offsets": {"BTCUSDT": 0}},
    )

    out = alignment.align_data_maps(data_maps, ["BTCUSDT"], "4h")
    assert out.cluster_id_1d is not None
    assert out.beta_vs_market_1d is not None
    assert out.cluster_size_1d is not None
    assert out.anchor_cluster_1d is not None
    assert out.cluster_id_1d.tolist() == [3.0]
    assert np.allclose(out.beta_vs_market_1d, np.array([1.2], dtype=np.float32))
    assert out.cluster_size_1d.tolist() == [2.0]
    assert out.anchor_cluster_1d.tolist() == [1.0]
