# ruff: noqa
def test_deployed_weights_append_and_truncate(tmp_path) -> None:
    import pandas as pd

    from src.live.deployed_weights import append_weight_row, load_weights_frame

    path = tmp_path / "deployed_target_weights.parquet"
    for i in range(6):
        d = pd.Timestamp("2026-08-20", tz="UTC") + pd.Timedelta(days=i)
        appended = append_weight_row(path, d, pd.Series({"BTCUSDT": 0.1 * i}), keep_rows=3)
        assert appended is True

    again = append_weight_row(path, pd.Timestamp("2026-08-25", tz="UTC"), pd.Series({"BTCUSDT": 9.9}), keep_rows=3)
    assert again is False

    frame = load_weights_frame(path)
    assert len(frame) == 3
    assert frame.index[-1] == pd.Timestamp("2026-08-25", tz="UTC")
    assert frame.index[0] == pd.Timestamp("2026-08-23", tz="UTC")

def test_weights_asof_holds_recent_row_and_flags_stale() -> None:
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.errors import StaleSignalError
    from src.live.deployed_weights import weights_asof

    idx = pd.date_range("2026-08-20", periods=3, freq="1D", tz="UTC")
    frame = pd.DataFrame({"BTCUSDT": [0.1, 0.2, 0.3]}, index=idx)

    row = weights_asof(frame, pd.Timestamp("2026-08-23", tz="UTC"), max_staleness=pd.Timedelta(days=4))
    assert float(row["BTCUSDT"]) == 0.3
    assert pd.Timestamp(row.name) == pd.Timestamp("2026-08-22", tz="UTC")

    with pytest.raises(StaleSignalError):
        weights_asof(frame, pd.Timestamp("2026-09-10", tz="UTC"), max_staleness=pd.Timedelta(days=4))
    with pytest.raises(DataIntegrityError):
        weights_asof(frame, pd.Timestamp("2026-08-01", tz="UTC"), max_staleness=pd.Timedelta(days=4))
