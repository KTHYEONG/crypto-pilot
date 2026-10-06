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


def test_sibling_artifact_paths_require_weights_token(tmp_path) -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import exposure_scale_path

    base = tmp_path / "deployed_target_weights.parquet.enc"
    assert exposure_scale_path(base) == tmp_path / "deployed_exposure_scale.parquet.enc"
    with pytest.raises(DataIntegrityError, match="deployed_target_weights"):
        exposure_scale_path(tmp_path / "w.parquet")


def test_weights_write_fsyncs_file_and_directory_plain_and_sealed(tmp_path, monkeypatch) -> None:
    """Both plain and sealed weight writes fsync the file before replace and the directory after."""
    import base64
    import os
    from pathlib import Path

    import pandas as pd
    from pydantic import SecretStr

    from src.live.deployed_weights import append_weight_row, load_weights_frame

    real_fsync = os.fsync
    file_fsyncs: list[str] = []
    dir_fsyncs: list[str] = []

    def _spy(fd: int) -> None:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            target = str(fd)
        (dir_fsyncs if Path(target).is_dir() else file_fsyncs).append(target)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    day = pd.Timestamp("2026-08-24", tz="UTC")
    row = pd.Series({"BTCUSDT": 0.1})
    plain = tmp_path / "deployed_target_weights.parquet"
    assert append_weight_row(plain, day, row) is True
    assert len(file_fsyncs) == 1
    assert len(dir_fsyncs) == 1
    file_fsyncs.clear()
    dir_fsyncs.clear()
    key = SecretStr(base64.b64encode(b"0" * 32).decode("ascii"))
    sealed_base = tmp_path / "sealed_deployed_target_weights.parquet"
    assert append_weight_row(sealed_base, day, row, artifact_key=key) is True
    assert len(file_fsyncs) == 1
    assert len(dir_fsyncs) == 1
    expected = pd.DataFrame([row], index=pd.DatetimeIndex([day]))
    pd.testing.assert_frame_equal(load_weights_frame(plain), expected)
    pd.testing.assert_frame_equal(load_weights_frame(sealed_base, artifact_key=key), expected)
