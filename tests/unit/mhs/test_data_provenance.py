

def test_input_manifest_detects_post_seal_mutation(tmp_path) -> None:
    import pandas as pd
    from src.mhs.data_provenance import DataEvidenceTier, seal_mhs_input_manifest, validate_mhs_input_manifest
    path = tmp_path/'BTCUSDT.parquet'
    pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [1.0]}).to_parquet(path)
    manifest = tmp_path/'inputs.json'
    seal_mhs_input_manifest([path], data_root=tmp_path, output_path=manifest)
    with path.open('ab') as handle:
        handle.write(b'x')
    result = validate_mhs_input_manifest(manifest, data_root=tmp_path, required_paths=[path])
    assert not result.valid
    assert result.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert 'STALE_INPUT_MANIFEST' in result.reason_codes


def test_input_manifest_requires_every_consumed_path(tmp_path) -> None:
    import pandas as pd
    from src.mhs.data_provenance import seal_mhs_input_manifest, validate_mhs_input_manifest
    a, b = tmp_path/'a.parquet', tmp_path/'b.parquet'
    frame = pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [1.0]})
    frame.to_parquet(a)
    frame.to_parquet(b)
    manifest = tmp_path/'inputs.json'
    seal_mhs_input_manifest([a], data_root=tmp_path, output_path=manifest)
    result = validate_mhs_input_manifest(manifest, data_root=tmp_path, required_paths=[a, b])
    assert not result.valid
    assert 'UNATTESTED_REQUIRED_FILE' in result.reason_codes


def test_input_manifest_numeric_timestamp_bounds_use_epoch_milliseconds(tmp_path) -> None:
    import json
    import pandas as pd
    from src.mhs.data_provenance import seal_mhs_input_manifest

    path = tmp_path / "BTCUSDT.parquet"
    pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    manifest = tmp_path / "inputs.json"
    seal_mhs_input_manifest([path], data_root=tmp_path, output_path=manifest)
    entry = json.loads(manifest.read_text())["files"][0]
    assert entry["first_timestamp"].startswith("2025-01-01T00:00:00")
    assert entry["last_timestamp"].startswith("2025-01-01T00:00:00")


def test_forward_observation_digest_mismatch_cannot_upgrade() -> None:
    import pandas as pd
    from src.mhs.data_provenance import DataEvidenceTier, validate_forward_execution_observations
    records = pd.DataFrame({'decision_time': [pd.Timestamp('2025-01-01', tz='UTC')], 'observed_at': [pd.Timestamp('2025-01-01 00:01', tz='UTC')], 'strategy_digest': ['other']})
    result = validate_forward_execution_observations(records, frozen_strategy_digest='frozen')
    assert not result.valid
    assert result.tier is not DataEvidenceTier.FORWARD_OBSERVED
    assert 'STRATEGY_DIGEST_MISMATCH' in result.reason_codes


def test_required_input_paths_include_panel_execution_mark_and_funding(tmp_path) -> None:
    from src.mhs.data_provenance import resolve_required_mhs_input_paths
    paths = resolve_required_mhs_input_paths(data_root=tmp_path, panel_symbols=['BTCUSDT'], execution_symbols=['BTCUSDT'], execution_timeframe='3m')
    rendered = {str(p.relative_to(tmp_path)) for p in paths}
    assert any('1h' in p and 'BTCUSDT' in p for p in rendered)
    assert any('3m' in p and 'BTCUSDT' in p for p in rendered)
    assert any('funding' in p and 'BTCUSDT' in p for p in rendered)
    assert not any('mark' in p for p in rendered)
    assert not any('metrics' in p or '1d' in p.split('/') for p in rendered)
    assert rendered == {
        'ohlcv/1h/BTCUSDT.parquet',
        'ohlcv/3m/BTCUSDT.parquet',
        'funding/BTCUSDT.parquet',
    }


def test_provenance_absent_manifest_and_seal_success(tmp_path) -> None:
    import pandas as pd
    from src.mhs.data_provenance import (
        DataEvidenceTier,
        seal_mhs_input_manifest,
        validate_forward_execution_observations,
        validate_mhs_input_manifest,
    )
    frame = pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [1.0]})
    path = tmp_path / 'a.parquet'
    frame.to_parquet(path)
    absent = validate_mhs_input_manifest(None, data_root=tmp_path, required_paths=[path])
    assert not absent.valid
    assert absent.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert 'MISSING_INPUT_MANIFEST' in absent.reason_codes
    manifest = tmp_path / 'inputs.json'
    digest = seal_mhs_input_manifest([path], data_root=tmp_path, output_path=manifest)
    assert isinstance(digest, str)
    assert len(digest) == 64
    ok_result = validate_mhs_input_manifest(manifest, data_root=tmp_path, required_paths=[path])
    assert ok_result.valid
    assert ok_result.tier is DataEvidenceTier.REPRODUCIBLE_ARCHIVE
    assert ok_result.manifest_digest == digest
    legacy = pd.DataFrame({'decision_time': [pd.Timestamp('2025-01-01', tz='UTC')]})
    schema_result = validate_forward_execution_observations(legacy, frozen_strategy_digest='frozen')
    assert not schema_result.valid
    assert 'FORWARD_OBSERVATION_SCHEMA_MISMATCH' in schema_result.reason_codes
    bad_time = pd.DataFrame({
        'decision_time': [pd.Timestamp('2025-01-02', tz='UTC')],
        'observed_at': [pd.Timestamp('2025-01-01', tz='UTC')],
        'strategy_digest': ['frozen'],
    })
    time_result = validate_forward_execution_observations(bad_time, frozen_strategy_digest='frozen')
    assert not time_result.valid
    assert 'FORWARD_OBSERVATION_TIME_INVALID' in time_result.reason_codes
    short = pd.DataFrame({
        'decision_time': [pd.Timestamp('2025-01-01', tz='UTC')],
        'observed_at': [pd.Timestamp('2025-01-01 00:01', tz='UTC')],
        'strategy_digest': ['frozen'],
    })
    short_result = validate_forward_execution_observations(short, frozen_strategy_digest='frozen')
    assert not short_result.valid
    assert 'FORWARD_EVIDENCE_INCOMPLETE' in short_result.reason_codes
    long_enough = pd.DataFrame({
        'decision_time': [pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-04-15', tz='UTC')],
        'observed_at': [pd.Timestamp('2025-01-01 00:01', tz='UTC'), pd.Timestamp('2025-04-15 00:01', tz='UTC')],
        'strategy_digest': ['frozen', 'frozen'],
    })
    good = validate_forward_execution_observations(long_enough, frozen_strategy_digest='frozen')
    assert good.valid
    assert good.tier is DataEvidenceTier.FORWARD_OBSERVED
    assert good.files_checked == 2

def test_mhs_sealable_input_paths_skips_incomplete_symbols(tmp_path) -> None:
    # Given: COMPLETEUSDT has all three required files, PARTIALUSDT lacks funding
    import pandas as pd

    from src.mhs.data_provenance import mhs_sealable_input_paths

    def _write(rel: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"timestamp": [pd.Timestamp("2025-01-01", tz="UTC")], "close": [1.0]}
        ).to_parquet(path)

    for rel in (
        "ohlcv/1h/COMPLETEUSDT.parquet",
        "ohlcv/3m/COMPLETEUSDT.parquet",
        "funding/COMPLETEUSDT.parquet",
        "ohlcv/1h/PARTIALUSDT.parquet",
        "ohlcv/3m/PARTIALUSDT.parquet",
    ):
        _write(rel)

    # When
    paths = mhs_sealable_input_paths(data_root=tmp_path, execution_timeframe="3m")

    # Then: existing required files are sealed, mark never enters the identity
    names = {p.relative_to(tmp_path).as_posix() for p in paths}
    assert names == {
        "ohlcv/1h/COMPLETEUSDT.parquet",
        "ohlcv/3m/COMPLETEUSDT.parquet",
        "funding/COMPLETEUSDT.parquet",
        "ohlcv/1h/PARTIALUSDT.parquet",
        "ohlcv/3m/PARTIALUSDT.parquet",
    }
    assert not any("mark" in name for name in names)
    assert len(paths) == len(set(paths))

def test_seal_then_validate_round_trip_is_reproducible_archive(tmp_path) -> None:
    # Given: a complete two-file-kind corpus for one symbol
    import pandas as pd

    from src.mhs.data_provenance import (
        DataEvidenceTier,
        mhs_sealable_input_paths,
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    for rel in (
        "ohlcv/1h/COMPLETEUSDT.parquet",
        "ohlcv/3m/COMPLETEUSDT.parquet",
        "funding/COMPLETEUSDT.parquet",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"timestamp": [pd.Timestamp("2025-01-01", tz="UTC")], "close": [1.0]}
        ).to_parquet(path)

    manifest = tmp_path / "input_manifest.json"
    sealable = mhs_sealable_input_paths(data_root=tmp_path, execution_timeframe="3m")

    # When
    digest = seal_mhs_input_manifest(sealable, data_root=tmp_path, output_path=manifest)
    required = resolve_required_mhs_input_paths(
        data_root=tmp_path, panel_symbols=["COMPLETEUSDT"],
        execution_symbols=["COMPLETEUSDT"], execution_timeframe="3m",
    )
    result = validate_mhs_input_manifest(manifest, data_root=tmp_path, required_paths=required)

    # Then: the sealed digest is carried and the tier clears the I3 gate
    assert result.valid is True
    assert result.tier is DataEvidenceTier.REPRODUCIBLE_ARCHIVE
    assert result.reason_codes == ()
    assert result.manifest_digest == digest


def test_required_files_are_exactly_consumed_sources() -> None:
    from pathlib import Path

    from src.mhs.data_provenance import resolve_required_mhs_input_paths

    paths = resolve_required_mhs_input_paths(
        data_root=Path("/data"),
        panel_symbols=["PANELONLYUSDT", "BOTHUSDT"],
        execution_symbols=["BOTHUSDT", "EXEConlyUSDT".upper()],
        execution_timeframe="3m",
    )
    rendered = {p.as_posix() for p in paths}
    assert "/data/ohlcv/1h/PANELONLYUSDT.parquet" in rendered
    assert "/data/funding/PANELONLYUSDT.parquet" in rendered
    assert not any("PANELONLYUSDT" in p and "/3m/" in p for p in rendered)
    assert "/data/ohlcv/1h/BOTHUSDT.parquet" in rendered
    assert "/data/ohlcv/3m/BOTHUSDT.parquet" in rendered
    assert "/data/funding/BOTHUSDT.parquet" in rendered
    assert "/data/ohlcv/3m/EXECONLYUSDT.parquet" in rendered
    assert "/data/funding/EXECONLYUSDT.parquet" in rendered
    assert not any("markPriceKlines" in p for p in rendered)
    assert not any("/metrics/" in p or p.endswith("/1d") for p in rendered)
    assert list(paths) == sorted(paths, key=lambda p: p.as_posix())


def test_mark_presence_cannot_change_digest(tmp_path) -> None:
    import pandas as pd

    from src.mhs.data_provenance import mhs_sealable_input_paths, seal_mhs_input_manifest

    def _write(rel: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)

    for rel in (
        "ohlcv/1h/AUSDT.parquet",
        "ohlcv/3m/AUSDT.parquet",
        "funding/AUSDT.parquet",
    ):
        _write(rel)
    first = mhs_sealable_input_paths(data_root=tmp_path, execution_timeframe="3m")
    digest_first = seal_mhs_input_manifest(first, data_root=tmp_path, output_path=tmp_path / "m1.json")
    _write("markPriceKlines/1h/AUSDT.parquet")
    second = mhs_sealable_input_paths(data_root=tmp_path, execution_timeframe="3m")
    digest_second = seal_mhs_input_manifest(second, data_root=tmp_path, output_path=tmp_path / "m2.json")
    assert digest_first == digest_second
    assert {p.as_posix() for p in first} == {p.as_posix() for p in second}


def test_missing_execution_source_remains_visible(tmp_path) -> None:
    import pandas as pd

    from src.mhs.data_provenance import (
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    for rel in ("ohlcv/1h/AUSDT.parquet", "funding/AUSDT.parquet"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    required = resolve_required_mhs_input_paths(
        data_root=tmp_path, panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    assert any(p.as_posix().endswith("ohlcv/3m/AUSDT.parquet") for p in required)
    existing = [p for p in required if p.exists()]
    digest = seal_mhs_input_manifest(existing, data_root=tmp_path, output_path=tmp_path / "m.json")
    assert isinstance(digest, str)
    result = validate_mhs_input_manifest(tmp_path / "m.json", data_root=tmp_path, required_paths=required)
    assert not result.valid
    assert "UNATTESTED_REQUIRED_FILE" in result.reason_codes


def test_missing_funding_remains_visible(tmp_path) -> None:
    import pandas as pd

    from src.mhs.data_provenance import (
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    for rel in ("ohlcv/1h/AUSDT.parquet", "ohlcv/3m/AUSDT.parquet"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    required = resolve_required_mhs_input_paths(
        data_root=tmp_path, panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    assert any(p.as_posix().endswith("funding/AUSDT.parquet") for p in required)
    existing = [p for p in required if p.exists()]
    seal_mhs_input_manifest(existing, data_root=tmp_path, output_path=tmp_path / "m.json")
    result = validate_mhs_input_manifest(tmp_path / "m.json", data_root=tmp_path, required_paths=required)
    assert not result.valid
    assert "UNATTESTED_REQUIRED_FILE" in result.reason_codes


def test_input_identity_is_deterministic(tmp_path) -> None:
    import pandas as pd

    from src.mhs.data_provenance import (
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
    )

    for rel in (
        "ohlcv/1h/AUSDT.parquet",
        "ohlcv/3m/AUSDT.parquet",
        "funding/AUSDT.parquet",
        "ohlcv/1h/BUSDT.parquet",
        "ohlcv/3m/BUSDT.parquet",
        "funding/BUSDT.parquet",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    forward = resolve_required_mhs_input_paths(
        data_root=tmp_path, panel_symbols=["AUSDT", "BUSDT"], execution_symbols=["BUSDT", "AUSDT"], execution_timeframe="3m",
    )
    reverse = resolve_required_mhs_input_paths(
        data_root=tmp_path, panel_symbols=["BUSDT", "AUSDT"], execution_symbols=["AUSDT", "BUSDT"], execution_timeframe="3m",
    )
    assert list(forward) == list(reverse)
    digest_forward = seal_mhs_input_manifest(list(forward), data_root=tmp_path, output_path=tmp_path / "m1.json")
    digest_reverse = seal_mhs_input_manifest(list(reversed(reverse)), data_root=tmp_path, output_path=tmp_path / "m2.json")
    assert digest_forward == digest_reverse
