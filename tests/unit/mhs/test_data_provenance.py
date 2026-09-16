

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
    assert any('mark' in p and 'BTCUSDT' in p for p in rendered)


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
    # Given: COMPLETEUSDT has all four required files, PARTIALUSDT lacks funding
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
        "markPriceKlines/1h/COMPLETEUSDT.parquet",
        "ohlcv/1h/PARTIALUSDT.parquet",
        "ohlcv/3m/PARTIALUSDT.parquet",
        "markPriceKlines/1h/PARTIALUSDT.parquet",
    ):
        _write(rel)

    # When
    paths = mhs_sealable_input_paths(data_root=tmp_path, execution_timeframe="3m")

    # Then: exactly the four complete-symbol files, no partial symbol at all
    names = {p.relative_to(tmp_path).as_posix() for p in paths}
    assert names == {
        "ohlcv/1h/COMPLETEUSDT.parquet",
        "ohlcv/3m/COMPLETEUSDT.parquet",
        "funding/COMPLETEUSDT.parquet",
        "markPriceKlines/1h/COMPLETEUSDT.parquet",
    }
    assert all("PARTIALUSDT" not in p.name for p in paths)
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
        "markPriceKlines/1h/COMPLETEUSDT.parquet",
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
