

from src.core.data_provenance import mhs_input_layout_for_lake


def test_input_manifest_detects_post_seal_mutation(tmp_path) -> None:
    import pandas as pd
    from src.core.data_provenance import DataEvidenceTier, seal_mhs_input_manifest, validate_mhs_input_manifest
    path = tmp_path/'BTCUSDT.parquet'
    pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [1.0]}).to_parquet(path)
    manifest = tmp_path/'inputs.json'
    seal_mhs_input_manifest([path], layout=mhs_input_layout_for_lake(tmp_path), output_path=manifest)
    with path.open('ab') as handle:
        handle.write(b'x')
    result = validate_mhs_input_manifest(manifest, layout=mhs_input_layout_for_lake(tmp_path), required_paths=[path])
    assert not result.valid
    assert result.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert 'STALE_INPUT_MANIFEST' in result.reason_codes


def test_input_manifest_requires_every_consumed_path(tmp_path) -> None:
    import pandas as pd
    from src.core.data_provenance import seal_mhs_input_manifest, validate_mhs_input_manifest
    a, b = tmp_path/'a.parquet', tmp_path/'b.parquet'
    frame = pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [1.0]})
    frame.to_parquet(a)
    frame.to_parquet(b)
    manifest = tmp_path/'inputs.json'
    seal_mhs_input_manifest([a], layout=mhs_input_layout_for_lake(tmp_path), output_path=manifest)
    result = validate_mhs_input_manifest(manifest, layout=mhs_input_layout_for_lake(tmp_path), required_paths=[a, b])
    assert not result.valid
    assert 'UNATTESTED_REQUIRED_FILE' in result.reason_codes


def test_input_manifest_numeric_timestamp_bounds_use_epoch_milliseconds(tmp_path) -> None:
    import json
    import pandas as pd
    from src.core.data_provenance import seal_mhs_input_manifest

    path = tmp_path / "BTCUSDT.parquet"
    pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    manifest = tmp_path / "inputs.json"
    seal_mhs_input_manifest([path], layout=mhs_input_layout_for_lake(tmp_path), output_path=manifest)
    entry = json.loads(manifest.read_text())["files"][0]
    assert entry["first_timestamp"].startswith("2025-01-01T00:00:00")
    assert entry["last_timestamp"].startswith("2025-01-01T00:00:00")


def test_forward_observation_digest_mismatch_cannot_upgrade() -> None:
    import pandas as pd
    from src.core.data_provenance import DataEvidenceTier, validate_forward_execution_observations
    records = pd.DataFrame({'decision_time': [pd.Timestamp('2025-01-01', tz='UTC')], 'observed_at': [pd.Timestamp('2025-01-01 00:01', tz='UTC')], 'strategy_digest': ['other']})
    result = validate_forward_execution_observations(records, strategy_digest='frozen')
    assert not result.valid
    assert result.tier is not DataEvidenceTier.FORWARD_OBSERVED
    assert 'STRATEGY_DIGEST_MISMATCH' in result.reason_codes


def test_required_input_paths_include_panel_execution_mark_and_funding(tmp_path) -> None:
    from src.core.data_provenance import resolve_required_mhs_input_paths
    paths = resolve_required_mhs_input_paths(layout=mhs_input_layout_for_lake(tmp_path), panel_symbols=['BTCUSDT'], execution_symbols=['BTCUSDT'], execution_timeframe='3m')
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
    from src.core.data_provenance import (
        DataEvidenceTier,
        seal_mhs_input_manifest,
        validate_forward_execution_observations,
        validate_mhs_input_manifest,
    )
    frame = pd.DataFrame({'datetime': [pd.Timestamp('2025-01-01', tz='UTC')], 'close': [1.0]})
    path = tmp_path / 'a.parquet'
    frame.to_parquet(path)
    absent = validate_mhs_input_manifest(None, layout=mhs_input_layout_for_lake(tmp_path), required_paths=[path])
    assert not absent.valid
    assert absent.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert 'MISSING_INPUT_MANIFEST' in absent.reason_codes
    manifest = tmp_path / 'inputs.json'
    digest = seal_mhs_input_manifest([path], layout=mhs_input_layout_for_lake(tmp_path), output_path=manifest)
    assert isinstance(digest, str)
    assert len(digest) == 64
    ok_result = validate_mhs_input_manifest(manifest, layout=mhs_input_layout_for_lake(tmp_path), required_paths=[path])
    assert ok_result.valid
    assert ok_result.tier is DataEvidenceTier.REPRODUCIBLE_ARCHIVE
    assert ok_result.manifest_digest == digest
    legacy = pd.DataFrame({'decision_time': [pd.Timestamp('2025-01-01', tz='UTC')]})
    schema_result = validate_forward_execution_observations(legacy, strategy_digest='frozen')
    assert not schema_result.valid
    assert 'FORWARD_OBSERVATION_SCHEMA_MISMATCH' in schema_result.reason_codes
    bad_time = pd.DataFrame({
        'decision_time': [pd.Timestamp('2025-01-02', tz='UTC')],
        'observed_at': [pd.Timestamp('2025-01-01', tz='UTC')],
        'strategy_digest': ['frozen'],
    })
    time_result = validate_forward_execution_observations(bad_time, strategy_digest='frozen')
    assert not time_result.valid
    assert 'FORWARD_OBSERVATION_TIME_INVALID' in time_result.reason_codes
    short = pd.DataFrame({
        'decision_time': [pd.Timestamp('2025-01-01', tz='UTC')],
        'observed_at': [pd.Timestamp('2025-01-01 00:01', tz='UTC')],
        'strategy_digest': ['frozen'],
    })
    short_result = validate_forward_execution_observations(short, strategy_digest='frozen')
    assert not short_result.valid
    assert 'FORWARD_EVIDENCE_INCOMPLETE' in short_result.reason_codes
    long_enough = pd.DataFrame({
        'decision_time': [pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-04-15', tz='UTC')],
        'observed_at': [pd.Timestamp('2025-01-01 00:01', tz='UTC'), pd.Timestamp('2025-04-15 00:01', tz='UTC')],
        'strategy_digest': ['frozen', 'frozen'],
    })
    good = validate_forward_execution_observations(long_enough, strategy_digest='frozen')
    assert good.valid
    assert good.tier is DataEvidenceTier.FORWARD_OBSERVED
    assert good.files_checked == 2

def test_mhs_sealable_input_paths_skips_incomplete_symbols(tmp_path) -> None:
    # Given: COMPLETEUSDT has all three required files, PARTIALUSDT lacks funding
    import pandas as pd

    from src.core.data_provenance import mhs_sealable_input_paths

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
    paths = mhs_sealable_input_paths(layout=mhs_input_layout_for_lake(tmp_path), execution_timeframe="3m")

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

    from src.core.data_provenance import (
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
    sealable = mhs_sealable_input_paths(layout=mhs_input_layout_for_lake(tmp_path), execution_timeframe="3m")

    # When
    digest = seal_mhs_input_manifest(sealable, layout=mhs_input_layout_for_lake(tmp_path), output_path=manifest)
    required = resolve_required_mhs_input_paths(
        layout=mhs_input_layout_for_lake(tmp_path), panel_symbols=["COMPLETEUSDT"],
        execution_symbols=["COMPLETEUSDT"], execution_timeframe="3m",
    )
    result = validate_mhs_input_manifest(manifest, layout=mhs_input_layout_for_lake(tmp_path), required_paths=required)

    # Then: the sealed digest is carried and the tier clears the I3 gate
    assert result.valid is True
    assert result.tier is DataEvidenceTier.REPRODUCIBLE_ARCHIVE
    assert result.reason_codes == ()
    assert result.manifest_digest == digest


def test_required_files_are_exactly_consumed_sources() -> None:
    from pathlib import Path

    from src.core.data_provenance import resolve_required_mhs_input_paths

    paths = resolve_required_mhs_input_paths(
        layout=mhs_input_layout_for_lake(Path("/data")),
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

    from src.core.data_provenance import mhs_sealable_input_paths, seal_mhs_input_manifest

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
    first = mhs_sealable_input_paths(layout=mhs_input_layout_for_lake(tmp_path), execution_timeframe="3m")
    digest_first = seal_mhs_input_manifest(first, layout=mhs_input_layout_for_lake(tmp_path), output_path=tmp_path / "m1.json")
    _write("markPriceKlines/1h/AUSDT.parquet")
    second = mhs_sealable_input_paths(layout=mhs_input_layout_for_lake(tmp_path), execution_timeframe="3m")
    digest_second = seal_mhs_input_manifest(second, layout=mhs_input_layout_for_lake(tmp_path), output_path=tmp_path / "m2.json")
    assert digest_first == digest_second
    assert {p.as_posix() for p in first} == {p.as_posix() for p in second}


def test_missing_execution_source_remains_visible(tmp_path) -> None:
    import pandas as pd

    from src.core.data_provenance import (
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    for rel in ("ohlcv/1h/AUSDT.parquet", "funding/AUSDT.parquet"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    required = resolve_required_mhs_input_paths(
        layout=mhs_input_layout_for_lake(tmp_path), panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    assert any(p.as_posix().endswith("ohlcv/3m/AUSDT.parquet") for p in required)
    existing = [p for p in required if p.exists()]
    digest = seal_mhs_input_manifest(existing, layout=mhs_input_layout_for_lake(tmp_path), output_path=tmp_path / "m.json")
    assert isinstance(digest, str)
    result = validate_mhs_input_manifest(tmp_path / "m.json", layout=mhs_input_layout_for_lake(tmp_path), required_paths=required)
    assert not result.valid
    assert "UNATTESTED_REQUIRED_FILE" in result.reason_codes


def test_missing_funding_remains_visible(tmp_path) -> None:
    import pandas as pd

    from src.core.data_provenance import (
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    for rel in ("ohlcv/1h/AUSDT.parquet", "ohlcv/3m/AUSDT.parquet"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)
    required = resolve_required_mhs_input_paths(
        layout=mhs_input_layout_for_lake(tmp_path), panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    assert any(p.as_posix().endswith("funding/AUSDT.parquet") for p in required)
    existing = [p for p in required if p.exists()]
    seal_mhs_input_manifest(existing, layout=mhs_input_layout_for_lake(tmp_path), output_path=tmp_path / "m.json")
    result = validate_mhs_input_manifest(tmp_path / "m.json", layout=mhs_input_layout_for_lake(tmp_path), required_paths=required)
    assert not result.valid
    assert "UNATTESTED_REQUIRED_FILE" in result.reason_codes


def test_input_identity_is_deterministic(tmp_path) -> None:
    import pandas as pd

    from src.core.data_provenance import (
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
        layout=mhs_input_layout_for_lake(tmp_path), panel_symbols=["AUSDT", "BUSDT"], execution_symbols=["BUSDT", "AUSDT"], execution_timeframe="3m",
    )
    reverse = resolve_required_mhs_input_paths(
        layout=mhs_input_layout_for_lake(tmp_path), panel_symbols=["BUSDT", "AUSDT"], execution_symbols=["AUSDT", "BUSDT"], execution_timeframe="3m",
    )
    assert list(forward) == list(reverse)
    digest_forward = seal_mhs_input_manifest(list(forward), layout=mhs_input_layout_for_lake(tmp_path), output_path=tmp_path / "m1.json")
    digest_reverse = seal_mhs_input_manifest(list(reversed(reverse)), layout=mhs_input_layout_for_lake(tmp_path), output_path=tmp_path / "m2.json")
    assert digest_forward == digest_reverse


def test_default_layout_is_canonical_lake() -> None:
    from src.common.paths import FUTURES_DATA_DIR
    from src.core.data_provenance import resolve_mhs_input_layout

    layout = resolve_mhs_input_layout(None)
    assert layout.ohlcv_root == FUTURES_DATA_DIR / "ohlcv"
    assert layout.funding_root == FUTURES_DATA_DIR / "funding"
    assert layout.lake_root == FUTURES_DATA_DIR
    assert resolve_mhs_input_layout(str(FUTURES_DATA_DIR / "ohlcv")).lake_root == FUTURES_DATA_DIR


def test_canonical_shaped_override_keeps_lake(tmp_path, monkeypatch) -> None:
    import src.core.data_provenance as provenance
    from src.core.data_provenance import resolve_mhs_input_layout

    lake = tmp_path / "futures"
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", lake)
    monkeypatch.setattr(provenance, "FUTURES_DATA_DIR", lake)
    layout = resolve_mhs_input_layout(lake / "ohlcv")
    assert layout.lake_root == lake
    assert layout.ohlcv_root == lake / "ohlcv"
    assert layout.funding_root == lake / "funding"


def test_foreign_override_has_no_lake(tmp_path) -> None:
    from src.common.paths import FUTURES_DATA_DIR
    from src.core.data_provenance import resolve_mhs_input_layout

    layout = resolve_mhs_input_layout(tmp_path / "synthetic")
    assert layout.lake_root is None
    assert layout.funding_root == FUTURES_DATA_DIR / "funding"
    assert layout.ohlcv_root == tmp_path / "synthetic"
    assert resolve_mhs_input_layout(tmp_path / "ohlcv").lake_root is None


def test_non_path_override_rejected() -> None:
    import pytest

    from src.core.data_provenance import resolve_mhs_input_layout

    with pytest.raises(TypeError):
        resolve_mhs_input_layout(123)  # type: ignore[arg-type]


def _write_lake_parquet(lake, rel: str) -> None:
    import pandas as pd

    path = lake / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [1735689600000], "close": [1.0]}).to_parquet(path)


def test_sealed_corpus_validates_through_ohlcv_root_override(tmp_path, monkeypatch) -> None:
    import src.core.data_provenance as provenance
    from src.core.data_provenance import (
        DataEvidenceTier,
        mhs_input_layout_for_lake,
        mhs_sealable_input_paths,
        resolve_mhs_input_layout,
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    lake = tmp_path / "futures"
    for rel in ("ohlcv/1h/AUSDT.parquet", "ohlcv/3m/AUSDT.parquet", "funding/AUSDT.parquet"):
        _write_lake_parquet(lake, rel)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", lake)
    monkeypatch.setattr(provenance, "FUTURES_DATA_DIR", lake)
    seal_layout = mhs_input_layout_for_lake(lake)
    manifest = tmp_path / "m.json"
    seal_mhs_input_manifest(
        mhs_sealable_input_paths(layout=seal_layout, execution_timeframe="3m"),
        layout=seal_layout, output_path=manifest,
    )
    layout = resolve_mhs_input_layout(lake / "ohlcv")
    required = resolve_required_mhs_input_paths(
        layout=layout, panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    assert not any("ohlcv/ohlcv" in p.as_posix() or "ohlcv/funding" in p.as_posix() for p in required)
    result = validate_mhs_input_manifest(manifest, layout=layout, required_paths=required)
    assert result.tier is DataEvidenceTier.REPRODUCIBLE_ARCHIVE
    assert result.valid is True


def test_default_validation_matches_ohlcv_override(tmp_path, monkeypatch) -> None:
    import src.core.data_provenance as provenance
    from src.core.data_provenance import (
        mhs_input_layout_for_lake,
        resolve_mhs_input_layout,
        resolve_required_mhs_input_paths,
        validate_mhs_input_manifest,
    )

    lake = tmp_path / "futures"
    for rel in ("ohlcv/1h/AUSDT.parquet", "ohlcv/3m/AUSDT.parquet", "funding/AUSDT.parquet"):
        _write_lake_parquet(lake, rel)
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", lake)
    monkeypatch.setattr(provenance, "FUTURES_DATA_DIR", lake)
    seal_layout = mhs_input_layout_for_lake(lake)
    manifest = tmp_path / "m.json"
    from src.core.data_provenance import mhs_sealable_input_paths, seal_mhs_input_manifest

    seal_mhs_input_manifest(
        mhs_sealable_input_paths(layout=seal_layout, execution_timeframe="3m"),
        layout=seal_layout, output_path=manifest,
    )
    override_required = resolve_required_mhs_input_paths(
        layout=resolve_mhs_input_layout(lake / "ohlcv"),
        panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    default_required = resolve_required_mhs_input_paths(
        layout=resolve_mhs_input_layout(None),
        panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    assert {p.as_posix() for p in override_required} == {p.as_posix() for p in default_required}
    override_result = validate_mhs_input_manifest(
        manifest, layout=resolve_mhs_input_layout(lake / "ohlcv"), required_paths=override_required,
    )
    default_result = validate_mhs_input_manifest(
        manifest, layout=resolve_mhs_input_layout(None), required_paths=default_required,
    )
    assert override_result.valid
    assert default_result.valid
    assert override_result.manifest_digest == default_result.manifest_digest


def test_foreign_root_never_claims_reproducibility(tmp_path) -> None:
    from src.core.data_provenance import (
        DataEvidenceTier,
        mhs_input_layout_for_lake,
        resolve_mhs_input_layout,
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    lake = tmp_path / "lake"
    _write_lake_parquet(lake, "ohlcv/1h/AUSDT.parquet")
    manifest = tmp_path / "m.json"
    seal_mhs_input_manifest([lake / "ohlcv" / "1h" / "AUSDT.parquet"], layout=mhs_input_layout_for_lake(lake), output_path=manifest)
    foreign = resolve_mhs_input_layout(tmp_path / "synthetic")
    assert foreign.lake_root is None
    required = resolve_required_mhs_input_paths(
        layout=foreign, panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    result = validate_mhs_input_manifest(manifest, layout=foreign, required_paths=required)
    assert result.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert result.valid is False
    assert result.reason_codes == ("NONCANONICAL_DATA_ROOT",)
    missing = validate_mhs_input_manifest(None, layout=foreign, required_paths=required)
    assert missing.reason_codes == ("MISSING_INPUT_MANIFEST",)


def test_foreign_root_with_unreadable_manifest_stays_unsealed(tmp_path) -> None:
    from src.core.data_provenance import (
        DataEvidenceTier,
        resolve_mhs_input_layout,
        resolve_required_mhs_input_paths,
        validate_mhs_input_manifest,
    )

    corrupt = tmp_path / "m.json"
    corrupt.write_bytes(b"\xff\xfe")
    foreign = resolve_mhs_input_layout(tmp_path / "synthetic")
    required = resolve_required_mhs_input_paths(
        layout=foreign, panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    result = validate_mhs_input_manifest(corrupt, layout=foreign, required_paths=required)
    assert result.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert result.valid is False
    assert result.reason_codes == ("NONCANONICAL_DATA_ROOT",)
    assert result.manifest_digest is None
    assert result.files_checked == 0


def test_required_path_outside_lake_raises(tmp_path) -> None:
    import pytest

    from src.core.data_provenance import seal_mhs_input_manifest, validate_mhs_input_manifest

    lake = tmp_path / "lake"
    _write_lake_parquet(lake, "ohlcv/1h/AUSDT.parquet")
    layout = mhs_input_layout_for_lake(lake)
    manifest = tmp_path / "m.json"
    seal_mhs_input_manifest([lake / "ohlcv/1h/AUSDT.parquet"], layout=layout, output_path=manifest)
    with pytest.raises(ValueError, match="not in the subpath"):
        validate_mhs_input_manifest(manifest, layout=layout, required_paths=[tmp_path / "foreign.parquet"])


def test_sealing_without_lake_fails_before_io(tmp_path) -> None:
    import pytest

    from src.core.data_provenance import resolve_mhs_input_layout, seal_mhs_input_manifest

    foreign = resolve_mhs_input_layout(tmp_path / "synthetic")
    out = tmp_path / "never.json"
    with pytest.raises(ValueError, match="single lake root"):
        seal_mhs_input_manifest([tmp_path / "x.parquet"], layout=foreign, output_path=out)
    assert not out.exists()


def test_stale_file_still_detected(tmp_path) -> None:
    import pandas as pd

    from src.core.data_provenance import (
        DataEvidenceTier,
        mhs_input_layout_for_lake,
        mhs_sealable_input_paths,
        resolve_required_mhs_input_paths,
        seal_mhs_input_manifest,
        validate_mhs_input_manifest,
    )

    for rel in ("ohlcv/1h/AUSDT.parquet", "ohlcv/3m/AUSDT.parquet", "funding/AUSDT.parquet"):
        _write_lake_parquet(tmp_path, rel)
    layout = mhs_input_layout_for_lake(tmp_path)
    manifest = tmp_path / "m.json"
    seal_mhs_input_manifest(
        mhs_sealable_input_paths(layout=layout, execution_timeframe="3m"),
        layout=layout, output_path=manifest,
    )
    target = tmp_path / "ohlcv" / "3m" / "AUSDT.parquet"
    pd.DataFrame({"timestamp": [1735776000000], "close": [2.0]}).to_parquet(target)
    required = resolve_required_mhs_input_paths(
        layout=layout, panel_symbols=["AUSDT"], execution_symbols=["AUSDT"], execution_timeframe="3m",
    )
    result = validate_mhs_input_manifest(manifest, layout=layout, required_paths=required)
    assert result.tier is DataEvidenceTier.UNSEALED_ARCHIVE
    assert "STALE_INPUT_MANIFEST" in result.reason_codes
