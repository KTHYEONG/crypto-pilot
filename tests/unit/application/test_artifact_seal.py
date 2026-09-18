"""Artifact seal source-ownership invariant guards."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest


def _write_key_env(monkeypatch: pytest.MonkeyPatch, key_bytes: bytes) -> str:
    encoded = base64.b64encode(key_bytes).decode("ascii")
    monkeypatch.setenv("LIVE_ARTIFACT_KEY", encoded)
    return encoded


def test_seal_round_trip_preserves_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    from src.application.ops.artifact_seal import main

    _write_key_env(monkeypatch, secrets.token_bytes(32))
    src = tmp_path / "plain.bin"
    src.write_bytes(b"artifact-payload-\x00\xff")
    sealed = tmp_path / "sealed.bin"
    opened = tmp_path / "opened.bin"
    assert main(["seal", "--in", str(src), "--out", str(sealed)]) == 0
    assert main(["unseal", "--in", str(sealed), "--out", str(opened)]) == 0
    assert opened.read_bytes() == b"artifact-payload-\x00\xff"


def test_unseal_legacy_envelope_matches_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    from src.application.ops.artifact_seal import main
    from src.live.crypto import MAGIC, derive_key, seal_bytes
    from pydantic import SecretStr

    raw_key = secrets.token_bytes(32)
    encoded = base64.b64encode(raw_key).decode("ascii")
    monkeypatch.setenv("LIVE_ARTIFACT_KEY", encoded)
    plaintext = b"legacy-envelope-check"
    blob = seal_bytes(plaintext, derive_key(SecretStr(encoded)))
    assert blob.startswith(MAGIC)
    sealed = tmp_path / "legacy.bin"
    sealed.write_bytes(blob)
    opened = tmp_path / "legacy.out"
    assert main(["unseal", "--in", str(sealed), "--out", str(opened)]) == 0
    assert opened.read_bytes() == plaintext


def test_seal_requires_key_without_exposure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from src.application.ops.artifact_seal import _key_from_env, main

    monkeypatch.delenv("LIVE_ARTIFACT_KEY", raising=False)
    src = tmp_path / "plain.bin"
    src.write_bytes(b"data")
    with pytest.raises(SystemExit) as exc:
        main(["seal", "--in", str(src), "--out", str(tmp_path / "out.bin")])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc2:
        _key_from_env()
    assert exc2.value.code == 2
    assert "LIVE_ARTIFACT_KEY" not in capsys.readouterr().out
    assert os.environ.get("LIVE_ARTIFACT_KEY") is None


def test_unseal_rejects_wrong_key_and_corrupt_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    from src.application.ops.artifact_seal import main

    _write_key_env(monkeypatch, secrets.token_bytes(32))
    src = tmp_path / "plain.bin"
    src.write_bytes(b"secret-data")
    sealed = tmp_path / "sealed.bin"
    assert main(["seal", "--in", str(src), "--out", str(sealed)]) == 0
    _write_key_env(monkeypatch, secrets.token_bytes(32))
    assert main(["unseal", "--in", str(sealed), "--out", str(tmp_path / "bad.out")]) == 1
    corrupt = tmp_path / "corrupt.bin"
    corrupt.write_bytes(b"not-a-valid-envelope")
    assert main(["unseal", "--in", str(corrupt), "--out", str(tmp_path / "corrupt.out")]) == 1


def test_keygen_emits_base64_32_bytes(capsys: pytest.CaptureFixture[str]) -> None:
    from src.application.ops.artifact_seal import main

    assert main(["keygen"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert len(base64.b64decode(out[0], validate=True)) == 32


def _expected_sealed_bytes(tmp_path: Path) -> bytes:
    from src.application.ops.artifact_seal import _key_from_env
    from src.live.crypto import derive_key, seal_bytes

    return seal_bytes((tmp_path / "plain.bin").read_bytes(), derive_key(_key_from_env()))


def test_cli_delegation_matches_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import secrets

    from src.cli.main import build_root_parser

    _write_key_env(monkeypatch, secrets.token_bytes(32))
    src = tmp_path / "plain.bin"
    src.write_bytes(b"cli-delegation")
    sealed = tmp_path / "sealed.bin"
    opened = tmp_path / "opened.bin"
    parser = build_root_parser()
    seal_args = parser.parse_args(["ops", "artifact-seal", "seal", "--in", str(src), "--out", str(sealed)])
    assert seal_args.handler(seal_args) is None
    assert sealed.read_bytes() == _expected_sealed_bytes(tmp_path)
    unseal_args = parser.parse_args(["ops", "artifact-seal", "unseal", "--in", str(sealed), "--out", str(opened)])
    assert unseal_args.handler(unseal_args) is None
    assert opened.read_bytes() == b"cli-delegation"
    key_args = parser.parse_args(["ops", "artifact-seal", "keygen"])
    assert key_args.handler(key_args) is None
    assert len(base64.b64decode(capsys.readouterr().out.strip().splitlines()[-1], validate=True)) == 32
    missing_args = parser.parse_args(["ops", "artifact-seal", "seal", "--in", str(tmp_path / "missing.bin"), "--out", str(tmp_path / "o.bin")])
    with pytest.raises(SystemExit) as exc:
        missing_args.handler(missing_args)
    assert exc.value.code == 2
    _write_key_env(monkeypatch, secrets.token_bytes(32))
    bad_args = parser.parse_args(["ops", "artifact-seal", "unseal", "--in", str(sealed), "--out", str(tmp_path / "bad.out")])
    with pytest.raises(SystemExit) as exc_bad:
        bad_args.handler(bad_args)
    assert exc_bad.value.code == 1


def test_ops_provision_and_migrate_still_registered() -> None:
    from src.cli.main import build_root_parser

    parser = build_root_parser()
    assert parser.parse_args(["ops", "provision-env", "--dry-run"]).ops_command == "provision-env"
    assert parser.parse_args(["ops", "artifact-seal", "keygen"]).ops_command == "artifact-seal"
    assert parser.parse_args(["ops", "daemon-idle-gate", "--heartbeat-file", "h", "--waited-s", "0"]).ops_command == "daemon-idle-gate"


def test_seal_io_failures_exit2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    from src.application.ops.artifact_seal import _read_bytes, _write_bytes, main

    _write_key_env(monkeypatch, secrets.token_bytes(32))
    with pytest.raises(SystemExit) as exc:
        _read_bytes(tmp_path / "does-not-exist.bin")
    assert exc.value.code == 2
    blocked = tmp_path / "blocked-dir"
    blocked.mkdir()
    with pytest.raises(SystemExit) as exc2:
        _write_bytes(blocked, b"")
    assert exc2.value.code == 2
    with pytest.raises(SystemExit) as exc3:
        main(["seal", "--in", str(tmp_path / "does-not-exist.bin"), "--out", str(tmp_path / "o.bin")])
    assert exc3.value.code == 2
