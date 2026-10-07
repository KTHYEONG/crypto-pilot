"""Ops CLI wiring contract (R8): provision-env builds before install, honors --dry-run."""

from __future__ import annotations

import pytest


def test_ops_provision_env_builds_before_install_and_respects_dry_run(tmp_path, monkeypatch) -> None:
    import src.cli.commands.ops as ops
    from src.cli.main import build_root_parser

    events: list[str] = []

    monkeypatch.setattr(ops, "build_runtime_fragment", lambda path: events.append("build") or "BINANCE_API_KEY=key\n")
    monkeypatch.setattr(ops, "install_runtime_fragment", lambda host, fragment: events.append(f"install:{host}"))

    source = tmp_path / ".quant.env"
    source.write_text("BINANCE_API_KEY=key\n", encoding="utf-8")

    parser = build_root_parser()

    args = parser.parse_args(["ops", "provision-env", "--host", "or-vps", "--source", str(source)])
    args.handler(args)
    assert events == ["build", "install:or-vps"]

    events.clear()
    dry_args = parser.parse_args(
        ["ops", "provision-env", "--host", "or-vps", "--source", str(source), "--dry-run"]
    )
    dry_args.handler(dry_args)
    assert events == ["build"]


def test_ops_procedure_registry_migrate_requires_explicit_paths() -> None:
    from src.cli.main import build_root_parser

    parser = build_root_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ops", "procedure-registry-migrate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["ops", "procedure-registry-migrate", "--legacy-path", "a"])


def test_ops_procedure_registry_migrate_moves_once(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.cli.main import build_root_parser

    legacy = tmp_path / "legacy.jsonl"
    target = tmp_path / "target.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"event": "registration"}\n', encoding="utf-8")
    payload = legacy.read_bytes()

    parser = build_root_parser()
    args = parser.parse_args(
        ["ops", "procedure-registry-migrate", "--legacy-path", str(legacy), "--target-path", str(target)]
    )
    with caplog.at_level(logging.INFO):
        args.handler(args)
    assert target.is_file()
    assert target.read_bytes() == payload
    assert not legacy.exists()
    assert any("moved=True" in r.message for r in caplog.records)

    caplog.clear()
    args = parser.parse_args(
        ["ops", "procedure-registry-migrate", "--legacy-path", str(legacy), "--target-path", str(target)]
    )
    with caplog.at_level(logging.INFO):
        args.handler(args)
    assert any("moved=False" in r.message for r in caplog.records)


def test_ops_procedure_registry_migrate_fails_closed_on_ambiguity(tmp_path) -> None:
    from src.cli.main import build_root_parser

    legacy = tmp_path / "legacy.jsonl"
    target = tmp_path / "target.jsonl"
    legacy.write_text('{"a": 1}\n', encoding="utf-8")
    target.write_text('{"b": 2}\n', encoding="utf-8")
    before_legacy = legacy.read_bytes()
    before_target = target.read_bytes()

    parser = build_root_parser()
    args = parser.parse_args(
        ["ops", "procedure-registry-migrate", "--legacy-path", str(legacy), "--target-path", str(target)]
    )
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1
    assert legacy.read_bytes() == before_legacy
    assert target.read_bytes() == before_target
