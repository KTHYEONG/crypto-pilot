"""VPS local-first restructuring: workstation-to-VPS secret provisioning contract (R1-R8)."""

from __future__ import annotations


def test_build_runtime_fragment_emits_declared_keys_in_canonical_order(tmp_path) -> None:
    from src.common.env_provisioning import RUNTIME_ENV_SPEC, build_runtime_fragment

    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "export BINANCE_API_KEY=binance-key",
                'export BINANCE_SECRET_KEY="binance-secret"',
                "LIVE_ARTIFACT_KEY='artifact-key'",
                "export LIVE_ALERT_GMAIL_USER=alert@example.com",
                "export LIVE_ALERT_GMAIL_APP_PASSWORD=alert-pass",
                "export LIVE_ORDER_API_KEY=order-key",
                "export LIVE_ORDER_API_SECRET=order-secret",
                "export LIVE_MODE=paper",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fragment = build_runtime_fragment(source)

    lines = fragment.splitlines()
    assert [line.partition("=")[0] for line in lines] == [key.target for key in RUNTIME_ENV_SPEC]
    assert len(lines) == 7
    assert fragment.endswith("\n")
    assert "export " not in fragment
    assert "BINANCE_SECRET_KEY=binance-secret" in lines
    assert "LIVE_ARTIFACT_KEY=artifact-key" in lines
    assert "LIVE_MODE" not in fragment


def test_build_runtime_fragment_accepts_binance_secret_alias(tmp_path) -> None:
    from src.common.env_provisioning import build_runtime_fragment

    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "BINANCE_API_KEY=binance-key",
                "BINANCE_SECRET=legacy-secret",
                "LIVE_ARTIFACT_KEY=artifact-key",
                "ALERT_GMAIL_USER=alert@example.com",
                "ALERT_GMAIL_APP_PASSWORD=alert-pass",
                "LIVE_ORDER_API_KEY=order-key",
                "LIVE_ORDER_API_SECRET=order-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fragment = build_runtime_fragment(source)
    lines = fragment.splitlines()

    assert "BINANCE_SECRET_KEY=legacy-secret" in lines
    assert "BINANCE_SECRET=" not in fragment
    assert "LIVE_ALERT_GMAIL_USER=alert@example.com" in lines
    assert "LIVE_ALERT_GMAIL_APP_PASSWORD=alert-pass" in lines


def test_build_runtime_fragment_rejects_missing_required_key(tmp_path) -> None:
    import pytest

    from src.common.env_provisioning import build_runtime_fragment
    from src.common.errors import ProvisioningError

    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "BINANCE_API_KEY=binance-key",
                "BINANCE_SECRET_KEY=binance-secret",
                "LIVE_ARTIFACT_KEY=artifact-key",
                "LIVE_ALERT_GMAIL_USER=alert@example.com",
                "LIVE_ALERT_GMAIL_APP_PASSWORD=alert-pass",
                "LIVE_ORDER_API_KEY=order-key",
                "LIVE_ORDER_API_SECRET=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProvisioningError, match="LIVE_ORDER_API_SECRET"):
        build_runtime_fragment(source)


def test_build_runtime_fragment_never_leaks_undeclared_keys(tmp_path) -> None:
    from src.common.env_provisioning import build_runtime_fragment

    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "BINANCE_API_KEY=binance-key",
                "BINANCE_SECRET_KEY=binance-secret",
                "LIVE_ARTIFACT_KEY=artifact-key",
                "LIVE_ALERT_GMAIL_USER=alert@example.com",
                "LIVE_ALERT_GMAIL_APP_PASSWORD=alert-pass",
                "LIVE_ORDER_API_KEY=order-key",
                "LIVE_ORDER_API_SECRET=order-secret",
                "KIS_APP_SECRET=kis-secret",
                "KIS_DATA_1_APP_SECRET=pool-secret",
                "TOSS_APP_SECRET=toss-secret",
                "LIVE_MODE=live_mainnet",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fragment = build_runtime_fragment(source)

    for forbidden in (
        "KIS_APP_SECRET",
        "kis-secret",
        "KIS_DATA_1_APP_SECRET",
        "pool-secret",
        "TOSS_APP_SECRET",
        "toss-secret",
        "LIVE_MODE",
        "live_mainnet",
    ):
        assert forbidden not in fragment


def test_runtime_env_spec_targets_are_valid_live_settings_fields() -> None:
    from src.common.env_provisioning import RUNTIME_ENV_SPEC
    from src.live.settings import LiveSettings

    accepted: set[str] = set()
    for field_name, field in LiveSettings.model_fields.items():
        accepted.add(f"LIVE_{field_name}".upper())
        alias = field.validation_alias
        choices = getattr(alias, "choices", None)
        if choices is not None:
            accepted.update(str(choice).upper() for choice in choices)
        elif isinstance(alias, str):
            accepted.add(alias.upper())

    live_targets = [key.target for key in RUNTIME_ENV_SPEC if key.target.startswith("LIVE_")]
    assert live_targets, "manifest must declare LIVE_* runtime keys"
    for target in live_targets:
        assert target in accepted, target
    assert "LIVE_MODE" not in [key.target for key in RUNTIME_ENV_SPEC]


def test_install_runtime_fragment_sends_values_only_on_ssh_stdin(monkeypatch) -> None:
    import subprocess

    import src.common.env_provisioning as provisioning

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(provisioning.subprocess, "run", fake_run)
    fragment = "BINANCE_SECRET_KEY=secret-value\n"

    provisioning.install_runtime_fragment("or-vps", fragment)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "ssh"
    assert args[1] == "or-vps"
    # 단일 문자열 인자: ssh가 argv[2:]를 공백으로 이어붙여 원격 셸에 전달하므로
    # 여러 인자로 나누면 개행 포함 스크립트가 재분리되어 환경변수가 유출된다.
    assert len(args) == 3
    assert args[2].startswith("bash -c ")
    assert kwargs["input"] == fragment
    assert kwargs["check"] is True
    assert kwargs["text"] is True
    assert "shell" not in kwargs
    assert all("secret-value" not in part for part in args)
    remote_script = args[2]
    assert provisioning.REMOTE_RUNTIME_ENV_PATH == "/home/ubuntu/quant-secrets/crypto-pilot.env"
    assert provisioning.REMOTE_RUNTIME_ENV_PATH in remote_script
    assert "set -euo pipefail" in remote_script
    assert "chmod 0700" in remote_script
    assert "chmod 600" in remote_script
    assert "chown ubuntu:ubuntu" in remote_script
    assert "mv -f" in remote_script


def test_parse_workstation_assignments_normalizes_and_rejects_duplicates(tmp_path) -> None:
    import pytest

    from src.common.env_provisioning import parse_workstation_assignments
    from src.common.errors import ProvisioningError

    accepted = frozenset({"ALPHA", "BETA", "GAMMA", "DELTA"})
    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "no_assignment_line",
                "export ALPHA=one",
                '  BETA = "two"  ',
                "GAMMA='three'",
                "DELTA=",
                "OUT_OF_SCOPE=first",
                "OUT_OF_SCOPE=second",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_workstation_assignments(source, accepted)

    assert parsed == {"ALPHA": "one", "BETA": "two", "GAMMA": "three"}

    duplicate = tmp_path / "dup.env"
    duplicate.write_text("ALPHA=one\nexport ALPHA=two\n", encoding="utf-8")

    with pytest.raises(ProvisioningError, match="ALPHA"):
        parse_workstation_assignments(duplicate, accepted)
