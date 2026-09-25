"""SCENARIO_LIVE_DAEMON_11: 배포 자산(Dockerfile/compose/.dockerignore) 정적 계약.

실제 docker build는 CI와 로컬 검증 명령(`docker build -t crypto-pilot-live:local .`)에서
수행하며, 여기서는 시크릿 미포함과 상태 보존 마운트를 소스 레벨로 검증한다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_SCENARIO_LIVE_DAEMON_11_DOCKERFILE_BUILDS() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["uv", "run", "python", "-m", "src.cli.main", "live", "daemon"]' in dockerfile
    # 어떤 레이어도 .env나 키 파일을 굽지 않는다(I-NO-SECRET-IN-IMAGE).
    assert ".env" not in dockerfile
    assert "pem" not in dockerfile

    # .dockerignore의 제외 패턴이 시크릿을 계속 걸러낸다.
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in dockerignore
    assert "*.pem" in dockerignore

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "env_file: /home/ubuntu/quant-secrets/crypto-pilot.env" in compose  # 시크릿은 env_file 주입만 허용된다
    assert "./data/state:/app/data/state" in compose  # I-STATE-SURVIVES-REDEPLOY
    assert "./logs:/app/logs" in compose


def test_docker_compose_has_independent_market_recorder_service() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # 라이브 전용 소스(bookTicker/premiumIndex/reference) + 청산 스트림은 live 데몬과 독립된 서비스로 24/7 가동된다.
    assert "src.market_data.streams.recorder_main" in compose
    assert "/app/.venv/bin/python" in compose
    assert "container_name: market-recorder" in compose
    assert "unless-stopped" in compose
    assert "./data/futures/liquidations:/app/data/futures/liquidations" in compose
    assert "./data/live_capture:/app/data/live_capture" in compose
    assert "mem_limit: 768m" in compose
    assert "stop_grace_period: 30s" in compose
    assert "liquidation-collector" not in compose
    # 기존 live 데몬 서비스 계약이 깨지지 않는다.
    assert "./data/state:/app/data/state" in compose
    # recorder 블록은 시크릿과 state 마운트 없이 최소 권한으로 동작한다.
    live_block, recorder_block = compose.split("  market-recorder:\n", 1)
    assert "env_file" not in recorder_block
    assert "./data/state" not in recorder_block
    assert "./logs:/app/logs" in recorder_block
    assert "env_file: /home/ubuntu/quant-secrets/crypto-pilot.env" in live_block
    assert "./data/state:/app/data/state" in live_block


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_11_DOCKERFILE_BUILDS",
    "test_docker_compose_has_independent_market_recorder_service",
    "test_dockerfile_keeps_uv_cache_out_of_image",
    "test_dockerignore_excludes_workspace_caches",
    "test_compose_uses_absolute_secret_path_and_declares_live_mode",
    "test_deploy_workflow_builds_native_arm64_and_tags_commit_sha",
)


def test_deploy_workflow_waits_for_daemon_idle_gate_before_recreate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

    gate = workflow.index("python3 -m src.application.ops.daemon_idle_gate")
    recreate = workflow.index("compose_recreate.sh")
    assert gate < recreate
    assert "cat ~/crypto-pilot/data/state/live_daemon_heartbeat.json" in workflow
    assert '--waited-s "$waited"' in workflow
    assert '"$rc" -ne 10' in workflow
    assert "sleep 60" in workflow
    assert 'scp $SSH_OPTS deploy/compose_recreate.sh "$REMOTE_USER@$HOST:~/crypto-pilot/deploy/"' in workflow
    # recorder는 fingerprint 스크립트가 판단한다. 베어 recreate로 recorder까지 재시작하면 안 된다.
    assert "up -d --force-recreate --remove-orphans\n" not in workflow
    assert "$C up -d --force-recreate" not in workflow


def test_docker_compose_memory_budget_fits_oci_a1_host() -> None:
    from pathlib import Path

    # Given
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    # When
    live_block, recorder_block = compose.split("  market-recorder:\n", 1)

    # Then
    assert "container_name: mhs-live-daemon" in live_block
    assert "mem_limit: 2g" in live_block
    assert "mem_limit: 1200m" not in compose
    assert "container_name: market-recorder" in recorder_block
    assert "mem_limit: 768m" in recorder_block
    assert "HARDWARE_MAX_WORKERS" not in compose


def test_dockerfile_keeps_uv_cache_out_of_image() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.splitlines()[0].startswith("# syntax=docker/dockerfile:1")
    assert "UV_CACHE_DIR=/root/.cache/uv" in dockerfile
    assert dockerfile.count("--mount=type=cache,target=/root/.cache/uv,sharing=locked") == 2
    assert "UV_NO_SYNC=1" in dockerfile
    assert 'CMD ["uv", "run", "python", "-m", "src.cli.main", "live", "daemon"]' in dockerfile
    assert "scratch/uv-cache" not in dockerfile


def test_dockerignore_excludes_workspace_caches() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")

    for entry in ("scratch/", "tmp/", ".mypy_cache", ".ruff_cache", ".serena"):
        assert entry in dockerignore, entry
    assert ".env" in dockerignore
    assert "*.pem" in dockerignore


def test_compose_uses_absolute_secret_path_and_declares_live_mode() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("env_file: /home/ubuntu/quant-secrets/crypto-pilot.env") == 1
    assert "env_file: .env" not in compose
    assert "LIVE_MODE=paper" in compose
    assert "live_mainnet" not in compose
    assert "./data/state:/app/data/state" in compose
    assert "./logs:/app/logs" in compose


def test_deploy_workflow_builds_native_arm64_and_tags_commit_sha() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04-arm" in workflow
    assert "platforms: linux/arm64" in workflow
    assert "linux/amd64" not in workflow
    assert "setup-qemu-action" not in workflow
    assert "sha-${{ github.sha }}" in workflow
    assert "${{ env.IMAGE }}:latest" in workflow
    assert "python3 -m src.application.ops.daemon_idle_gate" in workflow


def test_rclone_filter_keeps_live_capture() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "deploy" / "crypto-pilot.rclone-filter").read_text(encoding="utf-8")

    assert "+ /live_capture/**" in text
    assert text.index("+ /live_capture/**") < text.index("- /futures/**")
    assert text.index("+ /live_capture/**") < text.index("- **")
    rules = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    first_include = next(i for i, line in enumerate(rules) if line.startswith("+ "))
    assert "- *.tmp" in rules
    assert "- *.tmp-*" in rules
    assert rules.index("- *.tmp") < first_include
    assert rules.index("- *.tmp-*") < first_include


def test_rclone_filter_excludes_coverage_temp_names() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "deploy" / "crypto-pilot.rclone-filter").read_text(encoding="utf-8")
    rules = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    first_include = next(i for i, line in enumerate(rules) if line.startswith("+ "))
    # String-level check is sufficient: the two transient patterns must precede any include,
    # so `.20260924.jsonl.tmp-123` (-> `- *.tmp-*`) and `.10.parquet.tmp` (-> `- *.tmp`)
    # are matched by an exclusion before any include.
    assert rules.index("- *.tmp") < first_include
    assert rules.index("- *.tmp-*") < first_include


def test_gate_waits_on_fresh_busy_heartbeat() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import BUSY_STAGES, decide_deploy

    assert frozenset({"refresh", "signal", "execute"}) == BUSY_STAGES
    now = datetime.fromisoformat("2026-09-15T10:30:00+00:00")
    heartbeat = {"stage": "signal", "status": "RUNNING", "ts": "2026-09-15T10:29:00+00:00"}
    decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert decision.action == "wait"
    assert decision.reason == "busy:signal"


def test_gate_proceeds_on_idle_heartbeat() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy, main

    now = datetime.fromisoformat("2026-09-15T10:30:00+00:00")
    decision = decide_deploy(
        {"stage": "idle", "ts": "2026-09-15T10:29:00+00:00"},
        now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0,
    )
    assert (decision.action, decision.reason) == ("proceed", "idle")
    import json
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        hb = _Path(tmp) / "hb.json"
        hb.write_text(json.dumps({"stage": "idle", "ts": "2026-09-15T10:29:00+00:00"}), encoding="utf-8")
        assert main(["--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T10:30:00+00:00"]) == 0


def test_gate_preserves_strict_stale_boundary() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy

    now = datetime.fromisoformat("2026-09-15T10:30:00+00:00")
    at_threshold = {"stage": "execute", "ts": "2026-09-15T09:45:00+00:00"}
    over_threshold = {"stage": "execute", "ts": "2026-09-15T09:44:59+00:00"}
    at_decision = decide_deploy(at_threshold, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert at_decision.action == "wait"
    over_decision = decide_deploy(over_threshold, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert over_decision.action == "proceed_stale"
    assert over_decision.reason.startswith("stale:execute age_s=")


def test_gate_proceeds_on_wait_timeout() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy, main

    now = datetime.fromisoformat("2026-09-15T10:30:00+00:00")
    heartbeat = {"stage": "signal", "status": "RUNNING", "ts": "2026-09-15T10:29:30+00:00"}
    decision = decide_deploy(heartbeat, now=now, waited_s=3600.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert decision.action == "proceed_timeout"
    assert decision.reason == "max_wait:signal waited_s=3600"
    import json
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        hb = _Path(tmp) / "hb.json"
        hb.write_text(json.dumps(heartbeat), encoding="utf-8")
        assert main(["--heartbeat-file", str(hb), "--waited-s", "3600", "--max-wait-s", "3600", "--now", "2026-09-15T10:30:00+00:00"]) == 0


def test_gate_keeps_malformed_heartbeat_behavior(tmp_path, capsys) -> None:
    import json

    from src.application.ops.daemon_idle_gate import decide_deploy, main

    from datetime import datetime

    now = datetime.fromisoformat("2026-09-15T10:30:00+00:00")
    for heartbeat in (None, {}, {"stage": "signal"}, {"stage": "signal", "ts": "not-a-time"}):
        decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
        assert (decision.action, decision.reason) == ("proceed", "no_heartbeat")
    import pytest

    with pytest.raises(ValueError, match="tz-aware"):
        decide_deploy({"stage": "idle", "ts": "2026-09-15T10:29:00+00:00"}, now=datetime(2026, 9, 15, 1, 30), waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    busy = tmp_path / "busy.json"
    busy.write_text(json.dumps({"stage": "signal", "ts": "2026-09-15T10:29:00+00:00"}), encoding="utf-8")
    assert main(["--heartbeat-file", str(busy), "--waited-s", "0", "--now", "2026-09-15T10:30:00+00:00"]) == 10
    assert capsys.readouterr().out.strip() == "action=wait reason=busy:signal"
    for raw in ("", "<html>", "[1, 2]"):
        target = tmp_path / "case.json"
        target.write_text(raw, encoding="utf-8")
        assert main(["--heartbeat-file", str(target), "--waited-s", "0", "--now", "2026-09-15T10:30:00+00:00"]) == 0
        assert capsys.readouterr().out.strip() == "action=proceed reason=no_heartbeat"


def test_gate_runs_under_bare_python_and_matches_workflow(tmp_path) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "python3 -m src.application.ops.daemon_idle_gate" in workflow
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"stage": "refresh", "ts": "2026-09-15T10:29:00+00:00"}), encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter + the repo gate module
        [sys.executable, "-I", "-m", "src.application.ops.daemon_idle_gate", "--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T10:30:00+00:00"],
        capture_output=True, text=True, check=False, cwd=root,
    )
    assert result.returncode == 10
    assert result.stdout.strip() == "action=wait reason=busy:refresh"


def test_ops_cli_delegates_daemon_idle_gate(tmp_path, capsys) -> None:
    import json

    import pytest

    from src.cli.main import build_root_parser

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"stage": "signal", "ts": "2026-09-15T10:29:00+00:00"}), encoding="utf-8")
    parser = build_root_parser()
    args = parser.parse_args(["ops", "daemon-idle-gate", "--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T10:30:00+00:00"])
    with pytest.raises(SystemExit) as exc:
        args.handler(args)
    assert exc.value.code == 10
    assert capsys.readouterr().out.strip() == "action=wait reason=busy:signal"
    idle = tmp_path / "idle.json"
    idle.write_text(json.dumps({"stage": "idle", "ts": "2026-09-15T10:29:00+00:00"}), encoding="utf-8")
    idle_args = parser.parse_args(["ops", "daemon-idle-gate", "--heartbeat-file", str(idle), "--waited-s", "0", "--now", "2026-09-15T10:30:00+00:00"])
    assert idle_args.handler(idle_args) is None


def test_backup_timer_covers_both_slots() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    timer = (root / "deploy" / "backup" / "crypto-pilot-backup.timer").read_text(encoding="utf-8")
    assert "OnCalendar=*-*-* 00:15:00 UTC" in timer
    assert "OnCalendar=*-*-* 12:30:00 UTC" in timer
    assert "Persistent=true" in timer


def test_backup_service_alerts_on_failure_and_outlives_lock_wait() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    service = (root / "deploy" / "backup" / "crypto-pilot-backup.service").read_text(encoding="utf-8")
    assert "OnFailure=kca-alert@%n.service" in service
    assert "Type=oneshot" in service
    assert "ExecStart=%h/crypto-pilot/deploy/backup/crypto-pilot-backup.sh" in service
    assert "TimeoutStartSec=3h" in service


def test_deploy_workflow_installs_and_enables_backup_unit() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert 'deploy/crypto-pilot.rclone-filter "$REMOTE_USER@$HOST:~/crypto-pilot/deploy/crypto-pilot.rclone-filter.new"' in workflow
    assert 'deploy/backup/crypto-pilot-backup.sh "$REMOTE_USER@$HOST:~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh.new"' in workflow
    assert "mv -f ~/crypto-pilot/deploy/crypto-pilot.rclone-filter.new ~/crypto-pilot/deploy/crypto-pilot.rclone-filter" in workflow
    assert "mv -f ~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh.new ~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh" in workflow
    assert workflow.index("mv -f ~/crypto-pilot/deploy/crypto-pilot.rclone-filter.new") < workflow.index("chmod +x ~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh")
    assert workflow.index("mv -f ~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh.new") < workflow.index("chmod +x ~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh")
    assert workflow.index("mv -f ~/crypto-pilot/deploy/crypto-pilot.rclone-filter.new") < workflow.index("systemctl --user daemon-reload")
    assert workflow.index("mv -f ~/crypto-pilot/deploy/backup/crypto-pilot-backup.sh.new") < workflow.index("systemctl --user daemon-reload")
    assert "deploy/backup/crypto-pilot-backup.service" in workflow
    assert "deploy/backup/crypto-pilot-backup.timer" in workflow
    assert "systemctl --user daemon-reload" in workflow
    assert "systemctl --user enable --now crypto-pilot-backup.timer" in workflow


_FAKE_DOCKER_SCRIPT = """#!/usr/bin/env bash
echo "docker $*" >> "$FAKE_LOG"
op="$1"; shift || true
case "$op" in
  compose)
    sub="$1"; shift || true
    case "$sub" in
      version) exit 0 ;;
      pull) exit 0 ;;
      config) printf '%s\\n' "$FAKE_CONFIG_HASH"; exit 0 ;;
      up)
        if [ -n "${FAKE_FAIL_UP:-}" ]; then
          case "$*" in
            *"$FAKE_FAIL_UP"*) exit 1 ;;
          esac
        fi
        exit 0 ;;
      *) exit 0 ;;
    esac
    ;;
  exec)
    if [ "${FAKE_INSPECT_RUNNING:-}" != "true" ]; then exit 1; fi
    if [ "${FAKE_EXEC_FAIL:-0}" = "1" ]; then exit 1; fi
    printf '%s\\n' "$FAKE_RUNNING_FP"
    exit 0
    ;;
  run)
    if [ "${FAKE_RUN_FAIL:-0}" = "1" ]; then exit 1; fi
    printf '%s\\n' "$FAKE_IMAGE_FP"
    exit 0
    ;;
  inspect)
    if [ "${FAKE_INSPECT_RUNNING:-}" = "fail" ]; then exit 1; fi
    case "$*" in
      *State.Running*)
        if [ "${FAKE_INSPECT_RUNNING:-}" = "true" ]; then echo "true"; else echo "false"; fi
        exit 0 ;;
      *config-hash*) printf '%s\\n' "$FAKE_LABEL"; exit 0 ;;
      *) exit 1 ;;
    esac
    ;;
  image) exit 0 ;;
  *) exit 0 ;;
esac
"""


def _run_compose_recreate(tmp_path, env: dict[str, str]):
    """Replay `deploy/compose_recreate.sh` against a fake `docker` on PATH."""
    import os
    import shutil
    import stat
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(_FAKE_DOCKER_SCRIPT, encoding="utf-8")
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "argv.log"
    log.write_text("", encoding="utf-8")
    full_env = dict(os.environ)
    full_env.update(env)
    full_env["FAKE_LOG"] = str(log)
    full_env["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    result = subprocess.run(  # noqa: S603 - fixed argv: the repo script under test + a fake image ref
        [str(shutil.which("bash") or "/bin/bash"), str(ROOT / "deploy" / "compose_recreate.sh"), "fake-image:latest"],
        capture_output=True, text=True, check=False, cwd=tmp_path, env=full_env,
    )
    return result, [line for line in log.read_text(encoding="utf-8").splitlines() if line]


def _base_recreate_env() -> dict[str, str]:
    return {
        "FAKE_INSPECT_RUNNING": "true",
        "FAKE_RUNNING_FP": "sha256:abc",
        "FAKE_IMAGE_FP": "sha256:abc",
        "FAKE_CONFIG_HASH": "hash-1",
        "FAKE_LABEL": "hash-1",
        "FAKE_EXEC_FAIL": "0",
        "FAKE_RUN_FAIL": "0",
    }


def _up_lines(argv_log: list[str]) -> list[str]:
    return [line for line in argv_log if " compose up " in line or "compose up " in line]


def test_compose_recreate_keeps_unchanged_recorder(tmp_path) -> None:
    result, argv_log = _run_compose_recreate(tmp_path, _base_recreate_env())
    assert result.returncode == 0
    assert result.stdout.strip() == "[SYS] stage=deploy_recreate recorder_action=keep reason=unchanged"
    ups = _up_lines(argv_log)
    assert any("mhs-live" in line and "--force-recreate" in line for line in ups)
    assert not any("market-recorder" in line for line in ups)


def test_compose_recreate_recreates_recorder_before_daemon_on_fingerprint_change(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_IMAGE_FP"] = "sha256:changed"
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert result.stdout.strip() == "[SYS] stage=deploy_recreate recorder_action=recreate reason=fingerprint_changed"
    ups = _up_lines(argv_log)
    recorder_idx = next(i for i, line in enumerate(ups) if "market-recorder" in line)
    daemon_idx = next(i for i, line in enumerate(ups) if "mhs-live" in line)
    assert recorder_idx < daemon_idx


def test_compose_recreate_recreates_recorder_on_compose_config_change(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_LABEL"] = "hash-2"
    result, _ = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert result.stdout.strip() == "[SYS] stage=deploy_recreate recorder_action=recreate reason=compose_config_changed"


def test_compose_recreate_recreates_missing_recorder(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_INSPECT_RUNNING"] = "fail"
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert result.stdout.strip() == "[SYS] stage=deploy_recreate recorder_action=recreate reason=not_running"
    assert any("market-recorder" in line for line in _up_lines(argv_log))


def test_compose_recreate_recreates_when_running_fingerprint_unreadable(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_EXEC_FAIL"] = "1"
    result, _ = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert result.stdout.strip() == "[SYS] stage=deploy_recreate recorder_action=recreate reason=running_fp_unreadable"


def test_compose_recreate_recreates_when_image_fingerprint_unreadable(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_RUN_FAIL"] = "1"
    result, _ = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert result.stdout.strip() == "[SYS] stage=deploy_recreate recorder_action=recreate reason=image_fp_unreadable"


def test_compose_recreate_propagates_compose_failure(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_FAIL_UP"] = "mhs-live"
    result, _ = _run_compose_recreate(tmp_path, env)
    assert result.returncode != 0


def test_dockerfile_writes_recorder_fingerprint_after_dependency_sync() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    fingerprint_idx = dockerfile.index("recorder_fingerprint")
    assert fingerprint_idx > dockerfile.rindex("uv sync")
    assert "/app/.recorder_fingerprint" in dockerfile

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".recorder_fingerprint" in dockerignore



def test_gate_holds_inside_decision_window_until_cycle_complete() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy

    # 결정 윈도우(23:00 공개 15분 전 ~ 익일 02:00 UTC) 안에서는 idle 이어도 사이클 완료 전에는 막는다.
    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    idle = {"stage": "idle", "status": "AWAITING_DATA", "ts": "2026-09-15T01:29:00+00:00"}
    held = decide_deploy(idle, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert (held.action, held.reason) == ("wait", "decision_window:AWAITING_DATA")
    done = {
        "stage": "idle", "status": "COMPLETE",
        "decision_time": "2026-09-14T23:00:00+00:00", "ts": "2026-09-15T01:29:00+00:00",
    }
    released = decide_deploy(done, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert (released.action, released.reason) == ("proceed", "cycle_complete")
