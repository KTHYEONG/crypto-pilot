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


def test_docker_compose_has_blue_green_capture_slots() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # Raw-first 캡처는 blue/green 슬롯으로 24/7 가동되며, 정규화기는 파생 parquet만 쓴다.
    assert "src.capture.main" in compose
    assert '"--slot"' in compose or "--slot" in compose
    assert "container_name: market-capture-blue" in compose
    assert "container_name: market-capture-green" in compose
    assert "container_name: market-normalizer" in compose
    assert "market-recorder" not in compose
    assert "src.market_data.streams.recorder_main" not in compose
    assert "liquidation-collector" not in compose
    assert "unless-stopped" in compose
    assert "./data/futures/liquidations:/app/data/futures/liquidations" in compose
    assert "./data/live_capture:/app/data/live_capture" in compose
    assert "mem_limit: 768m" in compose
    assert "mem_limit: 256m" in compose
    assert "./data/state:/app/data/state" in compose
    _, blue_block = compose.split("  capture-blue:\n", 1)
    blue_block = blue_block.split("\n  capture-green:\n", 1)[0]
    green_block = compose.split("\n  capture-green:\n", 1)[1].split("\n  market-normalizer:\n", 1)[0]
    normalizer_block = compose.split("\n  market-normalizer:\n", 1)[1]
    live_block = compose.split("  capture-blue:\n", 1)[0]
    for block in (blue_block, green_block):
        assert "profiles:" in block
        assert "mem_limit: 256m" in block
        assert "--slot" in block
        assert "crypto-pilot-recorder.env" in block
        assert "required: false" in block
        assert "crypto-pilot.env" not in block.replace("crypto-pilot-recorder.env", "")
        assert "./data/state" not in block
        assert "./logs:/app/logs" in block
    assert "normalizer_main" in normalizer_block
    assert "./deploy/backup/status:/app/backup_status:ro" in normalizer_block
    assert "mem_limit: 768m" in normalizer_block
    assert "env_file" not in normalizer_block
    assert "env_file: /home/ubuntu/quant-secrets/crypto-pilot.env" in live_block
    assert "./data/state:/app/data/state" in live_block


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_11_DOCKERFILE_BUILDS",
    "test_docker_compose_has_blue_green_capture_slots",
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
    live_block = compose.split("  capture-blue:\n", 1)[0]

    # Then
    assert "container_name: mhs-live-daemon" in live_block
    assert "mem_limit: 2g" in live_block
    assert "mem_limit: 1200m" not in compose
    assert "container_name: market-capture-blue" in compose
    assert "container_name: market-capture-green" in compose
    assert compose.count("mem_limit: 256m") >= 2
    assert "container_name: market-normalizer" in compose
    assert "mem_limit: 768m" in compose
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
    if [ "${1:-}" = "--profile" ]; then echo "PROFILE_USED $2" >> "$FAKE_LOG"; shift 2; fi
    sub="$1"; shift || true
    case "$sub" in
      version) exit 0 ;;
      pull) exit 0 ;;
      config) printf '%s %s\\n' "${1:-unknown}" "$FAKE_CONFIG_HASH"; exit 0 ;;
      up)
        if [ -n "${FAKE_FAIL_UP:-}" ]; then
          case "$*" in
            *"$FAKE_FAIL_UP"*) exit 1 ;;
          esac
        fi
        case "$*" in
          *capture-green*)
            if [ "${FAKE_GREEN_CRASH:-0}" = "1" ]; then printf 'false\\n' > "$FAKE_STATE/green_running";
            else printf 'true\\n' > "$FAKE_STATE/green_running"; fi
            exit 0 ;;
          *capture-blue*)
            printf 'true\\n' > "$FAKE_STATE/blue_running"; exit 0 ;;
          *) exit 0 ;;
        esac
        ;;
      stop|rm) exit 0 ;;
      *) exit 0 ;;
    esac
    ;;
  exec)
    container="$1"; shift || true
    case "$container" in
      market-capture-blue)
        if [ "${FAKE_BLUE_RUNNING:-false}" != "true" ]; then exit 1; fi
        if [ "${FAKE_EXEC_FAIL:-0}" = "1" ]; then exit 1; fi
        printf '%s\\n' "$FAKE_BLUE_FP"; exit 0 ;;
      market-capture-green)
        state="false"
        if [ -f "$FAKE_STATE/green_running" ]; then state="$(cat "$FAKE_STATE/green_running")";
        else state="${FAKE_GREEN_RUNNING:-false}"; fi
        if [ "$state" != "true" ]; then exit 1; fi
        printf '%s\\n' "$FAKE_IMAGE_FP"; exit 0 ;;
      *) exit 1 ;;
    esac
    ;;
  run)
    if [ "${FAKE_RUN_FAIL:-0}" = "1" ]; then exit 1; fi
    printf '%s\\n' "$FAKE_IMAGE_FP"
    exit 0
    ;;
  inspect)
    case "$*" in
      *State.Running*market-capture-blue*)
        if [ -f "$FAKE_STATE/blue_running" ]; then cat "$FAKE_STATE/blue_running";
        else printf '%s\\n' "${FAKE_BLUE_RUNNING:-false}"; fi
        exit 0 ;;
      *State.Running*market-capture-green*)
        if [ -f "$FAKE_STATE/green_running" ]; then cat "$FAKE_STATE/green_running";
        else printf '%s\\n' "${FAKE_GREEN_RUNNING:-false}"; fi
        exit 0 ;;
      *State.Running*market-recorder*)
        if [ -f "$FAKE_STATE/legacy_removed" ]; then printf 'false\\n'; else printf '%s\\n' "${FAKE_LEGACY_RUNNING:-false}"; fi
        exit 0 ;;
      market-recorder)
        if [ "${FAKE_LEGACY_RUNNING:-false}" = "true" ] && [ ! -f "$FAKE_STATE/legacy_removed" ]; then exit 0; fi
        exit 1 ;;
      *State.StartedAt*) printf '%s\\n' "$FAKE_STARTED_AT"; exit 0 ;;
      *config-hash*market-capture-blue*) printf '%s\\n' "$FAKE_LABEL_BLUE"; exit 0 ;;
      *config-hash*market-capture-green*) printf '%s\\n' "$FAKE_LABEL_GREEN"; exit 0 ;;
      *) exit 1 ;;
    esac
    ;;
  stop|rm)
    case "$*" in
      *market-recorder*)
        if [ "${FAKE_LEGACY_STUCK:-0}" != "1" ] && [ "$op" = "rm" ]; then touch "$FAKE_STATE/legacy_removed"; fi ;;
    esac
    exit 0 ;;
  image) exit 0 ;;
  *) exit 0 ;;
esac
"""


def _write_ready_heartbeat(tmp_path, slot: str, started_at_iso: str) -> None:
    """Write a fully READY heartbeat for ``slot`` with a fresh ``ts``."""
    import json
    from datetime import UTC, datetime

    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "started_at": started_at_iso,
        "stopped_at": None,
        "ts": now_iso,
        "rest": {"book_ticker": {"first_ok_at": started_at_iso}},
        "ws": {"first_frame_at": started_at_iso},
        "flush_failures": 0,
    }
    raw_dir = tmp_path / "data" / "live_capture" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"capture_{slot}.json").write_text(json.dumps(payload), encoding="utf-8")


def _shift_iso(iso_text: str, seconds: int) -> str:
    from datetime import datetime, timedelta

    parsed = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


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
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "src" / "application" / "ops" / "capture_handover.py", deploy_dir / "capture_handover.py")
    log = tmp_path / "argv.log"
    log.write_text("", encoding="utf-8")
    full_env = dict(os.environ)
    full_env.update(env)
    full_env["FAKE_LOG"] = str(log)
    full_env["FAKE_STATE"] = str(state_dir)
    full_env["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    result = subprocess.run(  # noqa: S603 - fixed argv: the repo script under test + a fake image ref
        [str(shutil.which("bash") or "/bin/bash"), str(ROOT / "deploy" / "compose_recreate.sh"), "fake-image:latest"],
        capture_output=True, text=True, check=False, cwd=tmp_path, env=full_env,
    )
    return result, [line for line in log.read_text(encoding="utf-8").splitlines() if line]


def _base_recreate_env(started_at_iso: str = "2026-09-26T11:50:00+00:00") -> dict[str, str]:
    return {
        "FAKE_BLUE_RUNNING": "false",
        "FAKE_GREEN_RUNNING": "false",
        "FAKE_LEGACY_RUNNING": "false",
        "FAKE_BLUE_FP": "sha256:abc",
        "FAKE_IMAGE_FP": "sha256:abc",
        "FAKE_CONFIG_HASH": "hash-1",
        "FAKE_LABEL_BLUE": "hash-1",
        "FAKE_LABEL_GREEN": "hash-1",
        "FAKE_STARTED_AT": started_at_iso,
        "FAKE_EXEC_FAIL": "0",
        "FAKE_RUN_FAIL": "0",
        "FAKE_GREEN_CRASH": "0",
        "CAPTURE_HANDOVER_TIMEOUT_S": "5",
        "CAPTURE_HANDOVER_POLL_S": "1",
        "CAPTURE_HEARTBEAT_STALE_S": "30",
    }


def _up_lines(argv_log: list[str]) -> list[str]:
    return [line for line in argv_log if " compose up " in line or "compose up " in line]


def test_compose_recreate_keeps_unchanged_capture(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert "capture_action=keep" in result.stdout
    assert "PROFILE_USED capture-blue" in argv_log
    ups = _up_lines(argv_log)
    assert any("mhs-live" in line and "--force-recreate" in line for line in ups)
    assert not any("capture-" in line for line in ups)


def test_compose_handover_starts_idle_slot_before_retiring_old(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_IMAGE_FP"] = "sha256:changed"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    _write_ready_heartbeat(tmp_path, "green", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert "capture_action=handover" in result.stdout
    ups = _up_lines(argv_log)
    capture_idx = next(i for i, line in enumerate(ups) if "capture-green" in line)
    normalizer_idx = next(i for i, line in enumerate(ups) if "market-normalizer" in line)
    daemon_idx = next(i for i, line in enumerate(ups) if "mhs-live" in line)
    assert capture_idx < normalizer_idx < daemon_idx
    joined = "\n".join(argv_log)
    up_green = joined.index("up -d --no-deps --force-recreate capture-green")
    stop_blue = joined.index("capture-blue", up_green)
    assert up_green < stop_blue


def test_compose_handover_timeout_keeps_old_slot_and_fails_deploy(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_IMAGE_FP"] = "sha256:changed"
    env["CAPTURE_HANDOVER_TIMEOUT_S"] = "1"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 3
    joined = "\n".join(argv_log)
    assert "capture-green" in joined
    assert not any("capture-blue" in line and ("stop" in line or " rm" in line) for line in argv_log)
    assert any("capture-green" in line and ("stop" in line or " rm" in line) for line in argv_log)
    ups = _up_lines(argv_log)
    assert any("market-normalizer" in line for line in ups)
    assert any("mhs-live" in line for line in ups)


def test_compose_crashed_new_slot_fails_fast(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_IMAGE_FP"] = "sha256:changed"
    env["FAKE_GREEN_CRASH"] = "1"
    env["CAPTURE_HANDOVER_TIMEOUT_S"] = "5"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, _ = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 3
    assert "capture_handover=failed" in result.stdout


def test_compose_first_deploy_retires_legacy_recorder_only_after_ready(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_LEGACY_RUNNING"] = "true"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert "reason=legacy_migration" in result.stdout
    joined = "\n".join(argv_log)
    assert joined.index("capture-blue") < joined.index("market-recorder")


def test_compose_failed_migration_keeps_legacy_and_skips_normalizer(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_LEGACY_RUNNING"] = "true"
    env["CAPTURE_HANDOVER_TIMEOUT_S"] = "1"
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 3
    joined = "\n".join(argv_log)
    assert "market-recorder" not in [line for line in joined.splitlines() if "docker stop" in line and "market-recorder" in line]
    assert "normalizer_action=skipped reason=legacy_active" in result.stdout
    assert any("mhs-live" in line for line in _up_lines(argv_log))


def test_compose_both_running_host_reconciles(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_GREEN_RUNNING"] = "true"
    env["FAKE_BLUE_FP"] = "sha256:old"
    env["FAKE_IMAGE_FP"] = "sha256:abc"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    _write_ready_heartbeat(tmp_path, "green", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert "capture_action=reconcile" in result.stdout
    assert not any("capture-" in line for line in _up_lines(argv_log))
    stops = [line for line in argv_log if "stop" in line and "capture-" in line]
    assert len(stops) >= 1


def test_compose_script_has_no_remove_orphans() -> None:
    script = (ROOT / "deploy" / "compose_recreate.sh").read_text(encoding="utf-8")
    assert "--remove-orphans" not in script


def test_compose_recreate_propagates_compose_failure(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_FAIL_UP"] = "mhs-live"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, _ = _run_compose_recreate(tmp_path, env)
    assert result.returncode != 0


def test_compose_declares_profile_gated_slots_and_least_privilege_normalizer() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for slot in ("blue", "green"):
        assert '"--slot"' in compose or "--slot" in compose
        assert slot in compose
    assert "profiles:" in compose
    assert "mem_limit: 256m" in compose
    assert "src.capture.main" in compose
    assert "./data/state" not in compose.split("capture-blue:", 1)[1].split("market-normalizer:", 1)[0]
    assert "crypto-pilot.env" not in compose.split("capture-blue:", 1)[1].split("market-normalizer:", 1)[0].replace("crypto-pilot-recorder.env", "")
    assert "normalizer_main" in compose
    assert "./deploy/backup/status:/app/backup_status:ro" in compose
    assert "mem_limit: 768m" in compose
    normalizer_block = compose.split("market-normalizer:", 1)[1]
    assert "env_file" not in normalizer_block.split("mhs-live:", 1)[0] if "mhs-live:" in normalizer_block else "env_file" not in normalizer_block
    assert "market-recorder" not in compose


def test_deploy_workflow_ships_handover_helper_and_status_dir_before_recreate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "capture_handover.py" in workflow
    assert workflow.index("capture_handover.py.new") < workflow.index("compose_recreate.sh '$IMAGE:latest'")
    assert "deploy/backup/status" in workflow
    assert workflow.index("deploy/backup/status") < workflow.index("compose_recreate.sh '$IMAGE:latest'")
    assert "python3 -m src.application.ops.daemon_idle_gate" in workflow


def test_filter_excludes_hot_json_and_partial_before_capture_include() -> None:
    text = (ROOT / "deploy" / "crypto-pilot.rclone-filter").read_text(encoding="utf-8")
    capture_idx = text.index("+ /live_capture/**")
    for rule in ("- *.partial", "- /live_capture/raw/hot/**", "- /live_capture/raw/*.json", "+ /live_capture/raw/archive/**"):
        assert rule in text
        assert text.index(rule) < capture_idx
    rules = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert rules[-1] == "- **"


def test_backup_status_written_atomically_only_on_full_success(tmp_path) -> None:
    import json
    import os
    import shutil
    import stat
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_rclone.chmod(fake_rclone.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    lock = tmp_path / "test.lock"
    lock.write_text("", encoding="utf-8")
    script = ROOT / "deploy" / "backup" / "crypto-pilot-backup.sh"
    full_env = dict(os.environ)
    full_env["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    full_env["CRYPTO_PILOT_ROOT"] = str(tmp_path)
    full_env["RCLONE_BIN"] = str(fake_rclone)
    full_env["QUANT_GDRIVE_LOCK"] = str(lock)
    full_env["LOG_DIR"] = str(tmp_path / "logs")
    result = subprocess.run(  # noqa: S603 - fixed argv: the repo backup script under test
        [str(shutil.which("bash") or "/bin/bash"), str(script)],
        capture_output=True, text=True, check=False, cwd=tmp_path, env=full_env,
    )
    assert result.returncode == 0
    status = tmp_path / "deploy" / "backup" / "status" / "last_success.json"
    assert status.is_file()
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["rc"] == 0
    assert payload["started_at"] <= payload["finished_at"]
    assert list((tmp_path / "deploy" / "backup" / "status").glob("*.partial")) == []


def test_backup_failed_step_never_advances_status(tmp_path) -> None:
    import os
    import shutil
    import stat
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text("#!/usr/bin/env bash\ncase \"$*\" in *copy*) exit 1;; *) exit 0;; esac\n", encoding="utf-8")
    fake_rclone.chmod(fake_rclone.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    status_dir = tmp_path / "deploy" / "backup" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    before = '{"started_at": "2026-09-25T00:00:00Z", "finished_at": "2026-09-25T00:01:00Z", "rc": 0}'
    (status_dir / "last_success.json").write_text(before, encoding="utf-8")
    lock = tmp_path / "test.lock"
    lock.write_text("", encoding="utf-8")
    full_env = dict(os.environ)
    full_env["PATH"] = f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    full_env["CRYPTO_PILOT_ROOT"] = str(tmp_path)
    full_env["RCLONE_BIN"] = str(fake_rclone)
    full_env["QUANT_GDRIVE_LOCK"] = str(lock)
    full_env["LOG_DIR"] = str(tmp_path / "logs")
    result = subprocess.run(  # noqa: S603 - fixed argv: the repo backup script under test
        [str(shutil.which("bash") or "/bin/bash"), str(ROOT / "deploy" / "backup" / "crypto-pilot-backup.sh")],
        capture_output=True, text=True, check=False, cwd=tmp_path, env=full_env,
    )
    assert result.returncode == 1
    assert (status_dir / "last_success.json").read_text(encoding="utf-8") == before


def test_dockerfile_writes_capture_fingerprint_after_dependency_sync() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    fingerprint_idx = dockerfile.index("capture_fingerprint")
    assert fingerprint_idx > dockerfile.rindex("uv sync")
    assert "/app/.capture_fingerprint" in dockerfile
    assert "recorder_fingerprint" not in dockerfile
    assert ".recorder_fingerprint" not in dockerfile

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".capture_fingerprint" in dockerignore
    assert ".recorder_fingerprint" not in dockerignore



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


def test_liveness_units_are_well_formed() -> None:
    from pathlib import Path
    import subprocess

    root = Path(__file__).resolve().parents[2]
    timer = (root / "deploy" / "liveness" / "crypto-pilot-liveness.timer").read_text(encoding="utf-8")
    service = (root / "deploy" / "liveness" / "crypto-pilot-liveness.service").read_text(encoding="utf-8")
    script = (root / "deploy" / "liveness" / "crypto-pilot-liveness.sh").read_text(encoding="utf-8")

    assert "OnCalendar=*:0/5" in timer
    assert "Persistent=true" in timer
    assert "Type=oneshot" in service
    assert "OnFailure=kca-alert@%n.service" in service
    # user unit(%h)은 system unit docker.service에 의존할 수 없다: 의존 시 트랜잭션 단계에서
    # 실행 자체가 거부되어 OnFailure도 발동하지 않는다.
    assert "docker.service" not in service
    assert "--pull never" in script
    assert "data/state" in script
    assert "LIVE_DEADMAN_PING_URL=" not in script
    assert "LIVE_RECORDER_DEADMAN_PING_URL=" not in script
    assert "set -uo pipefail" in script
    assert "stage=liveness" in script
    assert subprocess.run(["/bin/bash", "-n", str(root / "deploy" / "liveness" / "crypto-pilot-liveness.sh")], capture_output=True, timeout=30).returncode == 0  # noqa: S603


def test_deploy_workflow_installs_and_enables_liveness_unit() -> None:
    """The host liveness watcher must be shipped and enabled by CI, or a dead daemon goes unreported."""
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert 'deploy/liveness/crypto-pilot-liveness.sh "$REMOTE_USER@$HOST:~/crypto-pilot/deploy/liveness/crypto-pilot-liveness.sh.new"' in workflow
    assert "deploy/liveness/crypto-pilot-liveness.service" in workflow
    assert "deploy/liveness/crypto-pilot-liveness.timer" in workflow
    assert workflow.index("mv -f ~/crypto-pilot/deploy/liveness/crypto-pilot-liveness.sh.new") < workflow.index("systemctl --user daemon-reload")
    assert workflow.index("systemctl --user daemon-reload") < workflow.index("systemctl --user enable --now crypto-pilot-liveness.timer")


def test_deploy_workflow_never_ships_sealed_artifacts() -> None:
    """Sealed .enc artifacts are untracked; a CI scp of them fails the whole deploy on a clean checkout."""
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert ".enc" not in "".join(line for line in workflow.splitlines() if line.lstrip().startswith("scp "))



def test_compose_up_failure_of_new_slot_continues_and_fails_deploy(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_IMAGE_FP"] = "sha256:changed"
    env["FAKE_FAIL_UP"] = "capture-green"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 3
    assert "capture_handover=failed reason=up_failed" in result.stdout
    ups = _up_lines(argv_log)
    assert any("market-normalizer" in line for line in ups)
    assert any("mhs-live" in line for line in ups)
    assert not any("capture-blue" in line and ("stop" in line or " rm" in line) for line in argv_log)


def test_compose_keep_still_retires_lingering_legacy_recorder(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_LEGACY_RUNNING"] = "true"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 0
    assert "legacy_recorder=retired" in result.stdout
    assert any("rm" in line and "market-recorder" in line for line in argv_log)
    assert any("market-normalizer" in line for line in _up_lines(argv_log))


def test_compose_stuck_legacy_recorder_blocks_normalizer_and_fails_deploy(tmp_path) -> None:
    env = _base_recreate_env()
    env["FAKE_BLUE_RUNNING"] = "true"
    env["FAKE_LEGACY_RUNNING"] = "true"
    env["FAKE_LEGACY_STUCK"] = "1"
    _write_ready_heartbeat(tmp_path, "blue", _shift_iso(env["FAKE_STARTED_AT"], 10))
    result, argv_log = _run_compose_recreate(tmp_path, env)
    assert result.returncode == 3
    assert "legacy_recorder=retire_failed" in result.stdout
    assert "normalizer_action=skipped reason=legacy_active" in result.stdout
    assert not any("market-normalizer" in line for line in _up_lines(argv_log))
    assert any("mhs-live" in line for line in _up_lines(argv_log))


def test_recreate_script_hashes_profile_gated_slot_services_with_their_profile() -> None:
    """Without --profile, compose reports "no such service", so every deploy would look like a config change."""
    root = Path(__file__).resolve().parents[2]
    script = (root / "deploy" / "compose_recreate.sh").read_text(encoding="utf-8")
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert '$C --profile "$service" config --hash "$service"' in script
    assert "print $NF" in script
    for slot in ("blue", "green"):
        assert f'profiles: ["capture-{slot}"]' in compose

