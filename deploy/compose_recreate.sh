#!/usr/bin/env bash
# Usage: compose_recreate.sh <image>
# Run from ~/crypto-pilot on the host after the daemon idle gate has passed.
#
# Blue/green capture handover: the new slot must be READY before the old slot
# retires, so a capture code change never leaves a coverage gap. The normalizer
# is recreated on every deploy; the daemon keeps its 120s grace semantics.
#
# Config (env overrides):
#   CAPTURE_HANDOVER_TIMEOUT_S=900  Max wait for the new slot to become READY.
#   CAPTURE_HANDOVER_POLL_S=5       Heartbeat poll period.
#   CAPTURE_HEARTBEAT_STALE_S=30    A heartbeat older than this is not READY.
#   CAPTURE_STOP_GRACE_S=20         Graceful stop of the old capture slot.
#   OWNERSHIP_WORKER_WAIT_S=300     Max wait for root-run one-shot workers before the ownership pass.
#   CRYPTO_OWNERSHIP_FIXUP          Ownership helper (default: crypto-ownership-fixup.sh beside this script).
set -euo pipefail

IMAGE="${1:?Usage: compose_recreate.sh <image>}"

CAPTURE_HANDOVER_TIMEOUT_S="${CAPTURE_HANDOVER_TIMEOUT_S:-900}"
CAPTURE_HANDOVER_POLL_S="${CAPTURE_HANDOVER_POLL_S:-5}"
CAPTURE_HEARTBEAT_STALE_S="${CAPTURE_HEARTBEAT_STALE_S:-30}"
CAPTURE_STOP_GRACE_S="${CAPTURE_STOP_GRACE_S:-20}"
OWNERSHIP_WORKER_WAIT_S="${OWNERSHIP_WORKER_WAIT_S:-300}"
OWNERSHIP_POLL_S=5

if docker compose version >/dev/null 2>&1; then
  C="docker compose"
else
  C="docker-compose"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/capture_handover.py" ]; then
  HELPER="$SCRIPT_DIR/capture_handover.py"
else
  HELPER="deploy/capture_handover.py"
fi

slot_running() {
  local container="$1"
  local out=""
  out="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)"
  [ "$out" = "true" ]
}

slot_started_at() {
  docker inspect -f '{{.State.StartedAt}}' "$1" 2>/dev/null || true
}

slot_fingerprint() {
  docker exec "$1" cat /app/.capture_fingerprint 2>/dev/null || true
}

slot_config_match() {
  local service="$1"
  local container="$2"
  local expected=""
  local actual=""
  # 슬롯 서비스는 프로파일이 걸려 있어 --profile 없이는 "no such service"로 빈 값이 나오고,
  # 그러면 매 배포마다 설정 변경으로 오판해 불필요한 인계가 일어난다.
  # 출력 형식은 "<서비스> <해시>"이므로 컨테이너 라벨(해시만)과 비교하려면 마지막 필드만 취한다.
  expected="$($C --profile "$service" config --hash "$service" 2>/dev/null | awk 'NR==1{print $NF}' || true)"
  actual="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.config-hash"}}' "$container" 2>/dev/null || true)"
  [ -n "$expected" ] && [ "$expected" = "$actual" ]
}

slot_ready() {
  local slot="$1"
  local started_at="$2"
  local hb="data/live_capture/raw/capture_${slot}.json"
  python3 "$HELPER" ready --heartbeat "$hb" --container-started-at "$started_at" --stale-s "$CAPTURE_HEARTBEAT_STALE_S" >/dev/null 2>&1
}

observe_slot() {
  local slot="$1"
  local container="market-capture-${slot}"
  local service="capture-${slot}"
  local running="0" fp="-" cfg="0" ready="0" started_at=""
  if slot_running "$container"; then
    running="1"
    fp="$(slot_fingerprint "$container" || true)"
    case "$fp" in
      sha256:?*) ;;
      *) fp="-" ;;
    esac
    if slot_config_match "$service" "$container"; then cfg="1"; else cfg="0"; fi
    started_at="$(slot_started_at "$container" || true)"
    if [ -n "$started_at" ] && slot_ready "$slot" "$started_at"; then ready="1"; else ready="0"; fi
  fi
  printf '%s:%s:%s:%s:%s' "$slot" "$running" "$fp" "$cfg" "$ready"
}

FIXUP="${CRYPTO_OWNERSHIP_FIXUP:-$SCRIPT_DIR/crypto-ownership-fixup.sh}"
MIGRATING=0
fixup_check_rc=0
"$FIXUP" check || fixup_check_rc=$?
case "$fixup_check_rc" in
  0) ;;
  10) MIGRATING=1 ;;
  *) echo "[SYS] stage=deploy_recreate ownership=check_failed rc=${fixup_check_rc}"; exit 4 ;;
esac

# Pass1: every writer is still a root container, so re-owning cannot break capture.
if [ "$MIGRATING" = "1" ]; then
  "$FIXUP" fix || { echo "[SYS] stage=deploy_recreate ownership=failed reason=pass1"; exit 4; }
fi

blue_obs="$(observe_slot blue)"
green_obs="$(observe_slot green)"

image_fp="$(docker run --rm --entrypoint cat "$IMAGE" /app/.capture_fingerprint 2>/dev/null || true)"
case "$image_fp" in
  sha256:?*) ;;
  *) image_fp="" ;;
esac

decide_args=(decide --slot "$blue_obs" --slot "$green_obs")
if [ -n "$image_fp" ]; then
  decide_args+=(--image-fp "$image_fp")
fi
decision="$(python3 "$HELPER" "${decide_args[@]}")"
action="$(printf '%s' "$decision" | sed -n 's/.*action=\([^ ]*\).*/\1/p')"
new_slot="$(printf '%s' "$decision" | sed -n 's/.*new=\([^ ]*\).*/\1/p')"
old_slot="$(printf '%s' "$decision" | sed -n 's/.*old=\([^ ]*\).*/\1/p')"
reason="$(printf '%s' "$decision" | sed -n 's/.*reason=\([^ ]*\).*/\1/p')"

echo "[SYS] stage=deploy_recreate capture_action=${action} new=${new_slot} old=${old_slot} reason=${reason}"

HANDOVER_FAILED=0

wait_ready() {
  local slot="$1"
  local container="market-capture-${slot}"
  local elapsed=0
  while [ "$elapsed" -lt "$CAPTURE_HANDOVER_TIMEOUT_S" ]; do
    if ! slot_running "$container"; then
      return 1
    fi
    local started_at=""
    started_at="$(slot_started_at "$container" || true)"
    if [ -n "$started_at" ] && slot_ready "$slot" "$started_at"; then
      return 0
    fi
    sleep "$CAPTURE_HANDOVER_POLL_S"
    elapsed=$((elapsed + CAPTURE_HANDOVER_POLL_S))
  done
  return 1
}

retire_container() {
  local service="$1"
  local container="$2"
  local grace="$3"
  $C stop -t "$grace" "$service" >/dev/null 2>&1 || docker stop -t "$grace" "$container" >/dev/null 2>&1 || true
  $C rm -f "$service" >/dev/null 2>&1 || docker rm "$container" >/dev/null 2>&1 || true
}

ownership_failed() {
  echo "[SYS] stage=deploy_recreate ownership=failed reason=$1"
  docker start market-normalizer mhs-live-daemon >/dev/null 2>&1 || true
  exit 4
}

root_workers() {
  local name image user
  docker ps --format '{{.Names}}' 2>/dev/null | while IFS= read -r name; do
    case "$name" in
      market-normalizer|mhs-live-daemon|"market-capture-${new_slot}") continue ;;
    esac
    image="$(docker inspect -f '{{.Config.Image}}' "$name" 2>/dev/null || true)"
    [ "$image" = "$IMAGE" ] || continue
    user="$(docker inspect -f '{{.Config.User}}' "$name" 2>/dev/null || true)"
    case "$user" in
      ""|0|0:*|root|root:*) printf '%s\n' "$name" ;;
    esac
  done
}

wait_root_workers() {
  local elapsed=0
  while [ -n "$(root_workers)" ]; do
    if [ "$elapsed" -ge "$OWNERSHIP_WORKER_WAIT_S" ]; then return 1; fi
    sleep "$OWNERSHIP_POLL_S"
    elapsed=$((elapsed + OWNERSHIP_POLL_S))
  done
}

# Pass2: the normalizer and daemon are the only root writers left; stop them, re-own, recreate as 1001.
migrate_pass2() {
  trap 'systemctl --user start crypto-pilot-liveness.timer >/dev/null 2>&1 || true' EXIT
  systemctl --user stop crypto-pilot-liveness.timer >/dev/null 2>&1 || true
  wait_root_workers || ownership_failed workers_running
  $C stop -t 30 market-normalizer || ownership_failed normalizer_stop
  $C stop -t 120 mhs-live || ownership_failed daemon_stop
  "$FIXUP" fix || ownership_failed reown
  $C up -d --no-deps --force-recreate market-normalizer || ownership_failed normalizer_up
  $C up -d --no-deps --force-recreate mhs-live || ownership_failed daemon_up
  echo "[SYS] stage=deploy_recreate ownership=ok"
}

case "$action" in
  keep)
    ;;
  start|handover)
    # up 실패가 set -e로 스크립트를 끊으면 normalizer/daemon 재생성까지 건너뛰므로 실패를 기록하고 계속한다.
    if ! $C up -d --no-deps --force-recreate "capture-${new_slot}"; then
      $C rm -f -s "capture-${new_slot}" >/dev/null 2>&1 || docker rm -f "market-capture-${new_slot}" >/dev/null 2>&1 || true
      echo "[SYS] stage=deploy_recreate capture_handover=failed reason=up_failed slot=${new_slot}"
      HANDOVER_FAILED=1
    elif wait_ready "$new_slot"; then
      echo "[SYS] stage=deploy_recreate capture_handover=ok slot=${new_slot}"
      if [ "$action" = "handover" ]; then
        retire_container "capture-${old_slot}" "market-capture-${old_slot}" "$CAPTURE_STOP_GRACE_S"
      fi
    else
      if slot_running "market-capture-${new_slot}"; then
        fail_reason="timeout"
      else
        fail_reason="exited"
      fi
      $C stop -t "$CAPTURE_STOP_GRACE_S" "capture-${new_slot}" >/dev/null 2>&1 || true
      $C rm -f "capture-${new_slot}" >/dev/null 2>&1 || true
      echo "[SYS] stage=deploy_recreate capture_handover=failed reason=${fail_reason}"
      HANDOVER_FAILED=1
    fi
    ;;
  reconcile)
    retire_container "capture-${old_slot}" "market-capture-${old_slot}" "$CAPTURE_STOP_GRACE_S"
    ;;
esac

if [ "$MIGRATING" = "1" ]; then
  if [ "$HANDOVER_FAILED" = "1" ]; then
    echo "[SYS] stage=deploy_recreate ownership=deferred reason=handover_failed"
    exit 3
  fi
  migrate_pass2
  exit 0
fi

$C up -d --no-deps --force-recreate market-normalizer
$C up -d --no-deps --force-recreate mhs-live

if [ "$HANDOVER_FAILED" = "1" ]; then
  exit 3
fi
exit 0
