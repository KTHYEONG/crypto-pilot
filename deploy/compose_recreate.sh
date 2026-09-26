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
#   LEGACY_RECORDER_STOP_GRACE_S=30 Graceful stop of the legacy market-recorder.
set -euo pipefail

IMAGE="${1:?Usage: compose_recreate.sh <image>}"

CAPTURE_HANDOVER_TIMEOUT_S="${CAPTURE_HANDOVER_TIMEOUT_S:-900}"
CAPTURE_HANDOVER_POLL_S="${CAPTURE_HANDOVER_POLL_S:-5}"
CAPTURE_HEARTBEAT_STALE_S="${CAPTURE_HEARTBEAT_STALE_S:-30}"
CAPTURE_STOP_GRACE_S="${CAPTURE_STOP_GRACE_S:-20}"
LEGACY_RECORDER_STOP_GRACE_S="${LEGACY_RECORDER_STOP_GRACE_S:-30}"

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

$C pull

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
  expected="$($C config --hash "$service" 2>/dev/null || true)"
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

blue_obs="$(observe_slot blue)"
green_obs="$(observe_slot green)"

image_fp="$(docker run --rm --entrypoint cat "$IMAGE" /app/.capture_fingerprint 2>/dev/null || true)"
case "$image_fp" in
  sha256:?*) ;;
  *) image_fp="" ;;
esac

legacy_running="0"
if slot_running "market-recorder"; then legacy_running="1"; else legacy_running="0"; fi

decide_args=(decide --slot "$blue_obs" --slot "$green_obs")
if [ -n "$image_fp" ]; then
  decide_args+=(--image-fp "$image_fp")
fi
if [ "$legacy_running" = "1" ]; then
  decide_args+=(--legacy-running)
fi
decision="$(python3 "$HELPER" "${decide_args[@]}")"
action="$(printf '%s' "$decision" | sed -n 's/.*action=\([^ ]*\).*/\1/p')"
new_slot="$(printf '%s' "$decision" | sed -n 's/.*new=\([^ ]*\).*/\1/p')"
old_slot="$(printf '%s' "$decision" | sed -n 's/.*old=\([^ ]*\).*/\1/p')"
legacy_flag="$(printf '%s' "$decision" | sed -n 's/.*legacy=\([^ ]*\).*/\1/p')"
reason="$(printf '%s' "$decision" | sed -n 's/.*reason=\([^ ]*\).*/\1/p')"

echo "[SYS] stage=deploy_recreate capture_action=${action} new=${new_slot} old=${old_slot} legacy=${legacy_flag} reason=${reason}"

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

retire_legacy() {
  docker stop -t "$LEGACY_RECORDER_STOP_GRACE_S" market-recorder >/dev/null 2>&1 || true
  docker rm market-recorder >/dev/null 2>&1 || true
  ! docker inspect market-recorder >/dev/null 2>&1
}

any_slot_ready() {
  local slot container started_at
  for slot in blue green; do
    container="market-capture-${slot}"
    if slot_running "$container"; then
      started_at="$(slot_started_at "$container" || true)"
      if [ -n "$started_at" ] && slot_ready "$slot" "$started_at"; then
        return 0
      fi
    fi
  done
  return 1
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

# 레거시 market-recorder는 결정 결과와 무관하게, READY 캡처 슬롯이 있으면 매 배포마다 퇴역시킨다.
LEGACY_FAILED=0
if docker inspect market-recorder >/dev/null 2>&1; then
  if any_slot_ready; then
    if retire_legacy; then
      echo "[SYS] stage=deploy_recreate legacy_recorder=retired"
    else
      echo "[SYS] stage=deploy_recreate legacy_recorder=retire_failed"
      LEGACY_FAILED=1
    fi
  else
    echo "[SYS] stage=deploy_recreate legacy_recorder=kept reason=no_ready_capture"
  fi
fi

# 레거시 레코더가 살아 있으면 normalizer와 같은 청산 파일에 동시에 쓰게 되므로 normalizer를 올리지 않는다.
if slot_running "market-recorder"; then
  echo "[SYS] stage=deploy_recreate normalizer_action=skipped reason=legacy_active"
  $C stop market-normalizer >/dev/null 2>&1 || true
  $C rm -f market-normalizer >/dev/null 2>&1 || true
  LEGACY_FAILED=1
else
  $C up -d --no-deps --force-recreate market-normalizer
fi
$C up -d --no-deps --force-recreate mhs-live
docker image prune -f

if [ "$HANDOVER_FAILED" = "1" ] || [ "$LEGACY_FAILED" = "1" ]; then
  exit 3
fi
exit 0
