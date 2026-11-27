#!/usr/bin/env bash
# Archive mirror night upload: builds a bounded chunk per pass and copies it to Drive immutably.
# Drive writes run only under the shared gdrive lock and never during a container run; nothing here deletes remote files.
set -uo pipefail

CRYPTO_PILOT_ROOT="${CRYPTO_PILOT_ROOT:-$HOME/crypto-pilot}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/crypto-pilot-archive-mirror-$(date -u +%F).log}"
RCLONE_BIN="${RCLONE_BIN:-$(command -v rclone 2>/dev/null || echo "$HOME/.local/bin/rclone")}"
MIRROR_REMOTE="${MIRROR_REMOTE:-gdrive:quant-lake/live/crypto-pilot/archive_mirror}"
DRIVE_ABOUT_REMOTE="${DRIVE_ABOUT_REMOTE:-gdrive:}"
QUANT_GDRIVE_LOCK="${QUANT_GDRIVE_LOCK:-/run/user/$(id -u)/quant-gdrive.lock}"
LOCK_WAIT_SEC="${LOCK_WAIT_SEC:-3600}"
MIRROR_DEADLINE_UTC="${MIRROR_DEADLINE_UTC:-11:00}"
MIRROR_MAX_CONFIRM_AGE_H="${MIRROR_MAX_CONFIRM_AGE_H:-72}"
# Optional space-separated subset for a supervised run; empty means the mirror's default datasets.
MIRROR_DATASETS="${MIRROR_DATASETS:-}"

ARCHIVE_ROOT="$CRYPTO_PILOT_ROOT/data/archive_mirror"
STATUS_DIR="$ARCHIVE_ROOT/_status"
OUTBOX="$ARCHIVE_ROOT/outbox"
CONTAINER_ROOT=/app/data/archive_mirror

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$(dirname "$QUANT_GDRIVE_LOCK")" || exit 1
cd "$CRYPTO_PILOT_ROOT" || exit 1

SEEN_CAPACITY=0
SEEN_ANOMALY=0
SEEN_STALE=0
SEEN_UPLOAD=0
SEEN_FAILED=0
HARD_RC=0

log_line() { printf '%s\n' "$1" >> "$LOG_FILE"; }
log_rc() { log_line "[SYS] stage=archive_mirror step=$1 rc=$2"; }

# Exit priority: capacity, anomaly, stale status, upload or probe failure, failed sources, other build code.
finish() {
  local code=0
  if [ "$SEEN_CAPACITY" -eq 1 ]; then code=4
  elif [ "$SEEN_ANOMALY" -eq 1 ]; then code=5
  elif [ "$SEEN_STALE" -eq 1 ]; then code=6
  elif [ "$SEEN_UPLOAD" -eq 1 ]; then code=1
  elif [ "$SEEN_FAILED" -eq 1 ]; then code=3
  elif [ "$HARD_RC" -ne 0 ]; then code=$HARD_RC
  fi
  log_line "[SYS] stage=archive_mirror step=done rc=$code"
  exit "$code"
}

deadline_passed() { [ "$((10#$(date -u +%H%M)))" -ge "$((10#${MIRROR_DEADLINE_UTC//:/}))" ]; }

mirror_cmd() {
  docker compose --profile archive run --rm --no-deps archive-mirror \
    /app/.venv/bin/python -m src.cli.main archive mirror "$@"
}

# Runs the given command while holding the shared Drive lock; returns the command's status or 1 when the lock times out.
with_drive_lock() {
  local rc=0
  exec {DRIVE_FD}>"$QUANT_GDRIVE_LOCK" || return 1
  if ! flock -w "$LOCK_WAIT_SEC" "$DRIVE_FD"; then
    exec {DRIVE_FD}>&-
    return 1
  fi
  "$@" || rc=$?
  exec {DRIVE_FD}>&-
  return "$rc"
}

sync_index() {
  local rc=0
  with_drive_lock "$RCLONE_BIN" copy "$MIRROR_REMOTE/index" "$ARCHIVE_ROOT/index" --immutable --fast-list \
    --stats 10m --stats-one-line --stats-log-level NOTICE >> "$LOG_FILE" 2>&1 || rc=$?
  log_rc index_sync "$rc"
  [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ]
}

upload_format_doc() {
  local tmp="$STATUS_DIR/FORMAT-v1.md.tmp" rc=0
  mirror_cmd format-doc > "$tmp" 2>>"$LOG_FILE" || rc=$?
  if [ "$rc" -eq 0 ]; then
    # Content-named: an edited description must never collide with the immutable copy and stop every later run.
    local digest
    digest="$(sha256sum "$tmp" | cut -c1-12)"
    with_drive_lock "$RCLONE_BIN" copyto "$tmp" "$MIRROR_REMOTE/FORMAT-v1-$digest.md" --immutable \
      --stats 10m --stats-one-line --stats-log-level NOTICE >> "$LOG_FILE" 2>&1 || rc=$?
  fi
  rm -f "$tmp"
  log_rc format_doc "$rc"
  [ "$rc" -eq 0 ]
}

# Snapshots the complete pairs once. An object without a sidecar is an unfinished build and must not reach Drive,
# and a build running by hand cannot slip a sidecar in without its object.
list_complete_pairs() {
  local objects_list="$1" index_list="$2" sidecar stem ext
  : > "$objects_list" && : > "$index_list" || return 1
  [ -d "$OUTBOX/index" ] || return 0
  while IFS= read -r sidecar; do
    stem="${sidecar%.json.zst}"
    for ext in parquet zip; do
      [ -f "$OUTBOX/objects/$stem.$ext" ] || continue
      printf '%s\n' "$stem.$ext" >> "$objects_list" && printf '%s\n' "$sidecar" >> "$index_list" || return 1
      break
    done
  done < <(cd "$OUTBOX/index" && find . -type f -name '*.json.zst' | sed 's|^\./||' | LC_ALL=C sort)
}

upload_outbox() {
  "$RCLONE_BIN" copy "$OUTBOX/objects" "$MIRROR_REMOTE/objects" --files-from "$1" --immutable --transfers 4 \
    --stats 10m --stats-one-line --stats-log-level NOTICE || return $?
  "$RCLONE_BIN" copy "$OUTBOX/index" "$MIRROR_REMOTE/index" --files-from "$2" --immutable \
    --stats 10m --stats-one-line --stats-log-level NOTICE
}

# Uploads objects before their sidecars, then confirms against a fresh remote listing; a failure never confirms.
flush() {
  local list="$STATUS_DIR/flush_paths.txt" listing="$STATUS_DIR/remote_listing.json" rc=0
  local objects_list="$STATUS_DIR/flush_objects.txt" index_list="$STATUS_DIR/flush_index.txt"
  rm -f "$list" "$listing" "$listing.tmp"
  if ! list_complete_pairs "$objects_list" "$index_list"; then
    log_rc list_outbox 1
    return 1
  fi
  [ -s "$index_list" ] || return 0
  with_drive_lock upload_outbox "$objects_list" "$index_list" >> "$LOG_FILE" 2>&1 || rc=$?
  log_rc upload_outbox "$rc"
  if [ "$rc" -eq 0 ]; then
    { sed 's|^|objects/|' "$objects_list" && sed 's|^|index/|' "$index_list"; } > "$list" || rc=$?
    if [ "$rc" -eq 0 ] && ! "$RCLONE_BIN" lsjson -R --hash --files-only --fast-list --files-from "$list" "$MIRROR_REMOTE" \
      > "$listing.tmp" 2>>"$LOG_FILE"; then
      rc=1
    fi
    [ "$rc" -eq 0 ] && { mv -f "$listing.tmp" "$listing" || rc=1; }
    log_rc list_remote "$rc"
  fi
  rm -f "$listing.tmp"
  if [ "$rc" -eq 0 ]; then
    mirror_cmd confirm --remote-listing "$CONTAINER_ROOT/_status/remote_listing.json" >> "$LOG_FILE" 2>&1 || rc=$?
    log_rc confirm "$rc"
  fi
  [ "$rc" -eq 0 ]
}

# Fails closed: a missing size is zero only for an absent remote directory (rclone exit 3).
remote_bytes() {
  local out rc=0
  out="$("$RCLONE_BIN" size --json "$MIRROR_REMOTE" 2>>"$LOG_FILE")" || rc=$?
  if [ "$rc" -eq 3 ]; then
    echo 0
    return 0
  fi
  [ "$rc" -eq 0 ] || return 1
  printf '%s' "$out" | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["bytes"]))'
}

drive_free() {
  local out
  out="$("$RCLONE_BIN" about --json "$DRIVE_ABOUT_REMOTE" 2>>"$LOG_FILE")" || return 1
  printf '%s' "$out" | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["free"]))'
}

# Runs build passes until the deadline, an empty pass, or a non-alert failure; alert codes 3 and 5 keep the backfill going.
build_loop() {
  local round=0 remote free out rc built more
  while ! deadline_passed; do
    round=$((round + 1))
    if ! flush; then
      SEEN_UPLOAD=1
      return 0
    fi
    if ! remote=$(remote_bytes) || ! free=$(drive_free); then
      log_line "[SYS] stage=archive_mirror step=capacity_probe round=$round status=failed rc=1"
      SEEN_UPLOAD=1
      return 0
    fi
    rc=0
    local -a dataset_args=()
    [ -z "$MIRROR_DATASETS" ] || read -r -a dataset_args <<< "--datasets $MIRROR_DATASETS"
    out=$(mirror_cmd build --execute --remote-bytes "$remote" --drive-free-bytes "$free" \
      --deadline "$MIRROR_DEADLINE_UTC" "${dataset_args[@]}" 2>>"$LOG_FILE" | tee -a "$LOG_FILE") || rc=$?
    log_line "[SYS] stage=archive_mirror step=build round=$round rc=$rc"
    case "$rc" in
      0) ;;
      3) SEEN_FAILED=1 ;;
      4) SEEN_CAPACITY=1; return 0 ;;
      5) SEEN_ANOMALY=1 ;;
      *) HARD_RC=$rc; return 0 ;;
    esac
    if ! read -r built more < <(printf '%s\n' "$out" | tail -n 1 | python3 -c \
      'import json,sys; d=json.loads(sys.stdin.read()); print(int(d["built_objects"]), str(bool(d["more"])).lower())'); then
      SEEN_UPLOAD=1
      return 0
    fi
    if ! flush; then
      SEEN_UPLOAD=1
      return 0
    fi
    if [ "$more" != true ] || [ "$built" -eq 0 ]; then
      return 0
    fi
  done
}

run_status() {
  local rc=0
  mirror_cmd status --max-confirm-age-hours "$MIRROR_MAX_CONFIRM_AGE_H" >> "$LOG_FILE" 2>&1 || rc=$?
  log_rc status "$rc"
  case "$rc" in
    0) ;;
    6) SEEN_STALE=1 ;;
    *) SEEN_UPLOAD=1 ;;
  esac
}

exec {RUN_FD}>"$STATUS_DIR/run.lock" || exit 1
if ! flock -n "$RUN_FD"; then
  log_line "[SYS] stage=archive_mirror step=run_lock status=busy rc=75"
  exit 75
fi

sync_index || { SEEN_UPLOAD=1; finish; }
upload_format_doc || { SEEN_UPLOAD=1; finish; }
build_loop
run_status
finish
