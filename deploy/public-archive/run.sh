#!/usr/bin/env bash
# Public-archive night job: incremental collect, backup-gated prune, then raw backfill rounds.
# Raw rounds write to Drive only through the backup script, which takes the shared gdrive lock itself;
# collection never overlaps the upload because the collector has exited before each backup.
set -uo pipefail

CRYPTO_PILOT_ROOT="${CRYPTO_PILOT_ROOT:-$HOME/crypto-pilot}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/crypto-pilot-public-archive-$(date -u +%F).log}"
RAW_BACKFILL_DEADLINE_UTC="${RAW_BACKFILL_DEADLINE_UTC:-21:30}"
RAW_BACKFILL_MAX_ROUNDS="${RAW_BACKFILL_MAX_ROUNDS:-6}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
CHUNK_GIB="${CHUNK_GIB:-30}"

mkdir -p "$LOG_DIR" || exit 1
cd "$CRYPTO_PILOT_ROOT" || exit 1

log_rc() {
  printf '[SYS] stage=public_archive step=%s rc=%s\n' "$1" "$2" >> "$LOG_FILE"
}

archive_busy() { docker ps -q --filter name=public-archive | grep -q .; }
pending_raw() { grep -l '"raw_only": true' data/public_archive/_jobs/pending/*.json 2>/dev/null | wc -l; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
utc_hhmm() { date -u +%H%M; }
deadline_passed() { [ "$((10#$(utc_hhmm)))" -ge "$((10#${RAW_BACKFILL_DEADLINE_UTC//:/}))" ]; }

# Budget = GiB already spent today (UTC) + one chunk; an unreadable or stale budget file counts as zero spend.
raw_budget_gib() {
  local spent
  spent=$(python3 -c 'import json,datetime as d;p=json.load(open("data/public_archive/_status/raw_backfill_budget.json"));print(p["bytes"]>>30 if p["utc_day"]==d.datetime.now(d.timezone.utc).strftime("%Y-%m-%d") else 0)' 2>/dev/null || echo 0)
  echo $((spent + CHUNK_GIB))
}

raw_cycle() {
  local round=0 stalled=0 before after collect_rc collect_skipped backup_rc prune_rc free
  while [ "$round" -lt "$RAW_BACKFILL_MAX_ROUNDS" ]; do
    [ "$(pending_raw)" -gt 0 ] || return 0
    deadline_passed && return 0
    round=$((round + 1))
    before=$(pending_raw)
    collect_rc=skipped
    collect_skipped=1
    if archive_busy; then
      backup_rc=skipped; prune_rc=skipped
    else
      free=$(free_gb)
      if [ "$free" -ge "$MIN_FREE_GB" ]; then
        collect_skipped=0
        docker compose --profile archive run --rm --no-deps public-archive \
          /app/.venv/bin/python -m src.cli.main archive collect --dataset all --execute --workers 2 \
          --drain-jobs-until "$RAW_BACKFILL_DEADLINE_UTC" --raw-budget-gib "$(raw_budget_gib)" >> "$LOG_FILE" 2>&1
        collect_rc=$?
      fi
      bash deploy/backup/crypto-pilot-backup.sh >> "$LOG_FILE" 2>&1
      backup_rc=$?
      docker compose --profile archive run --rm --no-deps public-archive \
        /app/.venv/bin/python -m src.cli.main archive prune >> "$LOG_FILE" 2>&1
      prune_rc=$?
    fi
    after=$(pending_raw)
    printf '[SYS] stage=public_archive step=3 round=%s collect_rc=%s backup_rc=%s prune_rc=%s pending_raw=%s free_gb=%s\n' \
      "$round" "$collect_rc" "${backup_rc:-skipped}" "${prune_rc:-skipped}" "$after" "$(free_gb)" >> "$LOG_FILE"
    if [ "$collect_skipped" -eq 1 ] && [ "$after" -ge "$before" ]; then
      stalled=$((stalled + 1))
      [ "$stalled" -ge 2 ] && return 0
    else
      stalled=0
    fi
  done
}

RC1=0
RC2=0

docker compose --profile archive run --rm public-archive >> "$LOG_FILE" 2>&1 || RC1=$?
log_rc 1 "$RC1"

docker compose --profile archive run --rm public-archive /app/.venv/bin/python -m src.cli.main archive prune >> "$LOG_FILE" 2>&1 || RC2=$?
log_rc 2 "$RC2"

raw_cycle >> "$LOG_FILE" 2>&1 || true

if [ "$RC1" -ne 0 ]; then
  exit "$RC1"
fi
exit 0
