# or-vps Operations Checklist (crypto-pilot)

AI-executable audit of the production host `or-vps`. It covers market-data capture, local
storage, the daily paper/live trading cycle, the trade records, alerting, and the Google
Drive backup. Work through the sections in order and report with the template in §10.

All times are UTC. The host is OCI Ampere arm64 (2 OCPU / 12 GB), shared with non-crypto
workloads (KCA, mt-etf). Paths below are relative to `~/crypto-pilot` on the host unless
they start with `/` or `~`.

---

## 0. Safety Rules (read before running anything)

- **Read-only by default.** Do not restart, recreate, stop or `docker exec` into
  containers. Do not run `systemctl start|stop|restart`. Do not run `rclone copy|sync|purge|delete`.
  Allowed on Drive: `rclone lsf|lsjson|size|about|check --one-way`.
- **Never execute project code inside prod containers that touches data caches.** A past
  audit called a cache loader inside the daemon container and deleted `BTCUSDT` data.
  To analyse records, copy files to the local machine under
  `scratch/` (or the session scratchpad) and analyse there.
- **Never dump container env or secrets.** Never run `docker inspect <c>` without `--format`
  (`Config.Env` holds API secrets, SMTP passwords and deadman URLs). When an env value is
  needed, filter to a single named non-secret key (`LIVE_MODE`, `LIVE_EXECUTION_POLICY`,
  `LIVE_RECORD_RUN_ID`, `LIVE_PAPER_FILL_MODEL`, `LIVE_NOTIONAL_EQUITY_USDT`). Never print
  `~/quant-secrets/*.env`, `LIVE_*_SECRET`, `LIVE_*_KEY`, `*_PASSWORD`, `*_PING_URL`.
- **Do not delete diagnostic evidence** (logs, quarantine files, stale heartbeats, old run
  dirs) even if it looks like residue. Report it instead.
- Run dirs are written by the container as `root`; the `ubuntu` user can read them but must
  not modify them.
- If a check needs a state change to fix, stop and report a recommended action. Do not apply it.

Convenience variables used below (run on the host):

```bash
cd ~/crypto-pilot
RUN_ID=$(docker inspect mhs-live-daemon --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^LIVE_RECORD_RUN_ID=//p')
RUN=data/state/runs/$RUN_ID
age() { echo $(( $(date +%s) - $(stat -c %Y "$1") ))s "$1"; }
```

---

## 1. System Map

| Component | Runs as | Produces | Health signal |
|---|---|---|---|
| `mhs-live-daemon` | compose service `mhs-live`, `mem_limit 2g`, `restart: unless-stopped` | daily decision cycle, orders (paper/live), run records under `data/state/runs/<run_id>/`, refreshed `data/futures/*`, `data/state/live_orderbook/`, `data/live_capture/exec_depth/` | `data/state/live_daemon_heartbeat.json`, healthchecks.io deadman, Gmail alerts |
| `market-capture-blue` / `-green` | compose profiles `capture-blue` / `capture-green`; exactly one active | raw REST/WS journal `data/live_capture/raw/hot/`, daily `reference/` snapshots | `data/live_capture/raw/capture_<slot>.json`, recorder deadman |
| `market-normalizer` | compose service, `mem_limit 768m` | snapshot parquet (`book_ticker/`, `premium_index/`), `data/futures/liquidations/`, `coverage/`, daily compaction to `raw/archive/`, retention prune | `data/live_capture/recorder_heartbeat.json` (schema v3); daemon `RecorderWatch` thread |
| `crypto-pilot-liveness.timer` | host systemd user timer, every 5 min | runs `live liveness-check` in a one-shot container | `~/logs/crypto-pilot-liveness-<date>.log`, `data/state/live_liveness_state.json` |
| `crypto-pilot-backup.timer` | host systemd user timer, 00:15 and 12:30 | `rclone copy` to `gdrive:quant-lake/live/crypto-pilot` | `deploy/backup/status/last_success.json`, `~/logs/crypto-pilot-backup-<date>.log` |
| GitHub Actions `deploy.yml` | on push to `main` | arm64 image to GHCR, tests, idle-gated deploy over Tailscale | workflow conclusion; `org.opencontainers.image.revision` label |

Configuration sources:

- `docker-compose.yml` (repo, copied to the host by CI on every deploy): non-secret runtime
  env for the daemon (`LIVE_MODE`, `LIVE_EXECUTION_POLICY`, `LIVE_PAPER_FILL_MODEL`,
  `LIVE_NOTIONAL_EQUITY_USDT`, `LIVE_RECORD_RUN_ID`, `LIVE_TAX_COLLECTION_ENABLED`).
  `environment:` entries override `env_file`.
- `~/quant-secrets/crypto-pilot.env` and `crypto-pilot-recorder.env` (host only, never
  deployed by CI): credentials, deadman URLs, alert SMTP.
- `deploy/mhs/` on the host: sealed bootstrap artifacts (`*.enc`) are host-authoritative;
  CI only uploads `venue_rules_*.json`.

## 2. Daily Timeline (UTC)

| Time | Event |
|---|---|
| every 60 s / 300 s | capture: `bookTicker` / `premiumIndex` REST grid; liquidations WS continuous |
| every 30 s | normalizer derive cycle; heartbeat v3 rewrite |
| 00:05 (+600 s retries) | capture: daily `reference/` snapshots (exchange_info, funding_info, asset_index) |
| 00:15, 12:30 | Drive backup |
| day end + 30 min | normalizer compacts the previous day's hot files into `raw/archive/` |
| hourly | normalizer retention prune (blocked unless backup succeeded within 72 h) |
| ~20:15 | daemon funding prefetch for decision day D (best effort) |
| ~23:03 | daemon decision cycle for D (`release_hour_utc=23` + 3 min buffer): refresh -> signal -> execute |
| cycle + up to ~36 min | passive execution window (`passive_timeout_minutes=30` + 2 x 180 s IOC) |
| 20:30 | shared `vps-image-gc.timer` |

A HALTed cycle retries with backoff 300/600/1200/2400 s, at most 5 attempts, then
`day_skipped` (CRITICAL) and the decision is marked processed. Missed decisions are never replayed.

---

## 3. Host, Containers and Deploy

| ID | Check | Command | Pass |
|---|---|---|---|
| H1 | Expected containers present | `docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}'` | `mhs-live-daemon`, `market-normalizer` Up; exactly one of `market-capture-blue`/`-green` Up; **no** `market-recorder` |
| H2 | Restarts / OOM / revision | `for c in mhs-live-daemon market-normalizer market-capture-blue market-capture-green; do docker inspect $c --format "$c {{.RestartCount}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} rev={{index .Config.Labels \"org.opencontainers.image.revision\"}}" 2>/dev/null; done` | restarts 0, oom=false; daemon and normalizer `rev` equal the latest successful `main` deploy SHA |
| H3 | Capture image version | `docker exec market-capture-<slot> cat /app/.capture_fingerprint` vs `docker run --rm --entrypoint cat ghcr.io/kthyeong/crypto-pilot-live:latest /app/.capture_fingerprint` | equal, or a handover is expected at next deploy. The capture slot may predate revision labels, so use the fingerprint (this `exec` is a read of a static file and is allowed) |
| H4 | Memory | `docker stats --no-stream --format '{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}'`; `free -m` | daemon well under 2 GiB (signal step peaks ~1.9 GiB on a 400-day panel); host available RAM > 2 GiB around 23:00 |
| H5 | Disk | `df -h /`; `du -sh data data/* data/live_capture/* data/futures/* logs` | root < 80 %; compare growth with the previous audit |
| H6 | Timers | `systemctl --user list-timers --all --no-pager \| grep -E 'crypto-pilot\|vps-image-gc'` | both crypto-pilot timers scheduled, last trigger within period |
| H7 | Unit results | `for u in crypto-pilot-backup crypto-pilot-liveness; do systemctl --user show $u.service -p ActiveState,Result,ExecMainStatus,ExecMainExitTimestamp; done`; `systemctl --user --failed` | `Result=success`, status 0; failed units belonging to other projects are noted, not actioned |
| H8 | CI/CD | `curl -s https://api.github.com/repos/KTHYEONG/crypto-pilot/actions/runs?per_page=3` | latest run on `main` `completed/success` and its `head_sha` equals H2 `rev`. A failed `test` job skips `deploy`; the host then still runs the previous image |
| H9 | Shared host load | `docker stats --no-stream` (all containers) | note non-crypto containers with high CPU/RAM near 23:00 |

## 4. Configuration and Run Manifest Consistency

This is the most common self-inflicted outage: the daemon refuses to execute when the
current settings disagree with the run's manifest.

| ID | Check | Command | Pass |
|---|---|---|---|
| C1 | Effective non-secret settings | `docker inspect mhs-live-daemon --format '{{range .Config.Env}}{{println .}}{{end}}' \| grep -E '^LIVE_(MODE\|EXECUTION_POLICY\|PAPER_FILL_MODEL\|NOTIONAL_EQUITY_USDT\|RECORD_RUN_ID\|TAX_COLLECTION_ENABLED)='` | matches repo `docker-compose.yml` |
| C2 | Manifest vs settings | `cat $RUN/run_manifest.json` | `execution_policy`, `mode`, `strategy_id`, `name_clip`, `unit_bootstrap_sha256` agree with C1 and the deployed code. Any difference on these keys makes `_assert_run_manifest_compatible` (`src/live/runner.py`) raise `DataIntegrityError` -> every attempt HALTs -> `day_skipped` |
| C3 | Policy/mode change discipline | review recent commits to `docker-compose.yml` | a change of `LIVE_EXECUTION_POLICY` or `LIVE_MODE` **must** ship together with a new `LIVE_RECORD_RUN_ID` (regex `^[a-z0-9][a-z0-9_]{7,63}$`); the new run starts flat with `seed_equity_usdt` |
| C4 | PAPER loop simulator | C1 | `strict_passive*` policies require `LIVE_PAPER_FILL_MODEL` other than `immediate_taker` (settings validator) |
| C5 | Bootstrap artifact present | `ls -la deploy/mhs/` | `frozen_unit_returns_maker.parquet.enc` and `venue_rules_*.json` present |

## 5. Daemon Cycle

| ID | Check | Command | Pass |
|---|---|---|---|
| D1 | Heartbeat | `cat data/state/live_daemon_heartbeat.json; age data/state/live_daemon_heartbeat.json` | age < 900 s; `status` COMPLETE or RUNNING; `consecutive_halts=0`; `attempts=0`; `decision_time` = yesterday 00:00 (or today after the ~23:03 cycle) |
| D2 | Stage deadline | heartbeat `expected_by` | `now < expected_by`. Idle: next wake + 900 s; refresh/signal: +1800 s; execute: +4860 s |
| D3 | Scheduler state | `cat data/state/live_daemon_last_run.json` | `pending_decision_time=null`, `attempts=0`, `last_processed_decision_time` = latest decision |
| D4 | Liveness episodes | `cat data/state/live_liveness_state.json` | `open_keys=[]` |
| D5 | Daemon log errors | `grep -nE 'ERROR\|CRITICAL\|halted\|HALT\|Traceback' logs/live/daemon.log \| tail -50` | none since the last audit; otherwise classify (refresh / signal / execute) |
| D6 | Decision continuity | `ls $RUN/audit/` and `ls logs/live/shadow_cycle/` | one file per calendar day since the run started. A missing day = a skipped cycle (e.g. 2026-09-26: funding prefetch failed, then refresh/signal HALT until retries exhausted) |
| D7 | Status values | heartbeat `status` | `AWAITING_DATA` (refresh failing, data > 30 h stale), `DEGRADED`, `STATE_CORRUPT`, `INTERRUPTED` all need an explanation |
| D8 | Universe inputs | `age data/state/non_crypto_symbols.json; ls -t data/state/venue_listing \| head -3; ls -t data/futures/venue_rules \| head -3` | refreshed by the last cycle; venue rules < 2 days old |
| D9 | Data quarantine | `ls -la data/state/signal_quarantine.json data/live_capture/quarantine 2>/dev/null` | absent or explained (`data_quarantine` alert; repair via `data repair-ohlcv --symbol`) |
| D10 | Market data freshness for the signal | `ls -t data/futures/ohlcv/1h \| head`, `ls -t data/futures/funding \| head` (mtime) | updated within the last cycle; staleness gates: signal 26 h, weights 96 h, market data 30 h, paper funding lag 24 h (HALT) |
| D11 | Run outputs written by the last cycle | `ls -la --time-style=+%FT%T $RUN $RUN/*/ \| head -60` | `frozen_unit_forward.parquet`, `frozen_signal_report.json`, `target_weights.parquet.enc`, `decision_ohlcv_close.parquet.enc`, `position_ledger.json`, monthly shards all touched at the last cycle |

## 6. Market Data Capture and Normalizer

| ID | Check | Command | Pass |
|---|---|---|---|
| M1 | Active capture slot heartbeat | `cat data/live_capture/raw/capture_<slot>.json` | `ts` < 120 s old; `ready=true`; `flush_failures=0`; every `rest.*.consecutive_failures` < 3; `ws.last_frame_at` < 600 s old |
| M2 | Permanent data loss counters | same file | `rest.*.dropped_records=0` and `ws.pending_dropped=0`. Any non-zero value is unrecoverable loss: report the stream and window |
| M3 | Dual-active slot | both `capture_blue.json` and `capture_green.json` | only one fresh (`ts` < 120 s) for no more than 1200 s; a stale slot file with `stopped_at` set is normal residue |
| M4 | Normalizer heartbeat | `python3 -m json.tool data/live_capture/recorder_heartbeat.json` | `schema_version=3`, `ts` < 600 s, `normalizer.lag_s` < 600, `normalizer.consecutive_failures` < 5, `last_error=null` |
| M5 | Sampler completeness | heartbeat `streams.book_ticker` / `streams.premium_index` | `last_persisted_at` < 1800 s; `window_captured_points / window_expected_points` >= 0.9; rejected fraction < 1 % |
| M6 | Liquidations | heartbeat `streams.force_order`; `ls data/futures/liquidations \| tail -30` | frame-to-persist lag < 1200 s; hourly `liquidations_<YYYYMMDD>_<HH>.parquet` continuous (quiet hours may legitimately be empty; cross-check `coverage/`) |
| M7 | Coverage gaps | `tail -n 20 data/live_capture/coverage/liquidations/$(date -u +%Y%m%d).jsonl` | intervals contiguous; gaps explained by deploy handovers or WS reconnects |
| M8 | Reference snapshots | heartbeat `reference`; `ls data/live_capture/reference/*/ \| tail` | today's endpoints `captured=true` after ~01:05; `previous_day_complete=true` |
| M9 | Compaction | `ls data/live_capture/raw/hot/*/`; `ls data/live_capture/raw/archive/*/ \| tail` | `hot/` holds only today and yesterday; every `<day>.jsonl.xz` has a `<day>.manifest.json`; heartbeat `compaction.last_error=null` |
| M10 | Retention gate | heartbeat `retention` | `prune_blocked=false`; if blocked, `blocked_reason` is `status_missing|status_invalid|status_stale` and is tied to §8 backup health; `footprint_bytes` < 8 GiB |
| M11 | Stray partials | `find data -name '*.partial' -mmin +60 -o -name '*.tmp' -mmin +60` | empty |
| M12 | Unbounded dirs | `du -sh data/live_capture/coverage data/live_capture/reference data/live_capture/exec_depth data/state/live_orderbook` | track growth; these are not pruned by the normalizer |
| M13 | Execution-window captures | `ls data/live_capture/exec_depth \| tail -3; ls data/state/live_orderbook \| tail -3` | a folder/file for the last decision day; otherwise check audit `exec_depth_skipped reason=` |
| M14 | Recorder alerts | `grep -n 'stage=recorder_watch' logs/live/daemon.log \| tail` | no `UNHEALTHY`, `CHECK_FAILED` or `ALERT_FAILED` without a later recovery |

## 7. Trade Records Integrity

Copy the run directory locally first, then check offline:

```bash
# local machine
mkdir -p scratch/vps_audit && scp -r or-vps:~/crypto-pilot/data/state/runs/<RUN_ID> scratch/vps_audit/
```

Layout of `data/state/runs/<run_id>/`:

| Path | Writer semantics |
|---|---|
| `run_manifest.json` | written once at run start |
| `position_ledger.json` | full atomic overwrite per commit; keeps the last >= 4 `position_history` snapshots |
| `order_journal.jsonl` | append-only WAL (submits, fills, terminals); the ledger commits only from it |
| `fills/fills_YYYYMM.parquet` | monthly, dedup by `fill_id` |
| `tax_ledger/tax_ledger_YYYYMM.jsonl` | append, idempotent `record_id` |
| `execution_quality/`, `microstructure/` | monthly parquet, append-only; corrupt partitions are quarantined |
| `portfolio_state/active.parquet` | one row per cycle; rotates at 256 KiB, keeps 12 archives |
| `audit/YYYY-MM-DD.jsonl` | mirror of `logs/live/shadow_cycle/<date>.jsonl` (the logs copy is pruned after 90 days; the mirror is not) |
| `target_weights.parquet.enc`, `decision_ohlcv_close.parquet.enc` | AES-256-GCM with `LIVE_ARTIFACT_KEY`; do not attempt to decrypt during an audit |
| `frozen_unit_forward.parquet`, `frozen_signal_report.json` | plaintext signal outputs |

Invariants (tolerances: `rtol=1e-9` on quantities, `atol=1e-6` USDT on cash):

| ID | Invariant |
|---|---|
| R1 | `journal_applied_fill_seq == journal_recorded_fill_seq` after a completed cycle (`recorded < applied` means evidence is pending; `recorded > applied` fails load) |
| R2 | Journal fill sequence numbers are contiguous |
| R3 | Ledger `positions[sym] == Σ fills.quantity_delta` per symbol (run started flat) == Σ signed journal execution fills |
| R4 | `cash_usdt == seed − Σ(qty·price) − Σ |qty·price|·fee_bps/1e4 + Σ funding`, equivalently `seed + Σ FUNDING_FEE.realized_pnl − Σ BUY(quote_qty + fee) + Σ SELL(quote_qty − fee)` from the tax ledger |
| R5 | `len(fills) == count(tax TRADE rows) == count(journal execution fills)`; `fill_id` and `record_id` unique |
| R6 | Per (decision_time, symbol): `execution_quality.filled_qty == Σ |fills.quantity_delta|`; `maker_fill_fraction == maker_qty / filled_qty` |
| R7 | `portfolio_state`: `decision_time` unique and ascending; `equity_high_water_mark_usdt` non-decreasing; `equity_usdt == cash_usdt + Σ qty·mark` |
| R8 | `funding_watermarks[sym]` within the last 8 h settlement for every held symbol; `funding_accrued_through` ≈ last cycle |
| R9 | `portfolio_state` rows == number of `audit/` days == executed decisions (gaps must match D6) |
| R10 | Order event log `logs/live/orders/<date>.jsonl` exists for each executed day |

Known by-design quirks (do not report as defects):

- PAPER `equity_source=virtual_mtm`: `unrealized_pnl_usdt` is always 0 and
  `wallet_balance_usdt` is the static venue wallet, not the paper book.
- `FUNDING_FEE` rows: `fee=0`; the cash amount is in `realized_pnl`; `quote_qty` is
  position notional, so never sum it as cash.
- `LIVE_TAX_COLLECTION_ENABLED=false` only disables venue income collection; PAPER
  TRADE/FUNDING rows are still written.

On mismatch the daemon behaves as follows: a cash reconcile miss raises a
`ledger_reconcile_mismatch` alert but does not halt; a corrupt ledger, journal gap or
regression, or manifest mismatch HALTs the cycle.

Execution quality review (optional, same local copy): maker fraction, slippage bps vs
`mark_price_at_decision`, latency per symbol; for `anchored_repeg` count `order_cancelled
reason=repeg` audit events per intent.

## 8. Alerting and Deadman

| ID | Check | Command | Pass |
|---|---|---|---|
| A1 | Outbox delivery | `python3 -c "import json;d=json.load(open('data/state/alert_outbox.json'));[print(r.get('event'),r.get('severity'),r.get('attempts'),r.get('pending_channels'),r.get('expired'),r.get('created_at')) for r in d['records'][-15:]]"` | every record has empty `pending_channels` and `expired=false`; no record with growing `attempts` (`expired=true` = delivery abandoned after 3 days) |
| A2 | CRITICAL events since last audit | same | each of `halt_streak`, `day_skipped`, `data_refresh_failed`, `recorder_unhealthy`, `ledger_reconcile_mismatch`, `order_journal_regressed`, `tax_ledger_corrupt`, `daemon_unresponsive`, `daemon_stage_overrun`, `daemon_container_down`, `cycle_degraded` has a matching recovery or explanation |
| A3 | Liveness log | `grep -hv 'status=OK' ~/logs/crypto-pilot-liveness-*.log \| tail` | no `CHECK_FAILED` / `CONTAINER_MISSING` |
| A4 | Deadman configuration | confirm (without printing values) that the keys exist: `grep -c '^LIVE_DEADMAN_PING_URL=' ~/quant-secrets/crypto-pilot.env; grep -c '^LIVE_RECORDER_DEADMAN_PING_URL=' ~/quant-secrets/crypto-pilot-recorder.env` | both 1. The recorder env file is `required: false` in compose; if missing the capture deadman is silently disabled |
| A5 | Daily digest | outbox | a `cycle_complete` NOTICE for each executed day |

## 9. Google Drive Backup

Remote root: `gdrive:quant-lake/live/crypto-pilot` (`data/`, `logs/live/orders/`, `_versions/<date>/`).
Copy-only (no deletes propagate), filtered by `deploy/crypto-pilot.rclone-filter`, overwritten
files moved to `_versions/<date>` and kept 30 days, serialized by the shared
`/run/user/<uid>/quant-gdrive.lock`.

| ID | Check | Command | Pass |
|---|---|---|---|
| B1 | Last success | `cat deploy/backup/status/last_success.json` | `finished_at` within 13 h (two runs/day); must be < 72 h or the normalizer prune blocks (M10) |
| B2 | Step results | `tail -n 8 ~/logs/crypto-pilot-backup-$(date -u +%F).log`; `grep -h 'status=failed' ~/logs/crypto-pilot-backup-*.log \| tail` | `data`, `orders`, `prune`, `status` all `ok` |
| B3 | Transfer warnings | `journalctl --user -u crypto-pilot-backup.service -p warning --since -7days --no-pager \| tail -40` | `md5 hashes differ` on the current hour's `liquidations_*.parquet` or today's `coverage/*.jsonl` is a known race (file still being appended) that the retry resolves; anything else is a finding |
| B4 | Scope coverage | `rclone lsf gdrive:quant-lake/live/crypto-pilot/data --dirs-only` | `state/`, `archive/`, `futures/liquidations/`, `futures/venue_rules/`, `live_capture/` present |
| B5 | Run records match | `rclone check data/state gdrive:quant-lake/live/crypto-pilot/data/state --one-way 2>&1 \| tail` | only continuously rewritten files differ (`live_daemon_heartbeat.json`, `live_liveness_state.json`); `$RUN/position_ledger.json` and current-month `fills_*.parquet` match after the 00:15 run |
| B6 | Live-only captures | `rclone check data/live_capture/raw/archive gdrive:quant-lake/live/crypto-pilot/data/live_capture/raw/archive --one-way` | no missing archives older than one backup cycle |
| B7 | Version retention | `rclone lsf gdrive:quant-lake/live/crypto-pilot/_versions --dirs-only` | dated folders only, none older than 30 days |
| B8 | Not backed up (by design) | n/a | re-downloadable `futures/ohlcv`, `funding` (and retired legacy feed dirs, if any); `raw/hot/`; capture heartbeats; `logs/` other than `live/orders`. Note: `logs/live/daemon.log` and `logs/live/shadow_cycle/` are not backed up (the run-dir `audit/` mirror is) |

## 10. Known Risks and Open Items

Re-verify each audit and update this list in place when resolved.

- Manifest trap (C2/C3): changing `LIVE_EXECUTION_POLICY` without a new `LIVE_RECORD_RUN_ID`
  HALTs every cycle.
- `logs/live/daemon.log` is a single unrotated file.
- `coverage/`, `reference/`, `exec_depth/`, `state/live_orderbook/` grow without local pruning.
- Backup copies files still being appended (B3), producing first-attempt md5 mismatches.
- Orphan run dir `data/state/runs/frozen_top20_v2_bayes_maker_20260921` and archived
  `data/archive/legacy_horizon_v2_20260902_20260920` (intentional, backed up) remain on disk.
- Capture slot images built before revision labels can only be versioned by fingerprint (H3).
- Missed decision days are not replayed; a failed funding prefetch/refresh near 23:00 costs a day (D6).
- Shared host: non-crypto workloads compete for RAM during the daemon signal step (H4, H9).
- Binance `fundingRate` requests with `limit>=300` can be blocked by AWS WAF (HTTP 403) per IP; repeated `403` in refresh logs is a WAF block, not a rate limit or ban.

## 11. Report Template

```
### or-vps audit <YYYY-MM-DD HH:MM UTC>
Verdict: OK | WARN | FAIL
Deploy: rev=<sha> (CI <status>), run_id=<id>, policy=<policy>, mode=<mode>
Daemon: heartbeat <status>/<stage> age=<s>, last decision=<date>, halts=<n>, missed days=<list>
Capture: slot=<blue|green> ready=<bool> dropped=<n>; normalizer lag=<s>; prune_blocked=<bool>
Records: R1-R10 <pass/fail list>; equity=<usdt> (seed <usdt>), fills=<n>, maker%=<x>
Alerts: outbox pending=<n>; CRITICAL since last audit=<list>
Backup: last_success=<ts> (<age>), failures 7d=<n>, state check=<diffs>
Disk/RAM: root <pct>, data <size>, host avail <MB>
Findings (ordered by severity): <ID> <what> <evidence> <recommended action>
```
