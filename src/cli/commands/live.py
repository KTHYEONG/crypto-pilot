"""``live`` 커맨드 그룹: 섬도우 사이클 1회 실행 또는 24/7 무인 데몬 구동."""

from __future__ import annotations

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.logging import LOG_DIR
from src.common.paths import DATA_DIR
from src.live.deployed_weights import default_weights_path

logger = logging.getLogger("LiveCli")

_DEFAULT_DAEMON_STATE_PATH = str(DATA_DIR / "state" / "live_daemon_last_run.json")

_LIVE_LOG_DIR: Path = LOG_DIR / "live"
LIVE_LOG_MAX_BYTES: int = 10 * 1024 * 1024
LIVE_LOG_BACKUP_COUNT: int = 5


def _attach_process_log(filename: str) -> Path:
    log_dir = _LIVE_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / filename
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(path):
            return path
    handler = RotatingFileHandler(path, maxBytes=LIVE_LOG_MAX_BYTES, backupCount=LIVE_LOG_BACKUP_COUNT, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(handler)
    return path


def _parse_decision_time(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        raise argparse.ArgumentTypeError("--decision-time must be tz-aware (UTC)")
    return ts.tz_convert("UTC")


def _settings_with_mode(args: argparse.Namespace) -> Any:
    """--mode 플래그로 LIVE_MODE 를 덮어쓴다(비밀값은 여전히 env 전용)."""
    from src.live.settings import ExecutionMode, LiveSettings

    m = getattr(args, "mode", None)
    return LiveSettings(mode=ExecutionMode(m)) if m else LiveSettings()


def _resolve_weights_path(explicit: str | None, settings: Any) -> Path:
    if explicit:
        return Path(explicit)
    weights_path = getattr(settings, "weights_path", None)
    if weights_path:
        return Path(weights_path)
    return default_weights_path()


def _run_shadow_cycle(args: argparse.Namespace) -> None:
    from src.live.runner import run_shadow_cycle

    settings = _settings_with_mode(args)
    report = run_shadow_cycle(
        settings,
        args.decision_time,
        _resolve_weights_path(args.artifact, settings),
    )
    logger.info(
        "[SYS] live shadow-cycle status=%s reason=%s intents=%d",
        report.status,
        report.reason,
        report.intent_count,
    )


def _run_daemon(args: argparse.Namespace) -> None:
    import os

    from src.live.alerting import dispatch_alert, drain_alerts
    from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers
    from src.live.recorder_watch import build_recorder_watchdog

    _attach_process_log("daemon.log")
    settings = _settings_with_mode(args)
    artifact = _resolve_weights_path(getattr(args, "artifact", None), settings)
    shutdown = ShutdownFlag()
    install_shutdown_handlers(shutdown)
    watchdog = build_recorder_watchdog(
        settings,
        alert=lambda event, detail: dispatch_alert(settings, event=event, detail=detail, decision_time=None, dedupe_key=f"{event}:{pd.Timestamp.now(tz='UTC').isoformat()}", now=pd.Timestamp.now(tz="UTC")),
    )
    if watchdog is not None:
        watchdog.start()
    try:
        try:
            from src.live.scheduler import run_daemon

            run_daemon(settings, artifact, Path(args.state_path), shutdown=shutdown)
        except Exception as exc:  # 프로세스 경계라 광역 except 허용
            logger.exception("[SYS] daemon crashed error=%s", type(exc).__name__)
            tick = pd.Timestamp.now(tz="UTC")
            dispatch_alert(settings, event="daemon_crashed", detail=f"error={type(exc).__name__}: {str(exc)[:300]}", decision_time=None, dedupe_key=f"daemon_crashed:{os.getpid()}:{tick.isoformat()}", now=tick)
            drain_alerts(settings, now=tick, blocking=True)
            raise
    finally:
        if watchdog is not None:
            watchdog.stop()


def _run_liveness_check(args: argparse.Namespace) -> None:
    import sys

    from src.live.liveness import ContainerObservation, _default_state_path, run_liveness_check

    settings = _settings_with_mode(args)
    started_at = pd.Timestamp(args.container_started_at) if args.container_started_at else None
    container = ContainerObservation(
        running=bool(args.container_running),
        restarting=bool(args.container_restarting),
        oom_killed=bool(args.container_oom),
        restart_count=int(args.restart_count),
        started_at=started_at,
    )
    state_path = Path(args.state_path) if args.state_path else _default_state_path()
    code = run_liveness_check(settings, container, now=pd.Timestamp.now(tz="UTC"), state_path=state_path)
    sys.exit(code)


def _run_strategy_step(args: argparse.Namespace) -> None:
    from src.live.scheduler import _default_strategy_step

    _attach_process_log("strategy_step.log")
    settings = _settings_with_mode(args)
    target = pd.Timestamp(args.date).tz_convert("UTC").normalize()
    artifact = _resolve_weights_path(getattr(args, "artifact", None), settings)
    try:
        report = _default_strategy_step(target, settings, artifact)
    except Exception as exc:
        logger.error("[EVAL] strategy_step status=FAILED decision_time=%s reason=%s", target.isoformat(), exc)
        raise SystemExit(1) from exc
    logger.info(
        "[EVAL] strategy_step decision_day=%s exposure=%.4f equity_usdt=%.2f unit_observations=%d venue=%s written=%s",
        report.decision_day.isoformat(), report.exposure, report.equity_usdt,
        report.unit_observations, report.venue_snapshot, report.written,
    )


def _run_status(args: argparse.Namespace) -> None:
    import json

    settings = _settings_with_mode(args)
    from src.live.scheduler import _resolve_heartbeat_path

    hb_path = _resolve_heartbeat_path(settings)
    if not hb_path.exists():
        logger.error("[SYS] status=NO_HEARTBEAT path=%s", hb_path)
        raise SystemExit(1)
    hb = json.loads(hb_path.read_text())
    status = str(hb.get("status", "UNKNOWN"))
    hb_ts = pd.Timestamp(hb["ts"])
    age_min = (pd.Timestamp.now(tz="UTC") - hb_ts).total_seconds() / 60
    logger.info(
        "[SYS] status=%s stage=%s decision_time=%s consecutive_halts=%s attempts=%s heartbeat_age_min=%.1f detail=%s",
        status,
        hb.get("stage"),
        hb.get("decision_time"),
        hb.get("consecutive_halts"),
        hb.get("attempts"),
        age_min,
        hb.get("detail", ""),
    )
    unhealthy = status in {"HALT", "AWAITING", "AWAITING_DATA", "STATE_CORRUPT"} or age_min > settings.max_signal_staleness_hours * 60
    raise SystemExit(1 if unhealthy else 0)


def _run_execution_quality_summary(args: argparse.Namespace) -> None:  # noqa: ARG001
    from src.live.execution_quality import summarize_execution_quality

    summary = summarize_execution_quality()
    logger.info("[EVAL] execution_quality %s", summary)


def _run_portfolio_state_summary(args: argparse.Namespace) -> None:  # noqa: ARG001
    from src.live.portfolio_state import summarize_portfolio_state

    summary = summarize_portfolio_state()
    logger.info("[EVAL] portfolio_state %s", summary)


def _run_ledger_resync(args: argparse.Namespace) -> None:
    import src.live.ledger_resync as _resync
    from src.common.errors import DataIntegrityError
    from src.live.errors import LiveTradingError

    settings = _settings_with_mode(args)
    try:
        plan = _resync.run_ledger_resync(
            settings, apply=bool(args.apply), now=pd.Timestamp.now(tz="UTC")
        )
    except (LiveTradingError, DataIntegrityError) as exc:
        logger.error("[PORTFOLIO] ledger_resync status=FAILED reason=%s", exc)
        raise SystemExit(1) from exc
    logger.info(
        "[PORTFOLIO] ledger_resync status=%s adjustments=%d",
        "APPLIED" if args.apply else "DRY_RUN",
        len(plan.adjustments),
    )
    for breach in plan.adjustments:
        logger.info(
            "[PORTFOLIO] ledger_resync adjustment symbol=%s venue=%s ledger=%s gap=%s",
            breach.symbol,
            breach.venue_qty,
            breach.ledger_qty,
            breach.gap,
        )


def _run_paper_funding_backfill(args: argparse.Namespace) -> None:
    import src.live.funding_backfill as _backfill
    from src.common.errors import DataIntegrityError

    settings = _settings_with_mode(args)
    try:
        plan = _backfill.run_paper_funding_backfill(
            settings, apply=bool(args.apply), now=pd.Timestamp.now(tz="UTC"), accrual_start=args.accrual_start
        )
    except DataIntegrityError as exc:
        logger.error("[PORTFOLIO] paper_funding_backfill status=FAILED reason=%s", exc)
        raise SystemExit(1) from exc
    logger.info(
        "[PORTFOLIO] paper_funding_backfill status=%s cash_delta=%s end=%s",
        "APPLIED" if args.apply else "DRY_RUN",
        plan.cash_delta,
        plan.end.isoformat(),
    )


def _run_preflight(args: argparse.Namespace) -> None:
    from src.live.preflight import run_preflight

    settings = _settings_with_mode(args)
    report = run_preflight(settings, _resolve_weights_path(args.artifact, settings))
    for check in report.checks:
        logger.info("[SYS] %s passed=%s detail=%s", check.name, check.passed, check.detail)
    if not report.passed:
        raise SystemExit(1)


def _run_tax_collect(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.live.account import assert_venue_configuration, fetch_account_snapshot
    from src.live.audit import AuditLog, default_audit_log_path
    from src.live.errors import LiveTradingError
    from src.live.rest import BinanceFuturesRestClient
    from src.live.tax_ledger import VenuePositionSnapshot, collect_and_persist_live_tax, resolve_tax_ledger_dir

    settings = _settings_with_mode(args)
    if settings.mode.suppresses_mutations:
        logger.error("[DATA] tax_collect status=REFUSED mode=%s reason=venue collection requires a live mode", settings.mode.value)
        raise SystemExit(1)
    now = pd.Timestamp.now(tz="UTC")
    audit = AuditLog(default_audit_log_path("tax_collect", for_date=now))
    client = BinanceFuturesRestClient(
        settings.order_base_url, settings.api_key, settings.api_secret, settings.mode, audit,
        recv_window_ms=settings.recv_window_ms,
    )
    ledger_dir = resolve_tax_ledger_dir(settings)
    try:
        client.sync_server_time()
        snapshot = fetch_account_snapshot(client, now=now)
        assert_venue_configuration(snapshot)
        written, issues = collect_and_persist_live_tax(
            client, (), ledger_dir, settings.mode.value, now=now, settings=settings,
            venue_snapshot=VenuePositionSnapshot(taken_at=snapshot.taken_at, positions=dict(snapshot.positions)),
        )
    except (DataIntegrityError, LiveTradingError) as exc:
        logger.error("[DATA] tax_collect status=FAILED ledger_dir=%s reason=%s", ledger_dir, exc)
        raise SystemExit(1) from exc
    for issue in issues:
        audit.record("tax_collect_issue", stream=issue.stream, stage=issue.stage, detail=issue.detail)
    logger.info("[DATA] tax_collect records=%d issues=%d ledger_dir=%s", written, len(issues), ledger_dir)


def _load_boundary_marks(path: Path) -> dict[str, Any]:
    import json
    from decimal import Decimal

    from src.live.tax_schema import (
        BOUNDARY_MARK_SOURCE_OPERATOR,
        BoundaryMark,
        parse_tax_decimal,
    )

    raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_float=Decimal)
    if not isinstance(raw, dict):
        raise ValueError(f"boundary marks must be a JSON object: {path}")
    out: dict[str, Any] = {}
    for symbol, value in raw.items():
        amount = parse_tax_decimal(value, field=f"boundary_marks.{symbol}")
        out[str(symbol)] = BoundaryMark(price=amount, source=BOUNDARY_MARK_SOURCE_OPERATOR)
    return out


def _run_tax_summary(args: argparse.Namespace) -> None:
    from pathlib import Path as _Path

    from src.common.errors import DataIntegrityError
    from src.live.tax_boundary_marks import derive_ohlcv_boundary_marks
    from src.live.tax_ledger import load_tax_watermark, resolve_tax_ledger_dir
    from src.live.tax_summary import (
        TaxSummaryConfig,
        summarize_tax_year,
        tax_source_for_mode,
        write_tax_summary,
    )

    settings = _settings_with_mode(args)
    ledger_dir = _Path(args.ledger_dir) if args.ledger_dir else resolve_tax_ledger_dir(settings)
    source = args.source or tax_source_for_mode(settings.mode)
    try:
        marks = _load_boundary_marks(_Path(args.boundary_marks)) if args.boundary_marks else None
        coverage = load_tax_watermark(ledger_dir / "watermark.json").coverage() if source == "venue" else None
        summary = summarize_tax_year(
            args.year, ledger_dir, source=source, config=TaxSummaryConfig.from_settings(settings),
            coverage=coverage, boundary_marks=marks,
            derive_boundary_marks=None if marks is not None else derive_ohlcv_boundary_marks,
        )
        output = _Path(args.output) if args.output else ledger_dir / "summaries" / f"tax_summary_{args.year}_{source}.json"
        write_tax_summary(summary, output)
    except (DataIntegrityError, ValueError, OSError) as exc:
        logger.error("[PORTFOLIO] tax_summary status=FAILED year=%d source=%s ledger_dir=%s reason=%s", args.year, source, ledger_dir, exc)
        raise SystemExit(1) from exc
    status = summary["reconciliation"]["status"]
    logger.info("[PORTFOLIO] tax_summary year=%d source=%s mode=%s status=%s output=%s", args.year, source, summary["mode"], status, output)
    if status == "incomplete":
        codes = sorted({issue["code"] for issue in summary["reconciliation"]["issues"]})
        logger.warning("[DATA] tax_summary incomplete year=%d issues=%s", args.year, ",".join(codes))
    print(str(output))  # noqa: T201 -- prints only the finalized summary path


def _run_orderbook_capture(args: argparse.Namespace) -> None:
    import time

    import pandas as pd

    from src.live.audit import AuditLog, default_audit_log_path
    from src.live.orderbook import append_order_book_snapshots, capture_order_books, default_orderbook_dir
    from src.live.rest import BinanceFuturesRestClient
    from src.live.settings import LiveSettings

    settings = LiveSettings()
    audit = AuditLog(default_audit_log_path("orderbook_capture", for_date=pd.Timestamp.now(tz="UTC")))
    client = BinanceFuturesRestClient(
        settings.market_data_base_url,
        settings.api_key,
        settings.api_secret,
        settings.mode,
        audit,
        recv_window_ms=settings.recv_window_ms,
    )
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    duration_s = float(args.duration_s)
    interval_s = float(args.interval_s)
    depth_limit = int(args.depth_limit)
    decision_time = pd.Timestamp.now(tz="UTC")
    snaps = capture_order_books(
        client,
        symbols,
        decision_time,
        mode=settings.mode.value,
        duration_s=duration_s,
        interval_s=interval_s,
        depth_limit=depth_limit,
        max_symbols=len(symbols),
        clock=time.time,
        sleep_fn=time.sleep,
        now_fn=lambda: pd.Timestamp.now(tz="UTC"),
    )
    orderbook_dir = default_orderbook_dir()
    append_order_book_snapshots(snaps, orderbook_dir)
    import logging

    logging.getLogger("LiveCli").info("[EVAL] orderbook_capture snapshots=%d", len(snaps))


def add_live_commands(live_parser: argparse.ArgumentParser) -> None:
    """``live`` 커맨드 그룹에 shadow-cycle/daemon 서브커맨드를 등록한다."""

    subparsers = live_parser.add_subparsers(dest="live_command", required=True)

    shadow = subparsers.add_parser("shadow-cycle", help="Run one daily shadow decision cycle")
    shadow.add_argument(
        "--decision-time",
        type=_parse_decision_time,
        required=True,
        help="Decision time T as ISO8601 UTC (e.g. 2026-08-24T00:00:00Z)",
    )
    shadow.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Path to the deployed_target_weights.parquet(.enc) artifact to consume (.enc requires LIVE_ARTIFACT_KEY)",
    )
    shadow.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    shadow.set_defaults(handler=_run_shadow_cycle)

    daemon = subparsers.add_parser("daemon", help="Run the 24/7 unattended shadow-cycle scheduler")
    daemon.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Override the deployed_target_weights path to consume (default: data/state/ forward ledger)",
    )
    daemon.add_argument(
        "--state-path",
        type=str,
        default=_DEFAULT_DAEMON_STATE_PATH,
        help="Path to the daemon last-processed decision_time state JSON",
    )
    daemon.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    daemon.set_defaults(handler=_run_daemon)

    status = subparsers.add_parser("status", help="Show daemon heartbeat status")
    status.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    status.set_defaults(handler=_run_status)

    strategy_step = subparsers.add_parser("strategy-step", help="Run one strategy signal step (manual/debug)")
    strategy_step.add_argument("--date", type=_parse_decision_time, required=True, help="Decision day as ISO8601 UTC (YYYY-MM-DD)")
    strategy_step.add_argument("--artifact", type=str, default=None, help="Override the strategy weights path to append")
    strategy_step.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    strategy_step.set_defaults(handler=_run_strategy_step)

    eq = subparsers.add_parser("execution-quality-summary", help="Summarize execution quality")
    eq.set_defaults(handler=_run_execution_quality_summary)

    portfolio_state = subparsers.add_parser("portfolio-state-summary", help="Summarize portfolio state")
    portfolio_state.set_defaults(handler=_run_portfolio_state_summary)

    preflight = subparsers.add_parser("preflight", help="Run preflight checks before live trading")
    preflight.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Weights artifact path (default: LIVE_WEIGHTS_PATH or run-scoped settings; .enc requires LIVE_ARTIFACT_KEY)",
    )
    preflight.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    preflight.set_defaults(handler=_run_preflight)

    backfill = subparsers.add_parser(
        "paper-funding-backfill", help="Backfill PAPER ledger funding never accrued (dry-run unless --apply)"
    )
    backfill.add_argument("--apply", action="store_true", default=False, help="Persist the backfill to the ledger")
    backfill.add_argument(
        "--accrual-start",
        type=_parse_decision_time,
        default=None,
        help="Watermark engine start (ISO8601 UTC) when not derivable from the ledger",
    )
    backfill.add_argument(
        "--mode",
        choices=["shadow", "paper", "live_testnet", "live_mainnet"],
        default=None,
        help="Override LIVE_MODE for this run",
    )
    backfill.set_defaults(handler=_run_paper_funding_backfill)

    resync = subparsers.add_parser("ledger-resync", help="Adopt the venue position snapshot into the ledger and clear de-risk-only mode (dry-run unless --apply)")
    resync.add_argument("--apply", action="store_true", default=False, help="Persist the adjustments (backs up the ledger first)")
    resync.add_argument("--mode", choices=["live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    resync.set_defaults(handler=_run_ledger_resync)

    tax_collect = subparsers.add_parser("tax-collect", help="Collect venue trades and income into the tax ledger (live modes only)")
    tax_collect.add_argument("--mode", choices=["live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    tax_collect.set_defaults(handler=_run_tax_collect)

    tax_summary = subparsers.add_parser("tax-summary", help="Summarize one tax year (local calendar year of LIVE_TAX_TIMEZONE)")
    tax_summary.add_argument("--year", type=int, required=True, help="Tax year in LIVE_TAX_TIMEZONE (default Asia/Seoul)")
    tax_summary.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE (selects ledger dir and source)")
    tax_summary.add_argument("--ledger-dir", type=str, default=None, help="Override the ledger directory resolved from settings")
    tax_summary.add_argument("--source", choices=["venue", "simulated"], default=None, help="Override the source derived from mode")
    tax_summary.add_argument("--boundary-marks", type=str, default=None, help="JSON object {symbol: decimal string} of regime-boundary market values; omitted = close of the 1h OHLCV bar ending at the regime boundary from the local lake (null with reason when unavailable)")
    tax_summary.add_argument("--output", type=str, default=None, help="Summary JSON path (default <ledger-dir>/summaries/tax_summary_<year>_<source>.json)")
    tax_summary.set_defaults(handler=_run_tax_summary)

    ob = subparsers.add_parser("orderbook-capture", help="Capture order book snapshots")
    ob.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    ob.add_argument("--duration-s", type=float, default=1800.0, help="Duration seconds")
    ob.add_argument("--interval-s", type=float, default=10.0, help="Interval seconds")
    ob.add_argument("--depth-limit", type=int, default=20, help="Depth limit")
    ob.set_defaults(handler=_run_orderbook_capture)

    liveness = subparsers.add_parser("liveness-check", help="Evaluate daemon liveness from host container state")
    liveness.add_argument("--container-running", type=int, choices=[0, 1], default=1)
    liveness.add_argument("--container-restarting", type=int, choices=[0, 1], default=0)
    liveness.add_argument("--container-oom", type=int, choices=[0, 1], default=0)
    liveness.add_argument("--restart-count", type=int, default=0)
    liveness.add_argument("--container-started-at", type=str, default=None)
    liveness.add_argument("--state-path", type=str, default=None)
    liveness.set_defaults(handler=_run_liveness_check)
