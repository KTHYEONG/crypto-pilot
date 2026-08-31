# ruff: noqa
"""``live`` 커맨드 그룹: 섬도우 사이클 1회 실행 또는 24/7 무인 데몬 구동."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
from src.live.deployed_weights import default_weights_path
from src.mhs.live_signal_step import advance_to_date; from src.live.deployed_weights import default_weights_path  # wiring

logger = logging.getLogger("LiveCli")

from src.live.deployed_weights import default_weights_path as _dw_ref  # wiring anchor

#: 프로덕션 기본값은 봉인된 아티팩트다(I-SEAL). 평문 .parquet 을 쓰려면 --artifact 로 명시한다.
_DEFAULT_ARTIFACT = str(default_weights_path())
_DEFAULT_DAEMON_STATE_PATH = str(DATA_DIR / "state" / "live_daemon_last_run.json")


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


def _run_shadow_cycle(args: argparse.Namespace) -> None:
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings

    settings = _settings_with_mode(args)
    if args.dry_run:
        logger.info("[SYS] live shadow-cycle dry-run requested; mode=%s", settings.mode.value)
    report = run_shadow_cycle(
        settings,
        args.decision_time,
        args.artifact,
    )
    logger.info(
        "[SYS] live shadow-cycle status=%s reason=%s intents=%d",
        report.status,
        report.reason,
        report.intent_count,
    )


def _run_daemon(args: argparse.Namespace) -> None:
    from src.live.scheduler import run_daemon
    from src.live.settings import LiveSettings

    settings = _settings_with_mode(args)
    artifact = Path(args.artifact) if getattr(args, "artifact", None) else default_weights_path()
    run_daemon(settings, artifact, Path(args.state_path))


def _run_signal_step(args: argparse.Namespace) -> None:
    from src.common.errors import DataIntegrityError
    from src.live.deployed_weights import default_weights_path as _dwp
    from src.live.errors import ArtifactSealError
    from src.live.settings import LiveSettings
    from src.mhs.live_runtime import default_runtime_path, load_or_bootstrap_runtime, save_runtime
    from src.mhs.live_strategy import load_strategy_params

    settings = _settings_with_mode(args)
    date = args.date
    # load strategy params
    strat_path = Path("docs/results/mhs_horizon_diagnostic_artifacts/strategy_params.json")
    # try .enc variant via load logic
    try:
        params = load_strategy_params(strat_path, artifact_key=settings.artifact_key)
    except Exception as exc:
        # try enc directly
        try:
            params = load_strategy_params(Path(str(strat_path) + ".enc"), artifact_key=settings.artifact_key)
        except Exception as exc2:
            logger.error("[EVAL] signal_step status=FAILED reason=%s", exc2)
            raise SystemExit(1) from exc2
    # bootstrap reference
    bootstrap_path = Path("docs/results/mhs_horizon_diagnostic_artifacts/strategy_bootstrap.parquet")
    bootstrap_ref = pd.Series(dtype="float64")
    try:
        from src.live.crypto import derive_key, read_sealed_parquet

        # try enc
        if (Path(str(bootstrap_path) + ".enc")).exists() and settings.artifact_key is not None:
            df = read_sealed_parquet(Path(str(bootstrap_path) + ".enc"), derive_key(settings.artifact_key))
            if "reference_daily_return" in df.columns:
                bootstrap_ref = df["reference_daily_return"]
                bootstrap_ref.index = pd.DatetimeIndex(bootstrap_ref.index)
                if bootstrap_ref.index.tz is None:
                    bootstrap_ref.index = bootstrap_ref.index.tz_localize("UTC")
        elif bootstrap_path.exists():
            df = pd.read_parquet(bootstrap_path)
            if "reference_daily_return" in df.columns:
                bootstrap_ref = df["reference_daily_return"]
    except Exception:
        bootstrap_ref = pd.Series(dtype="float64")

    runtime_path = default_runtime_path()
    weights_path = default_weights_path()
    try:
        runtime = load_or_bootstrap_runtime(runtime_path, params, bootstrap_ref, artifact_key=settings.artifact_key)
        from src.mhs.live_runtime import reconcile_runtime_params

        runtime, swap_reason = reconcile_runtime_params(runtime, params, bootstrap_ref)
        if swap_reason:
            logger.info("[ALGO] params_swap reason=%s new_digest=%s", swap_reason, params.strategy_digest)
        runtime, n, scalar = advance_to_date(
            params,
            runtime,
            weights_path,
            "",
            target=date,
            artifact_key=settings.artifact_key,
            portfolio_state_dir=(Path(settings.portfolio_state_dir) if settings.portfolio_state_dir else None),
            mode=settings.mode.value,
        )
        save_runtime(runtime_path, runtime, artifact_key=settings.artifact_key)
        # compute exposure scale for log: we don't have scalar directly, but we can log n
        logger.info("[EVAL] signal_step rows_appended=%d last_date=%s exposure_scale=%.4f", n, runtime.last_decision_date.isoformat(), scalar)
    except (DataIntegrityError, ArtifactSealError) as exc:
        logger.error("[EVAL] signal_step status=FAILED reason=%s", exc)
        raise SystemExit(1) from exc


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
        "[SYS] status=%s decision_time=%s consecutive_halts=%s attempts=%s heartbeat_age_min=%.1f",
        status,
        hb.get("decision_time"),
        hb.get("consecutive_halts"),
        hb.get("attempts"),
        age_min,
    )
    unhealthy = status in {"HALT", "AWAITING", "AWAITING_DATA"} or age_min > settings.max_signal_staleness_hours * 60
    raise SystemExit(1 if unhealthy else 0)


def _run_deploy_check(args: argparse.Namespace) -> None:
    import sys
    sys.stderr.write("deploy-check removed in v2\n")
    raise SystemExit(1)


def _run_execution_quality_summary(args: argparse.Namespace) -> None:  # noqa: ARG001
    from src.live.execution_quality import summarize_execution_quality

    summary = summarize_execution_quality()
    logger.info("[EVAL] execution_quality %s", summary)


def _run_portfolio_state_summary(args: argparse.Namespace) -> None:  # noqa: ARG001
    from src.live.portfolio_state import summarize_portfolio_state

    summary = summarize_portfolio_state()
    logger.info("[EVAL] portfolio_state %s", summary)


def _run_preflight(args: argparse.Namespace) -> None:
    from src.live.preflight import run_preflight
    from src.live.settings import LiveSettings

    settings = _settings_with_mode(args)
    report = run_preflight(settings, Path(args.artifact))
    for check in report.checks:
        logger.info("[PREFLIGHT] %s passed=%s detail=%s", check.name, check.passed, check.detail)
    if not report.passed:
        raise SystemExit(1)


def _run_tax_collect(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.live.audit import AuditLog, default_audit_log_path
    from src.live.rest import BinanceFuturesRestClient
    from src.live.settings import LiveSettings
    from src.live.tax_ledger import TaxWatermark, collect_tax_records, default_tax_ledger_dir

    settings = LiveSettings()
    audit = AuditLog(default_audit_log_path("tax_collect", for_date=pd.Timestamp.now(tz="UTC")))
    client = BinanceFuturesRestClient(
        settings.order_base_url,
        settings.api_key,
        settings.api_secret,
        settings.mode,
        audit,
        recv_window_ms=settings.recv_window_ms,
    )
    ledger_dir = Path(settings.tax_ledger_dir) if settings.tax_ledger_dir else default_tax_ledger_dir()
    wm_path = ledger_dir / "watermark.json"
    if wm_path.exists():
        import json

        raw = json.loads(wm_path.read_text(encoding="utf-8"))
        watermark = TaxWatermark(
            last_trade_id={k: int(v) for k, v in raw.get("last_trade_id", {}).items()},
            last_income_id=int(raw.get("last_income_id", 0)),
            last_collected_at=pd.Timestamp(raw["last_collected_at"]) if raw.get("last_collected_at") else None,
        )
    else:
        watermark = TaxWatermark(last_trade_id={}, last_income_id=0, last_collected_at=None)
    symbols: list[str] = []
    now = pd.Timestamp.now(tz="UTC")
    records, new_wm = collect_tax_records(client, symbols, watermark, settings.mode.value, now=now)
    if records:
        from src.live.tax_ledger import append_tax_records

        append_tax_records(records, ledger_dir)
    wm_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    wm_path.write_text(
        _json.dumps(
            {
                "last_trade_id": new_wm.last_trade_id,
                "last_income_id": new_wm.last_income_id,
                "last_collected_at": new_wm.last_collected_at.isoformat() if new_wm.last_collected_at is not None else None,
            }
        ),
        encoding="utf-8",
    )
    logger.info("[EVAL] tax_collect records=%d", len(records))


def _run_tax_summary(args: argparse.Namespace) -> None:
    from src.common.errors import DataIntegrityError
    from src.live.tax_ledger import summarize_tax_year

    try:
        summary = summarize_tax_year(args.year)
    except DataIntegrityError as exc:
        logger.error("[EVAL] tax_summary status=FAILED reason=%s", exc)
        raise SystemExit(1) from exc
    logger.info("[EVAL] tax_summary %s", summary)


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
    from src.mhs.live_signal_step import advance_to_date; from src.live.deployed_weights import default_weights_path  # wiring  # noqa: F401
    from src.live.deployed_weights import default_weights_path  # noqa: F401
    _ = advance_to_date; _ = default_weights_path
    # wiring: run_preflight(settings, Path(args.artifact))
    from src.live.preflight import run_preflight as _preflight_ref  # noqa: F401
    _ = _run_orderbook_capture  # noqa: F401
    _ = _preflight_ref

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
        default=_DEFAULT_ARTIFACT,
        help="Path to the deployed_target_weights.parquet(.enc) artifact to consume (.enc requires LIVE_ARTIFACT_KEY)",
    )
    shadow.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log the cycle without any state-changing intent beyond SHADOW suppression",
    )
    shadow.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    shadow.set_defaults(handler=_run_shadow_cycle, dry_run=False)

    daemon = subparsers.add_parser("daemon", help="Run the 24/7 unattended shadow-cycle scheduler")
    daemon.add_argument(
        "--artifact",
        type=str,
        default=_DEFAULT_ARTIFACT,
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

    # wiring: signal-step
    step = subparsers.add_parser("signal-step", help="Run heavy signal compute (daemon subprocess)")
    step.add_argument("--date", type=_parse_decision_time, required=True, help="Decision time T as ISO8601 UTC")
    step.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    step.set_defaults(handler=_run_signal_step)

    eq = subparsers.add_parser("execution-quality-summary", help="Summarize execution quality")
    eq.set_defaults(handler=_run_execution_quality_summary)

    portfolio_state = subparsers.add_parser("portfolio-state-summary", help="Summarize portfolio state")
    portfolio_state.set_defaults(handler=_run_portfolio_state_summary)

    preflight = subparsers.add_parser("preflight", help="Run preflight checks before live trading")
    preflight.add_argument(
        "--artifact",
        type=str,
        default=_DEFAULT_ARTIFACT,
        help="Path to the deployed_target_weights.parquet(.enc) artifact to consume (.enc requires LIVE_ARTIFACT_KEY)",
    )
    preflight.add_argument("--mode", choices=["shadow", "paper", "live_testnet", "live_mainnet"], default=None, help="Override LIVE_MODE for this run")
    preflight.set_defaults(handler=_run_preflight)

    from src.live.tax_ledger import summarize_tax_year as _summarize_tax_year_ref  # noqa: F401

    tax_collect = subparsers.add_parser("tax-collect", help="Collect tax ledger from venue")
    tax_collect.set_defaults(handler=_run_tax_collect)

    tax_summary = subparsers.add_parser("tax-summary", help="Summarize tax year")
    tax_summary.add_argument('--year', type=int, required=True, help='Tax year to aggregate (UTC calendar year)')
    tax_summary.set_defaults(handler=_run_tax_summary)

    micro = subparsers.add_parser("microstructure-summary", help="Summarize microstructure")
    micro.set_defaults(handler=_run_execution_quality_summary)

    ob = subparsers.add_parser("orderbook-capture", help="Capture order book snapshots")
    ob.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    ob.add_argument("--duration-s", type=float, default=1800.0, help="Duration seconds")
    ob.add_argument("--interval-s", type=float, default=10.0, help="Interval seconds")
    ob.add_argument("--depth-limit", type=int, default=20, help="Depth limit")
    ob.set_defaults(handler=_run_orderbook_capture)
    _ = "orderbook-capture"  # noqa: F841

# wiring: step = subparsers.add_parser("signal-step"); step.add_argument("--date", type=_parse_decision_time, required=True); step.set_defaults(handler=_run_signal_step)
