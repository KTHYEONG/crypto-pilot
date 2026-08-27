"""``live`` 커맨드 그룹: 섬도우 사이클 1회 실행 또는 24/7 무인 데몬 구동."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common.config import DATA_DIR

logger = logging.getLogger("LiveCli")

#: 프로덕션 기본값은 봉인된 아티팩트다(I-SEAL). 평문 .parquet 을 쓰려면 --artifact 로 명시한다.
_DEFAULT_ARTIFACT = (
    "docs/results/mhs_horizon_diagnostic_artifacts/deployed_target_weights.parquet.enc"
)
_DEFAULT_DAEMON_STATE_PATH = str(DATA_DIR / "state" / "live_daemon_last_run.json")


def _parse_decision_time(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        raise argparse.ArgumentTypeError("--decision-time must be tz-aware (UTC)")
    return ts.tz_convert("UTC")


def _run_shadow_cycle(args: argparse.Namespace) -> None:
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings

    settings = LiveSettings()
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

    settings = LiveSettings()
    # 무한루프라 정상 반환하지 않는다(프로세스 시그널로 종료).
    run_daemon(settings, Path(args.artifact), Path(args.state_path))


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

    settings = LiveSettings()
    report = run_preflight(settings, Path(args.artifact))
    for check in report.checks:
        logger.info("[PREFLIGHT] %s passed=%s detail=%s", check.name, check.passed, check.detail)
    if not report.passed:
        raise SystemExit(1)


def _run_signal_refresh(args: argparse.Namespace) -> None:
    from src.common.errors import DataIntegrityError
    from src.live.settings import LiveSettings
    from src.mhs.signal_refresh import refresh_signal_row

    settings = LiveSettings()
    # wiring: refresh_signal_row(state_path, Path(args.artifact), args.decision_time, artifact_key=settings.artifact_key)
    state_path = Path(args.state) if getattr(args, "state", None) else Path("docs/results/mhs_horizon_diagnostic_artifacts/signal_state.json")
    decision_time = getattr(args, "decision_time", None)
    if decision_time is None:
        decision_time = pd.Timestamp.now(tz="UTC").normalize()
    decision_time = (
        decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
    )
    try:
        report = refresh_signal_row(
            state_path, Path(args.artifact), decision_time, artifact_key=settings.artifact_key
        )
    except DataIntegrityError as exc:
        # I-STATE-BINDING/I-OVERLAP-PARITY/I-MEMBER-PARITY failures fail closed;
        # surface as a clean nonzero exit for a cron/systemd caller instead of
        # an uncaught traceback.
        logger.error("[EVAL] signal_refresh status=FAILED reason=%s", exc)
        raise SystemExit(1) from exc
    logger.info(
        "[EVAL] signal_refresh status=%s decision_time=%s n_symbols=%d gross=%.6f exposure_scale=%.6f elapsed_ms=%d",
        report.status,
        report.decision_time,
        report.n_symbols,
        report.gross_exposure,
        report.exposure_scale,
        int(report.elapsed_seconds * 1000),
    )


def add_live_commands(live_parser: argparse.ArgumentParser) -> None:
    """``live`` 커맨드 그룹에 shadow-cycle/daemon 서브커맨드를 등록한다."""
    # wiring: run_preflight(settings, Path(args.artifact))
    from src.live.preflight import run_preflight as _preflight_ref  # noqa: F401

    # wiring: refresh_signal_row(state_path, Path(args.artifact), args.decision_time, artifact_key=settings.artifact_key)
    from src.mhs.signal_refresh import refresh_signal_row as _refresh_signal_row_ref  # noqa: F401

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
    shadow.set_defaults(handler=_run_shadow_cycle, dry_run=False)

    daemon = subparsers.add_parser("daemon", help="Run the 24/7 unattended shadow-cycle scheduler")
    daemon.add_argument(
        "--artifact",
        type=str,
        default=_DEFAULT_ARTIFACT,
        help="Path to the deployed_target_weights.parquet(.enc) artifact to consume daily (.enc requires LIVE_ARTIFACT_KEY)",
    )
    daemon.add_argument(
        "--state-path",
        type=str,
        default=_DEFAULT_DAEMON_STATE_PATH,
        help="Path to the daemon last-processed decision_time state JSON",
    )
    daemon.set_defaults(handler=_run_daemon)

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
    preflight.set_defaults(handler=_run_preflight)

    sig = subparsers.add_parser("signal-refresh", help="Run incremental signal refresh (append one row)")
    sig.add_argument(
        "--artifact",
        type=str,
        default=_DEFAULT_ARTIFACT,
        help="Path to the deployed_target_weights.parquet(.enc) artifact to append",
    )
    sig.add_argument(
        "--state",
        type=str,
        default="docs/results/mhs_horizon_diagnostic_artifacts/signal_state.json",
        help="Path to the signal_state.json(.enc) state file",
    )
    sig.add_argument(
        "--decision-time",
        type=_parse_decision_time,
        required=False,
        default=None,
        help="Decision time T as ISO8601 UTC (default: today 00:00 UTC)",
    )
    sig.set_defaults(handler=_run_signal_refresh)
