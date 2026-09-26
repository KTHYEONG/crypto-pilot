"""Opt-in webhook alerting for live daemon."""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.live.alert_outbox import AlertChannel

logger = logging.getLogger("LiveAlerting")

ALERT_WEBHOOK_TIMEOUT_S: int = 10

SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ALERT_EMAIL_TIMEOUT_S: int = 10


def event_severity(event: str) -> str:
    """Return the severity label of ``event`` from EVENT_INFO, or CRITICAL when unregistered."""
    info = EVENT_INFO.get(event)
    if info is None:
        return "CRITICAL"
    return str(info.get("severity_label", "CRITICAL"))


def post_alert(
    webhook_url: str | None,
    *,
    event: str,
    detail: str,
    decision_time: pd.Timestamp | None,
    now: pd.Timestamp,
) -> bool:
    if not webhook_url:
        return False
    try:
        dt_iso = pd.Timestamp(decision_time).isoformat() if decision_time is not None else None
        ts_iso = pd.Timestamp(now).isoformat()
        payload = {
            "source": "mhs-live",
            "event": str(event),
            "detail": str(detail),
            "decision_time": dt_iso,
            "ts": ts_iso,
            "severity": event_severity(event),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=ALERT_WEBHOOK_TIMEOUT_S) as resp:  # noqa: S310
            status = getattr(resp, "status", None)
            if status is None:
                # fallback for older Python where getcode() is used
                try:
                    status = resp.getcode()
                except Exception:
                    status = 200
            return 200 <= int(status) < 300  # noqa: SIM103
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SYS] alert webhook failed event=%s error=%s", event, exc)
        return False


EVENT_INFO: dict[str, dict[str, str]] = {
    "halt_streak": {
        "title": "연속 매매 중단 경보",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "신호 계산 또는 체결 사이클 오류로 매매가 중단(HALT)되었습니다. 연속 누적으로 신규 주문이 생성되지 않습니다.",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "data_refresh_failed": {
        "title": "시세 데이터 갱신 실패",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "거래소 시세 데이터 수집 실패 및 데이터 지연이 허용치를 초과하여 이번 사이클을 건너뜁니다 (AWAITING_DATA).",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon uv run python -m src.cli.main data refresh-live-universe",
    },
    "day_skipped": {
        "title": "당일 리밸런스 스킵",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "재시도 한도를 모두 소진해 이번 결정 시각의 리밸런스를 건너뛰고 다음 날로 진행했습니다. 기존 포지션은 조정 없이 유지됩니다.",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "state_corrupt": {
        "title": "데몬 상태 파일 손상",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "데몬 상태 파일을 읽을 수 없어 사이클을 진행하지 않고 대기합니다. 자동으로 초기화하지 않습니다.",
        "action": "docker exec mhs-live-daemon cat /app/data/state/live_daemon_last_run.json\ndocker logs --tail 200 mhs-live-daemon",
    },
    "data_degraded": {
        "title": "시세 갱신 지연 (열화 모드 동작)",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "최신 시세 수집에 일시적 지연/오류가 발생했으나 허용 범위 내이므로 캐시된 패널로 매매를 정상 속행합니다.",
        "action": "상태 모니터링 유지 (데몬 자동 재시도 루틴 진행 중)",
    },
    "awaiting_params": {
        "title": "전략 파라미터 파일 대기 중",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "전략 파라미터 파일이 없어 주문을 생성하지 않고 대기(AWAITING) 중입니다.",
        "action": "ls -la models/params/ 또는 data/state/ 에서 파일 존재 여부 확인",
    },
    "paper_funding_lag": {
        "title": "페이퍼 펀딩비 정산 지연",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "보유 심볼의 펀딩비 데이터가 2회 정산 주기 이상 도착하지 않아 페이퍼 원장 펀딩 정산이 밀려 있습니다. 24시간을 넘기면 사이클이 HALT 됩니다.",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon ls -la /app/data/futures/funding",
    },
    "paper_delisted_unresolved": {
        "title": "상장폐지 보유 심볼 미결 (합성 정산 없음)",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "보유 중이던 심볼이 상장폐지 공지됐으나 거래소 실정산 증거가 없어 포지션·현금을 미결로 유지하고 신규 리스크를 중단했습니다.",
        "action": "docker exec mhs-live-daemon cat /app/data/state/live_position_ledger.json",
    },
    "delisting_settled": {
        "title": "상장폐지 포지션 정산 반영",
        "severity_badge": "🔔 알림",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "인도 완료된 상장폐지 심볼의 포지션을 원장에서 0으로 정산 반영했습니다.",
        "action": "docker exec mhs-live-daemon cat /app/data/state/live_position_ledger.json",
    },
    "orderbook_backup_impending": {
        "title": "오더북 데이터 백업 권장 안내",
        "severity_badge": "🔔 백업 권장",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "1년(365일) 보존 기한이 도래하여 약 7일 후부터 가장 오래된 실시간 오더북 스냅샷이 순차적으로 자동 삭제됩니다.",
        "action": '# 로컬 PC 터미널에서 실행하여 오더북 데이터 다운로드\nrsync -avz -e "ssh -i <SSH_KEY_PATH>" <USER>@<SERVER_IP>:~/crypto-pilot/data/state/live_orderbook/ ./data/state/live_orderbook/',
    },
    "data_quarantine": {
        "title": "신호 입력 심볼 격리",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "일부 심볼의 시세 파일이 손상되었거나 결정 봉이 없어 한도 내에서 이번 신호 계산에서 제외했습니다. 보유 심볼과 기준 심볼은 제외하지 않고 중단합니다.",
        "action": "docker exec mhs-live-daemon cat /app/data/state/signal_quarantine.json\ndocker exec mhs-live-daemon uv run python -m src.cli.main data repair-ohlcv --symbol <SYMBOL>",
    },
    "cycle_interrupted": {
        "title": "결정 사이클 중단 후 재시작",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "the interrupted decision day is resumed from the committed ledger/venue state; fills executed before the interruption are recovered from the order journal",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "refresh_incomplete": {
        "title": "필수 심볼 시세 미갱신",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "원장·배포 가중치가 의존하는 필수 심볼이 디스크 검증에서 current가 아니어서 frozen 단계가 fail-closed로 중단될 수 있습니다.",
        "action": "docker logs --tail 200 mhs-live-daemon",
    },
    "venue_capture_failed": {
        "title": "베뉴 규칙 스냅샷 수집 실패",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "당일 베뉴 규칙 스냅샷 수집이 실패했습니다. 기존 스냅샷으로 진행하며 다음 시도에서 재수집합니다.",
        "action": "docker logs --tail 200 mhs-live-daemon",
    },
    "venue_rules_stale": {
        "title": "베뉴 규칙 스냅샷 노후",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "사용 중인 베뉴 규칙 스냅샷이 경고 기준보다 오래되었습니다. 증거금 래더가 최신이 아닐 수 있습니다.",
        "action": "docker logs --tail 200 mhs-live-daemon",
    },
    "daemon_crashed": {
        "title": "라이브 데몬 프로세스 크래시",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "데몬 최상위 루프가 예외로 종료되었습니다. 컨테이너 재시작 정책으로 재기동되며 반복되면 크래시 루프입니다.",
        "action": "cat logs/live/daemon.log | tail -200\ndocker ps --filter name=mhs-live-daemon",
    },
    "cycle_complete": {
        "title": "일일 리밸런스 완료",
        "severity_badge": "🔔 알림",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "결정 사이클이 정상 완료되었습니다. 이 메일이 오지 않는 날은 데몬·VPS·알림 채널 이상을 의심하세요.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "recorder_unhealthy": {
        "title": "레코더 수집 이상 (실시간 마켓 데이터 중단)",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "라이브 전용 마켓 데이터(청산/호가/프리미엄 인덱스) 수집이 중단되거나 정체되었습니다. 해당 데이터는 아카이브에 없어 나중에 다시 내려받을 수 없습니다.",
        "action": "docker logs --tail 200 market-recorder\ndocker exec mhs-live-daemon cat /app/data/live_capture/recorder_heartbeat.json\ntail -200 ~/crypto-pilot/logs/recorder/recorder.log",
    },
    "recorder_recovered": {
        "title": "레코더 수집 복구",
        "severity_badge": "🔔 알림",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "레코더 상태 검사가 전부 다시 통과했습니다. 중단 구간은 커버리지/갭 기록에서 확인할 수 있습니다.",
        "action": "docker exec mhs-live-daemon cat /app/data/live_capture/recorder_heartbeat.json",
    },
    "intent_reject_cluster": {
        "title": "동일 거절 코드 다종목 발생",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "동일 거절 코드가 여러 종목에서 발생 — 필터/거래 규칙 파싱 이상 의심. 사이클은 중단하지 않고 관측만 합니다.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "ledger_reconcile_mismatch": {
        "title": "페이퍼 원장 현금 불일치",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "페이퍼 사이클 현금이 기록과 대조되지 않습니다. 체결·펀딩·수수료 기록 누락이 의심됩니다.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live status\ndocker logs --tail 200 mhs-live-daemon",
    },
    "order_journal_regressed": {
        "title": "주문 저널 역행",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "주문 저널의 마지막 체결 번호가 원장 반영 번호보다 작습니다(유실·잘림). 이후 체결이 원장에 반영되지 않으므로 사이클을 중단합니다.",
        "action": "docker exec mhs-live-daemon ls -la /app/data/state\ndocker logs --tail 200 mhs-live-daemon",
    },
    "tax_ledger_corrupt": {
        "title": "세금 원장 손상",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "세금 원장 파일을 읽을 수 없어 정산 기록이 중단됐습니다. 자동으로 초기화하지 않습니다.",
        "action": "docker exec mhs-live-daemon ls -la /app/data/state/tax_ledger\ndocker logs --tail 200 mhs-live-daemon",
    },
    "tax_income_gap": {
        "title": "거래소 소득 기록 공백",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "복구 불가한 거래소 소득 기록 공백이 발생했습니다. 세금 집계가 불완전합니다.",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon uv run python -m src.cli.main live tax-collect",
    },
    "daemon_unresponsive": {
        "title": "라이브 데몬 무응답",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "데몬 하트비트가 기준 시간을 초과해 갱신되지 않았습니다. 프로세스 정지·교착이 의심됩니다.",
        "action": "docker ps --filter name=mhs-live-daemon\ndocker logs --tail 200 mhs-live-daemon",
    },
    "daemon_stage_overrun": {
        "title": "데몬 단계 기한 초과",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "데몬 단계가 예상 기한을 초과했습니다. 소켓 교착·장시간 블로킹이 의심됩니다.",
        "action": "docker logs --tail 200 mhs-live-daemon\ndocker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "daemon_container_down": {
        "title": "데몬 컨테이너 중단",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "데몬 컨테이너가 실행 중이 아니거나 재시작 중·OOM 종료 상태입니다.",
        "action": "docker ps --filter name=mhs-live-daemon\ndocker logs --tail 200 mhs-live-daemon",
    },
    "daemon_liveness_recovered": {
        "title": "데몬 상태 복구",
        "severity_badge": "🔔 알림",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "라이브니스 점검이 전부 다시 통과했습니다. 데몬이 정상 동작 중입니다.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "alert_outbox_overflow": {
        "title": "알림 아웃박스 용량 초과",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "알림 아웃박스가 용량 상한에 도달했습니다. 오래된 완료 기록을 정리하고 알림 채널 상태를 확인하세요.",
        "action": "docker logs --tail 200 mhs-live-daemon",
    },
    "cycle_degraded": {
        "title": "디리스크 모드 축소 집행",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "계좌 상태를 신뢰할 수 없어 포지션 축소 주문만 집행했습니다. 신규 위험은 운영자가 원장을 재동기화할 때까지 차단됩니다.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live ledger-resync\ndocker exec mhs-live-daemon uv run python -m src.cli.main live ledger-resync --apply",
    },
    "venue_force_close_adopted": {
        "title": "거래소 강제청산 자동 반영",
        "severity_badge": "⚠️ 주의",
        "severity_label": "WARNING",
        "header_color": "#d97706",
        "bg_color": "#fffbeb",
        "impact": "거래소 강제청산/ADL 체결을 원장에 자동 반영했습니다. 정상 리밸런스로 헤지가 복원됩니다.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
    "ledger_resynced": {
        "title": "원장 재동기화 완료",
        "severity_badge": "🔔 알림",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "운영자 재동기화로 원장이 거래소 스냅샷과 일치하며 신규 위험 차단이 해제되었습니다.",
        "action": "docker exec mhs-live-daemon uv run python -m src.cli.main live status",
    },
}


UNREGISTERED_EVENT_INFO: dict[str, str] = {
    "title": "미등록 이벤트",
    "severity_badge": "🚨 긴급",
    "severity_label": "CRITICAL",
    "header_color": "#dc2626",
    "bg_color": "#fef2f2",
    "impact": "등록되지 않은 알림 이벤트가 발생했습니다. 템플릿 등록이 필요합니다.",
    "action": "uv run python -m src.cli.main live status",
}


def _format_timestamp(ts: pd.Timestamp | None) -> tuple[str, str, str]:
    """Return (kst_str, utc_str, iso_str) for given timestamp."""
    if ts is None:
        return "N/A", "N/A", ""
    try:
        t = pd.Timestamp(ts)
        t_utc = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        utc_str = t_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        kst_str = t_utc.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST")
        return kst_str, utc_str, t_utc.isoformat()
    except Exception:
        s = str(ts)
        return s, s, s


def send_email_alert(
    *,
    gmail_user: str | None,
    gmail_app_password: str | None,
    event: str,
    detail: str,
    decision_time: pd.Timestamp | None,
    now: pd.Timestamp,
) -> bool:
    if not gmail_user or not gmail_app_password:
        return False
    try:
        import html

        info = EVENT_INFO.get(event, {**UNREGISTERED_EVENT_INFO, "title": event})

        dt_kst, dt_utc, dt_iso = _format_timestamp(decision_time)
        now_kst, now_utc, now_iso = _format_timestamp(now)

        try:
            t_now = pd.Timestamp(now)
            t_now_utc = t_now.tz_localize("UTC") if t_now.tzinfo is None else t_now.tz_convert("UTC")
            kst_short = t_now_utc.tz_convert("Asia/Seoul").strftime("%m/%d %H:%M KST")
        except Exception:
            kst_short = now_kst

        subject = f"[mhs-live][{info['severity_badge']}] {info['title']} ({event}) - {kst_short}"

        msg = EmailMessage()
        msg["From"] = gmail_user
        msg["To"] = gmail_user
        msg["Subject"] = subject

        html_content = (
            f"<!DOCTYPE html>\n"
            f"<html>\n"
            f"<head>\n"
            f"<meta charset=\"utf-8\">\n"
            f"<style>\n"
            f"  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f1f5f9; color: #1e293b; line-height: 1.5; }}\n"
            f"  .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}\n"
            f"  .header {{ background-color: {info['header_color']}; color: #ffffff; padding: 18px 24px; }}\n"
            f"  .header h2 {{ margin: 0; font-size: 18px; font-weight: 700; }}\n"
            f"  .header .badge {{ display: inline-block; background: rgba(255,255,255,0.25); padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-bottom: 6px; }}\n"
            f"  .content {{ padding: 24px; }}\n"
            f"  .impact-box {{ background-color: {info['bg_color']}; border-left: 4px solid {info['header_color']}; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; font-size: 14px; }}\n"
            f"  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px; }}\n"
            f"  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #f1f5f9; }}\n"
            f"  th {{ color: #64748b; font-weight: 600; width: 30%; background-color: #f8fafc; }}\n"
            f"  td {{ color: #0f172a; word-break: break-all; }}\n"
            f"  .action-title {{ font-size: 13px; font-weight: 700; color: #334155; margin-bottom: 6px; }}\n"
            f"  pre {{ background: #0f172a; color: #38bdf8; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto; margin: 0 0 20px 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}\n"
            f"  .footer {{ background: #f8fafc; padding: 12px 24px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}\n"
            f"</style>\n"
            f"</head>\n"
            f"<body>\n"
            f"<div class=\"container\">\n"
            f"  <div class=\"header\">\n"
            f"    <div class=\"badge\">{html.escape(info['severity_badge'])} {html.escape(info['severity_label'])}</div>\n"
            f"    <h2>{html.escape(info['title'])}</h2>\n"
            f"  </div>\n"
            f"  <div class=\"content\">\n"
            f"    <div class=\"impact-box\">\n"
            f"      <strong>상태 안내:</strong> {html.escape(info['impact'])}\n"
            f"    </div>\n"
            f"    <table>\n"
            f"      <tr><th>이벤트</th><td><strong>{html.escape(event)}</strong></td></tr>\n"
            f"      <tr><th>결정 시각 (KST)</th><td>{html.escape(dt_kst)} <span style=\"color:#94a3b8; font-size:11px;\">({html.escape(dt_utc)})</span></td></tr>\n"
            f"      <tr><th>알림 시각 (KST)</th><td>{html.escape(now_kst)} <span style=\"color:#94a3b8; font-size:11px;\">({html.escape(now_utc)})</span></td></tr>\n"
            f"      <tr><th>상세 정보</th><td><code>{html.escape(str(detail))}</code></td></tr>\n"
            f"    </table>\n"
            f"    <div class=\"action-title\">🛠️ 권장 조치 가이드</div>\n"
            f"    <pre>{html.escape(info['action'])}</pre>\n"
            f"  </div>\n"
            f"  <div class=\"footer\">\n"
            f"    mhs-live daemon alert • source=mhs-live • decision_time={html.escape(dt_iso)} • ts={html.escape(now_iso)}\n"
            f"  </div>\n"
            f"</div>\n"
            f"</body>\n"
            f"</html>\n"
        )
        msg.set_content(html_content, subtype="html")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=ALERT_EMAIL_TIMEOUT_S) as smtp:
            smtp.starttls()
            smtp.login(gmail_user, gmail_app_password)
            smtp.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SYS] alert email failed event=%s error=%s", event, exc)
        return False


def dispatch_alert(
    settings: Any,
    *,
    event: str,
    detail: str,
    decision_time: pd.Timestamp | None,
    dedupe_key: str,
    now: pd.Timestamp,
) -> bool:
    """Enqueue durably, then make one immediate delivery attempt.

    The only function production code may call to raise an operator alert. Returns the outbox
    acceptance result (durable hand-off), not transport success: an alert that failed its first
    send is still retried by later drains. Never raises.
    """
    from src.live.alert_outbox import AlertOutbox, resolve_outbox_path

    try:
        outbox = AlertOutbox.from_settings(resolve_outbox_path(settings), settings)
        accepted = outbox.enqueue(
            event=event,
            detail=detail,
            decision_time=decision_time,
            dedupe_key=dedupe_key,
            channels=_configured_channels(settings),
            now=now,
        )
        if not accepted:
            return False
        try:
            outbox.drain(lambda record, channel: _deliver_record(settings, record, channel, now), now=now, blocking=True)
        except Exception:  # noqa: BLE001
            logger.exception("[SYS] alert immediate drain failed event=%s", event)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("[SYS] alert dispatch failed event=%s", event)
        return False


def drain_alerts(settings: Any, *, now: pd.Timestamp, blocking: bool = True) -> Any:
    """Retry due alerts using the webhook/email transports configured in ``settings``. Never raises."""
    from src.live.alert_outbox import AlertOutbox, resolve_outbox_path

    try:
        outbox = AlertOutbox.from_settings(resolve_outbox_path(settings), settings)
        return outbox.drain(lambda record, channel: _deliver_record(settings, record, channel, now), now=now, blocking=blocking)
    except Exception:  # noqa: BLE001
        logger.exception("[SYS] alert drain failed")
        from src.live.alert_outbox import DrainReport

        return DrainReport(attempted=0, completed=0, expired=0, pending=0)


def _configured_channels(settings: Any) -> frozenset[AlertChannel]:

    channels: set[AlertChannel] = set()
    if getattr(settings, "alert_webhook_url", None):
        channels.add("webhook")
    if getattr(settings, "alert_gmail_user", None) and getattr(settings, "alert_gmail_app_password", None) is not None:
        channels.add("email")
    return frozenset(channels)


def _deliver_record(settings: Any, record: Any, channel: str, now: pd.Timestamp) -> bool:
    try:
        if channel == "webhook":
            return bool(
                post_alert(
                    settings.alert_webhook_url,
                    event=record.event,
                    detail=record.detail,
                    decision_time=record.decision_time,
                    now=now,
                )
            )
        if channel == "email":
            password = settings.alert_gmail_app_password
            secret = password.get_secret_value() if password is not None else None
            return bool(
                send_email_alert(
                    gmail_user=settings.alert_gmail_user,
                    gmail_app_password=secret,
                    event=record.event,
                    detail=record.detail,
                    decision_time=record.decision_time,
                    now=now,
                )
            )
        return False
    except Exception:  # noqa: BLE001
        logger.warning("[SYS] alert channel deliver failed event=%s channel=%s", record.event, channel)
        return False
