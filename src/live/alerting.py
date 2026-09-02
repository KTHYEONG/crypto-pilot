"""Opt-in webhook alerting for live daemon."""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from email.message import EmailMessage

import pandas as pd

logger = logging.getLogger("LiveAlerting")

ALERT_WEBHOOK_TIMEOUT_S: int = 10

SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ALERT_EMAIL_TIMEOUT_S: int = 10


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
        "action": "uv run python -m src.cli.main live status\ntail -n 50 logs/live_daemon.log",
    },
    "data_refresh_failed": {
        "title": "시세 데이터 갱신 실패",
        "severity_badge": "🚨 긴급",
        "severity_label": "CRITICAL",
        "header_color": "#dc2626",
        "bg_color": "#fef2f2",
        "impact": "거래소 시세 데이터 수집 실패 및 데이터 지연이 허용치를 초과하여 이번 사이클을 건너뜁니다 (AWAITING_DATA).",
        "action": "uv run python -m src.cli.main data refresh-live-market-data",
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
    "orderbook_backup_impending": {
        "title": "오더북 데이터 백업 권장 안내",
        "severity_badge": "🔔 백업 권장",
        "severity_label": "NOTICE",
        "header_color": "#2563eb",
        "bg_color": "#eff6ff",
        "impact": "1년(365일) 보존 기한이 도래하여 약 7일 후부터 가장 오래된 실시간 오더북 스냅샷이 순차적으로 자동 삭제됩니다.",
        "action": '# 로컬 PC 터미널에서 실행하여 오더북 데이터 다운로드\nrsync -avz -e "ssh -i <SSH_KEY_PATH>" <USER>@<SERVER_IP>:~/crypto-pilot/data/state/live_orderbook/ ./data/state/live_orderbook/',
    },
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
    email_to: str | None,
    event: str,
    detail: str,
    decision_time: pd.Timestamp | None,
    now: pd.Timestamp,
) -> bool:
    if not gmail_user or not gmail_app_password:
        return False
    recipient = email_to or gmail_user
    try:
        import html

        info = EVENT_INFO.get(
            event,
            {
                "title": event,
                "severity_badge": "🔔 알림",
                "severity_label": "INFO",
                "header_color": "#475569",
                "bg_color": "#f8fafc",
                "impact": "라이브 데몬 이벤트가 발생했습니다.",
                "action": "uv run python -m src.cli.main live status",
            },
        )

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
        msg["To"] = recipient
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
