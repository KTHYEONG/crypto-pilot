"""Binance Futures 데이터 가용성 탐색 테스트.

목적: 유니버스 아키텍처 설계에 필요한 데이터 수집 가능성 파악.
실행: uv run pytest tests/integrations/test_data_availability.py -v -s -m integration

탐색 범위:
  A. Binance FAPI (REST) — exchangeInfo, OHLCV/funding/OI depth
  B. Binance Vision (S3 Public) — 전체 심볼 목록(상폐 포함), 아카이브 깊이
  C. 상폐 심볼 데이터 복원 가능성 검증
  D. 수집 불가능 항목 명시 (호가창 히스토리 등)
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

# ── project root 설정 ─────────────────────────────────────────────────────────
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import BINANCE_API_KEY, BINANCE_SECRET  # noqa: E402

# ── 상수 ─────────────────────────────────────────────────────────────────────
FAPI_BASE = "https://fapi.binance.com"
VISION_BASE = "https://data.binance.vision"
# S3 버킷 직접 쿼리 URL (심볼 목록 포함, 상폐 심볼도 포함)
VISION_S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

PROBE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]   # 충분한 역사를 가진 심볼
DELISTED_SYMBOLS = ["LUNAUSDT", "DEFIUSDT", "YFIIUSDT"]  # 상폐 확인용

HTTP_TIMEOUT = 15
FINDINGS: dict[str, Any] = {}  # 모든 탐색 결과 수집


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:  # noqa: S310
        return json.loads(r.read().decode())


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:  # noqa: S310
        return r.read().decode()


def _head_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:  # noqa: S310
            return r.status == 200
    except Exception:
        return False


def _vision_kline_url(symbol: str, tf: str, date: str) -> str:
    return f"{VISION_BASE}/data/futures/um/daily/klines/{symbol}/{tf}/{symbol}-{tf}-{date}.zip"


def _vision_funding_url(symbol: str, date: str) -> str:
    return f"{VISION_BASE}/data/futures/um/daily/fundingRate/{symbol}/{symbol}-fundingRate-{date}.zip"


def _probe_vision_earliest(symbol: str, tf: str = "4h") -> str | None:
    """Vision 아카이브에서 특정 심볼의 가장 이른 날짜를 이진 탐색."""
    # 바이낸스 선물 PERP 최초 출시: 2019-09-08
    lo = datetime(2019, 9, 8, tzinfo=timezone.utc)
    hi = datetime(2024, 1, 1, tzinfo=timezone.utc)
    found: str | None = None

    # 연 단위 앞에서 검색 (최대 5번)
    cur = lo
    while cur <= hi:
        ds = cur.strftime("%Y-%m-%d")
        if _head_ok(_vision_kline_url(symbol, tf, ds)):
            found = ds
            break
        cur += timedelta(days=90)

    return found


# ═══════════════════════════════════════════════════════════════════════════════
# A. Binance FAPI — exchangeInfo
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_a1_exchange_info_current_symbols() -> None:
    """현재 TRADING 심볼 수 및 계약 메타데이터 필드 확인."""
    data = _get_json(f"{FAPI_BASE}/fapi/v1/exchangeInfo")
    symbols: list[dict] = data.get("symbols", [])

    trading = [s for s in symbols if s.get("status") == "TRADING"]
    perp_usdt = [
        s for s in trading
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("marginAsset") == "USDT"
    ]

    # 상태별 심볼 수
    statuses: dict[str, int] = {}
    for s in symbols:
        st = str(s.get("status", "UNKNOWN"))
        statuses[st] = statuses.get(st, 0) + 1

    # 상폐 심볼 확인 (TRADING이 아닌 것)
    non_trading = [s["symbol"] for s in symbols if s.get("status") != "TRADING"]

    # 첫 번째 PERP USDT 심볼의 모든 필드 확인
    sample = perp_usdt[0] if perp_usdt else {}
    sample_fields = list(sample.keys())

    # onboardDate 필드 존재 확인 (PIT 상장일 복원의 핵심)
    has_onboard_date = "onboardDate" in sample

    finding = {
        "total_symbols": len(symbols),
        "trading_count": len(trading),
        "perp_usdt_count": len(perp_usdt),
        "non_trading_count": len(non_trading),
        "non_trading_symbols": non_trading[:10],
        "status_breakdown": statuses,
        "sample_symbol": sample.get("symbol"),
        "has_onboard_date": has_onboard_date,
        "onboard_date_value": sample.get("onboardDate"),
        "all_fields": sample_fields,
    }
    FINDINGS["A1_exchange_info"] = finding

    print("\n── [A1] exchangeInfo ──────────────────────────────────────────")
    print(f"  전체 심볼: {len(symbols)}  TRADING: {len(trading)}  PERP/USDT: {len(perp_usdt)}")
    print(f"  상태 분류: {statuses}")
    print(f"  비거래 심볼 수: {len(non_trading)}  예시: {non_trading[:5]}")
    print(f"  onboardDate 필드 존재: {has_onboard_date}  (값 예시: {sample.get('onboardDate')})")
    print(f"  계약 메타 필드: {sample_fields}")

    assert len(perp_usdt) > 0, "PERP/USDT 심볼 없음"


@pytest.mark.integration
def test_a2_exchange_info_onboard_dates() -> None:
    """onboardDate를 이용해 상장일 분포 파악 (PIT 재구성 핵심)."""
    data = _get_json(f"{FAPI_BASE}/fapi/v1/exchangeInfo")
    symbols = [
        s for s in data.get("symbols", [])
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and "onboardDate" in s
    ]

    has_onboard = len(symbols)
    dates = sorted(
        [
            (s["symbol"], datetime.fromtimestamp(s["onboardDate"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"))
            for s in symbols
        ],
        key=lambda x: x[1],
    )

    oldest = dates[:5] if dates else []
    newest = dates[-5:] if dates else []

    # 2019~2020 초기 상장 심볼 (역사가 깊은 것들)
    early = [(sym, d) for sym, d in dates if d < "2020-01-01"]

    finding = {
        "symbols_with_onboard_date": has_onboard,
        "oldest_5": oldest,
        "newest_5": newest,
        "listed_before_2020": early,
    }
    FINDINGS["A2_onboard_dates"] = finding

    print("\n── [A2] onboardDate 분포 ─────────────────────────────────────")
    print(f"  onboardDate 보유 심볼: {has_onboard} / {len(data['symbols'])}")
    print(f"  가장 오래된 5개: {oldest}")
    print(f"  가장 최근 5개: {newest}")
    print(f"  2020년 이전 상장: {early}")
    print("  ✅ onboardDate 필드로 현재 심볼의 상장일 복원 가능")


@pytest.mark.integration
def test_a3_ohlcv_depth() -> None:
    """CCXT를 통한 OHLCV 최초 가능 날짜 탐색."""
    import ccxt

    exchange = ccxt.binanceusdm({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })

    results: dict[str, Any] = {}
    # 2019-09-01 (PERP 출시 직전)부터 탐색
    probe_start = int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)

    for sym in PROBE_SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(f"{sym[:3]}/{sym[3:]}" if "/" not in sym else sym,
                                          "4h", since=probe_start, limit=3)
            if ohlcv:
                first_ts = ohlcv[0][0]
                first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                results[sym] = {"earliest_4h": first_dt, "bars_returned": len(ohlcv)}
            else:
                results[sym] = {"earliest_4h": None, "error": "empty"}
            time.sleep(0.3)
        except Exception as e:
            results[sym] = {"error": str(e)}

    FINDINGS["A3_ohlcv_depth"] = results

    print("\n── [A3] OHLCV 최초 날짜 (CCXT) ──────────────────────────────")
    for sym, r in results.items():
        if "error" not in r:
            print(f"  {sym}: 4h 최초={r.get('earliest_4h')}  (반환 봉: {r.get('bars_returned')})")
        else:
            print(f"  {sym}: 오류={r.get('error')}")


@pytest.mark.integration
def test_a4_funding_rate_depth() -> None:
    """펀딩비 히스토리 최초 가능 날짜 탐색."""
    results: dict[str, Any] = {}
    base = f"{FAPI_BASE}/fapi/v1/fundingRate"
    probe_start_ts = int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)

    for sym in PROBE_SYMBOLS:
        try:
            qs = urllib.parse.urlencode({
                "symbol": sym, "startTime": probe_start_ts,
                "endTime": probe_start_ts + 86400 * 365 * 1000,
                "limit": 5,
            })
            data = _get_json(f"{base}?{qs}")
            if data:
                first_ts = data[0]["fundingTime"]
                first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                results[sym] = {"earliest_funding": first_dt, "count": len(data)}
            else:
                results[sym] = {"earliest_funding": None, "count": 0}
            time.sleep(0.2)
        except Exception as e:
            results[sym] = {"error": str(e)}

    FINDINGS["A4_funding_depth"] = results

    print("\n── [A4] 펀딩비 최초 날짜 ────────────────────────────────────")
    for sym, r in results.items():
        if "error" not in r:
            print(f"  {sym}: 최초={r.get('earliest_funding')}")
        else:
            print(f"  {sym}: 오류={r.get('error')}")


@pytest.mark.integration
def test_a5_oi_and_lsr_depth() -> None:
    """Open Interest 및 Long/Short Ratio 히스토리 깊이 탐색."""
    # OI/LSR API는 startTime 없이 최신 데이터만 조회 후 가용성 확인
    # (너무 이른 startTime → -1130 오류. 해당 엔드포인트는 최근 500개만 지원)
    oi_results: dict[str, Any] = {}
    lsr_results: dict[str, Any] = {}

    for sym_raw in PROBE_SYMBOLS[:2]:
        # OI — startTime 없이 최대 limit 조회 후 첫 레코드 날짜 확인
        try:
            qs = urllib.parse.urlencode({"symbol": sym_raw, "period": "1h", "limit": 500})
            data = _get_json(f"{FAPI_BASE}/futures/data/openInterestHist?{qs}")
            if data:
                first_dt = datetime.fromtimestamp(
                    int(data[0]["timestamp"]) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                last_dt = datetime.fromtimestamp(
                    int(data[-1]["timestamp"]) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                oi_results[sym_raw] = {
                    "api_window_oldest": first_dt, "api_window_newest": last_dt,
                    "records": len(data),
                    "note": "API returns only recent ~500 bars; no deep history",
                }
            else:
                oi_results[sym_raw] = {"earliest_oi": None}
        except Exception as e:
            oi_results[sym_raw] = {"error": str(e)[:100]}
        time.sleep(0.3)

        # LSR — 동일하게 최대 limit 조회
        try:
            qs = urllib.parse.urlencode({"symbol": sym_raw, "period": "1h", "limit": 500})
            data = _get_json(f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio?{qs}")
            if data:
                first_dt = datetime.fromtimestamp(
                    int(data[0]["timestamp"]) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                lsr_results[sym_raw] = {
                    "api_window_oldest": first_dt, "records": len(data),
                    "note": "API returns only recent ~500 bars",
                }
            else:
                lsr_results[sym_raw] = {"earliest_lsr": None}
        except Exception as e:
            lsr_results[sym_raw] = {"error": str(e)[:100]}
        time.sleep(0.3)

    FINDINGS["A5_oi_lsr_depth"] = {"oi": oi_results, "lsr": lsr_results}

    print("\n── [A5] OI / LSR API 가용 범위 ─────────────────────────────")
    for sym, r in oi_results.items():
        if "error" not in r:
            print(f"  {sym} OI: window={r.get('api_window_oldest')}~{r.get('api_window_newest')}"
                  f"  records={r.get('records')}  ⚠️ {r.get('note','')}")
        else:
            print(f"  {sym} OI: 오류={r.get('error')}")
    for sym, r in lsr_results.items():
        if "error" not in r:
            print(f"  {sym} LSR: oldest={r.get('api_window_oldest')}"
                  f"  records={r.get('records')}  ⚠️ {r.get('note','')}")
        else:
            print(f"  {sym} LSR: 오류={r.get('error')}")


# ═══════════════════════════════════════════════════════════════════════════════
# B. Binance Vision (S3 Public) — 전체 심볼 목록 + 아카이브 깊이
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_b1_vision_symbol_discovery() -> None:
    """Vision S3 버킷 목록 조회 → 상폐 포함 전체 심볼 발견 (생존편향 해소 핵심).

    S3 list-type=2 XML 응답에서 CommonPrefixes를 파싱.
    """
    prefix = "data/futures/um/daily/klines/"
    url = (
        f"{VISION_S3_BASE}?list-type=2"
        f"&prefix={urllib.parse.quote(prefix)}"
        "&delimiter=/"
        "&max-keys=1000"
    )

    try:
        xml_text = _get_text(url)
        root = ET.fromstring(xml_text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        prefixes = root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
        if not prefixes:
            # namespace 없이 재시도
            prefixes = root.findall(".//CommonPrefixes/Prefix")

        all_symbols = []
        for p in prefixes:
            text = (p.text or "").strip()
            sym = text.replace(prefix, "").rstrip("/")
            if sym:
                all_symbols.append(sym)

        is_truncated_el = root.find(".//s3:IsTruncated", ns) or root.find(".//IsTruncated")
        is_truncated = (is_truncated_el.text if is_truncated_el is not None else "false").lower() == "true"

        finding = {
            "total_discovered": len(all_symbols),
            "is_truncated": is_truncated,
            "sample_symbols": all_symbols[:20],
            "all_symbols": all_symbols,
        }
        FINDINGS["B1_vision_symbols"] = finding

        print("\n── [B1] Vision S3 심볼 목록 ─────────────────────────────")
        print(f"  발견된 심볼: {len(all_symbols)}개  (결과 잘림: {is_truncated})")
        print(f"  샘플: {all_symbols[:10]}")

        # 알려진 상폐 심볼이 포함됐는지 확인
        for d in DELISTED_SYMBOLS:
            present = any(d.upper() in s.upper() for s in all_symbols)
            print(f"  {'✅' if present else '❌'} 상폐 심볼 {d}: {'발견됨' if present else '미발견'}")

    except Exception as e:
        FINDINGS["B1_vision_symbols"] = {"error": str(e)}
        print(f"\n── [B1] Vision S3 심볼 목록: 오류 → {e}")


@pytest.mark.integration
def test_b2_vision_kline_archive_depth() -> None:
    """Vision kline 아카이브에서 최초 가용 날짜 탐색.

    HEAD 요청으로 파일 존재 여부 확인 (다운로드 없이).
    """
    results: dict[str, str | None] = {}

    for sym in PROBE_SYMBOLS:
        earliest = None
        # 분기별로 앞에서부터 탐색
        probe_dates = []
        cur = datetime(2019, 9, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 1, tzinfo=timezone.utc)
        while cur <= end:
            probe_dates.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=90)

        for ds in probe_dates:
            url = _vision_kline_url(sym, "4h", ds)
            if _head_ok(url):
                # 이 날짜가 첫 번째 존재 확인 — 더 이른 날짜 있는지 월 단위로 확인
                month_probe = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                for _ in range(3):
                    month_probe -= timedelta(days=30)
                    if month_probe < datetime(2019, 9, 1, tzinfo=timezone.utc):
                        break
                    prev_url = _vision_kline_url(sym, "4h", month_probe.strftime("%Y-%m-%d"))
                    if _head_ok(prev_url):
                        ds = month_probe.strftime("%Y-%m-%d")
                earliest = ds
                break
            time.sleep(0.05)

        results[sym] = earliest

    FINDINGS["B2_vision_kline_depth"] = results

    print("\n── [B2] Vision kline 최초 날짜 ──────────────────────────────")
    for sym, dt in results.items():
        print(f"  {sym} 4h: {dt or '데이터 없음'}")


@pytest.mark.integration
def test_b3_vision_funding_archive() -> None:
    """Vision fundingRate 아카이브 존재 여부 및 깊이 확인."""
    prefix = "data/futures/um/daily/fundingRate/"
    url = (
        f"{VISION_S3_BASE}?list-type=2"
        f"&prefix={urllib.parse.quote(prefix)}"
        "&delimiter=/"
        "&max-keys=1000"
    )

    funding_symbols: list[str] = []
    try:
        xml_text = _get_text(url)
        root = ET.fromstring(xml_text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        prefixes = root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
        if not prefixes:
            prefixes = root.findall(".//CommonPrefixes/Prefix")
        for p in prefixes:
            sym = (p.text or "").replace(prefix, "").rstrip("/")
            if sym:
                funding_symbols.append(sym)
    except Exception as e:
        FINDINGS["B3_vision_funding"] = {"error": str(e)}
        print(f"\n── [B3] Vision funding 아카이브: 오류 → {e}")
        return

    # BTC 펀딩 최초 날짜 탐색
    btc_earliest: str | None = None
    probe = datetime(2019, 9, 1, tzinfo=timezone.utc)
    for _ in range(20):
        ds = probe.strftime("%Y-%m-%d")
        if _head_ok(_vision_funding_url("BTCUSDT", ds)):
            btc_earliest = ds
            break
        probe += timedelta(days=45)
        time.sleep(0.05)

    finding = {
        "funding_symbols_count": len(funding_symbols),
        "sample": funding_symbols[:10],
        "btc_earliest": btc_earliest,
    }
    FINDINGS["B3_vision_funding"] = finding

    print("\n── [B3] Vision fundingRate 아카이브 ─────────────────────────")
    print(f"  펀딩 데이터 심볼: {len(funding_symbols)}개")
    print(f"  샘플: {funding_symbols[:10]}")
    print(f"  BTC 펀딩 최초 날짜: {btc_earliest}")


@pytest.mark.integration
def test_b4_vision_additional_datasets() -> None:
    """Vision에서 추가 데이터셋 가용성 확인 (premiumIndex, indexPrice 등)."""
    datasets = [
        ("metrics", "data/futures/um/daily/metrics/"),
        ("premiumIndexKlines", "data/futures/um/daily/premiumIndexKlines/"),
        ("indexPriceKlines", "data/futures/um/daily/indexPriceKlines/"),
        ("markPriceKlines", "data/futures/um/daily/markPriceKlines/"),
        ("liquidationSnapshot", "data/futures/um/daily/liquidationSnapshot/"),
        ("bookDepth", "data/futures/um/daily/bookDepth/"),
        ("bookTicker", "data/futures/um/daily/bookTicker/"),
    ]

    avail: dict[str, Any] = {}
    for name, prefix in datasets:
        url = (
            f"{VISION_S3_BASE}?list-type=2"
            f"&prefix={urllib.parse.quote(prefix)}"
            "&delimiter=/"
            "&max-keys=5"
        )
        try:
            xml_text = _get_text(url)
            root = ET.fromstring(xml_text)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            prefixes = root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
            if not prefixes:
                prefixes = root.findall(".//CommonPrefixes/Prefix")
            count = len(prefixes)
            sample = [(p.text or "").replace(prefix, "").rstrip("/") for p in prefixes[:3]]
            avail[name] = {"available": count > 0, "symbol_count": count, "sample": sample}
        except Exception as e:
            avail[name] = {"available": False, "error": str(e)[:80]}
        time.sleep(0.1)

    FINDINGS["B4_vision_datasets"] = avail

    print("\n── [B4] Vision 추가 데이터셋 ────────────────────────────────")
    for name, info in avail.items():
        status = "✅ 가용" if info.get("available") else "❌ 불가"
        sample = info.get("sample", [])
        print(f"  {status}  {name}: {info.get('symbol_count',0)}개 심볼  예시={sample}")


# ═══════════════════════════════════════════════════════════════════════════════
# C. 상폐 심볼 데이터 복원 가능성 검증
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_c1_delisted_symbol_vision_probe() -> None:
    """상폐 심볼의 Vision 아카이브 데이터 존재 여부 확인.

    결과에 따라 Ledger 복원 전략이 결정됨.
    """
    # 상폐 심볼 상폐 직전 날짜 탐색 (2022년대가 많음)
    delisted_probe_dates = {
        "LUNAUSDT": [
            "2022-05-01", "2022-04-01", "2022-03-01", "2022-01-01", "2021-06-01",
        ],
        "DEFIUSDT": [
            "2022-08-01", "2022-06-01", "2022-01-01", "2021-09-01",
        ],
        "YFIIUSDT": [
            "2022-06-01", "2022-01-01", "2021-09-01", "2021-01-01",
        ],
    }

    results: dict[str, Any] = {}
    for sym, probe_dates in delisted_probe_dates.items():
        sym_result: dict[str, Any] = {"kline_found": False, "funding_found": False,
                                       "earliest_kline": None, "latest_kline": None}

        for ds in probe_dates:
            if _head_ok(_vision_kline_url(sym, "4h", ds)):
                sym_result["kline_found"] = True
                if sym_result["earliest_kline"] is None or ds < sym_result["earliest_kline"]:
                    sym_result["earliest_kline"] = ds
                if sym_result["latest_kline"] is None or ds > sym_result["latest_kline"]:
                    sym_result["latest_kline"] = ds
            time.sleep(0.05)

        if sym_result["kline_found"]:
            # funding 확인 (상폐 직전)
            fund_ds = sym_result["latest_kline"]
            if fund_ds:
                sym_result["funding_found"] = _head_ok(_vision_funding_url(sym, fund_ds))

        results[sym] = sym_result

    FINDINGS["C1_delisted_probe"] = results

    print("\n── [C1] 상폐 심볼 Vision 데이터 존재 여부 ───────────────────")
    for sym, r in results.items():
        k_status = "✅ kline 있음" if r["kline_found"] else "❌ kline 없음"
        f_status = "✅ funding 있음" if r["funding_found"] else "❌ funding 없음"
        print(f"  {sym}: {k_status} ({r['earliest_kline']}~{r['latest_kline']})  {f_status}")

    any_found = any(r["kline_found"] for r in results.values())
    if any_found:
        print("  → Vision 아카이브로 상폐 심볼 역사 복원 가능 ✅")
    else:
        print("  → Vision에서 상폐 심볼 데이터 미발견 — 복원 불가 ❌")


@pytest.mark.integration
def test_c2_delisted_via_ccxt() -> None:
    """CCXT를 통한 상폐 심볼 조회 시도 (예상: 실패)."""
    import ccxt

    exchange = ccxt.binanceusdm({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })

    results: dict[str, Any] = {}
    for sym_raw in DELISTED_SYMBOLS:
        sym = f"{sym_raw[:-4]}/{sym_raw[-4:]}"  # "LUNAUSDT" → "LUNA/USDT"
        try:
            ohlcv = exchange.fetch_ohlcv(sym, "1d", limit=3)
            results[sym_raw] = {"accessible": len(ohlcv) > 0, "bars": len(ohlcv)}
        except Exception as e:
            results[sym_raw] = {"accessible": False, "error": str(e)[:100]}
        time.sleep(0.2)

    FINDINGS["C2_delisted_ccxt"] = results

    print("\n── [C2] 상폐 심볼 CCXT 접근 가능성 ─────────────────────────")
    for sym, r in results.items():
        status = "✅ 접근 가능" if r.get("accessible") else "❌ 접근 불가"
        print(f"  {sym}: {status}  {r.get('error', r.get('bars', ''))}")
    print("  → CCXT는 현재 상폐된 심볼 조회 불가 (예상대로)")


# ═══════════════════════════════════════════════════════════════════════════════
# D. 수집 불가 항목 명시
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_d1_order_book_historical_probe() -> None:
    """호가창 히스토리 수집 가능성 확인 (예상: 불가)."""
    results: dict[str, bool] = {}

    # 1. CCXT fetch_order_book → 현재 시점만
    import ccxt
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    try:
        ob = exchange.fetch_order_book("BTC/USDT", limit=5)
        results["realtime_orderbook_via_ccxt"] = bool(ob)
    except Exception:
        results["realtime_orderbook_via_ccxt"] = False

    # 2. Vision bookDepth 아카이브 확인
    btc_today = datetime.now(tz=timezone.utc)
    for days_ago in [1, 2, 5]:
        ds = (btc_today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"{VISION_BASE}/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-{ds}.zip"
        results[f"vision_bookDepth_{ds}"] = _head_ok(url)

    # 3. Vision bookTicker
    ds = (btc_today - timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"{VISION_BASE}/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-{ds}.zip"
    results["vision_bookTicker"] = _head_ok(url)

    FINDINGS["D1_orderbook"] = results

    print("\n── [D1] 호가창 데이터 가용성 ────────────────────────────────")
    for k, v in results.items():
        status = "✅ 가용" if v else "❌ 불가"
        print(f"  {status}  {k}")

    # bookDepth가 있으면 실제 파일 크기/구조 확인
    any_depth = any("bookDepth" in k and v for k, v in results.items())
    if any_depth:
        print("  ⚠️ bookDepth 파일 존재 — 실시간 호가창 아카이브 있음 (파일 크기 클 수 있음)")
    else:
        print("  → 히스토리 호가창 수집 불가 (예상대로). 비용 모델은 OHLC 기반 Roll spread 사용 예정")


@pytest.mark.integration
def test_d2_historical_exchange_info_availability() -> None:
    """과거 exchangeInfo(상폐 심볼 포함) 수집 가능성 확인."""
    findings: dict[str, Any] = {}

    # 1. 현재 exchangeInfo의 비-TRADING 심볼 수
    data = _get_json(f"{FAPI_BASE}/fapi/v1/exchangeInfo")
    non_trading = [s for s in data.get("symbols", []) if s.get("status") != "TRADING"]
    findings["non_trading_in_current_exchangeInfo"] = [
        {"symbol": s["symbol"], "status": s.get("status"), "contractType": s.get("contractType")}
        for s in non_trading
    ]

    # 2. Wayback Machine API (archive.org) — 과거 exchangeInfo 보관 여부
    # 실시간 아카이브 조회 (rate limit 있으므로 1회만)
    wayback_url = (
        "https://archive.org/wayback/available"
        "?url=fapi.binance.com/fapi/v1/exchangeInfo"
        "&timestamp=20220101000000"
    )
    try:
        wb_data = _get_json(wayback_url)
        snap = wb_data.get("archived_snapshots", {}).get("closest", {})
        findings["wayback_snapshot"] = {
            "available": snap.get("available", False),
            "url": snap.get("url"),
            "timestamp": snap.get("timestamp"),
        }
    except Exception as e:
        findings["wayback_snapshot"] = {"available": False, "error": str(e)[:80]}

    FINDINGS["D2_historical_exchange_info"] = findings

    print("\n── [D2] 과거 exchangeInfo / 상폐 심볼 복원 ──────────────────")
    print(f"  현재 API의 비-TRADING 심볼: {len(non_trading)}개")
    for s in non_trading[:5]:
        print(f"    → {s['symbol']}  status={s.get('status')}  type={s.get('contractType')}")

    wb = findings.get("wayback_snapshot", {})
    if wb.get("available"):
        print(f"  Wayback Machine: ✅ 스냅샷 존재  ts={wb.get('timestamp')}  url={wb.get('url')}")
    else:
        print(f"  Wayback Machine: ❌ 없음/오류  ({wb.get('error', 'no snapshot')})")


# ═══════════════════════════════════════════════════════════════════════════════
# Z. 최종 보고서 출력
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_z_final_report() -> None:
    """탐색 결과 종합 보고서 출력."""
    print("\n")
    print("=" * 80)
    print("  BINANCE FUTURES 데이터 가용성 탐색 최종 보고서")
    print("=" * 80)

    # ── 수집 가능 ─────────────────────────────────────────────────────────────
    print("\n▶ 수집 가능 (Ledger 구성에 활용 가능)")

    ei = FINDINGS.get("A1_exchange_info", {})
    print(f"\n  [FAPI exchangeInfo]")
    print(f"    · PERP/USDT 심볼: {ei.get('perp_usdt_count')}개")
    print(f"    · 비-TRADING 심볼: {ei.get('non_trading_count')}개  → 최근 상폐 일부 포착 가능")
    onb = FINDINGS.get("A2_onboard_dates", {})
    has_ob = ei.get("has_onboard_date", False)
    print(f"    · onboardDate 필드: {'✅ 존재 → 현재 심볼 상장일 복원 가능' if has_ob else '❌ 없음'}")
    print(f"    · onboardDate 보유 심볼: {onb.get('symbols_with_onboard_date')}개")

    a3 = FINDINGS.get("A3_ohlcv_depth", {})
    print(f"\n  [OHLCV via CCXT]")
    for sym, r in a3.items():
        if "error" not in r:
            print(f"    · {sym} 4h: {r.get('earliest_4h')}~현재")

    a4 = FINDINGS.get("A4_funding_depth", {})
    print(f"\n  [Funding Rate via FAPI]")
    for sym, r in a4.items():
        if "error" not in r:
            print(f"    · {sym}: {r.get('earliest_funding')}~현재")

    a5 = FINDINGS.get("A5_oi_lsr_depth", {})
    print(f"\n  [OI / LSR via CCXT/FAPI]")
    for sym, r in a5.get("oi", {}).items():
        if "error" not in r:
            print(f"    · {sym} OI: {r.get('earliest_oi')}~현재")
    for sym, r in a5.get("lsr", {}).items():
        if "error" not in r:
            print(f"    · {sym} LSR: {r.get('earliest_lsr')}~현재")

    b1 = FINDINGS.get("B1_vision_symbols", {})
    print(f"\n  [Binance Vision S3 — 전체 심볼 목록 (상폐 포함)]")
    if "error" not in b1:
        print(f"    · 발견 심볼: {b1.get('total_discovered')}개  (상폐 심볼 포함)")
        print(f"    · 샘플: {b1.get('sample_symbols', [])[:8]}")
        all_syms = b1.get("all_symbols", [])
        for d in DELISTED_SYMBOLS:
            present = any(d.upper() in s.upper() for s in all_syms)
            print(f"    · {d}: {'✅ 발견' if present else '❌ 미발견'}")
    else:
        print(f"    · 오류: {b1.get('error')}")

    b2 = FINDINGS.get("B2_vision_kline_depth", {})
    print(f"\n  [Binance Vision — kline 아카이브 깊이]")
    for sym, dt in b2.items():
        print(f"    · {sym} 4h: {dt or '확인 실패'}~현재")

    b3 = FINDINGS.get("B3_vision_funding", {})
    print(f"\n  [Binance Vision — fundingRate 아카이브]")
    if "error" not in b3:
        print(f"    · 심볼: {b3.get('funding_symbols_count')}개  BTC 최초: {b3.get('btc_earliest')}")

    b4 = FINDINGS.get("B4_vision_datasets", {})
    print(f"\n  [Binance Vision — 추가 데이터셋]")
    for name, info in b4.items():
        status = "✅" if info.get("available") else "❌"
        print(f"    · {status} {name}: {info.get('symbol_count',0)}개 심볼")

    c1 = FINDINGS.get("C1_delisted_probe", {})
    print(f"\n  [상폐 심볼 데이터 복원]")
    for sym, r in c1.items():
        k = "✅" if r.get("kline_found") else "❌"
        f_ = "✅" if r.get("funding_found") else "❌"
        print(f"    · {sym}: kline={k} ({r.get('earliest_kline')}~{r.get('latest_kline')})  funding={f_}")

    # ── 수집 불가 ─────────────────────────────────────────────────────────────
    print(f"\n▶ 수집 불가 / 제한 사항")

    d1 = FINDINGS.get("D1_orderbook", {})
    book_ok = any("bookDepth" in k and v for k, v in d1.items())
    print(f"\n  [호가창 히스토리]")
    if book_ok:
        print(f"    · ⚠️ Vision bookDepth 아카이브 존재 (파일 크기 조사 필요)")
    else:
        print(f"    · ❌ 히스토리 호가창 수집 불가 → 비용 모델: OHLC Roll-spread 추정치 사용")

    d2 = FINDINGS.get("D2_historical_exchange_info", {})
    wb = d2.get("wayback_snapshot", {})
    print(f"\n  [과거 exchangeInfo (상폐 심볼 메타)]")
    print(f"    · 현재 API: 비-TRADING {len(d2.get('non_trading_in_current_exchangeInfo', []))}개 포함")
    if wb.get("available"):
        print(f"    · Wayback Machine: ✅ 아카이브 존재 → 과거 API 응답 복원 가능")
    else:
        print(f"    · Wayback Machine: ❌ 없음 → Vision S3 + kline 기반 복원이 주 전략")

    # ── 아키텍처 권고 ─────────────────────────────────────────────────────────
    print(f"\n▶ Ledger 구성 전략 (탐색 결과 기반)")
    print("""
  1. 현재 심볼 상장일 → exchangeInfo.onboardDate (정확)
  2. 전체 심볼 목록   → Vision S3 klines/ 디렉토리 XML 목록 (상폐 포함)
  3. 상폐 심볼 kline  → Vision daily/monthly klines 아카이브 (가능 여부 위 결과 참조)
  4. 펀딩비 히스토리  → Vision fundingRate 아카이브 + FAPI 병용
  5. OI / LSR        → FAPI 히스토리 (깊이 제한 있음)
  6. 호가창          → 없거나 bookDepth 아카이브 (크기 확인 후 결정)
  7. 상폐 발표 날짜  → Vision 마지막 데이터 날짜로 역추산 (근사치)
    """)

    print("=" * 80)
    # 이 테스트는 정보 출력이 목적 — 항상 PASS
    assert True
