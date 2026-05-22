"""Storage and synchronization utilities for futures universe data."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as _mad

from src.core.exchange.binance_client import BinanceClient
from src.core.exchange.binance_vision import BinanceVisionDownloader
from src.core.settings import BINANCE_API_KEY, BINANCE_SECRET, FUTURES_DATA_DIR, LOG_DIR

from .models import (
    FilterReport,
    LedgerRow,
    ManifestRow,
    RejectCode,
    SymbolMeta,
    UniverseSnapshot,
    update_ledger,
)

logger = logging.getLogger(__name__)


# --- Persistence Helpers ---

def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _symbol_meta_to_dict(meta: SymbolMeta) -> dict[str, Any]:
    payload = asdict(meta)
    payload["capacity_clip_usdt_list"] = list(meta.capacity_clip_usdt_list)
    return payload


def _symbol_meta_from_dict(payload: dict[str, Any]) -> SymbolMeta:
    return SymbolMeta(
        symbol=str(payload["symbol"]),
        role=str(payload["role"]),
        adv_usdt=float(payload["adv_usdt"]),
        execution_cost_bps=float(payload["execution_cost_bps"]),
        funding_carry_8h=float(payload["funding_carry_8h"]),
        beta_vs_market=float(payload["beta_vs_market"]),
        cluster_id=int(payload["cluster_id"]),
        tradeable_rank=int(payload["tradeable_rank"]),
        basis_annualized_mean=(
            float(payload["basis_annualized_mean"])
            if payload["basis_annualized_mean"] is not None
            else None
        ),
        basis_vol=float(payload["basis_vol"]) if payload["basis_vol"] is not None else None,
        capacity_clip_usdt_list=tuple(float(item) for item in payload["capacity_clip_usdt_list"]),
    )


def _filter_report_to_dict(report: FilterReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["stage1_reason"] = (
        report.stage1_reason.value if report.stage1_reason is not None else None
    )
    payload["stage2_reason"] = (
        report.stage2_reason.value if report.stage2_reason is not None else None
    )
    payload["stage3_reason"] = (
        report.stage3_reason.value if report.stage3_reason is not None else None
    )
    payload["stage4_reason"] = (
        report.stage4_reason.value if report.stage4_reason is not None else None
    )
    payload["stage5_reason"] = (
        report.stage5_reason.value if report.stage5_reason is not None else None
    )
    payload["stage6_reason"] = (
        report.stage6_reason.value if report.stage6_reason is not None else None
    )
    payload["audit_trail"] = list(report.audit_trail)
    return payload


def _reject_code_or_none(value: Any) -> RejectCode | None:
    if value is None:
        return None
    return RejectCode(str(value))


def _filter_report_from_dict(payload: dict[str, Any]) -> FilterReport:
    return FilterReport(
        symbol=str(payload["symbol"]),
        stage0_pass=bool(payload["stage0_pass"]),
        stage1_reason=_reject_code_or_none(payload["stage1_reason"]),
        stage1_metrics={str(k): float(v) for k, v in dict(payload["stage1_metrics"]).items()},
        stage2_reason=_reject_code_or_none(payload["stage2_reason"]),
        stage2_metrics={str(k): float(v) for k, v in dict(payload["stage2_metrics"]).items()},
        stage3_reason=_reject_code_or_none(payload["stage3_reason"]),
        stage3_metrics={str(k): float(v) for k, v in dict(payload["stage3_metrics"]).items()},
        stage4_reason=_reject_code_or_none(payload["stage4_reason"]),
        stage4_metrics={str(k): float(v) for k, v in dict(payload["stage4_metrics"]).items()},
        stage5_reason=_reject_code_or_none(payload["stage5_reason"]),
        stage5_metrics={str(k): float(v) for k, v in dict(payload["stage5_metrics"]).items()},
        stage6_reason=_reject_code_or_none(payload["stage6_reason"]),
        stage6_metrics={str(k): float(v) for k, v in dict(payload["stage6_metrics"]).items()},
        final_rank=int(payload["final_rank"]) if payload["final_rank"] is not None else None,
        final_cluster_id=(
            int(payload["final_cluster_id"])
            if payload["final_cluster_id"] is not None
            else None
        ),
        audit_trail=tuple(str(item) for item in payload["audit_trail"]),
    )


def snapshot_to_payload(snapshot: UniverseSnapshot) -> dict[str, Any]:
    """Serialize snapshot into a JSON-safe payload."""
    return {
        "as_of": snapshot.as_of,
        "tf": snapshot.tf,
        "schema_version": snapshot.schema_version,
        "config_hash": snapshot.config_hash,
        "data_manifest_hash": snapshot.data_manifest_hash,
        "basket_ref": list(snapshot.basket_ref),
        "basket_weights": list(snapshot.basket_weights),
        "selected": [_symbol_meta_to_dict(item) for item in snapshot.selected],
        "rejected": {
            symbol: _filter_report_to_dict(report)
            for symbol, report in snapshot.rejected.items()
        },
        "generated_at_utc": snapshot.generated_at_utc,
        "ledger_confidence": snapshot.ledger_confidence,
        "n_stage0": snapshot.n_stage0,
        "n_stage1_pass": snapshot.n_stage1_pass,
        "n_stage2_pass": snapshot.n_stage2_pass,
        "n_stage3_pass": snapshot.n_stage3_pass,
        "n_stage4_pass": snapshot.n_stage4_pass,
        "n_stage5_pass": snapshot.n_stage5_pass,
        "n_stage6_selected": snapshot.n_stage6_selected,
    }


def snapshot_from_payload(payload: dict[str, Any]) -> UniverseSnapshot:
    """Deserialize snapshot payload into contract object."""
    rejected_payload = {str(k): dict(v) for k, v in dict(payload["rejected"]).items()}
    return UniverseSnapshot(
        as_of=str(payload["as_of"]),
        tf=str(payload["tf"]),
        schema_version=int(payload["schema_version"]),
        config_hash=str(payload["config_hash"]),
        data_manifest_hash=str(payload["data_manifest_hash"]),
        basket_ref=tuple(str(item) for item in payload["basket_ref"]),
        basket_weights=tuple(float(item) for item in payload["basket_weights"]),
        selected=tuple(_symbol_meta_from_dict(dict(item)) for item in payload["selected"]),
        rejected={
            symbol: _filter_report_from_dict(report)
            for symbol, report in rejected_payload.items()
        },
        generated_at_utc=str(payload["generated_at_utc"]),
        ledger_confidence=str(payload["ledger_confidence"]),
        n_stage0=int(payload["n_stage0"]),
        n_stage1_pass=int(payload["n_stage1_pass"]),
        n_stage2_pass=int(payload["n_stage2_pass"]),
        n_stage3_pass=int(payload["n_stage3_pass"]),
        n_stage4_pass=int(payload["n_stage4_pass"]),
        n_stage5_pass=int(payload["n_stage5_pass"]),
        n_stage6_selected=int(payload["n_stage6_selected"]),
    )


def save_snapshot_json(snapshot: UniverseSnapshot, path: str | Path) -> Path:
    """Persist UniverseSnapshot as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot_to_payload(snapshot)
    target.write_text(_canonical_json(payload), encoding="utf-8")
    logger.info("universe_snapshot_json_saved path=%s", target)
    return target


def load_snapshot_json(path: str | Path) -> UniverseSnapshot:
    """Load UniverseSnapshot from JSON."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    logger.info("universe_snapshot_json_loaded path=%s", source)
    return snapshot_from_payload(dict(payload))


def save_snapshot_parquet(snapshot: UniverseSnapshot, path: str | Path) -> Path:
    """Persist UniverseSnapshot as one-row Parquet with JSON payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot_to_payload(snapshot)
    payload_json = _canonical_json(payload)
    frame = pd.DataFrame(
        [
            {
                "as_of": snapshot.as_of,
                "tf": snapshot.tf,
                "schema_version": snapshot.schema_version,
                "config_hash": snapshot.config_hash,
                "data_manifest_hash": snapshot.data_manifest_hash,
                "generated_at_utc": snapshot.generated_at_utc,
                "saved_at_utc": _utc_now_iso(),
                "payload_json": payload_json,
            }
        ]
    )
    frame.to_parquet(target, index=False)
    logger.info("universe_snapshot_parquet_saved path=%s", target)
    return target


def load_snapshot_parquet(path: str | Path) -> UniverseSnapshot:
    """Load UniverseSnapshot from one-row Parquet payload."""
    source = Path(path)
    frame = pd.read_parquet(source)
    if frame.empty:
        raise ValueError(f"Snapshot parquet is empty: {source}")
    payload_json = str(frame.iloc[0]["payload_json"])
    payload = json.loads(payload_json)
    logger.info("universe_snapshot_parquet_loaded path=%s", source)
    return snapshot_from_payload(dict(payload))


def hash_manifest_rows(rows: list[ManifestRow] | tuple[ManifestRow, ...]) -> str:
    """Compute deterministic SHA256 hash over manifest row set."""
    normalized = [
        {
            "symbol": row.symbol,
            "period": row.period,
            "sha256": row.sha256,
        }
        for row in rows
    ]
    normalized.sort(
        key=lambda item: (
            item["symbol"],
            item["period"],
            item["sha256"],
        )
    )
    return _hash_json({"rows": normalized})


# --- Sync Utilities ---

def _build_sync_coverage_report_rows(
    *,
    mode: str,
    start_date: date,
    end_date: date,
    symbols_total: int,
    sync_tasks_total: int,
    per_symbol_synced_days: dict[str, int],
) -> list[dict[str, object]]:
    """Build lightweight sync coverage rows for parquet persistence."""
    report_rows: list[dict[str, object]] = []
    run_ts = datetime.now().isoformat()
    synced_symbols = int(sum(1 for v in per_symbol_synced_days.values() if int(v) > 0))
    total_synced_days = int(sum(int(v) for v in per_symbol_synced_days.values()))
    coverage_ratio = (
        float(synced_symbols / max(1, sync_tasks_total)) if sync_tasks_total > 0 else 0.0
    )
    for symbol, synced_days in sorted(per_symbol_synced_days.items()):
        report_rows.append(
            {
                "run_ts_utc": run_ts,
                "sync_mode": mode,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbols_total": int(symbols_total),
                "sync_tasks_total": int(sync_tasks_total),
                "synced_symbols": synced_symbols,
                "total_synced_days": total_synced_days,
                "task_coverage_ratio": coverage_ratio,
                "symbol": str(symbol),
                "synced_days": int(synced_days),
                "is_synced": bool(int(synced_days) > 0),
            }
        )
    return report_rows


def _write_sync_coverage_report(report_rows: list[dict[str, object]]) -> None:
    """Append sync coverage report rows to parquet log file."""
    if not report_rows:
        return
    report_path = LOG_DIR / "futures/universe/sync_coverage_report.parquet"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(report_rows)
    if report_path.exists():
        try:
            df_old = pd.read_parquet(report_path)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as exc:
            logger.warning("sync_coverage_report read failed, overwrite: %s", exc)
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_parquet(report_path, index=False)
    logger.info(
        "[SYNC-COVERAGE] rows=%d file=%s",
        len(df_new),
        str(report_path),
    )


def _list_usdt_futures_symbols() -> list[str]:
    """Return all USDT perpetual futures symbols from exchangeInfo."""
    client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
    try:
        info = client.exchange.fapiPublicGetExchangeInfo()
        symbols: list[str] = []
        for sym_info in info.get("symbols", []):
            symbol = str(sym_info.get("symbol", "")).strip()
            quote_asset = str(sym_info.get("quoteAsset", "")).upper()
            contract_type = str(sym_info.get("contractType", "")).upper()
            if not symbol.endswith("USDT"):
                continue
            if quote_asset != "USDT":
                continue
            if contract_type not in {"PERPETUAL", ""}:
                continue
            symbols.append(symbol)
        return sorted(set(symbols))
    except Exception as e:
        logger.error("ExchangeInfo symbol discovery failed: %s", e)
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def smart_filter_symbols(limit: int | None = None) -> list[str]:
    """Fast-mode filter: top-volume elite symbols (optional acceleration mode)."""
    logger.info("Starting Smart Early-Exit Filtering (elite_fast)...")
    client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
    try:
        tickers = client.exchange.fapiPublicGetTicker24hr()
        df = pd.DataFrame(tickers)
        df["quoteVolume"] = pd.to_numeric(df["quoteVolume"])
        df = df[df["symbol"].str.endswith("USDT")]
        threshold_idx = int(len(df) * 0.4)
        df = df.sort_values("quoteVolume", ascending=False).head(threshold_idx)
        selected_symbols: list[str] = df["symbol"].tolist()
        if limit:
            selected_symbols = selected_symbols[:limit]
        logger.info("Smart Filter: Selected %d elite candidates.", len(selected_symbols))
        return selected_symbols
    except Exception as e:
        logger.error("Smart Filter failed: %s", e)
        return _list_usdt_futures_symbols()[: limit if limit else None]


def sync_single_symbol_data(
    symbol: str,
    start_date: date,
    end_date: date,
    downloader: BinanceVisionDownloader, # kept for signature compatibility
    onboard_date: date | None = None,
    collector: Any = None,
    sync_1d: bool = True,
    sync_4h: bool = True,
    sync_1m: bool = False,
) -> tuple[list[LedgerRow], int]:
    """개별 심볼을 동기화하고 ledger row를 생성한다."""
    if collector is None:
        from src.domain.futures.backtest.data_loader import DataCollector
        collector = DataCollector()
    
    # 1h 데이터 확보 (로컬 캐시 우선 사용)
    collector.ensure_ohlcv_data(symbol, "1h", str(start_date), str(end_date))
    
    # [Component 4] 백테스팅 필수 데이터 일괄 선 수집 (Pre-fetch)
    if sync_1d:
        collector.ensure_ohlcv_data(symbol, "1d", str(start_date), str(end_date))
    if sync_4h:
        collector.ensure_ohlcv_data(symbol, "4h", str(start_date), str(end_date))
    if sync_1m:
        collector.ensure_1m_data(symbol, str(start_date), str(end_date))

    klines_1h = collector._load_cache(symbol, "1h")
    if klines_1h.empty:
        return [], 0
    
    # 요청 범위로 필터링
    req_start = pd.to_datetime(start_date, utc=True)
    req_end = pd.to_datetime(end_date, utc=True)
    klines_1h = klines_1h[
        (klines_1h["datetime"] >= req_start) & (klines_1h["datetime"] <= req_end)
    ].copy()
    if klines_1h.empty:
        return [], 0

    # 4h 리샘플링 (Ledger용)
    agg_dict = {
        'timestamp': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    for col in ['quote_vol', 'taker_buy_base', 'taker_buy_quote']:
        if col in klines_1h.columns:
            agg_dict[col] = 'sum'

    klines_4h = (
        klines_1h.set_index("datetime")
        .resample("4h")
        .agg(agg_dict)
        .reset_index()
        .dropna(subset=["timestamp"])
    )

    # Ensure required columns exist for ledger computation
    for col in ['quote_vol', 'taker_buy_base', 'taker_buy_quote']:
        if col not in klines_4h.columns:
            klines_4h[col] = klines_4h['volume']


    # 펀딩 데이터 확보
    collector.ensure_funding_data(symbol, str(start_date), str(end_date))
    funding = pd.DataFrame()
    safe_symbol = symbol.replace("/", "_")
    funding_path = Path(FUTURES_DATA_DIR) / f"{safe_symbol}_funding.parquet"
    if funding_path.exists():
        funding = pd.read_parquet(funding_path)
        funding['datetime'] = pd.to_datetime(funding['timestamp'], unit='ms', utc=True)
        funding = funding[
            (funding["datetime"] >= req_start) & (funding["datetime"] <= req_end)
        ].copy()

    # LedgerRow 생성 (이하 기존 로직 동일)
    # vol_30d: 30일(4h*6*30=180바) 롤링 수익률 표준편차 연율화 (annualized)
    # funding_zscore: 30일 롤링 평균/std 기반 시계열 z-score
    bars_per_day = 6        # 4h 바 기준
    vol_window = 30 * bars_per_day      # 180바 = 30일
    funding_rolling_window = 30         # 펀딩비 일별 집계 후 30일 롤링

    klines_4h['date'] = klines_4h['datetime'].dt.date
    klines_4h = klines_4h.sort_values('datetime').reset_index(drop=True)
    first_date = klines_4h['date'].min()
    # 진짜 상장일: onboardDate 우선, 없으면 첫 kline
    true_first_date: date = onboard_date if onboard_date is not None else first_date

    # Fix 1: 일별 4h 봉 수 → 누적합 (PIT) — look-ahead 방지
    _bars_per_day = klines_4h.groupby('date').size()
    _cumulative_bars = _bars_per_day.cumsum()

    # 전체 시계열로 롤링 vol_30d 미리 계산 (벡터화)
    close_series = klines_4h['close'].astype(float)
    ret_series = close_series.pct_change()
    # min_periods=bars_per_day: 최소 하루치 데이터 있으면 추정값 제공
    rolling_vol = (
        ret_series.rolling(window=vol_window, min_periods=bars_per_day)
        .std()
        .mul(np.sqrt(bars_per_day * 365))
    )
    klines_4h['_vol_30d'] = rolling_vol.values

    # ADV: 일별 quote_vol 합산 (나중에 pandas groupby로 계산)
    klines_4h['_adv'] = klines_4h['quote_vol'].astype(float)
    klines_4h['_ret_abs'] = ret_series.abs()

    # Fix 5: Rolling MAD-based robust z-score (breakdown point 50%)
    # NOTE: funding_daily는 일별 집계 최대 ~2000행 소규모 배열 → Python loop 허용
    #       (CLAUDE.md Zero-Loop Policy는 OHLCV 시계열 대용량 배열에 해당)
    def _mad_zscore_series(series: pd.Series, window: int, min_periods: int = 3) -> pd.Series:
        """Compute rolling MAD-based robust z-score."""
        result = np.zeros(len(series), dtype=float)
        arr = series.to_numpy(dtype=float)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            window_data = arr[start : i + 1]
            if len(window_data) < min_periods:
                result[i] = 0.0
                continue
            med = np.median(window_data)
            mad_val = float(_mad(window_data, scale="normal"))  # 0.6745 x MAD
            z = (arr[i] - med) / mad_val if mad_val > 1e-10 else 0.0
            result[i] = float(np.clip(z, -50.0, 50.0))  # prevent extreme z values
        return pd.Series(result, index=series.index)

    # 펀딩비 일별 마지막 값 집계 + 30일 롤링 MAD z-score
    funding_daily: pd.DataFrame = pd.DataFrame()
    if not funding.empty:
        funding['_date'] = funding['datetime'].dt.date
        funding_daily = (
            funding.groupby('_date')['funding_rate']
            .last()
            .reset_index()
            .rename(columns={'_date': 'date', 'funding_rate': 'fr'})
            .sort_values('date')
        )
        funding_daily['fz'] = _mad_zscore_series(
            funding_daily['fr'], window=funding_rolling_window, min_periods=3
        )
        funding_daily = funding_daily.set_index('date')

    daily_groups = klines_4h.groupby('date')
    daily_rows = []

    for day, group in daily_groups:
        if day < start_date or day > end_date:
            continue

        adv = float(group['_adv'].sum())
        ret_abs_mean = float(group['_ret_abs'].mean()) if group['_ret_abs'].notna().any() else 0.0
        amihud = float(ret_abs_mean / adv) if adv > 0 else 0.0
        last_price = float(group['close'].iloc[-1])

        # vol_30d: 해당 일의 마지막 바 롤링값 사용
        vol_30d_val = float(group['_vol_30d'].iloc[-1])
        if not np.isfinite(vol_30d_val):
            vol_30d_val = 0.0

        fr = 0.0
        fz = 0.0
        if not funding_daily.empty and day in funding_daily.index:
            fr = float(funding_daily.loc[day, 'fr'])
            fz = float(funding_daily.loc[day, 'fz'])

        knowledge_date = (day + timedelta(days=1)).isoformat()
        listing_age = max(0, (day - true_first_date).days)
        # Fix 1: PIT 누적 봉 수 (look-ahead 방지)
        pit_bars = int(_cumulative_bars.get(day, 0))
        daily_rows.append(LedgerRow(
            symbol=symbol, date=day.isoformat(), knowledge_date=knowledge_date,
            is_listed=True, is_trading=True, status="TRADING",
            # Fix 2: true_first_date (onboardDate 우선) 사용
            first_kline_date=true_first_date.isoformat(),
            adv_usdt_median=adv, adv_usdt_mean=adv,
            has_kline=True, has_funding=not funding.empty, n_bar_gaps=0, max_gap_bars=0,
            frozen_bars=0, last_60d_coverage=1.0, n_zero_volume_bars_60d=0,
            funding_rate_8h=float(fr), listing_age_days=listing_age,
            vol_30d=vol_30d_val, risk_event_override=None,
            updated_at_utc=datetime.now().isoformat(),
            is_coverage=True,
            n_is_bars=pit_bars, expected_is_bars=pit_bars, tf="4h",
            amihud_30d=amihud, mark_price=last_price, funding_zscore=fz,
        ))
    return daily_rows, len(klines_4h)


_worker_collector = None
_worker_downloader = None


def _init_worker() -> None:
    """Initialize each worker process with singleton instances to reuse TCP sockets."""
    global _worker_collector, _worker_downloader
    from src.core.exchange.binance_vision import BinanceVisionDownloader
    from src.domain.futures.backtest.data_loader import DataCollector
    _worker_collector = DataCollector()
    _worker_downloader = BinanceVisionDownloader()


def _worker(
    args: tuple[str, date, date, date | None, bool, bool, bool],
) -> tuple[list[LedgerRow], int]:
    symbol, start, end, onboard_date, sync_1d, sync_4h, sync_1m = args
    global _worker_collector, _worker_downloader
    from src.core.exchange.binance_vision import BinanceVisionDownloader
    from src.domain.futures.backtest.data_loader import DataCollector

    if _worker_collector is None:
        _worker_collector = DataCollector()
    if _worker_downloader is None:
        _worker_downloader = BinanceVisionDownloader()
    return sync_single_symbol_data(
        symbol,
        start,
        end,
        _worker_downloader,
        onboard_date,
        collector=_worker_collector,
        sync_1d=sync_1d,
        sync_4h=sync_4h,
        sync_1m=sync_1m,
    )


def run_historical_sync(
    start_date: date,
    end_date: date,
    limit: int | None = None,
    force: bool = False,
    sync_mode: str = "full_history_master",
    symbols: list[str] | None = None,
    sync_1d: bool = True,
    sync_4h: bool = True,
    sync_1m: bool = False,
) -> None:
    """메인 동기화 오케스트레이터."""
    ledger_path = FUTURES_DATA_DIR / "universe_ledger.parquet"
    symbol_start_dates = {}
    if ledger_path.exists() and not force:
        try:
            df_ledger = pd.read_parquet(ledger_path, columns=["symbol", "date"])
            df_ledger["date"] = pd.to_datetime(df_ledger["date"], utc=True, errors="coerce").dt.date
            symbol_start_dates = df_ledger.groupby("symbol")["date"].max().to_dict()
        except Exception as e:
            logger.warning(f"Ledger load failed: {e}")

    mode = str(sync_mode or "full_history_master").strip().lower()
    if symbols is not None:
        symbols = list(dict.fromkeys(symbols))  # 중복 제거
        logger.info("Sync mode=%s targeted_symbols=%d", mode, len(symbols))
    elif mode == "elite_fast":
        symbols = smart_filter_symbols(limit=limit)
    else:
        symbols = _list_usdt_futures_symbols()
        if limit:
            symbols = symbols[:limit]
        logger.info("Sync mode=full_history_master symbols=%d", len(symbols))
    sync_tasks = []

    # Fix 2 Step C: FAPI exchangeInfo에서 onboardDate 일괄 조회
    onboard_dates: dict[str, date] = {}
    try:
        client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        info = client.exchange.fapiPublicGetExchangeInfo()
        for sym_info in info.get("symbols", []):
            s = sym_info.get("symbol", "")
            ob_ms = sym_info.get("onboardDate")
            if ob_ms:
                try:
                    onboard_dates[s] = date.fromtimestamp(int(ob_ms) / 1000)
                except Exception as e:
                    logger.debug("Invalid onboardDate for symbol=%s: %s", s, e)
        logger.info(f"onboardDate 조회 완료: {len(onboard_dates)}개 심볼")
    except Exception as e:
        logger.warning(f"onboardDate 조회 실패(fallback to first kline): {e}")

    # Ledger에 있는 데이터 중 가장 최신 날짜를 기준으로 상장 폐지 여부 판단 (180일 이상 지연시 중단)
    global_max = max(symbol_start_dates.values()) if symbol_start_dates else end_date

    for symbol in symbols:
        sym_start = start_date
        if symbol in symbol_start_dates:
            last = symbol_start_dates[symbol]
            if last >= end_date:
                continue
            # 180일 이상 데이터가 끊긴 경우 상장 폐지로 간주하고 스킵 (단, force인 경우는 제외)
            if not force and last < (global_max - timedelta(days=180)):
                continue
            sym_start = last.replace(day=1) # 월 단위 아카이브이므로 해당 월 초부터 다시 수집
        sync_tasks.append(
            (symbol, sym_start, end_date, onboard_dates.get(symbol), sync_1d, sync_4h, sync_1m)
        )

    if not sync_tasks:
        logger.info("All symbols are already up-to-date.")
        return

    logger.info(f"Syncing {len(sync_tasks)} symbols (Parallel)...")
    with multiprocessing.Pool(
        processes=min(multiprocessing.cpu_count(), 8),
        initializer=_init_worker,
    ) as pool:
        results = pool.map(_worker, sync_tasks)

    all_new = []
    per_symbol_synced_days: dict[str, int] = {}
    for (rows, _count), (symbol, _, _, _, _, _, _) in zip(results, sync_tasks, strict=False):
        per_symbol_synced_days[str(symbol)] = len(rows)
        if rows:
            df = pd.DataFrame([asdict(row) for row in rows])
            all_new.append(df)
            logger.info(f"  > {symbol} synced ({len(df)} days)")
        else:
            logger.info(f"  > {symbol} synced (0 days)")

    report_rows = _build_sync_coverage_report_rows(
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        symbols_total=len(symbols),
        sync_tasks_total=len(sync_tasks),
        per_symbol_synced_days=per_symbol_synced_days,
    )
    _write_sync_coverage_report(report_rows)

    if all_new:
        update_ledger(pd.concat(all_new, ignore_index=True), ledger_path=ledger_path)
        logger.info("Ledger update complete.")
