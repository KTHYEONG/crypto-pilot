import logging
import multiprocessing
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import BINANCE_API_KEY, BINANCE_SECRET, FUTURES_DATA_DIR
from src.core.exchange.binance_client import BinanceClient
from src.core.utils.binance_vision import BinanceVisionDownloader
from src.domain.futures.universe.contracts import LedgerRow
from src.domain.futures.universe.ledger import update_ledger

logger = logging.getLogger("SyncUtils")

def smart_filter_symbols(limit: int = None) -> list[str]:
    """거래량 기반 상위 40% 엘리트 심볼 필터링.
    """
    logger.info("Starting Smart Early-Exit Filtering...")
    client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
    try:
        tickers = client.exchange.fapiPublicGetTicker24hr()
        df = pd.DataFrame(tickers)
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        df = df[df['symbol'].str.endswith('USDT')]
        
        # 상위 40% 동적 추출
        threshold_idx = int(len(df) * 0.4)
        df = df.sort_values('quoteVolume', ascending=False).head(threshold_idx)
        
        selected_symbols = df['symbol'].tolist()
        logger.info(f"Smart Filter: Selected {len(selected_symbols)} elite candidates.")
        
        if limit:
            selected_symbols = selected_symbols[:limit]
        return selected_symbols
    except Exception as e:
        logger.error(f"Smart Filter failed: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"] # 최소한의 Fallback

def sync_single_symbol_data(symbol: str, start_date: date, end_date: date, downloader: BinanceVisionDownloader):
    """개별 심볼 동기화: 1h 수집 -> 4h 리샘플링 -> 로컬 저장 -> Ledger 데이터 생성.
    """
    current = start_date.replace(day=1)
    all_klines_1h = []
    all_funding = []
    
    while current <= end_date:
        df_k = downloader.fetch_klines_archive_monthly(symbol, "1h", current.year, current.month)
        if not df_k.empty:
            all_klines_1h.append(df_k)
        df_f = downloader.fetch_funding_rate_monthly(symbol, current.year, current.month)
        if not df_f.empty:
            all_funding.append(df_f)
        
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
            
    if not all_klines_1h:
        return [], 0
        
    klines_1h = pd.concat(all_klines_1h).copy()
    # 컬럼 설정 및 타입 변환
    col_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_vol', 'no_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
    klines_1h.columns = col_names[:klines_1h.shape[1]]
    for col in klines_1h.columns:
        if col not in ('datetime', 'date'):
            klines_1h[col] = pd.to_numeric(klines_1h[col], errors='coerce')
    klines_1h['datetime'] = pd.to_datetime(klines_1h['timestamp'], unit='ms', utc=True)
    klines_1h = klines_1h.sort_values('datetime')
    
    # 4h 리샘플링 (Ledger용)
    klines_4h = klines_1h.set_index('datetime').resample('4h').agg({
        'timestamp': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_vol': 'sum',
        'taker_buy_base': 'sum',
        'taker_buy_quote': 'sum'
    }).reset_index().dropna(subset=['timestamp'])
    
    funding = pd.DataFrame()
    if all_funding:
        funding = pd.concat(all_funding).copy()
        # Binance Vision Funding Rate columns: [timestamp, interval, fundingRate]
        col_names_f = ['timestamp', 'interval_hours', 'funding_rate']
        funding.columns = col_names_f[:funding.shape[1]]
        for col in funding.columns:
            funding[col] = pd.to_numeric(funding[col], errors='coerce')
        funding['datetime'] = pd.to_datetime(funding['timestamp'], unit='ms', utc=True)
        funding = funding.sort_values('datetime')
        
    # 로컬 캐시 저장
    safe_symbol = symbol.replace("/", "_")
    klines_1h[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'taker_buy_quote']].to_parquet(
        Path(FUTURES_DATA_DIR) / f"{safe_symbol}_1h.parquet", index=False
    )
    if not funding.empty:
        funding.to_parquet(Path(FUTURES_DATA_DIR) / f"{safe_symbol}_funding.parquet", index=False)
        
    # LedgerRow 생성
    # vol_30d: 30일(4h*6*30=180바) 롤링 수익률 표준편차 연율화 (annualized)
    # funding_zscore: 30일 롤링 평균/std 기반 시계열 z-score
    bars_per_day = 6        # 4h 바 기준
    vol_window = 30 * bars_per_day      # 180바 = 30일
    funding_rolling_window = 30         # 펀딩비 일별 집계 후 30일 롤링

    klines_4h['date'] = klines_4h['datetime'].dt.date
    klines_4h = klines_4h.sort_values('datetime').reset_index(drop=True)
    first_date = klines_4h['date'].min()

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

    # 펀딩비 일별 마지막 값 집계 + 30일 롤링 z-score
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
        # 30일 롤링 z-score: (fr - rolling_mean) / rolling_std
        fr_rolling_mean = funding_daily['fr'].rolling(window=funding_rolling_window, min_periods=3).mean()
        fr_rolling_std = funding_daily['fr'].rolling(window=funding_rolling_window, min_periods=3).std(ddof=0)
        # std=0 방어: 0인 경우 z-score=0
        fr_rolling_std = fr_rolling_std.replace(0.0, np.nan)
        funding_daily['fz'] = ((funding_daily['fr'] - fr_rolling_mean) / fr_rolling_std).fillna(0.0)
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

        daily_rows.append(LedgerRow(
            symbol=symbol, date=day.isoformat(), knowledge_date=(day + timedelta(days=1)).isoformat(),
            is_listed=True, is_trading=True, status="TRADING",
            first_kline_date=first_date.isoformat(), adv_usdt_median=adv, adv_usdt_mean=adv,
            has_kline=True, has_funding=not funding.empty, n_bar_gaps=0, max_gap_bars=0,
            frozen_bars=0, last_60d_coverage=1.0, n_zero_volume_bars_60d=0,
            funding_rate_8h=float(fr), listing_age_days=(day - first_date).days,
            vol_30d=vol_30d_val, risk_event_override=None, updated_at_utc=datetime.now().isoformat(),
            is_coverage=True, n_is_bars=len(klines_4h), expected_is_bars=len(klines_4h), tf="4h",
            amihud_30d=amihud, mark_price=last_price, funding_zscore=fz,
        ))
    return daily_rows, len(klines_4h)

def _worker(args):
    symbol, start, end = args
    return sync_single_symbol_data(symbol, start, end, BinanceVisionDownloader())

def run_historical_sync(start_date: date, end_date: date, limit: int = None, force: bool = False):
    """메인 동기화 오케스트레이터.
    """
    ledger_path = Path("data/futures/universe_ledger.parquet")
    symbol_start_dates = {}
    if ledger_path.exists() and not force:
        try:
            df_ledger = pd.read_parquet(ledger_path, columns=["symbol", "date"])
            df_ledger["date"] = pd.to_datetime(df_ledger["date"], utc=True, errors="coerce").dt.date
            symbol_start_dates = df_ledger.groupby("symbol")["date"].max().to_dict()
        except Exception as e:
            logger.warning(f"Ledger load failed: {e}")

    symbols = smart_filter_symbols(limit=limit)
    sync_tasks = []
    
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
        sync_tasks.append((symbol, sym_start, end_date))

    if not sync_tasks:
        logger.info("All symbols are already up-to-date.")
        return

    logger.info(f"Syncing {len(sync_tasks)} symbols (Parallel)...")
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 8)) as pool:
        results = pool.map(_worker, sync_tasks)
        
    all_new = []
    for (rows, count), (symbol, _, _) in zip(results, sync_tasks):
        if rows:
            df = pd.DataFrame([asdict(row) for row in rows])
            all_new.append(df)
            logger.info(f"  > {symbol} synced ({len(df)} days)")

    if all_new:
        update_ledger(pd.concat(all_new, ignore_index=True), ledger_path=ledger_path)
        logger.info("Ledger update complete.")
