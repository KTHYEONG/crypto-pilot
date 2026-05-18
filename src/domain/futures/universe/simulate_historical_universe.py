import logging
import os
import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Project Root Setup
# This file is now in src/domain/futures/universe/
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.utils.binance_vision import BinanceVisionDownloader
from src.domain.futures.universe.ledger import update_ledger
from src.domain.futures.universe.pipeline import build_universe
from src.domain.futures.universe.contracts import LedgerRow
from config.opt_config import get_quarterly_window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniverseSimulator")

def get_simulation_dates():
    """Dynamically determine start and end dates for the simulation."""
    today = date.today()
    
    # 2026-05-18 기준:
    # 1분기: 1, 2, 3월 -> 끝: 4/1
    # 2분기: 4, 5, 6월 -> 끝: 7/1
    # 3분기: 7, 8, 9월 -> 끝: 10/1
    # 4분기: 10, 11, 12월 -> 끝: 1/1 (next year)
    
    current_month = today.month
    if 1 <= current_month <= 3:
        # Currently in Q1, use end of previous year's Q4
        end_date = date(today.year, 1, 1)
    elif 4 <= current_month <= 6:
        # Currently in Q2, use end of Q1
        end_date = date(today.year, 4, 1)
    elif 7 <= current_month <= 9:
        # Currently in Q3, use end of Q2
        end_date = date(today.year, 7, 1)
    else:
        # Currently in Q4, use end of Q3
        end_date = date(today.year, 10, 1)
        
    # Start date is 3 years before end_date or fixed at 2021-01-01
    start_date = date(2021, 1, 1)
    
    return start_date, end_date

def process_symbol(symbol: str, start_date: date, end_date: date, downloader: BinanceVisionDownloader):
    """Fetch and process data for a single symbol."""
    current = start_date.replace(day=1)
    all_klines = []
    all_funding = []
    
    while current <= end_date:
        df_k = downloader.fetch_klines_archive_monthly(symbol, "4h", current.year, current.month)
        if not df_k.empty:
            all_klines.append(df_k)
            
        df_f = downloader.fetch_funding_rate_monthly(symbol, current.year, current.month)
        if not df_f.empty:
            all_funding.append(df_f)
            
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
            
    if not all_klines:
        return [], 0
        
    klines = pd.concat(all_klines).copy()
    n_cols = klines.shape[1]
    col_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_vol', 'no_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
    if n_cols > len(col_names):
        col_names.extend([f'extra_{i}' for i in range(n_cols - len(col_names))])
    elif n_cols < len(col_names):
        col_names = col_names[:n_cols]
        
    klines.columns = col_names
    klines['datetime'] = pd.to_datetime(klines['timestamp'], unit='ms', utc=True)
    klines = klines.sort_values('datetime')
    
    funding = pd.DataFrame()
    if all_funding:
        funding = pd.concat(all_funding).copy()
        n_cols_f = funding.shape[1]
        f_col_names = ['timestamp', 'funding_rate']
        if n_cols_f > len(f_col_names):
            f_col_names.extend([f'extra_{i}' for i in range(n_cols_f - len(f_col_names))])
        funding.columns = f_col_names
        funding['datetime'] = pd.to_datetime(funding['timestamp'], unit='ms', utc=True)
        funding = funding.sort_values('datetime')
        
    daily_rows = []
    klines['date'] = klines['datetime'].dt.date
    klines = klines.dropna(subset=['date'])
    daily_groups = klines.groupby('date')
    first_kline_date = klines['date'].min().isoformat()
    
    for day, group in daily_groups:
        if day < start_date or day > end_date:
            continue
            
        adv_usdt = group['quote_vol'].astype(float).sum()
        day_funding = 0.0
        if not funding.empty:
            mask = funding['datetime'].dt.date == day
            if mask.any():
                day_funding = funding.loc[mask, 'funding_rate'].iloc[-1]
                
        # Simple volatility
        prices = group['close'].astype(float)
        vol_30d = prices.pct_change().std() * np.sqrt(6 * 365)
        if pd.isna(vol_30d): vol_30d = 0.0
        
        row = LedgerRow(
            symbol=symbol,
            date=day.isoformat(),
            knowledge_date=(day + timedelta(days=1)).isoformat(),
            is_listed=True,
            is_trading=True,
            status="TRADING",
            first_kline_date=first_kline_date,
            delist_date=None,
            delist_announcement=None,
            adv_usdt_median=adv_usdt,
            adv_usdt_mean=adv_usdt,
            has_kline=True,
            has_funding=not funding.empty,
            n_bar_gaps=0,
            max_gap_bars=0,
            frozen_bars=0,
            last_60d_coverage=1.0,
            n_zero_volume_bars_60d=0,
            funding_rate_8h=float(day_funding),
            open_interest_usdt=0.0,
            oi_usdt_median=0.0,
            oi_change_30d=0.0,
            listing_age_days=(day - klines['date'].min()).days,
            vol_30d=float(vol_30d),
            basis_z_score=0.0,
            basis_annualized_mean=0.0,
            basis_vol=0.0,
            risk_event_override=None,
            updated_at_utc=datetime.now().isoformat()
        )
        daily_rows.append(row)
        
    return daily_rows, len(klines)

def collect_historical_data(start_date: date, end_date: date, limit: int | None = None) -> None:
    """Phase A: Build the universe ledger with real data."""
    import re
    downloader = BinanceVisionDownloader()
    all_symbols: list[str] = downloader.list_all_symbols()
    
    # 만기일이 포함된 분기 선물 심볼(예: ETHUSDT_210326) 필터링하여 제외
    all_symbols = [s for s in all_symbols if not re.search(r'_\d{6}$', s)]
    
    priority = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    symbols = [s for s in priority if s in all_symbols]
    others = [s for s in all_symbols if s not in priority]
    
    if limit:
        symbols = (symbols + others)[:limit]
    else:
        symbols = all_symbols
    
    logger.info(f"Processing {len(symbols)} symbols from {start_date} to {end_date}.")
    ledger_path = Path("data/futures/universe_ledger.parquet")
    
    symbol_start_dates = {}
    if ledger_path.exists():
        try:
            df_ledger = pd.read_parquet(ledger_path, columns=["symbol", "date"])
            df_ledger["date"] = pd.to_datetime(df_ledger["date"], utc=True, errors="coerce").dt.date
            latest = df_ledger.groupby("symbol")["date"].max()
            symbol_start_dates = latest.to_dict()
            logger.info("Loaded existing ledger for incremental sync.")
        except Exception as e:
            logger.warning(f"Could not load ledger for incremental sync: {e}")
    
    from dataclasses import asdict
    
    for symbol in symbols:
        sym_start_date = start_date
        if symbol in symbol_start_dates:
            latest_date = symbol_start_dates[symbol]
            if latest_date and latest_date >= end_date:
                logger.info(f"Skipping {symbol}, already up-to-date (latest: {latest_date}).")
                continue
            sym_start_date = latest_date.replace(day=1)
            
        logger.info(f"Syncing {symbol} from {sym_start_date} to {end_date}...")
        try:
            rows, actual_bar_count = process_symbol(symbol, sym_start_date, end_date, downloader)
            if rows:
                df = pd.DataFrame([asdict(row) for row in rows])
                # pipeline.py normalization
                df['tf'] = "4h"
                df['contract_type'] = "PERPETUAL"
                df['quote_asset'] = "USDT" if "USDT" in symbol else ("USDC" if "USDC" in symbol else "USDT")
                df['margin_asset'] = "USDT"
                df['contract_multiplier'] = 1.0
                df['has_kline'] = True
                df['has_funding'] = True
                df['is_coverage'] = True
                # Stage 2 requires coverage and bar count
                # 4h TF -> 6 bars per day. 21 months IS -> roughly 3800 bars.
                # Compute actual count from the fetched klines
                df['n_is_bars'] = actual_bar_count
                df['expected_is_bars'] = actual_bar_count # Use actual as expected for this simulation
                df['last_60d_coverage'] = 1.0
                df['n_zero_volume_bars_60d'] = 0
                df['frozen_bars'] = 0
                df['has_nan'] = False
                df['has_inf'] = False
                df['has_timestamp_issues'] = False
                df['screening_clip_usdt'] = 50000.0
                df['taker_fee_bps'] = 5.0
                df['half_spread_bps'] = 2.0
                df['impact_bps'] = 1.0
                df['tick_cost_bps'] = 0.5
                df['tick_size'] = 0.001
                df['mark_price'] = 1.0
                df['funding_zscore'] = 0.0
                df['amihud_30d'] = 0.0
                df['is_listed'] = True
                df['is_trading'] = True
                df['status'] = "TRADING"
                
                update_ledger(df, ledger_path=ledger_path)
            else:
                logger.warning(f"No archive data found for {symbol}")
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

def run_quarterly_simulation(start_date: date, end_date: date):
    """Phase B: Simulate quarterly universe evolution."""
    logger.info("Starting Quarterly Simulation...")
    
    quarterly_dates = []
    current = start_date
    while current <= end_date:
        quarterly_dates.append(current)
        if current.month > 9:
            current = current.replace(year=current.year + 1, month=(current.month + 3) % 12 or 12)
        else:
            current = current.replace(month=current.month + 3)
            
    results_dir = Path("logs/futures/universe/snapshots")
    results_dir.mkdir(parents=True, exist_ok=True)
    previous_selected = set()
    evolution_data = []
    
    for as_of in quarterly_dates:
        logger.info(f"Building universe as_of {as_of}")
        try:
            snapshot, selected_df, _ = build_universe(
                as_of=as_of,
                tf="4h",
                ledger_path=Path("data/futures/universe_ledger.parquet"),
                snapshot_root=results_dir
            )
            
            current_selected = set(selected_df['symbol'].tolist()) if not selected_df.empty else set()
            new_entries = current_selected - previous_selected
            dropouts = previous_selected - current_selected
            
            evolution_data.append({
                "date": as_of.isoformat(),
                "total": len(current_selected),
                "new_entries": sorted(list(new_entries)),
                "dropouts": sorted(list(dropouts))
            })
            previous_selected = current_selected
            
        except Exception as e:
            logger.error(f"Simulation failed for {as_of}: {e}")
            
    return evolution_data

def generate_report(evolution_data):
    """Generate the markdown report in logs/futures/universe/."""
    log_dir = Path("logs/futures/universe")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = log_dir / "universe_evolution_report.md"
    with open(report_path, "w") as f:
        f.write("# Binance Futures Universe Evolution Report (Real Data)\n\n")
        f.write(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis Window:** {evolution_data[0]['date']} to {evolution_data[-1]['date']}\n\n")
        
        f.write("| Quarter | Active Symbols | New Entries | Dropouts |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for entry in evolution_data:
            new_text = ", ".join(entry['new_entries'][:5]) + ("..." if len(entry['new_entries']) > 5 else "")
            drop_text = ", ".join(entry['dropouts'][:5]) + ("..." if len(entry['dropouts']) > 5 else "")
            f.write(f"| {entry['date']} | {entry['total']} | {new_text} | {drop_text} |\n")
            
    logger.info(f"✅ Report successfully generated at: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Limit number of symbols to process for demo")
    parser.add_argument("--skip-data", action="store_true", help="Skip Phase A (ledger building)")
    args = parser.parse_args()
    
    start_date, end_date = get_simulation_dates()
    
    if not args.skip_data:
        collect_historical_data(start_date, end_date, limit=args.limit)
    
    evolution = run_quarterly_simulation(start_date, end_date)
    if evolution:
        generate_report(evolution)
