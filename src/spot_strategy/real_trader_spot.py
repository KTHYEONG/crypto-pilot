
import time
import logging
import pandas as pd
import optuna
import os
import sys
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy
from config.settings import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY

# Setup Logging
from logging.handlers import RotatingFileHandler

log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        RotatingFileHandler(os.path.join(os.path.dirname(__file__), "logs/spot_trader.log"), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SpotTrader")

class SpotTrader:
    def __init__(self, symbols, db_path):
        self.symbols = symbols
        self.client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        
        # Load Strategy Params
        self.params = self._load_best_params(db_path)
        self.strategy = UltimateStrategy("RealSpot", self.params)
        
        # Risk Settings
        self.risk_per_trade = self.params.get('RISK_PER_TRADE', 0.02)
        
        # State Management
        self.timeframe = self.params.get('TIMEFRAME', '1h')
        logger.info(f"🤖 SpotTrader Initialized for {self.symbols}")
        logger.info(f"⏳ Timeframe: {self.timeframe}")
        logger.info(f"🔧 Strategy Params: {self.params}")

    def _load_best_params(self, db_path):
        try:
            storage = f"sqlite:///{db_path}"
            study = optuna.load_study(study_name="spot_strategy", storage=storage)
            logger.info("🏆 Best parameters loaded successfully.")
            return study.best_params
        except Exception as e:
            logger.error(f"⚠️ Failed to load params: {e}")
            logger.warning("Using default fallback parameters.")
            return {
                'TIMEFRAME': '1h',
                'ENTRY_TYPE': 'BOLLINGER', # Fallback defaults
                'ATR_PERIOD': 14,
                'STRENGTH_FILTER_PERIOD': 14,
                'EXIT_TYPE': 'ATR',
                'STOP_LOSS_TYPE': 'FIXED',
                'STOP_LOSS_PCT': 0.02,
                'RISK_PER_TRADE': 0.02
            }

    def run(self):
        logger.info("🚀 Bot Started! Waiting for next candle...")
        try:
            while True:
                try:
                    # Update Balance Cap dynamically
                    # ...
                    for symbol in self.symbols:
                        self._process_symbol(symbol)
                    
                    # Sleep to prevent API spam, check every 1 minute
                    # Display Heartbeat
                    print(".", end="", flush=True) 
                    time.sleep(60)
                    
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(f"⚠️ Network Error (Retrying in 60s): {e}")
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"🔥 Critical Error in Main Loop: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot Stopped by User (Ctrl+C). Exiting gracefully...")
            sys.exit(0)

    def _get_state_path(self):
        return os.path.join(os.path.dirname(__file__), 'trading_state.json')

    def _load_state(self):
        import json
        try:
            path = self._get_state_path()
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ Load State Error: {e}")
        return {}

    def _save_state(self, state):
        import json
        try:
            with open(self._get_state_path(), 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"⚠️ Save State Error: {e}")

    def _process_symbol(self, symbol):
        # 0. Load State
        state = self._load_state()
        symbol_state = state.get(symbol, {})
        entry_price = symbol_state.get('entry_price', 0.0)
        
        # 1. Fetch Data
        df = self.client.fetch_ohlcv(symbol, self.timeframe, limit=200)
        if df is None or len(df) < 50:
            logger.warning(f"⚠️ Insufficient data for {symbol}")
            return

        # 2. Generate Signals
        df = self.strategy.generate_signals(df)
        curr = df.iloc[-2]  # Confirmed Candle for signals
        last_price = df.iloc[-1]['close'] # Real-time price
        
        # 3. Check Position (Upbit Sync)
        pos = self.client.fetch_position(symbol)
        balance_coin = pos.get('amount', 0.0)
        current_value = balance_coin * last_price
        
        # Check if we are really in position
        in_position = current_value > 10000 
        
        # Sync State if mismatch (Manually sold or bought)
        if not in_position and entry_price > 0:
            logger.info("⚠️ Position mismatch (Empty on exchange). Clearing state.")
            entry_price = 0.0
            symbol_state = {}
            state[symbol] = symbol_state
            self._save_state(state)
            
        elif in_position and entry_price == 0:
            # We have position but no entry price? Use Current Price as approximate or avg_buy_price
            logger.info("⚠️ Position found without state. Using avg_buy_price.")
            entry_price = pos.get('avg_buy_price', last_price)
            symbol_state['entry_price'] = entry_price
            state[symbol] = symbol_state
            self._save_state(state)
        
        # 4. Trading Logic (EXIT LOGIC SKIPPED, ALREADY REPLACED ABOVE)
        
        if in_position:
            should_sell = False
            reason = ""
            
            # Indicators
            trend_dir = curr['trend_direction']
            atr_val = curr['atr']
            
            # Load Entry ATR for Consistent Exit
            entry_atr = symbol_state.get('entry_atr', atr_val) # Fallback to current if missing
            
            # Params
            sl_pct = self.params.get('STOP_LOSS_PCT', 0.02)
            atr_sl_mult = self.params.get('ATR_STOP_LOSS_MULT', 1.0)
            use_tp = self.params.get('USE_TAKE_PROFIT', False)
            tp_mult = self.params.get('TAKE_PROFIT_ATR_MULT', 2.0)
            sl_type = self.params.get('STOP_LOSS_TYPE', 'FIXED')
            exit_type = self.params.get('EXIT_TYPE', 'ATR')
            atr_mult = self.params.get('ATR_MULTIPLIER', 3.0)
            
            # --- State Management for Trailing Stop ---
            highest_high = symbol_state.get('highest_high', entry_price)
            current_high = max(curr['high'], last_price)
            if current_high > highest_high:
                highest_high = current_high
                symbol_state['highest_high'] = highest_high
                self._save_state(state) # Persist updates

            # 1. Main Trend Exit (ATR Trailing or SAR)
            if exit_type == 'ATR':
                # ATR Trailing Stop Logic using ENTRY ATR (Snapshot)
                trailing_stop = highest_high - (entry_atr * atr_mult)
                if last_price < trailing_stop:
                    should_sell = True
                    reason = f"ATR Trailing Stop (Hit {trailing_stop:,.0f})"
                    
            elif exit_type == 'PARABOLIC_SAR':
                # Parabolic SAR Exit
                p_sar = curr.get('parabolic_sar', 0)
                if p_sar > 0 and last_price < p_sar:
                    should_sell = True
                    reason = f"Parabolic SAR Exit (Hit {p_sar:,.0f})"

            # 2. Stop Loss Check
            stop_price = 0.0
            if sl_type == 'ATR':
                # Use Entry ATR for fixed SL distance
                stop_price = entry_price - (entry_atr * atr_sl_mult)  
            else:
                stop_price = entry_price * (1 - sl_pct)
            
            if not should_sell and last_price < stop_price:
                should_sell = True
                reason = f"Safety Stop Loss (Hit {stop_price:,.0f})"

            # 3. Trend Reversal
            if not should_sell and trend_dir == -1:
                should_sell = True
                reason = "Trend Reversal (Down)"
                
            # 4. Take Profit Check
            if not should_sell and use_tp:
                # Use Entry ATR for TP distance
                target_price = entry_price + (entry_atr * tp_mult)
                if last_price > target_price:
                    should_sell = True
                    reason = f"Take Profit (Hit {target_price:,.0f})"
            
            if should_sell:
                logger.info(f"🔴 SELL SIGNAL for {symbol} | Reason: {reason}")
                res = self.client.place_order(symbol, 'sell', amount=balance_coin)
                if res and 'uuid' in res:
                    self._send_telegram(f"🔻 SELL {symbol}\nPrice: {last_price:,.0f}\nReason: {reason}")
                    state[symbol] = {}
                    self._save_state(state)
                else:
                    logger.error(f"❌ Sell Order Failed: {res}")

        # B. Entry Logic
        else:
            entry_upper = curr['entry_upper']
            trend_dir = curr['trend_direction']
            strength = curr['strength_filter']
            vol_ratio = curr.get('volume_ratio', 1.0)
            
            use_vol = self.params.get('USE_VOLUME_FILTER', False)
            vol_thresh = self.params.get('VOLUME_THRESHOLD_MULT', 1.0)
            
            is_uptrend = trend_dir == 1
            breakout = last_price > entry_upper
            strong_momentum = strength == 1
            vol_ok = (not use_vol) or (vol_ratio >= vol_thresh)
            
            if is_uptrend and breakout and strong_momentum and vol_ok:
                total_krw, free_krw = self.client.fetch_balance()
                
                num_symbols = len(self.symbols)
                target_allocation = total_krw / num_symbols if num_symbols > 0 else 0
                
                max_cap = 100_000_000.0
                invest_amount = min(free_krw * 0.99, target_allocation, max_cap)
                
                if invest_amount > 5000:
                    logger.info(f"🟢 BUY SIGNAL for {symbol} | Price: {last_price:,.0f} | Invest: {invest_amount:,.0f} KRW")
                    res = self.client.place_order(symbol, 'buy', price=invest_amount)
                    
                    if res and 'uuid' in res:
                        self._send_telegram(f"🚀 BUY {symbol}\nPrice: {last_price:,.0f}\nAmt: {invest_amount:,.0f}")
                        
                        # Save State immediately including ATR Snapshot
                        state[symbol] = {
                            'entry_price': last_price,
                            'entry_time': datetime.now().isoformat(),
                            'amount': invest_amount,
                            'entry_atr': curr['atr'], # SAVE ATR SNAPSHOT
                            'highest_high': last_price # Init Highest High
                        }
                        self._save_state(state)
                    else:
                        logger.error(f"❌ Buy Order Failed: {res}")
                else:
                    logger.warning(f"⚠️ Insufficient funds to buy {symbol}. Free: {free_krw:,.0f}, Target: {target_allocation:,.0f}")

    def _send_telegram(self, msg):
        # Placeholder for notification
        logger.info(f"msg: {msg}")

if __name__ == "__main__":
    # Settings
    # Selected ETH based on higher total return (5418%) compared to BTC (828%) in verification
    SYMBOLS = ["KRW-ETH"] 
    DB_PATH = "spot_strategy.db"
    
    # Run
    bot = SpotTrader(SYMBOLS, DB_PATH)
    bot.run()
