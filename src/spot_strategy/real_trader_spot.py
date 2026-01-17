
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
                # ... Add minimal defaults if needed
            }

    def run(self):
        logger.info("🚀 Bot Started! Waiting for next candle...")
        try:
            while True:
                try:
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
        df = self.client.fetch_ohlcv(symbol, self.timeframe, count=200)
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
        
        # 4. Trading Logic
        
        # A. Exit Logic
        if in_position:
            should_sell = False
            reason = ""
            
            # Indicators
            trend_dir = curr['trend_direction']
            atr_val = curr['atr']
            
            # Params
            sl_pct = self.params.get('STOP_LOSS_PCT', 0.02)
            atr_sl_mult = self.params.get('ATR_STOP_LOSS_MULT', 1.0)
            use_tp = self.params.get('USE_TAKE_PROFIT', False)
            tp_mult = self.params.get('TAKE_PROFIT_ATR_MULT', 2.0)
            sl_type = self.params.get('STOP_LOSS_TYPE', 'FIXED')

            # 1. Trend Reversal
            if trend_dir == -1:
                should_sell = True
                reason = "Trend Reversal (Down)"
                
            # 2. Stop Loss Check
            stop_price = 0.0
            if sl_type == 'ATR':
                stop_price = entry_price - (atr_val * atr_sl_mult)
            else:
                stop_price = entry_price * (1 - sl_pct)
            
            if last_price < stop_price:
                should_sell = True
                reason = f"Stop Loss (Hit {stop_price:,.0f})"
                
            # 3. Take Profit Check
            if use_tp:
                target_price = entry_price + (atr_val * tp_mult)
                if last_price > target_price:
                    should_sell = True
                    reason = f"Take Profit (Hit {target_price:,.0f})"
            
            if should_sell:
                logger.info(f"🔴 SELL SIGNAL for {symbol} | Reason: {reason}")
                res = self.client.place_order(symbol, 'sell', amount=balance_coin)
                if res and 'uuid' in res:
                    self._send_telegram(f"🔻 SELL {symbol}\nPrice: {last_price:,.0f}\nReason: {reason}")
                    # Clear State
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
                
                # Dynamic Position Sizing
                # If we track N symbols, we allocate 1/N of Total Capital to each.
                # Or we can just use free_krw if we want to be aggressive but sequentially limited.
                # Safer: Target Allocation = Total Equity / len(self.symbols)
                
                num_symbols = len(self.symbols)
                target_allocation = total_krw / num_symbols if num_symbols > 0 else 0
                
                # Check if we already have exposure (although 'in_position' check above should handle strictly)
                # We use free_krw but cap it at target_allocation
                invest_amount = min(free_krw * 0.99, target_allocation)
                
                if invest_amount > 5000:
                    logger.info(f"🟢 BUY SIGNAL for {symbol} | Price: {last_price:,.0f} | Invest: {invest_amount:,.0f} KRW")
                    res = self.client.place_order(symbol, 'buy', price=invest_amount)
                    
                    if res and 'uuid' in res:
                        self._send_telegram(f"🚀 BUY {symbol}\nPrice: {last_price:,.0f}\nAmt: {invest_amount:,.0f}")
                        
                        # Save State immediately
                        state[symbol] = {
                            'entry_price': last_price,
                            'entry_time': datetime.now().isoformat(),
                            'amount': invest_amount # approx
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
