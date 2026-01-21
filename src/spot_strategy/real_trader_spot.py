"""
RealTrader Spot - 24시간 자동 현물 트레이딩 봇 (Production Grade - Upbit)
==========================================================================
P0/P1 개선사항 적용:
- 거래 기록 DB 영속화
- API 재시도 데코레이터 (tenacity)
- Health Check 메커니즘
- Graceful Shutdown (SIGTERM)
- 중복 코드 제거 (유틸 함수)
- 매직 넘버 → settings 이동
- 캔들 마감 동기화
- Structured JSON 로깅
- Oracle Cloud 최적화 (옵션)
"""

import os
import sys
import time
import signal
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# tenacity for retry logic


# Project Root Setup
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    sys.path.append(os.getcwd())

from config.settings import (
    UPBIT_ACCESS_KEY,
    UPBIT_SECRET_KEY,
    LOG_DIR,
    SPOT_STRATEGY_DB,
    TRADE_HISTORY_DB,
    SPOT_HEARTBEAT_FILE,
    SPOT_STATE_FILE,
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MIN,
    API_RETRY_WAIT_MAX,
    MIN_POSITION_VALUE_KRW,
    MIN_ORDER_VALUE_KRW,
    MAX_INVEST_CAP_KRW,
    SPOT_LOOP_INTERVAL_SECONDS,
    SPOT_SYMBOL_DELAY_SECONDS,
    ERROR_SLEEP_SECONDS,
    CANDLE_SYNC_OFFSET_SECONDS,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    SPOT_TARGET_SYMBOLS,
    SPOT_OPTUNA_STUDY_NAME,
    SPOT_ALLOCATION_WEIGHTS,
    MAX_TOTAL_BALANCE_KRW,
)

# 재사용 모듈 (Futures에서 구현한 공통 컴포넌트)
# 공통 유틸리티 및 컴포넌트
from src.common.utils import setup_logger, api_retry
from src.common.components import (
    TradeHistoryDB, 
    HealthCheckManager, 
    calculate_candle_wait_time
)

# Upbit 클라이언트
from src.spot_strategy.upbit_client import UpbitClient
from src.strategy.strategies import UltimateStrategy

# Oracle Cloud 최적화 (선택적)
try:
    from src.common.cloud_optimizer import CloudOptimizer
    CLOUD_OPTIMIZER_AVAILABLE = True
except ImportError:
    CLOUD_OPTIMIZER_AVAILABLE = False

# 로거 설정
logger = setup_logger("RealTraderSpot")


# ============================================================
# State Manager (JSON 파일 기반 - Upbit 현물 특성)
# ============================================================
class StateManager:
    """거래 상태 관리 (진입가, ATR 스냅샷 등)"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """상태 파일이 없으면 생성"""
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._save({})

    def _load(self) -> dict:
        """상태 로드"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ State load error: {e}")
            return {}

    def _save(self, state: dict):
        """상태 저장"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"⚠️ State save error: {e}")

    def get_symbol_state(self, symbol: str) -> dict:
        """특정 심볼의 상태 조회"""
        state = self._load()
        return state.get(symbol, {})

    def update_symbol_state(self, symbol: str, data: dict):
        """특정 심볼의 상태 업데이트"""
        state = self._load()
        if symbol not in state:
            state[symbol] = {}
        state[symbol].update(data)
        self._save(state)

    def clear_symbol_state(self, symbol: str):
        """특정 심볼의 상태 초기화"""
        state = self._load()
        if symbol in state:
            state[symbol] = {}
            self._save(state)


# ============================================================
# Main Trader Class
# ============================================================
class RealTraderSpot:
    """Production-grade 현물 트레이딩 봇 (Upbit)"""

    def __init__(
        self,
        db_path: str = None,
        enable_oracle_optimization: bool = False
    ):
        self.client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.db_path = db_path or str(SPOT_STRATEGY_DB)
        self.symbols = SPOT_TARGET_SYMBOLS.copy()

        # 신규 컴포넌트
        self.trade_db = TradeHistoryDB(TRADE_HISTORY_DB)
        self.health_manager = HealthCheckManager(SPOT_HEARTBEAT_FILE)
        self.state_manager = StateManager(SPOT_STATE_FILE)

        # 클라우드 최적화 (옵션)
        self.cloud_optimizer = None
        if enable_oracle_optimization and CLOUD_OPTIMIZER_AVAILABLE:
            self.cloud_optimizer = CloudOptimizer()
            logger.info("☁️ Cloud optimization enabled")

        # Shutdown 플래그
        self._shutdown_requested = False

        # Signal handlers 등록
        self._setup_signal_handlers()

        # 전략 로드
        self.params_map: Dict[str, dict] = {}
        self.strategies: Dict[str, UltimateStrategy] = {}
        self.load_strategies_from_db()

    def _setup_signal_handlers(self):
        """Graceful Shutdown 시그널 핸들러 등록"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def load_strategies_from_db(self):
        """Optuna DB에서 최적화된 파라미터 로드"""
        logger.info(f"📂 Loading strategies from {self.db_path}...")

        if not os.path.exists(self.db_path):
            logger.warning(f"⚠️ DB file not found: {self.db_path}, using defaults")
            self._use_default_params()
            return

        try:
            import optuna
            storage = f"sqlite:///{self.db_path}"
            study = optuna.load_study(study_name=SPOT_OPTUNA_STUDY_NAME, storage=storage)
            logger.info(f"✅ Loaded Study: '{SPOT_OPTUNA_STUDY_NAME}' (Score: {study.best_value:.4f})")

            best_params = study.best_params

            for symbol in self.symbols:
                self.params_map[symbol] = best_params.copy()
                strategy_name = f"RealSpot_{symbol.replace('KRW-', '')}"
                self.strategies[symbol] = UltimateStrategy(strategy_name, best_params)
                logger.info(f"🔹 Strategy initialized: {symbol} | TF: {best_params.get('TIMEFRAME')}")

        except Exception as e:
            logger.error(f"❌ Failed to load strategies: {e}")
            logger.warning("Using default fallback parameters.")
            self._use_default_params()

    def _use_default_params(self):
        """Fallback 기본 파라미터 사용"""
        default_params = {
            'TIMEFRAME': '1h',
            'ENTRY_TYPE': 'BOLLINGER',
            'ATR_PERIOD': 14,
            'STRENGTH_FILTER_PERIOD': 14,
            'EXIT_TYPE': 'ATR',
            'STOP_LOSS_TYPE': 'FIXED',
            'STOP_LOSS_PCT': 0.02,
            'RISK_PER_TRADE': 0.02,
            'ATR_MULTIPLIER': 3.0,
            'ATR_STOP_LOSS_MULT': 1.0,
            'USE_TAKE_PROFIT': False,
            'TAKE_PROFIT_ATR_MULT': 2.0,
        }

        for symbol in self.symbols:
            self.params_map[symbol] = default_params.copy()
            strategy_name = f"RealSpot_{symbol.replace('KRW-', '')}"
            self.strategies[symbol] = UltimateStrategy(strategy_name, default_params)
            logger.info(f"🔹 Default strategy for {symbol}")

    @api_retry
    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, limit: int):
        """안전한 OHLCV 조회 (재시도 적용)"""
        return self.client.fetch_ohlcv(symbol, timeframe, limit=limit)

    @api_retry
    def _fetch_position_safe(self, symbol: str) -> dict:
        """안전한 포지션 조회 (재시도 적용)"""
        return self.client.fetch_position(symbol)

    @api_retry
    def _fetch_balance_safe(self) -> tuple:
        """안전한 잔고 조회 (재시도 적용)"""
        return self.client.fetch_balance()

    @api_retry
    def _place_order_safe(self, symbol: str, side: str, **kwargs):
        """안전한 주문 실행 (재시도 적용)"""
        return self.client.place_order(symbol, side, **kwargs)

    def initialize(self):
        """초기화: 잔고 확인"""
        logger.info("🤖 RealTrader Spot Bot Initializing...")

        try:
            total_krw, free_krw = self._fetch_balance_safe()
            logger.info(f"💰 Account Balance: Total {total_krw:,.0f} KRW | Free {free_krw:,.0f} KRW")

            if free_krw < MIN_ORDER_VALUE_KRW:
                logger.warning(f"⚠️ Warning: Low balance (< {MIN_ORDER_VALUE_KRW:,} KRW)!")
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")

        # 초기 헬스체크
        self.health_manager.update_heartbeat(status="initialized")

        logger.info("🚀 Initialization Complete. Bot is Running...")

    def execute_logic(self, symbol: str):
        """핵심 매매 로직 실행"""
        try:
            params = self.params_map[symbol]
            strategy = self.strategies[symbol]
            timeframe = params.get('TIMEFRAME', '1h')

            # 1. 데이터 조회 (충분한 지표 계산을 위해 600개 이상 요청)
            df = self._fetch_ohlcv_safe(symbol, timeframe, limit=600)

            # 전략 지표(EMA 200, Ichimoku 등) 계산을 위해 최소 200개 이상 필요
            if df is None or len(df) < 200:
                logger.warning(
                    f"⚠️ Insufficient data for {symbol}: "
                    f"Got {len(df) if df is not None else 0}, need min 200 for indicators."
                )
                return

            # 2. 지표 계산
            df = strategy.generate_signals(df)

            # 3. 신호 확인 (-2: 확정된 마지막 봉, -1: 현재가)
            confirmed_candle = df.iloc[-2]
            last_price = df.iloc[-1]['close']

            # 4. 현재 포지션 조회
            pos = self._fetch_position_safe(symbol)
            balance_coin = pos.get('amount', 0.0)
            current_value = balance_coin * last_price

            in_position = current_value > MIN_POSITION_VALUE_KRW

            # 5. 상태 동기화
            state = self.state_manager.get_symbol_state(symbol)
            entry_price = state.get('entry_price', 0.0)

            # 상태 불일치 처리
            if not in_position and entry_price > 0:
                logger.info(f"⚠️ Position mismatch for {symbol} (Empty on exchange). Clearing state.")
                self.state_manager.clear_symbol_state(symbol)
                entry_price = 0.0
                state = {}

            elif in_position and entry_price == 0:
                # 포지션은 있는데 봇 상태가 없는 경우 (재시작 등)
                exchange_avg_price = pos.get('entryPrice', 0.0)
                
                if exchange_avg_price > 0:
                    logger.info(
                        f"⚠️ Position found without state for {symbol}. "
                        f"Recovering from exchange avg_buy_price: {exchange_avg_price:,.0f} KRW"
                    )
                    entry_price = exchange_avg_price
                    state['entry_price'] = entry_price
                    state['entry_atr'] = confirmed_candle.get('atr', 0.0)
                    state['highest_high'] = max(last_price, exchange_avg_price)
                    state['invest_amount'] = balance_coin * exchange_avg_price  # 원금 추정
                    self.state_manager.update_symbol_state(symbol, state)
                else:
                    logger.error(
                        f"🚨 CRITICAL: Position exists for {symbol} but CANNOT determine "
                        f"entry price! Amount: {balance_coin}, Value: {current_value:,.0f} KRW. "
                        f"Skipping all actions until manual verification."
                    )
                    return  # 안전을 위해 해당 심볼은 이번 루프 스킵

            # --- EXIT LOGIC ---
            if in_position:
                self._check_exit(
                    symbol, balance_coin, last_price, params,
                    confirmed_candle, state
                )

            # --- ENTRY LOGIC ---
            else:
                self._check_entry(
                    symbol, last_price, params,
                    confirmed_candle
                )

        except Exception as e:
            logger.error(f"🚨 Error executing logic for {symbol}: {e}")
            self.health_manager.record_error(e)

    def _check_exit(
        self,
        symbol: str,
        balance_coin: float,
        last_price: float,
        params: dict,
        candle: dict,
        state: dict
    ):
        """청산 로직"""
        try:
            should_sell = False
            reason = ""

            # 파라미터
            entry_price = state.get('entry_price', 0.0)
            # [Fix] Fallback to current candle ATR if state is missing (Critical for Safety)
            entry_atr = state.get('entry_atr', 0.0)
            if entry_atr == 0:
                entry_atr = candle.get('atr', 0.0)
                if entry_atr > 0:
                    logger.warning(f"⚠️ Recovered entry_atr from current candle for {symbol}: {entry_atr}")

            highest_high = state.get('highest_high', entry_price)

            trend_dir = candle.get('trend_direction', 0)
            current_high = max(candle.get('high', last_price), last_price)

            # Highest High 업데이트
            if current_high > highest_high:
                highest_high = current_high
                state['highest_high'] = highest_high
                self.state_manager.update_symbol_state(symbol, state)

            exit_type = params.get('EXIT_TYPE', 'ATR')
            atr_mult = params.get('ATR_MULTIPLIER', 3.0)
            sl_type = params.get('STOP_LOSS_TYPE', 'FIXED')
            sl_pct = params.get('STOP_LOSS_PCT', 0.02)
            atr_sl_mult = params.get('ATR_STOP_LOSS_MULT', 1.0)
            use_tp = params.get('USE_TAKE_PROFIT', False)
            tp_mult = params.get('TAKE_PROFIT_ATR_MULT', 2.0)

            # 1. Main Exit (ATR Trailing or SAR)
            if exit_type == 'ATR':
                trailing_stop = highest_high - (entry_atr * atr_mult)
                if last_price < trailing_stop:
                    should_sell = True
                    reason = f"ATR Trailing Stop ({trailing_stop:,.0f})"

            elif exit_type == 'PARABOLIC_SAR':
                p_sar = candle.get('parabolic_sar', 0)
                if p_sar > 0 and last_price < p_sar:
                    should_sell = True
                    reason = f"Parabolic SAR Exit ({p_sar:,.0f})"

            # 2. Stop Loss
            if not should_sell:
                if sl_type == 'ATR':
                    stop_price = entry_price - (entry_atr * atr_sl_mult)
                else:
                    stop_price = entry_price * (1 - sl_pct)

                if last_price < stop_price:
                    should_sell = True
                    reason = f"Stop Loss ({stop_price:,.0f})"

            # 3. Trend Reversal
            if not should_sell and trend_dir == -1:
                should_sell = True
                reason = "Trend Reversal"

            # 4. Take Profit
            if not should_sell and use_tp:
                target_price = entry_price + (entry_atr * tp_mult)
                if last_price > target_price:
                    should_sell = True
                    reason = f"Take Profit ({target_price:,.0f})"

            if should_sell:
                pnl = (last_price - entry_price) * balance_coin
                pnl_pct = ((last_price / entry_price) - 1) * 100 if entry_price > 0 else 0

                logger.info(
                    f"🛑 EXIT {symbol} | Price: {last_price:,.0f} | "
                    f"PnL: {pnl:,.0f} KRW ({pnl_pct:.2f}%) | Reason: {reason}"
                )

                res = self._place_order_safe(symbol, 'sell', amount=balance_coin)

                if res and 'uuid' in res:
                    self.trade_db.record_trade(
                        symbol=symbol,
                        side='LONG',
                        action='EXIT',
                        quantity=balance_coin,
                        price=last_price,
                        entry_price=entry_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        reason=reason
                    )
                    self.state_manager.clear_symbol_state(symbol)
                else:
                    logger.error(f"❌ Sell order failed: {res}")

        except Exception as e:
            logger.error(f"⚠️ Error in _check_exit: {e}")
            self.health_manager.record_error(e)

    def _calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        params: dict
    ) -> float:
        """포지션 사이즈 계산 (성과 기반 가중치, 리스크 관리 적용)"""
        try:
            total_krw, free_krw = self._fetch_balance_safe()
            
            # 1. 포트폴리오 총 가치 추정 (현금 + 투자원금 합계)
            # Upbit API는 total_krw에 코인 평가금을 포함하지 않으므로 직접 계산 필요
            current_invested_total = 0
            this_symbol_invested = 0
            
            state_map = self.state_manager._load()
            for s in self.symbols:
                st = state_map.get(s, {})
                invest_amt = st.get('invest_amount', 0)
                current_invested_total += invest_amt
                
                if s == symbol:
                    this_symbol_invested = invest_amt
            
            # [CRITICAL UPDATE] Single Entry Enforcement (Match Backtest)
            if this_symbol_invested > MIN_POSITION_VALUE_KRW:
                logger.info(f"⏭️ Skipping entry for {symbol}: Position already exists ({this_symbol_invested:,.0f} KRW).")
                return 0


            # 총 자산 (추정치) = KRW 현금 잔고 + 현재 투자 중인 원금 총액
            # (수익/손실은 무시하고 원금 기준으로 보수적 접근)
            estimated_total_equity = total_krw + current_invested_total
            
            # 2. 할당량 계산 (성과 기반 가중치 적용)
            # 예: ETH 70%, BTC 30%
            default_weight = 1.0 / len(self.symbols) if self.symbols else 0.5
            weight = SPOT_ALLOCATION_WEIGHTS.get(symbol, default_weight)
            target_amount = estimated_total_equity * weight
            
            # 3. 추가 매수 가능액 계산
            # 목표 금액 - 현재 투자 금액 (이미 진입해 있다면 추가 진입은 자제, 혹은 물타기?)
            # 여기서는 '신규 진입' 또는 '불타기' 관점
            buy_amount = target_amount - this_symbol_invested
            
            # 제약 조건 적용
            buy_amount = min(buy_amount, free_krw)            # 가용 현금 한도
            buy_amount = min(buy_amount, MAX_INVEST_CAP_KRW)  # 심볼당 최대 한도
            
            # 너무 작은 주문 방지
            if buy_amount < MIN_ORDER_VALUE_KRW:
                return 0
                
            logger.info(
                f"🧮 Sizing {symbol} (Weight {weight*100:.0f}%): "
                f"Equity ≈ {estimated_total_equity:,.0f} KRW | "
                f"Target {target_amount:,.0f} | Buy {buy_amount:,.0f}"
            )
            
            return buy_amount
            
        except Exception as e:
            logger.error(f"Sizing Error: {e}")
            return 0

    def _check_entry(
        self,
        symbol: str,
        last_price: float,
        params: dict,
        candle: dict
    ):
        """진입 로직"""
        try:
            entry_upper = candle.get('entry_upper', 0)
            trend_dir = candle.get('trend_direction', 0)
            strength = candle.get('strength_filter', 0)
            vol_ratio = candle.get('volume_ratio', 1.0)
            atr = candle.get('atr', 0.0)

            use_vol = params.get('USE_VOLUME_FILTER', False)
            vol_thresh = params.get('VOLUME_THRESHOLD_MULT', 1.0)

            is_uptrend = trend_dir == 1
            breakout = last_price > entry_upper
            strong_momentum = strength == 1
            vol_ok = (not use_vol) or (vol_ratio >= vol_thresh)

            if is_uptrend and breakout and strong_momentum and vol_ok:
                # 가중치 기반 동적 사이징 계산
                invest_amount = self._calculate_position_size(symbol, last_price, params)

                if invest_amount > MIN_ORDER_VALUE_KRW:
                    logger.info(
                        f"🟢 ENTRY {symbol} | Price: {last_price:,.0f} | "
                        f"Invest: {invest_amount:,.0f} KRW"
                    )

                    res = self._place_order_safe(symbol, 'buy', price=invest_amount)

                    if res and 'uuid' in res:
                        self.trade_db.record_trade(
                            symbol=symbol,
                            side='LONG',
                            action='ENTRY',
                            quantity=invest_amount / last_price,
                            price=last_price,
                            reason=f"Price > Upper ({entry_upper:,.0f})"
                        )

                        # 상태 저장
                        self.state_manager.update_symbol_state(symbol, {
                            'entry_price': last_price,
                            'entry_time': datetime.now().isoformat(),
                            'entry_atr': atr,
                            'highest_high': last_price,
                            'invest_amount': invest_amount
                        })
                    else:
                        logger.error(f"❌ Buy order failed: {res}")
                else:
                    logger.warning(
                        f"⚠️ Insufficient funds or allocation full for {symbol}. "
                    )

        except Exception as e:
            logger.error(f"⚠️ Error in _check_entry: {e}")
            self.health_manager.record_error(e)

    def _get_current_positions(self) -> dict:
        """현재 포지션 상태 조회 (헬스체크용)"""
        positions = {}
        for symbol in self.symbols:
            try:
                state = self.state_manager.get_symbol_state(symbol)
                if state.get('entry_price', 0) > 0:
                    pos = self._fetch_position_safe(symbol)
                    positions[symbol] = {
                        'entry_price': state.get('entry_price'),
                        'amount': pos.get('amount', 0),
                        'unrealized_pnl': (
                            (pos.get('amount', 0) * pos.get('current_price', 0)) -
                            (state.get('invest_amount', 0))
                        ) if pos.get('amount', 0) > 0 else 0
                    }
            except Exception:
                pass
        return positions

    def run_forever(self):
        """메인 무한 루프 (Graceful Shutdown 지원)"""
        try:
            self.initialize()
        except Exception as e:
            logger.error(f"🚨 Initialization failed: {e}")
            self.health_manager.update_heartbeat(status="init_failed")
            raise

        logger.info("⏳ Waiting for next candle close...")

        while not self._shutdown_requested:
            try:
                # 각 심볼 처리
                for symbol in self.symbols:
                    if self._shutdown_requested:
                        break
                    self.execute_logic(symbol)
                    time.sleep(SPOT_SYMBOL_DELAY_SECONDS)

                # 헬스체크 업데이트
                positions = self._get_current_positions()
                self.health_manager.update_heartbeat(
                    status="running",
                    positions=positions
                )

                # 클라우드 최적화 실행
                if self.cloud_optimizer:
                    # 1. 시간 동기화 검증 (거래소 API 필수)
                    if not self.cloud_optimizer.check_time_sync_ntp():
                        logger.error("⏰ Time drift detected! Bot may encounter API errors.")
                    
                    # 2. 리소스 모니터링 (10분마다)
                    if self.health_manager.loop_count % 20 == 0:
                        self.cloud_optimizer.log_resource_usage()
                    
                    # 3. DB 정리 (24시간마다, 90일 이상 오래된 거래 삭제)
                    if self.health_manager.loop_count % 2880 == 0:
                        self.cloud_optimizer.cleanup_db_old_records(
                            TRADE_HISTORY_DB, 
                            days_to_keep=90
                        )

                    # 5. 명시적 GC (2시간마다)
                    if self.health_manager.loop_count % 120 == 0:
                        self.cloud_optimizer.force_gc()

                # 캔들 동기화 대기
                if self.symbols and self.params_map:
                    tf = self.params_map[self.symbols[0]].get('TIMEFRAME', '1h')
                    wait_time = calculate_candle_wait_time(tf)

                    # 최소 대기 시간 적용
                    if wait_time < SPOT_LOOP_INTERVAL_SECONDS:
                        wait_time = SPOT_LOOP_INTERVAL_SECONDS

                    # 최대 대기 시간 제한 (1시간)
                    wait_time = min(wait_time, 3600)

                    logger.info(f"💤 Next execution in {wait_time}s...")

                    # Shutdown 체크하면서 대기
                    for _ in range(int(wait_time)):
                        if self._shutdown_requested:
                            break
                        time.sleep(1)
                else:
                    time.sleep(SPOT_LOOP_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"🚨 Critical Error in Main Loop: {e}")
                self.health_manager.record_error(e)
                self.health_manager.update_heartbeat(status="error")
                time.sleep(ERROR_SLEEP_SECONDS)

        # Graceful Shutdown 처리
        self._shutdown()

    def _shutdown(self):
        """Graceful Shutdown 처리"""
        logger.info("🛑 Shutting down gracefully...")

        # 현재 포지션 상태 기록
        positions = self._get_current_positions()
        if positions:
            logger.warning(f"⚠️ Open positions at shutdown: {positions}")

        self.health_manager.update_heartbeat(
            status="stopped",
            positions=positions,
            extra={"shutdown_time": datetime.utcnow().isoformat()}
        )

        logger.info("✅ Shutdown complete.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 RealTrader Spot - Production Grade Bot (Upbit)")
    logger.info("=" * 60)

    # Oracle Cloud 환경 변수로 활성화 결정 (기본값: True)
    enable_oracle_opt = os.getenv("ENABLE_ORACLE_OPTIMIZATION", "true").lower() == "true"

    bot = RealTraderSpot(enable_oracle_optimization=enable_oracle_opt)
    bot.run_forever()
