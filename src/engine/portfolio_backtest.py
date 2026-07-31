from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger
from src.core.types import CostModel, PortfolioSpec, StrategySpec
from src.data.loader import DataIntegrityError
from src.data.portfolio_universe import select_liquid_universe
from src.engine.backtest import BacktestResult, _align_funding_rates, calculate_position_size
from src.strategy.donchian import generate_signals

_logger = setup_logger("PortfolioBacktest")

# Contract invariant: sum(initial_risk) over open positions <= 2.5% of pre-entry
# total equity. This is five times the unchanged 0.5% per-position risk and is
# never a fitted leverage multiplier.
MAX_TOTAL_INITIAL_RISK = 0.025

_TRADE_COLUMNS = (
    "symbol",
    "entry_bar",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "qty",
    "reason",
    "pnl",
    "return_pct",
    "funding_pnl",
    "initial_risk",
    "portfolio_equity_before_entry",
)


@dataclass
class _Position:
    symbol: str
    qty: float
    entry_price: float
    stop_price: float
    entry_bar: int
    initial_risk: float
    equity_at_entry: float
    funding_pnl: float = 0.0


def _prepare_symbol_frame(
    symbol: str,
    df: pd.DataFrame,
    funding: pd.Series | None,
    strategy_spec: StrategySpec,
    signal_delay_bars: int,
) -> tuple[pd.DataFrame, bool, np.ndarray]:
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise DataIntegrityError(f"{symbol} df missing columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise DataIntegrityError(f"{symbol} index must be a tz-aware DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError(f"{symbol} index must be monotonic increasing")
    if df.index.has_duplicates:
        raise DataIntegrityError(f"{symbol} index must not contain duplicates")

    feat = generate_signals(df, strategy_spec)
    if signal_delay_bars > 0:
        feat["entry_signal"] = (
            feat["entry_signal"].shift(signal_delay_bars).fillna(False).astype(bool)
        )
        feat["exit_lower"] = feat["exit_lower"].shift(signal_delay_bars)

    if funding is None or len(funding) == 0:
        funding_eligible = False
        bar_funding = np.zeros(len(feat), dtype=np.float64)
    else:
        funding_eligible = True
        bar_funding = _align_funding_rates(funding, feat.index)
    return feat, funding_eligible, bar_funding


def run_portfolio_backtest(
    frames: Mapping[str, pd.DataFrame],
    funding_rates: Mapping[str, pd.Series],
    strategy_spec: StrategySpec,
    portfolio_spec: PortfolioSpec,
    costs: CostModel,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    """Execute one shared-cash long-only portfolio ledger across ``frames``.

    Signals reuse the frozen v1 Donchian(20)/EMA(200)/ATR(14, 2R) semantics once
    per symbol. At each UTC day boundary the liquid universe is reselected from
    trailing quote volume; entries fill no earlier than the next bar open, are
    ranked by causal ATR-normalized breakout, and are capped by slots, leverage,
    and the 2.5%-of-total-equity aggregate initial-risk invariant. Funding is
    accrued per open long on its own symbol. Inputs are never mutated.
    """
    if not frames:
        raise ValueError("frames must contain at least one symbol")
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")

    prepared: dict[str, tuple[pd.DataFrame, bool, np.ndarray]] = {}
    for symbol, df in frames.items():
        funding = funding_rates.get(symbol) if funding_rates is not None else None
        prepared[symbol] = _prepare_symbol_frame(
            symbol, df, funding, strategy_spec, signal_delay_bars,
        )

    grid = pd.DatetimeIndex(
        sorted(set.intersection(*(set(feat.index) for feat, _, _ in prepared.values())))
    )
    if len(grid) < 2:
        raise DataIntegrityError("frames share fewer than 2 common bars")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    funding_eligible: dict[str, bool] = {}
    for symbol, (feat, is_funding_eligible, bar_funding) in prepared.items():
        feat_g = feat.loc[grid]
        arrays[symbol] = {
            "open": feat_g["open"].to_numpy(dtype=np.float64),
            "high": feat_g["high"].to_numpy(dtype=np.float64),
            "low": feat_g["low"].to_numpy(dtype=np.float64),
            "close": feat_g["close"].to_numpy(dtype=np.float64),
            "upper": feat_g["upper"].to_numpy(dtype=np.float64),
            "exit_lower": feat_g["exit_lower"].to_numpy(dtype=np.float64),
            "atr": feat_g["atr"].to_numpy(dtype=np.float64),
            "entry_signal": feat_g["entry_signal"].to_numpy(dtype=bool),
            "bar_funding": pd.Series(bar_funding, index=feat.index)
            .loc[grid]
            .to_numpy(dtype=np.float64),
        }
        funding_eligible[symbol] = is_funding_eligible

    warmup = max(strategy_spec.ema_period, strategy_spec.entry_period, strategy_spec.atr_period) + 1 + signal_delay_bars

    equity_arr = np.full(len(grid), np.nan, dtype=np.float64)
    cash = initial_equity
    equity_prev = initial_equity
    positions: dict[str, _Position] = {}
    open_risk = 0.0
    touched: set[str] = set()
    pending_entry = dict.fromkeys(prepared, False)
    pending_strength: dict[str, float] = dict.fromkeys(prepared, -np.inf)
    pending_exit = dict.fromkeys(prepared, False)
    universe: set[str] = set()
    trades: list[dict[str, object]] = []

    def record_exit(
        symbol: str,
        pos: _Position,
        t: int,
        exit_price: float,
        reason: str,
    ) -> None:
        nonlocal cash, open_risk
        qty = pos.qty
        exit_fee = costs.fee_rate * abs(qty * exit_price)
        cash += qty * exit_price - exit_fee
        pnl = qty * (exit_price - pos.entry_price) - exit_fee + pos.funding_pnl
        return_pct = pnl / pos.equity_at_entry if pos.equity_at_entry != 0 else 0.0
        trades.append({
            "symbol": symbol,
            "entry_bar": pos.entry_bar,
            "entry_time": grid[pos.entry_bar],
            "exit_time": grid[t],
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "reason": reason,
            "pnl": pnl,
            "return_pct": return_pct,
            "funding_pnl": pos.funding_pnl,
            "initial_risk": pos.initial_risk,
            "portfolio_equity_before_entry": pos.equity_at_entry,
        })
        _logger.info(
            "bar=%d symbol=%s reason=%s entry=%.4f exit=%.4f qty=%.6f pnl=%.4f",
            t, symbol, reason, pos.entry_price, exit_price, qty, pnl,
            extra={"tag": "ALGO"},
        )
        open_risk -= pos.initial_risk
        del positions[symbol]
        touched.add(symbol)

    for t in range(len(grid)):
        ts = grid[t]
        if t == 0 or grid[t - 1].date() != ts.date():
            universe = set(select_liquid_universe(frames, as_of=ts, spec=portfolio_spec))

        for symbol in list(positions):
            pos = positions[symbol]
            arr = arrays[symbol]
            rate = float(arr["bar_funding"][t])
            if rate != 0.0:
                funding_pnl = -pos.qty * arr["open"][t] * rate
                cash += funding_pnl
                pos.funding_pnl += funding_pnl

        for symbol in sorted(positions):
            pos = positions[symbol]
            arr = arrays[symbol]
            if arr["low"][t] <= pos.stop_price:
                exit_price = min(arr["open"][t], pos.stop_price) * (1 - costs.slippage_rate)
                record_exit(symbol, pos, t, exit_price, "stop")
            elif pending_exit[symbol]:
                exit_price = arr["open"][t] * (1 - costs.slippage_rate)
                record_exit(symbol, pos, t, exit_price, "channel")

        if len(positions) < portfolio_spec.max_positions and t >= warmup:
            candidates: list[tuple[float, str]] = []
            for symbol in universe:
                if symbol not in arrays or symbol in positions or symbol in touched:
                    continue
                if not pending_entry[symbol] or not funding_eligible[symbol]:
                    continue
                strength = pending_strength[symbol]
                if not np.isfinite(strength):
                    continue
                candidates.append((strength, symbol))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _strength, symbol in candidates:
                if len(positions) >= portfolio_spec.max_positions:
                    break
                arr = arrays[symbol]
                stop_distance = strategy_spec.stop_atr_mult * arr["atr"][t - 1]
                if not np.isfinite(stop_distance) or stop_distance <= 0:
                    continue
                fill = arr["open"][t] * (1 + costs.slippage_rate)
                stop_price = fill - stop_distance
                qty = calculate_position_size(
                    equity=equity_prev,
                    risk_fraction=strategy_spec.risk_per_trade,
                    entry_price=fill,
                    stop_price=stop_price,
                    max_leverage=strategy_spec.max_leverage,
                )
                notional = qty * fill
                if notional < 5.0:
                    continue
                initial_risk = qty * stop_distance
                if open_risk + initial_risk > MAX_TOTAL_INITIAL_RISK * equity_prev:
                    continue
                entry_fee = costs.fee_rate * notional
                cash = cash - notional - entry_fee
                pos = _Position(
                    symbol=symbol,
                    qty=qty,
                    entry_price=fill,
                    stop_price=stop_price,
                    entry_bar=t,
                    initial_risk=initial_risk,
                    equity_at_entry=equity_prev,
                )
                positions[symbol] = pos
                open_risk += initial_risk
                touched.add(symbol)
                if arr["low"][t] <= stop_price:
                    exit_price = min(arr["open"][t], stop_price) * (1 - costs.slippage_rate)
                    record_exit(symbol, pos, t, exit_price, "stop_entrybar")

        marked = cash + sum(
            pos.qty * arrays[symbol]["close"][t] for symbol, pos in positions.items()
        )
        equity_arr[t] = marked
        equity_prev = marked

        for symbol in prepared:
            arr = arrays[symbol]
            entry_sig = bool(arr["entry_signal"][t])
            if symbol not in positions and symbol not in touched:
                pending_entry[symbol] = entry_sig
                pending_strength[symbol] = (
                    (arr["close"][t] - arr["upper"][t]) / arr["atr"][t]
                    if entry_sig and np.isfinite(arr["atr"][t]) and arr["atr"][t] > 0
                    else -np.inf
                )
            else:
                pending_entry[symbol] = False
                pending_strength[symbol] = -np.inf
            pending_exit[symbol] = (
                symbol in positions and bool(arr["close"][t] < arr["exit_lower"][t])
            )
        touched.clear()

    equity_series = pd.Series(equity_arr, index=grid, name="equity", dtype=np.float64)
    trades_df = (
        pd.DataFrame(trades, columns=list(_TRADE_COLUMNS))
        if trades
        else pd.DataFrame(columns=list(_TRADE_COLUMNS))
    )
    signals = pd.DataFrame(
        {symbol: arrays[symbol]["entry_signal"] for symbol in prepared},
        index=grid,
    )
    return BacktestResult(equity=equity_series, trades=trades_df, signals=signals)


def _check_contract() -> None:
    """Executable assertions locking the frozen portfolio surface."""
    assert run_portfolio_backtest.__name__ == "run_portfolio_backtest"
    assert MAX_TOTAL_INITIAL_RISK == 5 * 0.005
    from src.core.types import PortfolioSpec  # noqa: PLC0415

    spec = PortfolioSpec()
    assert (spec.universe_size, spec.max_positions, spec.liquidity_lookback_days) == (5, 5, 30)
    assert set(_TRADE_COLUMNS) >= {
        "symbol", "entry_time", "exit_time", "portfolio_equity_before_entry", "return_pct",
    }
    assert {f.name for f in fields(_Position)} == {
        "symbol", "qty", "entry_price", "stop_price", "entry_bar",
        "initial_risk", "equity_at_entry", "funding_pnl",
    }


_check_contract()
