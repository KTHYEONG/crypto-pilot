# Timeframe Census — 4h (v2, timeframe-invariant lookback scaling applied)

Structural activity-admission screen (`research run expert admission`, `technical-5symbol-rolling` gate params, BTC/ETH/SOL/BNB/XRP, 2022-06-01 -> 2025-12-31, pre-holdout). Indicator periods, router lookbacks, and the activity floor are now rescaled to a fixed calendar-time window relative to the 4h reference (`timeframe_invariant_lookback_scaling` spec) instead of being fixed bar counts.

`admitted` means the candidate cleared the **activity floor only** (`min_closed_trades>=20` and a calendar-time-equivalent `min_active_return_bars`) - a prerequisite gate, not a profitability/edge verdict. Edge (CAGR/LCB) is assessed downstream once a candidate reaches this floor (see `docs/results/rolling-res.md` for the 4h full-pipeline P&L reference).


## Summary

| Metric | Value |
| :--- | :--- |
| Candidates admitted (activity floor) | **80 / 90** (89%) |
| Downstream family-unique proposals generated | 374,489 |
| Wall-clock time | 8.48s |

## Family x activity-floor pass rate (out of 5 symbols)

| Family | Admitted/5 | Failing symbols |
| :--- | :--- | :--- |
| ADX/DI Regime [long] | 5/5 | - |
| ADX/DI Regime [short] | 5/5 | - |
| BB Squeeze Breakout [long] | 5/5 | - |
| BB Squeeze Breakout [short] | 5/5 | - |
| CCI Trend Pullback [long] | 5/5 | - |
| CCI Trend Pullback [short] | 5/5 | - |
| EMA Alignment [long] | 5/5 | - |
| EMA Alignment [short] | 5/5 | - |
| Ichimoku Cloud [long] | 5/5 | - |
| Ichimoku Cloud [short] | 5/5 | - |
| MACD Histogram Regime [long] | 5/5 | - |
| MACD Histogram Regime [short] | 5/5 | - |
| MFI Trend Pullback [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| MFI Trend Pullback [short] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| RSI Trend Pullback [long] | 5/5 | - |
| RSI Trend Pullback [short] | 5/5 | - |
| Stochastic Trend Pullback [long] | 5/5 | - |
| Stochastic Trend Pullback [short] | 5/5 | - |

## Real P&L edge verification (`research run expert backtest`, single-symbol, single-candidate)

Passing the activity floor above only means enough trades fired to be measurable — it says nothing about profitability. This section runs the actual backtest pipeline (net of fees/slippage, with the sealed observation reliability gate) for **every candidate that passed the activity floor at this timeframe** — not a subset.

| Family | N symbols | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | :--- | :--- | :--- |
| ADX/DI Regime [long] | 5 | -3.31% | -8.04% | 0/5 |
| ADX/DI Regime [short] | 5 | -1.23% | -2.67% | 0/5 |
| BB Squeeze Breakout [long] | 5 | -0.68% | -1.61% | 0/5 |
| BB Squeeze Breakout [short] | 5 | -5.46% | -7.48% | 0/5 |
| CCI Trend Pullback [long] | 5 | +2.35% | -2.33% | 0/5 |
| CCI Trend Pullback [short] | 5 | -0.55% | -0.54% | 0/5 |
| EMA Alignment [long] | 5 | -3.41% | -7.03% | 0/5 |
| EMA Alignment [short] | 5 | -4.55% | -7.30% | 0/5 |
| Ichimoku Cloud [long] | 5 | -5.32% | -12.00% | 0/5 |
| Ichimoku Cloud [short] | 5 | -7.95% | -10.01% | 0/5 |
| MACD Histogram Regime [long] | 5 | -4.61% | -10.91% | 0/5 |
| MACD Histogram Regime [short] | 5 | -3.82% | -6.64% | 0/5 |
| RSI Trend Pullback [long] | 5 | -4.99% | -8.70% | 0/5 |
| RSI Trend Pullback [short] | 5 | -2.18% | -1.36% | 0/5 |
| Stochastic Trend Pullback [long] | 5 | -1.41% | -3.13% | 0/5 |
| Stochastic Trend Pullback [short] | 5 | -0.59% | +0.00% | 0/5 |

**Verdict at 4h: 0/80 backtests passed the observation gate.** Every family/side tested at this timeframe shows a negative or flat mean CAGR. See `timeframe-census.md` §3 for the full cross-timeframe synthesis (0/499 across all 7 timeframes).
