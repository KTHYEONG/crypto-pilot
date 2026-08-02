# Timeframe Census — 6h (v2, timeframe-invariant lookback scaling applied)

Structural activity-admission screen (`research run expert admission`, `technical-5symbol-rolling` gate params, BTC/ETH/SOL/BNB/XRP, 2022-06-01 -> 2025-12-31, pre-holdout). Indicator periods, router lookbacks, and the activity floor are now rescaled to a fixed calendar-time window relative to the 4h reference (`timeframe_invariant_lookback_scaling` spec) instead of being fixed bar counts.

`admitted` means the candidate cleared the **activity floor only** (`min_closed_trades>=20` and a calendar-time-equivalent `min_active_return_bars`) - a prerequisite gate, not a profitability/edge verdict. Edge (CAGR/LCB) is assessed downstream once a candidate reaches this floor (see `docs/results/rolling-res.md` for the 4h full-pipeline P&L reference).


## Summary

| Metric | Value |
| :--- | :--- |
| Candidates admitted (activity floor) | **79 / 90** (88%) |
| Downstream family-unique proposals generated | 368,388 |
| Wall-clock time | 6.47s |

## Family x activity-floor pass rate (out of 5 symbols)

| Family | Admitted/5 | Failing symbols |
| :--- | :--- | :--- |
| ADX/DI Regime [long] | 5/5 | - |
| ADX/DI Regime [short] | 5/5 | - |
| BB Squeeze Breakout [long] | 3/5 | SOLUSDT, XRPUSDT |
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
| MFI Trend Pullback [short] | 1/5 | BNBUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| RSI Trend Pullback [long] | 5/5 | - |
| RSI Trend Pullback [short] | 5/5 | - |
| Stochastic Trend Pullback [long] | 5/5 | - |
| Stochastic Trend Pullback [short] | 5/5 | - |

## Real P&L edge verification (`research run expert backtest`, single-symbol, single-candidate)

Passing the activity floor above only means enough trades fired to be measurable — it says nothing about profitability. This section runs the actual backtest pipeline (net of fees/slippage, with the sealed observation reliability gate) for **every candidate that passed the activity floor at this timeframe** — not a subset.

| Family | N symbols | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | :--- | :--- | :--- |
| ADX/DI Regime [long] | 5 | -0.80% | -3.34% | 0/5 |
| ADX/DI Regime [short] | 5 | -4.92% | -9.23% | 0/5 |
| BB Squeeze Breakout [long] | 3 | -1.13% | -0.09% | 0/3 |
| BB Squeeze Breakout [short] | 5 | -1.92% | -2.90% | 0/5 |
| CCI Trend Pullback [long] | 5 | +0.63% | -4.18% | 0/5 |
| CCI Trend Pullback [short] | 5 | -2.67% | -2.75% | 0/5 |
| EMA Alignment [long] | 5 | -3.53% | -6.98% | 0/5 |
| EMA Alignment [short] | 5 | -3.11% | -6.72% | 0/5 |
| Ichimoku Cloud [long] | 5 | -4.16% | -13.78% | 0/5 |
| Ichimoku Cloud [short] | 5 | -1.58% | -5.23% | 0/5 |
| MACD Histogram Regime [long] | 5 | -2.03% | -7.07% | 0/5 |
| MACD Histogram Regime [short] | 5 | -3.79% | -7.24% | 0/5 |
| MFI Trend Pullback [short] | 1 | +0.00% | +0.00% | 0/1 |
| RSI Trend Pullback [long] | 5 | -2.51% | -8.22% | 0/5 |
| RSI Trend Pullback [short] | 5 | -10.11% | -9.59% | 0/5 |
| Stochastic Trend Pullback [long] | 5 | -1.87% | -2.17% | 0/5 |
| Stochastic Trend Pullback [short] | 5 | -1.03% | -1.10% | 0/5 |

**Verdict at 6h: 0/79 backtests passed the observation gate.** Every family/side tested at this timeframe shows a negative or flat mean CAGR. See `timeframe-census.md` §3 for the full cross-timeframe synthesis (0/499 across all 7 timeframes).
