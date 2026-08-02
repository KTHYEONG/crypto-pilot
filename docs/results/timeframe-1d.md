# Timeframe Census — 1d (v2, timeframe-invariant lookback scaling applied)

Structural activity-admission screen (`research run expert admission`, `technical-5symbol-rolling` gate params, BTC/ETH/SOL/BNB/XRP, 2022-06-01 -> 2025-12-31, pre-holdout). Indicator periods, router lookbacks, and the activity floor are now rescaled to a fixed calendar-time window relative to the 4h reference (`timeframe_invariant_lookback_scaling` spec) instead of being fixed bar counts.

`admitted` means the candidate cleared the **activity floor only** (`min_closed_trades>=20` and a calendar-time-equivalent `min_active_return_bars`) - a prerequisite gate, not a profitability/edge verdict. Edge (CAGR/LCB) is assessed downstream once a candidate reaches this floor (see `docs/results/rolling-res.md` for the 4h full-pipeline P&L reference).


## Summary

| Metric | Value |
| :--- | :--- |
| Candidates admitted (activity floor) | **65 / 90** (72%) |
| Downstream family-unique proposals generated | 104,676 |
| Wall-clock time | 1.96s |

## Family x activity-floor pass rate (out of 5 symbols)

| Family | Admitted/5 | Failing symbols |
| :--- | :--- | :--- |
| ADX/DI Regime [long] | 5/5 | - |
| ADX/DI Regime [short] | 5/5 | - |
| BB Squeeze Breakout [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| BB Squeeze Breakout [short] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| CCI Trend Pullback [long] | 5/5 | - |
| CCI Trend Pullback [short] | 5/5 | - |
| EMA Alignment [long] | 5/5 | - |
| EMA Alignment [short] | 5/5 | - |
| Ichimoku Cloud [long] | 5/5 | - |
| Ichimoku Cloud [short] | 5/5 | - |
| MACD Histogram Regime [long] | 5/5 | - |
| MACD Histogram Regime [short] | 5/5 | - |
| MFI Trend Pullback [long] | 5/5 | - |
| MFI Trend Pullback [short] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| RSI Trend Pullback [long] | 5/5 | - |
| RSI Trend Pullback [short] | 5/5 | - |
| Stochastic Trend Pullback [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| Stochastic Trend Pullback [short] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
## Real P&L edge verification (`research run expert backtest`, single-symbol, single-candidate)

Passing the activity floor above only means enough trades fired to be measurable — it says nothing about profitability. This section runs the actual backtest pipeline (net of fees/slippage, with the sealed observation reliability gate) for **every candidate that passed the activity floor at this timeframe** — not a subset.

| Family | N symbols | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | :--- | :--- | :--- |
| ADX/DI Regime [long] | 5 | -3.40% | -7.07% | 0/5 |
| ADX/DI Regime [short] | 5 | -6.41% | -13.04% | 0/5 |
| CCI Trend Pullback [long] | 5 | -1.40% | -4.14% | 0/5 |
| CCI Trend Pullback [short] | 5 | -0.81% | -3.01% | 0/5 |
| EMA Alignment [long] | 5 | -1.41% | -6.63% | 0/5 |
| EMA Alignment [short] | 5 | -5.59% | -10.06% | 0/5 |
| Ichimoku Cloud [long] | 5 | -1.87% | -9.85% | 0/5 |
| Ichimoku Cloud [short] | 5 | -1.23% | -8.42% | 0/5 |
| MACD Histogram Regime [long] | 5 | -3.82% | -8.24% | 0/5 |
| MACD Histogram Regime [short] | 5 | -3.88% | -6.28% | 0/5 |
| MFI Trend Pullback [long] | 5 | -1.64% | -7.24% | 0/5 |
| RSI Trend Pullback [long] | 5 | +0.81% | -7.39% | 0/5 |
| RSI Trend Pullback [short] | 5 | -7.30% | -10.55% | 0/5 |

**Verdict at 1d: 0/65 backtests passed the observation gate.** Every family/side tested at this timeframe shows a negative or flat mean CAGR. See `timeframe-census.md` §3 for the full cross-timeframe synthesis (0/499 across all 7 timeframes).
