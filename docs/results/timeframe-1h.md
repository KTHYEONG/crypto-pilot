# Timeframe Census — 1h (v2, timeframe-invariant lookback scaling applied)

Structural activity-admission screen (`research run expert admission`, `technical-5symbol-rolling` gate params, BTC/ETH/SOL/BNB/XRP, 2022-06-01 -> 2025-12-31, pre-holdout). Indicator periods, router lookbacks, and the activity floor are now rescaled to a fixed calendar-time window relative to the 4h reference (`timeframe_invariant_lookback_scaling` spec) instead of being fixed bar counts.

`admitted` means the candidate cleared the **activity floor only** (`min_closed_trades>=20` and a calendar-time-equivalent `min_active_return_bars`) - a prerequisite gate, not a profitability/edge verdict. Edge (CAGR/LCB) is assessed downstream once a candidate reaches this floor (see `docs/results/rolling-res.md` for the 4h full-pipeline P&L reference).


> `technical_rsi_trend_pullback_short_v1` was excluded from this run after it raised a fail-closed `DataIntegrityError` (equity exhausted) at this timeframe **even with timeframe-invariant lookback scaling applied** — confirming it is a systemic no-stop-loss engine defect (see `timeframe-census.md` §3), not a lookback-scaling artifact.

## Summary

| Metric | Value |
| :--- | :--- |
| Candidates admitted (activity floor) | **55 / 85** (65%) |
| Downstream family-unique proposals generated | 40,414 |
| Wall-clock time | 22.86s |

## Family x activity-floor pass rate (out of 5 symbols)

| Family | Admitted/5 | Failing symbols |
| :--- | :--- | :--- |
| ADX/DI Regime [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| ADX/DI Regime [short] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| BB Squeeze Breakout [long] | 5/5 | - |
| BB Squeeze Breakout [short] | 5/5 | - |
| CCI Trend Pullback [long] | 2/5 | BNBUSDT, ETHUSDT, XRPUSDT |
| CCI Trend Pullback [short] | 3/5 | BNBUSDT, SOLUSDT |
| EMA Alignment [long] | 5/5 | - |
| EMA Alignment [short] | 5/5 | - |
| Ichimoku Cloud [long] | 5/5 | - |
| Ichimoku Cloud [short] | 5/5 | - |
| MACD Histogram Regime [long] | 5/5 | - |
| MACD Histogram Regime [short] | 5/5 | - |
| MFI Trend Pullback [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| MFI Trend Pullback [short] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| RSI Trend Pullback [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| Stochastic Trend Pullback [long] | 5/5 | - |
| Stochastic Trend Pullback [short] | 5/5 | - |

## Real P&L edge verification (`research run expert backtest`, single-symbol, single-candidate)

Passing the activity floor above only means enough trades fired to be measurable — it says nothing about profitability. This section runs the actual backtest pipeline (net of fees/slippage, with the sealed observation reliability gate) for **every candidate that passed the activity floor at this timeframe** — not a subset.

| Family | N symbols | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | :--- | :--- | :--- |
| BB Squeeze Breakout [long] | 5 | -5.49% | -7.21% | 0/5 |
| BB Squeeze Breakout [short] | 5 | -6.86% | -6.20% | 0/5 |
| CCI Trend Pullback [long] | 2 | -1.14% | -1.87% | 0/2 |
| CCI Trend Pullback [short] | 3 | -0.60% | -1.57% | 0/3 |
| EMA Alignment [long] | 5 | -8.41% | -15.08% | 0/5 |
| EMA Alignment [short] | 5 | -5.20% | -5.00% | 0/5 |
| Ichimoku Cloud [long] | 5 | -11.41% | -17.52% | 0/5 |
| Ichimoku Cloud [short] | 5 | -7.14% | -12.11% | 0/5 |
| MACD Histogram Regime [long] | 5 | -11.61% | -19.50% | 0/5 |
| MACD Histogram Regime [short] | 5 | -9.20% | -9.81% | 0/5 |
| Stochastic Trend Pullback [long] | 5 | -2.72% | -3.55% | 0/5 |
| Stochastic Trend Pullback [short] | 5 | -2.64% | +0.00% | 0/5 |

**Verdict at 1h: 0/55 backtests passed the observation gate.** Every family/side tested at this timeframe shows a negative or flat mean CAGR. See `timeframe-census.md` §3 for the full cross-timeframe synthesis (0/499 across all 7 timeframes).
