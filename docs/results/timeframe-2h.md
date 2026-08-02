# Timeframe Census — 2h (v2, timeframe-invariant lookback scaling applied)

Structural activity-admission screen (`research run expert admission`, `technical-5symbol-rolling` gate params, BTC/ETH/SOL/BNB/XRP, 2022-06-01 -> 2025-12-31, pre-holdout). Indicator periods, router lookbacks, and the activity floor are now rescaled to a fixed calendar-time window relative to the 4h reference (`timeframe_invariant_lookback_scaling` spec) instead of being fixed bar counts.

`admitted` means the candidate cleared the **activity floor only** (`min_closed_trades>=20` and a calendar-time-equivalent `min_active_return_bars`) - a prerequisite gate, not a profitability/edge verdict. Edge (CAGR/LCB) is assessed downstream once a candidate reaches this floor (see `docs/results/rolling-res.md` for the 4h full-pipeline P&L reference).


> `technical_rsi_trend_pullback_short_v1` was excluded from this run after it raised a fail-closed `DataIntegrityError` (equity exhausted) at this timeframe **even with timeframe-invariant lookback scaling applied** — confirming it is a systemic no-stop-loss engine defect (see `timeframe-census.md` §3), not a lookback-scaling artifact.

## Summary

| Metric | Value |
| :--- | :--- |
| Candidates admitted (activity floor) | **65 / 85** (76%) |
| Downstream family-unique proposals generated | 117,078 |
| Wall-clock time | 13.30s |

## Family x activity-floor pass rate (out of 5 symbols)

| Family | Admitted/5 | Failing symbols |
| :--- | :--- | :--- |
| ADX/DI Regime [long] (dead) | 0/5 | BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT |
| ADX/DI Regime [short] | 1/5 | BNBUSDT, BTCUSDT, SOLUSDT, XRPUSDT |
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
| RSI Trend Pullback [long] | 4/5 | XRPUSDT |
| Stochastic Trend Pullback [long] | 5/5 | - |
| Stochastic Trend Pullback [short] | 5/5 | - |

## Real P&L edge verification (`research run expert backtest`, single-symbol, single-candidate)

Passing the activity floor above only means enough trades fired to be measurable — it says nothing about profitability. This section runs the actual backtest pipeline (net of fees/slippage, with the sealed observation reliability gate) for **every candidate that passed the activity floor at this timeframe** — not a subset.

| Family | N symbols | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | :--- | :--- | :--- |
| ADX/DI Regime [short] | 1 | +0.18% | -2.33% | 0/1 |
| BB Squeeze Breakout [long] | 5 | -0.71% | +0.00% | 0/5 |
| BB Squeeze Breakout [short] | 5 | -4.45% | -0.89% | 0/5 |
| CCI Trend Pullback [long] | 5 | -0.12% | +0.00% | 0/5 |
| CCI Trend Pullback [short] | 5 | -0.95% | -1.40% | 0/5 |
| EMA Alignment [long] | 5 | -3.59% | -10.06% | 0/5 |
| EMA Alignment [short] | 5 | -3.04% | -3.26% | 0/5 |
| Ichimoku Cloud [long] | 5 | -7.19% | -14.72% | 0/5 |
| Ichimoku Cloud [short] | 5 | -4.40% | -7.43% | 0/5 |
| MACD Histogram Regime [long] | 5 | -4.76% | -12.79% | 0/5 |
| MACD Histogram Regime [short] | 5 | -5.88% | -5.47% | 0/5 |
| RSI Trend Pullback [long] | 4 | -8.76% | -12.06% | 0/4 |
| Stochastic Trend Pullback [long] | 5 | -4.97% | -7.40% | 0/5 |
| Stochastic Trend Pullback [short] | 5 | -3.34% | -3.49% | 0/5 |

**Verdict at 2h: 0/65 backtests passed the observation gate.** Every family/side tested at this timeframe shows a negative or flat mean CAGR. See `timeframe-census.md` §3 for the full cross-timeframe synthesis (0/499 across all 7 timeframes).
