# Timeframe Census — 8h (v2, timeframe-invariant lookback scaling applied)

Structural activity-admission screen (`research run expert admission`, `technical-5symbol-rolling` gate params, BTC/ETH/SOL/BNB/XRP, 2022-06-01 -> 2025-12-31, pre-holdout). Indicator periods, router lookbacks, and the activity floor are now rescaled to a fixed calendar-time window relative to the 4h reference (`timeframe_invariant_lookback_scaling` spec) instead of being fixed bar counts.

`admitted` means the candidate cleared the **activity floor only** (`min_closed_trades>=20` and a calendar-time-equivalent `min_active_return_bars`) - a prerequisite gate, not a profitability/edge verdict. Edge (CAGR/LCB) is assessed downstream once a candidate reaches this floor (see `docs/results/rolling-res.md` for the 4h full-pipeline P&L reference).


## Summary

| Metric | Value |
| :--- | :--- |
| Candidates admitted (activity floor) | **85 / 90** (94%) |
| Downstream family-unique proposals generated | 579,062 |
| Wall-clock time | 6.65s |

## Family x activity-floor pass rate (out of 5 symbols)

| Family | Admitted/5 | Failing symbols |
| :--- | :--- | :--- |
| ADX/DI Regime [long] | 5/5 | - |
| ADX/DI Regime [short] | 5/5 | - |
| BB Squeeze Breakout [long] | 2/5 | ETHUSDT, SOLUSDT, XRPUSDT |
| BB Squeeze Breakout [short] | 5/5 | - |
| CCI Trend Pullback [long] | 5/5 | - |
| CCI Trend Pullback [short] | 5/5 | - |
| EMA Alignment [long] | 5/5 | - |
| EMA Alignment [short] | 5/5 | - |
| Ichimoku Cloud [long] | 5/5 | - |
| Ichimoku Cloud [short] | 5/5 | - |
| MACD Histogram Regime [long] | 5/5 | - |
| MACD Histogram Regime [short] | 5/5 | - |
| MFI Trend Pullback [long] | 5/5 | - |
| MFI Trend Pullback [short] | 3/5 | ETHUSDT, SOLUSDT |
| RSI Trend Pullback [long] | 5/5 | - |
| RSI Trend Pullback [short] | 5/5 | - |
| Stochastic Trend Pullback [long] | 5/5 | - |
| Stochastic Trend Pullback [short] | 5/5 | - |

## Real P&L edge verification (`research run expert backtest`, single-symbol, single-candidate)

Passing the activity floor above only means enough trades fired to be measurable — it says nothing about profitability. This section runs the actual backtest pipeline (net of fees/slippage, with the sealed observation reliability gate) for **every candidate that passed the activity floor at this timeframe** — not a subset.

| Family | N symbols | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | :--- | :--- | :--- |
| ADX/DI Regime [long] | 5 | -2.49% | -10.00% | 0/5 |
| ADX/DI Regime [short] | 5 | -4.48% | -4.97% | 0/5 |
| BB Squeeze Breakout [long] | 2 | +0.63% | -1.85% | 0/2 |
| BB Squeeze Breakout [short] | 5 | -0.71% | -1.66% | 0/5 |
| CCI Trend Pullback [long] | 5 | +0.24% | -3.46% | 0/5 |
| CCI Trend Pullback [short] | 5 | -2.12% | -2.06% | 0/5 |
| EMA Alignment [long] | 5 | -0.61% | -7.97% | 0/5 |
| EMA Alignment [short] | 5 | -1.14% | +0.00% | 0/5 |
| Ichimoku Cloud [long] | 5 | +0.26% | -9.96% | 0/5 |
| Ichimoku Cloud [short] | 5 | -4.45% | -5.11% | 0/5 |
| MACD Histogram Regime [long] | 5 | -0.71% | -8.40% | 0/5 |
| MACD Histogram Regime [short] | 5 | -3.54% | -5.30% | 0/5 |
| MFI Trend Pullback [long] | 5 | -1.11% | -0.73% | 0/5 |
| MFI Trend Pullback [short] | 3 | -0.16% | -1.13% | 0/3 |
| RSI Trend Pullback [long] | 5 | -4.58% | -6.25% | 0/5 |
| RSI Trend Pullback [short] | 5 | -4.51% | -5.92% | 0/5 |
| Stochastic Trend Pullback [long] | 5 | -1.72% | -3.51% | 0/5 |
| Stochastic Trend Pullback [short] | 5 | -1.34% | -2.28% | 0/5 |

**Verdict at 8h: 0/85 backtests passed the observation gate.** Every family/side tested at this timeframe shows a negative or flat mean CAGR. See `timeframe-census.md` §3 for the full cross-timeframe synthesis (0/499 across all 7 timeframes).

## Stop-loss exit sweep (`research run expert exit-sweep`, v3 fast batched tool)

The rows above are the `stop_loss_mode=None` control (no stop-loss) - kept exactly as measured. This section adds
the opt-in causal stop-loss engine, swept across all 4 exit combinations at 3 magnitudes each, for all 18 alive
candidates at 8h (partial-N families kept at their admitted symbol subset - BB Squeeze Breakout long:
BTCUSDT/BNBUSDT; MFI Trend Pullback short: BTCUSDT/BNBUSDT/XRPUSDT) - 1,170 cells (candidate x setting x symbol,
including the baseline row).

**Result: 0/1,170 individual cells passed the gate**, and aggregated per (candidate, setting) over each family's
correctly-admitted symbol subset, **0 of the 216 non-baseline settings (18 candidates x 12 settings) show any
gate PASS, and none show a positive median LCB90 CAGR either.** Every family/side stays at 0/N gate PASS across
every magnitude and anchor tested, matching the no-stop baseline's 0/85 finding. **Verdict: the stop-loss engine
did not flip any 8h family positive.** Full methodology and combined cross-timeframe verdict:
`docs/results/stop-loss-edge-check.md`.
