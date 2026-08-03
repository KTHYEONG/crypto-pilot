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

## Stop-loss exit sweep (`research run expert exit-sweep`, v3 fast batched tool)

The rows above are the `stop_loss_mode=None` control (no stop-loss) - kept exactly as measured. This section adds
the opt-in causal stop-loss engine, swept across all 4 exit combinations at 3 magnitudes each, for all 14 alive
candidates at 2h (partial-N families kept at their admitted symbol subset - ADX/DI short: ETHUSDT only; RSI
Trend Pullback long: BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT; MFI both sides excluded, confirmed still dead, 0/5 activity
floor, at 2h) - 910 cells (candidate x setting x symbol, including the baseline row).

**Result: 0/910 individual cells passed the gate**, and aggregated per (candidate, setting) over each family's
correctly-admitted symbol subset, **0 of the 168 non-baseline settings (14 candidates x 12 settings) show any
gate PASS, and none show a positive median LCB90 CAGR either.** Every family/side stays at 0/N gate PASS across
every magnitude and anchor tested, matching the no-stop baseline's 0/65 finding. **Verdict: the stop-loss engine
did not flip any 2h family positive.** Full methodology and combined cross-timeframe verdict:
`docs/results/stop-loss-edge-check.md`.
