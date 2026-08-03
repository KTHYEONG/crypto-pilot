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

## Stop-loss exit sweep (v2, `technical_expert_edge_recovery.md` §3)

The rows above are the `stop_loss_mode=None` control (no stop-loss) - kept exactly as measured. This section adds
the opt-in causal stop-loss engine, swept across all 4 exit combinations at 3 magnitudes each for all 13 alive
candidates (BB Squeeze Breakout both sides, Stochastic Trend Pullback both sides, and MFI Trend Pullback short
confirmed still dead, 0/5 activity floor, at 1d before spending sweep budget on them) - 780 additional
single-symbol backtests. Full grid: `docs/results/stop-loss-edge-check.md`.

**Result: 3/780 individual (candidate, symbol) cells passed the gate**, all at the single tightest setting
(`fixed_pct`, `stop_loss_value=0.03`, trailing) - one symbol each for EMA Alignment long (XRPUSDT), Ichimoku
Cloud short (ETHUSDT), and RSI Trend Pullback long (SOLUSDT). This is **not** read as recovered edge: the same
setting produces CAGR outliers between -5% and +522% across the other symbols of the same candidates and
inflates trade counts 2-4x (whipsaw churn from a 3% stop that is tighter than typical 1d crypto ATR), and no
other magnitude or the ATR-normalized variants reproduce a pass for the same candidates. Aggregated across all
5 symbols per candidate (the family-level measure the census/gate methodology actually uses), only 2 of the 156
1d settings show a positive **median** LCB90 CAGR (Ichimoku Cloud short +1.11%, RSI Trend Pullback long
+1.66%, both from the same noisy 3%-trailing setting), and the aggregate family verdict stays 0/5 gate PASS for
every family. **Verdict: the stop-loss engine did not flip any 1d family positive** either - see
`docs/results/stop-loss-edge-check.md` for full per-symbol detail and the honest noise-vs-edge discussion.
