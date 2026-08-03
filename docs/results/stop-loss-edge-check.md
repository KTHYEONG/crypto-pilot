# Stop-Loss / Exit-Mechanism Edge Check (v2 verification sweep, spec `technical_expert_edge_recovery.md` §3)

Full grid measurement of the opt-in causal stop-loss engine (`stop_loss_mode`, `stop_loss_value`, `atr_period`,
`trailing_stop` in `run_technical_expert_backtest` / `research run expert backtest`) against `docs/results/timeframe-census.md`'s
0/499 baseline (no stop-loss). Per spec §3, every priority family is swept across all four exit combinations
(`fixed_pct`/`atr_multiple` x static/trailing) at three magnitude values each, on the same 5 symbols
(BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT), 2022-06-01 -> 2025-12-31 pre-holdout window.

**Coverage**: all 7 timeframes are now swept. 4h and 1d were swept first, in full, via 1,740 individual CLI
`research run expert backtest` calls (see the original methodology below). 1h/2h/6h/8h/12h were swept in a second
pass using the newer, ~12-15x faster batched tool `research run expert exit-sweep`
(`src/application/research/expert/exit_sweep.py`, /check-verified), which produces the same
`observation.gate`-derived `cagr`/`lcb90_cagr`/`verdict` statistics per (candidate, symbol, setting) cell but
loads data once per (symbol, timeframe) and parallelizes across pairs via `ProcessPoolExecutor` instead of
spawning one CLI process per cell. **Total across all 7 timeframes: 6,550 cells**
(4h: 960, 1d: 780, 1h: 780, 2h: 910, 6h: 1,040, 8h: 1,170, 12h: 910 - candidate x setting x symbol, each including
the `stop_loss_mode=None` baseline row). Partial-N families (candidates where fewer than 5/5 symbols cleared the
activity floor at that timeframe, per each `timeframe-<TF>.md`'s admission table) were swept across the same
5-symbol grid as every other candidate for tool-batching efficiency, but aggregated (`mean_cagr` /
`median_lcb90_cagr` / `gate_pass_count`) only over the symbols that were actually admitted for that exact
candidate at that timeframe, to stay apples-to-apples with each file's existing baseline N columns - the excluded
symbols' cells are near-zero-trade/inert rows (`compute_metrics` returns gracefully for <2 equity points), not
errors, and are simply dropped from the aggregation, not double-counted as passes or fails. Per-timeframe detail:
`timeframe-1h.md`, `timeframe-2h.md`, `timeframe-6h.md`, `timeframe-8h.md`, `timeframe-12h.md` (each has its own
"Stop-loss exit sweep" section).

**Magnitudes tested**: `fixed_pct` in {3%, 5%, 8%} of entry price; `atr_multiple` in {1.5x, 2.5x, 4.0x} of the
causally-shifted `ATR(14)` (timeframe-scaled). Both crossed with static and trailing anchors -> 12 settings per
candidate per timeframe.

## Command template

```bash
uv run python -m src.cli.main research run expert backtest \
  --expert-id <return_source>:<SYMBOL> \
  --router-context-symbol BTCUSDT --router-trend-lookback-bars 48 \
  --router-volatility-lookback-bars 48 --router-min-context-history-bars 96 \
  --start 2022-06-01 --end 2025-12-31 --timeframe <4h|1d> --no-log-run \
  --stop-loss-mode <fixed_pct|atr_multiple> --stop-loss-value <VALUE> [--trailing-stop]
```

Router args match the canonical invocation pattern already used in `docs/guides/guide.md` §2.1/2.2 and
`tests/integration/cli/test_library_admission_backtest.py` (`BTCUSDT` / 48 / 48 / 96) - required by
`TechnicalLibraryAdmissionBacktestRequest` even though this per-candidate diagnostic use does not route by
regime. Per-family `Mean CAGR` / `Median LCB90 CAGR` / `Gate PASS` are aggregated across the 5 symbols from
`observation.gate.point_cagr` (reported as `metrics.cagr`) / `observation.gate.lcb90_cagr` /
`observation.gate.verdict=="PASS"` in the CLI's JSON report, exactly as `timeframe-4h.md` / `timeframe-1d.md`
already do for the no-stop baseline.

## Overall result

| Timeframe | Cells run (candidate x setting x symbol) | Gate PASS cells | Settings with any PASS | Settings with positive median LCB90 CAGR |
| :--- | ---: | ---: | ---: | ---: |
| 4h | 960 | 0 | 0 / 192 | 0 / 192 |
| 1d | 780 | 3 | 3 / 156 | 2 / 156 |
| 1h | 780 | 0 | 0 / 144 | 0 / 144 |
| 2h | 910 | 0 | 0 / 168 | 0 / 168 |
| 6h | 1,040 | 0 | 0 / 192 | 0 / 192 |
| 8h | 1,170 | 0 | 0 / 216 | 0 / 216 |
| 12h | 910 | 0 | 0 / 168 | 1 / 168 |
| **All 7** | **6,550** | **3** | **3 / 1,236** | **3 / 1,236** |

**1h/2h/6h/8h/12h: not a single one of the 4,810 additional cells passed the gate** - a stronger and cleaner null
than 4h/1d, since these five sweeps found zero gate-PASS cells even before any noise-vs-edge filtering (unlike
the 3 isolated 1d passes). 12h's single positive-median-LCB90 setting (Ichimoku Cloud short, `fixed_pct`/trailing
@3%, median LCB90 CAGR +0.15%) still shows 0/5 gate PASS and reproduces the exact same setting signature as the
1d noise passes (tightest fixed-percent stop x trailing) with no other magnitude, anchor, or family corroborating
it at 12h - read as the same whipsaw-sampling-noise pattern, not independent evidence of recovered edge. See each
`timeframe-<TF>.md`'s "Stop-loss exit sweep" section for detail.

**Combined 7-timeframe verdict: the stop-loss engine does not recover edge anywhere in the full 6,550-cell
census.** 6,547 of 6,550 cells (99.95%) are clean 0/5-gate-PASS negatives; the only 3 exceptions are the 1d
whipsaw-noise passes already diagnosed above as artifacts of a 3% stop being tighter than daily crypto ATR, not a
reproducible effect (no magnitude, anchor, or symbol-generalization corroborates them). The volatility-drag /
payoff-ratio mechanism the engine targets is real and visible in the raw cell deltas at every timeframe (trailing
variants routinely narrow the CAGR-vs-LCB90 gap for trend-confirmation families), but at no timeframe, on any of
the 24 magnitude/anchor settings, for any of the up-to-18 alive candidates, does it close the gap to the
observation gate's hurdle. This confirms the bottleneck diagnosed in `timeframe-census.md`'s cross-timeframe
0/499 finding is the absence of gross directional edge in these technical-indicator candidates, not the
choice of exit mechanism - stop-loss/exit-mechanism redesign is not a viable recovery lever for this expert
library, at any of the 7 timeframes tested.

**4h: no combination, at any magnitude, for any family/side, flipped a single gate PASS or a single positive
median-LCB90-CAGR family cell.** Every one of the 192 (family/side x combo x magnitude) 4h settings remains
`0/5` gate PASS, matching the pre-stop-loss `timeframe-4h.md` baseline's `0/80` finding — the stop-loss engine did
not recover edge at 4h for any of the 16 alive candidates.

**1d: 3 of 780 individual (candidate, symbol) cells passed the gate**, all under the same setting
(`fixed_pct`, `stop_loss_value=0.03`, `trailing_stop=True` - the tightest fixed-percent stop crossed with
trailing) - one symbol each for `technical_ema_alignment_long_v1` (XRPUSDT), `technical_ichimoku_cloud_short_v1`
(ETHUSDT), and `technical_rsi_trend_pullback_long_v1` (SOLUSDT). **This is not read as genuine recovered edge**:
a 3% trailing stop on daily bars is tighter than typical single-day crypto ATR (see per-symbol detail below - the
same setting produces CAGR outliers from -5% to +522% across the other 4 symbols of the same candidate, and
trade counts balloon 2-4x versus the no-stop baseline, e.g. EMA Alignment long BTCUSDT goes from ~100 trades to
302) - the classic signature of a stop distance pathologically tighter than the instrument's own daily range,
producing whipsaw churn whose sign is effectively a coin flip per symbol rather than a structural edge. Out of
1,740 cells tested, 3 isolated single-symbol passes (0.17%) at exactly the tightest, most whipsaw-prone setting is
consistent with sampling noise under multiple testing, not a reproducible per-family effect - no other magnitude
or the ATR-normalized variants (which should track true volatility better) reproduce it for the same candidates.

**Verdict: the stop-loss engine did not flip any family's aggregate (median-of-5-symbols) LCB90 CAGR positive at
either 4h or 1d.** The volatility-drag / payoff-ratio mechanism described in the spec's §1 diagnosis is real (the
engine visibly changes trade_count, cagr, and lcb90_cagr in the expected directions for many cells - e.g. trailing
variants often narrow the CAGR-vs-LCB90 gap for trend families, consistent with protecting profit_factor > 1
trade-level edge from full round-trip drawback) but is not large enough, at any of the 24 magnitude/anchor
settings tested, to overcome the underlying lack of gross edge already diagnosed in `timeframe-census.md`'s
0/499 finding. This matches the spec §4 expectation that volatility-scaled position sizing (deferred) may be the
next lever, not a foregone conclusion — reported honestly per `quant.md`'s "Logic Robustness Over Metrics".

## Per-symbol detail for the 3 gate-PASS 1d cells (fixed_pct, value=0.03, trailing=True)

| Candidate | Symbol | CAGR | LCB90 CAGR | Verdict | Trades |
| :--- | :--- | ---: | ---: | :--- | ---: |
| technical_ema_alignment_long_v1 | BTCUSDT | -4.96% | -8.77% | FAIL | 302 |
| technical_ema_alignment_long_v1 | ETHUSDT | -2.80% | -9.00% | FAIL | 366 |
| technical_ema_alignment_long_v1 | SOLUSDT | +521.86% | +193.92% | FAIL | 399 |
| technical_ema_alignment_long_v1 | BNBUSDT | +13.12% | -4.89% | FAIL | 377 |
| technical_ema_alignment_long_v1 | XRPUSDT | +173.05% | +37.82% | **PASS** | 325 |
| technical_ichimoku_cloud_short_v1 | BTCUSDT | -3.98% | -10.89% | FAIL | 267 |
| technical_ichimoku_cloud_short_v1 | ETHUSDT | +76.18% | +32.81% | **PASS** | 307 |
| technical_ichimoku_cloud_short_v1 | SOLUSDT | +133.90% | +47.07% | FAIL | 383 |
| technical_ichimoku_cloud_short_v1 | BNBUSDT | -7.00% | -15.75% | FAIL | 266 |
| technical_ichimoku_cloud_short_v1 | XRPUSDT | +25.49% | +1.11% | FAIL | 311 |
| technical_rsi_trend_pullback_long_v1 | BTCUSDT | +16.28% | +1.66% | FAIL | 176 |
| technical_rsi_trend_pullback_long_v1 | ETHUSDT | +4.42% | -5.42% | FAIL | 177 |
| technical_rsi_trend_pullback_long_v1 | SOLUSDT | +49.46% | +17.29% | **PASS** | 234 |
| technical_rsi_trend_pullback_long_v1 | BNBUSDT | +6.73% | -2.40% | FAIL | 193 |
| technical_rsi_trend_pullback_long_v1 | XRPUSDT | +45.75% | +14.94% | FAIL | 213 |

The 100x-plus CAGR swings within a single candidate at a fixed setting (e.g. EMA Alignment long: -5% to +522%
across 5 symbols) are themselves evidence against a structural effect — a genuine volatility-drag fix should
narrow cross-symbol dispersion, not explode it.

## Full grid (all cells, both timeframes)

### 4h full grid

| Family/Side | Combo | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | ---: | ---: | :--- |
| ADX/DI Regime [long] | atr_multiple/static @ 1.5x ATR | -3.07% | -4.59% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/static @ 2.5x ATR | -2.23% | -4.06% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/static @ 4.0x ATR | -2.01% | -4.29% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/trailing @ 1.5x ATR | -2.93% | -2.66% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/trailing @ 2.5x ATR | -4.59% | -6.78% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/trailing @ 4.0x ATR | -2.61% | -5.43% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/static @ 3% | -3.01% | -5.66% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/static @ 5% | -2.38% | -4.64% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/static @ 8% | -3.31% | -8.04% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/trailing @ 3% | -1.63% | -3.78% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/trailing @ 5% | -2.70% | -4.50% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/trailing @ 8% | -3.74% | -4.30% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/static @ 1.5x ATR | -1.09% | -2.89% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/static @ 2.5x ATR | -1.44% | -3.02% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/static @ 4.0x ATR | -1.23% | -2.67% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/trailing @ 1.5x ATR | -1.48% | +0.00% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/trailing @ 2.5x ATR | -0.05% | -0.87% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/trailing @ 4.0x ATR | -0.98% | -1.99% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/static @ 3% | -1.88% | -2.24% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/static @ 5% | -1.36% | -2.77% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/static @ 8% | -1.23% | -2.67% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/trailing @ 3% | -3.41% | -0.92% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/trailing @ 5% | -1.76% | -1.26% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/trailing @ 8% | -1.38% | -0.87% | 0/5 |
| BB Squeeze Breakout [long] | atr_multiple/static @ 1.5x ATR | -2.49% | -3.68% | 0/5 |
| BB Squeeze Breakout [long] | atr_multiple/static @ 2.5x ATR | -1.99% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | atr_multiple/static @ 4.0x ATR | -1.71% | -1.61% | 0/5 |
| BB Squeeze Breakout [long] | atr_multiple/trailing @ 1.5x ATR | -0.80% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | atr_multiple/trailing @ 2.5x ATR | -2.62% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | atr_multiple/trailing @ 4.0x ATR | -1.33% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | fixed_pct/static @ 3% | -1.80% | -3.01% | 0/5 |
| BB Squeeze Breakout [long] | fixed_pct/static @ 5% | -1.29% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | fixed_pct/static @ 8% | -0.81% | -3.17% | 0/5 |
| BB Squeeze Breakout [long] | fixed_pct/trailing @ 3% | -0.27% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | fixed_pct/trailing @ 5% | -0.13% | +0.00% | 0/5 |
| BB Squeeze Breakout [long] | fixed_pct/trailing @ 8% | -1.01% | +0.00% | 0/5 |
| BB Squeeze Breakout [short] | atr_multiple/static @ 1.5x ATR | -4.43% | -2.39% | 0/5 |
| BB Squeeze Breakout [short] | atr_multiple/static @ 2.5x ATR | -5.13% | -6.90% | 0/5 |
| BB Squeeze Breakout [short] | atr_multiple/static @ 4.0x ATR | -5.46% | -7.48% | 0/5 |
| BB Squeeze Breakout [short] | atr_multiple/trailing @ 1.5x ATR | -2.50% | -5.99% | 0/5 |
| BB Squeeze Breakout [short] | atr_multiple/trailing @ 2.5x ATR | -4.68% | -8.38% | 0/5 |
| BB Squeeze Breakout [short] | atr_multiple/trailing @ 4.0x ATR | -4.96% | -7.86% | 0/5 |
| BB Squeeze Breakout [short] | fixed_pct/static @ 3% | -4.74% | -6.51% | 0/5 |
| BB Squeeze Breakout [short] | fixed_pct/static @ 5% | -5.58% | -8.27% | 0/5 |
| BB Squeeze Breakout [short] | fixed_pct/static @ 8% | -5.46% | -7.48% | 0/5 |
| BB Squeeze Breakout [short] | fixed_pct/trailing @ 3% | -2.34% | -5.94% | 0/5 |
| BB Squeeze Breakout [short] | fixed_pct/trailing @ 5% | -3.16% | -3.87% | 0/5 |
| BB Squeeze Breakout [short] | fixed_pct/trailing @ 8% | -4.18% | -4.00% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/static @ 1.5x ATR | +1.05% | -3.71% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/static @ 2.5x ATR | +2.44% | -2.00% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/static @ 4.0x ATR | +2.03% | -2.33% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/trailing @ 1.5x ATR | -0.81% | -3.28% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/trailing @ 2.5x ATR | -0.65% | -3.61% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/trailing @ 4.0x ATR | +1.38% | -2.33% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/static @ 3% | -0.24% | -3.36% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/static @ 5% | +0.82% | -3.36% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/static @ 8% | +2.35% | -2.33% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/trailing @ 3% | -1.93% | -4.47% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/trailing @ 5% | -0.73% | -5.27% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/trailing @ 8% | +0.71% | -2.94% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/static @ 1.5x ATR | -0.37% | +0.00% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/static @ 2.5x ATR | -1.15% | -1.31% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/static @ 4.0x ATR | -1.15% | -1.31% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/trailing @ 1.5x ATR | -0.34% | +0.00% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/trailing @ 2.5x ATR | -0.71% | -1.31% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/trailing @ 4.0x ATR | -0.97% | -1.31% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/static @ 3% | -0.37% | +0.00% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/static @ 5% | -0.49% | -0.04% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/static @ 8% | -0.49% | -0.04% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/trailing @ 3% | -0.37% | +0.00% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/trailing @ 5% | -0.49% | -0.04% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/trailing @ 8% | -0.49% | -0.04% | 0/5 |
| EMA Alignment [long] | atr_multiple/static @ 1.5x ATR | -1.87% | -3.94% | 0/5 |
| EMA Alignment [long] | atr_multiple/static @ 2.5x ATR | -4.54% | -5.98% | 0/5 |
| EMA Alignment [long] | atr_multiple/static @ 4.0x ATR | -5.05% | -10.61% | 0/5 |
| EMA Alignment [long] | atr_multiple/trailing @ 1.5x ATR | -2.36% | -4.24% | 0/5 |
| EMA Alignment [long] | atr_multiple/trailing @ 2.5x ATR | -1.99% | -4.42% | 0/5 |
| EMA Alignment [long] | atr_multiple/trailing @ 4.0x ATR | -2.55% | -5.40% | 0/5 |
| EMA Alignment [long] | fixed_pct/static @ 3% | -3.31% | -5.67% | 0/5 |
| EMA Alignment [long] | fixed_pct/static @ 5% | -1.74% | -4.18% | 0/5 |
| EMA Alignment [long] | fixed_pct/static @ 8% | -4.53% | -7.73% | 0/5 |
| EMA Alignment [long] | fixed_pct/trailing @ 3% | -1.25% | -4.65% | 0/5 |
| EMA Alignment [long] | fixed_pct/trailing @ 5% | -1.00% | -1.07% | 0/5 |
| EMA Alignment [long] | fixed_pct/trailing @ 8% | -2.99% | -6.93% | 0/5 |
| EMA Alignment [short] | atr_multiple/static @ 1.5x ATR | -4.84% | -6.00% | 0/5 |
| EMA Alignment [short] | atr_multiple/static @ 2.5x ATR | -3.35% | -5.55% | 0/5 |
| EMA Alignment [short] | atr_multiple/static @ 4.0x ATR | -4.28% | -6.29% | 0/5 |
| EMA Alignment [short] | atr_multiple/trailing @ 1.5x ATR | -0.75% | -1.79% | 0/5 |
| EMA Alignment [short] | atr_multiple/trailing @ 2.5x ATR | -2.22% | -3.72% | 0/5 |
| EMA Alignment [short] | atr_multiple/trailing @ 4.0x ATR | -3.62% | -5.33% | 0/5 |
| EMA Alignment [short] | fixed_pct/static @ 3% | -3.80% | -5.67% | 0/5 |
| EMA Alignment [short] | fixed_pct/static @ 5% | -3.49% | -4.41% | 0/5 |
| EMA Alignment [short] | fixed_pct/static @ 8% | -4.21% | -7.30% | 0/5 |
| EMA Alignment [short] | fixed_pct/trailing @ 3% | +5.14% | +0.00% | 0/5 |
| EMA Alignment [short] | fixed_pct/trailing @ 5% | -1.33% | -1.69% | 0/5 |
| EMA Alignment [short] | fixed_pct/trailing @ 8% | -1.48% | -1.56% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/static @ 1.5x ATR | -4.45% | -7.55% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/static @ 2.5x ATR | -4.57% | -10.93% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/static @ 4.0x ATR | -4.76% | -10.44% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/trailing @ 1.5x ATR | -1.91% | -6.42% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/trailing @ 2.5x ATR | -1.99% | -6.58% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/trailing @ 4.0x ATR | -3.53% | -8.14% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/static @ 3% | -4.09% | -8.15% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/static @ 5% | -2.25% | -7.45% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/static @ 8% | -2.36% | -8.84% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/trailing @ 3% | -3.02% | -6.57% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/trailing @ 5% | -2.60% | -8.29% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/trailing @ 8% | -3.25% | -6.97% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/static @ 1.5x ATR | -6.01% | -8.12% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/static @ 2.5x ATR | -6.80% | -8.61% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/static @ 4.0x ATR | -8.20% | -9.37% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/trailing @ 1.5x ATR | -2.04% | +0.00% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/trailing @ 2.5x ATR | -3.14% | -4.27% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/trailing @ 4.0x ATR | -6.39% | -8.43% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/static @ 3% | -4.36% | -8.68% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/static @ 5% | -6.14% | -8.14% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/static @ 8% | -6.96% | -9.10% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/trailing @ 3% | +1.93% | -9.98% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/trailing @ 5% | -3.31% | -5.79% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/trailing @ 8% | -2.98% | -2.58% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/static @ 1.5x ATR | -3.76% | -5.29% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/static @ 2.5x ATR | -3.55% | -4.38% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/static @ 4.0x ATR | -3.75% | -7.40% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/trailing @ 1.5x ATR | -2.32% | -3.68% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/trailing @ 2.5x ATR | -2.67% | -2.74% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/trailing @ 4.0x ATR | -4.50% | -10.86% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/static @ 3% | -3.91% | -2.94% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/static @ 5% | -2.71% | -4.20% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/static @ 8% | -4.38% | -9.95% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/trailing @ 3% | -2.53% | -1.89% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/trailing @ 5% | -3.36% | -6.78% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/trailing @ 8% | -1.88% | -2.54% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/static @ 1.5x ATR | -2.26% | -4.23% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/static @ 2.5x ATR | -3.54% | -4.86% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/static @ 4.0x ATR | -4.23% | -6.89% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/trailing @ 1.5x ATR | -1.97% | -1.99% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/trailing @ 2.5x ATR | -2.35% | -4.70% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/trailing @ 4.0x ATR | -3.44% | -6.08% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/static @ 3% | -1.75% | -4.18% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/static @ 5% | -2.53% | -4.78% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/static @ 8% | -3.16% | -6.03% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/trailing @ 3% | -0.71% | +0.00% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/trailing @ 5% | -2.52% | -4.17% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/trailing @ 8% | -2.39% | -3.90% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/static @ 1.5x ATR | -5.96% | -7.87% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/static @ 2.5x ATR | -6.60% | -9.60% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/static @ 4.0x ATR | -6.61% | -7.70% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/trailing @ 1.5x ATR | -4.47% | -6.09% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/trailing @ 2.5x ATR | -6.65% | -8.43% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/trailing @ 4.0x ATR | -7.89% | -10.21% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/static @ 3% | -4.09% | -6.87% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/static @ 5% | -4.39% | -9.61% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/static @ 8% | -5.36% | -7.70% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/trailing @ 3% | -4.22% | -5.80% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/trailing @ 5% | -3.11% | -5.64% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/trailing @ 8% | -5.81% | -7.43% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/static @ 1.5x ATR | -0.00% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/static @ 2.5x ATR | -0.83% | -1.35% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/static @ 4.0x ATR | -0.68% | -1.35% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/trailing @ 1.5x ATR | -0.39% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/trailing @ 2.5x ATR | +0.00% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/trailing @ 4.0x ATR | -1.20% | -0.85% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/static @ 3% | -0.15% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/static @ 5% | +0.00% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/static @ 8% | +0.25% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/trailing @ 3% | +0.00% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/trailing @ 5% | +0.00% | +0.00% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/trailing @ 8% | -0.42% | +0.00% | 0/5 |
| Stochastic Trend Pullback [long] | atr_multiple/static @ 1.5x ATR | -0.96% | +0.00% | 0/5 |
| Stochastic Trend Pullback [long] | atr_multiple/static @ 2.5x ATR | -1.42% | -3.13% | 0/5 |
| Stochastic Trend Pullback [long] | atr_multiple/static @ 4.0x ATR | -1.25% | -3.13% | 0/5 |
| Stochastic Trend Pullback [long] | atr_multiple/trailing @ 1.5x ATR | -0.47% | +0.00% | 0/5 |
| Stochastic Trend Pullback [long] | atr_multiple/trailing @ 2.5x ATR | -1.76% | -3.13% | 0/5 |
| Stochastic Trend Pullback [long] | atr_multiple/trailing @ 4.0x ATR | -1.30% | -3.13% | 0/5 |
| Stochastic Trend Pullback [long] | fixed_pct/static @ 3% | -1.31% | -1.47% | 0/5 |
| Stochastic Trend Pullback [long] | fixed_pct/static @ 5% | -1.07% | -1.95% | 0/5 |
| Stochastic Trend Pullback [long] | fixed_pct/static @ 8% | -1.26% | -3.13% | 0/5 |
| Stochastic Trend Pullback [long] | fixed_pct/trailing @ 3% | -0.40% | +0.00% | 0/5 |
| Stochastic Trend Pullback [long] | fixed_pct/trailing @ 5% | -1.31% | -1.95% | 0/5 |
| Stochastic Trend Pullback [long] | fixed_pct/trailing @ 8% | -1.38% | -3.13% | 0/5 |
| Stochastic Trend Pullback [short] | atr_multiple/static @ 1.5x ATR | -0.64% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | atr_multiple/static @ 2.5x ATR | -0.31% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | atr_multiple/static @ 4.0x ATR | -0.31% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | atr_multiple/trailing @ 1.5x ATR | -0.10% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | atr_multiple/trailing @ 2.5x ATR | -0.50% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | atr_multiple/trailing @ 4.0x ATR | -0.31% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | fixed_pct/static @ 3% | -0.38% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | fixed_pct/static @ 5% | -0.42% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | fixed_pct/static @ 8% | -0.59% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | fixed_pct/trailing @ 3% | -0.10% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | fixed_pct/trailing @ 5% | -0.24% | +0.00% | 0/5 |
| Stochastic Trend Pullback [short] | fixed_pct/trailing @ 8% | -0.39% | +0.00% | 0/5 |
### 1d full grid

| Family/Side | Combo | Mean CAGR | Median LCB90 CAGR | Gate PASS |
| :--- | :--- | ---: | ---: | :--- |
| ADX/DI Regime [long] | atr_multiple/static @ 1.5x ATR | -2.40% | -8.13% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/static @ 2.5x ATR | -3.93% | -7.44% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/static @ 4.0x ATR | -3.30% | -7.07% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/trailing @ 1.5x ATR | -3.22% | -10.56% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/trailing @ 2.5x ATR | -5.08% | -11.63% | 0/5 |
| ADX/DI Regime [long] | atr_multiple/trailing @ 4.0x ATR | -3.62% | -7.14% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/static @ 3% | -2.64% | -9.91% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/static @ 5% | -2.59% | -7.56% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/static @ 8% | -4.53% | -9.26% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/trailing @ 3% | +124.57% | -7.48% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/trailing @ 5% | -3.87% | -10.79% | 0/5 |
| ADX/DI Regime [long] | fixed_pct/trailing @ 8% | -5.84% | -8.97% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/static @ 1.5x ATR | -6.14% | -12.33% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/static @ 2.5x ATR | -6.45% | -12.78% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/static @ 4.0x ATR | -5.21% | -11.24% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/trailing @ 1.5x ATR | -9.39% | -17.55% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/trailing @ 2.5x ATR | -9.08% | -17.70% | 0/5 |
| ADX/DI Regime [short] | atr_multiple/trailing @ 4.0x ATR | -5.67% | -12.27% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/static @ 3% | -7.72% | -11.35% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/static @ 5% | -6.79% | -14.06% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/static @ 8% | -8.07% | -18.40% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/trailing @ 3% | +67.17% | -1.41% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/trailing @ 5% | -6.46% | -18.91% | 0/5 |
| ADX/DI Regime [short] | fixed_pct/trailing @ 8% | -9.26% | -16.84% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/static @ 1.5x ATR | -1.49% | -3.68% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/static @ 2.5x ATR | -1.31% | -4.03% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/static @ 4.0x ATR | -1.40% | -4.14% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/trailing @ 1.5x ATR | -1.61% | -4.08% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/trailing @ 2.5x ATR | -2.03% | -5.65% | 0/5 |
| CCI Trend Pullback [long] | atr_multiple/trailing @ 4.0x ATR | -2.12% | -4.14% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/static @ 3% | -3.25% | -7.36% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/static @ 5% | -2.14% | -4.46% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/static @ 8% | -1.87% | -3.69% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/trailing @ 3% | +8.42% | -8.97% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/trailing @ 5% | -4.27% | -10.42% | 0/5 |
| CCI Trend Pullback [long] | fixed_pct/trailing @ 8% | -2.02% | -3.69% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/static @ 1.5x ATR | -1.00% | -3.06% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/static @ 2.5x ATR | -0.81% | -3.01% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/static @ 4.0x ATR | -0.81% | -3.01% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/trailing @ 1.5x ATR | -1.23% | -6.75% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/trailing @ 2.5x ATR | -0.61% | -3.01% | 0/5 |
| CCI Trend Pullback [short] | atr_multiple/trailing @ 4.0x ATR | -0.81% | -3.01% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/static @ 3% | -3.85% | -9.21% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/static @ 5% | -3.16% | -7.13% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/static @ 8% | -0.85% | -3.39% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/trailing @ 3% | +4.52% | -3.78% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/trailing @ 5% | -3.64% | -7.45% | 0/5 |
| CCI Trend Pullback [short] | fixed_pct/trailing @ 8% | -2.56% | -3.23% | 0/5 |
| EMA Alignment [long] | atr_multiple/static @ 1.5x ATR | -2.01% | -4.99% | 0/5 |
| EMA Alignment [long] | atr_multiple/static @ 2.5x ATR | -1.82% | -4.89% | 0/5 |
| EMA Alignment [long] | atr_multiple/static @ 4.0x ATR | -4.53% | -6.88% | 0/5 |
| EMA Alignment [long] | atr_multiple/trailing @ 1.5x ATR | -2.17% | -6.14% | 0/5 |
| EMA Alignment [long] | atr_multiple/trailing @ 2.5x ATR | -4.62% | -5.97% | 0/5 |
| EMA Alignment [long] | atr_multiple/trailing @ 4.0x ATR | -5.15% | -6.88% | 0/5 |
| EMA Alignment [long] | fixed_pct/static @ 3% | -2.22% | -9.12% | 0/5 |
| EMA Alignment [long] | fixed_pct/static @ 5% | -3.60% | -7.79% | 0/5 |
| EMA Alignment [long] | fixed_pct/static @ 8% | -2.20% | -4.07% | 0/5 |
| EMA Alignment [long] | fixed_pct/trailing @ 3% | +140.06% | -4.89% | 1/5 |
| EMA Alignment [long] | fixed_pct/trailing @ 5% | -7.13% | -9.76% | 0/5 |
| EMA Alignment [long] | fixed_pct/trailing @ 8% | -3.86% | -5.23% | 0/5 |
| EMA Alignment [short] | atr_multiple/static @ 1.5x ATR | +3.52% | -8.18% | 0/5 |
| EMA Alignment [short] | atr_multiple/static @ 2.5x ATR | +3.54% | -7.74% | 0/5 |
| EMA Alignment [short] | atr_multiple/static @ 4.0x ATR | -5.03% | -9.63% | 0/5 |
| EMA Alignment [short] | atr_multiple/trailing @ 1.5x ATR | +5.05% | -6.77% | 0/5 |
| EMA Alignment [short] | atr_multiple/trailing @ 2.5x ATR | +5.38% | -6.40% | 0/5 |
| EMA Alignment [short] | atr_multiple/trailing @ 4.0x ATR | -4.50% | -10.34% | 0/5 |
| EMA Alignment [short] | fixed_pct/static @ 3% | +2.20% | -11.31% | 0/5 |
| EMA Alignment [short] | fixed_pct/static @ 5% | +2.60% | -10.31% | 0/5 |
| EMA Alignment [short] | fixed_pct/static @ 8% | +2.80% | -8.26% | 0/5 |
| EMA Alignment [short] | fixed_pct/trailing @ 3% | +28.18% | -8.46% | 0/5 |
| EMA Alignment [short] | fixed_pct/trailing @ 5% | +4.96% | -9.14% | 0/5 |
| EMA Alignment [short] | fixed_pct/trailing @ 8% | +5.97% | -5.61% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/static @ 1.5x ATR | -1.42% | -10.70% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/static @ 2.5x ATR | -5.11% | -11.61% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/static @ 4.0x ATR | -4.61% | -10.48% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/trailing @ 1.5x ATR | -2.86% | -9.53% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/trailing @ 2.5x ATR | -0.19% | -7.89% | 0/5 |
| Ichimoku Cloud [long] | atr_multiple/trailing @ 4.0x ATR | -2.76% | -10.12% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/static @ 3% | -2.21% | -9.34% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/static @ 5% | -4.41% | -12.28% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/static @ 8% | -3.51% | -11.73% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/trailing @ 3% | +127.96% | -7.37% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/trailing @ 5% | -4.57% | -13.36% | 0/5 |
| Ichimoku Cloud [long] | fixed_pct/trailing @ 8% | -5.94% | -10.64% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/static @ 1.5x ATR | -2.70% | -10.41% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/static @ 2.5x ATR | -0.85% | -7.05% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/static @ 4.0x ATR | -1.93% | -8.42% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/trailing @ 1.5x ATR | -2.70% | -7.68% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/trailing @ 2.5x ATR | -1.48% | -7.73% | 0/5 |
| Ichimoku Cloud [short] | atr_multiple/trailing @ 4.0x ATR | -1.59% | -7.02% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/static @ 3% | -2.94% | -8.21% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/static @ 5% | -3.35% | -10.65% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/static @ 8% | -2.38% | -7.89% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/trailing @ 3% | +44.92% | +1.11% | 1/5 |
| Ichimoku Cloud [short] | fixed_pct/trailing @ 5% | -3.82% | -12.18% | 0/5 |
| Ichimoku Cloud [short] | fixed_pct/trailing @ 8% | -1.94% | -11.30% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/static @ 1.5x ATR | -2.87% | -5.85% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/static @ 2.5x ATR | -3.67% | -6.14% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/static @ 4.0x ATR | -4.97% | -6.83% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/trailing @ 1.5x ATR | -3.48% | -6.99% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/trailing @ 2.5x ATR | -4.00% | -7.45% | 0/5 |
| MACD Histogram Regime [long] | atr_multiple/trailing @ 4.0x ATR | -6.16% | -7.72% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/static @ 3% | -3.00% | -6.53% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/static @ 5% | -2.57% | -7.45% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/static @ 8% | -3.44% | -5.91% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/trailing @ 3% | +26.78% | -5.31% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/trailing @ 5% | -4.03% | -8.69% | 0/5 |
| MACD Histogram Regime [long] | fixed_pct/trailing @ 8% | -3.01% | -6.66% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/static @ 1.5x ATR | +6.47% | -5.25% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/static @ 2.5x ATR | +5.28% | -6.85% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/static @ 4.0x ATR | +4.41% | -6.28% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/trailing @ 1.5x ATR | +6.27% | -6.85% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/trailing @ 2.5x ATR | +5.41% | -6.85% | 0/5 |
| MACD Histogram Regime [short] | atr_multiple/trailing @ 4.0x ATR | +4.45% | -6.24% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/static @ 3% | -3.14% | -8.27% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/static @ 5% | +4.73% | -7.71% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/static @ 8% | +4.13% | -7.24% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/trailing @ 3% | -1.53% | -12.14% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/trailing @ 5% | -2.69% | -5.96% | 0/5 |
| MACD Histogram Regime [short] | fixed_pct/trailing @ 8% | +5.48% | -6.43% | 0/5 |
| MFI Trend Pullback [long] | atr_multiple/static @ 1.5x ATR | -6.08% | -6.60% | 0/5 |
| MFI Trend Pullback [long] | atr_multiple/static @ 2.5x ATR | -4.38% | -5.87% | 0/5 |
| MFI Trend Pullback [long] | atr_multiple/static @ 4.0x ATR | -4.30% | -13.10% | 0/5 |
| MFI Trend Pullback [long] | atr_multiple/trailing @ 1.5x ATR | -0.15% | -3.90% | 0/5 |
| MFI Trend Pullback [long] | atr_multiple/trailing @ 2.5x ATR | -4.66% | -5.87% | 0/5 |
| MFI Trend Pullback [long] | atr_multiple/trailing @ 4.0x ATR | -4.16% | -12.56% | 0/5 |
| MFI Trend Pullback [long] | fixed_pct/static @ 3% | -2.98% | -9.17% | 0/5 |
| MFI Trend Pullback [long] | fixed_pct/static @ 5% | -2.96% | -6.22% | 0/5 |
| MFI Trend Pullback [long] | fixed_pct/static @ 8% | -5.35% | -5.75% | 0/5 |
| MFI Trend Pullback [long] | fixed_pct/trailing @ 3% | +48.97% | -6.75% | 0/5 |
| MFI Trend Pullback [long] | fixed_pct/trailing @ 5% | +3.25% | -9.19% | 0/5 |
| MFI Trend Pullback [long] | fixed_pct/trailing @ 8% | -1.91% | -7.79% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/static @ 1.5x ATR | +2.50% | -2.48% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/static @ 2.5x ATR | +0.85% | -5.61% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/static @ 4.0x ATR | -1.22% | -7.39% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/trailing @ 1.5x ATR | +0.21% | -9.54% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/trailing @ 2.5x ATR | +0.42% | -5.73% | 0/5 |
| RSI Trend Pullback [long] | atr_multiple/trailing @ 4.0x ATR | -1.26% | -7.39% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/static @ 3% | +1.52% | -6.75% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/static @ 5% | +0.12% | -6.36% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/static @ 8% | -0.22% | -6.27% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/trailing @ 3% | +24.53% | +1.66% | 1/5 |
| RSI Trend Pullback [long] | fixed_pct/trailing @ 5% | +0.36% | -4.15% | 0/5 |
| RSI Trend Pullback [long] | fixed_pct/trailing @ 8% | -1.32% | -6.21% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/static @ 1.5x ATR | -3.82% | -11.74% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/static @ 2.5x ATR | -4.07% | -8.46% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/static @ 4.0x ATR | -7.86% | -11.45% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/trailing @ 1.5x ATR | -4.41% | -9.93% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/trailing @ 2.5x ATR | -3.69% | -9.68% | 0/5 |
| RSI Trend Pullback [short] | atr_multiple/trailing @ 4.0x ATR | -7.09% | -9.58% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/static @ 3% | -5.39% | -9.48% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/static @ 5% | -3.44% | -8.65% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/static @ 8% | -5.00% | -10.45% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/trailing @ 3% | -3.31% | -10.19% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/trailing @ 5% | -2.56% | -5.42% | 0/5 |
| RSI Trend Pullback [short] | fixed_pct/trailing @ 8% | -5.68% | -11.23% | 0/5 |


## Reproduction

Raw per-(candidate, symbol, setting) results: 1,740 JSON-lines records generated via
`research run expert backtest` CLI calls per the command template above, aggregated with `statistics.mean` /
`statistics.median` across the 5 symbols per (candidate, setting) cell. No source code was modified to produce
this sweep - measurement only, per this task's constraints.
