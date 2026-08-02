# Timeframe Census — Cross-Timeframe Comparison & Strategy Prune Recommendation (v2, scaled)

**This supersedes the first census pass.** After that pass, an audit found every indicator/router/activity-floor bar-count parameter was fixed regardless of `--timeframe` (e.g. "200 EMA regime" = 8.3 days at 1h vs. 200 days at 1d) — a structural confound that invalidated cross-timeframe comparison. The `timeframe_invariant_lookback_scaling` spec fixed this (calendar-time-invariant bar-count scaling, 4h reference, verified zero regression at 4h). This document reflects the **scaled** re-run; the raw v1 numbers are kept in each `timeframe-{tf}.md` file's git history for reference.

**Scope**: same as before — `technical-5symbol-rolling` profile (18 families x 5 symbols = 90 candidates), activity-admission screen (`research run expert admission`), 2022-06-01 -> 2025-12-31 pre-holdout. Still measures the activity floor (enough trades to be evaluable), not CAGR/Sharpe directly - `docs/results/rolling-res.md` remains the P&L reference (4h, CAGR +25.73%, LCB90 +0.95%, FAIL on observation reliability).

## 1. Before / after: per-timeframe activity-floor pass rate

| Timeframe | v1 (unscaled) | v2 (scaled) | Change |
| :--- | :--- | :--- | :--- |
| 1h | 79/85 (93%) | 55/85 (65%) | -28pp |
| 2h | 75/85 (88%) | 65/85 (76%) | -12pp |
| 4h | 80/90 (89%) | 80/90 (89%) | +0pp |
| 6h | 80/90 (89%) | 79/90 (88%) | -1pp |
| 8h | 64/85 (75%) | 85/90 (94%) | +19pp |
| 12h | 45/90 (50%) | 70/90 (78%) | +28pp |
| 1d | 22/85 (26%) | 65/90 (72%) | +46pp |

**The 12h/1d activity-floor collapse in v1 (50% -> 26%) was almost entirely the lookback-scaling artifact.** Once periods scale to a fixed calendar window, 1d recovers to 65/90 (72%) and 12h to 70/90 (78%) - both now comparable to 4h/6h rather than collapsing. Conversely, **1h/2h now show real degradation** (93%/88% -> 65%/76%) that v1's fixed-bar-count parameters had masked: at true calendar-equivalent lookbacks, several families structurally do not fire often enough at 1h/2h cadence - this is a genuine cadence effect, not an artifact.

## 2. Family x timeframe matrix (v2, scaled; symbols admitted out of 5)

| Family | 1h | 2h | 4h | 6h | 8h | 12h | 1d |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ADX/DI Regime [long] | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| ADX/DI Regime [short] | 0/5 | 1/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| BB Squeeze Breakout [long] | 5/5 | 5/5 | 5/5 | 3/5 | 2/5 | 0/5 | 0/5 |
| BB Squeeze Breakout [short] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| CCI Trend Pullback [long] | 2/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| CCI Trend Pullback [short] | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| EMA Alignment [long] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| EMA Alignment [short] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Ichimoku Cloud [long] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Ichimoku Cloud [short] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| MACD Histogram Regime [long] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| MACD Histogram Regime [short] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| MFI Trend Pullback [long] | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 |
| MFI Trend Pullback [short] | 0/5 | 0/5 | 0/5 | 1/5 | 3/5 | 5/5 | 0/5 |
| RSI Trend Pullback [long] | 0/5 | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| RSI Trend Pullback [short] | excl. | excl. | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Stochastic Trend Pullback [long] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| Stochastic Trend Pullback [short] | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 |

## 3. Strategy prune recommendation (final — exhaustive real P&L verification)

**Every candidate that passed the §2 activity floor was backtested for real P&L** — not a sample: all 499 admitted (family, side, symbol, timeframe) combinations across all 18 candidates and all 7 timeframes, via `research run expert backtest` (net fees/slippage, sealed observation reliability gate). **Result: 0/499 passed the observation gate.**

### Zero standalone edge, full stop

Every one of the 18 candidates — including EMA Alignment, Ichimoku Cloud, and MACD Histogram Regime, which the earlier activity-floor-only pass mischaracterized as the "most robust" tier — shows a negative or flat mean CAGR at every timeframe where it was tested. There is no family, side, or timeframe combination with genuine single-leg edge in this universe. The table below is the full per-family, all-timeframe summary (mean CAGR / median LCB90 CAGR across the admitted symbols at each timeframe; `-` = not admitted at that timeframe, so never backtested).

| Family | 1h | 2h | 4h | 6h | 8h | 12h | 1d |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ADX/DI Regime [long] | - | - | -3.3% | -0.8% | -2.5% | -3.3% | -3.4% |
| ADX/DI Regime [short] | - | +0.2% | -1.2% | -4.9% | -4.5% | -4.5% | -6.4% |
| BB Squeeze Breakout [long] | -5.5% | -0.7% | -0.7% | -1.1% | +0.6% | - | - |
| BB Squeeze Breakout [short] | -6.9% | -4.4% | -5.5% | -1.9% | -0.7% | - | - |
| CCI Trend Pullback [long] | -1.1% | -0.1% | +2.3% | +0.6% | +0.2% | -2.6% | -1.4% |
| CCI Trend Pullback [short] | -0.6% | -0.9% | -0.5% | -2.7% | -2.1% | -1.1% | -0.8% |
| EMA Alignment [long] | -8.4% | -3.6% | -3.4% | -3.5% | -0.6% | -2.0% | -1.4% |
| EMA Alignment [short] | -5.2% | -3.0% | -4.5% | -3.1% | -1.1% | -3.6% | -5.6% |
| Ichimoku Cloud [long] | -11.4% | -7.2% | -5.3% | -4.2% | +0.3% | -2.5% | -1.9% |
| Ichimoku Cloud [short] | -7.1% | -4.4% | -7.9% | -1.6% | -4.5% | -6.1% | -1.2% |
| MACD Histogram Regime [long] | -11.6% | -4.8% | -4.6% | -2.0% | -0.7% | -4.6% | -3.8% |
| MACD Histogram Regime [short] | -9.2% | -5.9% | -3.8% | -3.8% | -3.5% | -2.2% | -3.9% |
| MFI Trend Pullback [long] | - | - | - | - | -1.1% | -1.6% | -1.6% |
| MFI Trend Pullback [short] | - | - | - | +0.0% | -0.2% | -1.2% | - |
| RSI Trend Pullback [long] | - | -8.8% | -5.0% | -2.5% | -4.6% | -5.1% | +0.8% |
| RSI Trend Pullback [short] | - | - | -2.2% | -10.1% | -4.5% | -1.5% | -7.3% |
| Stochastic Trend Pullback [long] | -2.7% | -5.0% | -1.4% | -1.9% | -1.7% | - | - |
| Stochastic Trend Pullback [short] | -2.6% | -3.3% | -0.6% | -1.0% | -1.3% | - | - |

### What this means for the catalog

**None of the 18 technical-expert candidates should be deployed standalone at any timeframe.** This is not a call to delete the catalog wholesale, though: `docs/results/rolling-res.md`'s positive blended-portfolio CAGR (+25.73%, 4h) comes from the diversified multi-candidate construction — correlation/joint-negative screening plus per-symbol-winner routing across many simultaneously-held legs, which is a fundamentally different object than any one leg's standalone P&L. A leg with negative-to-flat standalone expectancy can still be a useful diversifier if it is weakly/negatively correlated with the rest of the blend — that is the actual design of `technical-5symbol-rolling`, and this exhaustive single-leg check does not invalidate it. What it does invalidate is treating any individual family's activity-floor pass as "this one works" — none of them do, alone.

This result is also consistent with, and now fully substantiates, the CASH-heavy state of the committed rolling ledger noted in §4/earlier reports (0 experts selected in 8 of the most recent quarters): the underlying single-leg technical signals genuinely lack standalone edge in this universe. That is not a bug in the admission gate — it is the honest state of the strategy library, matching this project's broader multi-cycle research history of edge-absence findings.

### Practical next step

The only path to a positive standalone result from this catalog is a genuinely different signal family (TradingView/prop-desk candidates not yet in the catalog — Supertrend, Parabolic SAR, Keltner Channel breakout), or accepting that this catalog's role is diversification inputs to the blended portfolio rather than standalone strategies, and focusing further validation effort on the blend construction (correlation screen, routing, sizing) rather than on any individual family.

## 4. What did NOT change

- The dynamic symbol universe verification (BNB -> DOGE swap at the 2026-07-07 snapshot) is unaffected by this fix and still holds.

- The known CASH-heavy state of the committed 4h rolling ledger (0 experts in the 8 most recent quarters) is unaffected and still needs separate investigation — and is now further corroborated by this section's finding that standalone technical edge is absent across the families tested.

## 5. Implementation note: a second timeframe-scaling gap found and fixed mid-session

The original `timeframe_invariant_lookback_scaling` spec wired calendar-time-invariant scaling into the `research run
expert admission` command's code path (`application/research/expert/admission.py`) but missed a second, separate code
path used by `research run expert backtest` (`application/research/expert/admission_backtest.py`), which still resolved
technical candidates and the router spec unscaled. This was caught before running the edge verification above (which
depends on `backtest`), fixed with the same `resolve_technical_candidate(..., timeframe=)` / `scale_router_spec(...)`
pattern, and verified byte-identical at the 4h reference timeframe before use. A third code path
(`application/research/expert/rolling_admission.py`, used by `research run expert rolling`) has the same unscaled router
gap in 6+ call sites and was deliberately **not** patched in this session (a partial fix there risked an internally
inconsistent state); it remains a known follow-up before any `rolling` walk-forward is run at a non-4h timeframe.
