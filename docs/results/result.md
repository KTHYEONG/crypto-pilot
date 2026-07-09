# L0/L1 Discovery Snapshot

- Date: `2026-07-09`
- Run id: `4h_1783585799`
- Command:
  `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. LOG_LEVEL=INFO timeout 1500 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --date 2026-07-09 --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
- Artifacts:
  - `logs/futures/alpha_foundry/4h_1783585799_{1h,2h,4h,6h,8h,12h}_report.json`
  - `logs/futures/alpha_foundry/4h_1783585799_{1h,2h,4h,6h,8h,12h}_evidence.parquet`

## Scope

- Test horizon: `2023-10-31 ~ 2026-06-30`
- IS / OOS split: `2026-01-01`
- Universe funnel:
  - discovered `414`
  - liquidity-selected `150`
  - loaded after integrity checks `25`

## L0 Gate Summary

| TF | Panels In | Bound | Evidence Rows | Gate Passed | Selected for L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1h | 8 | 8 | 8 | 0 | 0 |
| 2h | 10 | 10 | 10 | 0 | 1 |
| 4h | 52 | 44 | 44 | 4 | 3 |
| 6h | 19 | 19 | 19 | 1 | 1 |
| 8h | 19 | 19 | 19 | 1 | 1 |
| 12h | 19 | 18 | 18 | 2 | 2 |
| Total | 127 | 118 | 118 | 8 | 8 |

## Selected L1 Candidates

| TF | Family | Variant | Gate Passed | net_lcb_bps | nw_tstat | cost_drag_ratio |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2h | `btc_regime_pullback` | `btc_pullback_50` | False | 8.97 | 1.16 | 0.222 |
| 4h | `btc_regime_pullback` | `btc_pullback_50_4h` | True | 100.88 | 3.11 | 0.106 |
| 4h | `lsr_oi_regime_filter` | `lsr_oi_gate_42_4h` | True | 55.65 | 1.61 | 0.186 |
| 4h | `btc_regime_pullback` | `btc_pullback_50_rsi_4h` | True | 25.20 | 1.60 | 0.219 |
| 6h | `btc_regime_pullback` | `btc_pullback_50` | True | 16.73 | 1.32 | 0.173 |
| 8h | `trend_pullback_continuation` | `tpc_25_100` | True | 27.16 | 1.50 | 0.176 |
| 12h | `trend_pullback_continuation` | `tpc_33_133` | True | 112.21 | 1.80 | 0.070 |
| 12h | `btc_regime_pullback` | `btc_pullback_100_slow` | False | 4.27 | 1.04 | 0.112 |

## Failure Pattern

- Main reject reasons:
  - `non_positive_lcb`
  - `weak_tstat`
  - `excess_cost_drag`
- `tf_corroboration`:
  - all rows `0.0`
  - all selected rows `0.0`
- Family concentration among selected candidates:
  - `btc_regime_pullback`: `5`
  - `trend_pullback_continuation`: `2`
  - `lsr_oi_regime_filter`: `1`

## Critical Observations

1. Selection diversity is still weak.
   Current L1 handoff is concentrated in three families, and five of eight selections come from one family: `btc_regime_pullback`.

2. Two seed promotions are still not canonical gate passes.
   `2h btc_pullback_50` and `12h btc_pullback_100_slow` reached `selected_for_l1=True` while `gate_passed=False`. They are soft-admitted seed cases, not hard L0 winners.

3. Cross-timeframe corroboration is not functioning as an effective discriminator.
   The code path runs, but this execution produced `tf_corroboration=0.0` everywhere. In practice, the multi-timeframe layer is not adding useful separation yet.

4. `tf_probe` and Alpha Foundry readiness disagree.
   The runtime log reported `No data available for tf=4h/6h/8h/12h` in the TF probe path, but Alpha Foundry generated full per-TF JSON/parquet artifacts in the same run. This is a wiring or readiness-contract mismatch, not a true data absence event.

## Current Conclusion

- The current bottleneck is not lack of raw candidate generation.
  The bottleneck is that most candidates die on the same three axes: weak statistical significance, non-positive lower confidence bound, and cost drag.

- The current naming and observability surface makes diagnosis harder than it should be.
  Terms like `tf_probe`, `evidence`, `report`, and `manifest` are overloaded across unrelated layers, and file writes are separated from terminal diagnostics.

- Next implementation priority should be:
  1. unify naming for TF scan / gate result / persisted artifact concepts,
  2. align TF probe readiness with Alpha Foundry readiness,
  3. make JSON/CSV/parquet-producing paths DEBUG-loggable in a consistent terminal-friendly format.
