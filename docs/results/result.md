# L0/L1 Discovery Snapshot

- Date: `2026-07-11`
- Run id: `4h_1783736185`
- Command:
  `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. LOG_LEVEL=DEBUG timeout 1500 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --date 2026-07-11 --trials 1 --seed 42`
- Context: acceptance run for `[ADR_20260711_L0_ENTRY_EXIT_SIGNAL_EFFECTIVENESS_REDESIGN]` (barrier-aware evaluation, rising-edge sparse triggers, rolling-stat/entry-logic bug fixes, catalog cleanup). Supersedes the `2026-07-09` snapshot below in scope (this run additionally fixed a real-data crash the prior snapshot never hit).

## Scope

- Test horizon: `2023-10-31 ~ 2026-06-30`
- IS / OOS split: `2026-01-01`
- Universe funnel:
  - discovered `414`
  - liquidity-selected (capacity limit, Top-N) `150`
  - loaded after integrity checks `137`
  - admitted into L1 (post late-start drop) `126` (dropped `11`: `late_start`)
- 4h panel funnel (only TF with a logged panels_in/bound breakdown): `n_panels_in=54 → n_bound=46 → n_passed(cheap gate)=16 → n_rejected=30`

## L0 Gate Summary (real per-TF evidence, post barrier-aware fix)

| TF | Evidence Rows | Distinct Families | Gate Passed | Selected for L1 |
| --- | ---: | ---: | ---: | ---: |
| 1h | 8 | 4 | 0 | 0 |
| 2h | 10 | 4 | 0 | 0 |
| 4h | 46 | 28 | 16 | 13 |
| 6h | 19 | 11 | 0 | 0 |
| 8h | 19 | 11 | 0 | 0 |
| 12h | 18 | 10 | 0 | 0 |
| Total | 120 | - | 16 | 13 |

## Selected L1 Candidates (all from 4h; sorted by net_lcb_bps)

| Family | Variant | net_lcb_bps | nw_tstat | cost_drag_ratio | n_events |
| --- | --- | ---: | ---: | ---: | ---: |
| `trend_pullback_continuation` | `tpc_100_400_4h` | 96.78 | 6.19 | 0.091 | 7,287 |
| `trend_pullback_continuation` | `tpc_50_200_4h` | 77.18 | 7.43 | 0.119 | 12,188 |
| `trend_pullback_quality_v2` | `tpq_v2_100_400_4h` | 67.50 | 4.35 | 0.118 | 3,915 |
| `mtf_trend_pullback` | `mtf_tpb_100_30_4h` | 52.68 | 5.22 | 0.157 | 13,727 |
| `trend_pullback_quality_v2` | `tpq_v2_50_200` | 51.81 | 4.70 | 0.154 | 6,304 |
| `dual_momentum` | `dm_24_96_4h` | 51.78 | 6.62 | 0.167 | 28,510 |
| `vol_term_structure_gate` | `vts_gate_20_4h` | 42.30 | 5.33 | 0.190 | 33,675 |
| `macd_4h` | `macd_12_26_9` | 39.07 | 7.97 | 0.219 | 41,036 |
| `dual_momentum` | `dm_12_48_4h` | 37.12 | 7.47 | 0.225 | 40,642 |
| `trend_pullback_continuation` | `tpc_20_100_4h` | 36.97 | 6.52 | 0.214 | 22,949 |
| `taker_imbalance_momentum` | `tim_12_4h` | 33.41 | 11.41 | 0.258 | 57,304 |
| `trend_pullback_quality_v2` | `tpq_v2_20_100` | 23.05 | 3.37 | 0.268 | 9,204 |
| `oi_lsr_unwind` | `oiu_42` | 15.16 | 2.37 | 0.332 | 1,527 |

## Failure Pattern

- Main reject reasons (all TFs combined):
  - `non_positive_lcb`: 64
  - `weak_tstat`: 54
  - `tf_contradicted`: 39
  - `excess_cost_drag`: 10
  - `insufficient_events`: 14
  - `xs_spread_fail`: 10
  - `excess_turnover`: 8
  - `gross_lcb_below_cost`: 3
  - `non_positive_gross`: 2
  - `insufficient_effective_n`: 1
- 6h/8h/12h collapse pattern: identical reject-reason counts across all three TFs (`tf_contradicted=8, non_positive_lcb=15, weak_tstat=15, insufficient_events=4, xs_spread_fail=2`) — evidence rows are near-duplicated across these TFs, consistent with the same underlying 1h-sourced candles being resampled (per TF-PROBE readiness dashboard: `6h/8h/12h Mix: 1h:296`), not independent signal evaluation.
- L1 nested pairwise stage (downstream of L0, separate mechanism): even the 4h-selected set collapses to `0 qualified` pairs per as-of snapshot (`no_incremental_edge`, `negative_gross_edge` dominant), and final `L1_NESTED` fold readiness is `BLOCKED (fold_ratio 0.25<0.50)`. This is not part of the L0 entry/exit fix scope.

## Critical Observations

1. Barrier-aware evaluation (Fix 1) materially changed the 4h outcome shape.
   Compared to the `2026-07-09` snapshot (8 total selections, `net_lcb_bps` capped at ~112), this run produced **13 selections at 4h alone**, spanning **8 distinct families** (trend-pullback variants, `mtf_trend_pullback`, `dual_momentum`, `vol_term_structure_gate`, `macd_4h`, `taker_imbalance_momentum`, `oi_lsr_unwind`) instead of 5/8 concentrated in `btc_regime_pullback`. This is real breadth improvement at 4h, not just a numerical-stability fix.

2. The real-data crash found and fixed this session (`xs_spread_lcb_bps must be finite`) is confirmed resolved.
   All 6 TFs now produce complete, finite evidence CSVs end-to-end; no exception across the full run (628.94s wall clock).

3. Breadth improvement does not transfer past 4h.
   1h/2h/6h/8h/12h all show `gate_passed=0`. The 6h/8h/12h evidence is suspiciously identical in reject-reason shape, pointing to an HTF-resample data-sourcing artifact (all three built from the same `1h:296` source mix) rather than three independently-evaluated timeframes — a candidate follow-up investigation, separate from this spec's scope.

4. L0 breadth gain does not survive L1's nested pairwise-matching gate.
   Even the 13 4h-selected, gate-passed L0 candidates reduce to 0 qualified pairs in L1's own prequential evidence stage (`no_incremental_edge`/`negative_gross_edge`). This is a distinct downstream mechanism from the L0 entry/exit redesign and remains unresolved.

## Current Conclusion

- This session's fix (barrier-aware NaN/finite-masking bug) is verified by direct execution: no crash, and a genuine increase in 4h L0 gate-passed breadth/diversity versus the pre-fix baseline.
- The original diagnosis (`gross alpha 부재`, not implementation bug) still holds for 1h/2h/6h/8h/12h — those TFs remain fully blocked on `non_positive_lcb`/`weak_tstat` regardless of the entry-logic corrections.
- Next priority candidates (not yet started):
  1. Investigate why 6h/8h/12h evidence rows are near-identical (HTF resample sourcing artifact vs. genuine independent evaluation).
  2. Root-cause L1's nested pairwise `no_incremental_edge`/`negative_gross_edge` collapse — the mechanism that currently discards all L0 4h breadth gains before deployment.
