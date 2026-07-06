# L0/L1 Signal Discovery Real Run Report

- Run date: 2026-07-06
- Run id: `4h_1783337608`
- Command: `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
- Artifacts:
  - `logs/futures/alpha_foundry/4h_1783337608_report.json`
  - `logs/futures/alpha_foundry/4h_1783337608_evidence.parquet`

## Executive Summary

| Item | Value |
|---|---:|
| Universe loaded | 137 symbols |
| L1 admitted scope | 126 symbols |
| Alpha Foundry mode | gate |
| L0 panels in | 36 |
| L0 bound panels | 36 |
| L0 report passed | 3 |
| L0 report rejected | 33 |
| L0 elapsed | 1.25 sec |
| L1 final promoted, AF-gated block | 5 pairs |
| L1 final promoted, 6h block | 60 pairs |
| L1 final promoted, 8h block | 54 pairs |
| L1 final promoted, 12h block | 105 pairs |

TF probe audit still reported 0 winning cells across 4h/6h/8h/12h. The L1 run itself remained healthy and reached `4/4` fold readiness for the main blocks.

## L0 Evidence Snapshot

| Field | Count |
|---|---:|
| Evidence rows | 36 |
| `selected_for_l1=True` rows | 3 |
| `l1_budget_units > 0` rows | 3 |
| `discovery_tier=blocked` | 33 |
| `discovery_tier=seed` | 2 |
| `discovery_tier=candidate` | 1 |
| `gate_passed=True` | 1 |
| synthetic recipe source | 36 |
| cross-TF corroborated | 0 |
| insufficient TF coverage | 36 |

Reject reason counts:

| Reason | Count |
|---|---:|
| non_positive_lcb | 33 |
| excess_cost_drag | 27 |
| weak_tstat | 22 |
| excess_turnover | 2 |

Hard reject counts:

| Reason | Count |
|---|---:|
| deep_negative_lcb | 33 |
| excess_cost_drag | 27 |
| excess_turnover | 2 |

Soft flags:

| Flag | Count |
|---|---:|
| weak_tstat | 22 |
| bootstrap_disagree | 2 |
| below_conviction_floor | 1 |

## L0 Handoff Result

The handoff invariant is now aligned across candidate, evidence, and bridge layers.

| Family | Variant | Archetype | Tier | Mean net bps | LCB bps | t-stat | Soft flags | Priority | Budget |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| mtf_breakout_retest | mtf_bor_20_4h | trend | candidate | 33.15 | 11.95 | 1.55 |  | 17.25 | 1 |
| trend_pullback_continuation | tpc_50_200_4h | trend | seed | 37.08 | 5.94 | 1.19 | weak_tstat, bootstrap_disagree | 13.73 | 1 |
| lsr_oi_regime_filter | lsr_oi_gate_42_4h | hedge | seed | 29.93 | 2.73 | 1.10 | weak_tstat, bootstrap_disagree, below_conviction_floor | 9.53 | 1 |

Key checks:

- `selected_for_l1=True` rows: `3`
- `selected_for_l1=True` blocked rows: `0`
- `selected_for_l1=True` non-blocked rows: `3`
- `l1_budget_units > 0` blocked rows: `0`
- `report.n_passed == 3`
- `parquet selected_for_l1 == 3`

This confirms the previous bug is fixed:

- hard-rejected rows no longer receive L1 budget
- bridge mode no longer forwards blocked rows through a fallback path
- `min_conviction_lcb_bps` now behaves as a soft exploration floor, not a hard reject

## L1 Context

Main L1 readiness remained positive:

| Block | Folds ready | Promoted pairs | Top families observed |
|---|---:|---:|---|
| AF-gated first block | 3/4 | 5 | trend_pullback_continuation |
| 6h main block | 4/4 | 60 | trend_donchian, trend_pullback_continuation |
| 8h main block | 4/4 | 54 | trend_donchian, trend_pullback_continuation, dual_momentum |
| 12h main block | 4/4 | 105 | trend_donchian, dual_momentum |

The gate path is now meaningfully less noisy:

- earlier run: `selected_for_l1=True` 9, including 6 blocked rows
- current run: `selected_for_l1=True` 3, including 0 blocked rows

## Conclusion

The L0 handoff guard is now behaving as intended.

- hard rejects fail closed
- soft seeds survive when economically meaningful
- report/parquet/bridge are consistent
- the 4h run still yields only 3 L0 handoff candidates, but they are now valid candidates instead of leaked blockers

The remaining low count is a policy outcome, not an invariant failure. If more L0 flow is desired, the next tuning lever is gate strictness and timeframes, not the handoff wiring itself.
