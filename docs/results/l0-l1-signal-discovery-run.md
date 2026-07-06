# L0/L1 Signal Discovery Spec 적용 실측 보고

- Run date: 2026-07-06
- Command: `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 600 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
- Artifact:
  - `logs/futures/alpha_foundry/4h_1783333942_report.json`
  - `logs/futures/alpha_foundry/4h_1783333942_evidence.parquet`

## Execution Summary

| Item | Value |
|---|---:|
| Universe loaded | 137 symbols |
| L1 admitted scope | 126 symbols |
| Alpha Foundry mode | gate |
| L0 panels in | 36 |
| L0 bound panels | 36 |
| L0 report passed | 3 |
| L0 report rejected | 33 |
| L0 elapsed | 5.63 sec |
| L1 final promoted, first AF-gated block | 5 pairs |
| L1 final promoted, 6h block | 60 pairs |
| L1 final promoted, 8h block | 54 pairs |
| L1 final promoted, 12h block | 105 pairs |

TF probe audit still reported 0 winning cells across 4h/6h/8h/12h. Main L1 kept existing TF evidence after later validation parity checks.

## L0 Evidence Distribution

| Field | Count |
|---|---:|
| Evidence rows | 36 |
| `selected_for_l1=True` rows | 9 |
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

## Valid L0 Queue Candidates

These 3 rows have no hard reject and match the intended spec direction: send meaningful but not necessarily fully verified candidates to L1.

| Family | Variant | Archetype | Tier | Mean net bps | LCB bps | t-stat | Soft flags | Priority |
|---|---|---|---|---:|---:|---:|---|---:|
| mtf_breakout_retest | mtf_bor_20_4h | trend | candidate | 33.15 | 11.95 | 1.55 |  | 17.25 |
| trend_pullback_continuation | tpc_50_200_4h | trend | seed | 37.08 | 5.85 | 1.19 | weak_tstat, bootstrap_disagree | 13.65 |
| lsr_oi_regime_filter | lsr_oi_gate_42_4h | hedge | seed | 29.93 | 2.59 | 1.10 | weak_tstat, bootstrap_disagree | 9.42 |

Interpretation:

- `mtf_breakout_retest` is the only clean candidate: positive LCB, t-stat pass, no hard or soft flags.
- `trend_pullback_continuation` and `lsr_oi_regime_filter` are useful exploratory seeds: positive mean and LCB, but weak t-stat plus bootstrap disagreement.
- This is aligned with the new L0 design goal: L0 should not require full L1-grade proof before handoff.

## Spec Mismatch Found

`selected_for_l1=True` has 9 rows, but only 3 rows are non-blocked. Six hard-rejected rows still received `l1_budget_units=1`.

| Family | Variant | Archetype | Mean net bps | LCB bps | Hard reject reasons | Priority |
|---|---|---|---:|---:|---|---:|
| btc_regime_pullback | btc_pullback_50_4h | mean_reversion | -55.77 | -89.94 | excess_cost_drag, deep_negative_lcb | -81.40 |
| flow_exhaustion_reversal | fxr_24_4h | flow | -43.75 | -93.10 | excess_cost_drag, deep_negative_lcb | -80.76 |
| funding_carry | funding_24 | carry | -43.40 | -67.31 | deep_negative_lcb | -61.33 |
| mtf_breakout_retest | mtf_bor_40_4h | trend | 24.25 | -3.07 | deep_negative_lcb | 3.76 |
| trend_pullback_continuation | tpc_20_100_4h | trend | 2.32 | -8.53 | excess_cost_drag, deep_negative_lcb | -5.82 |
| xs_carry | xs_carry_96 | cross_sectional | -31.29 | -35.88 | excess_cost_drag, deep_negative_lcb | -34.73 |

Root cause from code path:

- `build_l0_signal_candidate()` correctly marks these as `discovery_tier="blocked"`.
- `allocate_global_l1_budget()` allocates archetype/timeframe seed slots from all bucket results without excluding blocked candidates.
- `run_alpha_foundry_l0_pipeline()` assigns `l1_budget_units=1` when a recipe is in `final_selected` and its bucket has any slot, without checking `discovery_tier != "blocked"`.
- Therefore hard rejects can be selected for L1, which violates the spec constraint: hard reject must fail closed.

## L1 Result Context

The broader L1 run itself passed readiness:

| Block | Folds ready | Promoted pairs | Top families observed |
|---|---:|---:|---|
| AF-gated first block | 3/4 | 5 | trend_pullback_continuation |
| 6h main block | 4/4 | 60 | trend_donchian, trend_pullback_continuation |
| 8h main block | 4/4 | 54 | trend_donchian, trend_pullback_continuation, dual_momentum |
| 12h main block | 4/4 | 105 | trend_donchian, dual_momentum |

Major-symbol gap diagnostics remained:

- BTC/ETH/BNB still show activation gaps in 12h and selected 6h/8h families.
- TF parity reports 1h/2h missing from main L1, even though fast-TF discovery can be enabled by config. This run did not enable `--alpha-foundry-enable-fast-tf`.

## Conclusion

The new spec direction is partially validated:

- L0 now identifies more than pure `gate_passed=True`: 3 useful non-blocked handoff candidates exist.
- Soft seeds are functioning conceptually: weak t-stat/bootstrap disagreement can still reach L1 as intended.
- However, hard-reject exclusion is not enforced in budget selection. This is a production blocker before trusting `selected_for_l1`.

Recommended next implementation fix:

1. Filter `candidate.discovery_tier != "blocked"` before bucket diversity, cross-bucket diversity, and budget allocation.
2. Assign `l1_budget_units > 0` only if `candidate.discovery_tier in {"seed", "candidate", "verified"}`.
3. Set report `n_passed` and parquet `selected_for_l1` from the same invariant: `l1_budget_units > 0 and not blocked`.
4. Add a regression test: hard-rejected archetype seed candidate must not receive L1 budget even when it is the only representative of that archetype.
