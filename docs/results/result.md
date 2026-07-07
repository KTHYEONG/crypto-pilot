# L0 Alpha Foundry Evidence - Run Comparison

- New run id: `4h_1783404539`
- Previous baseline: `4h_1783394043`
- Timeframe: `4h`
- Command: `UV_CACHE_DIR=/tmp/uv-cache LOG_LEVEL=DEBUG PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
- Source: `logs/futures/alpha_foundry/4h_1783404539_evidence.parquet` (34 rows, 23 families)
- Runtime log: `/tmp/alpha_foundry_result_run.log`
- Exit: `0`

## Executive Summary

이번 구현 후 L0 후보 수는 `28 -> 34`, family 수는 `20 -> 23`으로 늘었다. 하지만 strict `gate_passed=True`는 여전히 `1`개뿐이고, L1 selected도 `3`개로 증가하지 않았다.

가장 중요한 문제는 `selected_for_l1=True`인 3개 중 2개가 새 unified row 기준으로 `handoff_tier=blocked`라는 점이다. 즉, 현재 코드는 "L1로 넘겼다"와 "경제성 gate를 통과했다"의 의미가 아직 완전히 분리되지 않았거나, selection 단계가 blocked 상태를 다시 살리고 있다.

## Baseline Comparison

| metric | old | new | delta |
| --- | --- | --- | --- |
| rows | 28.0000 | 34.0000 | 6.0000 |
| families | 20.0000 | 23.0000 | 3.0000 |
| gate_passed | 1.0000 | 1.0000 | 0.0000 |
| selected_for_l1 | 3.0000 | 3.0000 | 0.0000 |
| positive_net | 8.0000 | 10.0000 | 2.0000 |
| positive_lcb | 3.0000 | 3.0000 | 0.0000 |
| cost_drag_ratio_gt_1 | 17.0000 | 21.0000 | 4.0000 |
| median_mean_net_bps | -11.0364 | -9.8426 | 1.1938 |
| best_mean_net_bps | 37.0773 | 37.0773 | 0.0000 |
| best_net_lcb_bps | 11.9516 | 11.9506 | -0.0011 |

Interpretation:

- Breadth improved: 신규 후보 6개, 신규 family 3개가 추가됐다.
- Quality did not materially improve: strict pass 수는 그대로 1개다.
- Positive average net은 `8 -> 10`으로 증가했지만, uncertainty-adjusted positive LCB는 `3`개로 그대로다.
- Cost death worsened: `cost_drag_ratio > 1` 후보가 `17 -> 21`로 증가했다.

## Newly Added Candidates

| family | variant | mean_net_bps | net_lcb_bps | cost_drag_ratio | gate_passed | handoff_tier | reject_reasons | soft_flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_breakout_retest_v2 | bor_v2_40 | 17.2724 | -0.5514 | 0.5159 | False | blocked | non_positive_lcb\|weak_tstat | weak_tstat |
| trend_pullback_quality_v2 | tpq_v2_50_200 | 13.3059 | -24.1293 | 0.6778 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| sparse_breakout_retest_v2 | bor_v2_20 | -0.5623 | -11.0975 | 1.0343 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| trend_pullback_quality_v2 | tpq_v2_20_100 | -3.8950 | -15.3774 | 1.2647 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| residual_momentum_xs | rm_xs_12 | -30.5902 | -33.7617 | 3.7746 | False | blocked | non_positive_lcb\|excess_cost_drag |  |
| residual_momentum_xs | rm_xs_24 | -32.5537 | -37.4293 | 5.3536 | False | blocked | non_positive_lcb\|excess_cost_drag |  |

Interpretation:

- `sparse_breakout_retest_v2 / bor_v2_40` is the only useful new near-pass. It has positive net `17.27bps`, cost drag `0.516`, and only slightly negative net LCB `-0.55bps`.
- `trend_pullback_quality_v2 / tpq_v2_50_200` has positive average net, but the LCB is too negative and cost drag is above the configured max `0.60`.
- `residual_momentum_xs` is structurally bad in the current formulation. It duplicates poor cross-sectional momentum economics with negative gross/net and very high cost drag.

## Full New L0 Evidence

| family | variant | archetype | n_events | effective_n | mean_gross_bps | mean_cost_bps | mean_net_bps | net_lcb_bps | nw_tstat | rank_ic | rank_ic_tstat | cost_drag_ratio | turnover_per_year | gate_passed | handoff_tier | reject_reasons | soft_flags | selected_for_l1 | l1_priority_score | l1_budget_units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trend_pullback_continuation | tpc_50_200_4h | trend | 15159 | 15159.0000 | 67.2619 | 30.1846 | 37.0773 | 5.8462 | 1.1874 | -0.0924 | 0.0000 | 0.4488 | 62.2939 | False | blocked | weak_tstat | weak_tstat\|bootstrap_disagree | True | 13.6540 | 1 |
| mtf_breakout_retest | mtf_bor_20_4h | trend | 11410 | 11410.0000 | 53.4975 | 20.3520 | 33.1454 | 11.9506 | 1.5546 | 0.0078 | 0.0000 | 0.3804 | 46.8158 | True | candidate |  | weak_rank_ic | True | 12.0745 | 1 |
| lsr_oi_regime_filter | lsr_oi_gate_42_4h | hedge | 1764 | 1764.0000 | 64.2528 | 34.3252 | 29.9276 | 2.7460 | 1.1039 | -0.0318 | 0.0000 | 0.5342 | 132.3275 | False | blocked | weak_tstat | weak_tstat\|bootstrap_disagree\|below_conviction_floor | True | 9.5414 | 1 |
| mtf_breakout_retest | mtf_bor_40_4h | trend | 7366 | 7366.0000 | 44.9978 | 20.7430 | 24.2548 | -3.0670 | 0.8902 | 0.0147 | 0.0000 | 0.4610 | 30.2133 | False | blocked | non_positive_lcb\|weak_tstat | weak_tstat | False | 3.7634 | 0 |
| sparse_breakout_retest_v2 | bor_v2_40 | trend | 10186 | 10186.0000 | 35.6772 | 18.4048 | 17.2724 | -0.5514 | 0.9693 | -0.0371 | 0.0000 | 0.5159 | 41.8070 | False | blocked | non_positive_lcb\|weak_tstat | weak_tstat | False | 3.9045 | 0 |
| trend_ma | ema_18_108 | trend | 6962 | 6962.0000 | 35.1545 | 19.8272 | 15.3273 | -16.9131 | 0.4731 | -0.0375 | 0.0000 | 0.5640 | 28.5204 | False | blocked | non_positive_lcb\|weak_tstat | weak_tstat | False | -8.8530 | 0 |
| trend_pullback_quality_v2 | tpq_v2_50_200 | trend | 8063 | 8063.0000 | 41.3010 | 27.9951 | 13.3059 | -24.1293 | 0.3383 | -0.0583 | 0.0000 | 0.6778 | 33.0943 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -14.7705 | 0 |
| mtf_trend_pullback | mtf_tpb_20_30_4h | trend | 8765 | 8765.0000 | 29.3110 | 21.5817 | 7.7293 | -22.3057 | 0.2318 | 0.0747 | 0.0000 | 0.7363 | 35.9691 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -14.7969 | 0 |
| mtf_trend_pullback | mtf_tpb_50_30_4h | trend | 11985 | 11985.0000 | 25.1563 | 20.5747 | 4.5816 | -23.9004 | 0.1288 | 0.0540 | 0.0000 | 0.8179 | 49.1818 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -16.7799 | 0 |
| trend_pullback_continuation | tpc_20_100_4h | trend | 28049 | 28049.0000 | 21.7844 | 19.4649 | 2.3195 | -8.5264 | 0.2144 | -0.0273 | 0.0000 | 0.8935 | 115.1651 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -5.8149 | 0 |
| sparse_breakout_retest_v2 | bor_v2_20 | trend | 17552 | 17552.0000 | 16.4108 | 16.9731 | -0.5623 | -11.0975 | -0.0534 | -0.0086 | 0.0000 | 1.0343 | 72.0408 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -8.4637 | 0 |
| trend_ma | ema_12_72 | trend | 10579 | 10579.0000 | 16.5631 | 20.3220 | -3.7589 | -27.5299 | -0.1956 | -0.0565 | 0.0000 | 1.2269 | 43.3952 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -21.5871 | 0 |
| trend_pullback_quality_v2 | tpq_v2_20_100 | trend | 16836 | 16836.0000 | 14.7131 | 18.6081 | -3.8950 | -15.3774 | -0.3433 | -0.0363 | 0.0000 | 1.2647 | 69.1557 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -12.5068 | 0 |
| residual_reversion | rr_24_4h | hedge | 26985 | 26985.0000 | 7.9969 | 13.8122 | -5.8153 | -11.2540 | -1.0697 | -0.0294 | 0.0000 | 1.7272 | 110.7041 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -9.8943 | 0 |
| dual_momentum | dm_24_96_4h | trend | 28413 | 28413.0000 | 21.4400 | 29.8665 | -8.4265 | -32.2705 | -0.3371 | -0.0209 | 0.0000 | 1.3930 | 116.8272 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -26.3095 | 0 |
| macd_4h | macd_12_26_9 | trend | 41038 | 41038.0000 | 11.5866 | 20.1605 | -8.5739 | -17.9198 | -0.9063 | -0.0290 | 0.0000 | 1.7400 | 168.5247 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -15.5833 | 0 |
| btc_regime_pullback | btc_pullback_50_4h | trend | 18638 | 18638.0000 | 15.7391 | 24.9281 | -9.1889 | -38.3496 | -0.3242 | -0.0095 | 0.0000 | 1.5838 | 76.5305 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -31.0594 | 0 |
| funding_extreme_reversal | fer_168_4h | flow | 16353 | 16353.0000 | -9.1037 | 1.3927 | -10.4963 | -33.9387 | -0.4475 | 0.0506 | 0.0000 | 0.1530 | 67.1837 | False | blocked | non_positive_lcb\|weak_tstat | weak_tstat | False | -28.0781 | 0 |
| dual_momentum | dm_12_48_4h | trend | 40376 | 40376.0000 | 8.6589 | 20.2354 | -11.5765 | -22.3674 | -1.0642 | 0.0096 | 0.0000 | 2.3369 | 165.8161 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -19.6697 | 0 |
| ichimoku_trend | ichi_9_26 | trend | 45205 | 45205.0000 | 14.4404 | 27.3644 | -12.9241 | -28.1540 | -0.8600 | -0.0243 | 0.0000 | 1.8950 | 8.5997 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -24.3465 | 0 |
| residual_reversion | rr_48_4h | hedge | 27201 | 27201.0000 | 0.6660 | 15.0539 | -14.3879 | -24.5306 | -1.4183 | -0.0141 | 0.0000 | 22.6032 | 111.6398 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -21.9950 | 0 |
| taker_imbalance_momentum | tim_12_4h | trend | 41552 | 41552.0000 | 4.9104 | 20.4982 | -15.5878 | -21.3187 | -2.6102 | -0.0130 | 0.0000 | 4.1745 | 256.2057 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -19.8860 | 0 |
| taker_imbalance_momentum | tim_24_4h | trend | 42991 | 42991.0000 | 4.4378 | 21.0059 | -16.5681 | -22.2312 | -2.8318 | -0.0136 | 0.0000 | 4.7334 | 265.0635 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -20.8154 | 0 |
| supertrend | st_10_4h | trend | 16347 | 16347.0000 | 5.7279 | 23.4742 | -17.7463 | -39.6138 | -0.8009 | -0.0545 | 0.0000 | 4.0982 | 67.1201 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -34.1469 | 0 |
| xs_flow | xs_flow_24_4h | cross_sectional | 147324 | 147324.0000 | -2.1522 | 20.3263 | -22.4785 | -24.0894 | -13.7738 | -0.0128 | 0.0000 | 9.4442 | 908.3113 | False | blocked | non_positive_lcb\|excess_cost_drag\|excess_turnover |  | False | -23.6867 | 0 |
| vol_term_structure_gate | vts_gate_20_4h | trend | 32985 | 32985.0000 | 3.5781 | 26.6493 | -23.0712 | -48.2250 | -0.9166 | -0.0312 | 0.0000 | 7.4479 | 135.4284 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -41.9365 | 0 |
| xs_momentum | xs_mom_48 | cross_sectional | 34404 | 34404.0000 | 3.0034 | 27.6735 | -24.6702 | -30.5201 | -4.2081 | -0.0278 | 0.0000 | 9.2142 | 140.9769 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -29.0576 | 0 |
| xs_momentum | xs_mom_12 | cross_sectional | 65072 | 65072.0000 | -6.4068 | 24.1834 | -30.5902 | -33.7617 | -9.5310 | -0.0396 | 0.0000 | 3.7746 | 267.0400 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -32.9688 | 0 |
| residual_momentum_xs | rm_xs_12 | cross_sectional | 65072 | 65072.0000 | -6.4068 | 24.1834 | -30.5902 | -33.7617 | -9.5310 | -0.0396 | 0.0000 | 3.7746 | 267.0400 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -32.9688 | 0 |
| residual_momentum_xs | rm_xs_24 | cross_sectional | 47453 | 47453.0000 | -5.1237 | 27.4300 | -32.5537 | -37.4293 | -6.6493 | -0.0404 | 0.0000 | 5.3536 | 194.6176 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -36.2104 | 0 |
| trend_donchian | donchian_72_4h | trend | 14660 | 14660.0000 | -4.7883 | 29.8878 | -34.6762 | -86.9679 | -0.7021 | 0.0368 | 0.0000 | 6.2418 | 60.3855 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -73.8949 | 0 |
| xs_oi_skew | xs_oi_42_4h | cross_sectional | 6058 | 6058.0000 | -6.9691 | 30.5715 | -37.5406 | -48.9049 | -3.6238 | 0.0194 | 0.0000 | 4.3867 | 453.6944 | False | blocked | non_positive_lcb\|excess_cost_drag\|excess_turnover |  | False | -46.0638 | 0 |
| vol_breakout | bb_compress_20_4h | trend | 5705 | 5705.0000 | -27.7031 | 26.8699 | -54.5730 | -82.2199 | -2.1348 | -0.0237 | 0.0000 | 0.9699 | 23.4500 | False | blocked | non_positive_lcb\|excess_cost_drag |  | False | -75.3082 | 0 |
| funding_flow_carry | ffc_96_4h | carry | 77 | 77.0000 | -75.5862 | 75.1878 | -150.7740 | -277.4220 | -0.9377 | 0.1841 | 0.0000 | 0.9947 | 0.4751 | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat | False | -245.7600 | 0 |

## L1 Runtime Observation

The same command continued into L1 multi-timeframe validation after L0.

| tf | readiness | promoted | probe_lcb_bps | run_l1_sec |
| --- | --- | --- | --- | --- |
| 4h | 3/4 folds ready | 5 | 88.931 | 5.9130 |
| 6h | 4/4 folds ready | 50 | 60.795 | 167.6381 |
| 8h | 4/4 folds ready | 45 | 26.583 | 156.2831 |
| 12h | 4/4 folds ready | 99 | 52.447 | 175.7043 |

Interpretation:

- L0 Alpha Foundry only reported the native 4h gate: `n_panels_in=34`, `n_bound=34`, `n_passed=3`, `n_rejected=31`.
- L1 still promoted many HTF candidates: 6h `50`, 8h `45`, 12h `99`.
- This confirms the core architecture risk remains: HTF candidates can still become L1 promotions without being represented in the L0 evidence table above.

## Implementation Gaps Observed In Runtime

1. `maybe_write_alpha_foundry_report()` is not wired.
   - Runtime still calls `_write_alpha_foundry_report()` directly.
   - Evidence artifacts were written even though `AlphaFoundryRuntimeConfig.artifact_write_enabled` defaults to `False`.

2. DEBUG gate summary is not wired.
   - `LOG_LEVEL=DEBUG` was set.
   - No `[EVAL] stage=af_gate`, `[ALGO] TOP`, or `[DATA] reject_reasons` lines appeared in `/tmp/alpha_foundry_result_run.log`.

3. `selected_for_l1` leaks blocked rows.
   - `trend_pullback_continuation/tpc_50_200_4h`: `gate_passed=False`, `handoff_tier=blocked`, `selected_for_l1=True`.
   - `lsr_oi_regime_filter/lsr_oi_gate_42_4h`: `gate_passed=False`, `handoff_tier=blocked`, `selected_for_l1=True`.
   - Only `mtf_breakout_retest/mtf_bor_20_4h` is both `gate_passed=True` and `selected_for_l1=True`.

## Plain-English Explanation

Think of L0 as the first security checkpoint and L1 as the deeper inspection room.

The new code added more people to the checkpoint line: 34 candidates instead of 28. But the checkpoint still truly clears only one person. Two others are still being sent into the deeper inspection room even though their new badge says `blocked`.

The useful new candidate is `sparse_breakout_retest_v2/bor_v2_40`. It is close to passing: it makes money on average after cost, but its safety margin is still slightly below zero. That means it is promising, but not deployable yet.

The HTF result is the bigger issue. While the 4h L0 gate is strict, 6h/8h/12h L1 still produces many promotions. That means the system can still find apparently strong slower-timeframe candidates, but they are not being audited by the same L0 evidence table. For production capital growth, that is dangerous because the slow-timeframe candidates may be good, but they are entering through a different door.

## Current Conclusion

- Alpha breadth improved.
- Alpha quality did not materially improve yet.
- Best next target is not adding more families blindly.
- Best next target is wiring HTF candidates through the same L0 gate and fixing `selected_for_l1` so blocked rows cannot receive L1 budget.
