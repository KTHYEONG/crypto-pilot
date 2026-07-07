# L0 Alpha Foundry Evidence — Raw Data

- Run id: `4h_1783394043`
- Timeframe: 4h
- Command: `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
- Source: `logs/futures/alpha_foundry/4h_1783394043_evidence.parquet` (28 rows, 20 families)
- 정렬: `mean_net_bps` 내림차순

| family | variant | archetype | n_events | effective_n | mean_gross_bps | mean_cost_bps | mean_net_bps | nw_tstat | block_lcb_bps | rank_ic | cost_drag_ratio | turnover_per_year | bootstrap_lcb_bps | bootstrap_agree | gate_passed | discovery_tier | reject_reasons | hard_reject_reasons | soft_flags | selected_for_l1 | l1_priority_score | l1_budget_units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trend_pullback_continuation | tpc_50_200_4h | trend | 15159 | 15159.0 | 67.2619 | 30.1846 | 37.0773 | 1.1873 | 5.8457 | -0.0924 | 0.4488 | 62.2939 | -14.0956 | False | False | seed | weak_tstat | | weak_tstat\|bootstrap_disagree | True | 13.6536 | 1 |
| mtf_breakout_retest | mtf_bor_20_4h | trend | 11410 | 11410.0 | 53.4975 | 20.3520 | 33.1454 | 1.5547 | 11.9516 | 0.0078 | 0.3804 | 46.8158 | 0.6837 | True | True | seed | | | weak_rank_ic | True | 12.0751 | 1 |
| lsr_oi_regime_filter | lsr_oi_gate_42_4h | hedge | 1764 | 1764.0 | 64.2528 | 34.3252 | 29.9276 | 1.0994 | 2.6383 | -0.0318 | 0.5342 | 132.3275 | -12.6602 | False | False | seed | weak_tstat | | weak_tstat\|bootstrap_disagree\|below_conviction_floor | True | 9.4606 | 1 |
| mtf_breakout_retest | mtf_bor_40_4h | trend | 7366 | 7366.0 | 44.9978 | 20.7430 | 24.2548 | 0.8900 | -3.0733 | 0.0147 | 0.4610 | 30.2133 | -15.2238 | True | False | blocked | non_positive_lcb\|weak_tstat | deep_negative_lcb | weak_tstat | False | 3.7587 | 0 |
| trend_ma | ema_18_108 | trend | 6962 | 6962.0 | 35.1545 | 19.8272 | 15.3273 | 0.4731 | -16.9131 | -0.0375 | 0.5640 | 28.5204 | -29.2790 | True | False | blocked | non_positive_lcb\|weak_tstat | deep_negative_lcb | weak_tstat | False | -8.8530 | 0 |
| mtf_trend_pullback | mtf_tpb_20_30_4h | trend | 8765 | 8765.0 | 29.3110 | 21.5817 | 7.7293 | 0.2318 | -22.3027 | 0.0747 | 0.7363 | 35.9691 | -42.4465 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -14.7947 | 0 |
| mtf_trend_pullback | mtf_tpb_50_30_4h | trend | 11985 | 11985.0 | 25.1563 | 20.5747 | 4.5816 | 0.1288 | -23.9016 | 0.0540 | 0.8179 | 49.1818 | -41.0724 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -16.7808 | 0 |
| trend_pullback_continuation | tpc_20_100_4h | trend | 28049 | 28049.0 | 21.7844 | 19.4649 | 2.3195 | 0.2144 | -8.5264 | -0.0273 | 0.8935 | 115.1651 | -13.3861 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -5.8149 | 0 |
| trend_ma | ema_12_72 | trend | 10579 | 10579.0 | 16.5631 | 20.3220 | -3.7589 | -0.1956 | -27.5299 | -0.0565 | 1.2269 | 43.3952 | -40.5609 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -21.5872 | 0 |
| residual_reversion | rr_24_4h | hedge | 26985 | 26985.0 | 7.9969 | 13.8122 | -5.8153 | -1.0689 | -11.2581 | -0.0294 | 1.7272 | 110.7041 | -14.6424 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -9.8974 | 0 |
| dual_momentum | dm_24_96_4h | trend | 28413 | 28413.0 | 21.4400 | 29.8665 | -8.4265 | -0.3370 | -32.2753 | -0.0209 | 1.3930 | 116.8272 | -46.8194 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -26.3131 | 0 |
| macd_4h | macd_12_26_9 | trend | 41038 | 41038.0 | 11.5866 | 20.1605 | -8.5739 | -0.9065 | -17.9175 | -0.0290 | 1.7400 | 168.5247 | -24.4701 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -15.5816 | 0 |
| btc_regime_pullback | btc_pullback_50_4h | trend | 18638 | 18638.0 | 15.7391 | 24.9281 | -9.1889 | -0.3242 | -38.3472 | -0.0095 | 1.5838 | 76.5305 | -50.6201 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -31.0576 | 0 |
| funding_extreme_reversal | fer_168_4h | flow | 16353 | 16353.0 | -9.1037 | 1.3927 | -10.4963 | -0.4475 | -33.9385 | 0.0506 | 0.1530 | 67.1837 | -45.6004 | True | False | blocked | non_positive_lcb\|weak_tstat | deep_negative_lcb | weak_tstat | False | -28.0779 | 0 |
| dual_momentum | dm_12_48_4h | trend | 40376 | 40376.0 | 8.6589 | 20.2354 | -11.5765 | -1.0644 | -22.3660 | 0.0096 | 2.3369 | 165.8161 | -30.8665 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -19.6686 | 0 |
| ichimoku_trend | ichi_9_26 | trend | 45205 | 45205.0 | 14.4404 | 27.3644 | -12.9241 | -0.8601 | -28.1508 | -0.0243 | 1.8950 | 8.5997 | -36.2168 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -24.3441 | 0 |
| residual_reversion | rr_48_4h | hedge | 27201 | 27201.0 | 0.6660 | 15.0539 | -14.3879 | -1.4177 | -24.5346 | -0.0141 | 22.6032 | 111.6398 | -28.5426 | True | False | blocked | non_positive_lcb\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | | False | -21.9979 | 0 |
| taker_imbalance_momentum | tim_12_4h | trend | 41552 | 41552.0 | 4.9104 | 20.4982 | -15.5878 | -2.6099 | -21.3194 | -0.0130 | 4.1745 | 256.2057 | -25.7045 | True | False | blocked | non_positive_lcb\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | | False | -19.8865 | 0 |
| taker_imbalance_momentum | tim_24_4h | trend | 42991 | 42991.0 | 4.4378 | 21.0059 | -16.5681 | -2.8317 | -22.2313 | -0.0136 | 4.7334 | 265.0635 | -25.5626 | True | False | blocked | non_positive_lcb\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | | False | -20.8155 | 0 |
| supertrend | st_10_4h | trend | 16347 | 16347.0 | 5.7279 | 23.4742 | -17.7463 | -0.8012 | -39.6051 | -0.0545 | 4.0982 | 67.1201 | -52.5325 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -34.1404 | 0 |
| xs_flow | xs_flow_24_4h | cross_sectional | 147324 | 147324.0 | -2.1522 | 20.3263 | -22.4785 | -13.7439 | -24.0930 | -0.0128 | 9.4442 | 908.3113 | -25.5327 | True | False | blocked | non_positive_lcb\|excess_cost_drag\|excess_turnover | excess_cost_drag\|excess_turnover\|deep_negative_lcb | | False | -23.6894 | 0 |
| vol_term_structure_gate | vts_gate_20_4h | trend | 32985 | 32985.0 | 3.5781 | 26.6493 | -23.0712 | -0.9164 | -48.2283 | -0.0312 | 7.4479 | 135.4284 | -57.4346 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -41.9390 | 0 |
| xs_momentum | xs_mom_48 | cross_sectional | 34404 | 34404.0 | 3.0034 | 27.6735 | -24.6702 | -4.2099 | -30.5177 | -0.0278 | 9.2142 | 140.9769 | -34.4864 | True | False | blocked | non_positive_lcb\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | | False | -29.0558 | 0 |
| xs_momentum | xs_mom_12 | cross_sectional | 65072 | 65072.0 | -6.4068 | 24.1834 | -30.5902 | -9.5272 | -33.7630 | -0.0396 | 3.7746 | 267.0400 | -35.5569 | True | False | blocked | non_positive_lcb\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | | False | -32.9698 | 0 |
| trend_donchian | donchian_72_4h | trend | 14660 | 14660.0 | -4.7883 | 29.8878 | -34.6762 | -0.7022 | -86.9637 | 0.0368 | 6.2418 | 60.3855 | -111.2376 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -73.8918 | 0 |
| xs_oi_skew | xs_oi_42_4h | cross_sectional | 6058 | 6058.0 | -6.9691 | 30.5715 | -37.5406 | -3.6041 | -48.9626 | 0.0194 | 4.3867 | 453.6944 | -54.9892 | True | False | blocked | non_positive_lcb\|excess_cost_drag\|excess_turnover | excess_cost_drag\|excess_turnover\|deep_negative_lcb | | False | -46.1071 | 0 |
| vol_breakout | bb_compress_20_4h | trend | 5705 | 5705.0 | -27.7031 | 26.8699 | -54.5730 | -2.1311 | -82.2660 | -0.0237 | 0.9699 | 23.4500 | -100.3154 | True | False | blocked | non_positive_lcb\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | | False | -75.3428 | 0 |
| funding_flow_carry | ffc_96_4h | carry | 77 | 77.0 | -75.5862 | 75.1878 | -150.7740 | -0.9377 | -277.4220 | 0.1841 | 0.9947 | 0.4751 | -277.4220 | True | False | blocked | non_positive_lcb\|weak_tstat\|excess_cost_drag | excess_cost_drag\|deep_negative_lcb | weak_tstat | False | -245.7600 | 0 |

## 필드 정의

- `mean_gross_bps`: 비용 차감 전 건당 평균 수익(bps)
- `mean_cost_bps`: 건당 평균 비용(bps, round-trip+funding)
- `mean_net_bps` = `mean_gross_bps` - `mean_cost_bps`
- `nw_tstat`: block-bootstrap 기반 Newey-West t-stat
- `block_lcb_bps`: block mean의 lower confidence bound(1-sigma)
- `rank_ic`: 신호 점수와 forward return의 Spearman 순위상관(비용 무관, 부호와 크기 모두 중요)
- `cost_drag_ratio`: 비용/|gross|. **1.0 초과 시 부호를 바꿔도 net이 음수가 될 수밖에 없음(수학적으로 증명됨)**
- `turnover_per_year`: 연간 거래 횟수
- `l1_priority_score`: L1 예산 배분 우선순위(음수 가능, 상대값)
- 제거된 family 7종(flow_exhaustion_reversal, funding_carry, funding_flow_unwind, funding_term_structure_carry, positioning_unwind, xs_carry, flow_trend_continuation)은 이 표에 없음 — 코드에서 완전 삭제됨(`docs/decisions/decisions.md` 참조)
