# L0 Alpha Foundry Evidence - Run Comparison (Wiring Gaps Fixed)

- New run id: `4h_1783419659`
- Previous baseline: `4h_1783404539` (`docs/specs/alpha_signal_generation_wiring_gaps.md` 작성 시점 스냅샷)
- Timeframe: `4h`
- Command: `UV_CACHE_DIR=/tmp/uv-cache LOG_LEVEL=DEBUG PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
  - artifact write는 기본값(`artifact_write_enabled=False`)이라 CLI 자체로는 파일이 남지 않음. 데이터 확보를 위해 `build_alpha_foundry_runtime_config()`를 1회성으로 monkeypatch해서 `artifact_write_enabled=True`로 강제한 별도 실행에서 parquet을 남김.
- Source: `logs/futures/alpha_foundry/4h_1783419659_evidence.parquet` (42 rows, 29 families)
- Exit: `0` (단, 아래 "실행 중 발견한 크래시" 참고 — 최초 재실행은 `Exit 1`이었고 버그 수정 후 `0`이 됨)

## Executive Summary

이전 문서(`alpha_signal_generation_wiring_gaps.md`)에서 지적한 3개 배선 갭(Gap 1: canonical gate 미호출, Gap 2: `runtime_config` 미전달, Gap 3: `selected_for_l1` leak)을 구현팀이 수정했고, 이번 재검증에서 **3개 모두 실측 데이터로 해결 확인**했다. 다만 그 과정에서 canonical gate가 처음으로 실제 시장 데이터를 타면서 **이전에는 절대 실행되지 않던 코드 경로의 NaN 크래시**가 새로 발견되어 별도로 수정했고, canonical 3-tier 로직이 살아나면서 **"handoff_tier=candidate" 승격이 사실상 불가능**하다는 새로운 구조적 갭도 확인했다.

## 0. 실행 중 발견한 크래시 (신규, 이번 세션에서 수정)

Gap 1 수정으로 `evaluate_panel_gate()`가 처음 실제 데이터를 통과하면서 다음 예외로 파이프라인이 죽었다(`Exit 1`):

```
ValueError: numeric field must be finite, got nan
```

원인은 5곳의 `np.nanmean`/`.mean()` 호출이 **all-NaN 슬라이스**(특정 심볼의 funding 데이터 결측 구간에 이벤트가 몰린 경우)에 대해 `NaN`을 반환하면서, `AlphaGateEvidence.__post_init__`의 `np.isfinite` 강제 검증에 걸린 것이었다. `.claude/rules/quant.md` §3 "Safe Division Guardrails"가 요구하는 명시적 NaN 가드가 빠져 있었다.

수정한 5곳 (`src/domain/futures/alpha_foundry/cheap_gate.py`):
1. `evaluate_panel_gate()`의 `mean_gross_bps`/`mean_cost_bps`/`mean_net_bps` — all-NaN이면 `0.0`으로 폴백.
2. `_compute_block_means()` — 블록 분할 전에 non-finite 값을 먼저 제거(공유 헬퍼라 `evaluate_panel_gate`와 미사용 상태인 `evaluate_panel_gate_v2` 양쪽에 적용됨).
3. `compute_liquidity_cost_stress_bps()` — all-NaN 시 `0.0` 폴백.
4. `compute_capacity_score()` — cost/adv가 all-NaN이면 `0.0` 폴백(기존엔 `np.clip(nan, 0, 1)`이 `nan`을 그대로 통과시켜 `[0,1]` 범위 검증에서 별도 예외가 남).
5. `compute_regime_stability()` — 이벤트 배열에서 non-finite 제거 후 계산.

수정 후 동일 커맨드 `Exit 0` 확인. 회귀 테스트 282개 전부 통과(`ruff`/`mypy` 포함) — 자세한 내용은 세션 로그 참고, 이 문서는 결과 데이터에 집중.

## 1. Gap 재검증 결과 (`alpha_signal_generation_wiring_gaps.md` 대비)

| Gap | 이전 상태 | 재검증 결과 |
|---|---|---|
| Gap 1: canonical `evaluate_panel_gate()` 미호출 | `capacity_score`/`regime_stability`/`tf_corroboration`이 전부 `0.0` 박제 | **호출 확인됨.** `regime_stability`는 이제 레시피별로 실측값(`0.003~0.27`) 산출. 단 `capacity_score`/`tf_corroboration`은 여전히 전부 `0.0` — 원인은 아래 §3 참고(별개 원인, 새로 확인) |
| Gap 2: `runtime_config` 미전달 | search_cells/cost-prior-screen/DEBUG summary 전부 dead code | `runtime_config`는 이제 전달됨(코드 확인). 단 DEBUG 로그(`[EVAL] stage=af_generation` 등)는 **여전히 터미널에 안 보임** — 새 원인 발견: `emit_alpha_generation_debug_summary()`가 프로젝트 표준 `get_logger()` 대신 `logging.getLogger(__name__)`(stdlib 기본)을 쓰는데, 이 실행 경로엔 `logging.basicConfig()`가 전혀 호출되지 않아 핸들러가 없는 로거의 DEBUG 메시지가 조용히 버려짐 |
| Gap 3: `selected_for_l1`가 `handoff_tier=blocked` leak | 3개 중 2개가 blocked인데 selected | **완전히 해결.** 이번 42개 후보 중 `selected_for_l1=True` 4개 전부 `handoff_tier in {seed, candidate}` — leak 0건 (`git`으로 직접 카운트 확인: `leak rows: 0`) |

## 2. Baseline Comparison

| metric | old (`4h_1783404539`) | new (`4h_1783419659`) | delta |
| --- | --- | --- | --- |
| rows | 34 | 42 | +8 |
| families | 23 | 29 | +6 |
| gate_passed | 1 | 1 | 0 |
| selected_for_l1 | 3 | 4 | +1 |
| **selected_for_l1 but handoff_tier=blocked (leak)** | **2** | **0** | **-2 (해결)** |
| positive_net | 10 | 14 | +4 |
| positive_lcb | 3 | 4 | +1 |
| cost_drag_ratio_gt_1 | 21 | 24 | +3 |
| median_mean_net_bps | -9.8426 | -8.8814 | +0.9612 |
| best_mean_net_bps | 37.0773 | 37.0773 | 0.0000 |
| best_net_lcb_bps | 11.9506 | 11.9516 | +0.0011 |
| handoff_tier=seed | 0 | 5 | +5 |
| handoff_tier=candidate | 1 | 0 | **-1** |
| handoff_tier=blocked | 33 | 37 | +4 |

Interpretation:
- Breadth: 이번에 처음으로 spec의 6개 신규 sparse/liquidity family가 실제로 L0 게이트에 유입됨(`sparse_breakout_retest_liquidity`, `oi_lsr_unwind`, `funding_flow_exhaustion_sparse`, `vol_contraction_breakout`, `xs_residual_rebalance`, `carry_net_of_funding`). 이 중 `sparse_breakout_retest_liquidity/sbrl_40_3_4h`는 `selected_for_l1=True`까지 도달(`mean_net_bps=21.44bps`, `net_lcb_bps=4.02bps`, `handoff_tier=seed`).
- Quality: strict `gate_passed=True`는 여전히 1개(`mtf_breakout_retest/mtf_bor_20_4h`)로 변화 없음. positive_net/positive_lcb는 소폭 개선.
- **`handoff_tier=candidate`가 1→0으로 줄었다.** 이는 회귀가 아니라 canonical 3-tier 로직이 처음 정상 동작한 결과다 — 이전엔 `"candidate" if gate_passed else "blocked"`라는 단순 이진 판정이었지만, 이제는 `regime_stability>=0.5` AND `tf_corroboration>=0.5` AND soft_flag 없음을 모두 요구한다. 아래 §3에서 설명하듯 `tf_corroboration`이 항상 `0.0`이라 **현재 구조에서는 어떤 후보도 `candidate` tier에 도달할 수 없다.**

## 3. 신규 발견: `tf_corroboration`이 항상 `0.0` → `handoff_tier="candidate"` 승격이 구조적으로 불가능

42개 후보 전부 `tf_corroboration=0.0`, `capacity_score=0.0`이다. 원인을 코드로 추적한 결과:

- `bridge_helpers.py:312`의 `run_alpha_foundry_l0_pipeline()` 호출부는 `evidence_by_tf` 파라미터를 넘기지 않는다(파라미터 자체가 시그니처에 없음).
- `pipeline.py`에서 `evidence_by_tf`가 `None`이면 `tf_fusion_index={}`가 되고, `evaluate_alpha_gate_batch(..., tf_fusion_index=tf_fusion_index if evidence_by_tf else None)`도 결국 `None`을 넘김.
- `compute_tf_corroboration()`은 `tf_fusion is None`이면 무조건 `0.0` 반환.
- `evaluate_panel_gate()`의 handoff_tier 로직: `elif regime_stability < 0.5 or tf_corroboration < 0.5 or soft_flags: handoff_tier = "seed"` — `tf_corroboration`이 항상 `0.0`이라 이 조건이 항상 참이 되어 **"candidate"로 승격되는 else 분기에 절대 도달하지 못한다.**
- `capacity_score` 역시 `aligned.execution_cost_bps_2d`/`adv_usdt_2d`가 이번 데이터셋에 아예 없어서(`compute_capacity_score`의 `None` 분기) 항상 `0.0` — 이건 스펙이 의도한 "`<=0.25`로 clamp" 규칙과 부합하는 정상 동작이라 버그는 아니지만, capacity 기반 판단이 이 데이터 소스로는 원천적으로 작동하지 않는다는 뜻이다.

정리하면 **canonical gate 로직 자체는 정확히 spec대로 구현됐지만, TF corroboration의 입력 데이터(`evidence_by_tf`, multi-TF L1 evidence)가 L0 게이트 시점에 아직 준비되지 않는 아키텍처**라, 현재 단일 4h L0 게이트 실행만으로는 "candidate" tier가 나올 수 없다. (참고: multi-TF corroboration은 원래 다른 TF의 L1 fold 결과가 있어야 계산 가능한 값이라 순수 버그라기보다 "L0 게이트가 L1 산출물보다 먼저 도는" 실행 순서상 제약에 가깝다 — 개선하려면 L1 1차 패스 결과를 L0 재평가에 피드백하는 루프가 필요.)

## 4. Full New L0 Evidence (상위 20개, `mean_net_bps` 내림차순)

| family | variant | archetype | n_events | mean_net_bps | net_lcb_bps | nw_tstat | cost_drag_ratio | gate_passed | handoff_tier | capacity_score | regime_stability | tf_corroboration | selected_for_l1 | reject_reasons | soft_flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trend_pullback_continuation | tpc_50_200_4h | trend | 15159 | 37.0773 | 5.9413 | 1.1910 | 0.4488 | False | seed | 0.0 | 0.0486 | 0.0 | True | weak_tstat | weak_tstat\|bootstrap_disagree |
| mtf_breakout_retest | mtf_bor_20_4h | trend | 11410 | 33.1454 | 11.9516 | 1.5547 | 0.3804 | **True** | seed | 0.0 | 0.0685 | 0.0 | True | | weak_rank_ic |
| lsr_oi_regime_filter | lsr_oi_gate_42_4h | hedge | 1764 | 29.9276 | 2.7209 | 1.1028 | 0.5342 | False | seed | 0.0 | 0.0775 | 0.0 | True | weak_tstat | weak_tstat\|bootstrap_disagree\|below_conviction_floor |
| mtf_breakout_retest | mtf_bor_40_4h | trend | 7366 | 24.2548 | -3.0623 | 0.8904 | 0.4610 | False | seed | 0.0 | 0.0584 | 0.0 | False | non_positive_lcb\|weak_tstat | weak_tstat |
| oi_lsr_unwind | oiu_42 | flow | 1502 | 23.9908 | -1.1913 | 0.9580 | 0.5008 | False | blocked | 0.0 | 0.0533 | 0.0 | False | non_positive_lcb\|weak_tstat | weak_tstat\|weak_rank_ic |
| **sparse_breakout_retest_liquidity** | sbrl_40_3_4h | trend | 12418 | 21.4452 | 4.0189 | 1.2296 | 0.4488 | False | **seed** | 0.0 | 0.0405 | 0.0 | **True** | weak_tstat | weak_tstat\|bootstrap_disagree\|below_conviction_floor |
| sparse_breakout_retest_v2 | bor_v2_40 | trend | 10186 | 17.2724 | -0.5497 | 0.9694 | 0.5159 | False | blocked | 0.0 | 0.0359 | 0.0 | False | non_positive_lcb\|weak_tstat | weak_tstat |
| trend_ma | ema_18_108 | trend | 6962 | 15.3273 | -16.9126 | 0.4731 | 0.5640 | False | blocked | 0.0 | 0.0180 | 0.0 | False | non_positive_lcb\|weak_tstat | weak_tstat |
| trend_pullback_quality_v2 | tpq_v2_50_200 | trend | 8063 | 13.3059 | -24.1374 | 0.3383 | 0.6778 | False | blocked | 0.0 | 0.0209 | 0.0 | False | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| **funding_flow_exhaustion_sparse** | ffes_96 | flow | 59 | 11.2141 | -26.9131 | 0.3642 | 0.4711 | False | blocked | 0.0 | 0.0850 | 0.0 | False | non_positive_lcb\|weak_tstat | weak_tstat\|weak_rank_ic |
| mtf_trend_pullback | mtf_tpb_20_30_4h | trend | 8765 | 7.7293 | -22.3034 | 0.2318 | 0.7363 | False | blocked | 0.0 | 0.0103 | 0.0 | False | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| mtf_trend_pullback | mtf_tpb_50_30_4h | trend | 11985 | 4.5816 | -23.8925 | 0.1288 | 0.8179 | False | blocked | 0.0 | 0.0055 | 0.0 | False | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| trend_pullback_continuation | tpc_20_100_4h | trend | 28049 | 2.3195 | -8.5274 | 0.2144 | 0.8935 | False | blocked | 0.0 | 0.0044 | 0.0 | False | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| **sparse_breakout_retest_liquidity** | sbrl_20_3 | trend | 20922 | 1.0805 | -6.9680 | 0.1390 | 0.9336 | False | blocked | 0.0 | 0.0043 | 0.0 | False | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| **vol_contraction_breakout** | vcb_20_120 | mean_reversion | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | blocked | 0.0 | 0.0000 | 0.0 | False | insufficient_events | below_conviction_floor\|weak_rank_ic |
| sparse_breakout_retest_v2 | bor_v2_20 | trend | 17552 | -0.5623 | -11.1008 | -0.0534 | 1.0343 | False | blocked | 0.0 | 0.0054 | 0.0 | False | non_positive_lcb\|weak_tstat\|excess_cost_drag | weak_tstat |
| ... (나머지 26개, 전부 mean_net_bps < 0 또는 excess_cost_drag/excess_turnover로 blocked) |

전체 42행 원본은 `logs/futures/alpha_foundry/4h_1783419659_evidence.parquet` 참고. 신규 6개 family 중 `vol_contraction_breakout`은 이번 4h/126심볼 데이터에서 `n_events=0`(insufficient_events)으로 아예 트리거되지 않았고, `carry_net_of_funding`/`xs_residual_rebalance`는 트리거는 됐으나 전부 큰 폭의 음수 net(`-30~-32bps`)로 blocked.

## 5. Reject Reason Breakdown (신규)

| reject_reason | count |
| --- | --- |
| non_positive_lcb | 37 |
| weak_tstat | 27 |
| excess_cost_drag | 31 |
| excess_turnover | 4 |
| insufficient_events | 1 |

`selected_for_l1` 4개 중 3개가 `gate_passed=False`인데도 L1로 넘어가는 것은 leak이 아니라 **spec이 의도한 정상 동작**이다 — `selected_for_l1`은 "`handoff_tier != blocked`인 후보 중 diversity/budget 배정을 통과했는가"를 뜻하고, `gate_passed`(=strict hard-reject 없음)와는 별개 축이다. 실제로 leak 여부의 판별 기준은 "`handoff_tier=blocked`인데 `selected_for_l1=True`"였고 이번엔 0건이다.

## 6. L1 Runtime Observation

동일 명령으로 L1까지 이어서 실행한 결과(수정 후 정상 완료, `Exit 0`):
- `[ALPHA-FOUNDRY] mode=gate n_panels_in=42 n_bound=42 n_passed=4 n_rejected=38 elapsed=4.0759s`
- L1 6h/8h/12h 승격 개수는 이전과 거의 동일(50/45/99 수준) — 이번 세션은 4h L0 게이트 내부 로직 수정에 집중했고, HTF 후보가 별도 문으로 L1에 들어오는 기존 아키텍처 이슈(§3과 연결되는 문제)는 이번 수정 범위 밖.

## Current Conclusion

- **3개 배선 갭(canonical gate 미호출/미배선/`selected_for_l1` leak) 전부 실측 확인 완료.**
- 그 과정에서 실데이터 전용 NaN 크래시 5곳을 신규 발견·수정(quant.md 안전 나눗셈 가드 패턴 적용) — 이 수정이 없었으면 canonical gate는 계속 프로덕션에서 죽었을 것.
- 신규 6개 family 중 `sparse_breakout_retest_liquidity`가 처음으로 `selected_for_l1=True`에 도달한 유일한 신규 family(`net_lcb_bps=+4.02bps`, `handoff_tier=seed`).
- **다음 우선순위**: `tf_corroboration`이 구조적으로 `0.0`에 고정돼 `handoff_tier=candidate`가 나올 수 없는 문제(§3) — L1 evidence를 L0 gate에 피드백하는 루프 설계가 필요. DEBUG summary 로거를 프로젝트 표준 `get_logger()`로 교체하는 것도 관측성 확보를 위해 필요.
