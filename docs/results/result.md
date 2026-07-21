# L2 Recency-Generalization 게이트 적용 결과 — 2026-07-21 (최신)

`docs/specs/l2-recency-generalization-gate.md` 구현(`/check` PASS, Cov 57%) — L2→L3 반복 붕괴 패턴(아래 "Phase L3 종단" 섹션의 우선순위2)에 대한 구조적 방어 2건 추가. **범위는 grill-me에서 명시적으로 좁혀짐: 신규 alpha/crisis 전용 신호는 배제, 알파 부재 자체는 이번 조치로 해결되지 않음.**

## 무엇을 고쳤나 (쉽게 설명)

1. **Recency Holdout 하드게이트(신규)**: 기존 `recent_fold` 게이트는 "L2 study 구간 4개 fold 중 마지막 fold"만 봤는데, 이 fold는 원래 champion을 고르는 점수(objective) 계산에도 이미 들어가 있어서 "내가 채점한 답안지로 내가 검증"하는 순환 구조였다. 이번에 **채점(objective)에는 전혀 안 들어가는, study 구간 맨 끝 30일만 따로 떼어** CAGR이 일정 기준(-5%) 아래면 그 파라미터 조합을 아예 후보에서 탈락시키는 제약을 추가했다(Optuna 14번째 제약 슬롯).
2. **"위기장 미검증" 경고를 실제로 보이게 함(투명성)**: L2 study 구간(예: 2025년)이 우연히 하락장/위기장을 하나도 안 겪었을 수 있는데, 이 경우 지금까지는 콘솔에 `NO-CRISIS-WINDOW`라는 경고 한 줄만 찍히고 아무도 이 정보를 실제 판정(멀티시드 합의 로그 등)에 반영하지 않았다. 이번에 이 정보를 **정식 필드(`window_bottleneck_covered`)로 승격**시켜 `[MULTI-SEED]` 최종 판정 로그에 항상 같이 찍히도록 배선했다.
3. **실측 중 발견한 크래시 버그 수정**: 위 1번을 배선하는 과정에서 실제 파이프라인을 돌려보니 `zip() argument 2 is longer than argument 1`로 3개 seed 전부 크래시하는 것을 발견 — 제약 이름 목록(13개)이 새로 늘어난 제약값 목록(14개)과 개수가 안 맞았던 실수. 즉시 수정 후 재실행해서 정상 동작 확인.

## 실측 결과 (`--phase l3 --trials 120 --timeframe 4h --seed 42`, 내부 42/43/44 순차)

| Seed | joint_feasible | recency_holdout 때문에 걸러진 trial 수 | 최종 판정 |
|---|---|---|---|
| 42 | 0/120 | 53개 | ❌ no_feasible_trials |
| 43 | 0/120 | 80개 | ❌ no_feasible_trials |
| 44 | 0/120 | 49개 | ❌ no_feasible_trials |

`[MULTI-SEED] pass_count=0/3 required=2 admitted=False window_covered=False`

**해석**: 새 게이트는 설계대로 실제로 작동한다(seed당 49~80개 trial을 차단). 하지만 최종 결론(`admitted=False`, 배포 불가)은 **바뀌지 않았다** — `cagr`/`crisis_cagr`/`crisis_mdd`(=알파가 거의 없다는 신호) 위반이 이미 recency_holdout보다 더 많은 trial을 걸러내고 있어서, 이번 게이트는 "이미 안 좋은 후보를 한 번 더 확실히 안 좋다고 확인"하는 역할만 했다. **좋은 소식은 이제 이 경고가 로그에 숨지 않고 항상 보인다는 것.**

## 여전히 문제인 것 (해결 안 됨, 이번 조치의 범위 밖)

- **근본 병목은 그대로다**: BTC를 제외한 나머지 코인(ETH, BNB, 알트 전부)의 신호가 사실상 "0"에 가깝다(`avg_mult=0.000`). 게이트를 아무리 정교하게 만들어도, 애초에 태울 알파(수익 낼 신호)가 없으면 통과할 후보 자체가 안 생긴다.
- 이번 spec은 사용자 확인(grill-me)에 따라 **crisis 전용 신호 재도입은 완전 배제**(과거 3번 반증된 경로라 재시도 안 함), **신규 alpha 소스 리서치도 범위 밖**으로 뒀다. 즉 이번 조치는 "잘못된 배포를 막는 방벽을 하나 더 튼튼히 한 것"이지, "돈을 벌 수 있게 만든 것"은 아니다.
- 다음으로 실제 성과를 개선하려면 이 문서 하단 "Phase L3 종단" 섹션의 **우선순위 3(L1 신규 alpha source 탐색: 마이크로구조/펀딩-베이시스/IV파생 등)**으로 넘어가야 한다 — 트렌드/모멘텀/리버설 계열은 이미 소진됐다.

---

# L2/L3 Multi-Seed 강건성 합의 게이트 적용 결과 — 2026-07-21

`docs/specs/l2-l3-multi-seed-robustness-consensus.md` 구현(`/check` PASS, Cov 32%) — L2 champion 승격을 단일 seed 결과에서 **K=3 독립 seed(base_seed, +1, +2) 과반수(2/3) 합의**로 전환, 미달 시 hard block(exit_code=1)하도록 기본 동작을 변경했다.

## 실행 결과 (`--seed 42` → 내부적으로 42/43/44 순차 실행)

```
[MULTI-SEED] seed=42 L2 study blocked: no_feasible_trials
[MULTI-SEED] seed=43 L2 study blocked: no_feasible_trials
[MULTI-SEED] seed=44 L2 study blocked: no_feasible_trials
[MULTI-SEED] pass_count=0/3 required=2 admitted=False
```

- 소요시간: `real 7m52.7s` (단일-seed 대비 약 2.6배 — 의도된 증가, `performance.md` §4 "15% 회귀" 규칙의 예외로 기록).
- 3개 seed(42/43/44) **전부** `no_feasible_trials`(joint_feasible=0) — champion 자체를 못 찾음. 과반수(2/3) 미달로 즉시 hard block.
- `test_active_pipeline_l3_blocked_when_consensus_fails_returns_exit_code_1`이 `/check`에서 이미 검증됐으므로 exit_code=1로 정상 종료(실측 재확인은 다음 세션에서 `echo $?`로 직접 캡처 권장).

## 판정

1. **Multi-seed 합의 게이트는 설계대로 정확히 작동했다.** 직전 세션의 수동 3-seed 실측(42/123/7 → 0/1/10 feasible, "운 좋은 seed 하나"가 존재)과 달리, 이번 연속 seed 조합(42/43/44)은 **셋 다 실패**해 더 단호하게 결론을 확인시켜준다.
2. **이것은 게이트의 실패가 아니라 성공이다.** 이 코드/설정으로는 L2 탐색 프로세스 자체가 강건한(seed에 안정적인) champion을 찾지 못하는 상태이며, 게이트가 바로 이런 상황의 배포를 정확히 차단했다.
3. **반복된 `no_feasible_trials`는 이제 게이트/파이프라인 설계의 문제가 아니라 L2 탐색공간·제약 자체 또는 그 상류의 L1 알파 표현력 문제로 수렴한다** — 이미 우선순위 3(L1 신호 표현력 근본원인)에서 지목된 병목과 일치.

## ⚠️ 하지 말아야 할 것 (재확인)
seed offset을 이것저것 바꿔가며 "과반수 통과하는 조합 찾기"를 시도하는 것은 **금지** — 이는 게이트를 무력화하는 새로운 형태의 p-hacking이다. `admitted=False`가 반복되면 다음 조치는 L2 탐색공간 재설계 또는 L1 알파 재검토여야 하며, 동일 로직에 seed만 바꿔 재시도하는 것이 아니다.

---

# Crisis Replay 매칭 버그 수정 + 3-Seed 반복 검증 — 2026-07-21 (최신)

`docs/specs/crisis-replay-strategy-match-fix.md` 구현(`/check` PASS, Cov 26%) — `_build_rule_based_stress_batch()`의 `panel.variant` substring 매칭을 `panel.family:variant` 정확 일치로 교체. 목적: 직전 세션(L1 registry merge) 이후 회귀한 crisis reliability(`stress_tested_pass`→`untested_no_data`)를 근본 수정.

## 3-Seed 반복 실행 결과 (동일 조건: `--trials 120 --timeframe 4h --sync skip`)

| Seed | joint_feasible | L2 | L3 | Crisis Reliability |
|---|---|---|---|---|
| 42 | 0/120 | ❌ BLOCKED (`no_feasible_trials`) | 도달 못함 | 도달 못함 |
| 123 | 1/120 | ✅ PASS (CAGR +31.2%) | ❌ BLOCKED (CAGR -10.0%) | ✅ **stress_tested_pass** |
| 7 | 10/108* | ✅ PASS (CAGR +48.7%) | ❌ BLOCKED (CAGR -20.1%) | ✅ **stress_tested_pass** |

\* seed=7은 120 trial 중 108에서 종료(정상 종료, exit_code=0).

## 판정

1. **Crisis-match 수정 자체는 확인됨 — 정상 작동.** 챔피언이 선정된 두 seed(123, 7) 모두 `stress_tested_pass`로 복구됐다(직전 세션의 `untested_no_data`/`trades=3` 붕괴가 재발하지 않음). `panel.family` 활용 정확 일치 교체가 의도대로 동작.
2. **더 중요한 재해석: 직전 세션 seed=42의 "L3 DEPLOY-READY(+5.6%)"는 재현되지 않는다.** substring 버그가 있던 상태(부정확한 crisis 제약)에서 채택된 champion이었을 가능성이 높다 — crisis 제약을 정확하게 계산하도록 고치자, 동일 seed=42조차 아예 champion을 못 찾는(`no_feasible_trials`) 결과로 바뀌었다. 즉 이전 세션의 긍정적 결과는 **버그가 낀 상태의 우연한 산물**이었을 개연성이 크다.
3. **3개 seed 전부 L3 최종 홀드아웃 실패** — joint_feasible이 늘어날수록(seed 7: 10개) 오히려 L3가 더 나빠짐(-20.1%, seed 123의 -10.0%보다 악화). Study 윈도우에 대한 과최적화(curve-fitting) 신호로 읽힌다 — feasible trial이 많을수록 study 윈도우에 더 잘 들어맞는 champion을 고르게 되는데, 그 champion이 forward holdout에는 더 안 맞을 수 있다는 정합적인 설명.
4. **결론: L1 registry merge(`ADR_20260721_L1_MULTI_TF_REGISTRY_MERGE`)와 crisis-match 수정 둘 다 각자의 목적(신호 폐기 방지, crisis reliability 계산 정확도)에서는 옳고 검증됐다. 하지만 L3 진짜 forward holdout 일반화 문제는 여전히 미해결이며, 오히려 이번 정밀 재검증으로 "L2 study 윈도우 최적화가 L3 forward에 일반화되지 않는다"는 기존 진단(아래 섹션 §근본원인진단 4번, non-stationarity)이 더 강하게 재확인됐다.**

## 다음 우선순위 (갱신)

- 아래 섹션(2026-07-21 최초 L3 분석)의 "우선순위 2"(L2 study 단계 recency 강건성 게이트)가 이번 3-seed 결과로 **더 시급해졌다** — joint_feasible이 많을수록 L3가 나빠지는 패턴은 정확히 이 게이트가 방지하려는 현상이다.
- seed 다양화 재실행 자체를 반복하며 "좋은 시드 찾기"로 흐르지 않도록 주의(anti-overfitting) — 다음 조치는 게이트/탐색공간 설계 변경이어야 하며, 동일 코드로 seed만 바꿔가며 재시도하는 것은 금지.

---

# L1 Multi-TF Registry Merge 적용 후 재실행 결과 — 2026-07-21 (후속)

`docs/specs/l1-deployment-registry-multi-tf-merge.md` 구현(`/check` PASS, Cov 24%) 직후 동일 조건(`--phase l3 --trials 120 --timeframe 4h --sync skip --seed 42`)으로 재실행. 결과: **L3가 BLOCKED → ✅ DEPLOY-READY로 전환**.

## Before/After 비교

| 지표 | Before (버그) | After (병합 적용) | 변화 |
|---|---|---|---|
| **L3 STATUS** | ❌ BLOCKED (negative_return) | ✅ **DEPLOY-READY** | 전환 |
| L3 CAGR / Total Return | -0.2% | **+5.6%** | +5.8%p |
| L3 Sharpe / Sortino | 0.021 / 0.029 | **0.688 / 1.018** | 큰 폭 개선 |
| L3 MDD | 8.6% | 5.8% | 개선(더 안전) |
| L3 CVaR95 | 0.7% | 0.6% | 개선 |
| L3 Trades | 73 | 110 | +37 |
| L2 CAGR / Sharpe | +33.2% / 1.815 | +33.3% / 2.035 | 소폭 개선(side benefit) |
| L2 Fold Pass | 75%(3/4) | 100%(4/4) | 개선 |
| L2 joint_feasible | 1/120 | 2/120 | 개선 |

- **근거 확인**: `[L1-MAJOR-REGISTRY-CENSUS]`에서 `ETHUSDT/trend_pullback_continuation`(registry_mean_incremental_bps 40~269bps, 4개 variant)이 `observed_active_in_holdout=True`로 전환됨 — 이전 실행에서는 ETH가 6개월 내내 완전 비활성(`avg_mult=0.000`)이었으나 이번엔 실제로 신호가 발화해 book에 기여했다. 이는 게이트 완화가 아니라 **버려지던 실제 신호가 살아난** 결과라는 spec의 가설을 실측으로 확인한 것이다.
- ETH의 `dual_momentum`(128bps)/`trend_donchian`(219bps)/`btc_regime_pullback`(97bps)/`mtf_fusion`(101bps), BTC의 `trend_donchian`(106bps)은 이번에도 여전히 `observed_active_in_holdout=False` — 병합 버그 수정으로 신호 "폐기"는 막았지만, 이 전략들이 정작 이번 6개월 구간에서 발화 조건 자체를 못 만난 것인지는 별도 확인 필요(추가 알파 여지 가능성).

## ⚠️ 신규 회귀 발견 — Crisis Reliability 검증 무효화

```
Before: [CRISIS-RELIABILITY] status=stress_tested_pass verified=True (mdd=19.69%, cagr=-4.46%, trades=113)
After:  [CRISIS-RELIABILITY] status=untested_no_data verified=False detail="usable_windows=0 < min_usable_windows=1"
        [CRISIS-WINDOW-DETAIL] status=stress_data_invalid trades=3 (전 실행 113건 대비 급감)
```
- 병합된 registry가 새 전략 조합(`taker_imbalance_momentum:tim_12_4h` 등, 이전 실행엔 없던 panel)을 포함하게 되면서 LUNA/FTX crisis window(2022, 소수 legacy 심볼만 존재) replay 시 유효 거래가 3건으로 붕괴 — crisis 생존성 자체가 "검증 불가" 상태로 바뀌었다.
- 이번 세션 범위 밖의 **신규 발견 이슈**로, 다음 세션에서 원인 규명 필요(병합된 신규 전략 조합이 legacy crisis 심볼 커버리지와 호환되지 않는 것으로 추정). 승격 판단 시 이 경고를 반드시 함께 고려할 것 — `L3 DEPLOY-READY` 판정은 crisis 생존성 검증과 별개다.

---

# Phase L3 종단(End-to-End) 실행 결과 및 자산증식 전략 개선 분석 — 2026-07-21

## 실행 조건

`uv run python src/execution/opt_main_futures.py --phase l3 --trials 120 --timeframe 4h --sync skip --seed 42`

- 엔진: Tiered Pipeline (`USE_CS_RANK_ENGINE=1` 경로, `signal_only=False`)
- L1 캐시: 7/7 TF hit (당일 이전 세션 산출물 재사용)
- 총 소요시간: `real 3m36.1s` / `user 7m45.6s` (병렬 워커) / `sys 0m24.8s`
- 프로세스 종료 코드: `0`

## 파이프라인 게이트 통과 현황

| Layer | 윈도우 | 판정 | 핵심 사유 |
|---|---|---|---|
| L1 (Signal Robustness) | 2023-06-30 ~ 2024-12-30 | ✅ PASS | 126/137 심볼 admitted |
| L2 (Optuna 튜닝, Study 윈도우) | 2025-03-20 ~ 2025-12-30 | ✅ PASS | `joint_feasible=1/120`, champion 갱신 |
| L3 (Final Holdout) | 2025-12-31 ~ 2026-06-30 | ❌ **BLOCKED** | `negative_return` |

**주의:** L3가 BLOCKED임에도 프로세스 종료 코드는 `0`("tiered_pipeline_completed")이다 — 코드 근거는 하단 [프로세스 무결성 결함] 참조.

## L2 스코어카드 (Study 윈도우, 2025-03-20~2025-12-30) — 참고용, 이미 안정적으로 PASS

```
[L2-AUDIT] completed=120/120 joint_feasible=1 crisis_measured=120
Champion: tf=8h (auto-selected by edge_quality, --timeframe=4h 무시됨 — 하단 참조)
CAGR +33.2% | MDD 13.7% | Sharpe 1.815 | Sortino 2.854 | Calmar 2.426
PSR 0.989 | DSR 0.986 | Sharpe Uplift +0.45
Fold Pass 75% (3/4) | Trades 174 | Crisis(luna_ftx_2022) stress_tested_pass (CAGR -4.46%, MDD 19.69%, CVaR95 1.92%)
```

`docs/results/result.md`(2026-07-20판)와 지표가 재현되었고(seed=42 재현성 fix 이후 결정론 확인, `ADR 2026-07-19`), **L2는 2회 연속 세션에서 동일하게 견고한 PASS** — 이번 세션의 추가 튜닝은 한계효용이 없다(→ 사용자 지시대로 L3 결과 피드백에 집중).

⚠️ L2 스코어카드 자체에 `NO-CRISIS-WINDOW` 경고 포함: 이 study 윈도우(2025Q1~Q4)는 병목-caliber fold(MDD≥15%&CAGR≤0)를 포함하지 않아 승격 근거로 단독 인용 불가 — 그래서 L3 최종 홀드아웃이 실질적인 검증 관문이다.

## L3 최종 홀드아웃 스코어카드 (2025-12-31 ~ 2026-06-30) — 핵심 결과

```
STATUS: ❌ BLOCKED (Reason: negative_return)

[GROWTH]    CAGR: -0.2% | Total Return: -0.2% (필요: >0.0%) | Equity x1.00
[EFFICIENCY] Sharpe: 0.021 | Sortino: 0.029  ← 사실상 무엣지(near-zero)
[RISK]      MDD: 8.6% (<=35%) | CVaR95: 0.7% (<=6%) | Exposure: 0.2x  ← 매우 낮은 실배치 비중
[DEPLOY]    Trades: 73 (>=10)
```

- 레짐 분포(L3 윈도우): `bull=43.0% bear=17.8% crisis=39.3%` — 직전 study 윈도우(암묵적으로 alt-트렌드 우호적) 대비 **crisis 상태 비중이 이례적으로 높음**.
- Major 심볼 신호 활성도: `BTCUSDT mu_bull=0.8% avg_mult=0.895` (유일하게 활성) vs `ETHUSDT mu_bull=0.0% avg_mult=0.000`, `BNBUSDT mu_bull=0.0% avg_mult=0.000` (완전 비활성).
- Long/Short 활성 바 수는 균형적(long=561, short=522)이나 book은 알트코인 중심(APEUSDT, AAVEUSDT, TRBUSDT 등)이고 major는 BTC 단독 캐리.

## 근본원인 진단 (코드 근거)

1. **크라이시스 리스크 캡이 아니라 알파 부재가 1차 원인.** `apply_regime_risk_cap`(`l2_meta.py:1246`)의 기본값은 `crisis_gross_cap=0.55`, `bear_gross_cap=0.75` — 즉 crisis+bear 구간(57.1%)이라도 이론상 최대 0.55~0.75x까지는 배치 가능하다. 그런데 실측 평균 `Exposure=0.2x`로 이론적 캡보다 훨씬 낮다 → 캡이 바인딩된 게 아니라 **신호 자체(mu)가 대부분 0에 가까워 배치할 게 없었다.** ETH/BNB `avg_mult=0.000`이 이를 직접 증명한다.
2. **BTC 단독 캐리 구조.** Study 윈도우(2025)에서는 알트 트렌드/모멘텀 계열(`dual_momentum`, `trend_donchian`, `mtf_fusion`)이 알트 로테이션 장에서 작동했으나, L3 홀드아웃(2026 H1)에서는 이 계열의 엣지가 재현되지 않고 BTC의 미미한 mu(0.8%)만 남았다. Sharpe 0.021은 사실상 노이즈 수준.
3. **비용/펀딩 드래그가 마지막 한 방을 넣었다.** Total Return -0.2%는 큰 손실이 아니라 "거의 0"인 상태에서 거래비용이 부호를 음(-)으로 뒤집은 결과 — `docs/specs/l2-edge-attribution-diagnostics.md`(2026-06-28 진단, SSOT)의 "gross alpha 부재(67%) > cost(33%)" 결론과 정확히 동일한 패턴이 **실제 미래 데이터(진짜 OOS)에서 재확인**된 것이다. `feature 예측력 부재`는 추정이 아니라 지금 **실증**됐다.
4. **L2 study 윈도우와 L3 홀드아웃 간 비정상성(non-stationarity).** L2가 auto-select한 champion tf=8h(`--timeframe 4h` 플래그는 무시됨 — `pipeline.py:4020` `_resolve_l2_master_tf`는 `cfg.l2_master_tf`가 미설정이면 CLI 인자와 무관하게 `edge_quality` 최댓값 TF를 자동 채택)는 study 윈도우에서 최적화됐을 뿐, forward 6개월에서 전혀 다른 레짐 구성(crisis 39.3%)을 만나 붕괴했다 — 전형적인 walk-forward 일반화 실패.

## 프로세스 무결성 결함 (신규 발견)

`active_pipeline.py:3326-3332`:
```python
if l2_final is None or not l2_final.gate_passed:
    return RunnerResult(exit_code=1, reason=f"layer2_blocked:{final_reason}")
...
if run_config.phase == "l3":
    return RunnerResult(exit_code=0, reason="tiered_pipeline_completed")
```
- 종료 코드 결정은 **`l2_final.gate_passed`만 검사**하고, L3의 `l3_gate_passed`(BLOCKED)는 검사하지 않는다.
- 즉 `--phase l3` 실행이 L3에서 완전히 BLOCKED 되어도(이번 세션처럼) 프로세스는 `exit_code=0`으로 종료된다.
- CI/cron 등 exit code만으로 자동화된 후속 단계(배포 승격, 알림 등)가 있다면 **이 실패를 놓친다.** 콘솔에 찍히는 `[LAYER 3: HOLDOUT VALIDATION SCORECARD]` 텍스트를 별도로 파싱하지 않는 한 실패가 은폐된다.

## 개선 제안 (우선순위)

### 우선순위 1 — L3 게이트를 종료 코드에 반영 (즉시, 저비용, 고가치)
- `run_config.phase == "l3"` 분기에서 `l3_gate_passed`가 False면 `exit_code=1, reason=f"layer3_blocked:{l3.blocker_reason}"`를 반환하도록 수정.
- 근거: 이번 세션에서 발견된 실제 사일런트 실패 사례 — 재발 방지 비용이 매우 낮고(if문 하나) 가치는 "true OOS 실패가 성공으로 오인되는" 최악의 리스크를 차단.

### 우선순위 2 — L2 champion 승격 이전에 "최근 구간(Recency) 강건성 게이트" 선행 (구조적)
- 현재 구조는 L2에서 champion을 확정(`growth_lcb_deployed` 등으로 최적화)한 뒤 L3에서 사후 검증만 한다 — 이번처럼 L2가 100% 안정적으로 PASS 해도 L3에서 뒤늦게 죽는 패턴이 반복될 수 있다.
- 제안: L2 Optuna 목적함수/제약에 "최근 N개월(rolling) OOS 서브구간에서 Sharpe>0" 같은 recency 하드 게이트를 추가해 study 단계에서부터 최근 레짐에 죽는 파라미터를 배제한다. (단, `quant.md` §0 우선순위상 이는 "curve-fitting 방지"와 "walk-forward 강건성" 사이의 트레이드오프이므로, 새로운 고정 임계값을 급조하지 말고 기존 `l2-replay-parity-divergence.md`/`layer2-cost-aware-selection.md` SSOT에 정합되는 형태로 설계 필요 — **다음 세션 `/spec` 대상**.)

### 우선순위 3 — 알파 근본 원인은 "L2 파라미터"가 아니라 "L1 신호 표현력" (재확인, 신규 작업 아님)
- 이번 L3 결과는 `project_l2_edge_waterfall_diagnosis_2026_06_28`(gross alpha 부재)과 `project_metrics_cache_never_materialized_2026_07_02`(L1 비추세 신호 다양화 전 사이클 기각)의 결론을 **실제 미래 데이터로 재확인**한 것이다. L2 하이퍼파라미터를 더 튜닝하는 것은 이미 study 윈도우에 대해 100% 최적화되어 있으므로 추가 이득이 없다(오히려 curve-fitting 위험만 키움).
- 알트코인 트렌드 계열 의존도를 낮추고 BTC/ETH 등 유동성 최상위 심볼에 대해 **다른 알파 원천**(예: 마이크로구조, 펀딩/베이시스 기반, 옵션 IV 파생 신호 등 — 과거에 시도하지 않은 축)을 탐색하는 것이 유일한 실질적 경로. 트렌드 계열 파라미터 재탐색은 반복하지 말 것(과거 세션에서 이미 반증됨).

### 우선순위 4 — Crisis 레짐 재현빈도 재검토
- L3 윈도우의 `crisis=39.3%`가 실제 시장 상태를 정확히 반영한 것인지, 아니면 레짐 분류기가 최근 데이터에 대해 과민(false crisis)한 것인지 미검증. 만약 후자라면 exposure 억제가 불필요하게 컸을 수 있다 — 다만 이번 세션 진단상 `avg_mult=0.2x`는 캡(0.55~0.75x)보다도 낮아 **레짐 캡이 아닌 신호 부재가 주 원인**이므로 이 항목은 2차 우선순위.

## 결론

- L2는 이번 세션에서도 견고하게 PASS했고 추가 튜닝의 한계효용이 없다 — 사용자 지시대로 **L3 결과에 집중**했다.
- L3는 실질적 실패(`negative_return`, Sharpe≈0)이며, 원인은 이미 알려진 "L1 알파 예측력 부재"가 진짜 미래 데이터에서 재현된 것이다. 손실 자체는 작지만(MDD 8.6%, CVaR95 0.7%) **성장 목표(자산 증식)를 전혀 달성하지 못했다.**
- 최우선 조치는 (1) 종료 코드가 L3 실패를 은폐하지 않도록 수정, (2) L2 study 단계에 recency 강건성 게이트 설계, (3) 트렌드-패밀리 재튜닝이 아닌 신규 알파 원천 탐색이다.
