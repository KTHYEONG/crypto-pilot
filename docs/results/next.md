# L1/L2 다음 Spec 방향성 (2026-07-01 갱신)

측정 훅: `L1_PROBE_DIAG`, `L1_XS_SPREAD_DIAG`, `L2_DIAG_ATTR`(+`[L2-ATTR-REGIME]`), `L2_TREND_EFFICIENCY_GATE`, `L2_REVERSAL_KILL`.

## 1. 확정된 진단 (측정 기반)

- **L1 edge ≈ 100% 시계열 trend-beta, 횡단면 alpha 부재**: 2회 독립 측정 확증.
  - XS factor(`xs_momentum/carry/flow/oi_skew`) 신규 4종 → **승격 0**. rank-IC pooled 음수(fold0 +, folds1-3 −0.02~−0.14), spread Sharpe+ 는 long_frac 52~54% **롱틸트 beta 누수**. SSOT: `docs/specs/l1-cross-sectional-alpha.md`, `l1-xs-factor-spread-diag.md`.
- **병목 = Fold#1 (24Q4-25Q1) −27.4%** = **시장 전체 방향전환**(Dec'24 ATH→Q1'25 −19% 급락). `[L2-ATTR-REGIME]` 실측: fold0 bull −0.027/bear −0.017/crisis −0.035 **全 regime 동시 음전**(regime-특정 아님). folds1-2는 全 regime 양전.
- **fold0는 인과적 선행 탐지 가능**(BTC 4h 실측): mean_dd 0.074(folds1-2 2배), neg_mom 74%, 첫 달부터 감지.

## 2. ⭐ 핵심 구조 발견: L\* 상쇄 (노출-크기 레버 무효)

- **노출-크기 레버 4종 전부 byte-identical**: regime cap / adaptive reliability(P3) / trend-efficiency gate. L2의 **L\* 레버리지 옵티마이저가 균일/비례 스케일을 목표변동성에 맞춰 재흡수** → magnitude 레버는 구조적으로 무효.
- **탈출 조건**: gross의 **상대 시간분포**를 바꾸는 **선택적·시간집중** de-gross만 L\*(단일 스칼라)가 복제 불가.

## 3. 검증된 A/B (200 trials, baseline CAGR +1.5%/fold#1 −27.4%)

| 레버 | 결과 | 판정 |
|---|---|---|
| XS family 4종 | 승격 0, rank-IC 음수 | **기각** (횡단면 alpha 부재) |
| trend-efficiency gate (`L2_TREND_EFFICIENCY_GATE`) | byte-identical (mean_er 균일→L\* 흡수), whipsaw 가설 반증(er_corr≈0) | **기각** |
| **reversal kill-switch (`L2_REVERSAL_KILL`)** | **byte-identical 아님(첫 L\* 탈출)**. fold#1 −27.4→**−21.6%(+5.8pp)**. 단 folds2-3 −5.5/−2.3pp(과발화). 종합 +0.3pp | **유망·튜닝 필요** |

## 4. 추가 실측 (2026-07-01, BTC 4h detector replay)

동일 `opt_main_futures.py` 데이터 로더/윈도우/fold 경로로 BTC 4h detector만 재생한 결과:

| Variant | Fold0 risk-off | Fold1 risk-off | Fold2 risk-off | 총 bars | 판정 |
|---|---:|---:|---:|---:|---|
| `legacy 0.06 / p1` | 330 (58.5%) | 87 (15.4%) | 106 (18.7%) | 523 | 과발화 |
| `0.10 / p2` | 148 (26.2%) | 13 (2.3%) | 6 (1.1%) | 167 | **균형 후보** |
| `0.10 / p3` | 137 (24.3%) | 10 (1.8%) | 5 (0.9%) | 152 | **균형 후보** |
| `current 0.12 / p3` | 50 (8.9%) | 2 (0.35%) | 0 (0.0%) | 52 | **과보수 가능성** |

- **확정된 사실 1:** spec 변경(`0.12 / p3`)은 detector **selectivity 자체는 성공**. folds1-2 과발화를 거의 제거.
- **확정된 사실 2:** 그러나 fold0도 `330 → 50 bars (-84.8%)`로 급감. 즉, **병목 방어 신호까지 과도하게 절단**했을 가능성 높음.
- **확정된 사실 3:** detector 관점 최적점은 `0.10 / p2` 또는 `0.10 / p3`가 더 유력. `0.12 / p3`는 production default로 바로 승격할 상태 아님.

## 5. 추가 실측 (2026-07-01, 동일 L2 경로 economic replay)

동일 `opt_main_futures.py` L2 실행 경로에서 `L2_REVERSAL_REPLAY=1`, `L2_OPTUNA_TRIALS=60`로 직접 실행한 결과:

- **최종 L2는 여전히 BLOCKED**: `CAGR -2.3%`, `MDD 18.9%`, `CVaR95 0.7%`, `Fold pass 66.7%`, `Trades 184`, `DSR 0.255`.
- **병목은 여전히 fold#1**: fold#1 `-24.7%`, fold#2 `+9.3%`, fold#3 `+13.3%`.
- **spec replay hook은 실제 동작 확인**: `docs/results/l2_reversal_replay.csv` 생성 및 variant별/fold별 replay 데이터 저장.
- **중요 잔존 이슈 = selection/final parity divergence**:
  - `best_evaluation` replay CAGR `-4.48%`
  - `final simulation` CAGR `-2.27%`
  - `MDD`, `fold_pass`, `trade_count`, `L*`는 동일
  - 즉, **선택 경로는 같지만 최종 metric 산출 경로가 다름**

| Variant | Aggregate CAGR | Aggregate MDD | adoption_passed | metric_parity | blocker |
|---|---:|---:|---|---|---|
| `baseline_off` | -4.48% | 18.91% | False | True | `baseline` |
| `legacy 0.06 / p1` | **-2.07%** | **16.67%** | **True** | False | - |
| `0.10 / p2` | -5.81% | 18.30% | False | False | `fold0_defense_below_70pct` |
| `0.10 / p3` | -6.72% | 18.78% | False | False | `fold0_defense_below_70pct` |
| `current 0.12 / p3` | -6.19% | 18.81% | False | False | `fold0_defense_below_70pct` |

- **확정된 사실 4:** detector selectivity만 보면 `0.10/p2~p3`가 균형 후보였지만, **economic replay에서는 legacy 0.06/p1만 adoption_passed**.
- **확정된 사실 5:** 그러나 `legacy 0.06/p1`조차 aggregate CAGR가 음수이므로, **채택 가능성은 "상대 우위" 수준이지 production 승격 수준은 아님**.
- **확정된 사실 6:** `selection_parity=True`가 전 variant에서 유지되므로, 현재 핵심 불확실성은 detector on/off가 아니라 **final metric/parity 일관성** 쪽이다.

## 5b. ✅ P0 해소 (2026-06-30, parity divergence 근본 정정)

- **확정 원인 = annualization-tf SSOT 위반** (이전 가설 "metric aggregation 비결정성"/"deploy_leverage 차이"는 反證).
  - L2 신호 데이터는 master tf(=`_resolve_l2_master_tf`, 예: 8h)에서 실행되는데, **study/champion-selection 경로만 `run_config.timeframe`(4h)로 연율화**.
  - 동일 sim(`sum_log1p` 동일)을 ½ horizon으로 연율화 → **CAGR ×2, Sharpe ×√2, MDD 불변**(horizon-invariant signature). Phase A 계측(`[L2-PARITY-FP]`)으로 직접 증명.
- **Fix**: runner가 `_resolve_l2_master_tf` 1회 해석 → `_run_tiered_l2_study`에 `l2_master_tf` 전달(study=final tf 일치). `Layer2Result.master_tf` 필드 + `annualization_tf_mismatch` blocker assert. reversal replay 경로·self-check tf도 동일 정렬.
- **재실행 실증**: `tag=selection`==`tag=final` `cagr_dep=−0.087468`(8h) 일치, **MAIN parity mismatch 0회**.
- **부수 효과(중요)**: 정직한 8h 연율화에서 챔피언이 `layer2_blocked:cagr`로 차단. 즉 **이전 PASS는 ×2 팽창 CAGR에 의한 거짓 admission**이었음. selection이 정직하게 엄격화됨.
- SSOT: `docs/specs/l2-replay-parity-divergence.md` (sync 시 ADR 승격 예정).

## 6. 다음 우선순위 (방향 재정렬)

### P0 — ✅ DONE: parity divergence 해소 (위 §5b). detector 비교의 신뢰도 회복.

### P0' (신규 최우선) — 8h master tf 기준으로 detector/economic replay **재측정**
- **이유**: §4–5의 detector replay·adoption 판정은 전부 **4h-팽창 지표 시대**의 산출물 → 신뢰 불가. parity 정정 후 수치가 바뀐다(예: 챔피언 자체가 cagr-block).
- **실행**:
  - `legacy 0.06/p1` 등 variant adoption 판정을 **8h 정직지표로 재생성** (`docs/results/l2_reversal_replay.csv` 재작성).
  - "selection PASS→final BLOCK" 패턴이 parity 정정만으로 얼마나 완화/악화되는지 재확인.
  - 주의: trials 적으면 master tf가 run마다 흔들림(8h/12h) → tf 고정(`cfg.l2_master_tf`) 후 비교 권장.

### P1 — detector는 `legacy 0.06/p1` 기준으로만 후속 비교 (P0' 재측정 후 유효)
- **이유**: 동일 economic replay에서 `0.10/p2~p3`, `0.12/p3`는 모두 `fold0_defense_below_70pct`로 탈락.
- **판정**: 현 시점 detector 후보군 중 비교 기준점은 `legacy 0.06/p1` 하나뿐이다.
- **의미**: 추가 threshold 탐색은 `legacy` 대비 aggregate CAGR 또는 fold#1 방어를 넘길 명확한 근거가 있을 때만 진행.

### P2 — 근본 방향 1: 단일 BTC trigger → **market-state panel**로 확장
- **핵심 문제**: 현재 kill-switch는 BTC 경로 하나만 본다. 알트 디커플링, breadth 붕괴, 포지셔닝 unwind 같은 **시장 내부 상태**를 반영하지 못함.
- **권장 확장 축**:
  - `breadth`: universe 내 음수 모멘텀 비율, dispersion, cross-sectional downside breadth
  - `positioning stress`: funding/oi/lsr unwind 동시 붕괴
  - `beta break`: BTC는 버티지만 알트 beta가 깨지는 구간 탐지
  - `recovery hysteresis`: kill on/off를 동일 기준으로 쓰지 않고 해제는 더 보수적으로
- **원칙**: magnitude 레버는 L\*에 흡수되므로, 여전히 **시간선택적·집중적 de-gross** 형태여야 함.

### P3 — 근본 방향 2: binary kill → **event-window budgeting** 검토
- **문제**: 현재는 `risk_off_floor`로 곧바로 hard kill. 선택성은 좋아도 fold0 방어 구간을 너무 짧게 만들 수 있음.
- **대안**: binary on/off 대신 “event cluster 진입 후 N bars만 강제 de-gross, 이후 단계적 복귀” 같은 **event-window 정책** 검토.
- **주의**: 단순 gross multiplier curve는 L\* 흡수 위험이 있으므로, **연속된 bar 묶음의 시간분포를 바꾸는 정책**이어야 함.

### P4 — 미적용 정리 (비차단)
- kill-switch/efficiency gate 적용부 `floor` 하드코딩(0.05/0.30) → `cfg.regime.*` 참조로 교체.
- XS spread 진단 er-pair 수집이 gate-block 내부 → `L2_DIAG_ATTR` 기준 분리(gate-off 측정 무력 버그).

## 7. 정직한 기대치 (실증 확정)

- **횡단면 alpha 부재 확정**(2회) + **노출-크기 레버 고갈 확정**(L\* 상쇄). 남은 유효 레버 = **선택적 시간집중 de-gross**(reversal kill) 단 하나.
- **단, detector tuning만으로는 30% 목표 불가 가능성 여전** — trend-beta 자체가 작고, reversal kill은 방어 레버일 뿐 신규 alpha가 아님.
- **parity divergence는 해소됨(§5b)** — 단, §4–5의 detector 판정은 4h-팽창 지표 기반이므로 8h 정직지표로 **재측정 전까지 신뢰 보류**(P0').
- **전략적 분기 (P0/P1 실패 시)**:
  - ① **데이터 확장**: bookDepth / half_spread / liquidation proxy 등 마이크로구조 입력 확보
  - ② **상태 표현 확장**: BTC 단일 경로가 아니라 market-state panel 기반 causal de-gross
  - ③ **horizon 확장**: 일/주 단위 regime·macro state 활용
  - ④ **목표 재검토**: 현 strategy class의 구조적 CAGR 상한 재평가
- **명시적 중단선**: XS·cap·regime·efficiency 같은 **L\* 흡수형 magnitude 레버**는 더 파지 말 것. 다음 작업은 반드시 **economic replay** 또는 **state representation 확장**이어야 함.

## 8. Phase A 실측 (2026-07-01, market-state panel H1 검정) — ❌ 반증

`docs/specs/l2-market-state-panel.md` 설계(BTC-axis ∧ breadth-axis AND-게이트, P2 대응)의 measure-first 게이트. `compute_xs_downside_breadth_1d`를 실제 `opt_main_futures.py` 파이프라인(`L2_REVERSAL_KILL=1 L2_DIAG_ATTR=1`, 8h master_tf, LOG_LEVEL=DEBUG)에 계측해 `[L2-PANEL-DIAG]`로 fold별 breadth 분포 실측(trial 무관 결정론적, 3회 반복 재현 동일값).

| Fold | breadth_mean | breadth_p90 | btc_off_bars |
|---|---:|---:|---:|
| fold0 (24Q4-25Q1, 병목) | 0.5729 | 0.9808 | 50 |
| fold1 | 0.5672 | 0.9750 | 2 |
| fold2 | 0.4856 | 0.9423 | 0 |

- **H1 가설**: fold0 breadth 분포가 folds1-2 대비 유의하게 높다(시장전체 break ⇒ 판별 가능).
- **실측**: fold0 vs fold1 — mean 차이 1.0%p(0.5729 vs 0.5672), p90 차이 0.6%p(0.9808 vs 0.9750) → **사실상 동일 분포**. fold1은 btc_off_bars=2(과발화 튜닝 대상 fold)인데, breadth 레벨로는 fold0(병목/진짜 시장전체 break)와 전혀 구분되지 않음.
- **판정: H1 반증**. `compute_xs_downside_breadth_1d`의 **절대 레벨**(unconditional mean/p90)은 판별력이 없다 — 암호화폐 유니버스는 상시 알트 음모멘텀 비율이 높은 baseline을 가져(전 fold breadth_p90 0.94~0.98), 시장-전체 break 국면과 평상 국면을 레벨만으로 분리하지 못함.
- **spec guard 발동**: `docs/specs/l2-market-state-panel.md` §Adoption Protocol 2항에 따라 **Phase B(AND-게이트 economic replay) 중단**. 구현된 `compute_xs_downside_breadth_1d`/`compute_market_state_risk_off_1d`는 코드에 남기되(`reversal_mode="panel"`은 opt-in, 기본값 `"btc"` 불변 — 프로덕션 경로 영향 없음) 채택 보류.
- **재해석**: breadth **레벨**이 아니라 **conditional 분포**(= btc_off 발동 시점에서의 breadth 값)가 진짜 판별축일 가능성은 남아있으나, 현 계측(`[L2-PANEL-DIAG]`)은 unconditional 통계만 제공 — 이 재설계는 별도 spec 필요(비용 대비 근거 약함, 우선순위 낮음).
- **결론 갱신(§7 분기)**: BTC 단일경로 대비 **cross-sectional breadth 레벨 확장(P2)은 기각**. 남은 유효 분기는 ①마이크로구조 데이터 확장, ③horizon 확장(일/주 regime), ④목표 재검토 — ②는 breadth 레벨 방식에 한해 반증(conditional 재설계는 미검증, 저우선).
