# L1→L2 Replay 결과 — 2026-07-18 (위기 재현성 게이트: 최초 동시 PASS 달성)

## 실행 조건

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-17 --seed 42`
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31~2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 103~137(run마다 변동); L1 admission 대부분 late_start 소수 제외.
- **주의**: 동일 `--seed 42`에도 Optuna champion이 run마다 크게 달라지는 비결정성 지속 확인(`ProcessPoolExecutor fork` 관련으로 추정, 별도 트래킹 필요) — 단일 run 내부 champion-고정 A/B 비교는 유효, run 간 절대 수치 비교는 confound.

## L1 결과 (안정 — 회귀 없음)

| Timeframe | Fold readiness | Probe LCB (bps) | 판정 |
| :--- | :---: | ---: | :---: |
| 1h~1d(7개 TF) | 3~4/4 | +37~+106 | 전부 PASS |

- Master TF는 `8h`(최대 breadth 기준)로 선정.

## 위기 재현성 게이트 — 정책 및 예산

`CrisisWindowMetrics`/`evaluate_crisis_survival()`(순수 함수)가 LUNA/FTX 2022 붕괴장(`2022-04-01~2023-02-15`, out-of-band 데이터)에 대해 champion의 rule-based 신호를 재생성해 생존 테스트한다. 위기 MDD 예산 `l2_max_mdd_abs×(1-l2_deploy_mdd_margin)=21%`, CAGR 하한 `l2_min_worst_fold_cagr=-5%`.

## 이전 수정 이력 (요약, 상세 서사는 decisions_archive.md 참고)

| 수정 | ADR | 핵심 결과 |
| :--- | :--- | :--- |
| BTC 레짐 데이터 무결성 | `ADR_20260717_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX` | has_btc False→True 복구, universe 확장(93→103)으로 champion drift(위기 MDD 29.01%→55.47%) |
| 레버리지 ceiling 구조 리팩토링 | `ADR_20260717_L2_LEVERAGE_CEILING_REFACTOR` | OOS-blend가 worst_fold/kelly ceiling을 더 이상 우회 못 하는 불변식 검증 |
| worst_fold 안전장치 기본 on | `ADR_20260717_L2_CRISIS_LEVERAGE_SAFETY_DEFAULT` | 위기 MDD 55.47%→46.53%, CAGR -38.44%→-28.04%(방향 개선, 예산 미달) |
| 롱/숏 방향 비대칭 opt-in 레버 | `ADR_20260717_L2_CRISIS_ASYMMETRIC_LONG_SHORT_CAP` | CAGR -28.04%→-14.19% 개선, MDD 평평(미개선) |
| 레짐 캡 해제 쿨다운 opt-in 레버 | `ADR_20260718_L2_CRISIS_REGIME_CAP_RELEASE_COOLDOWN` | 3개 레버 중 최선(MDD 46.5%→29.5%), 단독으로는 21% 예산 미달 |
| L2 위기 leverage 상한(l_crisis) + 방어 레버 탐색공간 편입(1차) | `ADR_20260718_L2_CRISIS_LEVERAGE_CEILING` | 탐색공간만 넓히고 objective가 crisis-blind라 Optuna가 방어 레버를 전부 off로 선택 — 정상장 CAGR만 악화(53%→34.6%), 위기 MDD 25.38%로 미해결 |
| L2 trial별 crisis MDD Optuna 제약(10번째 슬롯) | `ADR_20260718_L2_CRISIS_AWARE_OPTUNA_CONSTRAINT` | 방어 레버 사용률 0/200→154~198/200 반전, 제약 자체는 만족(MDD≈16.8%) — 그러나 정상장 CAGR+14.9%로 게이트 자체가 BLOCKED |
| 레짐 심각도 신호 재설계(방향-변동성 분리) | `ADR_20260718_L2_REGIME_SEVERITY_SIGNAL_REDESIGN` | 정상장 "crash" 오탐 40.2%→8.5%로 구조적 해소(실측 검증), gross exposure +14.7%(메커니즘 실작동 증명) — 그러나 이 시점엔 아직 탐색공간 미편입, 최종 CAGR/MDD 정밀 재검증 미완료 상태로 종료 |

## 신규: 배치-스케일 성장 목적함수 (`ADR_20260718_L2_DEPLOYED_SCALE_GROWTH_OBJECTIVE`) — 최초 동시 PASS

**근본 원인**: 세션 내내 정상장 CAGR이 계속 하락(53.2%→34.6%→14.9%→20.2%)한 이유를 목적함수 수학에서 직접 확인. `objective_l2_growth`의 1차항 `sortino_hac_unit`은 설계상 **scale-invariant**(leverage `k`배 변환 시 `E[r]`·`σ_down`이 동비율로 변해 Sortino 값 불변, `metrics.py` 독스트링에 명시)하다. worst_fold/kelly/l_crisis/crisis Optuna 제약을 계속 추가해 leverage 상한 후보를 늘릴수록 실제 배치 leverage(L*)는 단조적으로 낮아지는데, **objective는 이 억제를 전혀 못 본다** — `sortino_hac_unit`은 `_l_star` 계산 이전의 unit-leverage `rets_hybrid`로 계산되고, 유일한 leverage 인지 항인 `risk_util_realized` soft penalty는 가중치가 `≤0.03`로 캡핑돼 사실상 무의미(실측: 페널티 크기 <0.01, Sortino 값 0.5~3+에 압도됨). 기존 `growth_lcb_hybrid`(diagnostic)조차 이름과 달리 배치 전 `rets_hybrid`로 계산되는 동일한 잠복 버그를 갖고 있었음을 확인.

**구현**: `_dep.scaled_rets`(배치 후 수익률)로 `_contiguous_block_log_growth`/`_growth_lower_confidence_bound`(기존 함수 100% 재사용, 새 수학 없음)를 재계산해 `growth_lcb_deployed` 산출, `_shape_efficiency_l2_objective`에 `growth_lcb_weight`(기본 0.0=no-op) 블렌드 항으로 추가(Sortino shape 가드는 유지, 완전 대체 아님). `l2_objective_growth_lcb_weight`(0.0~1.0)와 `l2_regime_severity_gating_enabled`를 `L2_SEARCH_SPACE`에 정식 편입.

**실측 검증(200-trial 프로덕션 replay, 두 파라미터 모두 탐색공간 포함)**:

```
정상장 스코어카드: STATUS ✅ PASS — CAGR +35.1%, MDD 12.0%
위기 재현성 게이트: STATUS ✅ PASS — stress_tested_pass, verified=True
  LUNA/FTX MDD  = 16.72% (예산 21% 이내)
  LUNA/FTX CAGR = -4.94% (하한 -5% 이내, 근소)
  CVaR95        = 1.44%  (예산 6% 이내)
파이프라인 exit_code = 0
```

챔피언이 자체 선택한 파라미터: `l2_objective_growth_lcb_weight=0.8`(탐색 범위 상단 근처), `l2_regime_severity_gating_enabled=True`, `l2_regime_long_short_asymmetry_enabled=True`, `l2_regime_cap_release_cooldown_bars=34`. Optuna 제약 10개 전부 만족(전부 음수).

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS(회귀 없음).
- **L2 정상장 스코어카드 + 위기 재현성 게이트:** ✅ **동시 PASS 최초 달성**(exit_code=0) — 이 세션 전체(leverage 상한, crisis Optuna 제약, 레짐 심각도 신호, 배치-스케일 성장 목적함수) 4개 수정이 맞물려 작동함을 실측 확인.
- **`/check`:** PASS(Cov 100%, l2_search_space.py 대상).
- **주의**: 단일 run 결과. Optuna champion 비결정성이 여전히 미해결이라, **동일 조건 반복 재현성 검증이 다음 최우선 과제**.

## 다음 조치

1. **[최우선]** 동일 `--seed`로 반복 재현성 검증(champion 비결정성 하에서도 PASS가 일관되게 나오는지) — 지금까지 단 1회 run의 성공이며, run-to-run 편차가 이 세션 내내 관측된 만큼 확증 전까지 production 승격 보류.
2. `[LIMIT-01]` 2차 독립 위기 윈도우(2025-12-31~2026-06-30 BTC -32.8%) 검증 — 미해결, 유일한 검증 윈도우(LUNA/FTX)에 대한 과최적화 위험 여전히 존재.
3. Optuna champion 비결정성(동일 `--seed`에도 run마다 challenger 상이) 원인 규명 — 매 검증마다 confound를 일으키는 근본 이슈, 우선순위 상향.
4. `growth_lcb_weight` 과대 시(예: 2.0+) tail-risk 재발 여부 sweep — 안전 상한 문서화 미완료(spec 검증 프로토콜 §3, 미실행).
5. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
6. 위기장을 포함하는 정상 holdout 윈도우로 Uplift/CAGR 재검증 — 미해결(`NO-CRISIS-WINDOW` 경고 지속).
