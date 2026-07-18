# L1→L2 Replay 결과 — 2026-07-18 (위기 재현성 게이트: 레짐 심각도 신호 재설계)

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
| 롱/숏 방향 비대칭 opt-in 레버 | `ADR_20260717_L2_CRISIS_ASYMMETRIC_LONG_SHORT_CAP` | CAGR -28.04%→-14.19% 개선, **MDD 평평(미개선)** |
| 레짐 캡 해제 쿨다운 opt-in 레버 | `ADR_20260718_L2_CRISIS_REGIME_CAP_RELEASE_COOLDOWN` | 3개 레버 중 최선(MDD 46.5%→29.5%, cooldown=30 sweet spot), 단독으로는 21% 예산 미달 |
| **L2 위기 leverage 상한(l_crisis) + 방어 레버 탐색공간 편입** | `ADR_20260718_L2_CRISIS_LEVERAGE_CEILING` | 3-레버 조합 champion-고정 스윕(정식 코드화 전 진단): asym full-block+cooldown=30에서 crisis MDD 21.75%(예산 0.75pp 초과, 최선점). 그러나 프로덕션 코드화 후 200-trial replay: 탐색공간만 넓히고 objective가 crisis-blind라 Optuna가 방어 레버를 전부 off로 선택(asymmetry/cooldown 미사용) — 정상장 CAGR만 악화(53%→34.6%), 위기 MDD 25.38%로 미해결 |
| **L2 trial별 crisis MDD Optuna 제약(10번째 슬롯)** | `ADR_20260718_L2_CRISIS_AWARE_OPTUNA_CONSTRAINT` | 기존 정상장 게이트(9-tuple, `TPESampler(constraints_func=...)`에 이미 배선된 검증된 인프라)를 확장. 200-trial 실측: 방어 레버 사용률 0/200→154~198/200으로 반전, 챔피언의 crisis 제약값=-0.0424(예산 내, MDD≈16.8%) — **메커니즘은 설계대로 작동**. 그러나 정상장 CAGR+14.9%로 게이트 자체가 BLOCKED(cagr) — `_shape_efficiency_l2_objective`(Sortino 기반 scale-invariant, growth 미직접보상)와 신규 안전 제약이 만나 과도하게 보수적인 지점에 수렴 |

## 신규: 레짐 심각도 신호 재설계 (`ADR_20260718_L2_REGIME_SEVERITY_SIGNAL_REDESIGN`)

**근본 원인 재진단** — 3번의 할당(leverage/Optuna)-레이어 수정이 전부 "정상장 죽이거나 위기 못 막거나"의 동일한 딜레마로 되돌아온 원인을 신호 자체에서 직접 실측:

- **발견 1(구조적 결함, 정상장에서 확인)**: 6→3-state 압축맵(`_REGIME_COMPRESSION_MAP`)이 `transition`(방향 불확실, 단순 횡보)과 `crash`(CUSUM 진짜 급변)를 동일한 "crisis" 버킷으로 합산. 정상장(2023-2026) 실측: CUSUM 단독 8.5% vs 압축된 "crisis" 40.2% — **"위기" 라벨의 79%가 실은 단순 횡보**. 방어 레버를 강화할수록 정상장이 무너진 정확한 메커니즘.
- **발견 2(데이터 가용성 한계, LUNA/FTX 위기 검증에 국한)**: BTC 원시 가격 데이터가 정확히 2022-04-01부터 시작 — 위기 윈도우 시작일과 완전히 동일. 인과적(look-ahead 방지) 통계 설계상 위기 시작 시점엔 비교할 과거 기준선이 존재할 수 없음. 실측: CUSUM 발동률이 정상장 8.5% vs 위기장 8.2%로 **통계적으로 구분 불가** — 이 특정 위기 검증에 대해 레짐 신호의 판별력이 사실상 0.

**구현**: `MarketRegimeContext`에 `vol_scale_1d`/`crisis_active_1d` 신규 노출(기존 계산값 재사용, 추가 비용 0). `compute_risk_severity_code`(market_regime.py) 신설 — 방향 무관, 0=calm/1=elevated(causal quantile 기반 실현변동성)/2=crash(CUSUM). `Layer2AllocationConfig` opt-in 3필드(`l2_regime_severity_gating_enabled` 기본 False 등), `awf_sim.py`의 cap-gating 호출부(`apply_regime_risk_cap`/`apply_asymmetric_long_short_regime_cap`) 조건부 분기 — 기존 3-state 경로 완전 보존.

**실측 검증**:

| 검증 단계 | 결과 |
| :--- | :--- |
| 신호 품질(실제 구현 함수 직접 측정) | 정상장 "crash" 점유율 40.2%→**8.5%**(CUSUM 단독 수준까지 정확히 수렴), 위기장 33.1%→7.9% |
| 200-trial 챔피언 고정 A/B(severity_gating on/off) | 평균 gross exposure **0.3359→0.3852(+14.7%)** — 정상장에서 불필요한 억제가 풀리는 방향 확인(메커니즘 실작동 증명) |
| 최종 CAGR/MDD 정밀 비교 | **미완료** — fit_rets_hybrid(캘리브레이션 전용 부분구간) 기반 요약 함수가 전체 walk-forward 궤적의 차이를 못 잡음, 이번 챔피언도 정상장 CAGR+20.2%로 게이트(cagr) 미달 |

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS(회귀 없음).
- **레짐 심각도 신호 재설계:** ✅ 신호 품질 결함 구조적 해소(실측 검증), ✅ 프로덕션 배선 정상 작동(gross exposure 변화 실측 확인) — ⚠️ 최종 CAGR/MDD 정밀 재검증 미완료.
- **L2 위기 재현성 게이트:** FAIL-CLOSED 유지 — 레짐 신호 결함은 해소했으나, 3번의 champion 전부 정상장 CAGR 게이트(30%) 자체에서 막힘(4.2%/14.9%/20.2%, 점진적 개선 추세는 확인). Optuna 목적함수(`_shape_efficiency_l2_objective`, growth 미직접보상)와 안전 제약이 만나 과보수화되는 패턴이 반복 관측됨.
- **Optuna champion 비결정성:** 여전히 미해결, 매 run 결과 비교를 어렵게 함.

## 다음 조치

1. **[최우선]** `evaluate_l2_trial`의 정확한 fit-leg/전체궤적 metrics를 활용해 severity_gating on/off의 최종 CAGR/MDD를 정밀 재비교(직전 세션에서 fit_rets_hybrid 한계로 미완료) — 신호 수정의 실질 효과를 CAGR/MDD 단위로 확정.
2. **[최우선]** `_shape_efficiency_l2_objective`(Sortino 기반 scale-invariant, growth_lcb는 diagnostic으로 강등된 상태)가 안전 제약과 결합했을 때 과보수화로 수렴하는 패턴이 3회 연속 관측됨 — objective에 원시 CAGR/growth 보상을 재편입할지 검토 필요(단, 세션 내 규율에 따라 다른 변경과 분리해 독립적으로 A/B 검증할 것).
3. Optuna champion 비결정성(동일 `--seed`에도 run마다 challenger 상이) 원인 규명 — 매 검증마다 confound를 일으키는 근본 이슈, 우선순위 상향 검토.
4. `[LIMIT-01]` 2차 독립 위기 윈도우(2025-12-31~2026-06-30 BTC -32.8%) 검증 — 미해결.
5. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
6. 위기장을 포함하는 정상 holdout 윈도우로 Uplift/CAGR 재검증 — 미해결(`NO-CRISIS-WINDOW` 경고 지속).
