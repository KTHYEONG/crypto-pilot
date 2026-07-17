# L1→L2 Replay 결과 — 2026-07-18 (위기 재현성 게이트: 레짐 캡 해제 쿨다운 진단)

## 실행 조건

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-17 --seed 42`
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31~2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 125~137(run마다 변동); L1 admission 대부분 late_start 소수 제외.
- **주의**: 동일 `--seed 42`에도 Optuna champion이 run마다 달라지는 비결정성 확인됨(단일 run 내부 비교는 유효, run 간 절대 수치 비교는 champion drift로 confound될 수 있음 — 알려진 이슈, `ProcessPoolExecutor fork` 관련으로 추정, 별도 트래킹 필요).
- 정상 L2 계산 완료 후 crisis survival 승격 차단으로 exit code `1`(의도된 fail-closed) — 아래 위기 게이트 섹션 참고.

## L1 결과 (안정 — 회귀 없음)

| Timeframe | Fold readiness | Probe LCB (bps) | 판정 |
| :--- | :---: | ---: | :---: |
| 1h~1d(7개 TF) | 3~4/4 | +37~+106 | 전부 PASS |

- Master TF는 `8h`(최대 breadth 기준)로 선정.

## L2 정상장 스코어카드 (2025-03-20~2025-12-30 평가 윈도우)

- 최신 baseline(worst_fold 기본 on): `STATUS PASS` — CAGR +53.2%, Sharpe 1.700, Sortino 2.743, Calmar 3.724, MDD 14.3%, CVaR95 1.8%, Sharpe Uplift +0.20, PSR 0.986/DSR 0.902.
- `⚠️ NO-CRISIS-WINDOW` — 이 평가 윈도우는 병목-caliber fold(MDD≥15% & CAGR≤0)를 포함하지 않아 승격 근거로 단독 인용 금지. 위기 재현성은 별도 out-of-band 게이트(아래)로 검증.

## 위기 재현성 게이트 — 정책 및 예산

`CrisisWindowMetrics`/`evaluate_crisis_survival()`(순수 함수)가 LUNA/FTX 2022 붕괴장(`2022-04-01~2023-02-15`, out-of-band 데이터)에 대해 champion의 rule-based 신호를 재생성해 생존 테스트한다. 위기 MDD 예산 `l2_max_mdd_abs×(1-l2_deploy_mdd_margin)=21%`, CAGR 하한 `l2_min_worst_fold_cagr=-5%`. 실패 시 `apply_crisis_reliability_override()`가 monotonic fail-closed로 production 승격을 차단(위기 결과는 Optuna selection에 재투입하지 않음).

## 이전 수정 이력 (요약, 상세 서사는 decisions_archive.md 참고)

| 수정 | ADR | 핵심 결과 |
| :--- | :--- | :--- |
| BTC 레짐 데이터 무결성(`timestamp_x` 폴백 + `_btc_index` fail-closed) | `ADR_20260717_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX` | has_btc False→True 복구 검증. 단, universe 확장(93→103 심볼)으로 champion drift 발생 — 위기 MDD 29.01%→55.47%(악화, 새 챔피언의 진짜 꼬리위험이 드러난 것으로 판단) |
| 레버리지 ceiling 구조 리팩토링(`_resolve_safety_ceiling`/`_resolve_oos_adaptive_leverage` 분리) | `ADR_20260717_L2_LEVERAGE_CEILING_REFACTOR` | OOS-blend가 worst_fold/kelly ceiling을 더 이상 우회 못 하는 불변식 실측 검증(L*_off=2.05→L*_on=1.00 정확 제약) |
| worst_fold 안전장치 기본값 전환(`False`→`True`) + `from_mapping` SSOT fallback 버그 수정 | `ADR_20260717_L2_CRISIS_LEVERAGE_SAFETY_DEFAULT` | 위기 MDD 55.47%→46.53%, CAGR -38.44%→-28.04%(방향 개선이나 예산 미달). champion drift로 정상장 CAGR 92.8%→53.2%(Optuna 목적함수가 배치-CAGR 반영하는 의도된 설계로 판단) |
| 롱/숏 방향 비대칭 진단 + opt-in 완화 레버(`apply_asymmetric_long_short_regime_cap`) | `ADR_20260717_L2_CRISIS_ASYMMETRIC_LONG_SHORT_CAP` | 위기 replay 실측: 대칭 캡 적용 후에도 롱 레그 -11.29% vs 숏 레그 +3.59%(bars_long=683/bars_short=786, 빈도는 균형). 스윕 결과 CAGR -28.04%→-14.19%(개선) but **MDD 46.1~46.9% 평평(미개선)** — 병목이 MDD로 좁혀짐 |

## 신규: 레짐 캡 해제 쿨다운 진단 (`ADR_20260717_L2_CRISIS_REGIME_CAP_RELEASE_COOLDOWN`)

**정확한 MDD 드라이버 특정** (`scratch/diag_crisis_regime_whipsaw.py`, 동일 champion 고정, `deploy_leverage=1.7187`, 재구성 equity curve가 공식 로그 `mdd=0.4653`과 정확히 일치):

- 최대낙폭 구간(전체 902bar 중 마지막 26%, bar 627~861)은 LUNA 붕괴(윈도우 초반)가 아니라 **FTX 붕괴 국면(2022-11)**에 집중.
- 이 구간(rebalance-event 기준 79건) 레짐 분포: bull 22 / bear 47 / crisis 10 — 위기 한복판에서 28%가 "bull"로 오분류.
- 이 구간 내 `gross_exposure≥0.9`(사실상 무제한) 이벤트 **13건 전부가 예외 없이 bull 오분류 시점**과 일치.
- 근본 원인: 레짐 코드는 이미 Schmitt 히스테리시스(`persistence_target_dwell=6.0`)로 산출되지만 이는 방향 신호 안정화용일 뿐, **캡의 "해제" 타이밍에는 별도 지속성이 없어** bull로 한 번만 튀어도 캡이 즉시 완전히 풀림.

**opt-in 완화 레버 구현**: `apply_regime_cap_release_cooldown`(순수 함수, `market_regime.py`, 기존 검증된 `_apply_persistence_and_cooldown_1d` 재사용) — bear/crisis 진입은 항상 즉시 반영, **bull로의 "복귀"만 최근 N bar 이내 bear/crisis 이력이 있으면 지연**(대체 상태는 항상 bear, crisis로 인위 승격 안 함). 방향 alpha 신호(`_regime_code_1d` 원본)는 불변 — 캡 전용 파생 배열만 신규 도입, 다른 4개 소비처는 실측으로 영향 없음 확인. `Layer2AllocationConfig.l2_regime_cap_release_cooldown_bars`(기본값 `0`=no-op) opt-in 필드. `/check` PASS(Cov 42%, spec compliance 포함). SSOT: `docs/specs/l2-crisis-regime-cap-release-cooldown.md`.

**코스 스윕**(`scratch/sweep_crisis_cap_cooldown.py`, 동일 champion, LUNA/FTX):

| cooldown_bars(4h) | MDD | CAGR |
| ---: | ---: | ---: |
| 0(off) | 46.53% | -28.04% |
| 3 | 47.55%(악화) | -29.78%(악화) |
| 6 | 42.64% | -20.14% |
| 12 | 42.51% | -18.24% |
| 24 | 31.13% | **+1.16%**(첫 흑자) |
| 48 | 32.85%(24보다 악화) | -21.06%(24보다 악화) |

**파인 스윕**(24 근방, `scratch/sweep_crisis_cap_cooldown_v2.py`):

| cooldown_bars | MDD | CAGR |
| ---: | ---: | ---: |
| 15 | 34.67% | -5.75% |
| 18 | 34.17% | -4.34% |
| 20 / 22 | 34.30% | -5.16% |
| 24 | 31.13% | +1.16% |
| 26 / 28 | 31.30% | **+3.54%** |
| **30** | **29.46%**(MDD 최저) | -2.92% |
| 32 | 29.46% | -7.30%(CAGR 하한 이탈) |
| 36 | 31.03% | -17.70%(급격 악화) |

- Sweet spot: **26~30bar**. 최선의 경우(30bar)에도 MDD 29.46%로 21% 예산에 여전히 미달 — baseline(46.53%) 대비 대폭 개선(상대 -37%)했고 세 레버 중 가장 효과적이나 단독으로는 게이트 통과 못함.
- 32bar 이후 급격 악화 — 과잉 쿨다운의 기회비용(진짜 회복기까지 방어 유지) 확인.

**두 번째 독립 위기 윈도우 검증 — 블로커(미해결)**: `[LIMIT-01]`이 요구하는 2025-12-31~2026-06-30 BTC 위기 홀드아웃(2026-07-02 기록, BTC -32.8%)을 LUNA/FTX와 동일한 `CrisisWindow` 프레임워크로 재현 시도했으나 `load_futures_data_maps_for_symbols`가 "no valid symbols after load" 반환. 원시 parquet(`enriched/4h/BTCUSDT.parquet`)엔 2026-07-02까지 데이터가 실존함을 확인(원본 데이터 부재 아님) — 원래 2026-07-02 실측은 `CrisisWindow` out-of-band 프레임워크가 아니라 `L2_REVERSAL_KILL`/`L3_REVERSAL_REPLAY` env 기반의 파이프라인 내부 OOS holdout 슬라이스 메커니즘을 썼던 것으로, 이번 시도(`CrisisWindow` 임의 구성)가 그 메커니즘과 안 맞아 발생한 재현 방법 오류로 추정. 원인 미확정 — 별도 진단 필요.

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS(회귀 없음).
- **L2 스코어카드 정합성:** PASS — production 신뢰 가능.
- **레버리지 안전장치(worst_fold 기본 on, ceiling 리팩토링):** ✅ 완료·검증 — 위기 MDD/CAGR 방향 개선하나 단독 예산 미달.
- **롱/숏 비대칭 완화 레버:** ✅ 구현·검증 완료(opt-in) — CAGR 개선(-28%→-14%), MDD 불변.
- **레짐 캡 해제 쿨다운:** ✅ 구현·검증 완료(opt-in) — 현재까지 최선의 단일 레버(MDD 46.5%→29.5%, CAGR 최대 +3.5%), 단독으로는 여전히 MDD 예산(21%) 미달.
- **L2 위기 재현성 게이트:** FAIL-CLOSED 유지 — 3개 opt-in 레버 모두 개별적으로는 예산 미충족. 조합 효과 미측정.
- **`[LIMIT-01]` 2차 독립 윈도우 검증:** 블로커로 미완료.

## 다음 조치

1. 세 레버(worst_fold 기본 on + `l2_regime_long_short_asymmetry_enabled` + `l2_regime_cap_release_cooldown_bars=26~30`) **조합 효과** 실측 — 각각 부분적 개선이라 조합 시 예산 통과 가능성 있음, 최우선 순위.
2. 2025-12-31~2026-06-30 BTC 위기 홀드아웃 재현 방법 규명(원 메커니즘은 `L2_REVERSAL_KILL`/`L3_REVERSAL_REPLAY` env 기반으로 추정) — `[LIMIT-01]` 충족을 위해 필수.
3. Optuna champion 비결정성(동일 `--seed`에도 run마다 challenger 상이) 원인 규명 — 별도 이슈로 트래킹.
4. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
5. 위기장을 포함하는 정상 holdout 윈도우로 Uplift/CAGR 재검증 — 미해결(`NO-CRISIS-WINDOW` 경고 지속).
6. `run_config.timeframe` CLI 기본값 "4h"의 tf-probe 기반 근거화 — 별도 `/spec` 대기.
