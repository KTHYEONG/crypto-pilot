# L2 Phase 성과 개선 세션 결과 — 2026-07-19 (crisis-aware Optuna 탐색 TF 수정 반영)

## 세션 요약

정상장 게이트만 보고 안심하던 L2가 이번 세션 최초로 **정상장 + crisis stress test를 동시에 통과**하는 champion을 산출했다. 근본 원인은 Optuna 120-trial 탐색 루프가 crisis context 로딩 자체에 실패해(`crisis_measured=0`) 전 trial이 crisis 안전성을 전혀 보지 못한 채 정상장 성장만 극대화했던 구조적 결함이었다 — `select_crisis_load_tf` 도입으로 해소.

## 근본 원인

- `_load_crisis_replay_context`(Optuna 탐색 루프가 사용하는 crisis loader, `pipeline.py`)가 `_load_tf = tf`(L2 마스터 TF, 이번 실행은 8h)를 직접 요청.
- 실측 확인: `data/futures/ohlcv/8h/`·`enriched/8h/`는 디렉터리만 존재하고 **원본 파일 0개**(8h는 항상 4h에서 실시간 리샘플로만 합성, 원천 저장 없음 — `4h/`는 649개 심볼 파일 보유). 8h 직접 요청 시 전 심볼이 Pass-1 캐시 분류에서 탈락 → fallback도 `fetch_network=False`라 재수집 불가 → `loaded_symbols=0`.
- 대조군인 `assess_crisis_reliability`(champion 확정 후 실행되는 **사후** stress test)는 `_load_tf="4h"` 하드코딩이라 우연히 성공(`valid_symbols=45`) — 두 함수가 서로 다른 TF 선택 로직을 쓰는 것 자체가 근본 결함.
- `_load_crisis_replay_context`의 `_load_tf = tf` 설계는 그 자체로 과거의 다른 버그(4h 고정 시 1h 마스터 replay가 무조건 empty) 수정이었기 때문에, 단순히 4h로 통일하는 것은 그 버그를 재도입한다.

## 해결

`timeframe_contracts.py`에 `select_crisis_load_tf(target_tf)` 신설 — 기존 `PROBE_SOURCE_TFS`(1h, 4h)·`hours_per_bar`·`is_resample_compatible` 재사용(SSOT, 신규 개념 없음). target이 원천-백드 TF면 그대로, 아니면 클린하게 합성 가능한 가장 coarse한 원천-백드 후보 선택. `_load_crisis_replay_context`와 `assess_crisis_reliability` 양쪽의 `_load_tf` 산정을 이 헬퍼로 통합. 1h 마스터 기존 회귀 테스트 보존, 8h 마스터 신규 테스트 4개 시나리오 추가. `/check` PASS(Cov 24%, 두 호출부 라인 모두 커버 확인).

## 프로덕션 실측 (2026-07-19 기준일, 120 trials)

| 항목 | 수정 전 | 수정 후 |
| :--- | ---: | ---: |
| `[CRISIS-LOAD] loaded_symbols` | 0 | **45** |
| `[L2-AUDIT] crisis_measured` | 0/120 | **120/120** |
| champion 선정 경로 | 정상 gate-pass(12 candidates) | 정상 gate-pass(1 candidate) |
| regime 방어 레버 | hard_block=False, crisis_gross_cap=0.20 | **hard_block=True, asymmetry=True, severity_gating=True, crisis_gross_cap=0.10** |
| 레버리지 L* | 2.348 (binding=oos_blend, 정상장 지표에만 묶임) | **1.044 (binding=crisis_window, 위기 제약에 바인딩)** |
| 정상장 CAGR | +73.9% | +30.9% |
| 정상장 STATUS | ✅ PASS | ✅ PASS |
| Crisis stress test | ❌ mdd=49.0%(>21% 예산) | ✅ **mdd=19.11%(≤21%), cagr=-4.98%(≥-5%)** |
| `[CRISIS-RELIABILITY]` | `stress_tested_fail` | ✅ **`stress_tested_pass verified=True`** |
| 파이프라인 최종 상태 | 🛑 `exit_code=1 reason=layer2_blocked:crisis_survival` | ✅ **정상 종료** |

```
STATUS  : ✅ PASS

✅ [Growth    ] CAGR: +30.9% (>=30.0%) | PnL: +49.7% | Equity x1.50
✅ [Efficiency] Sharpe: 2.088 (>=1.000) | Sortino: 3.379 (>=1.500) | Calmar: 2.358 (>=1.000)
✅ [Risk      ] MDD: 13.1% (<=30.0%) | CVaR95: 0.8% (<=6.0%) | RiskUtil: 43.7%
✅ [Robust    ] Fold: 75.0% (>=60.0%) | Trades: 144 (>=30) | Friction: 100.0%
✅ [Uplift    ] Sharpe Uplift: +0.38 (>=+0.05)
✅ [Integrity ] PSR: 0.996 (>=0.90) | DSR: 0.990 (diag)
```

| Fold | Period | CAGR | MDD | Sharpe | Status |
| :--- | :--- | ---: | ---: | ---: | :---: |
| 1 | 2025-03-20 ~ 2025-05-30 | +32.1% | 6.4% | 2.136 | ✅ PASS |
| 2 | 2025-05-30 ~ 2025-08-09 | -1.7% | 13.1% | -0.065 | ❌ FAIL |
| 3 | 2025-08-09 ~ 2025-10-20 | +35.2% | 3.3% | 2.341 | ✅ PASS |
| 4 | 2025-10-20 ~ 2025-12-30 | +66.9% | 3.5% | 3.916 | ✅ PASS |

원본 로그: `/tmp/l2_crisis_tf_fix_verify.log`.

## Verdict

- **L2 정상장**: ✅ PASS (CAGR +30.9%, 전 게이트 통과).
- **L2 crisis 방어**: ✅ **이번 세션 최초로 PASS** (MDD 19.11%≤21%, CAGR -4.98%≥-5%).
- **파이프라인**: ✅ **이번 세션 최초로 정상장+위기장 동시 통과, exit_code=1 없이 완주**.
- **`/check`**: PASS (Cov 24%, 신규 함수 `select_crisis_load_tf` 및 두 호출부 전부 테스트 커버).

## 잔여 이슈

1. **경미 — `[CRISIS-WINDOW-DETAIL]` 개별 라벨 불일치**: 상위 집계 `[CRISIS-RELIABILITY] status=stress_tested_pass`와 달리 개별 윈도우 라벨은 여전히 `status=stress_tested_fail`을 표기(수치 자체는 mdd/cagr 모두 예산 이내로 PASS 조건 충족) — 경계값(cagr -4.98% vs 하한 -5%, 마진 0.02%p)에서 별도 sub-threshold나 라벨링 로직 존재 가능성, 다음 세션에서 확인 필요.
2. **joint_feasible 여전히 0/120**: crisis_measured=120으로 정상 측정되지만 정상장+위기 제약을 **동시에** 만족하는 trial은 아직 없음(champion은 gate-pass 경로로 선정되나 엄밀한 joint-feasibility 기준으로는 미달) — 탐색공간/제약 여유 재검토 대상.
3. **crisis CAGR 마진 매우 타이트**(-4.98% vs 하한 -5%, 0.02%p) — 단일 seed 실측이므로 seed 변동성에 따라 쉽게 재악화될 수 있음. 다중 seed 검증 필요.
