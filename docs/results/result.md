# L1→L2 최신 Replay 결과 — 2026-07-17 (L2 스코어카드 SSOT화 + 위기 재현성 게이트 적용 후)

## 실행 조건과 데이터 무결성

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-16 --seed 42`
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31 ~ 2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 125; L1 admission: 114/125 (`late_start` 11 제외).
- `docs/decisions/decisions.md`의 `ADR_20260717_L2_GATE_SCORECARD_AND_CRISIS_RELIABILITY` 구현 반영 후 재실행.

## L1 결과 (수정 전과 수치 완전 동일 — 회귀 없음)

| Timeframe | Fold readiness | Symbol breadth | Probe LCB (bps) | Final promoted pairs | 판정 |
| :--- | :---: | ---: | ---: | ---: | :---: |
| 1h | 4/4 | 68.554 | +68.850 | 237 | PASS |
| 2h | 4/4 | 86.185 | +106.455 | 164 | PASS |
| 4h | 3/4 | 44.579 | +37.238 | 59 | PASS |
| 6h | 4/4 | 29.755 | +49.121 | 14 | PASS |
| 8h | 4/4 | 105.662 | +80.662 | 186 | PASS |
| 12h | 4/4 | 106.333 | +73.830 | 98 | PASS |
| 1d | 4/4 | 95.006 | +83.277 | 3 | PASS |

- Master TF는 이번 run에서도 `8h`로 선정(최대 breadth, `assess_l1_tf_handoff` 기준).

## L2 결과 — 스코어카드 표시 버그 2건 수정 실측 확인

**[Bug-A1] 하드코딩 임계값 → config SSOT 수정 확인**:

| 항목 | 값 | 게이트(수정 전 표시) | 게이트(수정 후, config 기준) | 판정 |
| :--- | ---: | :--- | :--- | :---: |
| Sharpe Uplift | +0.10 | `>=+0.20` (하드코딩, ❌) | `>=+0.05` (`config.l2_min_sharpe_uplift`, ✅) | ✅ |

이전 replay에서 `Sharpe Uplift +0.10`이 `❌`로 표시됐던 것은 실제 게이트 임계값(0.05)과 무관한 화면 버그였음이 실측으로 확인됐다 — 전체 `STATUS: PASS`와 하위 라인의 자기모순이 해소됐다.

**[Bug-A2] DSR/PSR 표시 역할 정정 확인**:

```text
✅ [Integrity ] PSR: 0.999 (>=0.90) | DSR: 0.965 (diag)
```

실제 하드 블로커인 PSR이 게이트(✅/❌)로, diagnostic-only인 DSR이 `(diag)`로 올바르게 표시된다(수정 전엔 반대였음).

**[Bug-B] 위기 재현성 게이트 — 배선 완주 확인, 수치는 후속 검증 필요**:

`assess_crisis_reliability()`가 L1/L2/L3가 전혀 보지 않은 out-of-band 역사적 붕괴장(2022 LUNA+FTX 붕괴, `2022-04-01~2023-02-15`)에 대해 champion의 이미 확정된 전략 정체성(rule-based family/variant, 재학습 없음)을 그대로 적용해 생존 테스트한다:

```text
[CRISIS-STRESS] window=luna_ftx_2022_collapse registry_symbols=93 overlap_symbols=37
[CRISIS-STRESS] window=luna_ftx_2022_collapse valid_symbols=35
[CRISIS-STRESS] window=luna_ftx_2022_collapse resampled_symbols=35
[CRISIS-STRESS] window=luna_ftx_2022_collapse aligned_bars=903
[CRISIS-STRESS] window=luna_ftx_2022_collapse events=50532
[CRISIS-RELIABILITY] status=stress_tested_pass verified=True detail=luna_ftx_2022_collapse: mdd=0.0000 cagr=+0.0000 symbols=35
```

- 배선(데이터 로드 → 4h→8h 리샘플링 → 규칙기반 신호 재생성 → 시뮬레이션 → 게이트 반영)이 에러 없이 끝까지 실행되고 최종 `RunnerResult(exit_code=0)`로 정상 완료됨을 확인.
- **⚠️ `mdd=0.0000 cagr=+0.0000`이 정확히 0으로 산출** — 위험한도를 안전하게 지켰다기보다 시뮬레이션이 실제 포지션을 전혀 잡지 않았을 가능성이 있다. `verified=True`이므로 이번 run의 promotion 판정에 실질적 영향은 없었으나(원래도 정상 PASS 경로), **이 수치 자체를 "위기 생존 검증됨"의 근거로 인용하는 것은 금지** — 원인 규명 전까지는 배선 검증(파이프라인이 안 죽고 끝까지 돈다)만 확인된 상태로 취급한다.
- 배선 검증 과정에서 3개의 독립 버그를 추가 발견·수정: (1) 위기 구간 275일이 1d 최소 300-bar 요건 미달 → 320일로 확장, (2) `8h`는 정적 enriched 파일이 없는 파생 TF라 4h 로드 후 리샘플링 필요, (3) `align_data_maps()`의 `cache_result=False` 시 `UnboundLocalError`(pre-existing 버그, `cache_key` 미정의) — 이번 기능과 무관하게 상시 존재하던 버그였다.

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS (회귀 없음).
- **L2 스코어카드 정합성(Bug-A1/A2):** PASS — 실측 확인 완료, production 신뢰 가능.
- **L2 위기 재현성 게이트(Bug-B):** 배선 PASS(에러 없이 완주) — **수치(mdd/cagr=0.0000) 신뢰도는 미검증**, 원인 규명 전까지 참고용으로만 취급.

## 다음 조치

1. `mdd=0.0000/cagr=+0.0000` 정확히 0으로 나오는 원인 규명 — `_build_rule_based_stress_batch`의 사이징(quality_weight/Kelly weight)이 실제로 0이 되는지, 혹은 `run_l3_holdout` 내부에서 이 합성 registry/batch가 포지션을 만들지 못하는지 확인 필요.
2. 위기장(crisis-caliber fold)을 포함하는 정상 holdout 윈도우로 별도 replay해 Uplift/CAGR을 재검증할 것 — 여전히 미해결.
3. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
4. `docs/results/next.md` §1(`run_config.timeframe` CLI 기본값 "4h"의 tf-probe 기반 근거화)은 별도 `/spec` 대기 중.
