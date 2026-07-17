# L1→L2 최신 Replay 결과 — 2026-07-17 (TF-inclusion 게이트(C4) native-TF 회귀 수정 후)

## 실행 조건과 데이터 무결성

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-17 --seed 42`
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31 ~ 2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 125; L1 admission: 114/125 (`late_start` 11 제외).
- `docs/specs/l2-tf-inclusion-gate-native-tf-fix.md` 구현(`_build_sleeve_tf_lookup` 도입) 반영 후 재실행 — 이전 replay(2026-07-16)에서 노출된 `mdd=0.0000 cagr=+0.0000` 위기 재현성 게이트 회귀의 근본 수정.

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

## L2 결과 — 스코어카드 (2025-03-20 ~ 2025-12-30 평가 윈도우)

```text
STATUS  : ✅ PASS
✅ [Growth    ] CAGR: +61.2% (>=30.0%) | PnL: +67.2% | Equity x1.67
✅ [Efficiency] Sharpe: 2.026 (>=1.000) | Sortino: 3.875 (>=1.500) | Calmar: 3.082 (>=1.000)
✅ [Risk      ] MDD: 19.9% (<=30.0%) | CVaR95: 1.4% (<=6.0%) | RiskUtil: 66.2%
✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 1115 (>=30) | Friction: 99.4%
✅ [Uplift    ] Sharpe Uplift: +0.10 (>=+0.05)
✅ [Integrity ] PSR: 0.999 (>=0.90) | DSR: 0.965 (diag)
```

- 스코어카드 자체(Bug-A1/A2, 하드코딩 임계값→config SSOT, PSR/DSR 표시 역할)는 전날(2026-07-16) replay에서 이미 실측 확인 완료 — 이번 run에서도 수치 동일하게 재확인(회귀 없음).
- `⚠️ [WINDOW] NO-CRISIS-WINDOW` — 이 평가 윈도우(2025-03-20~2025-12-30)는 병목-caliber fold(MDD>=15% & CAGR<=0)를 포함하지 않으므로, 위 스코어카드 자체는 승격 근거로 단독 인용 금지(`docs/results/next.md` P0 기존 경고 유지). 위기 재현성은 별도 out-of-band 게이트(아래)로 검증한다.

## L2 위기 재현성 게이트 — TF-inclusion(C4) native-TF 회귀 수정 실측 확인

전날(2026-07-16) replay에서 `assess_crisis_reliability()`가 2022 LUNA+FTX 붕괴장(`2022-04-01~2023-02-15`)에 대해 champion의 rule-based 신호를 재생성해 생존 테스트했을 때 `mdd=0.0000 cagr=+0.0000`이 정확히 0으로 나와 원인 규명이 필요했다.

**근본 원인(코드 조사로 확정)**: crisis-stress 전용 버그가 아니라, 커밋 `c2831990`(2026-07-16, "L1→L2 네이티브 TF 핸드오프 무결성")이 남긴 전역 회귀였다. 그 커밋이 `strategy_id` 포맷을 `"donchian_72_8h"`(TF 접미사 포함) → `"family:variant"`(TF 접미사 없음, TF는 별도 `native_tf` 필드로 분리)로 바꿨는데, C4 TF-inclusion 게이트의 OOS 필터(`awf_sim.py:3205,3208`)가 옛 정규식 파서 `_parse_tf_from_strategy_id`로 여전히 `strategy_id`를 파싱하고 있었다. 이제 TF 접미사가 없으므로 파서는 항상 `"unk"`를 반환 → `included_tfs_by_fold`(실제 TF 문자열 집합)와 매칭 실패 → **매 OOS bar마다 전체 sleeve 탈락**. `l2_tf_inclusion_enabled`(기본값 `True`)가 켜진 모든 L2/L3 경로에 적용되는 회귀였다.

**수정**: `docs/specs/l2-tf-inclusion-gate-native-tf-fix.md` 구현 — `cache.sleeve_to_tf`(SSOT, `SignalSleeveKey.native_tf`)로 `(symbol, strategy_id) → native_tf` 룩업(`_build_sleeve_tf_lookup`)을 구축해 OOS 필터가 파싱 대신 직접 조회하도록 변경. 죽은 파서 함수와 스텁 테스트(S4)도 함께 정리. `/check` PASS(spec compliance + ruff/mypy/pytest, Cov 45%).

**A/B 실측 대조** (scratch 스크립트로 `assess_crisis_reliability`를 champion-형 registry에 직접 실행, `git stash`로 전/후 비교):

| | 수정 전(회귀) | 수정 후(fix) |
| :--- | ---: | ---: |
| status/verified | stress_tested_pass / True | stress_tested_pass / True |
| mdd | 0.0000 | 0.0635 |
| cagr | +0.0000 | -0.0093 |

**본 replay(2026-07-17, 전체 L2 파이프라인 실행, 실제 champion registry 93개 심볼)**:

```text
[CRISIS-STRESS] window=luna_ftx_2022_collapse registry_symbols=93 overlap_symbols=37
[CRISIS-STRESS] window=luna_ftx_2022_collapse valid_symbols=35
[CRISIS-STRESS] window=luna_ftx_2022_collapse resampled_symbols=35
[CRISIS-STRESS] window=luna_ftx_2022_collapse aligned_bars=903
[CRISIS-STRESS] window=luna_ftx_2022_collapse events=50532
[CRISIS-RELIABILITY] status=stress_tested_pass verified=True detail=luna_ftx_2022_collapse: mdd=0.2901 cagr=-0.3273 symbols=35
```

- 더 이상 0.0000이 아니며, 실제 포지션이 생성되고 계산되는 정상 동작으로 전환됨을 실측 확인.
- `mdd=0.2901 cagr=-0.3273` — LUNA/FTX 붕괴장에서 champion이 실제로는 자산의 29.0%를 잃고 연환산 -32.7% 손실을 냈다는 뜻이다. `verified=True`(파이프라인 완주 기준)이지만, **이 수치는 "위기를 견뎌냈다"는 근거가 아니라 오히려 위기 방어력 부재를 시사** — `l2_max_mdd_abs` 등 하드 리스크 한도 대비 이 수치가 실제로 게이트를 통과해야 하는지 별도 정책 검토 필요(다음 조치 참고).

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS (회귀 없음).
- **L2 스코어카드 정합성(Bug-A1/A2):** PASS — 실측 확인 완료, production 신뢰 가능.
- **L2 위기 재현성 게이트(TF-inclusion 회귀):** ✅ 수정 완료 및 실측 검증 — 수치가 더 이상 0이 아니며 파이프라인이 정상적으로 포지션을 생성·평가함을 확인. 단, 산출된 mdd/cagr 자체(-32.7% CAGR)가 위기 방어 관점에서 허용 가능한지는 별도 정책 판단 필요.

## 다음 조치

1. **[신규]** LUNA/FTX 붕괴장 실측 `mdd=0.2901 cagr=-0.3273`이 `l2_max_mdd_abs` 등 기존 위기 리스크 한도 대비 PASS/FAIL 판정에 실제로 반영되는지 확인 — 현재 `assess_crisis_reliability`는 `verified=True`(배선 성공)만 판단하고 임계값 비교(FAIL 조건)는 별도 구현 여부 확인 필요.
2. 위기장(crisis-caliber fold)을 포함하는 정상 holdout 윈도우로 별도 replay해 Uplift/CAGR을 재검증할 것 — 여전히 미해결(`⚠️ NO-CRISIS-WINDOW` 경고 지속).
3. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
4. `docs/results/next.md` §1(`run_config.timeframe` CLI 기본값 "4h"의 tf-probe 기반 근거화)은 별도 `/spec` 대기 중.
