# L1→L2 최신 Replay 결과 — 2026-07-17 (BTC 레짐 데이터 무결성 수정 후)

## 실행 조건과 데이터 무결성

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-17 --seed 42`
- 실행 결과: 정상 L2 계산 완료 후 crisis survival 승격 차단으로 exit code `1` (의도된 fail-closed)
- L2 Optuna: 200 trials, single-process replay
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31 ~ 2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 125; L1 admission: 114/125 (`late_start` 11 제외).
- native-TF lookup(`_build_sleeve_tf_lookup`) 구현 반영 후 재실행 — 2026-07-16 replay에서 노출된 `mdd=0.0000 cagr=+0.0000` 위기 재현성 게이트 회귀는 제거된 상태.
- **신규**: `_resolve_timestamp_column`(opt_data_utils.py) + `_btc_index` fail-closed 전환(market_regime.py) 반영 후 재실행 — 아래 "BTC 레짐 데이터 무결성 수정" 섹션 참고.

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

## L2 위기 재현성 게이트 — Crisis Survival fail-closed 실측 확인

전날(2026-07-16) replay에서 `assess_crisis_reliability()`가 2022 LUNA+FTX 붕괴장(`2022-04-01~2023-02-15`)에 대해 champion의 rule-based 신호를 재생성해 생존 테스트했을 때 `mdd=0.0000 cagr=+0.0000`이 정확히 0으로 나와 원인 규명이 필요했다.

**근본 원인(코드 조사로 확정)**: crisis-stress 전용 버그가 아니라, 커밋 `c2831990`(2026-07-16, "L1→L2 네이티브 TF 핸드오프 무결성")이 남긴 전역 회귀였다. 그 커밋이 `strategy_id` 포맷을 `"donchian_72_8h"`(TF 접미사 포함) → `"family:variant"`(TF 접미사 없음, TF는 별도 `native_tf` 필드로 분리)로 바꿨는데, C4 TF-inclusion 게이트의 OOS 필터(`awf_sim.py:3205,3208`)가 옛 정규식 파서 `_parse_tf_from_strategy_id`로 여전히 `strategy_id`를 파싱하고 있었다. 이제 TF 접미사가 없으므로 파서는 항상 `"unk"`를 반환 → `included_tfs_by_fold`(실제 TF 문자열 집합)와 매칭 실패 → **매 OOS bar마다 전체 sleeve 탈락**. `l2_tf_inclusion_enabled`(기본값 `True`)가 켜진 모든 L2/L3 경로에 적용되는 회귀였다.

**기존 C4 수정**: `cache.sleeve_to_tf`(SSOT, `SignalSleeveKey.native_tf`)로 `(symbol, strategy_id) → native_tf` 룩업(`_build_sleeve_tf_lookup`)을 구축해 OOS 필터가 파싱 대신 직접 조회하도록 변경. 이로써 0.0000 replay 회귀는 제거됐다.

**최신 정책 수정**: `CrisisWindowMetrics`와 순수 `evaluate_crisis_survival()`를 도입해 모든 configured crisis window를 데이터 충분성·MDD·CAGR·CVaR·거래 수로 함께 판정한다. 위기 MDD 예산은 `l2_max_mdd_abs × (1 - l2_deploy_mdd_margin) = 21%`, CAGR 하한은 `l2_min_worst_fold_cagr = -5%`이며, 실패 시 `apply_crisis_reliability_override()`가 L2 승격을 monotonic fail-closed로 차단한다. 위기 결과는 Optuna selection에 재투입하지 않는다.

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
[CRISIS-RELIABILITY] status=stress_tested_fail verified=False detail=luna_ftx_2022_collapse:mdd_abs; luna_ftx_2022_collapse:cagr
```

- 실제 replay는 35개 심볼·903 bars·50,532 events에서 포지션을 생성했다. 기존 동일 champion replay의 수치 `MDD=29.01%`, `CAGR=-32.73%`가 새 예산을 각각 초과/하회하여 fail-closed가 발생했다.
- 정상장 스코어카드가 PASS여도 위기 검증 실패 상태에서는 production 승격이 금지된다.

## BTC 레짐 데이터 무결성 수정 (`ADR_20260717_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX`)

**근본 원인(실측 확인)**: `assess_crisis_reliability`가 LUNA/FTX 윈도우를 로드할 때, BTCUSDT를 포함한 일부 심볼(전체 enriched 파일의 ~4%, 표본 확인)의 parquet가 `pd.merge(..., suffixes=("_x","_y"))` coalesce 누락으로 `timestamp` 대신 `timestamp_x`/`timestamp_y`만 갖고 있어 `load_single_symbol_data`의 3단 폴백(Arrow pushdown → full-read → `datetime` 파생)이 전부 실패, 해당 심볼이 예외 없이 `valid_symbols`에서 침묵 탈락했다(`scratch/probe_crisis_regime.py` 실행으로 `has_btc=False` 직접 확인). BTC가 빠지면 `market_regime._btc_index()`가 경고 없이 `return 0`(임의 심볼로 대체)해, 이미 프로덕션에 활성화된 regime-conditional 익스포저 캡(`apply_regime_risk_cap`, bull=1.0/bear=0.35/crisis=0.25)이 엉뚱한 심볼 가격으로 레짐을 오판정하고 있었다.

**수정**: `opt_data_utils.py`에 `_resolve_timestamp_column` 헬퍼 추가(`timestamp` 부재 시 `timestamp_x` 폴백). `market_regime._btc_index()`는 BTC 부재 시 `ValueError`로 fail-closed 전환. `active_pipeline.py`의 `[CRISIS-RELIABILITY]` 로그에 윈도우별 raw MDD/CAGR/CVaR detail 추가(다음 조치 §3 해소). 사전 존재하던 테스트 버그(`FUTURES_DATA_DIR`를 `opt_data_utils` 모듈에 잘못 monkeypatch — 실제 소유자는 `src.core.settings`)도 함께 수정해 xfail 7건을 실통과로 전환.

**메커니즘 레벨 검증** (`scratch/probe_crisis_regime.py`, `probe_crisis_regime_cap.py` 직접 실행):

| | 수정 전 | 수정 후 |
| :--- | ---: | ---: |
| has_btc | False | **True** |
| valid_symbols(58개 윈도우 심볼 기준) | 43 | **55** |
| BTC peak/trough | (오염됨) | **$42,418(2022-04-21) → $15,712.8(2022-11-21), 실제 역사와 일치** |

- BTC 앵커 복구 후 압축 3-state 분포: bull 27.0% / bear 39.4% / crisis 33.6% — 평균 gross cap 승수 0.492(원 노출의 절반 수준으로 자동 축소). LUNA 붕괴 구간(61봉)은 bull 0 / bear 34 / crisis 27로 전 구간 축소 모드.

**전체 파이프라인 재실행 결과(실제 champion registry, 2026-07-17)**:

```text
[CRISIS-STRESS] window=luna_ftx_2022_collapse registry_symbols=103 overlap_symbols=47
[CRISIS-STRESS] window=luna_ftx_2022_collapse valid_symbols=45
[CRISIS-STRESS] window=luna_ftx_2022_collapse resampled_symbols=45
[CRISIS-STRESS] window=luna_ftx_2022_collapse aligned_bars=903
[CRISIS-STRESS] window=luna_ftx_2022_collapse events=67593
[CRISIS-RELIABILITY] status=stress_tested_fail verified=False detail=luna_ftx_2022_collapse:mdd_abs; luna_ftx_2022_collapse:cagr
[CRISIS-WINDOW-DETAIL] label=luna_ftx_2022_collapse status=stress_tested_fail mdd=0.5547 cagr=-0.3844 cvar95=0.0458 trades=373 symbols=45
```

| | 수정 전 | 수정 후 |
| :--- | ---: | ---: |
| overlap_symbols | 37 | 47 |
| valid_symbols | 35 | 45 |
| events | 50,532 | 67,593 |
| 위기 MDD | 29.01% | **55.47%** |
| 위기 CAGR | -32.73% | **-38.44%** |
| 정상장 CAGR | +61.2% | +92.8% |
| 정상장 MDD | 19.9% | 17.4% |
| Uplift | +0.10 | +0.29 |

**해석**: 데이터 무결성 수정 자체는 검증됐다(BTC 정상 복구, valid_symbols 증가). 그러나 `timestamp_x` 스키마 버그는 위기 replay 경로에만 국한되지 않고 정상 L1/L2 유니버스 로딩에도 걸쳐 있었다 — 이번 수정으로 registry_symbols가 93→103으로 늘며 Optuna가 전혀 다른(더 공격적인) 챔피언을 선택했다(정상장 CAGR +61.2%→+92.8%, Uplift +0.10→+0.29). 위기 MDD/CAGR 악화는 이 새 챔피언의 진짜 꼬리위험이 이제야 정직하게 드러난 결과로 판단되며, 위기 게이트는 여전히(그리고 정당하게) `stress_tested_fail`로 승격을 차단 중이다. "이 수정이 위기 생존율을 개선한다"는 메커니즘 레벨의 사전 추정은 이번 실측으로 반증됐다 — 게이트가 설계대로 작동하고 있다는 뜻으로 해석한다.

## L2 배치 레버리지 Kelly/Worst-Fold 안전장치 + Ceiling 구조 리팩토링 (`ADR_20260717_L2_DEPLOY_LEVERAGE_KELLY_WORST_FOLD_SAFETY`, `ADR_20260717_L2_LEVERAGE_CEILING_REFACTOR`)

**출발점**: 새 챔피언(registry_symbols=103)의 위기 MDD 초과폭(55.47%/21% ≈ 2.64배)을 역산하면, fit-leg(2025 정상장) 단위 레버리지 MDD(~10.2%)와 2022 위기 구간 단위 레버리지 MDD(~27.0%)의 비율과 정확히 일치 — 레버리지 `L*`가 우연히 평온했던 과거 경로 하나에만 맞춰 산출되고 있음을 실측으로 확정.

**1차 수정(opt-in 안전장치 도입)**: `calibrate_deployment_leverage`에 신규 candidate 2종 추가 — (1) `select_worst_fold_returns`로 챔피언 자신의 walk-forward fold 중 unit MDD 최대 fold를 찾아 별도 MDD 제약으로 사용(위기 윈도우 미참조, look-ahead 없음), (2) `kelly_safety_fraction=0.25`(심볼 레벨에 이미 쓰이는 `KELLY_FRACTION`과 동일 상수) 기반 fractional-Kelly 이론 상한. `Layer2AllocationConfig.l2_deploy_worst_fold_gate_enabled`/`l2_deploy_kelly_safety_fraction`으로 기본 비활성(opt-in) 노출.

**버그 발견(동일 챔피언 고정 A/B)**: 게이트를 켜고 전체 파이프라인을 재실행하니 위기 MDD가 55.47%→47.13%로 개선됐으나, Optuna champion drift(정상장 CAGR 92.8%→65.5%로 다른 챔피언이 뽑힘)로 confound돼 순수 효과 판별 불가. 동일 챔피언의 fit-leg 데이터를 직접 캡처해 격리 비교한 결과, `l_worst_fold=1.0000`(가장 타이트한 후보)임에도 최종 `L*=2.0610`(게이트 off와 완전 동일)이 산출됨을 확인 — 원인은 candidates `min()`으로 후보를 모으는 1단계와 RC-2 OOS-blend가 그 결과를 조건부로 재상향하는 2단계가 분리돼 있고, `hard_cap`/`exchange_cap`만 함수 말미에서 재검증되고 `worst_fold`/`kelly`는 재검증 지점이 없는 비일관적 구조였기 때문.

**2차 수정(구조 리팩토링)**: `calibrate_deployment_leverage`를 `_resolve_safety_ceiling`(모든 절대 상한 후보를 모아 `l_full`(mdd/cvar 포함, OOS-blend가 재추정 가능) / `l_hard`(hard_cap/exchange_cap/worst_fold/kelly만 — 절대 상한) 반환) + `_resolve_oos_adaptive_leverage`(기존 RC-2 blend 로직 유지, 최종 후보를 `min(l_blend, oos_floor_cap, l_hard)`로 클램프 — 정확한 수정 지점) + `_apply_concentration_haircut`으로 분리. 공개 시그니처/반환 타입/binding 라벨은 전혀 변경 없음. 향후 신규 안전장치는 `_resolve_safety_ceiling`의 candidates 리스트에 한 줄만 추가하면 자동 강제되는 구조로 전환.

**실측 검증(리팩토링 후, 실제 챔피언 fit-leg 데이터 재캡처)**:

| capture | mu(per-bar) | l_hard_ceiling(raw) | L*_on(게이트 활성) | 불변식 `L* ≤ max(ceiling, l_floor)` |
| :--- | ---: | ---: | ---: | :---: |
| #1 | 0.0000045 | 0.0991 | **1.0000**(kelly_theoretical) | ✅ |
| #2(=이전 세션 실제 챔피언, L*=2.0543과 일치 확인) | 0.0000148 | 0.0447 | **1.0000**(kelly_theoretical) | ✅ |

- 수정 전 `L*_off=2.0543`이 그대로 유지됐던 바로 그 챔피언 데이터에서, 수정 후에는 `L*_on=1.0000`으로 정확히 제약이 걸림 — ceiling 우회 버그가 실측 재현 케이스에서 해소됨을 확인.
- **부가 발견**: quarter-Kelly 이론값(`l_kelly_raw`)이 0.04~0.10 수준 — `l_floor=1.0` 하한이 없었다면 사실상 무포지션 수준까지 de-lever됨. 이 챔피언의 실제 per-bar mu(~1~2×10⁻⁵)가 quarter-Kelly 기준으로는 매우 작아, **Kelly 게이트를 켜면 사실상 항상 1x로 강하게 de-lever되는 극단적 안전장치**로 작동 — 이 시스템의 엣지 크기에 quarter-Kelly가 과도하게 보수적인 기준일 수 있음을 시사(다음 조치 참고).
- `/check` PASS: `risk_deployment.py` 자체 Cov 92%, spec compliance 포함 전 항목 통과.

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS (회귀 없음).
- **L2 스코어카드 정합성(Bug-A1/A2):** PASS — 실측 확인 완료, production 신뢰 가능.
- **BTC 레짐 데이터 무결성(`_resolve_timestamp_column`/`_btc_index` fail-closed):** ✅ 수정 완료 및 메커니즘 레벨 실측 검증 — 단, 정상 유니버스 확장으로 챔피언이 달라져 위기 MDD/CAGR은 개선되지 않고 악화(55.47%/-38.44%).
- **L2 배치 레버리지 ceiling 구조 리팩토링:** ✅ 수정 완료 및 실제 챔피언 데이터로 불변식 검증 — worst_fold/kelly 게이트가 이제 RC-2 OOS-blend에 의해 우회되지 않음.
- **L2 worst_fold 게이트 기본값 전환:** ✅ 적용 완료·실측 검증 — 위기 MDD/CAGR 방향성 개선(55.47%→46.53%, -38.44%→-28.04%)했으나 예산 미달로 위기 게이트는 계속 차단. kelly는 과잉보정 확인되어 opt-in 유지.
- **L2 롱/숏 방향 비대칭 진단·완화 레버:** ✅ 근본원인(롱 레그 -11.29% vs 숏 레그 +3.59%, 대칭 cap 적용 후에도 존속) 실측 확정, opt-in 레버 구현·스윕 검증 완료 — CAGR은 대폭 개선(-28.04%→-14.19%)했으나 **MDD는 사실상 미개선(46.1~46.9% 평평)**, 위기 게이트의 실제 지배적 제약(MDD 예산)은 여전히 미해결. 기본값 비활성 유지.
- **L2 위기 재현성 게이트:** FAIL-CLOSED 유지 — replay는 정상 완주했으나 MDD/CAGR 생존 조건을 여전히 위반해 `stress_tested_fail`, `verified=False`, 최종 promotion 차단. 병목은 CAGR보다 **MDD**로 좁혀짐(현재 두 레버 모두 MDD를 21% 예산까지 끌어내리지 못함) — 다음 조치는 MDD 드라이버(peak-to-trough 구간 분해) 규명.

## 다음 조치

1. 위기장(crisis-caliber fold)을 포함하는 정상 holdout 윈도우로 Uplift/CAGR을 재검증할 것 — 여전히 미해결(`⚠️ NO-CRISIS-WINDOW` 경고 지속).
2. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
3. ~~crisis detail에 MDD/CAGR/CVaR 원시값을 함께 출력~~ — `[CRISIS-WINDOW-DETAIL]` 로그로 해소(2026-07-17).
4. `docs/results/next.md` §1(`run_config.timeframe` CLI 기본값 "4h"의 tf-probe 기반 근거화)은 별도 `/spec` 대기 중.
5. ~~새 챔피언이 왜 위기에 더 취약한지 원인 진단~~ — ceiling 우회 버그로 확정, 리팩토링으로 해소(2026-07-17).
6. ~~`l2_deploy_worst_fold_gate_enabled`을 프로덕션 기본값으로 전환할지 결정~~ — worst_fold 게이트만 기본 활성화, kelly는 opt-in 유지(아래 §L2 worst_fold 기본값 전환 참고). 활성화했으나 위기 게이트는 여전히 `stress_tested_fail` — 아래 §L2 롱/숏 방향 비대칭 진단으로 후속 조치 진행 중.

## L2 worst_fold 안전장치 프로덕션 기본값 전환 (`ADR_20260717_L2_CRISIS_LEVERAGE_SAFETY_DEFAULT`)

**변경**: `Layer2AllocationConfig.l2_deploy_worst_fold_gate_enabled` 기본값을 `False`→`True`로 전환. 동시에 `from_mapping`이 파라미터 키 부재 시 SSOT(`_dc.<field>`) 대신 하드코딩된 `False`로 침묵 복귀하던 버그를 수정(수정 없이 기본값만 바꿨다면 Optuna `best_l2_params`에 이 키가 보통 없어 여전히 비활성으로 침묵 복귀했을 것). `kelly_safety_fraction`은 이 시스템의 극소 mu(~1~2×10⁻⁵)에서 quarter-Kelly가 거의 항상 `l_floor=1.0`으로 수렴하는 과잉보정 특성이 확인되어 opt-in 유지.

**실측 검증(전체 파이프라인 재실행, 2026-07-17, `--seed 42`)**:

| 지표 | Before(게이트 off) | After(게이트 on, 기본값) | 변화 |
| :--- | ---: | ---: | ---: |
| 정상장 L* | ~2.06 | 1.72 | -16.5% |
| 정상장 CAGR | +92.8% | +53.2% | champion drift(아래 참고) |
| 정상장 MDD | 17.4% | 14.3% | 개선 |
| 위기 MDD | 55.47% | 46.53% | -8.9pp 개선(상대 -16%) |
| 위기 CAGR | -38.44% | -28.04% | +10.4pp 개선 |
| 위기 게이트 판정 | stress_tested_fail | stress_tested_fail(유지) | 방향은 맞으나 예산(MDD≤21%, CAGR≥-5%) 미달 |

**champion drift 원인(코드 조사로 확정)**: `build_layer2_deployable_score`의 `score = cagr + 0.10·sortino + 0.05·calmar - …`에서 `cagr` 항은 후보별 **배치된(leveraged) CAGR**이며 leverage-scale에 불변이 아니다. worst_fold 게이트가 후보마다 서로 다른 L*를 제약하면서, Optuna가 "raw Sharpe 최고" 후보 대신 "worst-fold 제약 하 배치 CAGR 최고" 후보를 선택하도록 목적함수가 사실상 바뀌었다(Sharpe 2.026→1.700). 프로젝트 목표(복리 성장 극대화)와 부합하는 의도된 설계로 판단, 결함 아님 — 단, "후보 품질 랭킹"과 "레버리지 사이징" 분리 여부는 별도 검토 과제로 남김.

**결론**: 게이트는 설계대로 작동(위기 MDD/CAGR 방향성 개선)하지만 단독으로는 위기 예산을 충족 못함 — worst_fold만으로는 부족함을 실측 확인. `/check` PASS(Cov 97%, spec compliance 포함). SSOT: `docs/specs/l2-crisis-leverage-safety-defaults.md`.

## L2 롱/숏 방향 비대칭 진단 및 opt-in 완화 레버 (`ADR_TBD_L2_CRISIS_ASYMMETRIC_LONG_SHORT_CAP`)

**진단 배경**: worst_fold 게이트 적용 후에도 위기 게이트가 실패해 근본원인을 추가 조사. `scratch/diag_crisis_longshort.py`로 실제 champion registry의 LUNA/FTX replay를 롱/숏 레그로 분리 측정(기존 `apply_regime_risk_cap` 방향-무관 대칭 축소가 이미 적용된 이후 값).

| 지표 | 롱 레그 | 숏 레그 |
| :--- | ---: | ---: |
| 활성 bar 수 | 683 | 786 |
| 실현 가격 P&L(비용 차감 전) | **-11.29%** | **+3.59%** |

- **숏 레그는 이미 위기장에서 순이익** — bar 빈도도 롱(683)보다 많음(786). "숏을 못한다"가 문제가 아니라 **롱의 손실 크기가 숏의 이익 크기의 3배 이상**인 것이 문제.
- 롱 상위 손실 종목(DOGE/ZIL/API3/VET/LPT/SNX/UNI/ANKR)이 전부 알트코인 — LUNA/FTX 전염발 상관붕괴 국면에서 추세추종 롱("눌림목 매수")이 반복 휩쏘당한 패턴과 일치. 기존 대칭적 `apply_regime_risk_cap`(bear=0.35x/crisis=0.25x)은 이 방향 비대칭을 교정하지 못함(위 수치가 그 적용 *이후* 값).

**opt-in 완화 레버 구현**: bear/crisis 레짐에서 롱 레그에만 추가 축소 배수를 적용하는 `apply_asymmetric_long_short_regime_cap`(순수 함수, `l2_meta.py`) + `Layer2AllocationConfig` 3개 opt-in 필드(`l2_regime_long_short_asymmetry_enabled`, `l2_regime_{bear,crisis}_long_extra_mult`, 기본값 전부 no-op) 추가. `/check` PASS(spec compliance 포함). SSOT: `docs/specs/l2-crisis-asymmetric-long-short-regime-cap.md`.

**스윕 실측(동일 champion 고정, `scratch/sweep_crisis_asymmetric_cap.py`, LUNA/FTX)**:

| long_extra_mult | MDD | CAGR |
| :--- | ---: | ---: |
| 1.00(off, baseline) | 46.53% | -28.04% |
| 0.70 | 46.11% | -23.57% |
| 0.50 | 46.19% | -20.72% |
| 0.30 | 46.47% | -18.00% |
| 0.15 | 46.68% | -16.05% |
| 0.00(롱 완전 차단) | 46.90% | **-14.19%** |

**혼재된 결과**: CAGR은 크게 개선(-28.04%→-14.19%, 손실 절반 이하로 축소)해 롱/숏 비대칭 가설을 CAGR 측면에서 강하게 확인. **그러나 MDD는 사실상 평평(46.1~46.9%)하고 롱을 완전 차단한 mult=0.00에서 오히려 소폭 악화** — 위기 게이트의 실제 지배적 제약(MDD≤21% 예산, 현재 2.2배 초과)을 이 레버가 전혀 못 건드림. 전 구간 `stress_tested_fail` 유지. MDD를 결정하는 최대낙폭 구간이 이 레그별 평균 P&L 비대칭과 다른 곳(특정 급락/반등 구간의 회전비용 또는 숏 스퀴즈 등)에서 발생하는 것으로 추정되나 현재 데이터로는 특정 못함 — 추가 진단(peak-to-trough 구간별 롱/숏/비용 분해) 필요.

**주의(전례 참고)**: 2026-07-02 reversal kill-switch 실측 반증(방향-무관 대칭적 de-gross가 실제 위기 홀드아웃에서 baseline보다 악화, `risk_off_realized_price` 전 variant 양수)과 달리 이번 레버는 방향-분리 설계라 같은 함정은 아니지만, 같은 실수(단일 윈도우 백테스트만으로 승격)를 반복하지 않기 위해 두 개의 독립 위기 윈도우(LUNA/FTX + 2025-12-31~2026-06-30 BTC 위기 홀드아웃) 모두에서 개선 확인 전까지 기본값 비활성 유지(spec `[LIMIT-01]`).
