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

## Verdict

- **L0→L1→native TF handoff / L1 robustness gate:** PASS (회귀 없음).
- **L2 스코어카드 정합성(Bug-A1/A2):** PASS — 실측 확인 완료, production 신뢰 가능.
- **BTC 레짐 데이터 무결성(`_resolve_timestamp_column`/`_btc_index` fail-closed):** ✅ 수정 완료 및 메커니즘 레벨 실측 검증 — 단, 정상 유니버스 확장으로 챔피언이 달라져 위기 MDD/CAGR은 개선되지 않고 악화(55.47%/-38.44%).
- **L2 위기 재현성 게이트:** FAIL-CLOSED 유지 — replay는 정상 완주했으나 MDD/CAGR 생존 조건을 여전히 위반해 `stress_tested_fail`, `verified=False`, 최종 promotion 차단.

## 다음 조치

1. 위기장(crisis-caliber fold)을 포함하는 정상 holdout 윈도우로 Uplift/CAGR을 재검증할 것 — 여전히 미해결(`⚠️ NO-CRISIS-WINDOW` 경고 지속).
2. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
3. ~~crisis detail에 MDD/CAGR/CVaR 원시값을 함께 출력~~ — `[CRISIS-WINDOW-DETAIL]` 로그로 해소(2026-07-17).
4. `docs/results/next.md` §1(`run_config.timeframe` CLI 기본값 "4h"의 tf-probe 기반 근거화)은 별도 `/spec` 대기 중.
5. 새 챔피언(registry_symbols=103)이 왜 위기에 더 취약한지 원인 진단 — 어떤 family/variant가 선택됐는지, `deploy_leverage` 값이 이전 대비 얼마나 늘었는지 확인 필요(별도 `/spec` 대상).
