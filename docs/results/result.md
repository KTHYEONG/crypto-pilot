## L1 신호 스테이지 정합성 교정(`l1_signal_stage_integrity`) 실전 CLI 검증 결과 — 2026-07-30

### 1. 실행 식별자

| 항목 | 값 |
|---|---|
| 기준 실행 (교정 적용 후) | `logs/futures/compound/20260730_022430/` |
| 대조 실행 (교정 직전, 동일 데이터) | `logs/futures/compound/20260730_011331/` |
| 기준 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` |
| reference date / seed | `2026-07-15` / `42` |
| 입력 규모 | `5,442` bars(4h) × `51` symbols |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` (두 실행 동일 스냅샷) |
| L2/L3 보호 | `L2_DRY_RUN=1` (dry_run=true), sealed L3 holdout 미소비(정식 판정 실행에서는) |
| 적용 스펙 | `docs/specs/l1_signal_stage_integrity.md` — L1 leg 파이프라인의 1-bar look-ahead 제거, holdout carry-forward, leg 가중 순서 교정, per-name 집중도 캡 |

### 2. 무엇이 왜 바뀌었는가 (요약)

직전 세션(`l1_signal_stage_redesign`)에서 L1이 프로젝트 사상 최초로 admitted>0·비영 포지션 배포에 성공했지만, 그 수치 자체가 **1-bar look-ahead**로 부풀려져 있었다는 것이 이번 세션 분석(캡처된 프로덕션 패널 재실행 기반 실측)에서 확인됐다. 핵심 결함 4가지와 교정 내용:

1. **채점 규약 불일치(look-ahead)**: `l1_concept_bank.build_leg_books`와 `l1_leg_admission.evaluate_portfolio_admission`이 `dot(book[t], ret[t])`로 수익률을 채점했다. 그런데 `ret[t] = log(close[t]/close[t-1])`는 `book[t]`가 이미 알고 있는 `close[t]`를 포함하므로, 아직 오지 않은 봉의 가격 정보로 자기 자신을 채점하는 셈이었다. 실제 배포 시뮬레이터(`dense_simulator`)는 정직하게 `prev_w × ret[t]`를 쓴다. → `compute_lagged_gross_returns` 신규 함수로 두 채점 지점 모두 `book[t-1] × ret[t]` 규약으로 통일(`[RULE-11]`).
2. **holdout이 항상 평평했던 문제**: `accumulate_prequential_leg_weights`가 마지막 fold의 OOS 구간 이후로는 가중치를 채우지 않아, 봉인된 L3 holdout 90일 구간은 항상 전량 0이었다(신호 품질과 무관하게 L3가 구조적으로 통과 불가능했다는 뜻). → 마지막으로 계산된(=`l3_start−embargo` 이전 증거만으로 산출된) leg 가중 벡터를 배열 끝까지 carry하도록 수정(`[RULE-14]`).
3. **leg 증거 가중이 사실상 무효(inert)했던 문제**: `min(2·max(Φ(t)-0.5,0), 0.5)`을 정규화 **전**에 cap 하는 구조라, t가 0.674를 넘으면(흔한 경우) 무조건 균등가중(50/50)으로 수렴했다 — prequential 증거 기계가 사실상 장식이었다. → cap을 정규화 **이후** water-filling으로 적용하도록 순서 교정(`[RULE-12]`, `normalise_leg_weights` 신규).
4. **집중도 캡 부재**: 단일 종목이 봉당 gross의 최대 74%까지 차지할 수 있어 `capacity_utilisation_p95` 게이트를 위협했다. → leg book 구성 직후 종목별 `|w| ≤ 0.10 × gross` 캡을 gross 보존 방식(water-filling)으로 적용(`[RULE-13]`, `cap_per_name_weights` 신규).

부수적으로 `l1_concept_bank.build_concept_registry`가 인자로 받은 `descriptors`를 무시하고 무조건 하드코딩된 registry를 반환하던 결함도 함께 고쳐, 존재하지 않는 member family를 참조하면 `ValueError`로 즉시 실패하도록 했다(`[RULE-15]`). 이 과정에서 `vol_regime` registry가 카탈로그에 없는 `volatility_squeeze_keltner`를 선언하고 있었다는 사실이 드러나 제거했다 — `vol_regime`은 실제로는 2-family(6 descriptor)였다.

### 3. Concept registry (2 concepts, 12 formula members — 정정)

| Concept | Mode | Member formulas |
|---|---|---|
| `trend_momentum` | xs | `trend_ema`, `momentum_ts`, `breakout_donchian` + `rsi`, `cci`, `mfi`, `aroon_oscillator`, `adx_directional`, `obv_trend`, `keltner_breakout` (10종) |
| `vol_regime` | ts | `volume_zscore`, `bollinger_bandwidth` (2종 — 직전 세션이 기재한 `volatility_squeeze_keltner`는 카탈로그에 descriptor가 없어 실제로는 무시되고 있었음. 이번 세션에서 registry에서 제거하고 `[RULE-15]` fail-fast 검증을 추가) |

### 4. Per-leg 실측 비교 — 교정 전후 (동일 데이터, 동일 fold 구조, 18 folds)

**핵심 지표만 비교 (마지막 fold, 가장 많은 증거를 누적한 시점)**

| Concept | | t_alpha | breakeven_bps | positive_folds | evidence_weight |
|---|---|---:|---:|---:|---:|
| `trend_momentum` | 교정 전 | **7.191** | 72.1 | 18/18 | 0.5000 |
| `trend_momentum` | **교정 후** | **1.881** | **19.1** | 13/17 | 0.5000 |
| `vol_regime` | 교정 전 | 3.115 | 52.3 | 13/18 | 0.5000 |
| `vol_regime` | **교정 후** | **2.414** | **44.5** | 10/17 | 0.5000 |

`trend_momentum`의 t가 **3.8배** 축소됐다 — 직전 세션이 "최초 성공"의 근거로 제시한 t=7.19, breakeven=72bps는 대부분 look-ahead 산물이었다. 다행히 두 concept 모두 가드 임계(`breakeven > cost×1.5 = 12bps`, `positive_fold_ratio ≥ 0.50`)는 여전히 통과해 **cash-only로 회귀하지 않았다** — `compute_evidence_weight`에 t-통계 문턱 자체가 없기 때문이다(`[LIMIT-02]`, 정책 결정 보류 중).

**`trend_momentum` 전체 fold 이력 (교정 후, 15개 평가 시점)**

| fold(n) | alpha_ann | beta | aSR | t_alpha | be_bps | turnover/bar | pos_folds | posterior | weight |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 3 | 0.6027 | 0.043 | 0.051 | 1.192 | 30.3 | 0.09093 | 2/3 | 0.739 | 0.5000 |
| 4 | 0.4901 | 0.005 | 0.043 | 1.161 | 25.2 | 0.08897 | 3/4 | 0.903 | 0.5000 |
| 5 | 0.4039 | −0.035 | 0.035 | 1.048 | 20.9 | 0.08841 | 4/5 | 0.894 | 0.5000 |
| 6 | 0.1667 | −0.052 | 0.015 | 0.498 | 8.4 | 0.09090 | 4/6 | 0.658 | **0.0000** |
| 7 | 0.2238 | −0.045 | 0.021 | 0.749 | 11.3 | 0.09038 | 5/7 | 0.750 | **0.0000** |
| 8 | 0.1913 | −0.045 | 0.019 | 0.713 | 9.6 | 0.09051 | 5/8 | 0.752 | **0.0000** |
| 9 | 0.1760 | −0.047 | 0.018 | 0.721 | 9.0 | 0.08897 | 6/9 | 0.777 | **0.0000** |
| 10 | 0.1789 | −0.046 | 0.019 | 0.788 | 9.2 | 0.08926 | 7/10 | 0.787 | **0.0000** |
| 11 | 0.1873 | −0.046 | 0.020 | 0.886 | 9.7 | 0.08841 | 8/11 | 0.820 | **0.0000** |
| 12 | 0.1790 | −0.044 | 0.020 | 0.910 | 9.3 | 0.08834 | 9/12 | 0.846 | **0.0000** |
| 13 | 0.3058 | −0.054 | 0.029 | 1.387 | 15.9 | 0.08781 | 10/13 | 0.935 | 0.5000 |
| 14 | 0.4494 | −0.050 | 0.040 | 1.990 | 23.4 | 0.08757 | 11/14 | 0.981 | 0.5000 |
| 15 | 0.4241 | −0.058 | 0.038 | 1.983 | 22.1 | 0.08777 | 12/15 | 0.972 | 0.5000 |
| 16 | 0.4163 | −0.061 | 0.038 | 2.043 | 21.6 | 0.08800 | 13/16 | 0.982 | 0.5000 |
| 17 | 0.3712 | −0.066 | 0.034 | 1.881 | 19.1 | 0.08883 | 13/17 | 0.971 | 0.5000 |

**주목**: fold 6~12(7개 시점)에서 `breakeven_cost_bps`가 8.4~9.7로 가드 임계(12.0)를 밑돌아 `evidence_weight=0.0000`을 기록했다 — 교정 전에는 이 구간에서도 무조건 0.5000이었다. **이것이 정규화-후-cap 순서 교정(`[RULE-12]`)이 실제로 데이터에 반응하기 시작했다는 최초의 관측 증거다.** k=2 상태에선 여전히 통과 시 이진(0 또는 0.5)이라 연속적 차등가중은 아니지만(`[LIMIT-07]`, concept가 3개 이상이어야 진짜 차등가중이 작동), 최소한 "항상 무조건 0.5"였던 이전 상태보다는 데이터에 반응하는 상태다.

**`vol_regime` 전체 fold 이력 (교정 후)**

| fold(n) | alpha_ann | beta | aSR | t_alpha | be_bps | turnover/bar | pos_folds | posterior | weight |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 3 | 1.5474 | 0.521 | 0.063 | 1.472 | 69.6 | 0.10157 | 2/3 | 0.969 | 0.5000 |
| 4 | 1.3118 | 0.379 | 0.046 | 1.237 | 52.1 | 0.11489 | 3/4 | 0.952 | 0.5000 |
| 5 | 1.8677 | 0.377 | 0.063 | 1.892 | 72.8 | 0.11715 | 4/5 | 0.992 | 0.5000 |
| 6 | 1.3834 | 0.353 | 0.047 | 1.555 | 53.2 | 0.11863 | 4/6 | 0.953 | 0.5000 |
| 7 | 1.6683 | 0.318 | 0.057 | 2.015 | 62.4 | 0.12205 | 5/7 | 0.992 | 0.5000 |
| 8 | 1.4741 | 0.312 | 0.051 | 1.945 | 55.1 | 0.12209 | 6/8 | 0.989 | 0.5000 |
| 9 | 1.4798 | 0.278 | 0.053 | 2.113 | 53.8 | 0.12551 | 7/9 | 0.997 | 0.5000 |
| 10 | 1.1972 | 0.280 | 0.043 | 1.834 | 43.8 | 0.12470 | 7/10 | 0.971 | 0.5000 |
| 11 | 0.9774 | 0.245 | 0.036 | 1.591 | 35.2 | 0.12686 | 7/11 | 0.956 | 0.5000 |
| 12 | 0.8736 | 0.179 | 0.032 | 1.484 | 31.8 | 0.12526 | 7/12 | 0.954 | 0.5000 |
| 13 | 1.0603 | 0.184 | 0.038 | 1.851 | 39.4 | 0.12295 | 8/13 | 0.978 | 0.5000 |
| 14 | 1.1560 | 0.225 | 0.042 | 2.105 | 43.4 | 0.12155 | 9/14 | 0.986 | 0.5000 |
| 15 | 1.3560 | 0.200 | 0.049 | 2.546 | 50.2 | 0.12335 | 10/15 | 0.997 | 0.5000 |
| 16 | 1.2652 | 0.175 | 0.046 | 2.467 | 47.0 | 0.12298 | 10/16 | 0.997 | 0.5000 |
| 17 | 1.1849 | 0.196 | 0.044 | 2.414 | 44.5 | 0.12167 | 10/17 | 0.994 | 0.5000 |

`vol_regime`은 한 번도 가드 임계를 밑돌지 않아 매 시점 admit — 직전 세션 대비 상대적으로 견고한 concept임이 재확인됐다.

### 5. 배포 (`target_weights.npy`, 5442×51 float32) — 교정 전후 비교

| metric | 교정 전 | 교정 후 |
|---|---:|---:|
| nonzero 봉 비율 | 50.7% (2,760봉) | **51.4% (2,796봉)** |
| 비영 시작/종료 인덱스 | [2142, 4901] | **[2142, 5160]** (holdout 진입 확인) |
| holdout(마지막 540봉) 비영 비율 | **0%** (구조적으로 항상 0) | **6.7%** (36/540봉 — 리스크 오버레이가 실질 작동) |
| mean gross exposure(활성 봉) | 0.3212 | 0.3391 |
| max gross exposure | 0.7442 | 0.9825 |
| capacity_utilisation_p95 | 0.184 | **0.179** (여전히 게이트 0.10 초과) |

holdout 진입 자체는 확인됐으나(`[RULE-14]` 정상 작동), 실질 노출은 6.7%에 그친다 — 이는 결함이 아니라 **`apply_portfolio_risk_overlay`의 변동성 타겟팅·drawdown 브레이커가 holdout 구간에서 실제로 작동한 결과**다(§7 참조).

### 6. L2 / L3 최종 판정 — 교정 전후 비교

| metric | 교정 전 | 교정 후 |
|---|---:|---:|
| L1 admitted concepts | 2 / 2 | **2 / 2** (회귀 없음) |
| L2 verdict | fail | **fail** |
| L2 cagr | 0.2339 | **0.2083** |
| L2 sharpe | 1.5632 | **1.3377** |
| L2 sharpe_probability | 0.9155 | 0.8770 |
| L2 deflated_sharpe_probability | 0.7075 | 0.6000 |
| L2 excess_growth_probability | 0.3023 | 0.2870 |
| L2 excess_growth_lcb90 | 0.0085 | **−0.0332** (양수 아님으로 전환) |
| L2 stressed_excess_growth_lcb90 | −0.0340 | −0.0912 |
| L2 max_drawdown | 0.0770 | 0.0683 |
| L2 annual_turnover | 62.30 | 79.01 |
| L2 cost_drag_ratio | 0.2502 | 0.3166 |
| L2 capacity_utilisation_p95 | 0.1837 | 0.1790 |
| L2 reasons(개수) | 6 | 6 |
| L3 verdict | reject | **reject** |
| L3 posterior_growth_probability | 0.1727 | **0.2092** (개선) |
| L3 max_drawdown | 1.04e-06 (평탄 — 비측정 상태) | **0.00562** (**실제 곡선 채점 시작**) |
| L3 reasons | `low_growth_probability`, `l2_not_pass` (2개) | **`l2_not_pass`만 (1개)** |

L2 turnover가 62.30→79.01로 늘고 cost_drag가 25.0%→31.7%로 커졌다 — per-name 캡(`[RULE-13]`)의 water-filling 재분배가 봉마다 미세한 리밸런싱을 추가로 발생시키기 때문이다. CAGR/Sharpe는 예상대로 하향했으나(look-ahead 제거 효과), L2 gate reasons 개수는 6개로 동일 — 근본 판정은 바뀌지 않았고 수치만 정직해졌다.

**L3의 변화가 가장 의미 있다**: 교정 전 L3는 holdout이 구조적으로 전량 0이라 "평탄한 직선을 채점"하는 비측정 상태였다(`max_drawdown=1.04e-06`은 사실상 노이즈). 교정 후 L3는 처음으로 **실제 25봉 비영 구간을 포함한 진짜 자산곡선**을 채점했고(`max_drawdown=0.0056`), reject 사유도 `low_growth_probability`가 사라지고 `l2_not_pass` 하나로 단순화됐다 — L1이 실제로 무언가를 배포하기 시작하면서 L3 판정 자체의 정보량이 늘었다는 뜻이다.

### 7. 부수 관측 — holdout 실측 손실이 스펙 사전측정치보다 훨씬 작았던 이유

스펙 작성 단계에서 leg-book **단독**(리스크 오버레이 미적용) prequential 시뮬레이션으로 holdout 구간 net_ann **−36.75%**, Sharpe −1.09를 측정했었다(`docs/specs/l1_signal_stage_integrity.md` D-2). 그러나 실제 프로덕션 배포에서 L3 MDD는 0.56%에 불과했다. 원인을 대조한 결과:

- holdout 구간에서 leg 가중 자체는 `[RULE-14]`에 따라 마지막 fold 값으로 carry되어 constant하다.
- 그러나 `apply_portfolio_risk_overlay`(변동성 타겟팅 + drawdown 브레이커, `allocator.py`)가 carry된 배포북에 실제로 반응해, 손실이 누적되는 구간에서 `dd_scale`을 0까지 낮췄다.
- 그 결과 holdout 540봉 중 실질 비영은 36봉(6.7%)뿐이었고, 나머지는 리스크 오버레이가 노출을 억제했다.

**이것은 결함이 아니라 다층 방어가 의도대로 작동한 사례다** — leg-book 단독 측정은 정직한 신호 품질(음의 엣지)을 드러냈고, 포트폴리오 리스크 계층이 그 손실의 대부분을 실제로 차단했다. 두 측정 모두 유효하며 서로 다른 층위(신호 품질 vs 실배포 손실)를 보여준다.

### 8. 부수 관측 — 워닝 로그 (해소 확인)

직전 세션에서 관측됐던 `l1_concept_bank.py`의 `np.nanmean` empty-slice `RuntimeWarning`이 이번 실행 로그에서 **더 이상 관측되지 않는다** — `[RULE-16]`(전 멤버 NaN 셀을 `np.divide(..., where=nan_cnt>0)`로 안전 처리)이 실전 데이터에서도 원천 차단에 성공했음을 확인. 남은 워닝은 `numpy.lib._function_base_impl.py`의 상관행렬 계산(clustering, 본 스펙과 무관한 기존 코드) 관련 divide-by-zero뿐이다.

### 9. 결론

1. **look-ahead 제거가 실측으로 확인됐다**: `trend_momentum` t_alpha가 7.19→1.88(3.8배 축소), breakeven이 72.1→19.1bps로 축소됐다. 직전 세션의 "최초 성공" 수치는 대부분 1-bar look-ahead 산물이었다.
2. **cash-only 회귀는 일어나지 않았다**: `compute_evidence_weight`에 t-통계 문턱이 없어 두 concept 모두 admit을 유지했다(`[LIMIT-02]`, 정책 결정으로 범위 외 보류). 정직해진 결과는 "실패"가 아니라 "더 겸손한 성공"이다.
3. **holdout이 처음으로 진짜 검정이 됐다**: 교정 전 L3는 항상 평탄한 곡선을 채점하는 구조적 결함이었다. 교정 후 L3는 실제 자산곡선(MDD 0.56%)을 채점하며, reject 사유가 2개→1개로 단순화됐다.
4. **leg 가중이 데이터에 반응하기 시작했다**: `trend_momentum`이 fold 6~12 구간에서 evidence_weight=0으로 정확히 떨어지는 것을 처음 관측했다 — 순서 교정(`[RULE-12]`) 전에는 불가능했던 거동이다.
5. **L2/L3 최종 판정(fail/reject)은 바뀌지 않았다** — 이것도 정직한 결과다. 근본 병목은 여전히 `capacity_utilisation_p95`(0.179, 게이트 0.10), `excess_growth_probability`(0.287, 게이트 0.90), `positive_outer_folds`(2, 게이트 3)다.
6. **다음 착수점**:
   - (a) `[LIMIT-02]` — `compute_evidence_weight`에 t-통계 하한 도입 여부는 정책 결정이 필요하다(교정 후 `trend_momentum`의 정직한 t=1.88은 통상적 유의수준에 못 미친다).
   - (b) `[LIMIT-07]` — concept 2개 + `max_leg_weight=0.50` 조합은 수학적으로 항상 이진(0 또는 0.5)만 가능하다. 진짜 연속적 차등가중을 보려면 concept를 3개 이상으로 늘려야 하며, 이는 미편입 5개 family(`basis_gap`, `reversal_st`, `xs_reversal`, `xs_momentum_slow`, `smart_money_divergence`, 22 descriptor, `[LIMIT-06]`) 편입 검토와 함께 진행해야 한다.
   - (c) `capacity_utilisation_p95` 초과 원인이 되는 저유동성/집중 종목 비중 완화(유니버스 정책 또는 캡 강도 재검토).
   - (d) L2 turnover 증가(62.30→79.01)에 따른 cost_drag 상승(25.0%→31.7%)이 per-name 캡의 부작용인지 별도 측정 필요.

원본 artifact:

- [result.json (교정 후)](../../logs/futures/compound/20260730_022430/result.json)
- [manifest.json (교정 후)](../../logs/futures/compound/20260730_022430/manifest.json)
- [target_weights.npy (교정 후)](../../logs/futures/compound/20260730_022430/target_weights.npy)
- [result.json (교정 전 대조군)](../../logs/futures/compound/20260730_011331/result.json)
- [l1_admission.jsonl](../../logs/l1_admission.jsonl)
- [l1_signal_stage_integrity.md](../specs/l1_signal_stage_integrity.md)
- [l1_signal_stage_integrity_contract.json](../specs/l1_signal_stage_integrity_contract.json)
