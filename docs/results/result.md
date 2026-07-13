# L0/L1 Discovery Snapshot

- **최신 측정일**: `2026-07-13` (run `4h_1783868413`, `--phase l1 --timeframe 4h --date 2026-07-12 --trials 1 --seed 42`, `LOG_LEVEL=DEBUG`)
- **재현성**: 이전 baseline(`4h_1783753822`, 2026-07-11)과 `n_evidence=120`, `gate_passed=78`, `selected_for_l1=72` 완전 동일. seed-42 파이프라인은 하루 이상 간격을 두고도 결정론적으로 재현됨.
- 과거의 방대한 최적화 반복 로그(2026-07-11~12, 20건 이상의 세션별 ADR)는 `docs/decisions/decisions.md`/`decisions_archive.md`에 보존되어 있음. 이 문서는 **현재 상태와 다음 액션**만 담는다.

## 1. L0 게이트 현황 (TF별)

| TF | 평가 Family 수 | 통과 Family 수 | Evidence Rows | Gate Passed | Selected for L1 | L1 `n_ready` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 4 | 2 | 8 | 5 | 5 | 0 (L1-nested 블로킹) |
| 2h | 4 | 4 | 10 | 10 | 10 | 19 |
| 4h | 28 | 12 | 46 | 24 | 18 | 0 (L1-nested 블로킹) |
| 6h | 11 | 6 | 19 | 13 | 13 | 0 (L1-nested 블로킹) |
| 8h | 11 | 6 | 19 | 13 | 13 | 53 |
| 12h | 10 | 7 | 18 | 13 | 13 | 98 |
| **합계** | - | - | **120** | **78** | **72** | - |

- L0 자체는 정상 작동 — 120개 후보 중 72개(-40%)로 압축해 L1에 넘김.
- **L1은 2h/8h/12h만 실제로 배포 가능**(`n_ready` > 0). 4h/6h/1h는 L0 통과 후에도 L1-nested 단계의 별도 버그(`empty_opportunities`)로 막혀 있음 — L0 품질과 무관한 하류 이슈.

## 2. 핵심 발견: "다양성"이 숫자만 있고 내용이 없다

**(A) Archetype 편중 — 4h TF, 28개 family 실측 분해**

| Archetype | 평가 Family 수 | 통과 | 통과율 |
| --- | ---: | ---: | ---: |
| `trend` (이평선/채널/눌림목/MACD/MTF조합 등) | 15 | 12 | **80%** |
| `flow` (미결제약정/펀딩/롱숏비율) | 8 | 1 | 12.5% |
| `cross_sectional` (횡단면 모멘텀/스큐) | 4 | 0 | **0%** |
| `유동성/거래량 미시구조` | 4 | 0 | **0%** |
| `mean-reversion` (평균회귀) | 1 | 0 | **0%** |
| 기타 단일지표(Ichimoku/Supertrend) | 2 | 0 | **0%** |

- 4h에서 게이트 통과 24건 중 **23건이 `trend`**. 6h/8h/12h/1h/2h는 **전 TF 100% `trend`**.
- 탈락 사유(`excess_cost_drag`, `non_positive_lcb`, `weak_tstat`)는 실제 경제성 부재이지 게이트 버그가 아님 — 과거 economic replay 전체 사이클로 이미 기각됐던 결론(비추세 archetype은 이 유니버스에서 durable edge 없음)의 독립적 재확인.

**(B) Cross-TF 중복 — 72개 중 진짜 독립적인 건 38개(53%)뿐**

같은 추세 테제(예: `btc_regime_pullback`)가 여러 TF에서 재측정되며 "72개 후보"처럼 보이지만, 실제 독립 전략 수는 38개.

**(C) 유일하게 잘 통과하는 패턴 = 상위TF 필터 × 하위TF 트리거 (MTF 융합)**

| Family | 방식 | net_lcb_bps |
| --- | --- | ---: |
| `mtf_trend_pullback` | 1D 이평선 기울기 필터 × 4h RSI(14) 진입 | 44.4~52.7 |
| `mtf_breakout_retest` | 1D 채널 돌파 × 4h 리테스트 진입 | 23.4~39.4 |
| `macd_4h` | MACD 히스토그램 크로스 (단독) | 39.1 |
| `vol_term_structure_gate` | 1D/4h 변동성 비율 × 채널 돌파 | 42.3 (통과했으나 config에서 -0.5 prior로 격하됨 — 모순) |

이 패턴은 외부 검증과도 일치: QuantPedia BTC D1+H1 MACD 전략 — 단독 Sharpe 0.33 → 상위TF 추세필터 추가 시 0.80 → 트레일링스탑까지 추가 시 1.07.

## 3. L0 목적 적합성 점수 — 56/100

| 항목 | 배점 | 점수 | 이유 |
| --- | ---: | ---: | --- |
| 연산 효율성 | 30 | 20 | 120→72 압축은 잘 작동하나, 전체 wall-clock의 **36%(158초)가 원인 불명 미계측 구간**이고, L1이 막혀있는 TF(4h/6h/1h)에도 여전히 풀 디스커버리를 돌려 컴퓨트 낭비 |
| 다양성 제공 | 40 | **18** (최약점) | 통과 후보의 96~100%가 `trend` 단일 archetype. 4h를 제외한 모든 TF에서 중복제거(BH-FDR)가 전혀 작동 안 함 |
| 자산증식 실현 | 20 | 10 | L1이 실제로 뚫린 TF는 6개 중 3개(2h/8h/12h)뿐 |
| 견고성(anti-overfitting) | 10 | 8 | 게이트 설계 자체는 견고하나, 문서(`layer0.md`)와 코드 설정값 drift, family prior가 최신 증거와 불일치 |

## 4. 결론: "새 지표 추가"는 답이 아니다

사용자가 제안한 피보나치·매물대 등은 근거 부족:
- **피보나치**: 웹 리서치 결과 학술적으로 반증됨("동전 던지기보다 낫지 않음") → 신규 archetype으로 추가 금지
- **매물대(Volume Profile)**: 단독으로는 약하고, 구조적으로 동일한 유동성-미시구조 계열이 이미 0% 통과율 실측 → 독립 전략이 아니라 기존 추세 전략의 보조 필터로만 검토 가치
- **RSI/MACD**: 이미 구현되어 있고 이미 통과 중 — "지표가 없는 게" 문제가 아니라 **"상위TF×하위TF 조합을 만드는 시스템이 3개만 하드코딩되어 있는 것"**이 진짜 병목
- **횡단면/평균회귀류**: 이번 실측과 과거 economic replay 양쪽에서 0% — 이 방향은 종결, 재시도 불필요

## 4-1. 보조지표 확장 검토 (2026-07-13, 사용자 요청)

피보나치/매물대 외에 스토캐스틱·스토캐스틱RSI·일목균형표·DEMA·HMA를 냉정하게 재검토 (`docs/specs/l0-mtf-recipe-factory.md` "Indicator Menu Review" 참조). 기준: "기존 신호와 다른 정보를 주는가, 아니면 같은 모멘텀을 다른 공식으로 재탕하는가."

| 지표 | 판정 | 근거 |
| --- | --- | --- |
| HMA | ✅ 채택 (상위TF 필터) | EMA 대비 지연/평활 트레이드오프가 실질적으로 다름 |
| DEMA | ❌ 기각 | HMA와 거의 동일 역할(지연 감소 이평선) — 웹 검증상 HMA가 더 우수, 중복 추가는 다중검정 비용만 늘림 |
| 일목균형표(구름대만) | ✅ 채택 (상위TF 필터) | 기존 독립형 `ichimoku_trend`는 이번 실측에서 `excess_turnover`(방향성 부재 아님)로 탈락 — 원인이 "4h 단독 풀시스템이 너무 자주 발동"이라는 뜻이라, 원래 저빈도인 상위TF 레짐필터 역할로 쓰면 실패 원인을 정확히 피해감 |
| 스토캐스틱(%K/%D 크로스) | ✅ 채택, 낮은 확신도 (하위TF 트리거) | RSI와 다른 크로스 메커니즘, 기존 버킷에 묶여 다중검정 비용 이미 통제됨 |
| 스토캐스틱RSI | ❌ 기각 | RSI의 RSI — 새 정보 없이 노이즈/허위신호만 증가 (RSI보다 민감하다는 게 실제로는 단점) |
| 호가창 깊이/스프레드(bookDepth) | ⏸ 보류 | 구조적으로는 새로운 정보(가격 모멘텀 아님)지만, 이미 실측에서 0% 통과한 유동성-미시구조 계열과 같은 실패 패턴 반복 위험 → 이 팩토리에 섞지 말고 별도 독립 스파이크로 격리 |

**결과(1차)**: 필터 축 2→4개, 트리거 축 3→4개로 확장 (4h 기준 최대 18→48 조합).

## 4-2. 보조지표 확장 검토 2차 (사용자 예시에 한정하지 않은 자율 검토)

1차는 사용자가 예로 든 지표만 다뤘다는 지적에 따라, 범위를 넓혀 ADX/DMI, 파라볼릭SAR, KAMA, MFI, OBV/CMF, VWAP, Choppiness Index, 볼린저밴드 스퀴즈까지 검토.

| 지표 | 판정 | 근거 |
| --- | --- | --- |
| **ADX/DMI** | ✅ 채택 (5번째 상위TF 필터) | 기존 필터 4개(EMA/MACD/HMA/일목) 전부 "방향"만 답하고 "추세가 존재하는가"는 아무도 안 답함 — 유일한 진짜 정보 공백. 횡보장(ADX<20)에서 신호 무효화하는 표준 관행이 웹 검증 다수 확인, MTF 구조(상위TF ADX로 확인 후 하위TF 진입)까지 그대로 일치. talib.ADX가 이미 프로젝트 의존성에 있음(1D 용도) |
| 파라볼릭SAR | ❌ 기각 | 결국 "방향"을 답하는 지표라 EMA/HMA/일목과 역할 중복 — 조합 캡(64개) 여유가 4칸뿐이라 정보량이 더 큰 ADX를 우선 |
| KAMA | ❌ 기각 | "추세일 때 빠르게, 횡보일 때 노이즈 무시"라는 목적 자체가 이미 채택한 HMA(지연감소)+ADX(횡보게이트) 조합으로 커버됨 |
| MFI, OBV, CMF | ❌ 기각 | 거래량 기반 방향성 지표인데, 이 프로젝트는 이미 더 정밀한 정보(taker 매수/매도 공격측 구분, `taker_imbalance_momentum`)를 쓰고 있어 후퇴 |
| VWAP | ❌ 기각 | 표준 용도(평균회귀)는 이미 0% 통과로 반증된 archetype. 실행/체결 기준가 용도는 L0 알파탐색이 아닌 다른 레이어의 일 |
| Choppiness Index | ❌ 기각 | ADX와 같은 질문(추세 vs 횡보)을 다른 공식으로 답함 — ADX와 같이 넣으면 또 DEMA/HMA식 중복 |
| **볼린저밴드(vol_breakout)** | 🔧 별도 즉시조치 | 이미 코드로 완성돼 있고 전역 `candidate_families`엔 있는데, **모든 TF의 `_DEFAULT_PER_TF_FAMILIES`에서 빠져있어 실제로는 한 번도 게이트를 통과 시도조차 안 한 상태**임을 확인. 새 코드 없이 설정 한 줄 추가 + 재실행이면 실측 가능 |

**결과(2차)**: 필터 축 4→5개(ADX/DMI 추가), 4h 기준 최대 48→60 조합 (`max_recipes_per_family=64` 캡 대비 여유 4칸). spec 갱신 완료.

## 5. 다음 액션 (우선순위)

0. **[신규, 제로코스트]** `vol_breakout`(볼린저밴드) family를 `_DEFAULT_PER_TF_FAMILIES`에 추가해 재실행 — 코드는 이미 있고 설정 누락으로 한 번도 실측된 적 없음. 구현 스펙 불필요, 설정 한 줄 + 게이트 1회 실행으로 바로 데이터 확보 가능.
1. **[진행 중]** MTF 융합 패턴(상위TF필터×하위TF트리거)을 팩토리화 — spec 완료: `docs/specs/l0-mtf-recipe-factory.md`
2. `DEPRIORITIZED_FAMILY_PRIOR` 재검증 — `vol_term_structure_gate` 등 실측 통과 중인데 격하되어 있는 모순 해소
3. Cross-TF pruning 활성화 — canonical-TF 충돌 해결해 47% 중복을 실제로 컴퓨트 절감으로 전환
4. L1-nested `empty_opportunities` 버그 해결 (4h/6h/1h가 막혀있는 근본 원인)
5. 158초 미계측 wall-clock 구간 정밀 계측

## 6. mtf_fusion Factory 실측 검증 (2026-07-13, run `4h_1783901398`)

`implement` 완료 후 동일 명령(`--phase l1 --timeframe 4h --date 2026-07-13 --trials 1 --seed 42`)으로 재실행해 factory가 실제로 유효한지 실측.

### 6-1. 통과율 — 예상보다 훨씬 강력

| 지표 | mtf_fusion 이전 (baseline) | mtf_fusion 이후 |
| --- | ---: | ---: |
| 전체 evidence rows | 120 | 300 |
| 전체 gate_passed | 78 | 255 |
| 전체 selected_for_l1 | 72 | 87 |
| mtf_fusion evidence | - | 180 |
| **mtf_fusion gate_passed** | - | **177 (98.3%)** |

4h/6h/8h/12h 전 TF에서 mtf_fusion 조합이 거의 전부(177/180) L0 cheap-gate를 통과. net_lcb_bps는 최저 7.5부터 최고 **107.2bps**(`mtf_fusion_1D_hma_slope_macd_cross`, 12h)까지 — 기존 최고 성과 family들과 동등하거나 상회.

### 6-2. Diversity Dedup이 이제 모든 TF에서 실제로 작동함 (기존 미해결 이슈 부수적 해소)

과거 result.md에 남아있던 미해결 항목("intra-TF dedup이 4h에서만 작동, 다른 TF는 selected==passed") — 원인이 게이트 버그가 아니라 **애초에 후보 밀도가 부족해 dedup이 프루닝할 대상이 없었던 것**으로 확인됨. mtf_fusion이 후보 수를 늘리자 6h/8h/12h 전부 `selected(18) < passed(33~83)`로 실제 프루닝 작동 확인:

| TF | selected_for_l1 중 mtf_fusion 비중 | 승자 패턴 |
| --- | --- | --- |
| 4h | 2/18 | `hma_slope+stochastic_cross`, `adx_dmi+macd_cross` |
| 6h | 5/18 | 대부분 `macd_cross` LTF 트리거 |
| 8h | 5/18 | 대부분 `macd_cross` LTF 트리거 |
| 12h | 5/18 | 대부분 `macd_cross` LTF 트리거 |

- **필터 축 5종(EMA/MACD/HMA/일목/ADX) 전부 최소 1개씩 살아남음** — 필터 다양성 실측 확인.
- **트리거 축은 `macd_cross`가 압도적으로 우세**, `rsi_band`/`donchian_retest`/`stochastic_cross`는 버킷 내 상관관계 프루닝으로 대부분 탈락 — 이번 사전 검토에서 "낮은 확신도"로 표시했던 `stochastic_cross`가 실제로도 약함을 실측 확인 (검토 당시 예측과 일치).

### 6-3. 정직하게 보고해야 할 트레이드오프

- **컴퓨트 비용 +41%**: 전체 wall-clock 439.81s → **619.59s**. 주 원인은 `panel_construction`이 26.99s→**98.68s**(+266%)로 급증 — `_htf_ichimoku_cloud_filter`/`_htf_adx_dmi_filter`가 심볼별 Python for-loop로 구현되어 있어(스펙의 구현 노트에서 예견했던 지점) 나머지 3개 필터의 벡터화 경로보다 훨씬 느림. `l0_phase1_cheap_evidence`도 85.19s→158.52s로 증가(평가 대상 3배 증가에 따른 자연 증가).
- **n_ready가 일부 TF에서 소폭 감소**: 8h `53→44`(-17%), 12h `98→92`(-6%), 2h는 `19`로 불변. L0가 더 많고 다양한 후보를 올려보냈음에도 L1 최종 배포 가능 후보 수는 오히려 줄었음 — 아직 원인 미확정(교체된 기존 후보들이 L1 outer-fold에서 근소하게 더 강했을 가능성 vs 단순 run-to-run 변동). **성급한 결론 금지, 후속 확인 필요.**
- 4h/6h/1h는 여전히 별도의 L1-nested `empty_opportunities` 버그로 막혀있음(예상대로, 이번 변경과 무관).

### 6-4. 종합 평가

L0의 "유효하고 다양한 전략 필터링"이라는 목적에 대해 **가설이 실측으로 검증됨**: 상위TF필터×하위TF트리거 팩토리화가 실제로 가동 가능한 다양성을 만들어냈고(5종 필터 전부 생존), 부수적으로 다른 TF들의 dedup 미작동 미스터리까지 해소했다. 다만 컴퓨트 비용 증가와 일부 TF n_ready 소폭 감소는 다음 조치 검토 대상:
1. `_htf_ichimoku_cloud_filter`/`_htf_adx_dmi_filter`의 심볼별 for-loop를 벡터화 (panel_construction 비용 절감)
2. n_ready 감소 원인 진단 — L1 outer-fold 단위로 어떤 후보가 밀려났는지 확인
3. `stochastic_cross`/`donchian_retest` 트리거가 실측으로 거의 항상 탈락 확인됐으므로, 조합 캡(`[LIMIT-08]`) 여유 확보를 위해 향후 축소 검토 가치 있음
