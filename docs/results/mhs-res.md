# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-13 (19차, `mhs_fast_reversal_overlay_redesign` 최초 실측 — §8 참고. 18차까지의 §1-§7은 아래 그대로 보존)
- **Document Date (18차)**: 2026-08-12 (18차, `mhs_universe_horizon_redesign` 적용 재실행 — `--slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon`)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`, `eligible_symbols=446`, `run_elapsed_seconds=706.7`
- **CLI**: `research run portfolio mhs-horizon-diagnostic --slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon`
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **성격**: `docs/specs/mhs_universe_horizon_redesign.md`(Q1-Q5 유니버스/호라이즌/자본배분 개선안 스펙)의 유일한 실행 계약 — `_FAST_BAND.horizons_hours`를 48h 단일값에서 `(24,48,72,96,120,144,168)` 7-후보 그리드로 확장하고, 기존 `--fold-safe-horizon`(slow 전용)에 fast의 leak-free fold-train-only 재검증을 나란히 배선한 **진단 전용** 변경. `PHASE_1_BOOK_BLEND_WEIGHTS`/실제 체결·원장 로직은 17차와 완전히 동일 — **§1/§2/§3의 blend·primary 수치는 17차와 바이트 동일(설계대로), 이번 표의 유일한 신규 정보는 §1-bis의 fold별 fast/slow horizon 재선정 결과다.**

## 1. Primary metrics (`slow_momentum` == `blend`, `fast_reversal` blend 자본 0%) — 16차(버그 있음) vs 17차(수정)

| metric | 16차 (연율화 버그) | 17차 (수정 후) | 배율 |
| :--- | ---: | ---: | ---: |
| `primary_autocorr_sharpe` | 0.5257 | 0.5257 | 1.0x(무영향, 설계대로) |
| `primary_naive_sharpe` | 0.1333 | **0.4616** | 3.46x (= √12) |
| `primary_net_ann` | 0.0081 | **0.0972** | 12.0x |
| `primary_geometric_cagr` | 0.0063 | **0.0784** | 12.4x |
| `primary_max_drawdown` | -0.2269 | -0.2269 | 1.0x(무영향, 설계대로) |
| `primary_annualized_turnover` | 3.56 | **42.68** | 12.0x |
| `stress_naive_sharpe` (x3 cost) | +0.0410 | **+0.1420** | 3.46x |
| `blend.failure` | null | null | — |

`docs/specs/mhs_execution_annualization_fix.md`가 예측한 정확히 그 배율(12x 지수, √12 Sharpe류)로 재현됨 — 버그 진단이 실측으로 확정됨. **연복리 수익률 7.84%/년**이 이번 코드베이스의 실행 스택이 실제로 만들어내고 있던 진짜 수치였고, 지금까지 문서화된 0.03~0.63%는 측정 오류였다.

**18차 재확인**: 위 17차 수치(`primary_autocorr_sharpe=0.5257`, `primary_geometric_cagr=7.84%`, `stress_naive_sharpe=+0.1420` 등)가 `--fold-safe-horizon` 신규 배선 이후에도 **소수점까지 완전히 동일**함을 직접 재실행으로 확인 — fast_reversal의 fold-train-only 재검증(§1-bis)이 진단 전용이며 실제 배분·리플레이 경로를 전혀 건드리지 않는다는 계약을 실측으로 증명한다.

## 1-bis. Fast/Slow horizon fold-train-only 재검증 (18차 신규, `mhs_universe_horizon_redesign.md` §4)

`docs/results/mhs-res.md` 17차까지 `fast_reversal`의 0% 배분 근거는 **48h 단일 밴드의 전역 prescreen**뿐이었다(discovery/qualification 7-후보 그리드가 fast에 한 번도 적용되지 않음, 스펙 §3.1). 이번 18차가 처음으로 각 anchored fold의 **train 구간에만 국한된(leak-free)** `fold_train_only_discovery_qualification`을 fast의 전체 후보 그리드(24~168h, sign=-1)에 대해 실행했다:

| fold | validation | `slow_horizon_hours` | `slow_horizon_source` | `fast_horizon_hours` | `fast_horizon_source` |
| ---: | :--- | ---: | :--- | ---: | :--- |
| 0 | 2023 | 168 | `frozen_default` | 48 | `frozen_default` |
| 1 | 2024 | 168 | `frozen_default` | 48 | `frozen_default` |
| 2 | 2025 | 168 | `frozen_default` | 48 | `frozen_default` |

3개 fold 전부 fast/slow 모두 `frozen_default`로 폴백 — **fold-train 표본 어디에서도 어떤 후보 호라이즌도 admission floor(`|t| >= 2.0`)를 넘지 못했다.** slow의 결과는 `ADR_20260811_MHS_FOLD_SAFE_HORIZON_SELECTION`의 기존 결론(admission_t가 fold-local 표본엔 과도하게 엄격)을 재확인하는 무변화이고, **fast의 결과가 이번 실행의 신규 정보다**: 48h 단일값이 아니라 7개 후보 전체를 leak-free로 훑어도 admission에 실패한다 — `docs/results/mhs_horizon_diagnostic.json`의 48h 전역 prescreen도 이번 실행에서 전 비용구간(0~8bps) `net_t`가 **-0.06 ~ -2.39로 전부 음수**(과거 문서화된 +0.577과 달리 이번 유니버스/EMA 스무딩 하에서는 0비용에서도 이미 음수)로 재확인됨.

**결론(Q4 최종 갱신)**: "fast 탐색이 무의미한가"라는 질문에 대해, 이번 결과는 **"48h만 봐서 성급했다"는 가설을 기각**한다 — 넓은 그리드에서도 동일한 결론이므로, 가격 기반 횡단면 반전 신호 자체가 이 유니버스·비용 구조에서 edge가 없다는 판정이 leak-free 증거로 강화되었다. 스펙 §4.2가 제시한 두 대안 방향(① 리스크/타이밍 오버레이로 재배선, ② top-30 로스터 밖 중간유동성 구간 재스캔)이 이제 유일하게 남은, 근거 있는 다음 단계다 — 자본 배분(`PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"]=0.0`)은 이번 진단만으로는 변경하지 않는다(진단 전용 계약).

## 2. Research GO gate

| field | 16차 | 17차 |
| :--- | ---: | ---: |
| `eligible` | `false` | `false` |
| `evaluated_folds` | 3 | 3 |
| `folds_passed` | 2 | 2 (무변화 — 게이트는 Sharpe/stress 기준이라 연율화 버그와 무관) |
| `reason_codes` | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` | 동일 |

여전히 `eligible=false` — Sharpe 플로어(0.6)와 `UNSPECIFIED_POLICY`가 차단. **Research GO 게이트 자체는 원래 Sharpe/일봉 기반이라 이번 버그의 영향을 받지 않았다** — 즉 이 버그는 GO/NO-GO 판정을 뒤집지 않았고, 오직 "자산증식이 얼마나 되는지"를 보여주는 CAGR 헤드라인만 왜곡했다.

## 3. Fold detail (17차, CAGR 신규 노출)

| fold | validation | `primary_autocorr_sharpe` | `primary_geometric_cagr` | `primary_max_drawdown` | `stress_naive_sharpe` | `failures` |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | 2023 | +0.8046 | **+9.36%** | -0.1089 | +0.2111 | (none) — **통과** |
| 1 | 2024 | -0.2672 | **-4.99%** | -0.1743 | -0.8201 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | 2025 | +1.5047 | **+48.22%** | -0.2185 | +0.9310 | (none) — **통과** |

fold별 CAGR이 처음으로 신뢰 가능한 수치로 노출됨 — fold1(2024)만 명확히 손실 구간, fold0/fold2는 양호.

## 4. 배포 준비도(꼬리위험) — 버그 수정의 부수 효과

`compute_deployment_readiness`의 블록부트스트랩이 `mean_block_bars=168`을 "168시간(1주)"이 아닌 "168개 5분바(14시간)"로 잘못 해석하던 것도 같이 고쳐짐:

| field | 16차(버그) | 17차(수정) |
| :--- | ---: | ---: |
| `probability_mdd_over_30pct` | 62.65% | **47.0%** |
| `probability_final_wealth_below_initial` | 21.35% | **16.8%** |
| `calmar` | 0.028 | **0.345** |

여전히 낙관적인 수치는 아니지만(30%대 낙폭 확률이 여전히 47%), 이전에 보고했던 "62.65% 확률로 30% 넘게 물린다"는 진짜보다 비관적인 추정이었다.

## 5. 부수 검증 — 인과성 수정(RC-3, 16차에서 이미 확인, 17차도 동일)

동일 실행에서 `xs_rank_ic`/`date_clustered_regression`이 처음으로 거래가능 선행수익률로 계산됨:

| field | 값 |
| :--- | ---: |
| `xs_rank_ic.mean_ic` | **-0.0409** (수정 전 겹침 통계는 +0.0957이었음) |
| `xs_rank_ic.t_stat` | **-46.07** |
| `date_clustered_regression.past_beta` | -0.0187 |
| `date_clustered_regression.past_t` | -1.39 |

부호 반전은 `docs/specs/mhs_alpha_engine.md` §3의 사전 스크래치 측정(`-0.0278`, t=-33)과 방향이 일치 — 헤드라인 "신호 품질" 지표가 지금까지 거래 불가능한 겹침 구간으로 계산돼 왔음을 실제 프로덕션 리포트에서도 확인.

## 6. 최근 진단·조사 계보

- **`OHLCV_IMMEDIATE_TAKER` 결손 가드**(`ADR_20260812_MHS_IMMEDIATE_TAKER_FILL_GUARD`): 진단 완주 안정화.
- **fold Sharpe 편차 근본 원인**(`ADR_20260812_MHS_MOMENTUM_REGIME_DIAGNOSIS`): 2022(LUNA/FTX) 전원 음수 — horizon 선택이 아닌 신호의 체계적 붕괴장 취약성.
- **전략 유형 재검토**(`ADR_20260812_MHS_MOMENTUM_STRATEGY_REDESIGN_REVIEW`): 완전 시장중립과 붕괴장 생존의 트레이드오프 확인, `src/mhs/regime.py` 레짐 프록시 신설.
- **크래시 레짐 방향성 틸트 오버레이**(`ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY`, 15차): `alpha=0.2` 실전 검증 — `folds_passed=1/3`, `stress_naive_sharpe=-0.0844`.
- **MHS 알파 엔진 재구축**(16차, `docs/specs/mhs_alpha_engine.md`): RC-1(종목별 데드밴드가 달러중립·레짐 디리스킹을 파괴) → 포트폴리오 리밸런스 트리거, RC-2(단일 호라이즌 argmax 고분산 선택) → 19-호라이즌 동일가중 앙상블. `primary_autocorr_sharpe` 0.182→0.526, MDD -0.392→-0.227, `folds_passed` 1→2/3.
- **실행 원장 연율화 버그 수정**(17차, `docs/specs/mhs_execution_annualization_fix.md`): `execution_timeframe=5m` 원장에 1시간 격자 상수를 적용하던 집계 버그(지수 12배 축소) 규명·수정. 전략 무변경, CAGR 0.63%→**7.84%**, 연율수익 12배, 낙폭확률(30%+) 62.65%→47.0%로 재노출. Research GO 게이트(Sharpe 기반)는 버그 영향을 받지 않아 무변화.
- **유니버스·호라이즌 개선안 스펙 + fast fold-safe 재검증**(이번 18차, `docs/specs/mhs_universe_horizon_redesign.md`): Q1(유동성 median-split 720h)·Q2(top-30 로스터)·Q3(호라이즌 탐색/앙상블 가중)·Q4(fast 0% 배분 재검토)·Q5(연관 백로그 우선순위)를 근거 기반으로 분석. 유일한 코드 계약은 `_FAST_BAND` 그리드를 48h→7-후보(24~168h)로 확장하고 `fold_train_only_discovery_qualification`을 fast에도 배선(진단 전용, 자본 불변). 실측 결과: 3개 fold 전부 `fast_horizon_source=frozen_default` — 넓은 그리드에서도 admission 실패, "48h만 봐서 성급했다"는 가설 기각. blend/primary 수치는 17차와 바이트 동일 확인.

## 7. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| `beta_neutralize`, `ensemble_signal=vol_normalized`, 트리거 임계값(`crash_regime_tilt_alpha`, `rebalance_filter`) — **일괄** fold-train-only 결합 탐색 설계 필요(개별 임의값 시도는 p-hack 우려로 이미 기각, `ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY`) | 미착수 (`mhs_universe_horizon_redesign.md` §5 순위 4) |
| 시계열(방향성) 절대모멘텀 재검토 — 사전스크린 Sharpe 1.0~1.4로 횡단면 앙상블(0.72~0.86)보다 높게 관측됨 | 미착수, 연율화 버그 수정 후 fold-train-only로 재검증 필요(`docs/specs/mhs_execution_annualization_fix.md` §3) |
| 펀딩비 캐리 신호 — 손익 분해 부호 모순 재검증 필요 | 미착수, 액면 그대로 신뢰 금지 |
| fold1(2024) 잔여 미달 원인 조사 | 미착수 — `yearly_net_t` 필드가 이미 노출되어 있어 `--discovery-gate` 재실행 후 2024열만 조회하면 됨(신규 계산 불필요, `mhs_universe_horizon_redesign.md` §5 순위 2) |
| `fast_reversal`의 독립적 `CAPITAL_INVARIANT_BREACH`(음의 자기자본) 근본 원인 조사 | **부분 해결 + 신규 결함 발견(19차, §8)**: root cause 규명(edge 없는 신호를 무방비 레버리지로 장기 리플레이한 결과, 데이터 버그 아님) + Pass-1 reference replay용 방어 플로어 구현 완료. 그러나 Pass-2(primary, 실제 보고값)는 미보호로 남아있어 여전히 크래시(더 악화: -144→**-66,427** 자기자본) — **Pass-2 플로어 확장이 신규 최우선 항목**(§8.3) |
| top-30 로스터 크기(N) fold-train-only 재검증(Q2) | 미착수, `ADR_20260810_MHS_BOOK_ADMISSION_VOL_MASK`가 미측정 상수로 인정한 항목(`mhs_universe_horizon_redesign.md` §5 순위 3) |
| fast 대안 방향 — ① 리스크/타이밍 오버레이 재배선, ② top-30 밖 중간유동성 구간 재스캔 | **① 착수(19차, §8)**: `trend_efficiency_scale` 오버레이 opt-in 구현·유닛테스트 완료(`docs/specs/mhs_fast_reversal_overlay_redesign.md`), fold-train-only 실측 비교는 미실행. ② 미착수. |
| 유동성 lookback 720h 민감도(Q1) | 미착수, 낮은 우선순위(4개 모듈 공유, 문제 증거 없음) |
| Pass-2(primary) 실행 replay에도 `min_equity_fraction` 플로어 확장 | **신규, 미착수 — 최우선(blocking)**: 19차 실측(§8.3)이 Pass-1만 보호하는 현재 구현이 Pass-2를 오히려 더 심하게 무방비로 노출시킴을 실증 — fast에 자본을 복원하는 어떤 경로든 이 확장이 선행되어야 함 |

## 8. 19차 — `fast_reversal` 오버레이 재설계 최초 실측 (`docs/specs/mhs_fast_reversal_overlay_redesign.md`)

**CLI**: `research run portfolio mhs-horizon-diagnostic --discovery-gate --fold-safe-horizon --output-tier full` (2021-2025 dev, `status=COMPLETE`).

**⚠️ 실행 설정 주의**: 이번 실행은 `--slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger`(18차가 사용한 알파엔진 개선 플래그)를 포함하지 않았다 — 기본값(`single_horizon`/`per_symbol_deadband`)으로 실행됨. 따라서 §8.2의 `slow_momentum`/`blend` 수치는 §1의 17-18차 헤드라인(`primary_autocorr_sharpe=0.5257`)과 **직접 비교 불가**하다(실행 설정 차이이지 이번 스펙 변경에 의한 회귀가 아님). §8.1(B)과 §8.3(A)의 `fast_reversal` 자체 수치는 이 플래그와 무관해 유효하다.

### 8.1 B — EMA 반전신호 스무딩 버그 수정: 확인됨

`fast_reversal` 48h 전역 prescreen, 0bps 시작점:

| cost tier | 18차(EMA 스무딩 버그 있음) | 19차(수정 후) |
| :--- | ---: | ---: |
| 0.0bps `net_t` | -0.06 (음수, 과거 문서화된 +0.577과도 불일치) | **+0.067 (양수로 복원)** |
| 2.0bps `net_t` | (음수 심화) | -0.55 |
| 8.0bps `net_t` | -2.39 | -2.41 (거의 동일) |

예측대로 재현: 스무딩 제거로 0비용 부호가 양수로 복원되어 `discovery.py`의 leak-free 스캔(스무딩 미적용)과 내부 정합성을 회복했다. 그러나 2bps부터 음수로 꺾이는 건 동일 — **"신호 자체에 근본적으로 edge 없음" 결론(18차 §1-bis)은 그대로 유지**, 버그는 진단 헤드라인의 왜곡만 고친 것이지 edge를 만들어내지 않았다.

### 8.2 슬로우/블렌드 (참고용, §1 헤드라인과 비교 불가 — 위 주의 참조)

| field | 값 (single_horizon/per_symbol_deadband) |
| :--- | ---: |
| `slow_momentum.primary_autocorr_sharpe` | 0.1933 |
| `slow_momentum.primary_geometric_cagr` | -0.28% |
| `slow_momentum.stress_naive_sharpe` | -0.2362 |
| `research_go.folds_passed` | 1/3 |
| `research_go.reason_codes` | `CAPITAL_INVARIANT_BREACH`, `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `RELEVANT_EXECUTION_DATA_GAP`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` |

`fold_safe_horizon_selection`은 3개 fold 전부 fast/slow `frozen_default` 유지 — 18차와 무변화.

### 8.3 A — 자본붕괴 방어: 부분 성공 + Pass-2 미보호 신규 발견

`_book_outcome`의 Pass-1(reference) replay에만 `min_equity_fraction=0.5` 방어 플로어를 배선했다:

- **Pass-1은 의도대로 크래시 없이 완주** — 단, `equity_floor_breached_at`가 **2021-09-14**부터 시작(2021-2025 전체 구간의 초반 8.5개월 만에 이미 초기자본 50%를 소진하고 있었다는 사실이 이번에 처음 노출됨).
- **Pass-2(실제 `primary`로 보고되는 replay)는 미보호로 남아 여전히 크래시**, 오히려 악화됨:

| field | 18차 | 19차 |
| :--- | ---: | ---: |
| `books.fast_reversal.failure.reason` | `CAPITAL_INVARIANT_BREACH` | `CAPITAL_INVARIANT_BREACH` (미해결) |
| 붕괴 시각 | 2025-07-14 | 2025-07-16 |
| `pre_trade_equity` | -144.52 | **-66,427.67** (약 460배 악화) |

**원인**: Pass-1이 2021-09부터 계속 강제 플랫이라 실현 변동성이 인위적으로 거의 0으로 측정되고, 여기서 파생되는 `_pnl_vol_target_scale`이 Pass-2를 사실상 축소 없이(거의 풀 노출로) 통과시킨다. Pass-2 자체엔 플로어가 없어 무방비 상태로 재차 파산한다 — Pass-1만 보호하는 현재 구현은 "크래시를 없앤다"는 원래 목표를 달성하지 못했고, 오히려 Pass-2의 파산 규모를 키우는 부작용을 만들었다.

**결론**: root cause 진단(edge 없는 신호를 무방비 레버리지로 장기 리플레이한 결과, 데이터 무결성 버그 아님)은 실측으로 확정됐으나, 구제 조치는 미완성이다. **Pass-2에도 동일 플로어를 확장하는 것이 fast_reversal 자본 복원 경로의 신규 최우선(blocking) 항목**이다(§7 갱신).

### 8.4 C — 트렌드효율성 오버레이: 코드/유닛테스트만 존재, 실측 비교 미실행

`trend_efficiency_scale`(`src/mhs/regime.py`) + `--trend-efficiency-overlay` opt-in 플래그 구현 및 10개 계약 시나리오 유닛테스트(104 passed) 완료. `regime_scale`-only vs `regime_scale * trend_efficiency_scale`의 fold-train-only Sharpe/MDD 비교(스펙 §4 3번 항목)는 이번 19차에 **실행하지 않았다** — 오버레이 기본값을 `True`로 승격하기 전 반드시 선행되어야 하는 실측이며, §7 백로그에 남아 있다.
