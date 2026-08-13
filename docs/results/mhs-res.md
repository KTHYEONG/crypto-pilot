# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-13 (21차, `mhs_strategy_foundation_reset` + `mhs_execution_friction_and_exposure_layers` 실측 완료 — §10 참고, `ADR_20260813_MHS_STRATEGY_FOUNDATION_RESET`. 20차까지의 §1-§9는 아래 그대로 보존)
- **Document Date (20차)**: 2026-08-13 (20차, `mhs_capital_floor_and_overlay_validation` 실측 완료 — §9 참고. 19차까지의 §1-§8은 아래 그대로 보존)
- **Document Date (19차)**: 2026-08-13 (19차, `mhs_fast_reversal_overlay_redesign` 최초 실측 — §8 참고. 18차까지의 §1-§7은 아래 그대로 보존)
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
| fold1(2024) 잔여 미달 원인 조사 | **정정(20차, §9.3)**: 19차가 기록한 "yearly_net_t로 조회 가능"은 틀린 전제였음 — 그 필드의 discovery window는 `DISCOVERY_END=2023-12-31`로 고정돼 2024/2025 열이 구조적으로 항상 NaN(신규 계산 필요, 노출된 필드로 불가). 대신 `fold1_strict` 원장(ledger.parquet, `--output-tier full`)의 월별 손익을 직접 조회해 진단: 12개월 중 8개월이 음수, 단일 크래시 월 없는 만성적 부진 패턴(2022 LUNA/FTX류 단일 붕괴와 다름) — trend_efficiency_overlay로 해결 시도했으나 §9.2에서 오히려 악화됨이 실증돼 가설 기각, 근본 해결은 미해결 |
| `fast_reversal`의 독립적 `CAPITAL_INVARIANT_BREACH`(음의 자기자본) 근본 원인 조사 | **완전 해결(20차, §9.1)**: Pass-2/stress/patient_reference/touch/ladder 전체에 동일 방어 플로어 확장 배선 완료. 실측 재실행으로 `books.fast_reversal.failure=None` 확인(크래시 완전 해소) |
| top-30 로스터 크기(N) fold-train-only 재검증(Q2) | 미착수, `ADR_20260810_MHS_BOOK_ADMISSION_VOL_MASK`가 미측정 상수로 인정한 항목(`mhs_universe_horizon_redesign.md` §5 순위 3) |
| fast 대안 방향 — ① 리스크/타이밍 오버레이 재배선, ② top-30 밖 중간유동성 구간 재스캔 | **① 실측 완료, 부정적 결과(20차, §9.2)**: fold-train-only로 `regime_scale`-only vs `regime_scale*trend_efficiency_scale` 비교 — 3-fold 전부 `stress_naive_sharpe` 악화, fold1(2024, 가장 필요했던 fold)은 primary/stress 둘 다 악화. 기본값 `False` 유지 권고, 이 신호 형태로는 재시도하지 않음. ② 미착수. |
| 유동성 lookback 720h 민감도(Q1) | 미착수, 낮은 우선순위(4개 모듈 공유, 문제 증거 없음) |
| top-level "blend" 북 실행 replay가 `regime_scale`을 반영하지 않던 결함 | **신규 발견 + 해결(20차, §9.4)**: `_run_books_concurrent`의 `blend_replay`가 `w_fast_execution`/`w_slow_execution`으로 독립 재구성되며 `regime_scale`(기존 `_regime_cash_scale` 포함, 신규 `trend_efficiency_overlay`도 마찬가지) 곱셈이 전혀 반영되지 않았음 — 17차 이래 보고된 `blend.primary_autocorr_sharpe=0.5257` 헤드라인이 실제로는 regime scale 없이 계산된 값이었음(코드 주석의 "prescreen/tail/execution diagnostics are comparable" 약속과 불일치). 수정 후 재실행 실측: `primary_autocorr_sharpe` 0.5257→**0.5196**, `primary_max_drawdown` -0.2269→**-0.2015**(개선), `stress_naive_sharpe` 0.1420→0.1295. Research-GO는 fold 경로(항상 정상 배선)로 판정되므로 게이트 결과 자체는 무변화(`folds_passed=2/3`) |

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

## 9. 20차 — 자본붕괴 완전방어·오버레이 실측·신규 결함 발견 (`docs/specs/mhs_capital_floor_and_overlay_validation.md`)

19차가 남긴 세 항목(Pass-2 미보호, 오버레이 미검증, fold1 원인 미상)을 실제 데이터로 검증·수정했다. **이번 차수는 판단이 아니라 코드 수정 + 실측 확인까지 완료**했다.

**CLI**: `research run portfolio mhs-horizon-diagnostic --slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon [--discovery-gate] [--trend-efficiency-overlay] --output-tier full` (2021-2025 dev, 4회 실행 — 베이스라인, 오버레이 on, 수정검증용, 총 `status=COMPLETE`).

### 9.1 A — 자본붕괴 방어: Pass-2/stress까지 확장, 완전 해결

`SCENARIO_MHS_CAPITAL_FLOOR_PASS2_STRESS_PROTECTED_03` (`docs/specs/mhs_capital_floor_and_overlay_validation_contract.json`) — 실데이터 재실행으로 검증됨.

19차는 `_book_outcome`의 Pass-1(reference)에만 `min_equity_fraction` 플로어를 배선해 Pass-2(실제 보고값)가 여전히 크래시했다(-144.52→-66,427.67로 악화). 이번 20차는 Pass-2/stress/patient_reference/touch/ladder 전체 replay 호출에 동일 플로어(`MHS_REFERENCE_PASS_EQUITY_FLOOR`)를 확장 배선했다.

| field | 19차 | 20차 |
| :--- | :--- | :--- |
| `books.fast_reversal.failure` | `CAPITAL_INVARIANT_BREACH` (Pass-2 미해결) | **`None` (완전 해소)** |
| `pre_vol_target_reference.equity_floor_breached_at` | (필드 자체는 19차부터 존재) | 6,086회 강제 플랫 처리 — 2021-09-14부터 시작 |
| `primary.equity_floor_breached_at` | — | 0회 (Pass-1 보호 덕분에 Pass-2 자체는 플로어를 칠 필요조차 없었음) |
| `fast_reversal.prescreen[0.0].net_t` | +0.067 | +0.067 (무변화 확인 — B의 결론 재확인) |

**결론**: root cause(edge 없는 신호를 무방비 레버리지로 장기 리플레이한 결과)는 그대로이지만, 진단 파이프라인이 이제 크래시 없이 완주하며 결과를 정직하게(타입드 필드로) 보고한다. §7의 최우선 blocking 항목 해소.

### 9.2 C — 트렌드효율성 오버레이 실측: 도움 안 됨, 기본값 유지

`SCENARIO_MHS_TREND_EFFICIENCY_OVERLAY_FOLD_VALIDATION_NEGATIVE_04` (`docs/specs/mhs_capital_floor_and_overlay_validation_contract.json`) — 실데이터 재실행, 부정적 결과.

`--slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon`(18차와 비교 가능한 정확한 플래그)로 오버레이 on/off 각각 재실행해 fold별 지표를 비교했다:

| fold | validation | `primary_autocorr_sharpe` (off→on) | `stress_naive_sharpe` (off→on) | `primary_max_drawdown` (off→on) |
| ---: | :--- | :--- | :--- | :--- |
| 0 | 2023 | 0.8046 → 0.8213 (소폭 개선) | **0.2111 → 0.0301 (악화)** | -0.1089 → -0.1020 (개선) |
| 1 | 2024 | **-0.2672 → -0.3395 (악화)** | **-0.8201 → -1.0475 (악화)** | -0.1743 → -0.1701 (소폭 개선) |
| 2 | 2025 | 1.5047 → 1.5303 (소폭 개선) | **0.9310 → 0.7774 (악화)** | -0.2185 → -0.2215 (소폭 악화) |

**stress_naive_sharpe가 3-fold 전부 악화**되고, **가장 도움이 필요했던 fold1(2024)이 primary/stress 둘 다 악화**됐다 — 애초에 "2024는 단일 크래시가 아니라 만성적 choppy 구간이니 trend_efficiency_overlay가 도움될 것"이라는 가설(§9.3)이 실측으로 기각됨. **`trend_efficiency_overlay` 기본값 `False` 유지 결정 — 이 신호 형태(efficiency_ratio 기반 gross 축소)로는 재시도하지 않는다.** 코드/opt-in 플래그/유닛테스트는 남겨두되(향후 다른 오버레이 신호의 배선 지점으로 재사용 가능), Research-GO 판정에 영향 없음(fold path는 항상 정상 배선이었으므로 이 실측 자체가 유효한 증거).

### 9.3 fold1(2024) 원인 조사 — 이전 문서의 오류 정정

19차/18차까지의 문서(§7)는 "yearly_net_t 필드가 이미 노출되어 있어 신규 계산 없이 조회만 하면 된다"고 기록했으나 **이는 틀린 전제였다** — `discovery_qualification.momentum.yearly_net_t`는 discovery window(`DISCOVERY_END=2023-12-31`)로 고정된 연도만 채워지며 2024/2025 열은 구조적으로 항상 `NaN`이다(신규 계산 없이는 확인 불가능).

정확한 방법: `--output-tier full`로 실행하면 `ledger.parquet`에 `replay_id='fold1_strict'`로 fold1 전용 원장이 보존된다(다른 replay와 파일을 공유하지만 `replay_id` 컬럼으로 구분됨, 덮어쓰기 아님). 이 원장의 월별 손익을 직접 조회한 결과:

| 월 | 손익 부호 | 비고 |
| :--- | :--- | :--- |
| 2024-02, 04, 06, 07, 08, 09, 11 | 음수 (7개월) | 단일 크래시 없이 산발적 손실 |
| 2024-03, 05, 10, 12 | 양수 (4개월) | |
| 최저 자기자본 | 2024-10-03, 초기자본의 -12.65% (0.8735) | 이후 12월까지 부분 회복 |

**12개월 중 8개월이 순손실**(순수익 4개월) — 2022(LUNA/FTX, `ADR_20260812_MHS_MOMENTUM_REGIME_DIAGNOSIS`)처럼 특정 사건 하나가 아니라 연중 만성적 부진 패턴. 이 관찰이 §9.2의 trend_efficiency_overlay 가설을 낳았으나 실측으로 기각됐다 — **fold1(2024) 미달의 진짜 근본 원인은 여전히 미해결**로 남는다.

### 9.4 신규 발견 + 해결 — top-level "blend" 실행 replay가 `regime_scale`을 전혀 반영하지 않던 결함

§9.2 실측 과정에서 top-level `blend.primary_autocorr_sharpe`가 `trend_efficiency_overlay` on/off와 무관하게 소수점까지 완전 동일(`0.525673922813482`)하게 나오는 이상 현상을 발견, 추적 결과 **17차 이래 존재해온 별개의 사전 결함**을 확인했다: `_run_books_concurrent`의 `blend_replay`(실제 primary/stress replay에 쓰이는 가중치)가 `w_fast_execution`/`w_slow_execution`으로부터 독립적으로 재구성되며, `blend_1h`/`blend_step`(prescreen/tail 전용)에는 적용되는 `regime_scale`(기존 `_regime_cash_scale` + 신규 `trend_efficiency_overlay` 둘 다) 곱셈이 **한 번도 반영된 적이 없었다**. 코드 주석은 "top-level prescreen/tail/execution diagnostics are comparable to fold primary evidence"라고 명시했지만 execution(replay) 쪽은 실제로 지켜지지 않고 있었다.

**수정**: `regime_scale`을 `_run_books_concurrent`에 새 옵셔널 파라미터로 전달해 `blend_replay`에도 동일하게 곱하도록 배선(`fast_reversal`/`slow_momentum` 개별 북은 영향 없음 — 포트폴리오 레벨 스케일이므로 blend에만 적용, fold 경로의 기존 설계와 정합). 회귀 테스트(`test_regime_scale_reaches_blend_replay_not_only_prescreen`, `tests/unit/application/research/mhs/test_evaluation.py`) 추가 — 수정 전 코드에서는 실패하고 수정 후 통과함을 확인.

실측 재실행 결과:

| field | 수정 전(버그, 18-19차 보고값과 동일) | 수정 후(20차) |
| :--- | ---: | ---: |
| `blend.primary_autocorr_sharpe` | 0.5257 | **0.5196** |
| `blend.primary_geometric_cagr` | 7.84% | **7.56%** |
| `blend.primary_max_drawdown` | -0.2269 | **-0.2015 (개선)** |
| `blend.stress_naive_sharpe` | 0.1420 | 0.1295 |
| `slow_momentum.primary_autocorr_sharpe`(개별 북, 영향 없어야 함) | 0.5257 | 0.5257 (무변화 확인) |
| `research_go.folds_passed` | 2/3 | 2/3 (무변화 — fold 경로는 항상 정상 배선이었음) |

효과 크기는 작다(Sharpe -1.2%, MDD 11% 개선) — `_regime_cash_scale`이 5년 전체에서 자주 발동하지 않기 때문으로 보인다. Research-GO 판정 자체(`eligible=False`)는 영향받지 않는다(게이트는 항상 정상 배선이던 fold 경로 기준). **이번 발견의 가치는 진단 리포트 헤드라인의 신뢰도 회복이며, Research-GO 판정 로직 자체의 변경은 아니다.**

### 9.5 요약

| 항목 | 상태 |
| :--- | :--- |
| A. 자본붕괴 방어 | ✅ 완전 해결 (Pass-1/2/stress/기타 전부 보호, 실측 확인) |
| B. EMA 반전신호 스무딩 | ✅ 19차 확인 유지 (재확인됨, +0.067) |
| C. 트렌드효율성 오버레이 | ✅ 실측 완료 — **부정적 결과**, 기본값 `False` 유지 |
| D. fold1(2024) 근본원인 | 부분 진단(만성적 choppy 패턴 확인) — **근본 해결은 미해결로 이월** |
| E. blend replay `regime_scale` 미반영 | ✅ 신규 발견 + 해결, 실측 확인 |

## 10. 21차 — 계측기 정합화(RC-1) 및 실행 마찰·노출 레이어 재조준 (`ADR_20260813_MHS_STRATEGY_FOUNDATION_RESET`)

> 20차까지 반복된 무개선의 **기계적 원인**을 규명했다: 모든 유의성 계측기(`prescreen`/`phase`/`tail`/`xs_rank_ic`)가
> **자본 0%인 참조책**을 측정해 왔고, `research_go.eligible`은 성과와 무관하게 **구조적으로 영구 False**였다.
> 스펙: `mhs_strategy_foundation_reset.md`(P0), `mhs_execution_friction_and_exposure_layers.md`(P1~P4).
> **CLI**: `research run portfolio mhs-horizon-diagnostic --slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --output-tier full` (2021-2025 dev, `status=COMPLETE`, `run_elapsed_seconds=570.7`).

### 10.1 RC-1 — 참조책 vs 집행책 실측 재현

`_book_outcome`은 `prescreen`/`phase`/`tail`을 자본 0%인 `weights_step`(446종목 단일 168h, 로스터·앙상블 없음)으로,
`primary`/`stress`는 100% 자본이 도는 `replay_weights_step`(로스터+19-호라이즌 앙상블+역변동성 틸트+regime scale)으로
계산해 왔다 — 두 개는 다른 포트폴리오인데 같은 book 이름 아래 보고돼 왔다. 이번 21차에 신설된 `executed_prescreen`이
동일 계측기(`cost_response_curve`)를 집행책에 겨눈 결과를 처음으로 나란히 노출한다:

| book | 지표 (4.18bps) | 참조책 (`prescreen`) | **집행책 (`executed_prescreen`)** |
| :--- | :--- | ---: | ---: |
| slow_momentum | net_t | 0.642 | **1.505** |
| slow_momentum | net_sharpe | 0.287 | **0.673** |
| blend | net_t | 0.779 | **1.729** |
| fast_reversal | net_t | -1.230 | **-1.669** |

**이것이 20차까지 "여러 번 수정에도 개선 없음"의 기계적 원인이다** — 계측기가 실제 전략에 연결돼 있지 않았으므로
어떤 수정도 계측값을 의도한 방향으로 움직일 수 없었다. slow_momentum/blend는 참조책 기준보다 뚜렷이 강하고,
fast_reversal은 참조책 기준보다 오히려 더 뚜렷하게 음수다(-1.230→-1.669) — 0% 배분 결정 자체는 이번 실측으로도 지지된다.

### 10.2 다중검정 회계 — 처음 노출

`trials_attempted`는 20차까지 `1`로 하드코딩돼 있었다. 2021-2025 dev 창에서 20차에 걸쳐 순차 탐색해 온 실제
시행수(`MHS_SEARCH_TRIALS_ATTEMPTED=20`)를 반영한 `deflated_sharpe_ratio`가 이번에 처음 계산됐다:

```
trials_attempted = 20
deflated_sharpe_ratio = 0.532
```

다중검정을 보정하면 진짜 Sharpe가 양수일 확률은 **53.2%** — 동전던지기 수준이다. `folds_passed=2/3`만으로
안도할 수 없다는 것이 수치로 확인됐다.

### 10.3 Research-GO 게이트 — 도달 가능해짐 (여전히 `eligible=False`)

`_mhs_research_go`는 `MHS_GO_REASON_UNSPECIFIED_POLICY`를 **무조건** append해 왔다(`reasons = list(book_reasons); ...;
reasons.append(UNSPECIFIED_POLICY)`) → `eligible = not reasons`가 성과와 무관하게 영구 `False`였다. 이번 21차부터
`MHS_REGISTERED_POLICY_THRESHOLDS`(`cap_30_roster`, `primary_annual_return`)가 **둘 다 등록될 때만** 이 append를
건너뛴다. 두 값 모두 아직 의도적으로 `None`(미등록)이므로:

```
research_go.eligible = False
research_go.reason_codes = [PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY]
```

— 20차까지와 표면적으로 동일한 결과이지만 의미가 다르다: **게이트가 이제 성과에 반응할 수 있는 구조**이고,
Sharpe 플로어(0.6)를 넘기면 정책 임계값 등록만으로 `eligible=True`가 가능하다. 임계값 등록은 성과와 무관한
사전등록 정책 결정이라 이번 계약 범위에서 값을 채우지 않았다(§11 후속 판단 필요).

### 10.4 P1 — 로스터 재정규화 회전율 가설: 실측 기각

`renormalize_within_mask`가 회전율의 주범이라는 원가설을 로스터 네이티브 랭킹으로 직접 반증했다:

| 변형 | 회전율/년 | net_t@0bps |
| :--- | ---: | ---: |
| R0 생산 (wide rank → roster 투영) | 42.74 | 2.042 |
| R1 로스터 네이티브 랭킹 (투영 제거) | 42.28 (**-1%**) | 2.058 |

투영을 완전히 제거해도 회전율은 거의 그대로다. 회전율은 **gross에 비례**했고(위상 트랜치 단계에서 이미
회전율/gross 125→51로 절반 이하), 20차가 재정규화를 "마찰 +37%"로 읽은 것은 실은 **레버리지 복원**이었다.
→ 로스터 재설계는 착수하지 않는다.

동일 gross(0.70)로 재스케일해 노출 레이어를 재심판한 결과, `_regime_cash_scale`은 Sharpe **+14%**(0.752→0.858)로
확인(20차 §9.4의 "효과가 작다"는 결론은 gross 미통제 비교였음 — 정정), `_pnl_vol_target_scale`은 두 짝짓기 모두에서
Sharpe를 깎았다(0.752→0.733, 0.858→0.841). → `pnl_vol_target` 플래그 신설(**기본값 `True` 유지, 무회귀**), 기본값
전환은 사전등록된 fold-train-only 기준(`mhs_execution_friction_and_exposure_layers.md` §6.1) 통과 후에만.

### 10.5 P2/P4 — 실현 집행비용·로스터 계약 노출

| field | 값 |
| :--- | ---: |
| `realized_execution_roster_size` | **41.93** (선언값 `execution_universe_size=30` 대비 +40%, exit multiplier 2.0의 히스테리시스 효과) |
| `slow_momentum.primary_realized_shortfall_bps` | 10.70 (모델 8.0bps 대비 초과) |
| `slow_momentum.stress_realized_shortfall_bps` | 26.71 |
| `fast_reversal.primary_realized_shortfall_bps` | 1.70 |

**⚠️ 신규 관찰(미해결, 다음 회귀에서 재확인 필요)**: `primary_fill_count=0`(모든 책) — 버그 아님, 기존 코드가
`fill_count`를 `OHLCV_STRICT_PROXY`/`LADDERED` 전용 지정가·타임아웃 분기 카운터로 정의해 왔고, `primary`가 쓰는
`OHLCV_IMMEDIATE_TAKER` 경로는 항상 `reason="timeout_taker"`로 떨어져 구조적으로 0이 나온다(`execution.py:1415` 부근).
실제 체결은 `slow_momentum.primary.fills.row_count=74,781`로 정상 확인됨 — 계약 요구("not None")는 충족했지만
필드명이 오독을 유발한다. 또한 `primary_notional_weighted_shortfall_bps=-46.6bps`가 단순평균(+10.7bps)과
**부호가 반대**로 나왔다 — 공식(`sum(shortfall·notional)/sum(notional)`) 자체는 검증된 순수함수이나, 소수의
초대형 명목가 체결이 평균을 강하게 끌어당기고 있을 가능성이 있다. 두 관찰 모두 원인 규명 없이 사실만 기록한다.

### 10.6 회귀 불변식 확인

`slow_momentum.primary_autocorr_sharpe=0.525673922813482`, 3개 fold 전부 20차 리포트와 **소수점까지 바이트 동일**.
`pnl_vol_target` 기본값(`True`)이 기존 파이프라인을 정확히 재현한다는 계약이 실측으로 검증됐다 — 이번 21차는
계측·게이트만 바꿨을 뿐 전략 로직/자본 배분은 무변경.

### 10.7 요약

| 항목 | 상태 |
| :--- | :--- |
| RC-1. 참조책/집행책 이중 계측 | ✅ 완료, 실측 확인 (`executed_prescreen`이 blend net_t 0.779→1.729로 노출) |
| RC-5. Research-GO 게이트 도달 가능화 | ✅ 완료, 임계값 미등록으로 `eligible` 여전히 False(의도됨) |
| RC-6. 다중검정 회계(DSR) | ✅ 완료, `deflated_sharpe_ratio=0.532` 최초 노출 |
| P1. 로스터 재정규화 회전율 가설 | ❌ **실측 기각**(-1%), 재설계 미착수 |
| P1. `_regime_cash_scale` 재평가 | ✅ 동일 gross에서 +14% 확인 — 20차 결론 정정, 유지 |
| P1. `_pnl_vol_target_scale` | ⚠️ 동일 gross에서 Sharpe 악화 확인, 플래그화(기본값 불변), 전환은 후속 fold 검증 대기 |
| P2. 실현 집행비용 노출 | ✅ 완료, `fill_count` 필드명 오독 이슈 발견(미해결, 후속 확인 필요) |
| P4. 로스터 계약 노출 | ✅ `realized_execution_roster_size=41.93` 노출 완료 |
| P3. 20차까지 참조책 기준 결론 재판정 | 미착수 — 후속 실행 대상 (`fast_reversal` 0% 배분은 §10.1에서 재확인됨) |
