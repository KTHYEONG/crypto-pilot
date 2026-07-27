## L2 게이트 정직화·리스크예산 스펙 구현 및 실측 — 진짜 첫 정직한 판정(FAIL) — 2026-07-27

- 실행일: `2026-07-27`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 검증 창: 2024-01-03 ~ 2026-07-01 (워밍업 90일 / L1 365일 / L2 365일 / L3 봉인 홀드아웃 90일)
- 데이터 축: **51개 CORE 완전 이력 심볼 × 5,460개 4h봉**
- 스펙: `docs/specs/l2-gate-honesty-and-risk-budget.md`
- exit_code: **0**
- 산출물: `logs/futures/compound/20260727_013707/`

### 배경 — "PASS"가 결함의 산물이었다

직전 항목(`20260726_142427`)은 L2 PASS(CAGR 31.05%, Sharpe 1.352, DSR 1.000000)를 기록했다. 사용자가 "excess_growth_probability(0.9400)·sharpe_probability(0.9405)가 임계값 0.90 대비 4%p 마진에 불과하다"는 의심을 제기했고, 실측 감사 결과 **마진이 얇은 게 아니라 PASS 자체가 3개 결함의 산물**임이 확인됐다.

`scratch/verify_l2_reconstruct.py`(재구성, turnover 6.634 정확 일치·equity_multiple 오차 0.03%)와 `scratch/verify_l2_gate_honesty.py`(E1~E8 다중가설 실험)로 확정한 결함:

| ID | 결함 | 실측 근거 |
|---|---|---|
| A-1 | DSR 주기 불일치 | 연율 Sharpe에 `√(N_daily−1)`를 곱해 DSR=1.000000이 포화 artifact. 주기정합 시 0.29~0.88 |
| A-2 | 확률 게이트 중복 | `excess_growth_probability`와 `sharpe_probability`는 동일 통계량(P(mean log-excess>0))의 draw 수만 다른 재현 — max\|diff\|=0.00e+00 |
| A-3 | 블록길이 오지정 | Politis-White 자동추정 22.0일 vs 하드코딩 5일. 정정 시 0.9420→0.8985로 문제의 4%p 마진을 정확히 설명 |
| A-4 | 벤치마크 일자 오정렬 | 전략 D일 수익이 벤치마크 D+1일 수익과 대응(corr −0.09→정렬 후 +0.22) |
| A-5 | funding carry 부호 역전 | allocator `+sign(mu)·fr` vs simulator `−Σw·fr` |
| A-6 | 종목별 비용 무력화 | `dense_simulator`가 `np.mean(cost_bps[t])`로 스칼라화, slippage/impact 하드 0 |
| A-7 | L3 holdout이 PROMOTE를 못 막음 | posterior 가중 구조상 P_prior≈1이면 홀드아웃 성장확률 0에서도 PROMOTE 가능 |
| A-8 | 실행 북 사실상 동결 | `target_weights.npy` 실측: 심볼당 동일가중 연속유지 중앙값 1,214봉(202일), 38/51 심볼 >1,000봉 |
| A-9 | 절대 CAGR 연율화 버그 | `log1p∘expm1` 상쇄로 산술평균 연율화(31.05%) vs 실제 복리(30.42%), +0.67pp 과대계상 |

반증된 초기 가설(정직하게 기록): ① "동결 북이 31% CAGR의 원천" — frozen 대조군 실측 −4.85% CAGR, live−frozen t=1.974로 **틀렸음** 확인. ② "비순환 bootstrap이 CI를 낙관화" — H0 하 size@0.90=0.0000으로 **틀렸음** 확인, 실제 원인은 블록길이 오지정(A-3).

### 설계 결정 — Phase 순서 강제

임계값 완화 0건 원칙 하에 4단계로 설계(`docs/specs/l2-gate-honesty-and-risk-budget.md`):

- **P1(측정계 정직화)**: DSR 주기정합, `sharpe_probability`를 게이트에서 제외하고 SPA(3-대조군: benchmark/cash/frozen-book) 신설, circular+Politis-White bootstrap, 벤치마크 일자정렬, 복리 연율화 분리.
- **P2(실행층 결함제거)**: 신호 소멸 시 support 강제청산, 심볼별 deadband, funding 부호 정정, 종목별 실제 비용 배선.
- **P3(리스크예산 회수, 폐루프 vol targeting)**: **P1·P2 완료 후로 강제** — 깨진 측정계에 대고 레버리지를 올리면 curve-fitting이 되기 때문.
- **P4(L3 holdout veto)**: 홀드아웃 성장확률이 낮으면 prior가 아무리 좋아도 PROMOTE 불가하도록 필요조건 추가.

### 구현 및 검증 (`/implement` → `/check`)

- 신규: `src/domain/futures/compound/bootstrap.py` (Politis-White 블록길이, circular bootstrap, SPA p-value)
- 수정: `multiplicity.py`(DSR 주기정합), `validation.py`(정렬·복리연율화·SPA 편입·frozen control), `allocator.py`(support 재적용·심볼별 band·carry 부호·폐루프 vol), `dense_simulator.py`(종목별 비용·slippage/impact), `engine.py`(frozen control 배선·L3 prior 일봉화·`window=None` 가드), `compound_main.py`(`l2_gate_inputs.npz` 영속화)
- **`/check` 1차 PASS 후 실전 실행에서 신규 결함 1건 추가 발견**: L3 prior 가드가 `len(daily_prior) > l2_prior_effective_days_cap(=60)`이면 무조건 `ValueError`를 던지도록 구현됐는데, 이 결함은 **내가 작성한 스펙 계약(contract.json) 자체의 오류**였다 — "60일 상한"을 "존재 가능한 최대 일수"로 잘못 정의해, L2 창이 실제로는 365일이므로 **모든 정상 실행에서 무조건 크래시**하는 구조였다(`ValueError: L3 prior returns length 365 exceeds daily-expected cap 60`). Mock 기반 유닛테스트(`test_l3_prior_length_check_raises`, 인위적으로 짧은 시나리오만 검증)가 이 결함을 못 잡았다. 가드를 제거(최근 60일 슬라이스 로직은 이미 정상 동작)하고 테스트를 실제 동작 검증(`test_l3_prior_slices_to_most_recent_cap_days`)으로 교체 → contract.json도 정정 → 재검증 PASS.
- 최종 판정: 🟢 PASS — Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ | Coverage 88%

### 실제 CLI 재실행 결과 — 이전(오염된 PASS) vs 신규(정직한 FAIL)

| 지표 | 이전 `20260726_142427` (PASS) | 신규 `20260727_013707` (**FAIL**) |
|---|---:|---:|
| **verdict** | PASS | **FAIL** |
| absolute CAGR | 31.05%(산술 연율화 오류 포함) | **6.75%**(복리, 정합) |
| Sharpe | 1.352 | 0.317 |
| sharpe_probability | 0.9405 | 0.724 |
| deflated_sharpe_probability | 1.000000(포화 artifact) | **0.4266** |
| excess_growth_probability | 0.9400 | **0.712** |
| excess_growth_lcb90 | +0.0527 | **−0.0883** |
| stressed_excess_growth_lcb90 | +0.1284 | **−0.0737** |
| spa_pvalue (신규 게이트) | — | **0.362**(기준 ≤0.10) |
| max drawdown | 4.23% | 10.68% |
| annual volatility | 10.06% | 12.11% |
| annual turnover | 6.63x | 7.13x |
| cost drag ratio | 2.74% | **10.17%** |
| capacity utilisation p95 | 5.62% | 2.58% |
| integrity | `true` | `true` |
| L3 verdict | shadow(미소진, `dry_run_holdout_not_consumed`) | **reject**(`l2_not_pass`) |

내부 정합성: `l2_gate_inputs.npz`에 영속화된 일별 계열로부터 excess Sharpe(0.31741345 vs 보고 0.3174), 복리 절대 CAGR(0.067475 vs 보고 0.0675), MDD(0.106809 vs 보고 0.1068)를 독립 재계산해 정확히 일치함을 확인 — 보고 수치가 실제 산출 파이프라인과 정합함을 검증했다.

### 결과 해석 — 왜 이렇게 크게 바뀌었나

1. **통계 게이트 3종이 독립적으로 동일 결론(엣지 없음)에 도달**했다: DSR 1.000000→0.4266(스펙 예측 σ_SR≈1.0 시나리오 0.4282와 거의 일치), excess_growth_probability 0.9400→0.712, 신규 SPA p-value=0.362(기준 0.10 대비 3.6배 초과). 세 검정이 서로 다른 귀무가설에서 같은 결론에 도달했다는 것이 결함이 아니라 진짜 신호 부재를 가리키는 강한 증거.
2. **CAGR 붕괴(31%→6.75%)는 산술→복리 정정(예상 −0.6pp)만으로 설명되지 않는 크기**다. 실제 원인은 실행층 결함 4건 동시 수정: 신호 소멸 종목 방치 포지션 강제청산(A-8) + 종목별 실제 비용 배선(A-6, cost drag 2.74%→10.17% 급증 — 비유동 알트코인 비용이 이제 반영됨) + funding 부호 정정(A-5) + deadband 위치 이동(A-4). 즉 이전 31%는 "동결 포지션 + 저평가 알트코인 비용 + 펀딩 부호 오류"가 겹친 **복합 착시**였다.
3. **MDD·변동성은 오히려 악화**(4.2%→10.7%, 10.1%→12.1%)했다. 스펙 설계 시 "리스크 예산에 여유가 있다"고 판단한 근거는 오염된(옛) 실행 경로 기준이었다. 정직화된 실행 경로에서는 실제 비용·청산 로직이 리스크를 더 많이 소모한다 — **P3(성장 레버 확대)를 지금 적용해서는 안 된다**는 스펙의 순서 강제(P3는 P1·P2 이후)가 실측으로 정당화됐다.
4. **L3 holdout veto가 설계대로 작동**했다. 이전엔 `L2_DRY_RUN=1`이라 봉인 홀드아웃이 아예 소비되지 않고 `shadow`로 방치됐으나, 이번엔 L2가 FAIL하자마자 P4 로직이 즉시 `reject`(`l2_not_pass`)로 응답해 잘못된 PASS가 배포 후보로 이어지는 경로를 차단했다.

### 부수 발견 (범위 외, 별도 확인 필요)

`logs/futures/compound/`에 `n_bars=2~3`짜리 더미값 결과가 다수 섞여 있다(`tests/unit/application/futures/runner/test_compound_main.py`의 일부 mock 테스트가 `tmp_path`를 쓰지 않고 실제 프로덕션 로그 경로에 값을 씀 — 이번 세션 이전부터 존재). 재감사 시 실제 실행과 테스트 산출물을 혼동할 위험이 있어 별도 수정이 필요하다.

### 최종 판정

- 정직화 이전 "PASS, CAGR 31%"는 **버그(DSR 포화·게이트 중복·블록길이 오지정·일자 오정렬·동결 포지션·비용 무력화)가 만든 숫자**였다.
- 정직화 이후 "FAIL, CAGR 6.75%"가 이 전략의 **현재 진짜 실력**이다. 실패가 아니라 이번 작업의 목표(측정계 정직화)가 실측으로 검증된 결과.
- A-8(PIT 유니버스 51종목 생존편향·breadth 확장)은 이번 스펙 범위 밖(`[LIMIT-08]`) — 후속 스펙에서 다룬다. breadth 확장 없이는 통계적 증거가 늘지 않는다.
- **여전히 실전 매매에 사용하지 않는다.** `L2_DRY_RUN=0`은 P4(L3 veto) 검증이 실측 완료된 지금도 사용자 별도 결정 사항이며, 이번 실행에서 전환하지 않았다.
- 다음 검토 시점: A-8 PIT 유니버스 스펙 착수 여부, 그리고 `logs/futures/compound/` 테스트 오염 정리 여부.
