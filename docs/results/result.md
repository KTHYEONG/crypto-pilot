## L1 측정계 정직화(Driscoll-Kraay HAC + 단일 포트폴리오 게이트 + block bootstrap) 구현 및 신호별 DEBUG 실측 — 2026-07-28

- 실행일: `2026-07-28`
- 실행 명령: `L1_DEBUG=1 L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15`
- 검증 규모: 51개 CORE 심볼 × 5,442개 4h봉, 신호 27종(family 7개 × speed 5단), fold 5개 × cluster ~4개 = sleeve 후보 455개
- 스펙: `docs/specs/l1-measurement-integrity-restore.md` (+ contract.json)
- 산출물: `logs/futures/compound/20260728_102005/`, `logs/l1_admission.jsonl`(455 ALGO + 1 EVAL row)
- `/check` PASS: Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ (145 passed) | Coverage 92%
- exit_code 관련 `l2.verdict`: **no_evidence** (현금 100% 보존, `rebalances=0`)

### 배경 — 어제 복원한 fail-closed 게이트의 판정 근거 자체가 결함

전날(`CAPITAL_DEPLOYMENT_FAILCLOSED_RESTORE`) admission 게이트(`has_admitted`)를 복원했으나, 그 게이트가 참조하는 집계 통계량 자체를 실측(`scratch/verify_l1_gate_signal_independence.py`, `scratch/verify_l1_gate_se_validity.py`)으로 검증한 결과 3건의 결함을 확정:

| ID | 결함 | 실측 근거 |
|---|---|---|
| D-1 | 집계 게이트 계열이 매매 신호와 무관 — 클러스터 구성원 등가중 롱온리 바스켓 수익률만 측정 | 부호 반대 신호 2개 투입 시 계열이 비트단위로 완전 동일(`max diff=0.0`) |
| D-2 | sleeve가 (fold×cluster)로만 정의되어 540개 "독립" sleeve가 실제로는 고유 계열 4개(135배 중복) | 실측 검증 스크립트 결과 |
| D-3 | pooled OLS 표준오차가 지속성 있는 실제 신호(EWM 평활)에서 41배 과신(P(p>0.99) 명목1%→40.8%) | 지속성 널 400회, KS p=3.7e-59 |
| D-5 | 이전 세션 성능 최적화 커밋(`f2f6d7aa`)이 Politis-White block bootstrap을 i.i.d.로 조용히 되돌림(3회차 동일 패턴) | `git log -S` 확인 |

### 구현 (`/implement` → `/check`)

- `l1_sleeves.py`: `compute_l1_oos_portfolio_returns` 신규(OOS 구간만 stitch, 리스크패리티 사이징, 비용 차감) → `build_exit_aware_handoff`가 단일 포트폴리오 시계열 1개만 `circular_stationary_bootstrap_growth`(block bootstrap)에 투입. `_cluster_masked_beta`를 Driscoll-Kraay HAC SE로 교체(횡단면 집계 + Bartlett 커널 Newey-West).
- `config.py`: `HandoffConfig.min_sleeve_posterior_probability` 0.52→**0.95**(하드코딩 리터럴 제거), `hac_lag_cap=120` 신규.
- `l1_diagnostics.py`(신규): `L1AdmissionRecorder` — `L1_DEBUG=1`일 때만 `logs/l1_admission.jsonl`에 `[ALGO]`(sleeve별) / `[EVAL]`(게이트 1회) 기록, 비활성 시 완전 무비용(logging.md 토큰경제 준수).
- 후속 수정: `pw_block` 필드가 `0.0` 하드코딩 스텁이었던 것을 실제 `politis_white_block_length` 계산값 배선으로 교정(회귀 테스트 `test_build_exit_aware_handoff_records_resolved_pw_block` 추가). `estimate_cluster_sleeve_posteriors`에 `recorder.record_sleeve` 배선 추가 — 이전엔 클래스만 존재하고 프로덕션에서 호출된 적이 없어 sleeve별 데이터가 전혀 수집되지 않고 있었음.
- `/check` PASS: Mypy Strict ✅ | Regression Test ✅ (145 passed, `test_config.py`의 무관한 사전 결함 1건은 베이스라인에서도 동일 재현 확인 후 범위 제외)

### 실전 CLI 실행 결과 — 정직화 이후 다시 NO_EVIDENCE

| 지표 | 값 |
|---|---:|
| `l2.verdict` | **no_evidence** |
| `active_days_ratio` | 0.0 |
| `rebalances` | 0 |
| `l3.verdict` | reject (`l2_not_pass`, `low_growth_probability`) |

집계 게이트 내부 실측(`[EVAL]`, `logs/l1_admission.jsonl`):

| 지표 | 값 |
|---|---:|
| `admitted_sleeves` | 15 / 455 후보 (임계값 0.52→0.95 상향 효과) |
| `distinct_series` | **1** (D-2 회귀 없음 확인) |
| `oos_bars` | 1,700 |
| `ann_growth`(평균) | +18.14% |
| `ann_lcb90`(block bootstrap 하한) | **−78.87%** → `admitted=False` |
| `pw_block`(Politis-White, 정상 배선 확인) | 2.89 |
| `turnover`(내부 진단용 프록시) | 446.87x |

**해석**: 평균 성장률(+18.1%)만 보면 양호해 보이나, 살아남은 sleeve가 15개뿐이고(distinct_series=1이라 진짜 독립정보량은 이보다도 적음) 회전율이 극단적으로 높아 신뢰구간 하한이 −78.9%로 폭락 — "운이 좋으면 벌 수도 있으나 확신할 근거가 없다"는 정직한 판정. 임계값을 완화하지 않고 그대로 기록한다.

### 신호(Family)별 DEBUG 실측 — L1 signal 상세 진단용 데이터 (455 sleeve 후보 전수)

| family | n | admit | admit% | mean\|beta\| | mean_prob | mean_se_ratio(HAC/OLS) | mean_n_blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| **momentum_ts** | 95 | **15** | **15.8%** | 0.0762 | 0.657 | **7.84** | 85.2 |
| trend_ema | 95 | 0 | 0.0% | 0.0037 | 0.518 | 68.41 | 85.2 |
| breakout_donchian | 95 | 0 | 0.0% | 0.0208 | 0.610 | 26.21 | 85.2 |
| basis_gap | 75 | 0 | 0.0% | 0.0109 | 0.564 | 39.63 | 95.0 |
| xs_momentum_slow | 38 | 0 | 0.0% | 0.0087 | 0.509 | 37.85 | 20.9 |
| xs_reversal | 38 | 0 | 0.0% | 0.0015 | 0.487 | 19.18 | 170.8 |
| reversal_st | 19 | 0 | 0.0% | 0.0015 | 0.462 | 17.90 | 256.6 |

speed 세분(momentum_ts만 admit 발생, 상위 4개):

| signal_id | n | admit | mean_beta | mean_prob | mean_se_ratio | mean_n_blocks |
|---|---:|---:|---:|---:|---:|---:|
| momentum_ts:medium | 19 | 6 | 0.0267 | 0.841 | 7.15 | 85.0 |
| momentum_ts:very_slow | 19 | 6 | **0.2265** | 0.687 | 8.15 | **13.8** |
| momentum_ts:moderate | 19 | 2 | 0.0172 | 0.535 | 8.07 | 42.3 |
| momentum_ts:slow | 19 | 1 | 0.0389 | 0.662 | 8.27 | 28.1 |

### 데이터 기반 관찰 (L1 signal 개선 착수점)

1. **momentum_ts family가 유일한 생존 신호**다 — 다른 6개 family 전부 admit=0%. beta 절대값(0.076)과 SE 팽창 배수(7.84x)가 전 family 중 가장 양호(다른 family는 17.9~68.4x로 3~9배 더 심하게 팽창) — 즉 이 신호만 실제로 자기상관 구조 대비 유효한 예측력을 갖고 있을 가능성이 상대적으로 높다.
2. **trend_ema는 사실상 사멸 신호**다 — beta가 전 speed에서 0.0001~0.0087로 0에 수렴하는데, SE 팽창 배수는 오히려 속도가 느릴수록 급증(fast 20.98x → very_slow 98.33x). "느린 추세 신호일수록 지속성만 크고 예측력은 없다"는 전형적 과최적화 위험 신호.
3. **momentum_ts:very_slow의 beta=0.2265**는 다른 모든 signal×speed 조합(대부분 0.001~0.07)보다 한 자릿수 크나, `n_blocks=13.8`로 독립 표본이 매우 적다 — 진짜 엣지인지 소표본 아티팩트인지 구분 불가. 과신 금지, 후속 검증 필요.
4. **basis_gap·xs_reversal**(2026-07-24 horizon term structure 연구에서 "완전 독립" 신호로 확인된 바 있음)이 이번 클러스터-베타 측정에서는 admit=0%로 나타남 — 상충이 아니라 **측정 방법이 다름**(당시는 신호간 상관구조, 이번은 클러스터 내 pooled beta)에 유의. 두 결과를 동일 선상에서 비교하려면 별도 스펙 필요.
5. `n_blocks`는 예상대로 speed에 반비례(fast 256.6개 ↔ very_slow 13.8개) — 느린 신호일수록 admission 판정의 통계적 신뢰도 자체가 구조적으로 낮아진다는 것을 재확인.

### 최종 판정

- L1 게이트가 이제 **실제 배포될 북(포지션·비용 반영)의 신호의존적 수익률**을 측정하며, 표본 부풀리기·i.i.d. 부트스트랩 회귀가 제거됨을 코드·테스트로 확정.
- 결과는 여전히 NO_EVIDENCE — 이는 실패가 아니라 "지금까지 momentum_ts 외 신호는 통계적으로 유의한 엣지가 없다"는 정직한 확인.
- 신규 확보된 sleeve별 DEBUG 데이터(`logs/l1_admission.jsonl`)로 momentum_ts 집중 강화 및 trend_ema 계열 폐기/재설계가 다음 스펙의 데이터 기반 후보로 확정됨.
- Exit Policy(소비자 0개) 결함은 이번 스펙에서 사용자 결정으로 보류 유지.
