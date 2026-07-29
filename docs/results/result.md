## R-4 실전 CLI 재검증 — 증거창 누적 로직 수정 확인, no_evidence는 데이터 부족으로 재분류 — 2026-07-29

- 실행일(KST): `2026-07-29`
- 실행 명령: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42`
- 프로세스: `exit_code=0`, `integrity_ok=true` (크래시 없음)
- 결과 artifact: `logs/futures/compound/20260729_030134/`
- 라우터 attribution 재검증: `scratch/reverify_r4_attribution.py`(`_build_prequential_expert_route_impl` 논인베이시브 패치, `logs/scratch/r4_reverify_attribution.json`)
- 검증 규모: `51` symbols × `5,442` 1h bars(내부 L1 4h), `L2_DRY_RUN=1`, sealed L3 holdout 미소비

### 배경 — 선행 스펙(`l1-signal-evaluation-architecture-fix`) 적용 후 정직한 실전 확인

직전 스펙에서 P0(증거창 누적 로직), P1(sleeve OOS 확증 AND-게이트), P2(regime 하드게이트→overlay 강등)를 구현하고 `/check` PASS(Mypy strict, Cov 94%)까지 마쳤으나, `/check` 자체는 mypy/wiring/pytest만 검증하고 "실제로 프로덕션 데이터에서 게이트가 도달 가능해졌는지"는 증명하지 않는다. 이번 실행은 그 실전 확인이다.

### 결과 데이터

| 항목 | 값 |
|---|---:|
| `l2.verdict` | `no_evidence`(불변) |
| `l2.integrity_ok` | **true** |
| `l2.reasons` | `active_days_ratio=0.0000<0.1`, `rebalances=0<30` |
| `l3.verdict` | `reject` |
| `admitted_sleeves`(L1, `logs/l1_admission.jsonl` EVAL) | **5** (직전 세션 15 → P1 OOS 확증 게이트로 정직하게 축소) |
| target weights 비영 비율 | `0.0%` |

### R-4 수정의 직접 증거 — fold 3 `momentum_ts:very_slow`

```json
{"fold": 3, "signal_id": "momentum_ts:very_slow", "n_evidence_bars": 0, "reasons": ["insufficient_evidence_window"]}
```

이 신호가 라우터에 **처음** 등장하는 시점(fold 1~2엔 L1 admission 자체가 없었음)에 `n_evidence_bars=0`이 기록됐다. 회귀 버그(2026-07-29 앞선 실행, "P0 ExpertReturnTape" 절 참조) 상태였다면 이 값은 fold 자신의 OOS 폭(340)으로 나왔을 것이다 — 실측 0은 "누적 이력이 아직 없다"는 정직한 반영이며 누적 로직이 실제로 작동함을 직접 증명한다. fold 4에서 같은 신호는 `n_evidence_bars=340`(fold3 1개 record 누적)으로 정확히 이어졌다.

### 잔여 no_evidence의 재해석

fold 4 시점 누적치(340) < `min_evidence_bars=900`으로 게이트 미통과가 지속되나, 이는 게이트 버그가 아니라 **이 신호가 5-fold 창의 후반(fold 3)에야 처음 admit돼 누적 시간이 구조적으로 부족**한 정직한 데이터 부족 사유다(`[LIMIT-01]`/`[LIMIT-11]` 기대대로). `admitted_sleeves`가 15→5로 준 것도 P1의 sleeve OOS 확증 게이트가 실제로 과적합 후보를 걸러내고 있다는 정합적 증거다.

### 판정

1. R-1(크래시)·R-4(증거창 누적)는 실전 CLI로 재확인 완료 — `exit_code=0`, `integrity_ok=true`, `n_evidence_bars`가 신호 첫 등장 시 0으로 정직하게 기록됨.
2. P1(sleeve OOS AND-게이트) 실전 효과 확인 — admitted sleeves 15→5.
3. `no_evidence`는 이번 실행에서도 유지되나 원인이 "게이트 계산 버그"에서 "R-6(엣지 부재) + 신호별 fold 내 등장 시점에 따른 누적 이력 부족"으로 완전히 재분류됨. 임계값 완화 없이 기록.
4. 다음 착수점: R-6(엣지 자체 부재, 선행 세션 실측 pooled rank IC t=0.31) 재설계, 혹은 신호가 fold 0~1부터 꾸준히 admit되도록 L1 sleeve 안정성 자체를 다루는 후속 스펙.

- 결과 원본: `logs/futures/compound/20260729_030134/result.json`, `logs/scratch/r4_reverify_attribution.json`, `logs/l1_admission.jsonl`

---

## 최신 DEBUG 실행 — P2 ExpertReturnTape 항등식 예외로 cash-only fallback — 2026-07-29

- 실행일(KST): `2026-07-29`
- 실행 명령: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42`
- 프로세스: `exit_code=1` (P2 예외가 `integrity_ok=false`로 전파됨)
- 결과 artifact: `logs/futures/compound/20260728_232739/`
- DEBUG 원본: `logs/futures/compound/debug_20260729.log` (25 lines, 1,808 bytes)
- 검증 규모: `51` symbols × `5,442` 1h bars (내부 L1 4h), `L2_DRY_RUN=1`, sealed L3 holdout 미소비

### 결과 데이터

| 항목 | 값 |
|---|---:|
| `l2.verdict` | `no_evidence` |
| `l2.integrity_ok` | **false** |
| `l2.reasons` | `p2_pipeline_error:ValueError` |
| `l3.verdict` | `reject` |
| `l3.reasons` | `p2_pipeline_error:ValueError`, `l2_not_pass` |
| CAGR / annualized log growth | `0.00% / 0.00%` |
| Sharpe / MDD / volatility | `0.00 / 0.00% / 0.00%` |
| equity multiple | `1.00x` |
| target weights shape | `(5442, 51)` |
| non-zero target weight ratio | `0.0000%` |
| max `abs(target_weight)` | `0.0` |

### DEBUG 예외 trace 및 원인 후보

```text
P2 pipeline failed, using cash-only fallback
... l1_regime_routing.py:276 -> ExpertReturnTape(...)
contracts.py:801: ValueError: net != gross + execution_cost + funding
```

- 실패 지점: `build_fold_local_shadow_tape()`가 `ExpertReturnTape`를 생성하는 시점.
- 계약: `net_return_1d == gross_return_1d + execution_cost_return_1d + funding_return_1d` (`np.isclose` 검증).
- 이번 실행은 전략 성과가 음수라서 거부된 것이 아니라 **수익 분해 배열의 수치/정렬 불일치로 L1 라우팅 자체가 중단**된 결과다.
- 따라서 이번 `no_evidence`는 정상적인 통계적 현금 대기가 아니라 `integrity_failure` 성격의 보호 동작으로 분류해야 한다.
- 동반 DEBUG 경고: `smart_money_divergence`에 `top_trader_long_short_ratio/long_short_ratio` 필드가 없어 해당 데이터가 fallback 처리됨(2회).
- 추가 런타임 경고: `numpy` 상관계수 계산 중 zero-variance 입력으로 `invalid value encountered in divide` 발생(2건).

### 다음 분석용 원자료

- 전체 stdout/stderr: `logs/futures/compound/debug_20260729.log`
- 구조화 결과: `logs/futures/compound/20260728_232739/result.json`
- 현금 결과 weights: `logs/futures/compound/20260728_232739/target_weights.npy`
- L1 admission 누적 JSONL: `logs/l1_admission.jsonl` (실행 후 총 `2,073` lines; 이번 DEBUG 로그에는 REGIME tape 완료 행이 없음)
- 우선 점검: gross/cost/funding/net 배열의 동일 인덱스 정렬, float32→float64 변환 시점, `NaN/inf` 전파, 마지막 bar의 `prev_pos`/funding slice 경계.

---

## 최신 실행 — causal regime expert routing 적용 후 fail-closed 현금 상태 — 2026-07-28

- 실행일: `2026-07-28`
- 실행 명령: `L2_DRY_RUN=1 L1_DEBUG=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42`
- 실행 산출물: `logs/futures/compound/20260728_131507/`
- 검증 규모: 51개 심볼 × 5,442개 1h 기준봉(내부 L1 4h), L2 dry-run, 봉인 L3 holdout 미소비
- 실행 상태: 데이터 준비·신호 계산 완료 후 `l2_gate_inputs`가 빈 상태로 `no_evidence`; 프로세스 `exit_code=1`은 성과 게이트 실패 반환

### 성과

| 지표 | 최신 결과 |
|---|---:|
| `l2.verdict` | **no_evidence** |
| `l3.verdict` | **reject** |
| 연환산 log growth / CAGR | 0.00% |
| Sharpe | 0.00 |
| MDD | 0.00% |
| 연 변동성 | 0.00% |
| 연 turnover | 0.00 |
| cost drag | 0.00% |
| equity multiple | 1.00x |
| active days ratio | 0.0000 |
| rebalances | 0 |
| fold 성장률 | `[0.00%, 0.00%, 0.00%, 0.00%, 0.00%]` |

### 해석

- `admitted_sleeves=15` 후보는 존재했지만 causal regime/expert routing 후 실제 배포 `weights_2d`가 전부 0이 됐다.
- `target_weights.npy` shape은 `(5442, 51)`, 비영 weight 비율은 `0.0%`; 따라서 이번 실행은 손실도 수익도 발생시키지 않은 현금 보존 결과다.
- L2 거부 사유는 `active_days_ratio=0.0000<0.1`, `rebalances=0<30`; L3는 `low_growth_probability`, `l2_not_pass`로 거부됐다.
- 직전 활성 북 실행(`20260728_112010`)의 `ann_growth=-8.01%`, `ann_lcb90=-24.75%`, `positive_folds=2/5`와 달리, 이번 결과는 성과 개선을 의미하지 않는다. 새 라우터가 증거 부족 시 거래를 차단한 안전 상태다.
- 현재 `logs/l1_admission.jsonl`에는 최종 EVAL은 기록되지만 regime-level rejection 원인이 충분히 분해되지 않는다. 다음 실행 전에는 expert×regime별 `effective_blocks`, `growth_lcb90`, `growth_2x_cost`, `positive_inner_folds`, `scale` 계측을 확인해야 한다.

- 결과 원본: `logs/futures/compound/20260728_131507/result.json`, `target_weights.npy`

---

## L1 포지션 구성(Construction) 정직화 — 게이트/배포 북 일치, 순노출 캡, fold 일관성 게이트 활성화 — 2026-07-28

- 실행일: `2026-07-28`
- 실행 명령: `L1_DEBUG=1 L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15`
- 검증 규모: 51개 CORE 심볼 × 5,442개 4h봉, 신호 27종(family 7개 × speed 5단), fold 5개 × cluster ~4개 = sleeve 후보 455개, admitted sleeve 15개(멤버 27종목)
- 스펙: `docs/specs/l1-position-construction-integrity.md` (+ contract.json) — 직전 스펙 `l1-measurement-integrity-restore.md` 의 후속
- 산출물: `logs/futures/compound/20260728_112010/`, `logs/l1_admission.jsonl`(455 ALGO + 1 EVAL row), `logs/scratch/l1_gate_inputs.npz`(재현용 게이트 입력 덤프)
- `/check` PASS: Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ (216 passed) | Coverage 91%
- `l2.verdict`: **no_evidence** (현금 100% 보존, `rebalances=0`, `active_days_ratio=0.0`)
- `l3.verdict`: reject (`l2_not_pass`, `low_growth_probability`)

---

### 배경 — 직전 세션(L1 측정계 정직화)의 결과가 통계적으로 유의미했는지 재검증하다가 구성 결함 발견

직전 스펙(`l1-measurement-integrity-restore`, D-1~D-5 수정: 신호 무관 게이트·135배 중복 표본·SE 41배 과신·i.i.d 부트스트랩 회귀 해결)을 적용한 뒤 실행한 phase full 결과가 `admitted_sleeves=15`, `ann_growth=+18.14%`, `ann_lcb90=−78.87%`, `turnover=446.87x`, `cost_drag=44.69%` 였다. 평균 수익은 양호해 보였으나 하한이 −79%로 폭락하는 것이 이상해 실제 게이트 입력(`logs/scratch/l1_gate_inputs.npz`)을 덤프해 신호(`mu_2d`)를 완전히 고정한 채 포지션 구성 방식만 바꿔가며 원인을 분해했다(`scratch/verify_l1_portfolio_variants.py`, `scratch/verify_net_cap_sweep.py`).

### 확정된 결함 (전부 실측, signal 품질과 무관)

| ID | 결함 | 실측 근거 |
|---|---|---|
| C-1 | **게이트가 허수아비 포트폴리오를 채점** — 실제 배포 allocator(`compute_dynamic_compounding_path`)는 스무딩(`alpha_smooth=0.08`)·데드밴드(`band_frac=0.60`)·vol타게팅(12%)을 모두 갖췄으나, 직전 스펙의 `compute_l1_oos_portfolio_returns`는 자체 리스크패리티 사이징만 쓰고 셋 다 없었음. 게이트가 실제 북이라면 결코 내지 않을 연 57.6% 비용 드래그 + 27.9% vol 드래그를 스스로 만들어 자신을 탈락시킴 | 구성 변형 A(원 게이트) turnover 576/y·vol 74.6% vs 실제 allocator 대응 변형(스무딩+밴드+voltarget) turnover 87/y·vol 15.0% |
| C-2 | **포트폴리오 분산의 85%가 의도치 않은 방향성(net exposure) 베팅** — 그로스 1.0 북의 순노출이 평균\|net\|=0.532, std=0.600으로 부호까지 뒤집으며 진동. 무조건부 베타(−0.19)로는 은폐되어 있었음(평균 net≈−0.08이라 상쇄) | `var(net×market)/var(portfolio_return)` = 85.0% (횡단면 중립화 시 0.0%로 소멸) |
| C-3 | `HandoffConfig.min_positive_outer_folds=4`가 정의만 되고 **어디서도 검사되지 않는 dead 파라미터** — fold 5개 중 1~2개 극단치가 전체 통과를 좌우할 수 있는 구조적 허점 | 전 소스 grep 결과 참조처 없음 |
| C-4 | OOS 0.776년 창에서 LCB90>0 요구 시 **레버리지 무관 최소 Sharpe ≈ 1.46** 필요(창 길이의 함수, 범위 밖) | `1.282/√0.776` |

### 실측 A/B — `|net|` 노출 캡 sweep (신호 mu_2d 완전 고정, 구성만 변경)

| `|net|` 캡 | 평균\|net\| | net growth | vol | Sharpe | turn/y | LCB90 | folds+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 없음(구 게이트) | 0.703 | 4.2% | 26.8% | 0.29 | 47 | −31.5% | 2/5 |
| 0.50 | 0.437 | 11.9% | 21.0% | 0.67 | 51 | −16.3% | 2/5 |
| 0.30 | 0.277 | 17.7% | 17.7% | 1.09 | 57 | −7.6% | 2/5 |
| 0.20 | 0.189 | 22.3% | 15.9% | 1.48 | 62 | −0.1% | 1/5 |
| **0.10(채택)** | 0.097 | 25.9% | 14.4% | 1.88 | 69 | +6.6% | 2/5 |
| 0.05 | 0.049 | 28.5% | 13.8% | 2.14 | 73 | +9.1% | 3/5 |
| 0.00(완전중립) | 0.000 | 28.7% | 13.3% | 2.22 | 76 | +8.7% | 3/5 |

관계가 **완전 단조** — 순노출을 줄일수록 예외 없이 Sharpe 개선. 현 신호군에 마켓타이밍 알파가 전무하다는 직접 증거. `0.10` 채택 근거는 LCB 부호가 아니라 **방향성 채널의 분산 기여를 ≤10%로 억제**하면서(완전중립 대비 Sharpe 손실 0.34에 불과) 장래 마켓타이밍 알파 실재 시를 위한 표현력을 구조적으로 남기기 위함.

**반증된 최초 가설(정직 기록)**: "무조건부 시장베타 제거가 이득의 원천" — 정렬 정정(±1봉 시프트 버그 수정) 후 베타는 −0.186→−0.065로만 변해 vol 60.7%→19.0% 축소를 설명하지 못함. 진짜 메커니즘은 부호가 뒤집히는 **시변 순노출**이며 무조건부 베타로는 원리적으로 탐지 불가.

### 구현 (`/implement` → `/check`)

- `allocator.py`: `apply_net_exposure_cap` 신규 — support 마스킹 직후, `dd_scale`/`vol_scale` 적용 이전에 순노출을 그로스 대비 `max_net_exposure` 비율로 클램프. 스케일 불변(레버리지 스케일링과 직교), `max_net_exposure=1.0`이면 완전 무연산(롤백 경로 보장).
- `config.py`: `DynamicCompoundingConfig.max_net_exposure=0.10` 신규.
- `l1_sleeves.py`: `compute_l1_oos_portfolio_returns` 시그니처를 `mu_2d`/`sigma_2d` 대신 **완성된 `weights_2d` 수취**로 교체(자체 포지션 구성 로직 완전 삭제). `compute_fold_growths` 신규(fold별 독립 성장률, 결측 fold는 skip). `build_exit_aware_handoff`가 `min_positive_outer_folds` 실제 검사(`admitted = ann_lcb90>0 AND positive_folds>=config.min_positive_outer_folds`).
- `engine.py`: `weights_2d`를 게이트 호출 **이전** 1회 계산해 핸드오프에 주입 — 게이트 채점과 실제 배포가 동일 배열을 공유(C-1 재발 구조적 차단).
- `l1_diagnostics.py`: `record_gate`에 `positive_folds`, `fold_growths`, `mean_abs_net` 필드 추가.
- `/check` PASS: Mypy Strict ✅ | Regression Test ✅ (216 passed, 무관한 사전 결함 `test_config.py::test_dynamic_compounding_config_default_band_and_smoothing`도 이번 구현 과정에서 함께 해소됨을 확인) | Coverage 91%

### 실전 CLI 재실행 결과 — 게이트가 실제 배포 북과 정렬되며 수치가 극적으로 정직해짐

| 지표 | 수정 전(구 게이트, 허수아비 북) | 수정 후(실제 배포 북 채점) |
|---|---:|---:|
| `admitted_sleeves` | 15 | 15 (신호 admission 로직 무변경, 기대대로 동일) |
| `distinct_series` | 1 | 1 |
| `oos_bars` | 1,700 | 1,700 |
| `ann_growth` | +18.14% | **−8.01%** |
| `ann_lcb90` | **−78.87%** | **−24.75%** (폭 3배 이상 축소) |
| `pw_block` | 2.89 | 2.61 |
| `turnover`(연) | 446.87x | **7.50x** (60배 감소) |
| `cost_drag` | 44.69% | **0.75%** |
| `positive_folds`(신규 계측) | 미검사 | **2 / 5** |
| `mean_abs_net`(신규 계측) | 미측정 | **0.062** (캡 0.10 이내 정상 작동 확인) |
| fold별 성장률(신규 계측) | — | `[+6.13%, −12.43%, −44.54%, +16.10%, −6.86%]` |
| `admitted` | False | **False** |
| `l2.verdict` | no_evidence | **no_evidence**(불변) |

**판정**: 구성 결함(C-1/C-2) 해소로 turnover·cost_drag·LCB90 폭 전부 큰 폭 개선됐으나, `positive_folds=2/5`로 신규 활성화된 fold 일관성 게이트(C-3, 임계치 4)를 통과하지 못해 admission 실패 유지. fold 성장률을 보면 5개 중 3개가 마이너스이고 fold 2(−44.5%)가 특히 크게 손실 — **fold 간 일관성(비정상성) 자체가 근본적으로 부족**함을 시사. 임계값을 낮추지 않고 그대로 기록한다.

### 신호(Family)별 DEBUG 실측 — 이번 실행에서도 재확인(455 sleeve 후보 전수, 포지션 구성 변경은 sleeve-level admission에 영향 없음을 확인)

| family | n | admit | admit% | mean\|beta\| | mean_prob | mean_se_ratio(HAC/OLS) | mean_n_blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| **momentum_ts** | 95 | **15** | **15.8%** | 0.0762 | 0.657 | **7.84** | 85.2 |
| trend_ema | 95 | 0 | 0.0% | 0.0037 | 0.518 | 68.41 | 85.2 |
| breakout_donchian | 95 | 0 | 0.0% | 0.0208 | 0.610 | 26.21 | 85.2 |
| basis_gap | 75 | 0 | 0.0% | 0.0109 | 0.564 | 39.63 | 95.0 |
| xs_momentum_slow | 38 | 0 | 0.0% | 0.0087 | 0.509 | 37.85 | 20.9 |
| xs_reversal | 38 | 0 | 0.0% | 0.0015 | 0.487 | 19.18 | 170.8 |
| reversal_st | 19 | 0 | 0.0% | 0.0015 | 0.462 | 17.90 | 256.6 |

speed 세분(momentum_ts만 admit 발생):

| signal_id | n | admit | mean_beta | mean_prob | mean_se_ratio | mean_n_blocks |
|---|---:|---:|---:|---:|---:|---:|
| momentum_ts:medium | 19 | 6 | 0.0267 | 0.841 | 7.15 | 85.0 |
| momentum_ts:very_slow | 19 | 6 | **0.2265** | 0.687 | 8.15 | **13.8**(소표본 주의) |
| momentum_ts:moderate | 19 | 2 | 0.0172 | 0.535 | 8.07 | 42.3 |
| momentum_ts:slow | 19 | 1 | 0.0389 | 0.662 | 8.27 | 28.1 |

### 다음 스펙 검토를 위한 상세 데이터 (재현 자료)

- **재현 스크립트**: `scratch/dump_l1_gate_inputs.py`(게이트 입력 npz 덤프, `logs/scratch/l1_gate_inputs.npz`), `scratch/verify_l1_portfolio_variants.py`(구성 변형 A~P + fold별 안정성 비교), `scratch/verify_net_cap_sweep.py`(`|net|` 캡 sweep, 위 표의 원본).
- **fold별 안정성 원자료**(구성 변형 H: 스무딩24+밴드0.002+voltarget12% 기준, 이번 정식 구현과 파라미터 동일):

  | fold | bars | net growth | Sharpe |
  |---:|---:|---:|---:|
  | 0 | 340 | −8.7% | −0.51 |
  | 1 | 340 | −17.9% | −1.44 |
  | 2 | 340 | **+120.7%** | **+6.93**(극단치) |
  | 3 | 340 | +46.7% | +3.01 |
  | 4 | 340 | +7.9% | +0.63 |

  → fold 2의 극단적 양의 성과가 스펙 작성 시점 sweep 표의 총합 지표를 크게 끌어올리고 있었음(다중검정 편향 주의, 스펙 [LIMIT-02]에 기록됨). 실제 프로덕션 파라미터로 재실행한 이번 결과(`fold_growths` 위 참조)는 부호가 다시 섞여(fold 2가 오히려 최대 손실 −44.5%) window/seed 민감도가 매우 높음을 보여준다 — **fold 성과의 부호 자체가 안정적이지 않다는 것이 핵심 병목**.
- **momentum_ts 15개 admitted sleeve의 fold×cluster 분포**, `se_ols_ratio`, `n_blocks` 등 sleeve별 원자료는 `logs/l1_admission.jsonl`의 `tag=ALGO` 행에서 `signal_id`가 `momentum_ts:*`인 항목으로 전수 조회 가능(재현 시드 42, `--date 2026-07-15 --sync local`).

### 최종 판정 및 다음 착수점

1. 포지션 구성 결함(C-1: 게이트/배포 북 불일치, C-2: 방향성 베팅 은폐)은 완전히 해소 — turnover 60배·cost_drag 60배·LCB90 폭 3배 개선을 코드·실측으로 확정.
2. **여전히 NO_EVIDENCE** — 병목이 구성에서 **fold 간 일관성 부족(비정상성)**으로 이동. `positive_folds=2/5`, fold 성과 부호가 window마다 크게 흔들림.
3. 다음 스펙 후보(우선순위순): (a) fold 간 비정상성 자체를 겨냥한 스펙 — regime 조건화, fold 수 확대, 혹은 momentum_ts 자체의 시간가변 강건성 진단. (b) momentum_ts:very_slow(beta=0.2265, n_blocks=13.8) 소표본 아티팩트 여부 별도 검증. (c) trend_ema family(전 speed beta≈0, SE 팽창 최대) 폐기 여부 결정.
4. Exit Policy(소비자 0개) 결함은 사용자 결정으로 계속 보류.
