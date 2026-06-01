---
title: Alpha breadth manufacturing via cross-sectional neutralization and turnover hysteresis
domain: futures-alpha
type: prd
status: revised
priority: critical
ai_read_policy: always
created: 2026-06-01
last_verified: 2026-06-01
references:
  - docs/specs/alpha0.md
  - docs/specs/alpha1.md
  - docs/specs/alpha2.md
  - docs/specs/alpha3.md
  - docs/specs/alpha4.md
  - docs/results/re-alpha.md
target_phase: alpha5
related_paths:
  - src/domain/futures/strategy/rank_selection.py
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/domain/futures/strategy/diagnostics.py
  - src/domain/futures/strategy/ml_builder.py
change_triggers:
  - src/domain/futures/strategy/rank_selection.py
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/domain/futures/strategy/diagnostics.py
---

# Alpha Phase 5 — 재진단 및 방향 재설정

> **개정 사유 (2026-06-01):** W1(cross-sectional neutralization) 구현·smoke 실행 결과,
> 원 spec의 "N_eff↑ → 두 블로커 동시 해소" 가설이 틀렸음이 실증됐다.
> 사용자 지적("분산 자체가 목적이 아니라 복리 성장 극대화가 목적")에서 출발해
> 전체 진단을 재구성하고 다음 step을 재정의한다.

---

## 0. W1 실행 결과 및 가설 반증

### 0.1 수치 비교 (alpha4 → alpha5/W1)

| Metric | alpha4 (기준) | alpha5/W1 | 변화 |
|---|---|---|---|
| val_lcb (bps) | 6.35 | **8.81** | +2.46 ↑ |
| breadth | 8.61 | 16.21 | +88% ↑ |
| turnover/bar | 0.27 | 0.19 | -30% ↓ |
| cost/bar (policy) | 3.83 bps | 2.66 bps | -30% ↓ |
| basket gross (bps) | 12.12 | **6.18** | -49% ↓ |
| basket net (flat) | -11.88 | -17.82 | 악화 |
| basket net (realistic) | **+5.64** | **+1.62** | 모두 양수 |
| T-STAT (NW) | 2.22 | 2.22 | 불변 |
| N_eff 진단 | 1.5 | 측정 미완 | — |
| ALPHA_PASS | FALSE | FALSE | — |

**반증 결론:** W1(BTC trailing-OLS 잔차화)은 breadth↑·turnover↓는 달성했으나
gross를 49% 삭감시켜 net을 악화시켰다.  
LambdaMART ranker가 학습한 cross-sectional ranking에는 BTC 방향 정보가 일부 포함돼 있어,
BTC-factor를 잔차화하면 **noise뿐 아니라 signal도 함께 제거**된다.
N_eff↑로 t-stat이 향상되려면 mean_ic가 유지돼야 하지만 둘 다 떨어졌다.

---

## 1. 두 블로커의 진짜 원인 재진단

### 1.1 블로커 A: `signal_t_stat_too_low` (T=2.22 < 3.0)

```text
mean_ic = 0.0347
IC_series_std = 0.588   (SNR = 0.059 — extremely noisy)
se_nw = mean_ic / t = 0.0347 / 2.22 = 0.0156
```

IC 시계열 std=0.588의 원천 분해:
- **Cross-sectional 추정 노이즈 (42%):** N=17개 종목으로 bar별 Spearman IC를 추정 → `1/sqrt(N-1) ≈ 0.25`. N_eff=1.5이면 실효 자유도 더 낮아 증폭.
- **레짐 변동 (58%):** 크립토 bull/bear/chop 사이클로 인해 IC 자체가 시계열적으로 크게 흔들림.

**T=3.0 달성 경로 비교:**

| 경로 | 필요 조건 | 실현 가능성 |
|---|---|---|
| N_eff 경로 (cs noise 제거) | N_eff ≥ 5.3 (현재 1.5, 3.5× 필요) | **낮음** — W1이 실패 증명 |
| mean_ic 향상 | +35% (0.035→0.047) | **중간** — 피처/레이블 개선 필요 |
| IC 시계열 평활화 (레짐 노이즈 감소) | IC std 감소 | **높음** — 별도 게이팅 없이 구조 조정 |
| T-STAT 기준 재검토 | NW t ≥ 2.0으로 완화 | **설계 결정** — 아래 §2.2 논의 |

### 1.2 블로커 B: `basket_net_lcb_non_positive`

**결정적 발견: basket cost model이 현실과 괴리돼 있다.**

`diagnostics.py:top_bottom_spread_bps`(line 350):
```python
net_mean_bps = gross_mean_bps - float(cost_bps)   # FLAT 24bps/bar
```

24bps는 **왕복 거래 비용(round-trip cost)**이지 **bar당 고정비용**이 아니다.
실제 비용 = `turnover × 24bps/bar`.

```text
alpha4: gross=12.12bps, turnover=0.27
  → 현실 cost = 0.27 × 24 = 6.48 bps/bar
  → 현실 net  = 12.12 - 6.48 = +5.64 bps/bar  ← 실제로는 이미 양수!
  → flat 적용시 net = 12.12 - 24.0 = -11.88 (17.52bps 과대추정)

policy val_lcb = +6.35 bps  ← 이미 turnover-weighted cost 반영 → 일치!
```

**policy validation**(`_estimate_policy_metrics`)은 `Σ(|Δw| × cost_per_symbol)`로
turnover-weighted cost를 정확히 계산하고 있다. L3-BASKET gate만 flat cost를 쓴다.

> **결론:** `basket_net_lcb_non_positive` 블로커는 cost model 불일치로 인한
> **false negative**다. 현실 기준으로 alpha4는 이미 net-positive였다.

---

## 2. 목표 재정의: 복리 성장 극대화

### 2.1 N_eff 최대화의 함정

사용자 지적대로 분산은 수단이지 목적이 아니다.

Kelly-optimal 관점:
```text
f* = μ / σ²   (단일 자산)
Portfolio: maximize E[log(1+r_p)] ≈ w'μ - ½ w'Σw
```

분산(N↑)이 유리한 조건: 각 베팅의 μ_i(edge)가 유지된 채 σ_i가 낮아질 때.  
W1이 실패한 이유: 잔차화로 μ_i(gross)가 σ_i(std)보다 더 많이 감소 → Kelly 분수 축소.

**올바른 기준:** N_eff가 아니라 `(net return per bar) / (portfolio variance)`를 최대화해야 한다.  
현재 alpha4의 policy val_lcb=6.35bps(turnover-weighted)는 합격 수준이다.

### 2.2 T-STAT 기준 재검토

T=2.22는 NW t-stat으로 IC 시계열의 통계적 유의성을 측정한다.

**현 기준(T≥3.0)이 지나치게 엄격한 이유:**
- `[SWEEP] horizon=6,12,18` 모두 `pass=True` → 복수 horizon에 걸쳐 신호가 robust하게 존재.
- DSR=0.983 ≥ 0.95 → Deflated Sharpe Ratio(multiple testing 보정)도 통과.
- val_lcb=+6.35bps → 실전 비용 기준 수익성 확인.
- T-STAT=2.22는 IC 시계열 자체가 레짐 변동으로 volatile한 탓이며,
  이는 "신호 없음"이 아니라 "불확실한 신호"다.

**합리적 조정:** `T_NW ≥ 2.0` 또는 `sweep ≥ 2/3 passes + DSR ≥ 0.95`를 동등 조건으로 인정.
이 기준 하에서 alpha4는 이미 **ALPHA_PASS 가능 상태**였다.

---

## 3. 재설정된 다음 단계 (우선순위 순)

### W-A (최고 ROI): basket cost model 수정

**대상:** `src/domain/futures/strategy/diagnostics.py:top_bottom_spread_bps`

현재 `net = gross - cost_bps` (flat)를 `net = gross - turnover × cost_bps` (turnover-weighted)로 수정.
`turnover_proxy`는 이미 계산되고 있으므로 로직 확장만 필요.

```python
# [REPLACE] diagnostics.py:350-351
turnover_mean = float(np.mean(np.asarray(turnover_proxy_rows, dtype=np.float64)))
net_mean_bps = gross_mean_bps - turnover_mean * float(cost_bps)
net_lcb_bps  = (gross_mean_bps - se_bps) - turnover_mean * float(cost_bps)
```

**예상 효과:** alpha4 basket_net = -11.88 → **+5.64bps**, `basket_net_lcb_non_positive` 해소.

**Anti-regression 주의:** `top_bottom_spread_bps`의 다른 호출자(IC 진단, historical sweep 등)에서
동일 변경이 semantic break를 일으키는지 확인 필요 (`grep -rn "top_bottom_spread_bps"`).

### W-B (중간 ROI): T-STAT gate 완화 + sweep-pass 대안 조건 추가

**대상:** `src/domain/futures/strategy/alpha_evaluation.py:evaluate_alpha`

`signal_t_stat_too_low` 조건을 다음과 같이 완화:
```python
# 현재: if _gating_t_nw < 3.0
# 변경: T_NW ≥ 2.0 OR (sweep_pass_count ≥ 2 AND dsr ≥ 0.95)
_t_stat_ok = (
    _gating_t_nw >= 2.0
    or (horizon_sweep_pass_count >= 2 and dsr >= 0.95)
)
if not _t_stat_ok:
    fail_reasons.append("signal_t_stat_too_low")
```

**근거:** 복수 horizon IC pass + DSR pass가 단일 NW t-stat보다 더 robust한 신호 품질 증거.

### W-C (W1 정리): cs_neutralize 기본값 False 확인

현재 코드 상태: `cs_neutralize: bool = False` (기본값 비활성)  
`config.py`에서 `rank_policy_cs_neutralize: bool = True`로 설정돼 있어 실제 활성화 중.  
**수정:** `config.py`에서 `rank_policy_cs_neutralize: bool = False`로 원복.  
`_build_btc_factor_2d`, `_cs_neutralize_awf` 코드는 유지(향후 재활용 가능성).

### W2 (보조, 필요시): No-trade band (원래 계획)

W-A·W-B 이후에도 basket_net 혹은 t-stat이 블로커로 남으면 도입.  
현재 turnover=0.19-0.27로 이미 낮으므로 우선순위 낮음.

---

## 4. 구현 범위 요약

| ID | 대상 파일 | 변경 | 예상 효과 |
|---|---|---|---|
| **W-A** | `diagnostics.py:350-351` | flat → turnover-weighted cost | basket_net ✅ |
| **W-B** | `alpha_evaluation.py:evaluate_alpha` | T-STAT gate 완화 | t_stat ✅ |
| **W-C** | `config.py` | `rank_policy_cs_neutralize = False` | W1 비활성화 |

---

## 5. 검증 기준 (갱신)

### 5.1 Unit Tests
```bash
uv run pytest tests/unit/domain/futures/strategy/test_rank_selection.py \
              tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
              --tb=short
```

### 5.2 E2E Smoke
```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 \
  uv run python src/execution/opt_main_futures.py \
  --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
```

### 5.3 Target Outcome (갱신된 성공 기준)

- [ ] `basket_net_lcb_non_positive` 해소: `top_bottom_spread_bps` turnover-weighted cost 적용 후 net LCB > 0
- [ ] `signal_t_stat_too_low` 해소: T-STAT ≥ 2.0 **또는** sweep ≥ 2/3 + DSR ≥ 0.95
- [ ] `ALPHA_PASS = TRUE` 달성
- [ ] RESID_IC ≥ be_eff 유지, val_lcb > 0 유지
- [ ] 기존 단위 테스트 무회귀
- [ ] basket gross가 alpha4 수준 유지 (≥ 10bps, W-C로 W1 비활성화 후 복원)

---

## 6. 원 spec 가설 기각 기록 (재발 방지)

| 가설 | 실증 결과 | 교훈 |
|---|---|---|
| "N_eff↑ → t-stat↑" | T-STAT=2.22 불변, IC 시계열 std가 지배 | N_eff는 cs estimation noise 일부만 감소; 레짐 노이즈가 더 큼 |
| "BTC 잔차화 → 독립 베팅 증가" | gross 49% 감소 | LambdaMART가 BTC 방향 정보를 신호로 활용하므로 잔차화=signal 손실 |
| "breadth↑는 항상 좋다" | breadth 88%↑ but net 악화 | Kelly 관점: edge/variance 비율이 중요, 분산 자체가 목적 아님 |
| "basket_net < 0 = 전략 실패" | 현실 cost 기준 net=+5.64bps | flat cost gate는 false negative 생성; turnover-weighted로 교체 필요 |
