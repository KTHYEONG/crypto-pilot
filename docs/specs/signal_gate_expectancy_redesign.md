---
title: Signal Gate Expectancy Redesign
type: spec
status: proposal
created: 2026-06-05
---

# 🎯 Objective
현재 임계값 완화로 pass-through화된 signal 승격 게이트를 **경제적 판별력 게이트**로 교체한다.
ML 진행 전 rule 후보가 순수하게 비용 초과 mean-net을 생성함을 블렌드 수준에서 입증한다.

# 💡 Strategy (근본 진단)

## 현상
- Phase 1~3 적용 결과 KEEP 12/20 variants 생성 성공 (이전 0/20).
- 그러나 `apply_variant_promotions` 출력 n=2355 == raw labeled n=2355 → 선택이 실질적으로 no-op.
- `validate_candidate_signals` 블렌드: `net_p50=−117bps`, `net_stress_p50=−128bps`, `overall_pass=False`.
- 승격 집합이 경제성을 전혀 개선하지 못함.

## 근본 원인
`median_edge ≥ −100` + `p10_edge ≥ −600` 완화가 임계값을 풀어 통계적 선별을 제거.
올바른 처방은 median/p10 임계값 조정이 아니라 **게이트 목적함수 자체를 교체**:
- **현재:** tail 절대값(median ≥ 0, p10 ≥ −150) → 구조적으로 skew payoff 전략 전멸.
- **필요:** net mean expectancy(mean − cost > breakeven) + fold 안정성.

## 설계 원칙
1. **mean-net이 primary 경제 게이트** — `oos_mean_edge_bps − rt_cost_bps ≥ min_net_edge_bps`.
2. **median/p10은 soft outlier filter only** — 절대 임계 대신 mean 대비 ratio 기준.
3. **블렌드 검증을 승격 조건에 편입** — keep-set의 `mean-net > 0` + IR ≥ min_rule_ir_t.
4. **임계값 완화를 환원** — Phase 1의 −100/−600 롤백, 구조적 정당화 기준으로 재설정.

---

# 🛠️ Surgical Plan

## Step A — config.py: 임계값 환원 + mean-net 게이트 추가

### [ACTION: UPDATE] `src/domain/futures/strategy/config.py` — `CandidateStrategyConfig`

```python
# 기존 완화값 → 환원 + net 기반 재설정
min_variant_oos_median_edge_bps: float = -50.0   # 완화(-100) 환원; 강한 좌쏠림 outlier만 차단
min_variant_oos_p10_edge_bps: float = -400.0      # 환원(-600→-400); ATR stop 구조 반영하되 outlier 차단
p10_edge_relative_to_stop: bool = False            # 유지(단순화)

# 신규: mean-net 게이트 (핵심 추가)
min_variant_oos_net_mean_edge_bps: float = 0.0    # oos_mean_edge_bps - rt_cost >= 0 (비용 최소 회수)
min_variant_oos_net_mean_edge_bps_strict: float = 3.0  # 엄격 모드(선택): 비용의 ~40% 초과
use_net_mean_gate: bool = True                     # mean_edge check를 net 기준으로 전환

# 블렌드 검증을 승격 조건에 편입
blend_survival_gate_enabled: bool = True           # keep-set 블렌드의 mean-net > 0 요구
blend_survival_min_ir_t: float = 1.0               # keep-set 블렌드의 IR t-stat 요구
```

`__post_init__`에 검증 추가:
```python
if self.min_variant_oos_net_mean_edge_bps < 0.0:
    raise ValueError("min_variant_oos_net_mean_edge_bps must be >= 0")
```

## Step B — rule_diagnostics.py: mean-net 게이트 적용

### [ACTION: UPDATE] `_recommendation_threshold_checks` (line ~636)

```python
# rt_cost를 row에서 읽거나 config 기본값 사용
rt_cost = float(row.get("rt_cost_bps", cfg.cost_floor_bps))

# mean_edge 체크를 net 기준으로 전환
if cfg.use_net_mean_gate:
    mean_net = float(row.get("oos_mean_edge_bps", float("nan"))) - rt_cost
    mean_ok = mean_net >= cfg.min_variant_oos_net_mean_edge_bps
else:
    mean_ok = float(row.get("oos_mean_edge_bps", float("nan"))) >= cfg.min_variant_oos_edge_bps
```

dict 내 `"mean_edge"` 항목 교체:
```python
"mean_edge": mean_ok,
```

## Step C — ablation.py: `survives_cost` 판정을 mean 기반으로 정정

### [ACTION: UPDATE] `validate_candidate_signals` (line ~1175)

현재: `net_stress_p50 > 0` (median 기반)
교체:
```python
# mean-net으로 교체: mean이 stress-rt를 초과하면 pass
mean_net = float(np.mean(finite_edge)) if finite_edge.size > 0 else float("nan")
mean_stress_net = mean_net - stress_rt if np.isfinite(mean_net) else float("nan")

survives = (
    np.isfinite(mean_stress_net)
    and mean_stress_net > 0.0
    and ir_t >= cfg.min_rule_ir_t
)
```

보고 필드에 추가:
```python
net_edge_bps_mean=mean_net,            # mean-net (median 대신)
net_edge_bps_stress_mean=mean_stress_net,
```

## Step D — rule_diagnostics.py: blend 검증 편입 (선택적, blend_survival_gate_enabled)

### [ACTION: UPDATE] `_summarize_recommendation_variants` 반환 직전 또는 호출부

`keep_variants` 확정 후 블렌드 mean-net 검증을 추가하는 단계:
- 이미 keep-set의 이벤트는 `apply_variant_promotions`로 필터 가능.
- 필터된 OOS 이벤트의 `edge_after_hurdle_bps`에서 mean/IR 계산.
- `cfg.blend_survival_gate_enabled=True`이고 블렌드 mean-net ≤ 0이면 → keep_variants를 top 50%로 축소하거나 경고 로그.
- **구현 복잡도 주의:** 이 단계는 `diag` 객체 생성 후이므로 `RuleDiagnosticsResult`를 반환하기 전에 추가해야 함. 단순 버전은 경고 로그만 발행.

---

# ✅ Verification

```bash
# 1. 단위 테스트
uv run pytest tests/unit/domain/futures/strategy/ -q

# 2. signal phase 재실행 — 기대 결과:
#    - KEEP 집합이 경제적으로 선별됨 (이전보다 적을 수 있음)
#    - rule_promo_no_leak의 mean-net > 0, survives_cost=True
#    - overall_pass=True
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
  --phase signal --sync skip --timeframe 4h --trials 1 --date 2026-05-01 2>&1 | \
  grep -E "SIGNAL-VALIDATION|overall_pass|survives|Active|Status|KEEP"
```

**기대 출력:**
```
[SIGNAL-VALIDATION] variant=rule_promo_no_leak ... survives=True
[SIGNAL-VALIDATION] overall_pass=True
```

# ⚠️ Risk

- **과강화 위험:** mean-net ≥ 0 (breakeven)이 너무 느슨하면 여전히 통과. `min_variant_oos_net_mean_edge_bps_strict=3.0`으로 두 번째 기준 적용 고려.
- **KEEP 0 재발 가능:** 진짜 알파가 없으면 net-mean이 모든 variant에서 음수 → 0 KEEP. 이 경우는 신호 추가/redesign 논의 필요.
- **Step D 복잡도:** blend 검증 편입은 `RuleDiagnosticsResult` 계약 변경을 수반할 수 있음. 먼저 Step A~C로 `overall_pass=True` 확보 후 Step D 별도 진행 권장.
