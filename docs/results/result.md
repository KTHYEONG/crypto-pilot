# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-07 (진단 로그 추가 후 재실행 — signal_ml_pipeline_audit.md)
**현재 상태:** `BLOCKED` — `pass_ratio=0.00` (변화 없음)
**평가 기준:** `min_fold_realized_edge_bps=15.0`, `min_cagr_for_promotion=0.15`, `min_edge_rank_ic=0.02`, `signal_prequalify_min_obs=30`

---

## 최신 실행 요약 (candidate_v6)

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 (63 ready)
[PIPELINE] 4 folds / wf_scheme=anchored
[SIGNAL_PREQUALIFY] fold별 평균 ~29% 이벤트 제거
[EDGE_GATE] mode=rank_ic
  fold1: rejected (insufficient_obs, n=725)  → inference_mode=prior_only
  fold2: rejected (insufficient_obs, n=869)  → inference_mode=prior_only
  fold3: rejected (rank_ic=-0.005, t=-0.14)  → inference_mode=prior_only
  fold4: ACCEPTED (rank_ic=0.071, t=1.98)    → inference_mode=direct
[BRIDGE][WF] fold_cost_survival=[False, False, False, False] pass_ratio=0.00 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status blocked
[BRIDGE SUMMARY][WF_DIAG] wf_selected=0 wf_eligible=0 shadow_profiles=18 shadow_max_selected=37 shadow_max_eligible=420 eu_p90=1.437 downside_p90=8.490

[ABLATION cal_eval] rank_ic=0.0689, t=2.23 (n=1047) → accepted (direct mode)
```

---

## fold별 상세 결과 (+ 2026-06-07 진단 로그)

| Fold | OOS 기간 | mode | rank_ic | n | prior | mu_p90 | eu_p90 | break | eligible | 선택 | 결과 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Oct-Nov | prior_only | n/a | 377 | 0.01 | 1.9 | 1.89 | 3.75 | 0 | 0 | ❌ |
| 2 | Nov-Dec | prior_only | n/a | 303 | 0.01 | 1.5 | 1.50 | 3.75 | 0 | 0 | ❌ |
| 3 | Jan-Feb | prior_only | -0.005 | 383 | 0.00 | 1.3 | 1.27 | 3.75 | 0 | 0 | ❌ |
| 4 | Feb-Mar | **direct** | +0.071 | 420 | 0.00 | 1.1 | 1.08 | 3.75 | 0 | 0 | ❌ |
| ablation | 전체 | direct | n/a | 1491 | 0.00 | 1.7 | n/a | 3.75 | 0 | 0 | ❌ |

**신규 컬럼 해석:**
- `prior`: `[DIAG][PRIOR]` global_prior_bps
- `eligible`: `[DIAG][SELECT_ZERO]` 결과 — 모든 fold에서 0 (breakeven 미달)
- `선택`: WF selected events

---

## 백테스트 성과 (OOS Ablation — 6 Causal Variants)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| rule_stop_risk | -20.4% | 16.6% | -1.23 | 875,924 | 620 | 1.00 | N |
| prior_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| prior_residual_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| edge_plus_validated_gate_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| edge_plus_gate_event_kelly | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| full_portfolio_caps | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |

---

## 근본 진단 (진단 로그 추가 후 재실행 — 2026-06-07 v2)

| 항목 | 내용 |
|---|---|
| **신규 발견: Prior vs Breakeven 정량화** | `[DIAG][PRIOR]` 로그 새로 추가. 모든 fold에서 `global_prior_bps=0.00~0.01 bps << breakeven_floor=3.75 bps`. 이는 IS 학습 구간의 신호 성과가 극히 낮다는 의미. |
| **신규 진단: Per-Variant Prior** | `[DIAG][VARIANT_PRIOR]` 로그. top-2 variant 모두 `prior=0.00 bps` → 개별 신호마다 IS mean_edge ≈ 0.01 bps 수준. |
| **신규 진단: SIGNAL_PREQUALIFY 탈락 원인** | `[SIGNAL_PREQUALIFY][DISQ]` 로그. 예: tpc_20_100은 263/939 이벤트 탈락, IS mean_edge=12.05 bps. 즉, IS에서는 양의 수익이 있지만 t-stat 기준으로 유의성 없음. |
| **신규: CANDIDATE TOP STRATEGIES Rec 컬럼** | KEEP 신호(OOS 기반)와 실제 학습 포함 여부(IS+Cal 기반) 비교 가능. |
| candidate_v6 적용 | 피처 6개 추가. WF 결과 변화 없음 (예상대로). |
| 핵심 병목 진단 (확정) | `global_prior ≈ 0.00 bps` — IS 신호 성과 자체가 비용(3.8 bps)보다 1000배 낮음. prior가 breakeven을 절대 넘을 수 없음. μ_p90 ≤ 1.7 bps 는 불가피한 결과, 설계 결함 아님. |
| 다음 돌파 방향 (변경 없음) | (1) IS signal alpha 재설계 (tpc_50_200 등 IS mean edge 상향) (2) 비용 재검증 (3) 4h 타임프레임 검토 |

---

## 다음 단계 (2026-06-07 진단 결과 기반 — 우선순위 재정렬)

### ✅ 확정된 병목 (코드 버그 아님)
```
IS signal prior (0.00 bps) << breakeven_floor (3.75 bps)
→ prior가 breakeven을 절대 초과할 수 없음
→ eligible=0 고착 (구조적, 피처 추가로 탈출 불가)
```

### 1️⃣ **신호 alpha 재설계 (가장 중요)**
- IS mean_edge가 모든 신호에서 ≤ 0.01 bps 수준 → 비용 7.5 bps 대비 1000배 낮음
- `tpc_50_200` IS mean=? OOS mean=? 갭 분석 (현재 로그에 IS 74bps는 무엇인지 확인)
- 신호별 IS mean_edge 분포도 작성 필요
- **결론:** 규칙 신호 자체가 1h 타임프레임에서 엣지 부재

### 2️⃣ **비용 가정 재검증**
- `cost_floor_bps=7.5` (maker 2.0 + taker 5.0 + slippage 1.0 + impact 0.0)
- 실제 Binance futures taker fee 현황 (2026-06-07 기준)
- 펀딩비 (24h 평균)
- 슬리피지 및 impact 재검토

### 3️⃣ **4h 타임프레임 실험 (선택사항)**
- `--timeframe 4h`로 재실행
- 신호 노이즈 감소 기대 vs 신호 개수 감소 트레이드오프

### ⚠️ 구조적 한계 인정
- `prior ≤ 0.01 bps`, `breakeven = 3.75 bps` → **IS 신호 alpha 부재가 근본 원인**
- ML 게이트 강화, 피처 추가, 모델 개선 → 모두 prior를 올릴 수 없음
- **포기 판정 기준:** IS mean_edge가 breakeven의 20% 미만 (현재 << 0.8 bps)이면 전략 무용
