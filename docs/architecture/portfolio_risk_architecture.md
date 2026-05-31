---
title: Portfolio & Risk 계층 아키텍처 (as-built + 갭 분석)
domain: futures-portfolio
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/forecast/compose.py
  - src/domain/futures/portfolio/portfolio_constructor.py
  - src/domain/futures/portfolio/portfolio_optimizer.py
  - src/domain/futures/portfolio/risk_controls.py
  - src/domain/futures/portfolio/execution_sim.py
  - src/domain/futures/backtest/engine.py
  - src/domain/futures/optimization/objectives.py
  - src/domain/futures/optimization/opt_config.py
change_triggers:
  - src/domain/futures/portfolio/**
  - src/domain/futures/backtest/engine.py
last_verified: 2026-06-01
dependencies:
  documents: [docs/specs/alpha0.md, docs/specs/alpha2.md]
---

# Portfolio & Risk 아키텍처 (실측 기반)

> **정정 고지**: 본 문서 초판은 "portfolio/ 모듈은 미연결 dead, inverse-vol 엔진" 이라 서술했으나 **사실과 다름**(손상된 read 기반 오류). 코드 재검증 결과 portfolio/ 스택은 **이미 연결·활성이며 기관급**이다. 본 개정판이 정답.

## 1. As-Built 데이터 플로우 (검증됨)

```
AlphaForecast(alpha_long/short, rank_score) + CostForecast
   │
[L1] compose_mu  (forecast/compose.py)              ← objectives.py:443 호출
   │   · μ = β_alpha·alpha − cost_frac(maker/taker/slippage 복합) − (amortize 옵션)
   │   · soft-hurdle(sigmoid) → rank-sizing tanh  →  xs_long/short(±1), mu_long/short(EV)
   │   · admission_mode: ev_gate | rank_then_ev_gate | rank_cs_neutral
   ▼
[L2] precompute_rebalance_weights (portfolio_constructor.py)  ← objectives.py:538
   │   · μ = mu_long − mu_short (EV, policy_inputs 우선)
   │   · **fractional Kelly** f=μ/σ²·κ, KELLY_FRACTION=0.25, f_max=min(F_KELLY_MAX, KELLY_IC_UPPER=0.5)
   │   · **Ledoit-Wolf 공분산**(rolling, by-TF lookback) → 상관 반영
   │   · **portfolio-level vol target**(σ_target_ann)
   │   · **5-cap projection**(project_all_caps): gross / per_symbol / net / **β-neutral** / vol
   │   · no-trade buffer(risk_controls.apply_no_trade_buffer), minNotional 양자화
   ▼
[L3] backtest_target_weights_numba (backtest/engine.py)  ← objectives.py:18,745
   │   · shared-margin 포트 sim: MtM PnL − funding − turnover·cost
   │   · intrabar 변형(backtest_target_weights_intrabar_numba) + execution_sim(1049L) 체결 모델
   ▼
equity / gross_exp / trade_pnl → AWF 게이트 → Optuna objective
```

## 2. 이미 구현·연결된 것 (검증됨) — "새로 만들 필요 없음"

| 기능 | 위치 | 상태 |
|---|---|---|
| EV-aware **fractional Kelly**(0.25) | portfolio_constructor `_kelly_*` | ✅ active |
| **Ledoit-Wolf** shrinkage 공분산(상관) | portfolio_constructor `rolling_ledoit_wolf_cov` | ✅ active |
| **portfolio vol-target** | `solve_constrained_weights` / `project_all_caps` cap5 | ✅ active |
| **β-neutral cap**(BTC beta) | `project_all_caps` cap4 | ✅ active |
| gross/per-symbol/net cap | `project_all_caps` cap1-3 | ✅ active |
| turnover no-trade buffer | risk_controls `apply_no_trade_buffer` | ✅ imported(objectives:26) |
| **DD throttle** | numba 엔진 `DD_SCALING_THRESHOLD` 인자(objectives:736,769) | ✅ active (엔진 내부) |
| funding+turnover 비용 회계 | execution_sim `backtest_target_weights_numba`(backtest/engine 재노출) | ✅ active |
| intrabar 체결 현실 | execution_sim intrabar kernel(1049L) | ✅ active |
| rank-sizing + soft-hurdle | compose.py `_rank_weight_1d`/`_soft_hurdle` | ✅ active |
| bear/crisis 위험조정 | `DYNAMIC_RA_BEAR_COEF`(엔진 params 도달: objectives:238, ml_context:819), `CRISIS_GAMMA/GATE_PROB`, `CRISIS_LONG_Z_BOOST` | ✅ params 전달(최종 적용=dyn_leverage, §4 확인) |

→ **결론: 포트폴리오/리스크 계층은 이미 기관급. 신규 모듈 신설은 불필요(중복 위험).**

## 3. 복리극대화 관점 — 정합성 평가 (긍정)

- alpha 게이트가 **β-residualized** target IC를 보고, 포트는 **β-neutral cap** → 신호 평가와 실거래가 같은 market-neutral 가정 위에 정렬. **불협화음 없음.**
- bear IC≈0 의 재해석: 포트가 β-neutral이므로 bear 구간은 "XS 엣지 없음 → 대체로 flat", **방향성 손실이 아님**. 이전 분석(docs/specs/alpha2.md C6)의 "하락장 누수" 우려는 β-neutral 가정 하에서 **완화**됨(전면 철회는 아님 — XS 잔차상관 잔존).
- thin edge(IC 0.038)에 Kelly가 공격적일 수 있으나 κ=0.35 · KELLY_FRACTION 0.25 · f_max 0.5 · LW shrinkage 4중 보수화로 estimation-error 증폭은 구조적으로 억제됨.

## 4. 진짜 갭 (미검증·검토 필요 — 신규 개발 아님, 배선/보정 확인)

1. **COST_GATE_AMORTIZE 불일치 (HIGH)**: `opt_config.py:195` 기본값 `True`. alpha0.md §objectives 명세는 "default must be False, tests must assert". → 비용 amortize가 켜져 gate가 관대해졌을 수 있음. **검증 후 통일.**
2. **DD 방어 이중화/dead (LOW-MED)**: 실제 DD throttle은 numba 엔진 `DD_SCALING_THRESHOLD`로 **active**. 반면 `risk_controls.compute_drawdown_gross_scale`(tier 0.7/0.4)는 **호출처 0 = dead 중복**. → 죽은 함수 제거 또는 단일화(혼동 방지).
3. **bear/crisis 최종 적용점 (MED)**: `DYNAMIC_RA_BEAR_COEF`·`CRISIS_*`는 엔진 params까지 도달 확인(objectives:238). 단 weight/leverage에 실제 곱해지는 지점(dyn_leverage 산출)은 1-line 확인 필요 — bear 선제 축소가 실효적인지 검증.
4. **Kelly μ 단위 정합 (MED)**: compose_mu가 xs(rank ±1)와 mu(EV)를 모두 산출. L2는 `policy_inputs.mu_long-mu_short`(EV) 사용 확인 → 정상. 단 EV가 β-resid·rank 경로를 거친 뒤 **return-fraction 스케일이 보존**되는지(σ² 대비) 수치 점검 권장.

## 5. 권고 (ROI 순) — "만들기" 아님, "검증·보정"

- **P1**: COST_GATE_AMORTIZE 정합(§4.1) + alpha 게이트 정직화(docs/specs/alpha2.md). 저비용·고ROI.
- **P2**: §4.2/4.3 배선 audit — DD-tier·bear coef가 실제 weight에 닿는지 grep+단위테스트. 죽었으면 1개 경로로 연결.
- **P3**: thin-edge 보정 진단 — Kelly κ/f_max sweep이 OOS Calmar에 미치는 민감도. 과레버 구간 컷.
- **(불권장)** 신규 portfolio/risk 모듈 작성 — 4번째 단편 생성, 중복.

## 6. OOS 설정 (Q2 정정)

검증값(opt_config / get_quarterly_window):
- **IS = 24개월, OOS = 6개월**(quarter_start −6mo ~ −1d). (사용자가 인지한 3개월과 상이 — 경로/버전 확인 필요.)
- **AWF(anchored walk-forward) 이미 가동**: `AWF_K_LEGS=5`, `WF_OOS_LEGS=5`, `use_anchored_awf_geometry=True`, `IS_POOL_FRAC 0.65~0.70`, embargo 4h=42bars.
- ML alpha OOS(신호) = fold test slice ≈ 6주(re-alpha 254bar) — AWF leg과 별개.

→ **"12개월 OOS"는 단일 holdout 확장이 아니라 AWF leg 수/OOS-pool span 조정 문제.** 인프라(5-leg anchored AWF)는 이미 존재. 권장: OOS pool을 12개월로 늘려 5-leg가 12개월을 타일링하도록 `FUTURES_AWF_IS_POOL_FRAC`·윈도우 조정(별도 spec). 단일 12mo holdout 금지(최근데이터·IS 손실).

## 7. Acceptance (검증 항목)
- COST_GATE_AMORTIZE 정책이 alpha gate와 backtest에서 동일.
- DD-tier / bear-coef 가 활성 weight 경로에 도달함을 테스트로 확인(또는 의도적 제거).
- portfolio/ 스택을 중복하는 신규 모듈을 만들지 않음(SSOT 유지).
