---
title: Universe Architecture — 3-Cohort Inference/Live/Execution 분리 (통합 명세서)
domain: strategy-ml
type: domain-spec
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/universe/selection.py
  - src/domain/futures/universe/membership.py
  - src/domain/futures/universe/pipeline.py
  - src/domain/futures/universe/models.py
  - src/domain/futures/universe/config.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/labels.py
  - src/domain/futures/strategy/ml_builder.py
  - src/application/futures/optimization/universe_service.py
  - src/application/futures/optimization/strategy_service.py
change_triggers:
  - src/domain/futures/universe/**
  - src/domain/futures/strategy/**
  - src/application/futures/optimization/strategy_service.py
  - src/application/futures/optimization/universe_service.py
dependencies:
  documents:
    - docs/specs/ml_alpha_specs_consolidated.md
    - docs/results/alpha0.md
last_verified: 2026-05-28
supersedes:
  - docs/specs/universe-fix.md  # Merged into this document; "방향 B" rejected, "방향 A" adopted
  - docs/specs/phase2_alpha_universe_decoupling.md  # Merged; P1 breadth expansion rationale integrated
---

# Universe Architecture — 3-Cohort Inference/Live/Execution 분리

## Executive Summary

현행 `Stage6 = tradeable_score 상위 K` 단일 유니버스가 (a) ML 학습 cross-section을 18~24개로 축소시켜 IR을 구조적으로 붕괴(`net_ic=0.0021`, `breadth=2.88`, `t_stat=1.07`)시키고, (b) "거래 가능"만 측정할 뿐 "수익 가능"을 측정하지 않아 자산증식 기여도가 낮다.

초기 해법들의 실패:
- **Phase 1 시도** (`phase2_alpha_universe_decoupling.md` P1): Stage5 전체(80~150개)로 학습하되 active_mask는 Stage6 timeline에 강결합 → `eligible=0.2866`으로 붕괴(자격 있는 행이 전체의 28.7%만 학습 참여).
- **우회로** (`universe-fix.md` "방향 B"): Historical Stage6 union(20~35개)로 제한 → Stage6 boundary churn 문제 해결 못함 + breadth 손실 가장 심함.

본 명세는 **3-Cohort 아키텍처**로 정공법 재설계한다 — active_mask를 Stage5 timeline으로 분리재구성하여 breadth 손실 없이 Phase 1의 정당성을 살린다:

| Cohort | 역할 | 규모 | active_mask 출처 | 우선 최적화 목표 |
|--------|------|------|------------------|-----------------|
| **C1 Inference (Historical Stage5 Union)** | ML 학습·IC 측정 | 80~150 | Stage5 quarterly membership | breadth √N (Grinold-Kahn) |
| **C2 Live Inference (Current Stage5 Passed)** | OOS 알파 발산 | 50~80 | Stage5 current quarter | live prediction coverage |
| **C3 Execution (Enhanced Stage6 Selected)** | 실제 진입 집행 | 18~24 | Stage6 current quarter | net IC × capacity × diversification |

추가로 Stage6 점수 함수를 단일 `tradeable_score`(거래 용이성 100%)에서 **3축 다목표 점수**(friction · alpha_capacity · diversification)로 확장하여 "거래 가능한 후보군"을 "효과적 자산증식 후보군"으로 끌어올린다.

---

## 0. 역사적 맥락 — 왜 3-Cohort인가? (Historical Context)

### 0.1 Problem Statement — Phase 1 active_mask 강결합 버그

Phase 1 구현 후 OOS 실측에서 다음 현상이 발견되었다:

```
symbols_loaded=37, eligible=0.2866  ← 기대값(0.914) 대비 69% 붕괴
ev_mean fold0=-2.96e-4, fold1=+1.04e-3
```

**근본 원인**: `universe_active_mask`는 **Stage6 quarterly membership timeline** 기반으로만 계산되었다.

```python
# src/domain/futures/universe/membership.py:50 (버그)
active_quarters = {q for q, syms in norm_timeline.items() if sym_norm in syms}
# norm_timeline = Stage6 quarterly selected 집합
```

Stage5는 통과했지만 **Stage6에 한 번도 선택된 적 없는 심볼**(예: 과거에 거래 가능했으나 최근 거래량 감소)은:
- `active_quarters = {}` (빈 집합)
- `universe_active_mask = False` (전 기간)
- `entry_block_mask = True` (전 기간)  
- `eligible = False` (학습 불참)

결과: 37개 심볼을 로드했지만 학습 대상은 10.6개(28.7%)에 불과 → 실질적 breadth = 2.88 → **P1의 breadth 확장 효과 전무**.

### 0.2 First Solution Attempt — "Historical Stage6 Union" (universe-fix.md "방향 B")

**제안**: Training panel을 "과거 Stage6 진입 이력이 있는 심볼"(historical_trading_panel)로 한정.

**이론적 근거**:
- Stage6 진입 경험이 있는 심볼은 이미 Stage2~5 모든 품질 필터 통과.
- 실제 거래 집행 이력이 있으므로 "비현실적 신호"를 학습할 위험 낮음.
- 실장이 간단: active_mask 재구성 불필요, 기존 timeline 그대로 활용.

**문제점**:
1. **Breadth 손실 심각**: Historical Stage6 union = 20~35개 (vs. 기대 80~150) → √N: 4.5~5.9 (vs. 9~12.2) → √BR 효과 **50% 손실**.
2. **Boundary Noise 해결 못함**: Stage6의 hard cutoff(rank K)는 분기마다 rank 15~25 심볼이 진입/이탈 → 학습 분포 흔들림 지속. 과거 진입 이력을 쌓아도 최신 상태는 여전히 경계에서 진동.
3. **정치**의 회피: 근본 원인(active_mask Stage6 강결합)을 해결하지 않고 우회로 → 향후 다른 문제 유발 가능성.

### 0.3 Root Cause Fix — 3-Cohort Architecture (본 명세)

**핵심 통찰**: 추론(Inference)과 집행(Execution)은 본질적으로 다른 요구사항을 가진다 (§1.1).

- **추론**: 횡단면 정보 밀도 최대화 → 객관 필터(Stage1~5)로 충분.
- **집행**: 거래 가능성 + 알파 수익성 동시 만족 → 다목표 점수 필요.

**재설계**:

1. **Active_mask 이중화**: `universe_active_mask`(Stage6) + `inference_active_mask`(Stage5) 별도 구성.
   ```python
   # inference_active_mask[s,t] = True ⟺ s ∈ stage5_passed(quarter_of(t))
   # universe_active_mask[s,t] = True ⟺ s ∈ stage6_selected(quarter_of(t))  [기존, 거래용]
   ```

2. **ML 학습이 inference_active_mask 사용**: C1 = historical Stage5 union, active_mask는 분기별 Stage5 기준 계산 → boundary noise 제거.

3. **C3 집행은 universe_active_mask 유지**: Stage6 선택 심볼만 거래 → 비용 모델 일관성 유지.

결과: breadth 손실 0, Phase 1의 정당성 보존, boundary noise 제거, 실장 복잡도는 active_mask 이중화만 추가.

---

## 1. 이론적 근거 (Theoretical Foundations)

### 1.0 Quant Framework — 대수적 필연성

본 설계는 다음 선행 논문/프레임워크의 수학적 결과에 기초한다:

**Fundamental Law of Active Management (Grinold-Kahn, 1989)**:
```
IR = IC · √BR · TC
```

- `IC`: 신호의 평균 정확도 ([-1, 1])
- `BR`: Breadth = 시간당 독립 베팅 수 (≈ N_effective)
- `TC`: Transfer Coefficient = alpha → 실현 PnL 전환 효율
- `IR`: Information Ratio = 초과 수익률 / 추적 오차

현행 N=18 → √BR ≈ 4.24에서, Stage5 전체 N=70~150으로 확장 시 √BR ≈ 8.4~12.2 → **√BR이 2~2.9배 증폭**. 같은 IC에서도 비용 장벽 통과 확률이 2배 상승.

**López de Prado, Advances in FML (2018)**:
- Ch. 6: 자산 수익률은 local stationary이며, 단일 IS 모델은 regime shift에 무방비.
  - 해법: (a) sample weight time-decay, (b) walk-forward 재학습 + embargo, (c) 멀티호라이즌.
- Ch. 7: "Boundary symbols inject distribution shift. When training set membership = portfolio membership, the model wastes capacity learning the boundary, not the alpha."
  - 학습 패널을 거래 패널보다 충분히 넓게 두는 것이 표준.

**LambdaMART/LambdaRank Pair Count** (ranking loss 특화):
```
Learning signal ∝ N(N-1)/2 pairs per timestep
- N=18:   153 pairs/step
- N=70: 2,415 pairs/step  → 15.8× 학습 신호 밀도
```

현재 `lgbm_lambdarank` 모델을 쓰면서 N=18로 제약하는 것은 자기모순. 모델과 데이터 사이즈의 mismatch.

**Survivorship Bias 회피** (Banz 1981, Davis et al. 2000):
- Universe selection에 ex-post 정보(미래에 강한 것으로 알려진 심볼)를 넣으면 IS/OOS divergence 폭발.
- Stage1~5는 PIT objective filter이므로 그 union은 survivorship-free.

### 1.1 Universe 분리의 필연성 — 두 가지 직교 목적의 혼동

학계·산업계에서 유니버스는 두 개의 본질적으로 다른 함수를 수행한다:

**A. 추론 유니버스 (Inference Universe)**

- 목표: 횡단면 정보밀도 최대화 → 통계검정력 확보
- 제약: PIT(point-in-time) objectivity, no survivorship
- 최적해: **다양하고 넓은** 종목군. Asness-Frazzini-Pedersen(2014, "Quality Minus Junk")은 4,000+ 종목으로 신호 추정.

**B. 집행 유니버스 (Execution Universe)**

- 목표: 알파 → 실현 PnL 전환률(`TC` Transfer Coefficient) 최대화
- 제약: 슬리피지·체결 가능성·capacity
- 최적해: **소수의 깊은 유동성**. Israel-Moskowitz(2013)는 신호 자체와 집행 가능성을 명확히 분리.

**두 함수의 최적해가 다르므로**, 단일 유니버스로 양자를 동시에 만족시키는 것은 수학적으로 dominated solution이다. 현재 시스템은 (A)를 (B)에 종속시켜 학습의 cross-section을 N=18로 강제 축소 → 통계검정력 √N 손실.

### 1.2 Grinold-Kahn Fundamental Law — Breadth는 절대 통계량

```
IR = IC · √BR · TC
```

| 항 | 의미 | 현재 | 분리 후 (예상) |
|---|---|---|---|
| `IC` | 신호의 평균 정확도 | 0.0021 | 0.005~0.010 (개선 별도) |
| `BR` (Breadth) | 시간당 독립 베팅 수 ≈ N_effective | 2.88 | **8~12 (3~4배)** |
| `√BR` | breadth 증폭 | 1.70 | **2.83~3.46** |
| `TC` | alpha → 포지션 전환 | ~0.7 | 변화 없음 (집행은 C3 그대로) |

**핵심**: IC를 직접 개선하는 것보다 **breadth 확장이 통계학적으로 더 쉬운 승리**다. N=18 → N=70은 √BR을 2.04× 증폭. 같은 IC에서도 비용 장벽 통과 가능성 2배.

### 1.3 LambdaMART/LambdaRank — N² 의존성

LambdaRank는 group(쿼리) 내 pairwise 비교로 학습한다:

```
∂L/∂s_i = Σ_j ΔNDCG_{ij} · σ(s_i - s_j)
```

- 학습 신호량 ∝ **N(N-1)/2 pairs per group**
- N=18: 153 pairs/timestep
- N=70: 2,415 pairs/timestep → **15.8× 학습 신호 밀도**

즉, ranker에 한해서는 breadth 효과가 √N이 아니라 **거의 N²에 가깝다**(Pareto 부분에서 reweighting되긴 함). 현재 `lgbm_lambdarank`(config.py:301)를 쓰면서 N=18로 묶어둔 것은 모델 선택의 자기모순.

### 1.4 López de Prado AFML — Boundary Universe Noise

`Advances in Financial Machine Learning` Ch. 7:

> "When training set membership coincides with portfolio membership, the boundary symbols (those near the inclusion threshold) inject distribution shift into the loss function. The model wastes capacity learning the boundary, not the alpha."

현재 Stage6의 K=18~24 hard cutoff는 임계점 부근(rank 15~25) 종목이 분기마다 진입/이탈 → 학습 분포가 분기별로 흔들림 → 모델은 "boundary discrimination"에 capacity를 소모. 학습 패널을 거래 패널보다 **충분히 넓게** 두는 것이 표준 해법.

### 1.5 Survivorship 회피 원칙 — PIT Ledger 활용

- 현재 `universe_ledger.parquet`는 분기별 Stage1~6 통과/탈락 이력을 기록 (`models.py:LedgerRow`).
- Stage1~5는 **PIT objective filters**(데이터 품질·유동성·리스크 이벤트 임계치)이므로 그 집합의 union은 survivorship-free.
- 분기별 Stage5 통과 집합의 시계열 union = **historical Stage5 union**. 이 집합은 미래 정보를 사용하지 않으면서 학습 패널을 최대로 확장.

이것이 `universe-fix.md`의 "Historical Stage6 union"보다 **이론적으로 우월**한 이유:
- Stage6는 `tradeable_score` 임계 컷오프 → 임계 인근 종목이 분기마다 진입/이탈 → boundary noise 그대로
- Stage5는 객관 임계 → boundary가 데이터 품질·유동성 절대값에 의해 결정되므로 분기 변동이 작음

### 1.6 "거래 가능" vs "수익 가능" — Stage6 점수 함수의 결함

현행 `tradeable_score`(selection.py:221):

```python
tradeable_score = 0.4·liq + 0.3·cost_inv + 0.2·quality + 0.1·stability
```

이는 **체결 마찰의 역수**일 뿐 **자산증식 기여**의 추정량이 아니다. 결정적으로 빠진 차원:

| 누락 차원 | 이론 | 자산증식 영향 |
|----------|------|--------------|
| **Idiosyncratic Volatility** | Markowitz(1952) — 동일 IC에서 dollar PnL ∝ vol | 저변동성 종목은 신호가 맞아도 PnL 작음 |
| **Cross-Sectional Dispersion Contribution** | Grinold-Kahn — effective N은 ENB(Effective Number of Bets) | 거대 코인 위주면 ENB ≈ 1 (사실상 단일 베팅) |
| **Regime Independence** | Jegadeesh-Titman(1993), Asness et al.(2013) | BTC와 100% 상관인 종목 추가는 breadth 기여 0 |
| **Funding Drag Stability** | Perpetual futures specific | 만성적 음수 funding 종목은 long position 점진 침식 |
| **HRP-style Cluster Representation** | López de Prado(2016) | 단일 클러스터(예: meme L1) 과대표현 시 drawdown correlated |

Stage6는 friction-only를 유지하면서 alpha-capacity·diversification 점수를 **별도 축으로 추가**해야 한다. 단일 점수에 합치면 trade-off가 가려진다(Pareto-optimal 해를 잃음).

---

## 2. 설계 결정 (Design Decisions)

### 2.1 D1. Universe 분리는 필연이다 (User Q1 답변)

**결론**: 추론 패널(C1)·집행 패널(C3)의 분리는 *option*이 아닌 **이론적 필수**. 단일 유니버스는 Pareto-dominated.

이 결정은 `phase2_alpha_universe_decoupling.md` §2.1과 동일하나, 본 명세는 더 나아가 **C1의 active_mask를 Stage6와 무관하게 Stage5 timeline으로 재정의**한다 — `universe-fix.md`가 우회한 "방향 A"의 정공법 구현.

### 2.2 D2. Stage6 점수 함수를 3축 다목표로 확장 (User Q2 답변)

**결론**: Stage6의 단일 `tradeable_score`(friction 100%)를 **3축 점수 + cluster-aware selection**으로 교체.

```
final_score(s) = w_F·friction_score(s) + w_A·alpha_capacity_score(s) + w_D·diversification_score(s)
```

| 축 | 가중치 (초기) | 구성 |
|----|--------------|------|
| Friction (`w_F` = 0.50) | 50% | 기존 `tradeable_score` 그대로 (0.4 liq + 0.3 cost_inv + 0.2 quality + 0.1 stability) |
| Alpha Capacity (`w_A` = 0.30) | 30% | 0.40·vol_ex_ante_z + 0.30·dispersion_contrib + 0.30·regime_independence |
| Diversification (`w_D` = 0.20) | 20% | cluster representation bonus (HRP-style) |

선발은 **점수 정렬 후 cluster cap**: 동일 correlation cluster에서 최대 `max_per_cluster=3` 종목만 채택, 이후 다음 cluster로.

`w_F`가 여전히 가장 크다 — execution feasibility는 self-evidence (못 거래하면 알파가 무의미). 단, `w_A`/`w_D`를 추가하여 "유동성 깊지만 알파 자산증식 기여 0"인 stable·high-cap·low-vol 종목의 과대표현을 방지.

**가중치 결정 절차 (Optuna 검증 필수)**:
- 초기값(0.50/0.30/0.20)은 이론적 prior.
- Optuna로 `w_F ∈ [0.40, 0.70]`, `w_A ∈ [0.15, 0.40]`, `w_D ∈ [0.10, 0.30]` (정규화) 탐색.
- objective: walk-forward OOS Calmar (집행 비용 차감 후).

### 2.3 D3. ML 학습 유니버스 = Historical Stage5 Union (User Q3 답변)

**결론**: 학습 패널은 **분기별 Stage5 통과 집합의 시계열 union**.

```python
historical_stage5_union = ⋃_{q ∈ training_window} stage5_passed(q)
```

| 비교 항목 | universe-fix.md "Historical Stage6" | 본 명세 "Historical Stage5" |
|----------|-------------------------------------|---------------------------|
| 출처 | Stage6 quarterly selected union | Stage5 quarterly passed union |
| 예상 규모 | 20~35 | 80~150 |
| Breadth | √N ≈ 5.4 | √N ≈ 9.5~12.2 |
| Pair count (ranker) | 380~595 | 3,160~11,175 |
| Boundary noise | Stage6 cutoff churn 그대로 | Stage5 hard cutoff (수치 임계) — 변동 작음 |
| Survivorship | OK (PIT) | OK (PIT) |
| 구현 비용 | active_mask 재구성 불필요 | active_mask **Stage5 기준 재구성 필수** ← 본 명세의 정공법 |

각 심볼의 `universe_active_mask[t]` = `True ⟺ symbol ∈ stage5_passed(quarter_of(t))`.

집행 마스킹은 별도: `trading_active_mask[t]` = `True ⟺ symbol ∈ stage6_selected(quarter_of(t))`. 알파 발산은 C1·C2 전체로, 진입은 C3만.

### 2.4 D4. Sample Weighting의 3중 보정 (P2 — Time-Decay & Diversification)

학습 데이터 가중치를 다음으로 확장:

```
w_{i,t} = base · time_decay · quality · cluster_balance
```

| 항 | 식 | 근거 |
|---|---|------|
| `base` | `1 + 2·|y_ev_i,t|` (기존 유지) | extreme observations 강조 |
| `time_decay` | `exp(-λ·(T-t))`, `λ = ln2/halflife_bars` | **P2**: López de Prado AFML Ch.6, 비정상성 보정 |
| `quality` | `coverage_60d_i,t` (clipped [0.5, 1.0]) | 저품질 심볼 영향 제한, Stage5 확대의 안전장치 |
| `cluster_balance` | `1 / √cluster_size_t` | 거대 cluster(예: ETH-correlated 30종) 과대표현 차단 |

**근거**: 초기값(halflife=12개월)은 경험적 prior. López de Prado는 비정상 시계열에 시간 감쇠 가중치가 표준 보정이라고 제시. 24개월 IS에서 오래된 데이터와 최근 데이터가 동일 가중이면 regime shift를 학습하지 못함 → OOS 성능 악화.

**핵심**: Stage5 전체로 확장하면 데이터 품질·cluster 다양성이 떨어질 수 있으므로 이 가중치들이 **확장의 safety net**.

### 2.5 P1 + P2 통합 수용 결정 (Adoption Decision)

| Priority | 작업 | 이론 기반 | 기대 효과 | 상태 |
| --- | --- | --- | --- | --- |
| **P1** | Universe-Training Decoupling + 3-Cohort | Grinold-Kahn FL, breadth × √N, boundary noise | net_ic +50~100%, breadth +200~300% | **This spec** |
| **P2** | Exponential Time-Decay Sample Weighting | López de Prado FML Ch.6, concept drift | regime-shift robust IS, OOS bias 감소 | §2.4 integrated |

P1과 P2는 **독립적이지 않다**. P1(breadth 확장)을 할 때 Stage5 전체를 포함하면 과거 품질·regime 변동이 크므로, P2(time-decay 가중치)가 필수 안정장치. 둘을 함께 실장해야 개선 효과를 정확히 측정할 수 있다.

**보류된 작업**:
- **P3 Multi-Horizon Stack** (h=6, 12, 18 동시): 분산 감소 가능하나, P1+P2 효과 입증 후 별도 트랙.
- **P4 Huber/Quantile Loss**: 효과 가능하나 `lgbm_huber` 모드는 별도 검증. P1+P2 후 A/B 실험.
- **P5 신규 특징** (DeFi TVL, Stablecoin Supply): 데이터 파이프라인 변경 비용 > 이득 (우선순위 낮음).
- **P6 Ensemble/Stacking**: Phase 2 (M1) 실패 교훈 — 모델 복잡도 증가는 영구 금지.

---

## 3. 아키텍처 (3-Cohort Model)

```
┌────────────────────────────────────────────────────────────────────────┐
│ COHORT 1 — INFERENCE (Historical Stage5 Union)                         │
│   • Symbols: ⋃_q stage5_passed(q)  for q in [IS_start, OOS_end]        │
│   • Size: 80~150 (예상)                                                  │
│   • active_mask[s,t]: True ⟺ s ∈ stage5_passed(quarter_of(t))          │
│   • 데이터 로드: 전체 시계열 (delisted 포함)                              │
│   • 용도: feature 계산, label 생성, ranker 학습, IC 측정                 │
│                                                                          │
│   ↓ 모델 예측 (alpha_long, alpha_short for all C1 symbols)              │
│                                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ COHORT 2 — LIVE INFERENCE (Current Stage5 Passed)                      │
│   • Symbols: stage5_passed(current_quarter)                            │
│   • Size: 50~80                                                         │
│   • 용도: 신호 발산 (alpha emission) — 미래 거래 후보 후보군              │
│   • 통계: net_ic_live, breadth_live (training panel과 일관성 점검)       │
│                                                                          │
│   ↓ tradability filter (multi-objective Stage6)                         │
│                                                                          │
├────────────────────────────────────────────────────────────────────────┤
│ COHORT 3 — EXECUTION (Enhanced Stage6 Selected)                        │
│   • Symbols: stage6_selected(current_quarter, multi_objective_score)   │
│   • Size: 18~24 (k_in)                                                  │
│   • 점수: w_F·friction + w_A·alpha_capacity + w_D·diversification       │
│   • 선발: cluster-aware (max_per_cluster=3, anchors forced)            │
│   • 용도: 실제 포지션 진입 — 알파 마스킹 `trading_mask`                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Data Flow

```
build_universe_timeline()
  └─► for each quarter q:
      ├─► run pipeline Stage1~5: stage5_passed(q)  ─────┐
      └─► run Stage6 (multi-obj): stage6_selected(q) ───┤
                                                          │
  ⋃_q stage5_passed(q) ──► UniverseSnapshot.inference_panel  (C1)
  ⋃_q stage6_selected(q) ──► UniverseSnapshot.historical_trading_panel  (보조)
  stage5_passed(OOS_q) ──► UniverseSnapshot.live_inference_panel  (C2)
  stage6_selected(OOS_q) ──► UniverseSnapshot.selected  (C3, 기존 필드)

run_active_strategy_output_bridge(
    symbols=C1.inference_panel,         # 데이터 로드 + 학습
    training_symbols=C1.inference_panel,
    trading_symbols=C3.selected,         # 알파 마스킹
)

ml_builder:
  ├─ training: aligned panel = C1 (active_mask = Stage5 timeline)
  ├─ inference: alpha = predict(C1)
  ├─ masking: alpha_long[:, ¬trading_mask] = 0  (C3 only)
  └─ evaluation: panel="inference" (C1 stat) + panel="trading" (C3 stat)
```

---

## 4. Contract (Data Structures)

### 4.1 `UniverseSnapshot` 확장 (`src/domain/futures/universe/models.py`)

```python
@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    as_of: date
    selected: tuple[SymbolMeta, ...]                                        # C3 (기존)
    rejected: dict[str, FilterReport]                                       # 기존
    config_hash: str                                                        # 기존
    # ── 신규 필드 (frozenset for set ops, tuple for order-stable iteration) ──
    inference_panel: tuple[str, ...] = field(default_factory=tuple)         # C1 — historical Stage5 union
    live_inference_panel: tuple[str, ...] = field(default_factory=tuple)    # C2 — current Stage5 passed
    historical_trading_panel: tuple[str, ...] = field(default_factory=tuple)# 보조 — historical Stage6 union (참고/마이그레이션용)
    # ── 메타 (멱등성·재현성) ──
    inference_panel_quarter_membership: dict[date, tuple[str, ...]] = field(default_factory=dict)
    # 분기별 Stage5 통과 집합 (active_mask 계산용 권위 데이터)
```

`inference_panel_quarter_membership`이 **C1의 active_mask 재구성을 위한 SSOT**. 기존 Stage6 timeline은 그대로 두고 별도 timeline을 추가한다(C3 집행 로직 영향 0).

### 4.2 `MembershipMaskBundle` 확장 (`src/domain/futures/universe/membership.py`)

기존 `MembershipMaskBundle`은 단일 timeline 기반이다. 두 종류의 mask를 동시에 노출:

```python
@dataclass(slots=True, frozen=True)
class MembershipMaskBundle:
    # ── 기존 (C3 trading용) ──
    universe_active_mask: np.ndarray            # = trading_active_mask (Stage6 timeline)
    universe_entry_warm_mask: np.ndarray
    membership_kill_signal: np.ndarray
    entry_block_mask: np.ndarray
    kill_signal: np.ndarray
    # ── 신규 (C1 inference용) ──
    inference_active_mask: np.ndarray           # Stage5 timeline 기반
    inference_entry_warm_mask: np.ndarray
```

`inject_membership_masks_into_maps`는 두 timeline을 받아 두 종류 mask를 모두 frame에 주입.

```python
def inject_membership_masks_into_maps(
    *,
    data_maps: dict, oos_data_maps: dict, symbols: list[str], tf: str,
    trading_timeline: Mapping[date, frozenset[str]],       # Stage6 (기존 timeline)
    inference_timeline: Mapping[date, frozenset[str]],     # Stage5 (신규)
    warmup_bars_required: int,
) -> None
```

### 4.3 `StrategyMLConfig` 확장 (`src/domain/futures/strategy/config.py`)

```python
training_universe_scope: Literal[
    "stage5_passed",              # 기존 — 단일 quarter Stage5 (현행 P1)
    "stage6_selected",            # 기존 — 단일 quarter Stage6 (회귀 테스트)
    "historical_stage6",          # universe-fix.md 호환 (deprecated 예정)
    "historical_stage5_union",    # ← 신규 기본값 (C1)
] = "historical_stage5_union"

# C1 active_mask 사용 여부 — True면 inference_active_mask, False면 universe_active_mask
use_inference_active_mask: bool = True

# Stage6 다목표 점수 가중치 (sum = 1.0)
stage6_weight_friction: float = 0.50
stage6_weight_alpha_capacity: float = 0.30
stage6_weight_diversification: float = 0.20
stage6_max_per_cluster: int = 3

# Sample weighting 보정
sample_weight_quality_clip_min: float = 0.50         # quality factor의 하한
sample_weight_cluster_balance_enabled: bool = True   # cluster_balance 곱셈 활성화
# 기존 sample_weight_time_decay_halflife_bars는 그대로
```

`__post_init__`:

```python
if self.training_universe_scope not in {
    "stage5_passed", "stage6_selected", "historical_stage6", "historical_stage5_union"
}:
    raise ValueError(...)
weights_sum = self.stage6_weight_friction + self.stage6_weight_alpha_capacity + self.stage6_weight_diversification
if not math.isclose(weights_sum, 1.0, abs_tol=1e-6):
    raise ValueError(f"stage6 weights must sum to 1.0 (got {weights_sum})")
if not all(0.0 <= w <= 1.0 for w in (self.stage6_weight_friction,
                                      self.stage6_weight_alpha_capacity,
                                      self.stage6_weight_diversification)):
    raise ValueError("stage6 weights must each be in [0, 1]")
if self.stage6_max_per_cluster < 1:
    raise ValueError("stage6_max_per_cluster must be >= 1")
if not (0.0 < self.sample_weight_quality_clip_min <= 1.0):
    raise ValueError("sample_weight_quality_clip_min must satisfy 0 < x <= 1")
```

### 4.4 `Stage6Config` 확장 (`src/domain/futures/universe/config.py`)

```python
@dataclass(frozen=True, slots=True)
class Stage6Config:
    # 기존
    k_in: int = 20
    k_out: int = 35
    min_dwell_days: int = 90
    anchor_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    basket_ref: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    basket_weights: tuple[float, ...] = (0.45, 0.25, 0.08)
    corr_cluster_threshold: float = 0.70
    # ── 신규 (다목표 점수) ──
    weight_friction: float = 0.50
    weight_alpha_capacity: float = 0.30
    weight_diversification: float = 0.20
    max_per_cluster: int = 3
    # alpha_capacity 하위 가중치
    capacity_w_volatility: float = 0.40
    capacity_w_dispersion: float = 0.30
    capacity_w_regime_independence: float = 0.30
    # ex-ante volatility 측정 윈도우
    vol_lookback_bars: int = 540   # 4h * 90d
    # regime independence: β_vs_market의 |β - 1| z-score
```

### 4.5 `UniverseTimelineResult` 확장 (`src/application/futures/optimization/universe_service.py`)

```python
@dataclass(frozen=True, slots=True)
class UniverseTimelineResult:
    symbols: tuple[str, ...]                       # 기존: ⋃ Stage6 selected (C3 union)
    timeline: UniverseMembershipTimeline           # 기존: Stage6 timeline
    snapshots: tuple[UniverseSnapshot, ...]
    snapshot: UniverseSnapshot                     # OOS snapshot
    report: pd.DataFrame
    # ── 신규 ──
    inference_symbols: tuple[str, ...] = ()        # C1: ⋃ Stage5 passed (sorted)
    inference_timeline: UniverseMembershipTimeline | None = None  # Stage5 quarterly timeline
    inference_panel_quarter_membership: dict[date, frozenset[str]] = field(default_factory=dict)
```

---

## 5. Surgical Plan

### Phase A — Universe Pipeline (C1 데이터 노출)

#### A1. `[src/domain/futures/universe/pipeline.py:466 build_universe]` `[REPLACE]`

Stage5 통과 frame을 Stage6 호출 직전 capture, snapshot에 노출.

```python
# 의사코드 — 현재 build_universe 내부 흐름 보완
stage5_passed_frame = apply_risk_events_stage(...)  # 기존
stage5_passed_symbols = tuple(sorted(stage5_passed_frame["symbol"].dropna().astype(str).tolist()))

selected_frame, stage6_report = apply_selection_stage(stage5_passed_frame, config=cfg.stage6, ...)

snapshot = UniverseSnapshot(
    as_of=as_of_date,
    selected=tuple(symbol_metas),
    rejected=rejected_map,
    config_hash=config_hash,
    live_inference_panel=stage5_passed_symbols,    # C2 (current quarter)
    # inference_panel과 inference_panel_quarter_membership은 timeline 빌더가 채움
)
```

**불변성**: `apply_selection_stage`는 변경하지 않는다(Phase B에서 별도 작업).

#### A2. `[src/application/futures/optimization/universe_service.py:100 discover_universe_timeline]` `[REPLACE]`

각 분기 snapshot에서 `live_inference_panel`을 수집 → quarterly membership으로 적재.

```python
inference_panel_quarter_membership: dict[date, frozenset[str]] = {}
inference_symbols_set: set[str] = set()

for quarter_start, snapshot, _report in snapshots_by_quarter:
    quarter_stage5 = frozenset(snapshot.live_inference_panel)
    inference_panel_quarter_membership[quarter_start] = quarter_stage5
    inference_symbols_set.update(quarter_stage5)

# Stage5 timeline 빌드 (Stage6 timeline과 동일 구조)
inference_windows: list[UniverseMembershipWindow] = []
prev_stage5: frozenset[str] = frozenset()
for idx, (q_start, _snap, _) in enumerate(snapshots_by_quarter):
    members = inference_panel_quarter_membership[q_start]
    next_start = (pd.Timestamp(snapshots_by_quarter[idx+1][0])
                  if idx+1 < len(snapshots_by_quarter) else None)
    inference_windows.append(UniverseMembershipWindow(
        effective_from=pd.Timestamp(q_start),
        effective_to=next_start,
        snapshot_as_of=_snap.as_of,
        active_symbols=tuple(sorted(members)),
        entry_symbols=tuple(sorted(members - prev_stage5)),
        exit_symbols=tuple(sorted(prev_stage5 - members)),
    ))
    prev_stage5 = members

inference_timeline = UniverseMembershipTimeline(tf=tf, windows=tuple(inference_windows))

# OOS snapshot에 C1 union 주입 (frozen → replace)
from dataclasses import replace
oos_snapshot = replace(
    oos_snapshot,
    inference_panel=tuple(sorted(inference_symbols_set)),
    inference_panel_quarter_membership={
        k: tuple(sorted(v)) for k, v in inference_panel_quarter_membership.items()
    },
    # universe-fix 호환: historical_trading_panel = ⋃ Stage6 selected (기존 all_symbols)
    historical_trading_panel=tuple(sorted(all_symbols)),
)

return UniverseTimelineResult(
    symbols=tuple(sorted(all_symbols)),
    timeline=UniverseMembershipTimeline(tf=tf, windows=tuple(windows)),  # Stage6 (기존)
    snapshots=tuple(snap for _, snap, _ in snapshots_by_quarter),
    snapshot=oos_snapshot,
    report=oos_report,
    inference_symbols=tuple(sorted(inference_symbols_set)),
    inference_timeline=inference_timeline,
    inference_panel_quarter_membership={
        k: frozenset(v) for k, v in inference_panel_quarter_membership.items()
    },
)
```

#### A3. `[src/domain/futures/universe/storage.py snapshot_to_payload/from_payload]` `[ADD]`

`snapshot_to_payload`:

```python
"inference_panel": list(snapshot.inference_panel),
"live_inference_panel": list(snapshot.live_inference_panel),
"historical_trading_panel": list(snapshot.historical_trading_panel),
"inference_panel_quarter_membership": {
    qd.isoformat(): list(syms)
    for qd, syms in snapshot.inference_panel_quarter_membership.items()
},
```

`snapshot_from_payload`:

```python
inference_panel=tuple(str(s) for s in payload.get("inference_panel", [])),
live_inference_panel=tuple(str(s) for s in payload.get("live_inference_panel", [])),
historical_trading_panel=tuple(str(s) for s in payload.get("historical_trading_panel", [])),
inference_panel_quarter_membership={
    date.fromisoformat(k): tuple(str(s) for s in v)
    for k, v in payload.get("inference_panel_quarter_membership", {}).items()
},
```

### Phase B — Membership Mask 이중화 (C1·C3 분리)

#### B1. `[src/domain/futures/universe/membership.py build_membership_mask_bundle]` `[REPLACE]`

두 timeline을 받고 두 종류 mask 모두 반환:

```python
def build_membership_mask_bundle(
    *,
    datetimes: pd.Series,
    symbol: str,
    trading_timeline: Mapping[date, frozenset[str] | set[str]],     # Stage6
    inference_timeline: Mapping[date, frozenset[str] | set[str]],   # Stage5 (None이면 trading과 동일)
    warmup_bars_required: int,
    raw_kill_signal: np.ndarray | None = None,
) -> MembershipMaskBundle:
    # 기존 로직을 trading_timeline에 적용 → universe_active_mask, ...
    # 동일 로직을 inference_timeline에 적용 → inference_active_mask, inference_entry_warm_mask
    # inference에 한해 kill_signal·entry_block은 적용하지 않음 (학습 마스킹 외 use case 없음)
```

기존 호출자는 `inference_timeline=trading_timeline`(또는 `None`)으로 호출 시 backward-compatible.

#### B2. `[src/domain/futures/universe/membership.py inject_membership_masks_into_maps]` `[REPLACE]`

`trading_timeline`, `inference_timeline` 두 인자 받고 frame에 두 mask 모두 주입:

```python
frame.loc[:, "universe_active_mask"] = bundle.universe_active_mask          # 기존
frame.loc[:, "universe_entry_warm_mask"] = bundle.universe_entry_warm_mask
frame.loc[:, "membership_kill_signal"] = bundle.membership_kill_signal
frame.loc[:, "entry_block_mask"] = bundle.entry_block_mask
frame.loc[:, "kill_signal"] = bundle.kill_signal
frame.loc[:, "inference_active_mask"] = bundle.inference_active_mask        # 신규
frame.loc[:, "inference_entry_warm_mask"] = bundle.inference_entry_warm_mask
```

### Phase C — Strategy Service: C1 학습 + C3 집행

#### C1. `[src/application/futures/optimization/strategy_service.py run_active_strategy_output_bridge]` `[REPLACE]`

```python
def run_active_strategy_output_bridge(
    *,
    run_config: FuturesRunConfig,
    symbols: list[str],                              # C3 (기존, 호환 유지)
    tf: str,
    fetch_start: str | None,
    end_date: str | None,
    opt_config: dict[str, Any],
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    inference_panel: tuple[str, ...] | None = None,           # C1
    live_inference_panel: tuple[str, ...] | None = None,      # C2 (선택, 평가용)
    trading_symbols: tuple[str, ...] | None = None,           # C3
    historical_trading_panel: tuple[str, ...] | None = None,  # universe-fix 호환
) -> MLPipelineOutput:
    ...
    strategy_cfg = StrategyConfig(name=run_config.strategy)
    ml_scope = strategy_cfg.ml.training_universe_scope

    if ml_scope == "historical_stage5_union" and inference_panel:
        effective_symbols = list(inference_panel)
    elif ml_scope == "historical_stage6" and historical_trading_panel:
        effective_symbols = list(historical_trading_panel)
    elif ml_scope == "stage5_passed" and live_inference_panel:
        effective_symbols = list(live_inference_panel)
    else:  # stage6_selected
        effective_symbols = symbols

    # trading_mask 전달용 — C3에서만 알파 발산
    from dataclasses import replace
    if trading_symbols:
        updated_ml = replace(strategy_cfg.ml, trading_symbols=tuple(trading_symbols))
        strategy_cfg = replace(strategy_cfg, ml=updated_ml)

    return run_ml_pipeline_for_universe(
        symbols=effective_symbols,
        tf=tf, fetch_start=fetch_start, end_date=end_date,
        opt_config=opt_config, strategy_cfg=strategy_cfg,
        preloaded_data_maps=preloaded_data_maps,
    )
```

#### C2. `[src/execution/opt_main_futures.py 또는 application/runner]` `[REPLACE]`

`UniverseTimelineResult` 결과를 위 함수에 매핑:

```python
result = discover_universe_timeline(...)
ml_out = run_active_strategy_output_bridge(
    run_config=run_config,
    symbols=tuple(meta.symbol for meta in result.snapshot.selected),  # C3
    tf=tf,
    ...,
    inference_panel=result.snapshot.inference_panel,                  # C1
    live_inference_panel=result.snapshot.live_inference_panel,        # C2
    trading_symbols=tuple(meta.symbol for meta in result.snapshot.selected),
    historical_trading_panel=result.snapshot.historical_trading_panel,
)
```

#### C3. `[Membership 주입 지점]` `[REPLACE]`

`inject_membership_masks_into_maps` 호출 지점(opt_main_futures 또는 strategy 데이터 로더):

```python
inject_membership_masks_into_maps(
    data_maps=data_maps,
    oos_data_maps=oos_data_maps,
    symbols=loaded_symbols,                                # = inference_panel
    tf=tf,
    trading_timeline=result.timeline.as_mapping(),         # Stage6 (기존)
    inference_timeline=result.inference_timeline.as_mapping(),  # Stage5 (신규)
    warmup_bars_required=warmup,
)
```

### Phase D — ML Builder: active_mask 출처 분기

#### D1. `[src/domain/futures/strategy/ml_builder.py 학습 데이터 마스킹부]` `[REPLACE]`

학습 시 `active_mask` 선택을 `use_inference_active_mask`로 분기:

```python
# 학습 row eligibility 계산 부근 (기존 universe_active_mask 사용처)
mask_col = (
    "inference_active_mask"
    if ml_cfg.use_inference_active_mask and "inference_active_mask" in frame.columns
    else "universe_active_mask"
)
eligible = frame[mask_col].to_numpy(dtype=bool) & ~frame["entry_block_mask"].to_numpy(dtype=bool)
# 단, entry_block_mask는 trading용이므로 inference 학습 시에는 별도 검토:
# inference 시에는 inference_entry_warm_mask 사용
```

`trading_mask` 적용은 **alpha 출력 직전**만(기존 P1과 동일):

```python
trading_keys = set(_effective_trading or ())
trading_mask_array = np.array(
    [sym in trading_keys for sym in aligned_symbols], dtype=np.bool_,
)
alpha_long_final[:, ~trading_mask_array] = 0.0
alpha_short_final[:, ~trading_mask_array] = 0.0
```

(이미 ml_builder.py:1113, :1578에 골격 존재 — 동작 보존).

#### D2. `[src/domain/futures/strategy/labels.py:252 sample_weight]` `[REPLACE]`

```python
# 기존
sample_weight = (original_weight * (1.0 + 2.0 * y_ev_abs))

# 신규
sample_weight = (original_weight * (1.0 + 2.0 * y_ev_abs))

# Quality factor (frame에 coverage_60d 필요 — Stage5 통과 기준 0.95 ≤ × ≤ 1.0이지만 fallback)
if "coverage_60d" in features.columns:
    quality = features["coverage_60d"].to_numpy(dtype=np.float32)
    quality = np.clip(quality, cfg.sample_weight_quality_clip_min, 1.0)
    # quality는 [N_obs] 1D → broadcast 가능하도록 처리 후 sample_weight에 곱
    sample_weight = sample_weight * quality.reshape(-1, 1)  # shape 맞춤

# Cluster balance (cluster_id가 features에 있을 때만)
if cfg.sample_weight_cluster_balance_enabled and "cluster_id" in features.columns:
    cluster_ids = features["cluster_id"].to_numpy(dtype=np.int32)
    # 각 (t, cluster) 에서 size 계산 → 1/sqrt(size)
    # 단순화: 전체 학습셋의 cluster size로 정적 가중치 (per-bar 동적은 비용 큼)
    unique, counts = np.unique(cluster_ids[cluster_ids >= 0], return_counts=True)
    size_map = dict(zip(unique.tolist(), counts.tolist(), strict=False))
    cluster_w = np.array(
        [1.0 / math.sqrt(size_map.get(int(c), 1)) if c >= 0 else 1.0 for c in cluster_ids],
        dtype=np.float32,
    )
    sample_weight = sample_weight * cluster_w.reshape(-1, 1)

# Time decay (기존 P2 작업 — phase2_alpha_universe_decoupling.md §3.2와 동일)
if cfg.sample_weight_time_decay_halflife_bars is not None:
    hl = float(cfg.sample_weight_time_decay_halflife_bars)
    lam = math.log(2.0) / max(hl, 1.0)
    T = signed.shape[0]
    time_decay = np.exp(-lam * (T - 1 - np.arange(T, dtype=np.float64))).astype(np.float32)
    sample_weight = (sample_weight.T * time_decay).T
```

**주의**: 위 코드는 features의 shape에 따라 axis 정합성 검토 필수 (`labels.py`의 `signed`/`features` 실제 shape에 맞춰 implement 단계에서 보정).

### Phase E — Stage 6 다목표 점수

#### E1. `[src/domain/futures/universe/selection.py apply_selection_stage]` `[REPLACE]`

`tradeable_score` 계산 후 다음을 추가:

```python
# 1) Friction score (기존 tradeable_score를 friction_score로 rename)
friction_score = (
    DEFAULT_W_LIQ * liq_norm
    + DEFAULT_W_COST_INV * cost_inv_norm
    + DEFAULT_W_QUALITY * quality_norm
    + DEFAULT_W_STABILITY * stability_norm
)
out["friction_score"] = friction_score

# 2) Alpha capacity score
# 2-a) ex-ante volatility: out["vol_30d"] 또는 return_vector로부터 σ 추정
vol_series = pd.to_numeric(
    out.get("vol_30d", pd.Series(np.nan, index=out.index)), errors="coerce",
)
vol_z = _zscore(vol_series.fillna(vol_series.median()))
vol_norm = _normalize_unit(vol_z)  # [0,1]로 압축

# 2-b) dispersion contribution: |β_vs_market - 1|이 클수록 distinct
beta = pd.to_numeric(out.get("beta_vs_market", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
dispersion_norm = _normalize_unit((beta - 1.0).abs())

# 2-c) regime independence: anchor와의 상관 < threshold인 정도
# CORR_CLUSTER_THRESHOLD 미만 cluster에 속한 종목에 가중 부여
cluster_ids = out["cluster_id"]
anchor_clusters = set(
    out.loc[out["_symbol_key"].isin(set(_symbol_key(a) for a in ANCHORS)), "cluster_id"].tolist()
)
regime_indep = (~cluster_ids.isin(anchor_clusters)).astype(float)

alpha_capacity = (
    cfg.capacity_w_volatility * vol_norm
    + cfg.capacity_w_dispersion * dispersion_norm
    + cfg.capacity_w_regime_independence * regime_indep
)
out["alpha_capacity_score"] = alpha_capacity

# 3) Diversification score: cluster size의 역수
cluster_size = out["cluster_id"].map(out["cluster_id"].value_counts().to_dict()).fillna(1)
diversification = 1.0 / np.sqrt(cluster_size.astype(float))
out["diversification_score"] = _normalize_unit(diversification)

# 4) Combined final score
out["tradeable_score"] = (  # 이름 유지 (호환), 의미는 multi-objective 합
    cfg.weight_friction * friction_score
    + cfg.weight_alpha_capacity * alpha_capacity
    + cfg.weight_diversification * out["diversification_score"]
)
```

#### E2. `[src/domain/futures/universe/selection.py apply_selection_stage]` `[REPLACE]` (cluster cap)

`out = out.sort_values("tradeable_score", ascending=False)` 직후, anchor 강제 전:

```python
# Cluster cap: 동일 cluster에서 max_per_cluster 초과 선택 금지
max_per_cluster = int(cfg.max_per_cluster)
cluster_counts: dict[int, int] = {}
selected_indices: list[int] = []
for idx in out.index:
    cid = int(out.at[idx, "cluster_id"])
    if cid < 0:
        # 비클러스터(-1)는 cap 미적용 (개별 종목)
        selected_indices.append(idx)
    else:
        if cluster_counts.get(cid, 0) < max_per_cluster:
            selected_indices.append(idx)
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    if len(selected_indices) >= max_symbols * 2:  # buffer for anchor forcing
        break

out = out.loc[selected_indices].copy()  # cluster cap 적용된 후보 풀
out["rank"] = np.arange(1, len(out) + 1)
# 이후 기존 `selected = out.head(max_symbols).copy()` 흐름 유지
```

### Phase F — Alpha Evaluation Panel 분리

#### F1. `[src/domain/futures/strategy/alpha_evaluation.py AlphaEvaluationReport]` `[REPLACE]`

`panel: Literal["inference", "trading"]` 인자 추가, 두 패널 모두 계산:

```python
# net_ic, ic_t_stat_nw, effective_breadth는 patient subset에 따라 분리
inference_metrics = compute_alpha_metrics(alpha_panel, panel="inference")  # C1 전체
trading_metrics = compute_alpha_metrics(alpha_panel, panel="trading")       # trading_mask True만
```

리포트 dict에 `metrics_by_panel: {"inference": {...}, "trading": {...}}` 추가. 게이팅은 **inference panel 기준**(N=70 통계량) — 기존 임계값(`ic_t_stat ≥ 2.5`, `breadth ≥ 3.0`)을 *inference에 한정 적용*하면서 trading panel은 정보 표시.

---

## 6. 알고리즘 의사코드 (Stage6 Multi-Objective Selection)

```text
INPUT: stage5_passed_frame (columns: symbol, adv_usdt_median, execution_cost_bps,
                                     last_60d_coverage, listing_age_days, vol_30d,
                                     beta_vs_market, return_vector or precomputed cluster_id)
       cfg: Stage6Config
       anchors = (BTCUSDT, ETHUSDT)
       previous_selection (optional)

ALGORITHM:
  1. CLUSTER_ASSIGN:
       cluster_id = compute_correlation_clusters(return_vector, threshold=cfg.corr_cluster_threshold)
       # 기존 _compute_cluster_ids 재사용

  2. SCORE_FRICTION:
       friction = 0.4·norm(adv) + 0.3·norm(1/cost) + 0.2·norm(coverage) + 0.1·norm(age)

  3. SCORE_ALPHA_CAPACITY:
       vol_n = normalize_unit(vol_30d)
       disp_n = normalize_unit(|beta - 1|)
       indep = 1 if cluster_id ∉ anchor_clusters else 0
       capacity = 0.40·vol_n + 0.30·disp_n + 0.30·indep

  4. SCORE_DIVERSIFICATION:
       cluster_size[cid] = count(cluster_id == cid)
       div = normalize_unit(1 / sqrt(cluster_size[cluster_id]))

  5. COMBINE:
       score = w_F·friction + w_A·capacity + w_D·div

  6. CLUSTER_CAP_SELECTION:
       sorted_symbols = argsort(score, desc)
       chosen = []
       cluster_count = defaultdict(int)
       for s in sorted_symbols:
           cid = cluster_id[s]
           if cid < 0 or cluster_count[cid] < cfg.max_per_cluster:
               chosen.append(s)
               cluster_count[cid] += 1
           if len(chosen) >= 2 * k_in:  # buffer
               break

  7. HYSTERESIS (기존 로직 유지):
       apply k_in / k_out / min_dwell_days using `previous_selection`

  8. ANCHOR_FORCING (기존 로직 유지):
       ensure BTCUSDT, ETHUSDT ∈ chosen (synthetic row if absent)

  9. FINALIZE:
       selected = chosen[:k_in]  # 또는 max_symbols
       report = build_stage6_report(...)

OUTPUT: (selected_frame, report)
```

**계산 복잡도**:
- `compute_correlation_clusters`: O(N²·T) where T = return window length — 기존과 동일
- Score computations: O(N) per axis — 무시할 수준
- Cluster cap selection: O(N·log N) for sort + O(N) for loop

전체 비용 증가는 5% 미만 추정.

---

## 7. 검증 기준 (Acceptance Criteria)

### 7.1 단위 테스트

| 테스트 | 파일 | 핵심 assertion |
|--------|------|--------------|
| `test_universe_snapshot_inference_panel` | `tests/unit/domain/futures/universe/test_models.py` | `snapshot.inference_panel` ⊇ `snapshot.selected` |
| `test_membership_dual_mask` | `tests/unit/domain/futures/universe/test_membership.py` | `inference_active_mask` ⊇ `universe_active_mask` (per bar) |
| `test_stage6_cluster_cap` | `tests/unit/domain/futures/universe/test_selection.py` | 동일 cluster에서 `max_per_cluster` 초과 선택 없음 |
| `test_stage6_multi_objective_weights` | `tests/unit/domain/futures/universe/test_selection.py` | 가중치 변경 시 점수 정렬 변화 검증 |
| `test_strategy_service_inference_panel_branch` | `tests/unit/application/futures/optimization/test_strategy_service.py` | `historical_stage5_union` 분기 시 `effective_symbols == inference_panel` |
| `test_sample_weight_quality_clip` | `tests/unit/domain/futures/strategy/test_labels.py` | `coverage_60d < min` 행이 `min` 이상으로 clipped |
| `test_sample_weight_cluster_balance` | `tests/unit/domain/futures/strategy/test_labels.py` | 큰 cluster 가중 < 작은 cluster 가중 |
| `test_alpha_evaluation_dual_panel` | `tests/unit/domain/futures/strategy/test_alpha_evaluation.py` | `metrics_by_panel`에 `inference`·`trading` 모두 존재 |

### 7.2 OOS 실측 (집행 비용 차감 후)

| 지표 | 현재 (alpha0.md) | universe-fix.md 목표 | **본 명세 목표** |
|------|-----------------|----------------------|-----------------|
| `inference_symbols` (C1) | 9 | 20~35 | **80~150** |
| `effective_breadth` (inference) | 2.88 | ≥ 6.0 | **≥ 8.0** |
| `eligible` (학습 행 비율) | 0.91 (Stage6) / 0.29 (Stage5 buggy) | ≥ 0.70 | **≥ 0.85** (Stage5 mask 정렬 후) |
| `net_ic` (inference panel) | 0.0021 | ≥ 0.008 | **≥ 0.010** |
| `ic_t_stat_nw` (inference) | 1.07 | ≥ 2.50 | **≥ 3.00** |
| `net_ic` (trading panel C3) | n/a | n/a (미정의) | **≥ 0.006** (집행 가능 알파) |
| `chop IC × √breadth` (trading) | 0.0148 | ≥ 0.0250 | **≥ 0.0300** |
| Walk-forward OOS Calmar (집행 비용 후) | (alpha0.md) | (측정 안 됨) | **≥ 0.6** (현재 대비 +50%) |

### 7.3 Quant Integrity 게이트 (Mandatory)

| 항목 | 검증 명령 / 기준 |
|------|---------------|
| Look-ahead | `inference_panel` 멤버십이 분기 boundary에서만 변화, `inference_active_mask`가 `as_of` 이후 데이터 사용 없음 |
| Survivorship | C1에 delisted 심볼 포함 (분기별 active=True/False 토글) — `test_inference_panel_includes_delisted` |
| Walk-forward | 기존 fold 구조 유지 (`embargo_bars ≥ label_horizon_bars`), `inference_active_mask`가 purge 영역에서 False |
| Cluster diversity | 선발된 C3의 ENB ≥ 5 (`ENB = (Σw)² / Σw² with equal w = N/(1 + (N-1)·avg_corr)`) |

### 7.4 검증 명령

```bash
# 단위 테스트
uv run pytest tests/unit/domain/futures/universe/ tests/unit/domain/futures/strategy/ \
    tests/unit/application/futures/optimization/ --tb=short

# Forecast layer 통합
uv run pytest tests/unit/domain/futures/forecast/ --tb=short

# Alpha 진단 모드 (alpha2.md 신규 작성)
PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
    --mode alpha --skip-universe --skip-data-sync \
    --strategy ml_lambdamart_v1

# Stage6 점수 분포 진단 (신규 스크립트)
PYTHONPATH=. uv run python -m src.execution.diagnose_stage6_scores \
    --output docs/results/stage6_score_distribution.md
```

---

## 8. 작업 순서 (Sequencing)

| Step | Phase | 작업 | 검증 |
|------|-------|------|------|
| 1 | A | `UniverseSnapshot` 필드 추가 + `Stage6Config` 가중치 필드 추가 + `StrategyMLConfig` Literal 확장 | `pytest tests/unit/domain/futures/universe/test_models.py test_config.py` |
| 2 | A | `build_universe`가 `live_inference_panel` 채움 + `discover_universe_timeline`이 `inference_*` 노출 | `test_quarterly_selection_audit.py` 회귀 + 신규 `test_inference_panel_aggregation` |
| 3 | A | `storage.py` 직렬화 round-trip | `test_storage_roundtrip` 신규 |
| 4 | B | `MembershipMaskBundle` 이중화 + `build_membership_mask_bundle` dual timeline | `test_membership_dual_mask` 신규 |
| 5 | B | `inject_membership_masks_into_maps` dual timeline + 호출 지점 마이그레이션 | 통합 테스트 |
| 6 | C | `run_active_strategy_output_bridge`에 4분기 분기 + `opt_main_futures` 연결 | `test_strategy_service_inference_panel_branch` |
| 7 | D | `ml_builder.py` active_mask 출처 분기 | 회귀 + `test_use_inference_active_mask` |
| 8 | D | `labels.py` quality·cluster·time-decay 3중 sample weighting | 4개 단위 테스트 (clip/cluster/time/joint) |
| 9 | E | `apply_selection_stage` 3축 점수 + cluster cap | `test_stage6_cluster_cap` + `test_stage6_multi_objective_weights` |
| 10 | F | `AlphaEvaluationReport` dual panel | `test_alpha_evaluation_dual_panel` + alpha0.md 재측정 |
| 11 | — | OOS 실측 → `docs/results/alpha2.md` 작성 + acceptance criteria 검증 | 7.2 표 전 지표 측정 |
| 12 | — | Optuna로 `(w_F, w_A, w_D)` 탐색 → 최적값 commit | walk-forward OOS Calmar 최대화 |

각 Step은 독립 PR. Step 1~3 마무리 전 Step 4 진행 금지(데이터 구조 의존).

---

## 9. 마이그레이션 & 호환성

### 9.1 `universe-fix.md` 호환

- `historical_trading_panel` 필드는 유지 (universe-fix가 이미 사용 중)
- `training_universe_scope = "historical_stage6"` Literal 값은 유지 (deprecated 표시, 1 사이클 후 제거)
- 두 명세를 동시에 적용 가능 — 새 기본값 `historical_stage5_union`이 우선

### 9.2 `phase2_alpha_universe_decoupling.md` 호환

- `training_universe_scope = "stage5_passed"`는 그대로 동작 (단일 quarter Stage5 = `live_inference_panel`)
- Sample weighting의 time-decay 부분은 §3.2를 그대로 흡수, quality·cluster 항을 추가

### 9.3 Rollback Trigger

| Trigger | 대응 |
|---------|------|
| `eligible (inference) < 0.70` | `training_universe_scope = "historical_stage6"` 회귀 (universe-fix 모드) |
| `net_ic (trading panel) < 0.003` 또는 `Calmar < 0.4` | Stage6 가중치 `(w_F, w_A, w_D) = (1.0, 0.0, 0.0)` 회귀 (현행 friction-only) |
| `inference panel`이 `cluster_balance` 부재 시 dominated | `sample_weight_cluster_balance_enabled = False`로 끄고 재측정 |

각 회귀 경로는 **config flag 한 줄 변경**으로 가능 (코드 revert 불필요).

---

## 10. Out-of-Scope (별도 트랙)

| 항목 | 사유 |
|------|------|
| 모델 구조 변경 (Huber/Quantile loss, stacking) | `phase2_alpha_universe_decoupling.md` §6 영구 금지 |
| Multi-horizon stack (h=6/12/18) | Phase 2.1 후속 트랙 |
| DeFi TVL / Stablecoin Supply 등 외부 데이터 | Phase 3 후속 |
| Stage5 자체 임계값 완화 (예: `min_adv_usdt_median` ↓) | universe quality regression 위험 — 본 명세에서 손대지 않음 |
| `cluster_balance`의 per-bar 동적 가중 | O(T·N) 비용 — 정적 가중 효과 측정 후 결정 |
| Funding drag을 alpha_capacity에 통합 | 데이터 가용성 검증 필요 (8h funding history) — Phase 후속 |
| Maker (post-only) 주문 모델링 | `alpha0.md` §6 차세대 과제 |

---

## 11. 이론적 근거 참고문헌

- **Grinold, R. C.** (1989). "The Fundamental Law of Active Management". *Journal of Portfolio Management*.
- **Grinold, R. C., & Kahn, R. N.** (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- **López de Prado, M.** (2016). "Building Diversified Portfolios that Outperform Out of Sample". *Journal of Portfolio Management* (HRP).
- **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley. Ch. 4 (Sample Weights), Ch. 6 (Time-Decay), Ch. 7 (Cross-Validation in Finance).
- **Asness, C., Frazzini, A., & Pedersen, L. H.** (2014). "Quality Minus Junk". AQR Working Paper.
- **Israel, R., & Moskowitz, T. J.** (2013). "The role of shorting, firm size, and time on market anomalies". *Journal of Financial Economics*.
- **Burges, C. J. C.** (2010). "From RankNet to LambdaRank to LambdaMART: An Overview". *Microsoft Research Technical Report MSR-TR-2010-82*.
- **Markowitz, H.** (1952). "Portfolio Selection". *Journal of Finance*.

---

## 12. 결론 요약

| 질문 | 답변 |
|------|------|
| Q1. 추론/집행 유니버스 분리는 합리적인가? | **이론적 필수**. 단일 유니버스는 Pareto-dominated. 두 함수의 최적해가 직교. |
| Q2. 거래가능 후보군 → 자산증식 후보군으로의 진화? | Stage6 점수를 friction-only에서 **friction · alpha_capacity · diversification 3축**으로 확장, **cluster-aware selection**으로 ENB 보호. |
| Q3. ML 학습 후보군은 어떻게 구성해야 효과적 ranking이 가능한가? | **Historical Stage5 Union** (분기별 Stage5 통과의 시계열 union) — PIT-safe, breadth 5~10×, LambdaMART pair count 15×. `inference_active_mask`를 Stage5 timeline으로 별도 정의. |

핵심 아키텍처: **3-Cohort Model** (Inference / Live Inference / Execution), **Dual Membership Mask**, **Multi-Objective Stage6 with Cluster Cap**, **Triple-Boosted Sample Weighting** (time·quality·cluster).
