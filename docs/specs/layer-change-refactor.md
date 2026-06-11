---
title: Tiered Hybrid Architecture — 냉정한 검토 및 리팩토링 블루프린트
domain: futures/strategy
type: spec
status: proposal
priority: critical
related_paths:
  - src/domain/futures/strategy/candidate_workflow.py
  - src/domain/futures/portfolio/portfolio_constructor.py
  - src/application/futures/optimization/universe_service.py
  - src/domain/futures/optimization/opt_config.py
source_doc: docs/decisions/layer-change.md
last_verified: 2026-06-11
---

# 🎯 Objective
`layer-change.md`(2계층 하이브리드)의 이론·수식·ML·Optuna·데이터정책을 코드 실측 대비 검증하고, 결함을 보정한 실행 가능한 리팩토링 설계도 + 파일 정리 + universe 데이터기간 정책을 확정한다.

---

# 0. SOTA 판정 (요청 1)

**판정: "SOTA"가 아니라 "Industry-Standard Robust". 방향은 옳고, 코인 시장에 적합하나 신규성 없음.**

- 제안 구조(per-symbol signal → cross-sectional rank → Top-K → diagonal Kelly)는 **Grinold-Kahn / AQR식 cross-sectional alpha+risk model**의 정석. 검증된 안전한 설계이나 2024+ SOTA(예: end-to-end differentiable portfolio, deep CS factor model, RL execution)는 아님.
- **냉정한 핵심 지적:** 이 리팩토링은 *아키텍처 문제*를 푼다. 그러나 [[project_wf_fold_reporting_fix_2026_06_07]]·[[project_candidate_v6_signal_context]]에 기록된 **실제 블로커는 "feature 예측력 부재(Rank IC≈0.000, μ≈1bps < breakeven 3.8bps)"** 이다. 구조를 바꿔도 알파가 없으면 friction filter가 전 종목을 0으로 만들어 **정직하게 현금 보유**할 뿐 — 수익은 생기지 않는다. **구조 개선과 알파 탐색은 별개 트랙으로 병행해야 함.**

---

# 1. 구조적 결함 (요청 2 — Cross-cutting)

| # | 결함 | 근거 | 보정 |
|---|------|------|------|
| S1 | **"Pointwise를 푼다"는 주장과 모순** | `[Raw Mu, Vol]`의 Mu는 심볼별 독립 시계열 예측. 이를 사후 랭킹/Z-score해도 underlying은 여전히 pointwise → market-beta 노출 잔존 | 각 t에서 **랭킹 전에 cross-sectional demean/beta-neutralize** (μ_i ← μ_i − β_i·μ_mkt 또는 CS 평균 차감). Phase2 LambdaMART은 rank를 직접 target |
| S2 | **선택과 사이징의 σ 지수 불일치** | 랭킹=Sharpe(μ/σ), 사이징=Kelly(μ/σ²). 동일 위험조정량을 쓰지 않음 | 의도된 분리면 문서에 명시. σ²는 vol 추정오차를 제곱 증폭 → **vol targeting/cap** 또는 σ shrinkage 추가(epsilon floor는 0만 방어, 추정오차 미방어) |
| S3 | **순수 Diagonal은 코인 공통인자(BTC-beta) 무시** | 코인 분산의 ~70-80%가 단일 시장인자. Diagonal은 상관 높은 alt에 **숨은 집중** 유발 | LW/BL 폐기는 옳음(역행렬 오차 회피). 단 **single-factor(BTC-beta) risk model**로 절충: w∝μ/σ²에 beta-cap 또는 market-neutral. Sector Cap≠factor |
| S4 | **Top-K 경계 whipsaw — Bayesian admission과 동일 문제 재발** | K=3~4(`K_RANK` 1-4) / 20유니버스에서 rank4↔5 flip이 회전율 폭증. hysteresis는 breadth에만 적용, Top-K 경계엔 미적용 | Top-K 경계에 **buffer/hysteresis**(보유분은 rank≤K+buffer까지 유지) 또는 rank-weighted soft-K + **no-trade band**(Δw>τ일 때만 거래) |
| S5 | **Rank-to-Mu Scaling이 "Raw Mu 사이징"과 충돌** | item9 `Rank-to-Mu`는 ordinal→magnitude 재합성 = Kelly가 필요로 하는 크기정보 파괴 후 가짜로 재주입 | **사이징은 Raw Mu만 사용**(magnitude-aware), rank는 selection 전용. `Rank-to-Mu` 함수 폐기 |

---

# 2. Layer별 세부 검토 (요청 2)

## Layer 1 (Signal)
| 항목 | 제안값 | 검토 | 권고 |
|------|--------|------|------|
| IC pass | Mean IC≥0.02, t≥1.96 | IC 0.02는 institutional 하한(통상 0.03-0.05). **plain t-stat은 overlapping label autocorr로 팽창** → 거짓 통과 | **HAC(Newey-West) t-stat 또는 deflated** 사용(`FUTURES_ML_IC_FILTER_USE_HAC=True` 이미 존재). IC 목표 0.03 권장. **현 실측 IC≈0**임을 명시 |
| Breadth pass | per-symbol≥30%, ≥4종목 시점≥80% | liveness 체크로 타당하나 30/80은 magic number | rebalance 빈도·K와 결합해 도출. ≥4는 K=3~4와 정합 |
| QC | t-stat + min_obs | 타당. 단 min_obs는 label_horizon·embargo와 연동 필요 | embargo ≥ max(label_horizon_bars)=18 강제 |
| 인터페이스 | `[Raw Mu, Vol]` tuple, vol epsilon floor 1e-6 | floor는 zero-div만 방어. 1e-6는 per-bar σ엔 과소(거의 무한 레버리지 허용) | floor를 **유의미한 하한**(예: 자산 최소 historical σ의 일정비율)으로. 또는 vol-cap |

## Layer 2 (Allocation)
| 항목 | 제안 | 검토 | 권고 |
|------|------|------|------|
| LW+BL 폐기→Diagonal | O | 코인에 타당(DeMiguel 2009 1/N). 단 S3(공통인자) | single-factor로 절충 |
| Friction Filter | breakeven을 랭킹게이트→사이징필터 이동 | 옳음. 단 **funding(perp 8h) 미포함** | hurdle = 2·taker + 2·slip + **funding_carry** |
| Diagonal Kelly | w∝μ/σ² | σ² 노이즈 민감(S2) | kelly_fraction·vol-target 동반 |
| Caps | per-symbol/sector/gross | 타당. gross가 유일 레버리지 통제 | **kill-switch 연동**(trading_bot.md): gross 위반·연속에러시 flat |

## ML / Optuna
| 항목 | 제안 | 검토 | 권고 |
|------|------|------|------|
| Phase2 LambdaMART | rank 학습 | query group≈20·K@3-4로 gradient 약함. **무알파 feature 위 트리=과적합** | **Phase1(Z-score) IC>0 입증 전까지 보류.** 즉시 도입 X |
| CPCV(L1) | combinatorial purged | overlapping label에 정석(López de Prado), quant.md 정합. 단 1500trial×CPCV = 고비용 | compute budget 명시. embargo=max horizon |
| AWF(L2) + OOS Stacking | anchored WF, L1 OOS만 입력 | **leakage 방지 정석(nested/stacked).** 설계 우수 | 유지 |
| Decoupled Optuna | L1=IC, L2=Sharpe 분리 | 현재 단일 `ENGINE_PARAM_SPACE_FUTURES` 합본 → **2 study 분리 필요(실 구현 delta)**. 분리는 과적합·탐색공간 축소로 健全 | L1 study(lookback·filter·model HP) / L2 study(K_RANK·kelly·sector·friction) |
| total_trials=1500 | 합본 | 단일공간 1500은 PBO 위험. (DSR/PBO 패널티는 이미 존재 — 양호) | 분리 후 per-layer 축소 |

---

# 3. Contract Changes (signatures)

```python
# Layer1 표준 출력 — signal_composer.py
@dataclass(frozen=True, slots=True)
class SymbolSignal:
    raw_mu: float        # 절대 기대수익(사이징용, magnitude 보존)
    volatility: float    # per-bar σ, floor=VOL_FLOOR (1e-6 아님)
    n_obs: int           # QC
    t_stat: float        # HAC 권장
    valid: bool          # Reliability QC 통과

# Layer2 — cross-sectional 단계 (신규: cs_rank.py)
def neutralize_cross_section(            # S1 보정
    mu: NDArray[float64],                # [N] at t
    beta_btc: NDArray[float64] | None,   # single-factor면 사용
) -> NDArray[float64]: ...               # CS-demeaned/beta-neutral mu

def rank_and_select(
    signals: Mapping[str, SymbolSignal],
    *, k_rank: int, sector_cap: int,
    prev_selection: frozenset[str],      # S4 hysteresis 입력
    rank_buffer: int,                    # 경계 buffer
) -> tuple[frozenset[str], dict[str, float]]: ...  # (selected, z_scores)

# Sizing — portfolio_constructor.py (LW/BL 제거 후)
def diagonal_kelly_weights(
    mu: NDArray[float64], sigma: NDArray[float64],
    *, kelly_fraction: float, vol_target: float | None,
    friction_hurdle_bps: NDArray[float64],  # funding 포함
    caps: PortfolioCaps,
    prev_w: NDArray[float64], no_trade_band: float,  # S4
) -> NDArray[float64]: ...

# Universe 3-way window — opt_config.py / universe_service.py
def get_layered_window(                  # 요청 4
    reference_date: date,
    *, l1_months: int = 18, l2_months: int = 12, holdout_months: int = 6,
    regime_floor: date,                  # 하드 플로어(아래 §6 결정 필요)
) -> LayeredWindow: ...                  # fetch/l1_start/l2_start/holdout_start/end

def discover_universe_timeline(          # 시그니처 확장
    *, tf, l1_start, l2_start, holdout_start, end_date,   # oos_start→2분할
    min_history_bars: int,               # 신규: 학습창 데이터충분성 게이트
    force_rebuild=False,
) -> UniverseTimelineResult: ...
```

---

# 4. Surgical Plan (유지·수정 파일)

- **`candidate_workflow.py::run_candidate_walk_forward`**: 2-way(is/oos)→**3-way(L1/L2/holdout)** 분리. L1 OOS 예측을 L2 입력으로 stacking. CPCV(L1)/AWF(L2) 호출 분기.
- **`portfolio_constructor.py`**: `rolling_ledoit_wolf_cov`·BL·full-cov 경로 **제거**. `diagonal_kelly_weights` 신설. `project_all_caps`·`quantize_weights` 유지. `_kelly_raw/_kelly_scaled` 재사용. `no_trade_band` 추가.
- **(신규) `strategy/cs_rank.py`**: `neutralize_cross_section`·`rank_and_select`. S1/S4 보정 격리.
- **`signal_composer.py` / `rule_signals.py`**: 출력 `SymbolSignal` tuple 규격화 + VOL_FLOOR 상수화.
- **`candidate_gate.py`**: breakeven hard gate → **사이징단 friction filter**로 역할 이동(랭킹 비차단).
- **`opt_config.py`**: `ENGINE_PARAM_SPACE_FUTURES` → **`L1_ALPHA_SPACE` / `L2_ALLOC_SPACE`** 2분할. `get_layered_window` 추가. `REGIME_FLOOR` 상수.
- **`universe_service.py`**: `discover_universe_timeline` 3-경계화 + `min_history_bars` 게이트.
- **유지(삭제 금지):** `market_regime.py`(breadth risk-off/crisis 오버레이로 재활용 — E3), `friction_model.py`, `execution_cost.py`, `covariance.py`(축소).

---

# 5. File Separation → legacy/ (요청 3, **참조감사 후 이동**)

> ⚠️ Anti-Pattern 방지: 아래는 *legacy 후보*. 이동 전 `find_referencing_symbols`로 import 0 확인 필수. 추측 이동 금지.

| Legacy 후보 | 사유 | 선행 확인 |
|-------------|------|-----------|
| `candidate_ensemble.py` (RegimeConditionalEnsemble) | "Ensemble B0" = 폐기 대상 본체 | workflow/evaluation 참조 끊기 |
| `regime_evaluation.py` | regime-conditional means 폐기. (단 regime *detection* 일부는 market_regime로 이관) | 부분 이관 후 |
| `portfolio_constructor.py::rolling_ledoit_wolf_cov` + BL 분기 | LW/BL 폐기 | 함수단 제거(파일 유지) |
| Bayesian admission 로직([[project_bayesian_admission_2026_06_10]]) | binary admission→continuous rank 대체 | 위치 식별(candidate_gate?) 후 |
| `candidate_edge.py` / `ablation.py` (해당시) | B0 실험 잔재 가능성 | 참조감사로 판정 |

**유지 핵심:** workflow·portfolio_constructor·universe_service·opt_config·rule_signals·signal_composer·friction_model·market_regime·validation/*·optimization/*(study 분리만).

---

# 6. Data Period & Universe (요청 4)

## 현 상태(실측)
- `get_quarterly_window`: IS=24mo, OOS=6mo, fetch=is_start−365d → **2-way only**.
- `discover_universe_timeline(is_start, oos_start, end_date)`: 분기별(3mo) 재빌드, `oos_start` 단일 경계.
- universe 7-stage(`universe.md`)는 liquidity·cost 게이트는 있으나 **학습창 데이터충분성 게이트 부재**.

## 보정 설계
1. **3-way 슬라이딩 윈도우**: L1(18mo)→L2(12mo)→holdout(6mo). `discover_universe_timeline`에 `l2_start` 경계 추가(oos→2분할). 분기 재빌드 cadence는 기존과 정합(충돌 없음).
2. **Regime hard floor 강제**: 현재 상대계산이라 윈도우 확장 시 pre-FTX 유입 가능. `REGIME_FLOOR` 상수로 클램프.
   - ⚠️ **문서 사실오류**: "2022 이전(FTX 사태 이전)" — FTX 붕괴는 **2022-11**. "pre-2022"≠"pre-FTX". §9 결정 필요.
3. **데이터충분성 = universe 멤버십 조건화(핵심 인사이트)**: 멤버는 as_of 시점에 **≥(L1 18mo + embargo) bars** 보유해야 CPCV fold 비퇴화. `min_history_bars` 게이트를 7-stage S1(structure)에 추가. → 신규상장 코인이 단순 liquidity 통과로 들어와 CPCV를 깨는 것 방지.
4. **데이터 확보 흐름**: `fetch_start = l1_start − buffer(embargo+지표 warmup)`. 멤버별 가용 bars 검증 후 부족분은 universe에서 제외(전체 fetch 실패 아님).

---

# 7. 🧪 Test Scenario Design

**T1 CS Neutralization (S1)** — Given 3종목 μ=[2,1,0]bps·동일 β=1, mkt_μ=1 → When `neutralize_cross_section` → Then 결과 평균≈0, 순위 보존.
**T2 Top-K Hysteresis (S4)** — Given prev_selection={A,B,C}, A가 rank K→K+1(buffer 내) → When `rank_and_select` → Then A **유지**(회전 미발생). rank>K+buffer로 떨어지면 교체.
**T3 Friction Filter + Funding** — Given μ_net=2bps, hurdle=3.8bps(funding 포함) → Then weight=0(현금). μ_net=5bps → weight>0.
**T4 Diagonal Kelly σ-edge** — Given σ→VOL_FLOOR 근접 → Then weight가 vol_target/cap에 의해 유한(무한 레버리지 방지). σ=0 입력 → ValueError 또는 floor 적용(zero-div 없음).
**T5 No-trade band** — Given Δw=0.5%<band(1%) → Then 주문 미발생(prev_w 유지). Δw=2% → 리밸런스.
**T6 3-way window leakage** — Given L1 학습구간 t, L2 입력은 t의 L1 *OOS* 예측만 → Then L2가 L1 in-sample 예측 참조 시 assert 실패(leakage guard).
**T7 Universe min_history** — Given 신규상장(history<18mo+embargo) → Then universe 제외. 충분 종목 → 포함.
**T8 Regime floor** — Given reference_date로 l1_start<REGIME_FLOOR 계산 → Then l1_start=REGIME_FLOOR 클램프, fetch도 동반 클램프.
**T9 Decoupled study** — Given L1_ALPHA_SPACE trial → Then objective=IC만 평가(Sharpe 미참조). L2 역.

---

# 8. Verification
```bash
uv run ruff check --fix src/domain/futures/strategy/cs_rank.py src/domain/futures/portfolio/portfolio_constructor.py
uv run mypy src/domain/futures/strategy/cs_rank.py
uv run pytest tests/unit/domain/futures -k "cs_rank or kelly or friction or window or universe_history" --tb=short
uv run pytest tests/unit/application/futures/optimization/test_universe_service.py -k "layered or history or floor" --tb=short
```

---

# 9. Logging Contract (결과 출력 설계)

각 Layer 실행 완료 시 `logger.info`로 출력할 표 형식 스펙. 구현 시 `result.md` 양식과 동일한 파이프 테이블 포맷 사용.

## 9.1 실행 윈도우 (모든 Phase 공통)

```text
[WINDOW: TIERED] ------------------------------------
| Segment      | Start      | End        | Duration  |
| ------------ | ---------- | ---------- | --------- |
| Regime Floor | {floor}    | —          | (hard LB) |
| L1 (CPCV)   | {l1_start} | {l2_start} | 18 months |
| L2 (AWF)    | {l2_start} | {ho_start} | 12 months |
| Holdout     | {ho_start} | {ho_end}   | 6 months  |
------------------------------------------------------
```

## 9.2 Layer 1 — Signal Quality (CPCV 완료 후 출력)

```text
[LAYER 1: AGGREGATE SIGNAL QUALITY] ----------------
| Metric              | Value    | Gate    | Pass?  |
| ------------------- | -------- | ------- | ------ |
| Mean IC (HAC)       | {ic:.3f} | ≥ 0.030 | {p}    |
| IC t-stat (HAC)     | {ts:.2f} | ≥ 1.96  | {p}    |
| Symbol Breadth (avg)| {br:.1%} | ≥ 30%   | {p}    |
| Valid Coverage (≥K) | {vc:.1%} | ≥ 80%   | {p}    |
| Valid Symbols / N   | {vs}/{n} |         |        |
| CPCV Fold Pass Ratio| {fp:.2f} | ≥ 0.60  | {p}    |
| L1 Gate             | —        | —       | {PASS/BLOCKED} |
-----------------------------------------------------

[LAYER 1: CPCV FOLD DETAILS] -----------------------
| Fold | IC (HAC) | Breadth | n_valid | n_events | Pass |
| ---- | -------- | ------- | ------- | -------- | ---- |
| {f}  | {ic:.3f} | {br:.1%}| {nv}    | {ne}     | {p}  |
-----------------------------------------------------

[LAYER 1: PER-SYMBOL DIAGNOSTICS (top 10 by IC)] ---
| Symbol      | Raw Mu(bps) | Vol    | t-stat | IC     | Valid |
| ----------- | ----------- | ------ | ------ | ------ | ----- |
| {sym}       | {mu:.1f}    | {v:.4f}| {ts:.2f}| {ic:.3f}| {Y/N}|
-----------------------------------------------------
(★ Valid=N 심볼은 Layer 2 랭킹 입력 제외)
```

**출력 위치:** `candidate_workflow.py::run_l1_cpcv` 완료 후
**출력 조건:** L1 BLOCKED이면 이후 Layer 2/3 섹션 출력 생략 (BLOCKED 메시지만 출력)

## 9.3 Layer 2 — CS Ranking & Allocation (AWF 완료 후 출력)

```text
[LAYER 2: ALLOCATION EFFICIENCY] -------------------
| Metric                  | Value    | Gate    | Pass? |
| ----------------------- | -------- | ------- | ----- |
| Top-K                   | {k}      | Optuna  |       |
| Friction Filter Pass %  | {fp:.1%} | ≥ 40%   | {p}   |
| Sharpe (Hybrid)         | {sh:.2f} |         |       |
| Sharpe (1/N Baseline)   | {sb:.2f} |         |       |
| Sharpe vs 1/N           | {sv:+.1%}| ≥ +20%  | {p}   |
| MDD (Hybrid)            | {md:.1%} |         |       |
| MDD (1/N Baseline)      | {mb:.1%} |         |       |
| MDD Reduced             | {mr}     | Y       | {p}   |
| Avg Active Positions    | {ap:.1f} |         |       |
| Turnover / rebal        | {to:.1%} |         |       |
| L2 Gate                 | —        | —       | {PASS/BLOCKED} |
-----------------------------------------------------

[LAYER 2: AWF FOLD DETAILS] -----------------------
| Fold | Sharpe | Friction% | Active_K | Turnover | Pass |
| ---- | ------ | --------- | -------- | -------- | ---- |
| {f}  | {sh:.2f}| {fp:.1%} | {k}      | {to:.1%} | {p}  |
-----------------------------------------------------

[LAYER 2: TOP-K SELECTION (OOS last t)] ------------
| Rank | Symbol   | Raw Mu(bps)| β_BTC | CS_Z  | w_kelly |
| ---- | -------- | ---------- | ----- | ----- | ------- |
| {r}  | {sym}    | {mu:.1f}   | {b:.2f}| {z:.2f}| {w:.3f}|
-----------------------------------------------------
(★ w_kelly: vol-cap·sector-cap·gross-cap·no-trade-band 적용 후)
(★ Friction Filter 제외 종목은 w=0.000으로 표시)
```

**출력 위치:** `candidate_workflow.py::run_l2_awf` 완료 후

## 9.4 Layer 3 — Hold-out Backtest (1-shot)

```text
[LAYER 3: HOLD-OUT BACKTEST ({ho_start} ~ {ho_end})] --------
| Model         |   CAGR |  MaxDD |  Sharpe |   MAR | Pass  |
| ------------- | ------ | ------ | ------- | ----- | ----- |
| L1+L2 Hybrid  | {ca:.1%}| {md:.1%}| {sh:.2f}| {mar:.2f}| {p}|
| 1/N Baseline  | {ca:.1%}| {md:.1%}| {sh:.2f}| {mar:.2f}| — |
| vs Baseline   | {dca:+.1%}|{dmd:+.1%}|{dsh:+.2f}|{dmar:+.2f}|{p}|
| L3 Gate       | —      | —      | —       | —     | {PASS/BLOCKED} |
--------------------------------------------------------------
(★ 단 1회 실행. 파라미터 frozen. Optuna 재사용 금지)
```

**출력 위치:** `candidate_workflow.py::run_l3_holdout` 완료 후

## 9.5 시스템 상태 헤더 (전체 파이프라인 시작 시)

```text
[SYSTEM STATUS] ------------------------------------
| Layer       | Status  | Blocker (if any)          |
| ----------- | ------- | ------------------------- |
| Layer 1     | {status}| {blocker or "—"}          |
| Layer 2     | {status}| {blocker or "—"}          |
| Layer 3     | {status}| {blocker or "—"}          |
-----------------------------------------------------
```

**Status 값:** `PASS` / `BLOCKED` / `PENDING` / `SKIP` (L1 BLOCKED → L2/L3 = SKIP)

---

# 10. Confirmed Decisions (2026-06-11 확정)
1. **Regime floor = `2023-01-01`** (post-FTX 정착 + 연 경계). `REGIME_FLOOR: date = date(2023,1,1)`. fetch/l1_start 동반 클램프(§6-T8).
2. **Risk model = Single-factor BTC-beta.** 랭킹 전 CS beta-neutralize + 사이징 beta-cap. `neutralize_cross_section(beta_btc=...)` 활성. portfolio_constructor에 rolling β 추정 추가. 순수 Diagonal 폐기(S3 방어).
3. **Phase2 LambdaMART = 보류.** Phase1(CS Z-score)로 Rank IC>0 입증 전까지 미구현. 즉시 작업은 Phase1만.
4. **Sizing basis = Raw Mu only.** `Rank-to-Mu Scaling`(원안 item9) **폐기**. rank=selection 전용, Kelly μ=Layer1 raw_mu(S5 해결).

> 영향: §2-S3→single-factor 확정, §3 `neutralize_cross_section` β 인자 필수, §6 `REGIME_FLOOR=2023-01-01`, ML 트랙은 Phase1 격리.
