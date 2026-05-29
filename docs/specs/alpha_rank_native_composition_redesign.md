---
title: Alpha Rank-Native 합성 재설계 — 절대 EV 게이트 폐기, IC 기반 포트폴리오 비용 게이트, 결정론적 리스크 오버레이
domain: strategy-ml
type: prd
status: proposal
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/forecast/contracts.py
  - src/domain/futures/forecast/alpha.py
  - src/domain/futures/forecast/compose.py
  - src/domain/futures/optimization/objectives.py
  - src/domain/futures/portfolio/portfolio_constructor.py
  - src/domain/futures/strategy/regime_gate.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/alpha_evaluation.py
change_triggers:
  - src/domain/futures/forecast/**
  - src/domain/futures/portfolio/**
  - src/domain/futures/strategy/**
dependencies:
  documents:
    - docs/architecture/alpha.md
    - docs/specs/alpha_breadth_decontamination.md
    - docs/specs/master_alpha_universe_architecture.md
last_verified: 2026-05-29
---

# Alpha Rank-Native 합성 재설계 PRD

## 0. TL;DR

실측 로그(96-asset, --mode alpha, ml_lambdamart_v1)는 명확한 진단을 제공한다:

- **Ranker score OOS Rank IC = 0.0731 (t=9.71)** — 신호 자체는 강력하고 OOS 일반화됨.
- 그러나 최종 C3 `NET_IC = 0.0027`, `breadth = 0.95` — 신호가 포트폴리오로 전달되지 못함.
- **근본 원인은 3중 결함의 복합**이며, 모두 **신호→포트폴리오 합성 계층**에 있다. 모델/피처 계층은 정상이다.

핵심 수학(`compute_breakeven_ic`): `BE_IC = cost / (σ_r · √breadth)`. 현재 breadth≈1 → BE_IC=0.0508(난공불락). **breadth=8이면 BE_IC≈0.018 ≪ IC 0.073.** breadth가 지배 레버이며, breadth로 가는 길은 **OOS 일반화되는 rank로 사이징**하는 것이지 **일반화 실패하는 절대 EV로 게이팅**하는 것이 아니다.

**설계 철학:** "순서(ordering)는 믿고, 크기(magnitude)는 믿지 않는다." LightGBM ranker의 횡단면 순서(IC 0.073)를 1차 자산으로 삼고, calibrator의 절대 EV(IS 75bps→OOS 4.5bps, 15× 붕괴)는 게이트에서 배제한다.

---

## A. 검증된 실패 연쇄 (Root-Cause Chain, 코드 실측)

### A.1 절대 EV는 OOS 일반화에 실패한다 (약한 고리)
EV-PRECLIP 진단: IS fold는 clip 천장(75bps)에 포화(p90=p95=75), OOS(vrefit) p95=**4.5bps**, neg=**64.2%**. 크립토 4h 절대 기대수익은 비용벽(24bps)을 OOS에서 넘을 수 없다. 이는 튜닝 문제가 아니라 **예측 과제 난이도**의 문제다(magnitude 예측 ≫ ordering 예측).

### A.2 `compose_mu`가 절대 EV로 게이팅한다
`src/domain/futures/forecast/compose.py` 실측:
```python
mu_long = beta_a * alpha.alpha_long_2d - cost_frac      # cost_frac ≈ 24bps
...
xs_long = np.where(mu_long >= hurdle, mu_long, 0.0)      # hurdle = +10bps
```
OOS EV(4.5bps) ≪ cost(24bps) → `mu_long ≈ -20bps` 전역 → `mu_long >= +10bps` 거의 불성립 → **xs_long 전역 0 → breadth≈1**.

### A.3 강력한 rank 신호가 포트폴리오에 도달하지 못한다 (Dead Code)
- `AlphaForecast` 컨트랙트(`contracts.py`)에 **`rank_score_long_2d` 필드가 없다.**
- `compose_mu`는 `getattr(alpha, "rank_score_long_2d", None)` → **항상 None** → `rank_then_ev_gate` 블록 전체가 **조용히 스킵**되고 순수 EV 게이트로 폴백.
- `objectives.py:~410`의 인라인 `AlphaForecast` 생성도 rank score를 전달하지 않음(전 q-field None).
- **결론: IC 0.073 신호는 계산·로깅된 뒤 폐기된다.**

### A.4 Binary regime gate가 60% bar를 0으로 셧다운
`regime_gate.py` + `config.py`: `regime_exposure_bear = 0.0`. Bear(2905) + Chop(1096) = 60%+ bars. `_apply_ls_balance`는 단방향/소거된 북을 그대로 통과시키므로(`단방향 포트폴리오는 그대로 통과`), bear-zeroing은 달러중립 헤지조차 없이 알파를 폐기한다.

### A.5 정상 작동 중인 자산 (건드리지 않음)
`portfolio_constructor.py`는 이미 정교하다: Ledoit-Wolf shrinkage 공분산(`rolling_ledoit_wolf_cov`), Kelly 사이징(`_kelly_scaled`), L/S 밸런스(`_apply_ls_balance`), L1/L∞ 제약 투영(`_project_l1_linf_numba`). **constructor는 정상이며, 입력 신호만 퇴화되어 있다.**

---

## B. 설계 — 이론적 근거 (Theory-Grounded Architecture)

### B.1 Pillar 1 — Rank-Native 횡단면 사이징 (절대 EV 게이트 폐기)

**이론:** Grinold-Kahn Fundamental Law (`IR = IC·√breadth`)와 characteristic portfolio. 순수 IC 신호의 최적 횡단면 가중치는 표준화 신호 z에 비례(`w ∝ Σ⁻¹z`, Σ는 이미 보유). 절대 magnitude는 불필요.

**메커니즘** (신규 admission mode `rank_cs_neutral`):
1. ranker score를 매 bar 횡단면 z-score로 표준화: `z_i,t = (s_i,t − mean_t(s)) / std_t(s)`.
2. **분위 선택으로 breadth 확보:** top quantile(예: 상위 1/3) → long, bottom quantile → short. 96 자산 기준 측당 ~32 후보 → raw breadth ≫ 8 (상관 조정 후 effective ≥ 8 목표). 기존 `RANK_PORTFOLIO_TOP_K=4`(측당 4)는 breadth 기근의 직접 원인이므로 **분위 기반으로 대체**.
3. 선택 종목 사이징은 `z`에 비례(downstream Kelly+공분산 솔버가 위험조정). 절대 EV(`alpha_long_2d`) magnitude는 사이징에서 미사용.
4. xs는 더 이상 `mu>=hurdle`로 0 처리하지 않음 — 분위 마스크가 breadth를 결정.

### B.2 Pillar 2 — 포트폴리오 레벨 IC 기반 비용 게이트 (per-asset hurdle 폐기)

**이론:** per-asset 절대 EV가 비용을 넘는지 묻는 대신, **횡단면 L/S 북의 기대 gross 스프레드**가 amortized 비용을 넘는지 묻는다.
```
E[gross spread_t] ≈ IC_prior · σ_r,t · (z̄_long,t − z̄_short,t)
net_edge_t = E[gross spread_t] − cost_rt / holding_bars
```
- `IC_prior`: **leak-free 보수적 상수**(예: 0.03, 관측 0.073의 절반 이하). 실현 IC 사용 금지(look-ahead).
- `σ_r,t`: 횡단면 forward-return std(관측 가능, bps).
- bar별 게이트: `net_edge_t > 0`인 bar만 배포. 기회 없는 bar는 자연 skip — 전역 0이 아님.
- 비용 amortization: `COST_GATE_AMORTIZE`(기존 존재) 활성화. SWEEP 실측상 18h 보유 시 BE_IC 0.0508→0.0284로 절반.

### B.3 Pillar 3 — 결정론적 리스크 오버레이 (binary regime gate 대체)

사용자 질의(ML regime 등)에 대한 이론적 판단:

**채택: Beta-중립화 + 변동성 타게팅 (예측 불요, 결정론적)**
1. **Beta-neutralization:** `RiskForecast.beta_2d`(보유) + Ledoit-Wolf 공분산으로 북의 net market beta ≈ 0 강제. bull/bear 시장 방향을 **예측이 아니라 헤지**한다 → regime gate의 본래 의도(악조건 디리스크)를 예측 취약성 없이 달성.
2. **Volatility targeting (Moreira & Muir 2017, "Volatility-Managed Portfolios"):** 트레일링 실현 포트폴리오 vol로 gross exposure를 연속 스케일링하여 목표 연율 vol 유지. binary {bull:1,bear:0,chop:1}를 **연속·leak-free 스칼라**로 대체. 격변(주로 bear)기 자동 디리스크하되 알파를 0으로 죽이지 않음.

**ML regime 분류를 1차 권장하지 않는 근거:**
- **HMM regime은 2026-05-24에 의도적으로 제거됨**(futures 백테/최적화 경로, 24파일 −1151줄, MEMORY.md 기록). 재도입은 최근 결정과 충돌.
- regime 분류는 그 자체로 hard OOS 일반화 예측 문제 — 지금 싸우는 바로 그 실패 모드. 오분류 시 잘못된 binary 승수가 적용됨.
- 결정론적 beta-hedge + vol-target은 **예측 없이** 동일 목적 달성 → 더 견고.
- (선택) 확률적 regime이 추후 필요하면 GMM/HMM 소프트 확률을 vol 스칼라에 **블렌드**(hard 0 금지)하고 strict purging — 단 본 PRD 1차 범위 밖.

### B.4 Pillar 4 — 절대 EV는 2차 횡단면 틸트로만 (선택)
calibrator의 절대 magnitude는 폐기하되, **EV의 횡단면 rank**는 잔존 정보 가능. `final_z = w_rank·rank_z + w_ev·ev_rank_z` (기본 `w_ev=0`, ranker-primary). calibrator 전면 폐기 회피 + magnitude 불신.

### B.5 비범위 (Out of Scope)
- **모델/피처 계층 불변:** IC 0.073은 충분. 신호 과적합 회피("정상 작동 자산 비수정").
- cost.py는 동적 amortization 활성화만, 재설계 없음.

---

## C. Surgical Plan (단계별 블루프린트)

### Phase 1 — Rank-Native 사이징 + rank 신호 배선 + 포트폴리오 비용 게이트 (최고 레버, 최저 위험)

**C.1 `src/domain/futures/forecast/contracts.py` — [ACTION: REPLACE] `AlphaForecast`**
`artifact_hash` 뒤에 default 필드 2개 추가(frozen slots: defaulted 필드는 반드시 마지막):
```python
    artifact_hash: AlphaArtifactHash
    rank_score_long_2d: np.ndarray | None = None
    rank_score_short_2d: np.ndarray | None = None
```

**C.2 `src/domain/futures/forecast/alpha.py` — [ACTION: REPLACE] `to_alpha_forecast` 반환부**
v3 메타데이터에서 rank score reshape하여 채움(`_reshape` 재사용):
```python
        rank_score_long_2d=_reshape("rank_score_long"),
        rank_score_short_2d=_reshape("rank_score_short"),
```
(ml_builder가 `alpha_forecast_v3`에 `rank_score_long`/`rank_score_short` flat array를 넣도록 보장 — 미존재 시 None 폴백 안전.)

**C.3 `src/domain/futures/optimization/objectives.py:~410` — [ACTION: REPLACE] 인라인 `AlphaForecast`**
aligned에서 rank score를 추출해 전달(있으면). 없으면 None.

**C.4 `src/domain/futures/forecast/compose.py` — [ACTION: REPLACE] `compose_mu`**
신규 admission mode `rank_cs_neutral` 추가:
- `_cs_zscore(score_2d)` 헬퍼: bar별 횡단면 표준화(NaN-safe, finite 마스크).
- long/short 분위 선택: `RANK_SELECT_QUANTILE`(기본 0.33) 또는 `RANK_PORTFOLIO_TOP_K`를 측당 K로 사용하되 **K ≥ ceil(target_breadth/2)** 강제.
- xs 출력 = 선택 마스크 내 `z`(또는 `final_z`), 외부 0.
- 포트폴리오 비용 게이트: bar별 `net_edge_t` 계산 → `net_edge_t <= 0`인 bar는 측 전체 0.
- `IC_PRIOR_FOR_GATE`(기본 0.03), `RANK_SELECT_QUANTILE`, `TARGET_BREADTH`(기본 8) 파라미터 도입.
- 기존 `ev_gate`/`rank_then_ev_gate` 경로는 하위호환 유지(deprecated 주석).

**C.5 `src/domain/futures/strategy/config.py` — [ACTION: ADD] 신규 파라미터**
```python
    post_cost_admission_mode: Literal["ev_gate", "rank_then_ev_gate", "rank_cs_neutral"] = "rank_cs_neutral"
    rank_select_quantile: float = 0.33
    target_breadth: int = 8
    ic_prior_for_gate: float = 0.03
    ev_secondary_tilt_weight: float = 0.0
```
`__post_init__`에 범위 검증 추가(0<quantile<0.5, target_breadth>=2, 0<=ic_prior<=0.2).

**Phase 1 검증:**
```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode alpha --strategy ml_lambdamart_v1 --skip-data-sync --skip-universe
```
기대: `ALPHA SCOREBOARD` breadth ≥ 8, NET_IC ≥ 0.02, t-stat ≥ 2.0. `[SCORE-IC] emit_breadth`가 1 → ≥8로 상승.

### Phase 2 — 결정론적 리스크 오버레이 (binary regime gate 대체)

**C.6 `src/domain/futures/portfolio/portfolio_constructor.py` — [ACTION: ADD]**
- `neutralize_market_beta(w, beta_1d) -> w_neutral`: `w' = w − (wᵀβ / βᵀβ)·β` (net beta=0 투영). zero-beta 가드.
- `volatility_target_scalar(realized_vol_ann, target_vol_ann, *, cap) -> float`: `min(target/realized, cap)` 연속 스칼라.
- `precompute_rebalance_weights` 파이프라인에 beta-neutral 투영(L1/L∞ 투영 전) + vol-target 스칼라(gross 적용) 삽입.

**C.7 `src/domain/futures/strategy/regime_gate.py` — [ACTION: REPLACE]**
binary 승수 폐기. `regime_gate_enabled=False`를 신규 기본값으로(config), 함수는 하위호환 유지하되 vol-target 경로로 위임. 또는 `apply_risk_overlay`로 신설 후 regime_gate deprecate.

**C.8 `config.py` — [ACTION: ADD]** `target_vol_annual: float = 0.20`, `vol_target_cap: float = 2.0`, `beta_neutralize_enabled: bool = True`, `regime_gate_enabled` 기본 `False`로 변경.

**Phase 2 검증:** REGIME IC가 bear에서도 양수 유지, NET_IC가 bull/bear/chop 전반 균질화. net market beta ≈ 0 로깅.

### Phase 3 — No-Trade Band / 턴오버 최적화 (선택, Garleanu-Pedersen)
proportional cost 하 최적 정책: aim 포트폴리오로 부분 이동. `|w* − w| > band`일 때만 리밸런스. `NO_TRADE_BAND_BPS` 파라미터. Phase 1·2 안정화 후 진행.

---

## D. 평가 / 검증 표준 (Quant)

1. **Leakage Check:** `IC_PRIOR_FOR_GATE`는 상수(실현 IC 미사용). vol-target은 트레일링(strict-causal). beta 추정 trailing. Purge/Embargo 불변.
2. **Stability Check:** 시드 앙상블 불변. 분위 선택은 z-score 횡단면 표준화로 outlier-robust(MAD 옵션 고려).
3. **Friction Check:** 포트폴리오 비용 게이트가 amortized round-trip(24bps) 적용 후에도 net_edge>0 입증. breadth≥8에서 BE_IC≈0.018 < IC 0.073.
4. **회귀 테스트:** `tests/unit/domain/futures/forecast/test_alpha_forecast.py`, `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`에 신규 모드/헬퍼 단위테스트 1:1 추가.

## E. 완료 기준 (Acceptance Criteria)
- [ ] AlphaForecast가 rank_score를 보유·전달, compose_mu `rank_cs_neutral` 경로 동작(getattr None 데드코드 해소).
- [ ] `ALPHA SCOREBOARD` breadth ≥ 8 (현 0.95).
- [ ] OOS NET_IC ≥ 0.020, t-stat ≥ 2.0 (현 0.0027 / 1.06).
- [ ] binary regime gate 제거, net market beta ≈ 0 + 연속 vol-target 적용.
- [ ] mypy/ruff clean, 신규 단위테스트 통과(coverage ≥ 90% on changed modules).
- [ ] `docs/architecture/alpha.md` §6 갱신 + `last_verified` 갱신.

## F. 위험 / 한계
- breadth↑가 BE_IC를 낮추지만, 분위 확장은 신호 약한 종목 포함으로 **실현 IC를 희석**할 수 있음 → top/bottom 분위와 IC 트레이드오프를 검증 단계에서 스윕.
- vol-target은 격변기 레버리지 축소로 일부 수익 기회 포기(보수적, 의도된 trade-off).
- beta-neutralization은 beta 추정 오차에 민감 → Ledoit-Wolf shrinkage로 완화.
