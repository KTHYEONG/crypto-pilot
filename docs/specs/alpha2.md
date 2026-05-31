---
title: ML Alpha 실전투입 견고화 — Phase 2.5 & 3 정공법: 비용 보존 및 엣지 강화
domain: futures-alpha
type: prd
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/ml_builder.py
last_verified: 2026-05-31
---

# ML Alpha 실전투입 견고화 — Phase 2.5 & 3 정공법: 비용 보존 및 엣지 강화

> **목표:** 1회 왕복 물리 거래비용인 **24bps 장벽**을 엄격히 고정(끼워 맞추기식 비용 깎기 배제).
> 포트폴리오 분산 효과(Breadth) 최적화와 Idiosyncratic Alpha 강화(타깃 잔차화)를 통해 통계적 breakeven을 정면 돌파.

---

## 1. 유저 지적 수용 및 설계 배경

### 1.1. 물리적 마찰비용(Round-trip Cost)의 보존
- **물리적 진실:** 타임프레임이나 예측 호라이즌(6h, 12h, 18h)이 길어진다고 해서 1회 진입/청산 왕복에 따르는 테이커 수수료 및 슬리피지가 마법처럼 감소하지 않습니다. 물리적 최소 마찰은 **24bps**로 무조건 보존되어야 합니다.
- **기존 설계의 착시(Curve-fitting) 철회:** 이전의 호라이즌별 비용 상각 방식(`12h=12bps`, `18h=8bps`)은 OOS sweep 통과 요건을 맞추기 위해 통계적 리스크를 과소평가(underestimate)한 "수치 끼워 맞추기"였습니다. 이를 전면 폐기하고 모든 호라이즌 평가에서 **24bps 고정 장벽**을 적용합니다.

### 1.2. 근본적 실패 원인 해부
- 엄격한 24bps 적용 시, 요구되는 $\text{Breakeven IC} = \frac{24\text{bps}}{\sigma_r \times \sqrt{N_{\text{eff}}}} = 0.0343$ 입니다.
- 이때 $\sigma_r$ (선물 횡단면 변동성)은 약 $310\text{bps}$이며, 실질 베팅 개수 $N_{\text{eff}}$는 **$2.2$** 입니다.
- $N_{\text{eff}} = 2.2$로 지나치게 작아 분산 효과(Breadth)를 얻지 못하므로 통계적으로 24bps 장벽을 넘으려면 $0.0343$ 이라는 무리한 IC 수준이 요구되는 것입니다.

---

## 2. 정공법 개선 설계 (Contracts & Logic)

### 2.1. [Phase 2.5] Portfolio Breadth ($N_{\text{eff}}$) 최적화
- **논리:** L/S 선택 분위수인 `rank_select_quantile`가 `0.33`으로 과도하게 좁아 포지션이 소수 자산에 집중되어 $N_{\text{eff}}$가 2.2로 극소화되었습니다.
- **구현:** `rank_select_quantile`를 `0.45` 수준으로 상향 조절하는 실험을 수행합니다.
  - 횡단면 극단 선택 범위를 늘려 실질 독립 베팅 개수($N_{\text{eff}}$)를 $3.5 \sim 4.5$ 이상으로 다변화합니다.
  - 다변화 효과로 인해 $\text{Breakeven IC}$ 장벽은 자연스럽게 $0.024$ 수준으로 합리화됩니다.
  - 이 상태에서 IR(정보비율) 및 넷 수익성(`basket_net_bps`)과의 트레이드오프를 평가하여 최적점을 선택합니다.

### 2.2. [Phase 3] Idiosyncratic Alpha 강화 (타깃 잔차화)
- **논리:** 단순히 유니버스를 넓히는 것만으로는 부족하며, 시그널 엣지 자체를 강화해야 합니다. 현재 `resid_ic = 0.0141`은 목표값 `0.030` 대비 절대적으로 부족합니다.
- **구현:** 
  1. **Beta-Residualized Target 학습:** 단순 gross return 대신, BTC 베타를 정교하게 제거한 순수 잔차 리턴(`beta_residualized`)을 직접 모델 학습 타깃으로 주입하여 노이즈(시장 베타) 학습을 완전 차단합니다.
  2. **Idiosyncratic Features 추가:** BTC 대비 lead-lag 팩터, sector-relative, funding basis 이격 등 순수 개별 자산의 엣지 팩터를 보강합니다.

### 2.3. Bear Market Basket PnL 실측화 (엄격화)
- `bear_market_basket_safe` 검증 시 24bps 비용 장벽을 완벽하게 적용한 뒤, BTC 레짐 라벨러(`_compute_regime_labels`)를 사용하여 하락장 국면에서의 basket PnL을 정밀 실측합니다.

---

## 3. Surgical Plan (코드 수정 및 실험 계획)

### Target File 1: `src/domain/futures/strategy/config.py`
- `rank_select_quantile` 기본값을 `0.33` -> `0.45` 로 상향 조절하여 breadth $N_{\text{eff}}$를 높입니다.

```python
<<<<
    rank_select_quantile: float = 0.33          # top/bottom quantile for L/S selection
====
    rank_select_quantile: float = 0.45          # top/bottom quantile for L/S selection (N_eff breadth 상향)
>>>>
```

### Target File 2: `src/execution/opt_main_futures.py`
- 24bps 고정 비용 정책을 엄격화하고, `bear_market_basket_safe`에 실제 레짐별 PnL을 실측 주입합니다.
- `_basket_spread_diag`에서 `cost_per_bar_bps`로 **고정 24bps**를 강제합니다.

```python
    # 24bps 고정 거래비용 적용 (상각 제거, 물리적 진실 보존)
    _fixed_cost = 24.0

    report = evaluate_alpha(
        alpha_long_2d=al,
        alpha_short_2d=as_,
        realized_fwd_ret_2d=real_resid,
        inference_signed_2d=_dense_pred_for_eval,
        btc_close_1d=btc_close_1d,
        n_trials=_n_trials_dsr,
        horizon_bars=horizon,
        cost_floor_bps=_fixed_cost,  
    )
```

---

## 4. Verification (검증 및 대조군 평가)

### 4.1. 다변화 효과(Breadth) 최적화 OOS 평가
```bash
# rank_select_quantile 수정 후 OOS IC 및 N_eff 실측 확인
uv run python -m src.execution.opt_main_futures --mode alpha
```
**통과 판정 모니터링:**
- $N_{\text{eff}}$ 가 $3.5$ 이상으로 상향되었는지 확인.
- 이에 따라 `breakeven_ic` (be_raw)가 $0.025$ 이하로 건전하게 안착하는지 확인.
- `portfolio_ic_above_breakeven` 게이트가 "비용 조작 없이" 정직하게 통과되는지 대조 확인.

### 4.2. Phase 3 타깃 잔차화 실험 개시
- Breadth 조정 후에도 엣지가 부족할 경우, 즉각 `calibrator_target`을 `beta_residualized`로 변경하는 실험(Surgical Plan)을 가동합니다.
