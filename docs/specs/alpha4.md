---
title: ML Alpha idiosyncratic redesign — target residualization & feature engineering
domain: futures-alpha
type: spec
status: proposal
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/labels.py
  - src/domain/futures/strategy/ml_builder.py
  - tests/unit/domain/futures/strategy/test_ml_config.py
last_verified: 2026-05-31
---

# ML Alpha Idiosyncratic Redesign

> **목표:** 24bps의 엄격한 물리 마찰비용 장벽을 우회하지 않고(비용 상각 기각), 모델 학습의 근본적 타깃 및 피처 설계를 개별 자산의 특이적 성분(Idiosyncratic Alpha)에만 집중하도록 재설계하여 **Breakeven IC(0.029+)**를 정면 돌파한다.

---

## 1. 근본적 한계 분석 및 해결 논리 (Why & How)

### 1.1. 물리 비용 24bps 장벽과 비용 상각(옵션 A) 기각
- **진단:** 예측 호라이즌이 길어진다고 해서 테이커 수수료(Taker Fee)와 슬리피지(Slippage)로 구성된 물리적 round-trip 거래비용(24bps)이 상각(amortize)되어 감소하지 않습니다. 18h 호라이즌에 대해 임의로 8bps를 대입하는 등의 "끼워 맞춤형 비용 깎기"는 백테스트의 착시를 유발하여 live 투입 시 즉각 손실로 이어집니다.
- **결정:** **비용 상각안(옵션 A)을 전면 기각**하고 모든 평가에 24bps 물리 장벽을 고수합니다.

### 1.2. 포트폴리오 다변화(Breadth) 최적화의 수용
- **진단:** 기존의 L/S 선택 분위수인 `0.33`은 베팅 집중도를 높여 실질 베팅 개수 $N_{\text{eff}}$를 2.2 수준으로 묶어두었습니다. 이 경우 통계적 요구 breakeven IC가 무려 `0.0343`에 달하게 됩니다.
- **결정:** 분위수를 **`0.45`**로 상향(Phase 2.5)하여 실질 독립 베팅 개수를 $N_{\text{eff}} = 3.08$ 이상으로 다변화합니다. 분산 효과로 breakeven IC 요구치가 **`0.029`** 이하로 현실화됩니다.

### 1.3. 핵심 병목: Target Residualization (학습 타깃 잔차화) 부재
- **진단:** 기존 모델은 `rank_target_mode`가 `"forward_gross_rank"`로 되어 있어, 시장 지수 성분(Beta)이 그대로 묻어있는 gross log return의 랭킹을 학습했습니다. 그 결과 모델이 BTC 등 시장 전체 지수의 움직임(Market Beta)에 동조하여 과적합되었습니다. OOS 평가 시에는 베타를 제거한 순수 특이 수익률(`real_resid`)을 대상으로 `net_ic`를 구하므로, 시장 동조형 모델의 OOS IC가 `0.0126` 수준으로 붕괴하여 breakeven 장벽을 넘지 못했던 것입니다.
- **결정:**
  1. **`rank_target_mode` 강제 전환:** `"forward_gross_rank"` $\rightarrow$ **`"cs_residual"`**
     - 모델이 학습 단계에서부터 시장 베타가 정교하게 OLS residualization되고 크로스섹션 중앙화(CS-demean)된 순수 idiosyncratic alpha 랭크(`signed_net_ret`)만 직접 학습하도록 강제합니다.
  2. **`calibrator_target` 강제 전환:** `"gross"` $\rightarrow$ **`"beta_residualized"`**
     - magnitude를 추정하는 calibrator 역시 gross return 대신 잔차 리턴(`exec_net_ret`)의 절댓값을 목표로 학습하여 오버피팅을 원천 방지합니다.

---

## 2. Idiosyncratic 피처 엔지니어링 보강 (Phase 3)

시그널 엣지(`resid_ic`) 자체를 `0.030+` 이상으로 끌어올리기 위해 피처 엔지니어링 단에서 idiosyncratic 팩터를 대대적으로 보강합니다.

1. **BTC Lead-Lag Residual Feature (`feat_btc_lag_resid`)**
   - 개별 코인의 리턴에서 BTC의 1-bar, 2-bar lag 리턴을 OLS 잔차화하여, 단순 시장 추종이 아닌 "시장 대비 반응 속도의 차이" 및 "시차적 특이 엣지"를 추출합니다.
2. **Funding-Basis Divergence Feature (`feat_funding_basis_diverged`)**
   - 전체 선물 유니버스의 평균 펀딩비 대비 개별 코인의 펀딩비 이격도를 계산하여, 시장 참여자들의 비정상적 극단 쏠림(Crowding) 및 숏스퀴즈/청산 압력을 포착합니다.
3. **Sector-Relative Momentum Feature (`feat_sector_rel_mom`)**
   - 개별 코인이 속한 테마/섹터(예: L1, DeFi, Meme)의 평균 모멘텀을 차감한 섹터 중립 모멘텀을 피처로 활용하여 섹터 베타를 상쇄합니다.

---

## 3. 학습 설정 및 목적 함수 튜닝

- **목적 함수 및 규제 강화:**
  - `LGBMRanker`의 과적합을 제한하기 위해 L2 규제 파라미터 `ranker_lambda_l2`를 `20.0` $\rightarrow$ `30.0`으로 강화합니다.
  - 트리 깊이 `max_depth`를 `4`로, `num_leaves`를 `15`로 타이트하게 고정하여 복잡한 노이즈 패턴 학습을 차단합니다.
- **Purge & Embargo Walk-Forward 보존:**
  - 레이블 호라이즌 겹침으로 인한 미래 정보 누출(Data Leakage)을 방지하기 위해 `purge_bars = 12` 및 `embargo_bars = 12` 설정을 엄격히 유지합니다.

---

## 4. Blueprint & Surgical Plan

### 4.1. Target Files & Action Plan

#### 1. `src/domain/futures/strategy/config.py` [REPLACE]
`StrategyMLConfig` 내의 `rank_target_mode` 및 `calibrator_target` 디폴트 값을 완전한 idiosyncratic alpha 훈련 계약으로 변경합니다.

```python
# ... existing code ...
    # Calibrator target: which return series to use as y_ev for magnitude learning.
    # "beta_residualized" = current default: exec_net_ret after beta-resid, pre-CS-demean
    # "gross"             = raw log return minus funding only (no beta removal, no fee)
    calibrator_target: Literal["beta_residualized", "gross"] = "beta_residualized"
    model_family: Literal["lgbm_regression", "lgbm_huber", "lgbm_lambdarank"] = "lgbm_regression"
    ranking_mode: Literal["pointwise", "group_ndcg"] = "group_ndcg"
    # False: skip ranker stage; calibrator uses zero rank_score
    # (C3 ablation A/B — empirically better OOS IC)
    ranker_enabled: bool = True
    rank_target_mode: Literal["cs_residual", "forward_gross_rank"] = "cs_residual"
# ... existing code ...
```

#### 2. `tests/unit/domain/futures/strategy/test_ml_config.py` [REPLACE]
기본 설정 변경에 맞춰 config 기본값 검증 테스트 코드를 함께 갱신합니다.

```python
# ... existing code ...
def test_strategy_ml_config_new_modes_defaults() -> None:
    cfg = StrategyMLConfig()
    assert cfg.rank_target_mode == "cs_residual"
    assert cfg.calibrator_target_mode == "rank_confidence"
    assert cfg.post_cost_admission_mode == "rank_cs_neutral"
    assert cfg.oos_ic_target_source == "forward_gross_ret"
# ... existing code ...
```

---

## 5. Verification Plan

변경 완료 후, 다음 Verification 명령어를 통해 OOS(Out Of Sample) 테스트 결과가 24bps 고정 마찰비용 장벽 하에서도 `portfolio_ic_above_breakeven` 게이트를 정직하게 돌파하고 `ALPHA_PASS`가 `TRUE`로 전환되는지 검증합니다.

### 5.1. 단위 테스트 검증
```bash
uv run pytest tests/unit/domain/futures/strategy/test_ml_config.py -k "test_strategy_ml_config_new_modes_defaults" --tb=short
```
- **기대 결과:** `test_ml_config.py` 내의 기본값 검증 단정문이 성공적으로 통과(`passed`).

### 5.2. OOS Alpha 파이프라인 전체 실행 검증
```bash
uv run python -m src.execution.opt_main_futures --mode alpha
```
- **기대 결과:**
  1. 학습 타깃으로 `cs_residual` 및 `beta_residualized`가 자동 바인딩되어 훈련 진행.
  2. OOS `resid_ic` (C3 RANK-IC) 가 기존 `0.0141` $\rightarrow$ **`0.030+`** 수준으로 비약적 향상.
  3. `portfolio_ic_above_breakeven` 게이트 판정이 **`✅ (Passed)`**로 변경됨.
  4. 최종 `ALPHA_PASS`가 **`TRUE`**로 전환됨.
