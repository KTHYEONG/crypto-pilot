---
title: MHS Architecture - 03. Committee & Regime Smoothing
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/committee.py
  - src/mhs/pipeline/stages/committee.py
  - src/mhs/params.py
  - src/mhs/types.py
change_triggers:
  - src/mhs/committee.py
  - src/mhs/pipeline/stages/committee.py
last_verified: 2026-09-03
---

# 03. 경제적 신호 위원회 및 레짐 적응형 트랜치 평활

## 1. 개요 (Overview)
단일 알파 신호나 단일 호라이즌에만 의존하는 포트폴리오는 시장의 구조적 레짐 변화(유동성 경색, 테이커 주도 장세, 횡보장 등)에 취약합니다.

MHS는 서로 다른 경제적 동인(Flow Imbalance, Cross-Sectional Momentum, Idiosyncratic Residual, Skewness 등)을 대변하는 $k=5$개의 독립 신호를 위원회(Committee)로 구성하고, 전략 자체의 자기상관(Autocorrelation)에 따라 평활 강도를 실시간 조절하는 **레짐 적응형 트랜치 평활(Regime Adaptive Tranche Smoothing)**을 통해 최적의 위험 대비 수익을 달성합니다.

---

## 2. 경제적 신호 패밀리 및 위원회 구성 (Committee Architecture)

위원회는 상관관계가 낮고 상호 보완적인 5개의 신호 멤버로 구성됩니다 ([`COMMITTEE_MEMBERS`](file:///home/kth/crypto-pilot/src/mhs/params.py)):

### 기본 멤버 세트: `flow_momentum`
1. **`flow_imb_720h`**: 최근 30일(720시간) 테이커 매수/매도 거래대금 불균형 (장기 유동성 유입 추세).
2. **`flow_imb_168h`**: 최근 7일(168시간) 단기 테이커 공격적 매수세 불균형.
3. **`xs_mom_336h`**: 14일 횡단면 가격 모멘텀.
4. **`xs_idio_mom_336h`**: 시장 베타(BTC) 영향을 직교화하여 제거한 순수 개별 자산 고유 모멘텀 (Idiosyncratic Momentum).
5. **`mom3_skew_168h`**: 최근 7일 수익률의 왜도(Skewness) 기반 모멘텀 비대칭성 신호.

*(대안 세트: `risk_premia` - `flow_imb_720h`, `flow_imb_168h`, `mom3_skew_168h`, `lowvol_168h`, `rev_24h`)*

---

## 3. Train-Only 증거 기반 가중치 (Evidence Weighting)

위원회 각 멤버의 결합 가중치는 엄격하게 Train 구간의 실측 성과를 바탕으로 인과적으로 결정됩니다 ([`train_evidence_weights`](file:///home/kth/crypto-pilot/src/mhs/committee.py)).

### 1) 부호 안전 비용 분해 (Sign-Safe Cost Decomposition)
- 서로 다른 두 비용 티어(예: Low 2.64bps, High 6.07bps)에서의 PnL 패널을 분석하여 순수 총이익(`gross`)과 회전율 비용(`turnover_cost`)을 완벽히 분리합니다 ([`decompose_cost`](file:///home/kth/crypto-pilot/src/mhs/committee.py)):
  $$\text{turnover\_cost} = \frac{\text{net\_low} - \text{net\_high}}{\text{bps\_high} - \text{bps\_low}}$$
  $$\text{gross} = \text{net\_low} + \text{turnover\_cost} \times \text{bps\_low}$$
- **Sign-Safe 회귀 방어**: 신호에 음수(-) 가중치를 부여하더라도 거래 비용은 항상 양수의 손실로 작용해야 하며, 비용이 이익으로 환급되는 수학적 결함을 원천 방지합니다.

### 2) 단일 배포 구성 불변식 (I-SINGLE-CONFIGURATION)
- 전략이 실제로 배포(Deploy)할 위원회 가중치 믹스는 상위 경계(`top_level`)에서 단 한 번 확정됩니다.
- 검증 폴드(Anchored-Purged Folds)들은 폴드마다 가중치를 다시 피팅하는 누출을 금지하고, 실제 배포될 동일한 단일 가중치 믹스를 그대로 평가하여 실전 일치성을 보장합니다.

---

## 4. 위원회 레짐 적응형 트랜치 평활 (Regime Adaptive Tranche)

### 1) 기존 고정 평활의 딜레마
- **노이즈 억제 목적의 고정 평활**: 신호를 여러 행(Tranche)에 걸쳐 평균화하면 횡보장의 휩소(Whipsaw) 손실과 거래 비용은 줄어들지만, 강력한 추세가 시작될 때 포지션 진입이 지연되어 알파 기회비용이 발생합니다.
- 고정 평활 하나로는 항상 한쪽 레짐(추세 지속 vs 휩소 횡보)을 희생시키는 트레이드오프가 존재했습니다.

### 2) 레짐 적응형 솔루션 ([ADR-20260817-MHS-COMMITTEE-REGIME-ADAPTIVE-TRANCHE])
위원회 북 자신의 **인과적 롤링 Lag-1 자기상관(Autocorrelation)**을 실시간 측정하여 평활 여부를 동적으로 전환합니다 ([`COMMITTEE_REGIME_ADAPTIVE_WINDOW = 15`](file:///home/kth/crypto-pilot/src/mhs/params.py)):

```text
위원회 북의 Causal Trailing Lag-1 Autocorrelation 측정
       │
       ├─ [음수 Autocorr < 0 : Whipsaw 레짐]
       │    └──► 3행 트랜치 평활 (COMMITTEE_TRANCHE_COUNT = 3) 적용
       │         (진동 억제, 불필요한 체결 비용 및 슬리피지 방어)
       │
       └─ [양수 Autocorr >= 0 : Trend-Continuation 레짐]
            └──► Raw 신호 즉각 채택 (Tranche Smoothing = 1)
                 (지연 없는 신속한 추세 편승, 알파 수익 극대화)
```

### 3) 실측 개선 효과
- 고정 평활의 트레이드오프를 완벽히 제거하여 전략의 기본 Sharpe를 **0.5257에서 1.0792로 2배 이상 급상승**시켰으며, 비용 3배 스트레스 상황에서도 양의 수익률(+0.8334)을 안정적으로 방어했습니다.

---

## 5. 성장 위험 포락선 (Growth Risk Envelope)

자본 보존과 기하학적 복리 성장을 위해 MHS는 불변의 단일 위험 제약 조건인 [`GrowthRiskEnvelope`](file:///home/kth/crypto-pilot/src/mhs/params.py)을 등록하여 운용합니다:

```python
@dataclass(frozen=True, slots=True)
class GrowthRiskEnvelope:
    name: str                   # 'conservative', 'balanced', 'growth_extreme' 등
    max_drawdown: float         # 최대 허용 낙폭 예산 (예: 0.25 ~ 0.60)
    max_drawdown_prob: float    # 해당 낙폭 초과 허용 확률 (예: 0.10)
    ruin_fraction: float        # 파산 정의 잔고 비율 (기본 0.60, 40% 이상 손실)
    max_ruin_prob: float        # 파산 허용 상한 확률 (기본 0.01, 1%)
    horizon_years: float        # 운용 지평 (3.0년)
    leverage_ceiling: float     # 최대 레버리지 상한 (1.0 ~ 3.0)
```

- 현재 CLI 및 프로덕션 기본값은 [`growth_extreme_budgeted`](file:///home/kth/crypto-pilot/src/mhs/params.py)으로, 낙폭 예산 60% 및 파산 확률 1% 미만의 엄격한 제약 하에서 자본 효율성을 극대화합니다.

---

## 6. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 클래스 |
|---|---|---|
| 비용 분해 및 가중치 | [src/mhs/committee.py](file:///home/kth/crypto-pilot/src/mhs/committee.py) | `decompose_cost`, `score_weighted_net`, `train_evidence_weights` |
| 위원회 파이프라인 스테이지 | [src/mhs/pipeline/stages/committee.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/committee.py) | `build_committee` |
| 위원회 및 리스크 파라미터 | [src/mhs/params.py](file:///home/kth/crypto-pilot/src/mhs/params.py) | `COMMITTEE_MEMBERS`, `COMMITTEE_REGIME_ADAPTIVE_WINDOW`, `GROWTH_RISK_ENVELOPES` |
| 성장 최적 리스크 산출 | [src/mhs/scaling.py](file:///home/kth/crypto-pilot/src/mhs/scaling.py) | `solve_growth_optimal_risk` |
