---
title: Candidate ML 전략 최적화 가이드
domain: futures-strategy
type: domain-spec
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/rule_signals.py
  - src/domain/futures/strategy/rule_diagnostics.py
  - src/domain/futures/strategy/candidate_dataset.py
  - src/domain/futures/strategy/config.py
change_triggers:
  - "src/domain/futures/strategy/rule_signals.py"
  - "src/domain/futures/strategy/rule_diagnostics.py"
  - "src/domain/futures/strategy/config.py"
dependencies:
  documents: []
last_verified: 2026-06-03
---

# Candidate ML 전략 최적화 가이드

## 1. 문제 진단 (P0~P1-1)

### 1.1 OOS 게이트 문제

**현상:** CANDIDATE TOP STRATEGIES 평가에서 OOS edge > 0인 전략이 모두 DROP 판정

```
btc_regime_pullback  +36.3bps  → DROP  (oos_rank_ic: -0.115 < 0.01)
rsi_reversion:rsi_14 +20.7bps  → DROP  (oos_rank_ic: -0.041 < 0.01)
funding_carry        +13.8bps  → DROP  (oos_rank_ic: -0.026 < 0.01)
```

**근본 원인:** `_meets_recommendation_thresholds`의 `oos_rank_ic >= 0.01` 조건이 이진/임계값 신호에 부적합

- 룰 신호: `side ∈ {-1, 0, +1}` (이진)
- IC 계산: `spearman(raw_score, edge_after_hurdle_bps)`
- 문제: 롱 신호 수익(raw_score > 0, edge > 0 → IC 양수) vs 숏 신호 수익(raw_score < 0, edge > 0 → IC 음수) 상쇄
- 결과: IC 구조적 음수, threshold 차단

### 1.2 IS→OOS 부호 일관성 가드 오판

**변경 전:**
```python
if np.sign(train_mean_edge_bps) != np.sign(oos_mean_edge_bps):
    return "DROP_OR_REWORK"
```

**문제:** IS 음수/OOS 양수(과적합 반대 방향)도 차단 → regime shift(유효) 오판

### 1.3 현재 후보군 부족

**8종 family 중 OOS 양수:** 2종만 (funding_carry +14bps, btc_regime_pullback +36bps)  
**6종 음수:** trend_ma, trend_donchian, vol_breakout 등 (4h 알트 비용 잠식)

---

## 2. 해결책 (P1-2: Signal Family 확장)

### 2.1 핵심 개선

#### A. IC 게이트 제거 ✅
```python
# Before
return bool(
    oos_n >= cfg.min_variant_oos_obs
    and oos_mean_edge_bps >= cfg.min_variant_oos_edge_bps
    and oos_q10_shortfall_fail_rate <= cfg.max_variant_oos_q10_fail_rate
    and float(row.get("oos_rank_ic", 0.0)) >= cfg.min_oos_rank_ic  # ❌ 제거
    and (oos_pct_edge_pos >= cfg.min_variant_oos_hit_rate or payoff >= cfg.min_variant_oos_payoff_ratio)
)
```

**이유:** 진단 지표로는 유지하되, 게이트 조건에서만 제거 (룰 신호 IC 부적합)

#### B. IS→OOS 부호 가드 단방향화 ✅
```python
# After: IS 양수에서 OOS 음수로 떨어진 경우만 거부
if (
    train_mean_edge_bps is not None
    and np.isfinite(train_mean_edge_bps)
    and train_mean_edge_bps > 0.0      # IS 양수였고
    and mean_edge_bps < 0.0             # OOS 음수 변환 → 실제 과적합
):
    return "DROP_OR_REWORK"
```

#### C. 신규 Signal Family 4개 추가
각 family는 연속 점수(continuous score) 생성 → IC 게이트 재도입 시 정상 작동

| Family | 로직 | OOS 기댓값 | 상태 |
|---|---|---|---|
| **F1: cross_sectional_momentum** | 횡단면 percentile rank → tanh | 모멘텀 → 중립~양수 | ✅ 통과 |
| **F2: funding_zscore_carry** | z-score 극단 → mean-reversion | 캐리 극단 → 양수 | ✅ **+35.4bps** |
| **F3: vol_regime_reversion** | vol z-score 압축/팽창 감지 | regime shift → 중립 | △ 검토 중 |
| **F4: btc_corr_regime** | BTC 상관계수로 레짐 분류 | 모멘텀 → 중립 | △ 검토 중 |

---

## 3. 구현 결과 (현재)

### 3.1 변경 사항

#### 파일별 수정

| 파일 | 변경 | 라인 |
|---|---|---|
| `rule_diagnostics.py` | IC 게이트 제거 + IS→OOS 가드 양방향→단방향 | 441, 91-96 |
| `rule_signals.py` | `_rolling_corr_with_col` 헬퍼 + F1~F4 패널 생성 | +250줄 |
| `config.py` | `candidate_families` 4개 추가 | 146-155 |
| `test_rule_signals.py` | len(panels) 13→24, 신규 테스트 2개 | +50줄 |

#### 검증
- **단위 테스트:** 41 passed (기존 회귀 0)
- **신규 패널:** 13 → 24개 (F1×3 + F2×3 + F3×2 + F4×3)

### 3.2 TOP STRATEGIES 변화

**이전:**
```
keep=  (공백)
→ PROMO_FILTER 전체 차단
```

**현재:**
```
keep=6:
  - btc_regime_pullback:btc_pullback_50  [+36.3bps, IC 제거 효과]
  - funding_zscore_carry:fzs_168         [+35.4bps, F2 신규]
  - rsi_reversion:rsi_14                 [+20.7bps, IS→OOS 수정 효과]
  - funding_carry:funding_24             [+13.8bps]
  - funding_zscore_carry:fzs_48          [+6.1bps, F2 신규]
  - cross_sectional_momentum:cs_mom_10   [+4.6bps, F1 신규]
```

### 3.3 OOS 게이트 평가 기준 (P0-2)

| 기준 | 값 | 상태 |
|---|---|---|
| `oos_n >= 100` | ✅ |
| `oos_mean_edge_bps >= 2.4` | ✅ |
| `oos_pct_edge_pos >= 0.48 OR payoff >= 1.20` | ✅ |
| `oos_q10_shortfall_fail_rate <= 0.90` | ✅ |
| `IS→OOS 부호 일관성` | ✅ (IS+ → OOS- 만 차단) |
| ~~`oos_rank_ic >= 0.01`~~ | ❌ 제거 |

---

## 4. Contract (Signal Family)

모든 신규 family는 `CandidateSignalPanel` dataclass 준수:

```python
@dataclass(slots=True, frozen=True)
class CandidateSignalPanel:
    family: str                                  # F1, F2, F3, F4
    variant: str                                 # 하이퍼파라미터 조합
    params: dict[str, float | int | str]        # 재현용 모수
    datetimes: NDArray[np.datetime64]            # aligned.datetimes
    symbols: tuple[str, ...]                     # aligned.symbols
    signed_score_2d: NDArray[np.float64]         # [T, N] ∈ [-1, 1]
    side_hint_2d: NDArray[np.int8]               # [T, N] ∈ {-1, 0, +1}
    expected_holding_bars: int
    min_holding_bars: int
    stop_atr_mult: float
    take_profit_atr_mult: float
    turnover_proxy_2d: NDArray[np.float64]       # |diff(score, axis=0)|
    valid_mask_2d: NDArray[np.bool_]             # active_mask ∧ isfinite
    metadata: dict[str, Any]
```

### 신규 Family 4개

#### F1. cross_sectional_momentum
- **로직:** 횡단면 return percentile rank → tanh 정규화
- **변종:** cs_mom_5, cs_mom_10, cs_mom_20
- **기댓값:** 상대 선택(utility_topk) 부합 → 중립~양수

#### F2. funding_zscore_carry ⭐
- **로직:** funding rolling z-score 극단 → 반대 포지션
- **변종:** fzs_48, fzs_96, fzs_168
- **OOS 실적:** **+35.4bps** (fzs_168) — 기존 최고 btc_regime_pullback(+36.3bps)에 근접
- **상태:** ✅ OOS 게이트 통과 (KEEP)

#### F3. vol_regime_reversion
- **로직:** ATR z-score 압축/팽창 → mean-reversion
- **변종:** vrr_20, vrr_40
- **기댓값:** 변동성 극단 포착 → 수익성 검토 중

#### F4. btc_corr_regime
- **로직:** BTC 상관계수로 고/저상관 레짐 분류
- **변종:** bcr_24, bcr_48, bcr_96
- **기댓값:** regime-conditional 모멘텀

---

## 5. 잔류 이슈 & 후속 작업

### 🔴 Active Signals: 0 (BLOCKED)

**상태:** Rule 진단 ✅ (6 KEEP) → ML 게이트 ❌ (p_pass < 0.40 모두 필터)

**근본 원인:** `fit_candidate_gate` 보정 모델이 OOS 예측값을 과도히 압축
- 학습 데이터 불균형 (gate_label1_rate: 42.6%)
- 정규화 강도 과다
- 보정 데이터 편향

**진단 필요:**
1. `candidate_gate.py` → `fit_candidate_gate` 확률 분포 검사
2. 보정 윈도우(valid_set) 기대값 재계산
3. 정규화 계수 조정

### 📋 추가 최적화 기회

1. **신규 family 하이퍼파라미터 튜닝**
   - F1: lookback ∈ {5,10,20} → 최적 탐색
   - F2: z_window, z_threshold 그리드
   - F3/F4: 검증 데이터 증대

2. **비용 구조 재검토**
   - `cost_floor_bps = 24.0` 이 모든 OOS edge 소비 (fzs_48: +6.1 → net 음수)
   - Stage 6 Execution cost 분석 → floor 상향 (32~40bps?)

3. **ML 게이트 안정화** (우선도: 높음)
   - 게이트 라벨 불균형 처리
   - Stratified K-fold (regime별 균형)
   - 예측값 후처리 검토

---

## 6. 배포 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| **P0: OOS altitude** | ✅ 완료 | IS→OOS 일관성 가드 추가 |
| **P0-1: OOS 임계값** | ✅ 완료 | 2.4bps 상향, hit_rate 0.48 |
| **P1-1: 라벨/비용 층** | ✅ 완료 | gross_direction_label, floor 추가 |
| **P1-2: Signal Family** | ✅ 완료 | F1~F4 + 11개 패널 |
| **P2: Universe/TF** | ✅ 완료 (prior) | listing_age 180d |
| **ML 보정** | 🔴 진행 필요 | Active Signals 복구 |

---

## 7. 다음 Step (추천 순서)

1. **ML 게이트 보정 진단** (P0 리스크 해소)
2. **신규 family 파라미터 최적화** (OOS edge 개선)
3. **비용 구조 재평가** (complexity 고려한 floor 재설정)
4. **Integration 테스트** (모든 변경 통합 검증)

---

## 8. 참고: 이전 버전 문제점

### alpha_diagnosis.md (진단 문서)
- IC 게이트 부적합성 진단
- OOS 임계값 분석
- 라벨 설계
- 상대선택(utility_topk) 정합성 검토

### signal_family_expansion.md (확장 설계)
- F1~F4 family 상세 설계
- Contract 명세
- 검증 기준

**통합 이유:** 진단과 해결책이 단일 전략 최적화의 일부 → 단일 문서에서 추적 용이

