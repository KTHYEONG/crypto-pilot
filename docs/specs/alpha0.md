---
title: Futures ML Alpha Rebuild and Core Repair History (Phases 0-3)
domain: futures-alpha
type: compilation
status: completed
priority: critical
ai_read_policy: always
last_verified: 2026-05-31
---

# Futures ML Alpha Rebuild and Core Repair History (Phases 0-3)

이 문서는 Futures ML Alpha의 재구축(Phase 0)부터 시작하여, 초기 구현 및 평가 파이프라인에서 발견된 치명적인 버그들을 수정한 기록(Phase 1-3)을 통합한 것입니다.

## 0. Phase 0: Rebuild Strategy (Simple Rank-Native Architecture)

### 목적
복잡한 `dual-side ranker + quantile calibrator + conservative EV + rank-sized emit + alpha_p95 wall` 구조를 걷어내고, `LightGBM LGBMRanker` 단일 모델 기반의 단순하고 정직한 아키텍처로 전환.

### 핵심 설계 및 구현
- **Model:** LambdaMART 단일 ranker (`LGBMRanker`)의 cross-sectional ranking 학습.
- **Feature Set:** Price momentum, volatility, liquidity, funding, basis, market context 등 32개 이내의 단단한 피처셋 유지.
- **Signal Contract:** `return_unit_grinold_rank`
  ```python
  alpha_signed = score_z * sigma_resid_trailing * ic_lcb_fold
  ```
- **Gate Honesty (Admission Gates):**
  - `rank_ic_lcb >= breakeven_ic_eff`
  - `basket_net_bps_lcb_24bps > 0`
  - `ev_cost_ratio >= 1.5`, `turnover_cost_ratio <= 0.35`
- **주요 수정 파일:** `src/domain/futures/strategy/` 내 `config.py`, `features.py`, `ranker.py`, `ml_builder.py`, `diagnostics.py`, `alpha_evaluation.py`.

---

## 1. Phase 1: Realized Return Shifting Bug Fix (Bug-Fix)

### 문제
OOS 평가 시 `reindex`를 수행한 후 `shift(-h)`를 적용하여 시계열 데이터 유실 및 수익률 계산 왜곡 발생. 이로 인해 `portfolio_ic=0.0000`, `basket_net_bps=nan` 등 지표 파괴됨.

### 해결
전체 연속 시계열에서 수익률을 먼저 계산한 뒤 대상 인덱스로 `reindex` 하도록 수정하여 시계열 무결성 보존.
```python
# Fixed pattern
fwd_ret = np.log(df["close"].shift(-horizon) / df["close"]).reindex(target_index)
```

---

## 2. Phase 2: Gate Honesty and OOS Pipeline Repair (Spec-Lite)

### 문제
- `clip_pres` 임계값(0.5)이 너무 낮고, `DSR` 계산 시점이 pre-clip으로 설정되어 지표 왜곡.
- OOS 평가 단계에서 심볼 커버리지 불일치(`common_syms` vs `alpha_panel` symbols)로 인해 `common_idx`가 비어버리는 현상.

### 해결
- **Gate 보정:** `clip_pres` 임계값 상향 (0.7), `DSR`을 post-clip 기준으로 변경.
- **Pipeline 수정:** `_oos_dt_mask` 계산 시 C3 필터링 전 전체 심볼을 사용하여 시간 인덱스를 확보하도록 로직 수정.

---

## 3. Phase 3: Signed-Rank Contract Repair (Bug-Fix)

### 문제
`rank_score_long - rank_score_short` 수식 사용 시, 단일 랭커 모드에서 롱/숏 점수가 동일하여 신호가 0으로 상쇄됨 (`[RANK-QUALITY L1] breadth=0.0`).

### 해결
`derive_signed_rank_signal` 함수를 도입하여 동일 점수 모드를 감지하고, 이 경우 뺄셈 대신 원본 점수를 유지하도록 계약 보수(Contract Repair) 수행.
```python
# src/domain/futures/strategy/alpha_evaluation.py
def derive_signed_rank_signal(long_arr, short_arr):
    # If cofinite values are same, return raw signed score.
    # Else, return 0.5 * (long - short).
```

---

## 4. Verification History (Final State of Phase 3)

Phase 3 완료 시점의 smoke run 결과 (`--mode alpha`):
- `[OOS-DIAG] rank_cols=96 finite_rows=1417` (인덱스 복구)
- `[RANK-CONTRACT] c3_signed_nz=0.586` (신호 복구)
- `[RANK-IC C3] ic=0.0174, breadth=8.79`
- `ALPHA_PASS: FALSE` (기능적 버그는 모두 해결되었으나, 전략의 실력이 비용을 극복하지 못한 상태)

---
*이후의 개선 사항(Calibration 등)은 `alpha1.md`에서 계속됩니다.*
