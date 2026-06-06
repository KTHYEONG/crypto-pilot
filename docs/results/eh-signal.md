# Signal Evaluation Hardening — 논리 압축

> **목적**: "여러 좋은 후보군 → ML 동적 배분 → 복리 극대화" 달성을 위한 signal 평가 기준 재설계 논리 기록.
> 다른 AI / 채팅 세션에서 논리구조 참조용.

---

## 핵심 아이디어 (Why)

전략: 여러 signal variant(family×param 조합)를 동시에 유지하고, ML이 각 bar에서 가장 유리한 variant를 동적으로 선택.
문제: variant 품질 필터가 없으면 IS에서 음수 edge인 variant도 ML 학습 데이터에 포함됨 → **garbage in, garbage out**.

---

## Signal 구조 개요

```
rule_signals.py
  └── family × variant → Candidate Events (sparse, per bar)
        └── Triple-Barrier Labeling → net_event_bps (taker fee + funding + hurdle 차감)
              └── ML feature engineering → fit/cal/oos split
```

- **family**: 전략 종류 (trend-pullback, fading-zscore, bb-compress 등)
- **variant**: 파라미터 조합 (예: tpc_50_200, tpc_20_100, fzs_96 등)
- **net_event_bps** = gross_bps - execution_cost_bps - funding_bps - hurdle_bps
- **edge_after_hurdle_bps** = net_event_bps (backward-compat alias)

---

## Signal Pre-Qualification (Layer 0)

### 문제
IS 구간에서 `mean(edge_after_hurdle_bps) < 0`인 variant가 ML 학습에 포함됨.
Huber regression이 이 잡음 패턴을 학습 → OOS residual 예측을 왜곡.

### 기준 (IS fit window 기준)

| 조건 | 임계값 | 처리 |
|---|---|---|
| 최소 이벤트 수 | `n_events >= 30` | 미달 → global prior만 (uniqueness_weight=0) |
| IS 평균 edge | `mean_edge_bps > 0` | 음수 → 학습 기여 제거 (uniqueness_weight=0) |

### 구현 방식
- Hard exclusion이 아닌 `uniqueness_weight = 0` 설정
- OOS inference 시에는 여전히 global prior 적용 (비율 관리)
- 적용 시점: `build_candidate_dataset(is_fit_split=True)` 호출 시에만

### 실측 효과
- fold당 26~29% 이벤트 제거 (382~608개/fold)
- 학습 데이터 정제 완료. 단, 남은 데이터만으로 IC 0.02 달성은 아직 불충분

---

## Triple-Barrier Labeling 핵심 계약

- **SL**: ATR 기반 동적 설정 (진입 시점 기준)
- **TP**: ATR × TP/SL ratio
- **Time Exit**: `entry_idx + expected_holding_bars`의 next-open 가격 (close 미사용 → look-ahead 방지)
- **비용 모델**: baseline floor는 7.5bps지만 label 단계에서는 `max(dynamic_cost, taker_round_trip_bps)`를 사용한다. signal stress는 고정 11.25bps 대체가 아니라 `stress_multiplier × ex_ante_cost_bps`를 event별로 적용해야 한다.
- **구조적 비대칭**: ATR SL < TP → base hit_rate < 50% (≈43%) → gate binary classification 상한 구조적으로 0.496

---

## 경제적 최소선 (Signal → Fold Survival 연결)

| 기준 | 값 | 근거 |
|---|---|---|
| `min_fold_realized_edge_bps` | 15.0 bps | RT cost(7.5bps) × 2배 최소선 |
| `min_cagr_for_promotion` | 15% | crypto 위험 프리미엄 최소선 |
| `min_fold_selected_events` | 20 | 통계적 최소 샘플 수 |
| `min_wf_fold_pass_ratio` | 0.60 | 4 fold 중 최소 3/4 통과 |

---

## 현재 상태 (2026-06-06, integrity fix 후)

- Signal Pre-Qual 구현 완료 및 동작 확인
- raw baseline과 promoted signal을 분리해서 봐야 함:
  - `rule_only_equal_size`: `n=111101`, `decision_bars=1296`, `mean=8.9bps`, `stress_mean=2.9bps`, `hac_t=0.12` → FAIL
  - `rule_promo_no_leak`: `n=2355`, `decision_bars=831`, `mean=12.9bps`, `stress_mean=6.9bps`, `hac_t=1.42` → PASS
- 따라서 "signal 전체가 강하다"가 아니라 "promotion된 subset만 marginally survivable"가 더 정확한 해석이다.
- shadow profile의 realized 결과는 OOS profile search 산물이므로 진단 전용이다.
