---
title: ML Regime-Conditional 동적 배분 통합 Spec
domain: futures.strategy
type: domain-spec
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/regime_evaluation.py
  - src/domain/futures/strategy/market_regime.py
  - src/domain/futures/strategy/candidate_edge.py
  - src/domain/futures/strategy/candidate_dataset.py
  - src/execution/opt_main_futures.py
change_triggers:
  - "src/domain/futures/strategy/market_regime.py"
  - "src/domain/futures/strategy/regime_evaluation.py"
  - "src/domain/futures/strategy/candidate_edge.py"
  - "src/domain/futures/strategy/candidate_dataset.py"
dependencies:
  documents:
    - docs/results/result.md
    - docs/architecture/backtest-logic.md
last_verified: 2026-06-07
---

# 🎯 Objective

시장 레짐(CUSUM 6-state)에 따라 복수의 검증된 전략 중 적재적소의 것을 동적으로 배분하여 복리 자산증식하는 아키텍처를 확정한다.

---

# ✅ 완료된 작업 (압축 요약)

## Phase R1 — 평가 프레임워크 구현 (완료)
- `regime_evaluation.py` 신규: `RegimeScoreCard` + `evaluate_regime_classifier(C2~C5)` 구현
- `rule_diagnostics.py`: `log_regime_scorecard` 테이블 surface
- `opt_main_futures.py` + `config.py`: `--phase regime` early-exit 추가 (universe↔signal 사이)
- `test_regime_evaluation.py`: 20개 단위테스트, 93% coverage

## Phase R2 — 실측 및 결정 (완료, 2026-06-07)

| 축 | 점수 | 판정 | 근거 |
|----|-----:|------|------|
| C1 Look-ahead | 9/10 | ✅ | CUSUM/EMA causal, entry-1 소비 |
| C2 Persistence | 8/10 | ✅ | dwell=6.0, tr=0.116 |
| C3 Distinctness | 6/10 | ❌ | kw_p≈0이나 flip=False (방향 전환 없음) |
| C4 OOS Stability | 4/10 | ❌ | rho=0.100 << 0.5 |
| C5 Coverage | 10/10 | ✅ | n_eff=4.65, transition=0%(dead state) |

**가중 종합(C2-C5)=0.450. 결정: 분기 B** — 이산 code 배분-게이팅 폐기, 연속 overlay 직행.
주의: ML-Ready 신호 3개 기반 측정 → 신호 풀 6+개 확장 후 **재측정 전제** 하의 잠정 결정.

## Phase R3-0 — 이중 시스템 정리 (완료, 2026-06-07)
제거 목록:
- `opt_data_utils.py`: `infer_regime_codes` stub (항상 0 반환, HMM 제거 잔재), `compute_oos_regime_attribution`, `compute_regime_drift`
- `final_evaluator.py`: 위 3개 함수 import 및 호출 블록
- `dashboard.py`: `REGIME_NAMES`(4-state), `log_oos_regime_attribution`

**CUSUM 6-state `market_regime.py`가 유일 SSOT로 확정.**

---

# 🔶 현재 상태 (Baseline)

```
상태: WF_ELIGIBLE (ML Phase)
Fold: 3/4 PASS | sel=614 | EU_p90 62~104bps | prior_rank +1.4% CAGR
근본 블로커: ML Rank-IC≈0 (ML-Ready 신호 3개×~1500 이벤트, 변별력 부족)
배분 방식: prior_only (이산 code 폐기 → 연속 overlay는 portfolio/dataset 경로에만 활성)
```

SSOT 실행 결과: `docs/results/result.md`

---

# 🏗️ 아키텍처 결정

## 배분 방향 (B 채택)

| 방안 | 의도 부합 | 판정 |
|------|----------|------|
| **B. 레짐별 전략 배분** (regime-conditional prior → Hedge 동적 배분) | ✅ "적재적소" = 레짐에 맞는 전략 | **주 방향** |
| A. 이진 분류 (메타라벨링) | 부분 — 게이트/사이징 레이어 | B 위의 보조 도구 |
| C. prior_only 최적화 | 정적 — "ML이 동적으로" 위배 | fallback 안전망 |

**채택 조합**: ① Regime-conditional prior → ② Hedge 동적 배분 → ③ 메타라벨 사이징

## 연속 overlay 현황 (기존 활성 경로)

| 경로 | 파일 | 상태 |
|------|------|------|
| weight 곱 | `candidate_portfolio.py:858` `signed_w *= overlay_mult` | ✅ 활성 |
| ML 피처 | `candidate_dataset.py:421` `overlay_mult_entry` | ✅ 활성 |
| Regime context | `rule_signals.py:310`, `candidate_dataset.py:659` | ✅ 활성 |

---

# 🔧 잔여 로드맵

## Priority 0 — 신호 풀 확장 (선행 블로커)

**의존도**: Priority 1의 regime-conditional prior가 의미 있으려면 ML-Ready 신호 3개 → **6+개** 확장이 선행 필요. C3/C4 재측정도 이에 의존.

| 작업 | 상태 | 비고 |
|------|------|------|
| ML-Ready 신호 6+개 확보 | ⬜ 대기 | SSOT: 이 파일 |
| C3/C4 재측정 | ⬜ 신호 확장 후 | `--phase ml` 실행 후 scorecard 재기록 |

## Priority 1 — Regime-conditional Prior

**목적**: prior 키 `family:variant` → `family:variant:regime_code`로 확장. ML 없이도 레짐 조건부 전략 선택.

### Surgical Plan

`[FILE]` `src/domain/futures/strategy/candidate_dataset.py`
`[ACTION]` UPDATE — `build_candidate_dataset`의 event_index에 `regime_code` 컬럼 추가
`[INSTRUCTION]` `compute_market_regime_context(aligned).code_1d`를 각 이벤트의 `entry_idx`로 인덱싱. look-ahead 금지: `code[entry_idx - 1]` 소비.

`[FILE]` `src/domain/futures/strategy/candidate_edge.py`
`[ACTION]` UPDATE — `_variant_keys` 레짐 확장 + 계층 backoff
`[INSTRUCTION]` `regime_code` 컬럼 존재 시 키를 `family:variant:regime`으로 확장. 레짐별 obs < `edge_prior_min_obs`이면 `family:variant`로 fallback (계층적 shrinkage).

**Risk**:
- 레짐 분할 시 셀당 obs 감소 → 계층 backoff 필수
- 레짐 경계 prior 불연속 → EB shrinkage로 완화

**완료 기준**:
- [ ] 동일 전략이 레짐별 상이한 prior 생성 확인
- [ ] `--phase ml` 로그에서 레짐별 selection 분화 확인

## Priority 2 — Strategy-level Hedge 동적 배분 (Priority 1 검증 후)

**목적**: 각 전략을 expert로 보고 레짐 조건부 최근 성과로 비중을 동적 갱신.

수학:
```
w_i(t+1) = w_i(t) · exp(η · r_i(t)) / Σ_j w_j(t) · exp(η · r_j(t))
regret ≤ √(T ln N)  (N=전략수, T=기간)
```

`[FILE]` `src/domain/futures/strategy/strategy_allocator.py` (신규)
`[ACTION]` CREATE — `build_candidate_target_weights` 상류에 전략 비중 레이어 삽입

**완료 기준**:
- [ ] ablation `regime_hedge_allocation` 변형 추가
- [ ] CAGR > prior_rank(+1.4%), deploy_ratio ≥ 0.5

## Priority 3 — Meta-labeling 사이징 (Priority 2 검증 후)

`[INSTRUCTION]` 1차 신호 방향 결정 후, 2차 이진 ML이 `p(win|features, regime)`으로 베팅 크기 조절. `barrier_label` 컬럼 재사용.

**완료 기준**:
- [ ] 메타라벨 적용 후 MaxDD 축소 확인

## R3-B — 연속 overlay 정합성 검증 (C3/C4 개선 후)

**조건**: C3/C4가 신호 풀 확장 후 재측정에서도 현 수준(rho=0.10, flip=False)이면 연속 overlay 경로(`overlay_mult`/`trend_scale`/`vol_scale`)를 allocation prior에 연결하는 방안 재검토.

`transition`=0% dead state 처리: 6→5 state 축소 or hysteresis 재설계 (보류).

---

# 📏 Regime 평가 프레임워크 (8축 루브릭)

| ID | 축 | PASS 임계 | 가중 |
|----|----|----------|------|
| C1 | Look-ahead | hard gate (entry는 `code[t-1]` 소비) | 필수 |
| C2 | Persistence | dwell ≥ 6 bars, 전이율 ≤ 0.15 | 0.15 |
| C3 | Distinctness | KW p<0.05 **AND** 최소 1쌍 부호반전 | **0.25** |
| C4 | OOS Stability | ρ ≥ 0.5 | **0.20** |
| C5 | Coverage | 5% ≤ 각 occ ≤ 60% | 0.10 |
| C6 | 통계 기초 | 데이터 기반 breakpoint | 0.15 |
| C7 | 견고성 | ±20% 파라미터 점유 안정 | 0.10 |
| C8 | 재현성/테스트 | distinctness 테스트 존재 | 0.05 |

**현재 점수(C1-C5 기준)**: 가중 0.450. C3·C4 ❌가 결정적 약점.

---

# Verification

```bash
# Regime stage 단독 실행
uv run python -m src.execution.opt_main_futures --phase regime --timeframe 4h --sync skip 2>&1 | grep -A15 "REGIME_SCORECARD"

# Stub 제거 확인 (출력 없어야 정상)
grep -rn "infer_regime_codes\|compute_oos_regime_attribution\|compute_regime_drift" src/ --include="*.py"

# 단위 테스트
uv run pytest tests/unit/domain/futures/strategy/test_regime_evaluation.py --tb=short -q

# Priority 1 구현 후
LOG_LEVEL=DEBUG uv run python -m src.execution.opt_main_futures --phase ml --timeframe 4h --sync skip 2>&1 | grep -E "regime|PRIOR|ABLATION"
```

---

# 완료 기준 전체

- [x] R1: `[REGIME_SCORECARD]` 테이블 로그 출력
- [x] R1: C3 kw_p, MI, flip 실측치 확보
- [x] R1: `--phase regime` 단독 실행 경로
- [x] R2: 이산 vs 연속 결정 근거 문서화 (분기 B)
- [x] R3-0: 죽은 백테스팅 stub 제거 (CUSUM 6-state SSOT 단일화)
- [ ] P0: ML-Ready 신호 6+개 확보 → C3/C4 재측정
- [ ] P1: 레짐 조건부 prior (candidate_edge + candidate_dataset)
- [ ] P2: Hedge 동적 배분 (ablation CAGR > +1.4%)
- [ ] P3: 메타라벨 사이징 (MaxDD 축소)
- [ ] R3-B: 종합 점수 ≥ 70/100 (신호 풀 확장 후)
