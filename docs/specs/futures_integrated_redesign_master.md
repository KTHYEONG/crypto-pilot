# 🎯 Futures Integrated Redesign & Ablation Master Spec

> **Note:** 본 문서는 아키텍처 재설계, 레짐 격리(Ablation), 진입 게이팅 스펙을 통합한 단일 진실 공급원(SSOT)입니다. `l1_signal_l2_allocation_redesign.md`의 실행 전략을 포함합니다.

## 1. 개요 및 진단 (Objective & Diagnostics)
*   **목적:** 복잡한 ML(LGBM) 의존도를 낮추고, **L1(신호 하드 게이트)** 및 **L2(B0 통계 앙상블)** 중심의 해석 가능한 6단계 파이프라인으로 재편.
*   **핵심 진단:** 현재 신호 풀은 '추세' 일색으로 레짐별 분화(C3 flip)가 부재함. 연료(신호 Edge)가 부족한 상태에서의 엔진(ML) 튜닝은 무의미함.

## 2. 목표 아키텍처: 6-Layer 파이프라인
1.  **[L0] Universe:** ADV 기반 유동성 및 상관 클러스터링 (현행 유지)
2.  **[L1] Signal (Alpha) ★:** Standalone Breakeven 하드 게이트 + Archetype별 공정 평가.
3.  **[L2] Allocation (배분) ★:** B0 Regime-conditional Shrinkage Ensemble (ML은 챌린저로 격하).
4.  **[L3] Sizing:** Fractional Kelly (κ≤0.25) + Vol-target (현행 유지)
5.  **[L4] Portfolio/Cost:** 상관 Netting 및 현실적 비용 반영 (현행 유지)
6.  **[L5] WF Backtest:** AWF + CPCV + DSR 기반 OOS 검증 (현행 유지)

## 3. L1/L2 개편 상세 (The Core Redesign)

### L1 — Signal: 공정 평가 및 진입 게이팅
*   **Hard Breakeven Gate:** Archetype-valid 레짐 내 Net Edge가 비용 허들(HAC-t)을 못 넘으면 풀 진입 원천 차단.
*   **MR Entry Gating:** `mean_reversion` 신호는 추세장(Volatile/Crash) 진입을 강제 차단(`side_hint=0`). 이를 통해 역추세 신호가 불리한 장세에서 도태되는 것을 방지.

### L2 — Allocation: B0 Shrinkage Ensemble
*   **수학적 기초:** $\hat{\mu}(a,g) = \frac{n_{a,g} \cdot \bar{e}_{a,g} + k \cdot \bar{e}_{global}}{n_{a,g} + k}$ ($k$: shrinkage 강도, 기본 50)
*   **운영:** 학습 없이 Train Window 데이터만으로 적합(Look-ahead 차단). 결정론적이며 압도적으로 빠름.

## 4. 레짐 역할 격리 및 절제 사다리 (Ablation Ladder)
Regime의 기여도를 측정하기 위해 기능을 4단계로 분리하여 측정함.
*   **L0 (Baseline):** Signal + B0/ML + Kelly Sizing (Regime 전부 OFF)
*   **L1:** L0 + **Regime Feature** 주입 (`ln_entry`, `ln_code`)
*   **L2:** L1 + **Regime Risk Overlay** 활성 (`overlay_mult`)
*   **L3:** L2 + **MR Entry Gating** (L1 개편안 핵심 반영)

## 5. 단계별 검증 게이트 (Phase Ladder)
*   **G1 (Signal):** 신호가 Standalone Breakeven을 통과하는가? (1차 병목)
*   **G2 (Structure):** 전략 엣지가 레짐별로 분화(C3 Magnitude)되는가? (2차 병목)
*   **G3 (Allocation):** B0 앙상블이 균등 배분(Equal-Weight)을 이기는가?
*   **G4 (ML Rank):** B1(ML)이 B0를 DSR-유의하게 이기는가? (No면 B0 채택)
*   **G5 (Optimization):** 최종 Optuna AWF/CPCV/DSR 최적화.

## 6. 핵심 설정 (Contract Changes - `config.py`)
```python
# L1/L2 Redesign
allocation_backend: Literal["ensemble_b0", "ml_edge"] = "ensemble_b0"
standalone_breakeven_hard_gate_enabled: bool = True
ensemble_shrinkage_k: float = 50.0

# Regime Ablation & Gating
regime_feature_enabled: bool = True               # L1
regime_overlay_enabled: bool = True               # L2
mean_reversion_regime_entry_gating_enabled: bool = True # L3
regime_scorecard_enabled: bool = False            # 진단 분리 (Main Path 제외)
```

## 7. 대상 파일 (Target Files)
*   **신규:** `src/domain/futures/strategy/candidate_ensemble.py` (B0 구현)
*   **수정:** `config.py`, `candidate_workflow.py` (Backend 분기), `rule_signals.py` (L1 게이팅), `rule_diagnostics.py` (하드 게이트), `opt_main_futures.py` (Ablation Runner).
