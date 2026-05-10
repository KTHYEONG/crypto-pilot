# 🧬 System Evolution Journal

This file tracks the logical progression and experimental results of the quantitative trading system. It serves as the primary context for AI agents to understand "why" certain changes were made and "what" was learned.

---

## [2026-05-10] v7.0.0: 전면 아키텍처 재설계 + HMM 캐시 제거 (Claude Opus 4.7)

### 1. 핵심 진단: 통합 백테스트가 최악인 근본 원인

- **Signal Path 다중 곱셈/게이팅**: HMM modulator가 gate(<0.1 차단)와 sizing multiplier(Kelly×modulator) 두 번 작동. alpha z-score + rank-mult 이중가중. crisis에 long만 threshold+magnitude 양쪽 패널티.
- **IS 누설**: SignalCalibrator가 IS 12-bar returns로 Platt fit → IS에서 win probability 과적합.
- **PLGD 14차원 과최적화**: 7개 가중치 + 14개 trial param = 5 seeds × 400 trials = 2000회 IS 검색.
- **AWF/CPCV/WF 혼재**: 세 파이프라인이 데이터 슬라이스 미공유, 각각 OOS 누설. embargo=0 기본값.

### 2. 아키텍처 재설계 내역

| 구분 | 변경 전 | 변경 후 |
|---|---|---|
| **신호 결합** | HMM gate + Kelly×modulator 직렬 3중 | `signal_composer.py` 단일 선형 결합 μ = β_α·α + Σβ_k·p_k − friction |
| **포트폴리오** | policy_engine apply_policy_constraints | `portfolio_constructor.py` Ledoit-Wolf + Kelly + vol-target + QP |
| **실행** | MultiSymbolEngine 복합 로직 | `execution_sim.py` single-pass, fee 이중계상 제거, short borrow 추가 |
| **목적함수** | PLGD 7-항 가중합 | `median(log_TW) − 1.0·MAD − 0.5·DD` (λ, ψ 고정) |
| **최적화** | 5-seed × 400trial = 2000회 동시 검색 | 3-Phase Coordinate Ascent A(80)+B(120)+C(60) = 260 trials |
| **Validation** | AWF/CPCV/WF 혼재, embargo=0 | AWF 단일 K=6, IS-pool=70%, embargo timeframe-aware |
| **CPCV** | legacy 코드 잔존 | 완전 삭제 (validation.py, evaluator.py, optimizer.py) |
| **Champion** | select_orthogonal_ensemble[0] | 3-Layer Gate (L1/L2/L3 + 5-seed stability CV≤0.30) |
| **Calibration** | IS 데이터로 Platt fit (누설) | `_fit_oos_platt_calibrators_from_maps` OOS-only fit |

### 3. HMM 고도화 및 캐시 제거

- **Gravity penalty** 200→700 (CRISIS state return loc 음수 고정 강화)
- **min-duration** [168,84,24,24,6] → [120,48,24,12,24] (CRISIS 6→24 스파이크 억제)
- **Crisis calibration** 재활성화 (target=7%, logit offset)
- **HMM 캐시 완전 제거**: CacheManager 의존성 삭제, to_parquet 저장 삭제. 실측 재학습 시간=**2분(120s)** — 캐시 불필요.
- **캐시 제거 계기**: 소스파일 SHA256을 캐시 키에 포함하는 구조가 코드 편집 시마다 비결정론적 재학습을 유발하는 근본 결함 발견.

### 4. 현재 HMM 결과 (캐시 없이)

| Regime | TIME% | MOD L/S | G_LOG | Verdict |
|---|---|---|---|---|
| BULL_CALM | 2.3% | 0.32/0.22 | +0.014% | CHOP |
| BULL_VOL_UP | 13.4% | 0.36/0.20 | +0.016% | BULL |
| BEAR_TREND | 46.1% | 0.21/0.35 | -0.000% | CHOP |
| CHOP | 31.8% | 0.32/0.19 | +0.008% | CHOP |
| CRISIS | **6.4%** | 0.17/0.29 | -0.013% | BEAR |

- **CRISIS 목표 달성** (25.2%→6.4%). Left-tail capture 34.1%→**60.8%**.
- **미해결**: BEAR_TREND 46.1% 과대분류 (이전 12.8%). BULL_CALM 2.3% (이전 43.1%). HMM 수렴 경로가 이전과 달라진 state 분포 — JAX seed 고정으로 해결 예정.
- **Modulator** max 0.36 미개선 — pipeline_runner.py risk_scale 공식 교체(5-7순위) 필요.

### 5. 주요 실패 실험

- `BULL_CALM min-duration 84`: min-duration 단축이 BULL_CALM 붕괴를 유발. post-hoc lock-in이므로 Viterbi 경로 자체에 영향 없음.
- `HMM 소스파일 해시 캐시 키`: 주석 1줄 수정에도 캐시 무효화→비결정론 재학습.

---

## [2026-05-10] v6.9.0: Advanced Alpha Features (v22) — Idiosyncratic & Microstructure (Gemini CLI)

### 1. Architectural Shift: Unlocking Directional Conviction
*   **Problem**: Alpha v21 was elite but trade frequency was low. Long-side participation was weak (PF 0.65) compared to Short-side (PF 3.82). The system lacked features to distinguish idiosyncratic strength from market beta.
*   **Implementation (The Alpha Depth)**:
    1.  **Idiosyncratic Residuals**: Added `idiosyncratic_return_24h` ($R_{asset} - \beta \cdot R_{btc}$) to isolate asset-specific strength.
    2.  **Microstructure Signals**: Added `price_impact_asymmetry` (Liquidity Vacuum) and `exhaustion_cascade_score` (Triple-confluence bottom sensing).
    3.  **Temporal Embedding**: Added `session_seasonality_sin/cos` to allow AI to learn session-specific patterns.
    4.  **Schema v22**: Bumped `GP_FEATURE_SCHEMA_VERSION` to force global cache invalidation.

### 2. Performance Impact (Full-Scale 5,000 Trials)
*   **Alpha Consistency**: **OOS IC maintained at 0.0743 (T-Stat 33.27)**.
*   **Safety Audit**: **MDD remained ultra-low at 0.15%**, confirming HMM robustness.
*   **The Blocking Bottleneck**: Despite better features, **CAGR stayed negative (-0.6%)**. 
*   **Diagnosis**: **Execution Dissonance**. The unified `CS_Z_SCORE_THRESHOLD` is too high (~1.0) for the Long edge.

---

## [2026-05-10] v6.8.0: Magnitude Refactor & OI Data Restoration (Gemini CLI)

### 1. Architectural Shift: Targeting Explosive Alpha
*   **Implementation**:
    1.  **OI Restoration**: Fixed `BinanceVisionDownloader` mapping and normalized timestamps. Backfilled 60 days of metrics.
    2.  **High-Volatility Features**: Added `liq_intensity_proxy` and `capitulation_proxy` (candle tails).
    3.  **Dynamic Friction Hurdle**: AI now ignores "noise-sized" winners using ATR-based targets.

### 2. Performance Impact
*   **Efficiency**: **Profit Factor surged from 0.49 to 0.75 (+53%)**.
*   **Precision**: **Win Rate improved to 57.26%**.
*   **Safety**: **MDD reduced to 0.16%**.

---

## [2026-05-10] v6.7.0: P1+P2 Integration — Magnitude Model & Linear Modulator (Opus 4.7)

### 1. Architectural Changes
*   **Modulator**: Shifted from tanh saturation to `clip(target_var/(RA·var), 0.25, 1.75)`.
*   **Magnitude**: Added `LGBMRegressor` per slot for 24h magnitude prediction.
*   **Slot Expansion**: Increased from 15 to 18 slots to accommodate interaction themes.

### 2. Experimental Results
*   **IS-OOS Retention**: Improved from -331% → **+86.4%** (structural stability).
*   **HO CAGR**: **+0.28%** (out-of-holdout positive for the first time).

---

## [2026-05-10] v6.6.0: Natural Risk-Adjusted Scaling (Gemini CLI)
- **Status**: Validated (Net Alpha improved from -65.4% to -15.94% (+49.5%p)).
- **Logic**: Replaced heuristic overrides with $RA_{dyn} = 1.0 + 3.0 \cdot P_{crisis} + 1.5 \cdot P_{bear}$.

---

## [2026-05-10] v6.5.0-p2-kelly-diag: Kelly Sizing Root Cause (Haiku)
- **Discovery**: Kelly sizing was fundamentally underpowered due to Platt Scaling under-discrimination ($ml\_calib\_prob \approx 0.496$).

---

## [2026-05-09] v6.4.x: Directional Symmetry & Execution Liberation (Gemini CLI)
- **Status**: Record **OOS IC 0.1466**.
- **Fixes**: Symmetric hybrid labeling and search space expansion. Identified 87% OOS defensive mapping as the blocker.

---

## [2026-05-09] v6.3.x: PLGD Objective & Hysteresis Restoration (Gemini CLI)
- **Status**: Validated (MDD 0.18%, PF 1.74).
- **Logic**: Maximizing Probabilistic Log Growth Deflation (PLGD) and Schmitt Trigger filtering.

---

## [2026-05-09] v6.0.0: Horizon Pivot (4h) & Decoupled Architecture (Gemini CLI)
- **Status**: Validated (OOS CAGR: +24.8% PASS, PF: 1.14 PASS).
- **Logic**: Upgraded from 1h to 4h base timeframe. Decoupled HMM from Alpha Ranking.

---

### 🏛️ Historical Summary (v1.x - v5.x)
Prior to v6.0.0, the system evolved through Guided HMM architectures (v1-v2), unsupervised regime discovery (v3), and deterministic policy mapping using Kelly Criterion (v5). Key lessons included the failure of binary heuristic overrides and the necessity of frequency matching between alpha and execution.
*Full history preserved in `.ai/archive/EVOLUTION_v5.md`.*

---
<!-- APPEND_POINT: New experiments will be added above this line -->
