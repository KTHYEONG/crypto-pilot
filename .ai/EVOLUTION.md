# 🧬 System Evolution Journal

## [2026-05-15] v11.6.0 HMM Architecture & Extreme GPU Optimization (Gemini CLI)

### 1. 개요 (Context)
기존 HMM v11.0의 'CHOP Sink' 현상(모든 데이터를 CHOP 상태로 분류하여 방어 불능) 및 GPU 연산 비효율성 해결을 위한 대규모 아키텍처 개편.

### 2. 주요 변경 사항 (Logic Shift)
- **HMM v11.1 (Structural):** `Occupancy Prior` 및 `CHOP Semantic Penalty` 도입으로 상태 붕괴 해결. `Direct Tail-Penalty` 주입으로 Tail-Capture 성능 52% 달성.
- **HMM v11.3 (GPU Native):** `jax.lax.scan` 및 `dynamic_slice`를 이용한 GPU 내재화 루프 구현. PCIe 데이터 전송 병목 제거 및 `TF32` 활성화.
- **HMM v11.5 (Mathematical):** `Relative Tolerance` 기반 조기 종료(Early Stopping) 버그 수정. 루프 내 중복 연산(Loop Invariant) 외부 호이스팅.
- **HMM v11.6 (Python/Pandas):** `rolling.quantile` 제거 및 `Numba` 가속 전처리 도입. 전체 소요 시간 189초 → 35초(약 5.4배 가속) 달성.

### 3. 실험 결과 (Metrics)
- **Regime Tail-Capture:** 0.0% → **52.1%** (목표 40% 초과 달성)
- **Avg-Duration:** 21,865 bars(붕괴) → **35.9 bars** (안정적 국면 유지)
- **Execution Speed:** **35s** (v11.6 GPU-Native)
- **Verdict:** 🟢 **CONDITION_READY**

### 4. 향후 과제
- **FlatGate-Prec (29.6%) 개선:** 상위 Quantile 임계값 튜닝(0.90 → 0.93) 필요.
- **Phase D (Portfolio Optimization):** 고성능 HMM 엔진을 활용한 전체 수익률/MDD 개선 검증.

---

<!-- APPEND_POINT: New experiments will be added above this line -->

# 🧬 System Evolution Journal (Legacy)
... [기존 내용] ...
