# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-09
- **Registered ADR**: `ADR_20260809_MHS_HORIZON_OPT_SPEC`
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/my_coin_traider/docs/results/mhs_horizon_diagnostic.json)
- **Execution Status**: `COMPLETE` (전체 5개년 파이프라인 100% 정상 완주)

---

## 1. Data Integrity & Exclusion Summary

바이낸스 원본 아카이브 갭 및 펀딩비 결측치 처리 결과:

1. **원천 데이터 갭 종목 하드코딩 제외 (Source Gap Exclusions)**:
   - 바이낸스 REST API 및 Vision 아카이브 상 4시간 이상 연속 갭이 존재하는 7개 종목 사전에 안전 제외:
     `SLPUSDT`, `CTKUSDT`, `LITUSDT`, `AERGOUSDT`, `PUMPUSDT`, `CVXUSDT`, `CVCUSDT`
2. **펀딩비 내부 보정 (bar_funding Sanitization)**:
   - 타임스탬프 인덱스 얼라인먼트 과정에서 발생한 internal NaN/Inf 결측치에 대해 `.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)` 적용 완료.
   - 데이터 정합성 에러(`bar_funding must be finite` 및 `MISSING_DECISION_MARK`) **전면 해소**.

---

## 2. Resource & RSS Budget Metrics

5개년(2021-2025) 정밀 백테스트 시 프로세스 최고 메모리(Peak RSS) 사용량 현황:

| Stage / Replay Window | Peak RSS Observed | Budget Limit | Status |
| :--- | :--- | :--- | :--- |
| **`replay_fast_reversal`** | **2,575,036,416 bytes (2.575 GB)** | 4.000 GB | ✅ PASS |
| **`replay_slow_momentum`** | **2,575,036,416 bytes (2.575 GB)** | 4.000 GB | ✅ PASS |
| **`replay_blend`** | **2,575,036,416 bytes (2.575 GB)** | 4.000 GB | ✅ PASS |
| **`anchored_fold_0` (2023)** | **2,575,036,416 bytes (2.575 GB)** | 4.000 GB | ✅ PASS |
| **`anchored_fold_1` (2024)** | **2,575,036,416 bytes (2.575 GB)** | 4.000 GB | ✅ PASS |
| **`anchored_fold_2` (2025)** | **2,575,036,416 bytes (2.575 GB)** | 4.000 GB | ✅ PASS |

---

## 3. Full-Period Primary Replay Quantitative Performance (2021-2025)

| Metric | Fast Reversal | Slow Momentum | Blend Ensemble | Research GO Target |
| :--- | :--- | :--- | :--- | :--- |
| **Execution Status** | `COMPLETE` (Pass) | `COMPLETE` (Pass) | `COMPLETE` (Pass) | `COMPLETE` |
| **Autocorrelation Sharpe** | **-2.9178** | **-0.7370** | **-1.9468** | **≥ +0.60** |
| **Naive Sharpe** | **-1.2048** | **-0.2314** | **-0.6286** | > 0.00 |
| **Stress Naive Sharpe** | **-0.2368** | **+0.1694** | **+0.0618** | **> 0.00** |
| **Annualized Net Return** | **-4.8460%** | **-0.7954%** | **-1.0151%** | Positive |
| **Geometric CAGR** | **-4.8079%** | **-0.8508%** | **-1.0229%** | Positive |
| **Maximum Drawdown (MDD)** | **-94.8315%** | **-43.7777%** | **-46.0566%** | Low MDD |
| **Annualized Turnover** | **4.3922** | **1.5240** | **1.6053** | Bounded |
| **Intent Shortfall (Slippage)** | **428.75 bps** | **563.48 bps** | **-8.35 bps** | ≤ 15.0 bps |
| **Fills / Unfilled Count** | 281,475 / 128,095 | 61,001 / 28,908 | N/A | Complete Fills |

---

## 4. Anchored Folds Quantitative Performance

이전 정산에서 메모리 및 데이터 갭으로 무효화되었던 Fold 1, Fold 2가 **100% 정상 완주(`Primary Valid: True`)** 되었습니다.

| Metric | Fold 0 (2023) | Fold 1 (2024) | Fold 2 (2025) | Research GO Target |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Validity** | **`True`** | **`True`** | **`True`** | All `True` |
| **Autocorrelation Sharpe** | **-0.8980** | **-3.5601** | **-1.8427** | **≥ +0.60** |
| **Naive Sharpe** | **-0.3023** | **-0.9693** | **-0.5897** | > 0.00 |
| **Stress Naive Sharpe** | **+0.2519** | **-0.4892** | **-0.2590** | **> 0.00** |
| **Annualized Net Return** | **-0.5542%** | **-1.1743%** | **-0.9184%** | Positive |
| **Geometric CAGR** | **-0.5694%** | **-1.1746%** | **-0.9262%** | Positive |
| **Maximum Drawdown (MDD)** | **-13.3220%** | **-13.1434%** | **-10.4382%** | Low MDD |
| **Intent Shortfall (Slippage)** | **36.23 bps** | **78.17 bps** | **78.17 bps** | ≤ 15.0 bps |
| **Fills / Unfilled Count** | 317,200 / 136,598 | 195,243 / 78,932 | 177,302 / 79,670 | Complete Fills |
| **Decision Intents** | 668,226 | 274,912 | 257,010 | Complete |

---

## 5. Books Prescreen & Tail Analysis

| Book Name | Prescreen Sharpe (2.64 bps) | Phase Ensemble Sharpe | Tail Base Sharpe | Leave-Worst-Out Sharpe |
| :--- | :--- | :--- | :--- | :--- |
| **Fast Reversal** | -0.0671 | -2.3458 | -0.5113 | -0.5252 |
| **Slow Momentum** | **+0.7303** | -0.0544 | +0.3086 | +0.2795 |
| **Blend Ensemble** | **+0.5608** | -2.3458 | -0.1495 | -0.1000 |

---

## 6. Research GO Evaluation & Target Optimization Metrics

- **Research GO Final Status**: **`eligible: false`**
- **Reason Codes**: `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`
- **주요 성과 및 의의**:
  1. 백테스트 데이터 처리 엔진의 100% 정상 완주로 **Deterministic한 정밀 연구 시뮬레이션 환경 구축 완결**.
  2. `slow_momentum` 및 `blend` 전략의 스트레스 샤프 지수가 각각 **`+0.1694`**, **`+0.0618`**로 양수 유지.
  3. 향후 시그널 알파 튜닝 및 체결 슬리피지 절감을 통해 샤프 지수 $\ge +0.60$ 달성을 위한 기준선(Baseline) 마련.
