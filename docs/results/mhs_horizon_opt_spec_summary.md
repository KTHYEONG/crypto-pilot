# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-09
- **Registered ADR**: `ADR_20260809_MHS_HORIZON_OPT_SPEC`
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/my_coin_traider/docs/results/mhs_horizon_diagnostic.json)

---

## 1. Resource & RSS Budget Metrics

실행 시 프로세스 RSS 메모리 예산은 **2.500 GB (2,500,000,000 bytes)** 로 설정되었으나, 아래 모든 replay 세그먼트에서 초과하여 `RESOURCE_BUDGET_BREACH` 발생.

| Replay Segment | Observed Peak RSS | Budget Limit | Breach Amount | Status |
| :--- | :--- | :--- | :--- | :--- |
| `replay_fast_reversal` | **2,554,400,768 bytes (2.554 GB)** | 2.500 GB | +54.40 MB | ❌ BREACH |
| `replay_slow_momentum` | **2,552,385,536 bytes (2.552 GB)** | 2.500 GB | +52.39 MB | ❌ BREACH |
| `replay_blend` | **2,762,870,784 bytes (2.763 GB)** | 2.500 GB | +262.87 MB | ❌ BREACH |

---

## 2. Anchored Folds Quantitative Performance

총 3개 Fold 중 Fold 0만 Strict Ledger를 생성하였으며, Fold 1/2는 메모리 초과로 무효화되었습니다.

| Metric | Fold 0 (2023) | Fold 1 (2024) | Fold 2 (2025) | Research GO Target |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Validity** | `True` | `False` | `False` | All `True` |
| **Autocorrelation Sharpe** | **-6.3830** | `NaN` | `NaN` | **≥ +0.60** |
| **Naive Sharpe** | **-1.8074** | `NaN` | `NaN` | > 0.00 |
| **Stress Naive Sharpe** | **-0.2864** | `NaN` | `NaN` | **> 0.00** |
| **Annualized Net Return** | **-3.3358%** | `NaN` | `NaN` | Positive |
| **Geometric CAGR** | **-3.2972%** | `NaN` | `NaN` | Positive |
| **Maximum Drawdown (MDD)** | **-34.3414%** | `NaN` | `NaN` | Low MDD |
| **Intent Shortfall (Slippage)** | **39.2694 bps** | N/A | N/A | ≤ 15.0 bps |
| **Fills / Unfilled Count** | 180,722 / 78,437 | 0 / 0 | 0 / 0 | Complete Fills |
| **Failure Reasons** | `AUTOCORR_SHARPE_BELOW_0_6`<br>`STRESS_SHARPE_NOT_POSITIVE` | `INVALID_PRIMARY_LEDGER` | `INVALID_PRIMARY_LEDGER` | None |

---

## 3. Books Prescreen & Tail Analysis

| Book Name | Prescreen Sharpe (2.64 bps) | Phase Ensemble Sharpe | Tail Base Sharpe | Leave-Worst-Out Sharpe |
| :--- | :--- | :--- | :--- | :--- |
| **Fast Reversal** | -0.1316 | -2.3719 | -0.6109 | -0.6388 |
| **Slow Momentum** | **+0.6779** | -0.0343 | +0.2520 | +0.2225 |
| **Blend Ensemble** | +0.5232 | -2.3719 | -0.1794 | -0.1542 |

---

## 4. Target Optimization Metrics (`mhs_horizon_opt` Spec)

설계안([`mhs_horizon_opt.md`](file:///home/kth/my_coin_traider/docs/specs/mhs_horizon_opt.md)) 적용 후 달성 목표 수치:

1. **Peak Memory**: Process RSS < **1.80 GB** (예산 2.50 GB 대비 700 MB 마진)
2. **Autocorrelation Sharpe**: Fold 0/1/2 전구간 **≥ +0.60**
3. **Stress Naive Sharpe**: **> 0.00**
4. **Intent Shortfall**: **≤ 15.0 bps** (현재 39.27 bps 대비 60% 감축)
5. **Passed Folds**: **3 / 3 (100% Pass)**
