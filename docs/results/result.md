# L0/L1 최신 파이프라인 결과 (2026-07-16, Slow-TF XS Challenger)

## 실행 및 데이터 무결성

- **Baseline**: `logs/futures/diagnostics/l1_cross_tf/treatment.json` — 1h 포함 7개 TF, 6h/1d BLOCKED.
- **Challenger**: `slow_tf_xs_challenger_enabled=True`를 process-local config에만 주입하여 동일 L1 replay를 2회 실행했다. 기본 production config는 `False`로 유지된다.
  - `logs/futures/diagnostics/l1_cross_tf/challenger.json`
  - `logs/futures/diagnostics/l1_cross_tf/challenger_repeat.json`
- **변경 범위**: 6h/1d pool에 기존 `residual_momentum_xs`, `xs_residual_rebalance` 4개 panel을 opt-in 추가하고, 두 TF의 effective config에만 XS factor-level admission을 활성화했다. FDR/LCB/quality threshold는 완화하지 않았다.
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01.
- **Universe**: Pool 377 → Selected 150 → Loaded 106; L1 admission 101/106 (`late_start` 5개 제외).
- **재현성 범위**: 두 challenger run의 최종 `l1_result` snapshot은 모든 TF에서 동일했다. 그러나 native panel/canonical L0/delivery-event digest는 동일 seed에서도 달라진다. 이는 기존 `ProcessPoolExecutor` fork 비결정성 이슈와 일치하며, full-trace reproducibility는 아직 미해결이다.

## L1 판정 — Challenger 최신 실측

| Timeframe | 판정 | Symbol-Breadth | probe_lcb_bps | 승급 신호 수 | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1h** | **✅ PASS** | 77.606 (≥5.00) | +37.651 | 100 | baseline과 최종 승급 수 동일 |
| **2h** | **✅ PASS** | 84.390 (≥5.00) | +93.414 | 87 | baseline과 동일 |
| **4h** | **✅ PASS** | 40.552 (≥3.00) | +26.275 | 33 | baseline과 동일 |
| **6h** | **✅ PASS** | 55.148 (≥3.00) | +35.357 | **8** | **BLOCKED 0 → PASS 8**; `xs_residual_rebalance:xsrr_8`가 AVAXUSDT에서 승급 |
| **8h** | **✅ PASS** | 81.281 (≥2.00) | +27.964 | 44 | baseline과 동일 |
| **12h** | **✅ PASS** | 61.504 (≥1.00) | +104.238 | 21 | baseline과 동일 |
| **1d** | **✅ PASS** | 80.803 (≥1.00) | +60.073 | **1** | **BLOCKED 0 → PASS 1**; `xs_residual_rebalance:xsrr_2`가 SKLUSDT에서 승급 |

## 6h/1d Waterfall

| TF | Native panels | Canonical L0 routes | L1 delivery events | Final winning signals |
| :--- | :---: | :---: | :---: | :---: |
| 6h baseline | 74 | 3 | 34,948 | 0 |
| 6h challenger | 78 | 4 | 94,884 | 8 |
| 1d baseline | 12 | 3 | 26,190 | 0 |
| 1d challenger | 16 | 5 | 67,970 | 1 |

- 6h의 최종 XS 승급 예: `AVAXUSDT / xs_residual_rebalance:xsrr_8`, edge `+144.6bps`, LCB `+102.5bps`, 4/4 folds.
- 1d의 최종 XS 승급: `SKLUSDT / xs_residual_rebalance:xsrr_2`, edge `+317.0bps`, LCB `+204.8bps`, 3/4 folds.
- 비목표 TF(1h/2h/4h/8h/12h)는 baseline과 `gate_passed` 및 winning signal 수가 모두 동일하다.

## Verdict and Remaining Constraints

- **과거 replay 가설 검증**: PASS. 기존 slow-TF pool에 없던 cross-sectional residual alpha가 6h/1d의 개별 pair 검정력 병목을 해소했고, strict final admission 아래에서도 실제 승급을 만들었다.
- **최종 L1 판정 재현**: PASS. 2회 challenger run에서 6h=`PASS/8`, 1d=`PASS/1` 및 전 TF `l1_result` snapshot이 동일했다.
- **Production promotion**: 보류. 현재 replay data는 이미 후보 설계와 분석에 사용된 2026-03-31까지이며 독립 holdout이 아니다. full-trace digest/승급 ID 수준의 재현성도 아직 증명되지 않았다.
- **금지된 후속 조치**: 이 구간의 PASS 수를 늘리기 위한 FDR, LCB, bootstrap probability 또는 quality-weight threshold 재완화.

## Next Action

1. `ProcessPoolExecutor` 기반 upstream artifact digest 비결정성의 최소 재현 사례를 고정하고, winning strategy ID를 trace에 기록해 full-trace parity를 검증한다.
2. 2026-04-01 이후 사전 고정 독립 구간에서 challenger를 2회 sequential replay한다. 6h/1d PASS, 비용 기준 factor LCB, no-regression, full-trace parity를 모두 만족할 때만 production default 전환을 별도 ADR로 결정한다.
