# Mode Full (ML) — 최신 검증 결과

**실행 일시:** 2026-06-05 (Phase 1 signal gate 교정 후)  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase full --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 433 (sel=113)`, `Status: PROMOTED`

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[PIPELINE] raw=272819 labeled=6350 promoted=6350 fit=9869 cal=9165 oos=2355 n_folds=4 wf_scheme=anchored
[BRIDGE][WF] fold_cost_survival=[True, True, True, True] pass_ratio=1.00
[SIGNAL-VALIDATION] overall_pass=True (mean=17.8bps, stress_mean=14.1bps)
[BRIDGE] Active Signals: 433 (sel=113) | Status: PROMOTED
```

---

## 백테스트 성과 (OOS)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Kelly (No ML) | -0.1% | 0.4% | 0.00 | 997,350 | 2149 | 0.66 | N |
| ML Gate | -0.0% | 0.0% | 0.00 | 999,663 | 117 | 0.05 | N |
| ML Gate+Edge | -3.2% | 9.8% | -0.33 | 906,347 | 69 | 0.02 | N |
| ML Full (Capped) | -0.0% | 0.0% | 0.00 | 999,647 | 115 | 0.05 | N |
| Cand. ML | -0.1% | 0.0% | 0.00 | 999,563 | 115 | 0.27 | N |

**전 모델 Pass=N** — 배포 판정 실패.

---

## Signal Gate 교정 결과 (Phase 1)

| 항목 | 이전 | 이후 |
|---|---|---|
| 블렌드 판정 기준 | median (`net_p50=−117bps`) | **mean** (`mean=+17.8bps`) |
| stress 비용 적용 | 이중 차감 (base+stress=18.75bps) | **stress 1회** (11.25bps 총) |
| any_passes | False | **True** |
| overall_pass | False | **True** |
| SIGNAL-VALIDATION | BLOCKED | **PROMOTED** |

---

## ML Layer 블로커 (신규 진단)

### 블로커 1 — Calibration Probability Collapse

모든 4개 WF fold에서 sigmoid 캘리브레이션 붕괴:

```text
raw_std ≈ 0.12~0.13 → cal_std ≈ 0.003~0.017  (판별력 소멸)
calibrated=False  reason=calibration_probability_collapse
gate_p50=0.449  pct_ge55=0.208
```

- `gate>=0.55` 기준에 gate_pass=491/2355 (21%)만 통과
- 전체 eligible=1135 중 selected=113 — gate가 실질적 병목

### 블로커 2 — OOS Edge 붕괴

- ML Gate+Edge: CAGR -3.2%, 69 trades → 과소 배포 + 비용 미회수
- IS에서 mean +17.8bps인 edge가 OOS에서 음(−) 실현

---

## Candidate Top Strategies (KEEP 12/20)

| Rank | Strategy | OOS n | Profit(bps) | Win Rate | P/L | Action |
|---|---|---:|---:|---:|---:|---|
| 1 | `trend_pullback_continuation:tpc_20_100` | 312 | **90.9** | 37.5% | 1.63 | **KEEP** |
| 3 | `funding_zscore_carry:fzs_48` | 691 | 36.9 | 46.2% | 1.69 | **KEEP** |
| 4 | `funding_carry:funding_24` | 1560 | 33.6 | 45.4% | 1.68 | **KEEP** |
| 5 | `funding_zscore_carry:fzs_168` | 1306 | 31.8 | 46.4% | 1.48 | **KEEP** |
| 6 | `rsi_reversion:rsi_14` | 1838 | 28.2 | 44.9% | 1.38 | **KEEP** |
| 10 | `cross_sectional_momentum:cs_mom_10` | 4488 | 21.3 | 44.0% | 1.23 | **KEEP** |
| 12 | `funding_zscore_carry:fzs_96` | 659 | 20.1 | 43.2% | 1.41 | **KEEP** |
| 13 | `funding_acceleration_carry:fac_48` | 6743 | 19.7 | 44.2% | 1.22 | **KEEP** |
| 14 | `cross_sectional_momentum:cs_mom_20` | 6480 | 19.0 | 44.3% | 1.21 | **KEEP** |
| 16 | `funding_acceleration_carry:fac_168` | 6933 | 18.2 | 43.9% | 1.24 | **KEEP** |
| 18 | `btc_corr_regime:bcr_48` | 9711 | 15.8 | 44.1% | 1.25 | **KEEP** |
| 20 | `btc_corr_regime:bcr_96` | 9896 | 14.0 | 43.9% | 1.22 | **KEEP** |

---

## 다음 단계

1. **캘리브레이션 붕괴 대응** — `isotonic` 교체 또는 uncalibrated raw score 활용 경로 검토
2. **OOS edge 붕괴 원인 진단** — IS/OOS mean 乖離 근본 원인 (선택편향 / 다중검정)
3. **Phase 2 spec** — deflated Sharpe / BH-FDR 다중검정 보정, WF fold positive-ratio 게이트
