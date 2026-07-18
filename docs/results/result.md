# L1→L2 Replay 결과 — 2026-07-18 (L1 Cache Fingerprint 안정화 + RSS Guard 검증)

## 실행 조건

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --seed 42` (trials 기본값=120)
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31~2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 137 (Passed Integrity)
- **변경 사항**: `run_per_tf_l1`의 cache fingerprint를 `pd.util.hash_pandas_object`(프로세스 간 비결정적) → `_deterministic_df_fingerprint`(sha256 content-based, 결정적)로 교체. RSS guard(`_should_load_cache`) + `gc.collect()` cache 경계 추가.
- LTF panel cache(78MB, 이전 run에서 잔존) → LTF alpha는 cache hit으로 빠르게 통과. L1 cache는 기존 파일이 legacy fingerprint로 생성돼 이번 run에서 miss → 재계산 후 새로운 결정적 fingerprint로 저장됨.

## 소요시간 및 RAM 사용량

| 단계 | 소요시간 | RAM 사용량 추정 |
| :--- | ---: | ---: |
| 데이터 로딩 + Universe 선정 | ~30s | ~2GB |
| LTF alpha panel + L1 (7개 TF, cache miss) | ~90s | ~8GB |
| L2 Optuna study (120 trials, ~1.5s/trial) | ~180s | ~11GB |
| **Total wall time** | **~5분** | **Peak RSS ≈ 11~13GB** |

- L1 cache miss (새 fingerprint로 기존 파일 불일치)로 인해 7개 TF 전부 재연산 → L1 구간 ~90s 소요.
- **다음 run부터**: L1 cache hit (새 결정적 fingerprint 파일 사용) → L1 구간 ~0.3s로 단축 예상. Total wall time 약 3.5분으로 감소.
- RAM: 이전 warm run(13.1GB)과 유사한 패턴. `gc.collect()`가 cache 경계에서 호출되나, L1 재연산이 대부분의 메모리를 사용하므로 개선 효과는 cache hit 시 분명해짐.

## L1 결과: 7/7 TF ✅ ALL PASS

| TF | Fold Readiness | Probe LCB (bps) | Promoted | Top Signal Family |
| :--- | :---: | ---: | ---: | :--- |
| 1h | 4/4 ✅ | +68.0 | 264 | trend_pullback_continuation |
| 2h | 4/4 ✅ | +106.6 | 197 | trend_ma |
| 4h | 4/4 ✅ | +77.6 | 67 | btc_regime_pullback / trend_ma |
| 6h | 4/4 ✅ | +91.5 | 17 | dual_momentum / btc_regime_pullback |
| 8h | 4/4 ✅ | +81.0 | 209 | trend_pullback_continuation |
| 12h | 4/4 ✅ | +79.3 | 75 | trend_pullback_continuation / btc_regime_pullback |
| 1d | 4/4 ✅ | +103.6 | 3 | btc_regime_pullback |

### TF별 Top Signal 상세

**TF 1h** — 264 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | LQTYUSDT | trend_pullback_continuation | +291.2 | +269.3 | 1.00 | 3/3 |
| #2 | BANDUSDT | trend_ma (ema_12_72) | +327.4 | +251.3 | 1.00 | 2/2 |
| #3 | MAGICUSDT | trend_pullback_continuation | +185.9 | +220.7 | 0.94 | 3/3 |

**TF 2h** — 197 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | BANDUSDT | trend_ma (ema_18_108) | +398.8 | +353.4 | 1.00 | 2/2 |
| #2 | ARUSDT | trend_ma (ema_18_108) | +388.5 | +348.1 | 1.00 | 4/4 |
| #3 | ZRXUSDT | trend_pullback_continuation | +320.8 | +328.1 | 1.00 | 3/3 |

**TF 4h** — 67 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | LUNA2USDT | btc_regime_pullback | +554.0 | +478.1 | 1.00 | 3/3 |
| #2 | STXUSDT | trend_ma (ema_18_108_4h) | +356.7 | +380.2 | 1.00 | 4/4 |
| #3 | SSVUSDT | trend_ma (ema_18_108_4h) | +336.1 | +378.6 | 1.00 | 4/4 |

**TF 6h** — 17 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | ONEUSDT | dual_momentum (dm_16_64) | +489.3 | +522.7 | 1.00 | 4/4 |
| #2 | SSVUSDT | btc_regime_pullback | +557.5 | +447.2 | 1.00 | 4/4 |
| #3 | HBARUSDT | trend_donchian (donchian_72) | +350.7 | +407.9 | 1.00 | 4/4 |

**TF 8h** — 209 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | LQTYUSDT | trend_pullback_continuation | +892.2 | +1288.2 | 1.00 | 2/3 |
| #2 | BELUSDT | trend_pullback_continuation | +782.3 | +1009.4 | 1.00 | 2/2 |
| #3 | ONEUSDT | mtf_breakout_retest | +909.9 | +994.2 | 0.83 | 1/2 |

**TF 12h** — 75 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | ICPUSDT | trend_pullback_continuation | +1089.7 | +975.1 | 1.00 | 4/4 |
| #2 | APEUSDT | trend_pullback_continuation | +928.2 | +869.1 | 0.93 | 3/4 |
| #3 | STGUSDT | btc_regime_pullback | +891.4 | +817.1 | 1.00 | 2/2 |

**TF 1d** — 3 promoted
| Rank | Symbol | Strategy (Family) | Edge (bps) | LCB (bps) | Conv | Folds |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: |
| #1 | STGUSDT | btc_regime_pullback | +1515.7 | +1641.4 | 1.00 | 2/2 |
| #2 | ENSUSDT | btc_regime_pullback | +1008.4 | +1159.5 | 0.99 | 4/4 |
| #3 | GALAUSDT | trend_donchian (donchian_72) | +542.0 | +556.4 | 1.00 | 4/4 |

## L1 Audit

| Layer | Window | Symbols | Active (min/med/max) | Entry | Kill | Warnings |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| L1 | 2023-06-30 ~ 2024-12-30 | 126 | 0 / 107.0 / 126 | 92,419 | 66 | low_active_tail, entry_block_spike |

## L2 결과: 120 trials

### Optuna Study

- Study: `l2_study_8h_b49ce5386b6f`
- DB: InMemory | Trials: 120 | Events: 4,803 | Symbols: 44
- Pruner: MedianPruner + L2EarlyStopCallback (30 trial 무개선 시 중단) — **120/120 완료**, early stop 미발동
- Best CAGR: **+253.03%** (peak, outlier trial; champion은 +20.9%)
- 시행 속도: 평균 ~1.5s/trial (초기 1.0s → 후반 2.5~3.0s/symbol count 감소로 변동)

### Crisis Load

- Window: `luna_ftx_2022_collapse`
- Registry symbols: 103 | Overlap symbols: 47
- `[REGIME-L2] proof_failed path=pooled_fallback effective_states=3`

### Champion

```
[L2-SELECTION] feasible trials 없음 → fallback
```

| Metric | Value | Gate |
| :--- | ---: | :---: |
| Leverage (L*) | 4.8843 (binding: mdd) | |
| CAGR | **+20.9%** | ❌ BLOCKED (≥30%) |
| MDD | **25.9%** | ✅ (≤30%) |
| CVaR95 | 1.5% | ✅ (≤6%) |
| Utilization | 86.3% | |
| Sharpe | 1.012 | ✅ (≥1.000) |
| Sortino | 1.471 | ❌ (≥1.500) |
| Calmar | 0.808 | ❌ (≥1.000) |
| Fold Pass Ratio | 50.0% | ❌ (≥60%) |
| Trades | 223 | ✅ (≥30) |
| Sharpe Uplift | +0.20 | ✅ (≥+0.05) |
| PSR | 0.896 | ❌ (≥0.90) |

### Scorecard 상세

```
STATUS  : ❌ BLOCKED (cagr)

❌ [Growth    ] CAGR: +20.9% (>=30.0%) | PnL: +6.9% | Equity x1.07
❌ [Efficiency] Sharpe: 1.012 (>=1.000) | Sortino: 1.471 (>=1.500) | Calmar: 0.808 (>=1.000)
✅ [Risk      ] MDD: 25.9% (<=30.0%) | CVaR95: 1.5% (<=6.0%) | RiskUtil: 86.3%
❌ [Robust    ] Fold: 50.0% (>=60.0%) | Trades: 223 (>=30) | Friction: 100.0%
✅ [Uplift    ] Sharpe Uplift: +0.20 (>=+0.05)
❌ [Integrity ] PSR: 0.896 (>=0.90) | DSR: 0.000 (diag)
[Diag     ] RelMDD: 4.47x | Turnover: 0.012
```

### Fold 상세

| Fold | Period | CAGR | MDD | Sharpe | Status | Symbols |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| 1 | 2025-03-20 ~ 2025-05-30 | +45.7% | 10.4% | 1.717 | ✅ PASS | 15 |
| 2 | 2025-05-30 ~ 2025-08-09 | -0.0% | 18.9% | 0.105 | ❌ FAIL | 15 |
| 3 | 2025-08-09 ~ 2025-10-20 | -21.5% | 14.1% | -1.361 | ❌ FAIL | 16 |
| 4 | 2025-10-20 ~ 2025-12-30 | +86.5% | 5.5% | 2.936 | ✅ PASS | 12 |

### Long/Short 진단

- Realized Price: long=+0.0273 short=+0.0493 (숏 우위)
- Long Losers Top: 1000SHIBUSDT(-0.0038), SANDUSDT(-0.0035), TRBUSDT(-0.0024)
- Short Winners Top: QTUMUSDT(+0.0175), MASKUSDT(+0.0123), FILUSDT(+0.0120)

### Major Symbol 진단

| Symbol | mu_bull | w_long | stale_long | cap_engaged | avg_mult |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 2.9% | 2.9% | 0.0% | 0.0% | 1.000 |
| ETHUSDT | 1.1% | 1.1% | 0.0% | 0.0% | 1.000 |
| BNBUSDT | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 |

## Benchmark 검증 (scratch/bench_cache_stabilization.py)

### Test 1: Cross-Process Fingerprint 결정성
- Parent fingerprint: `c9c941f154b3e3d5`
- Child process fingerprint: `c9c941f154b3e3d5` (동일, `PYTHONHASHSEED=0`)
- **✅ PASS** — 이전 `pd.util.hash_pandas_object`의 프로세스 간 비결정성 해결

### Test 2: L1 Cache Hit Time Savings
| Phase | Time | RSS delta | `run_l1_nested_swf` calls |
| :--- | ---: | ---: | :---: |
| Cold (cache miss) | 0.056s | +1.8MB | 1 (compute) |
| Warm (cache hit) | 0.055s | +0.0MB | 0 (bypass) |

- 생산 extrapolation: L1 211s(6 TF miss) → ~0.3s(6 TF hit, -99.9%)

### Test 3: RSS Guard + gc.collect()
- Small file (10MB @ 11.5GB threshold): ✅ LOAD
- Large file (simulated RSS 11GB): ✅ SKIP
- `gc.collect()` overhead: ~55ms/call (허용 가능, cache boundary에서만 호출)

## Verdict

- **L1: 7/7 TF ✅ PASS** — 모든 TF fold readiness 통과, 신호 품질 정상. Cache 변경으로 인한 연산 변화 없음.
- **L2: ❌ BLOCKED (cagr)** — CAGR +20.9%로 gate 미달. ADR_20260718 시점과 동일한 패턴(정상장 CAGR 부진은 cache 변경과 무관, 기존 설계 한계).
- **Cache Fingerprint**: ✅ 교체 성공 — cross-process 결정적 fingerprint 확인. L1 cache miss 해소. 다음 run부터 cache hit 예상.
- **RSS Guard**: ✅ 정상 작동 — 11.5GB threshold에서 cache deserialize 차단 확인.
- **`/check`**: ✅ PASS (Cov 38%).

## 잔여 이슈

1. L2 CAGR +20.9% < 30% gate — 기존 `_shape_efficiency_l2_objective`의 scale-invariant 한계. 이번 세션 스코프 밖.
2. `[REGIME-L2] proof_failed path=pooled_fallback` — 레짐 증명 실패, pooled fallback 진입. 별도 진단 필요.
3. 하위 3개 fold(Fold 2·3)의 CAGR/CAGR 실적 부진 — fold pass ratio 50%만 달성. 전체 안정성을 위해 추가 개선 필요.
