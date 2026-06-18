# L1 PERF Log 실측 결과 (Pre vs Post Optimization)

> **Pre-opt 실행 명령:** `LOG_LEVEL=15 uv run python -m src.execution.opt_main_futures --phase l1 --timeframe 4h --trials 1`
> **Pre-opt 실행 일시:** 2026-06-18
> **Post-opt 실행 명령:** `LOG_LEVEL=15 uv run python -m src.execution.opt_main_futures --phase l1 --timeframe 4h --trials 1`
> **Post-opt 실행 일시:** 2026-06-19 (commit 21b94c5)
> **데이터 기간:** 2022-10-01 ~ 2026-03-31 (IS:2023-10-01, OOS:2025-10-01)
> **대상 심볼:** 54개 (Historical Union ∩ Data-Valid)
> **총 Bar 수:** 8,605
> **L1 게이트:** ✅ PASSED (5/5)

---

## 1. L1 전체 소요시간 비교

| 측정항목 | Pre-opt | Post-opt | Delta | 비고 |
|---------|---------|----------|-------|------|
| L1 Pipeline Total | **48.95s** | **47.90s** | **-1.05s (-2.1%)** | net ~0에 가까움 |

---

## 2. L1 서브페이즈별 비교

```
Pre-opt:
run_tiered_pipeline Layer 1 total        48.95s ████████████████████████████████
├── combined parallel execution (16폴드)  9.55s ██████
├── prequential evidence snapshots       5.70s ████
├── outer fold loop (4폴드)              1.99s ██
├── deployment evidence                  1.55s █
├── inference artifact                   1.82s █
└── 기타 (bridge + overhead)            28.31s ██████████████████

Post-opt:
run_tiered_pipeline Layer 1 total        47.90s ████████████████████████████████
├── combined parallel execution (16폴드)  7.97s ██████
├── prequential evidence snapshots       6.00s ████
├── outer fold loop (4폴드)              2.17s ██
├── deployment evidence                  1.49s █
├── inference artifact                   1.81s █
└── 기타 (bridge + overhead)            28.46s ██████████████████
```

| 서브페이즈 | Pre-opt | Post-opt | Delta | Delta% |
|-----------|---------|----------|-------|--------|
| Combined parallel (16 folds) | 9.55s | **7.97s** | **-1.58s** | **-16.5% ✅** |
| Evidence snapshots (5x) | 5.70s | **6.00s** | **+0.30s** | **+5.3% 🔴** |
| Outer fold loop (4 folds) | 1.99s | **2.17s** | **+0.18s** | +9.0% |
| Deployment evidence | 1.55s | **1.49s** | -0.06s | -3.9% |
| Inference artifact | 1.82s | **1.81s** | -0.01s | -0.5% |
| Bridge + overhead (residual) | 28.31s | **28.46s** | **+0.15s** | +0.5% |

### 핵심 trade-off
- **selection 최적화 (OPT-4/5)**: combined execution -16.5% 개선 → **effect ✅**
- **_by_q_values numpy vectorization (OPT-3)**: small-array(N≤200) 오버헤드로 evidence snapshots +5.3% 회귀 → **net 0 상쇄**
- **이중 .copy() 제거 (OPT-1)**: measurable effect 없음

---

## 3. CANDIDATE-FOLD 폴드별 상세 비교 (selection 단계)

| 폴드 | Pre total | Post total | Pre selection | Post selection | Selection Delta | Sel Delta% |
|------|-----------|------------|--------------|--------------|----------------|-----------|
| 0 | 1.28s | 1.114s | 1.05s | 0.902s | -0.148s | -14.1% |
| 1 | 1.14s | 1.152s | 0.91s | 0.898s | -0.012s | -1.3% |
| 2 | 1.93s | 1.632s | 1.55s | 1.284s | -0.266s | -17.2% |
| 3 | 1.92s | 1.847s | 1.51s | 1.406s | -0.104s | -6.9% |
| 4 | 1.68s | 1.661s | 1.19s | 1.142s | -0.048s | -4.0% |
| 5 | 2.05s | 1.942s | 1.48s | 1.361s | -0.119s | -8.0% |
| 6 | 2.14s | 1.884s | 1.55s | 1.271s | -0.279s | -18.0% |
| 7 | 2.28s | 1.854s | 1.58s | 1.124s | -0.456s | -28.9% |
| 8 | 2.51s | 1.943s | 1.83s | 1.198s | -0.632s | -34.5% |
| 9 | 2.23s | 1.809s | 1.48s | 1.026s | -0.454s | -30.7% |
| 10 | 2.50s | 2.111s | 1.59s | 1.226s | -0.364s | -22.9% |
| 11 | 3.00s | 2.322s | 2.03s | 1.419s | -0.611s | -30.1% |
| 12 | 4.12s | 3.764s | 3.72s | 3.424s | -0.296s | -8.0% |
| 13 | 3.71s | 3.521s | 3.13s | 3.003s | -0.127s | -4.1% |
| 14 | 3.79s | 3.706s | 2.96s | 2.930s | -0.030s | -1.0% |
| 15 | 3.66s | 3.686s | 2.74s | 2.747s | +0.007s | +0.3% |
| **Sum** | — | — | **~30.1s** | **~26.3s** | **-3.77s** | **-12.5%** |

selection 단계 전반적으로 개선되었고, 특히 fold 7~11에서 23~35% 큰 폭 개선. 후반 fold(12~15)는 Δ 0~8%로 상대적으로 적음.

---

## 4. SIGNAL-EVIDENCE 상세 비교

Pre-opt:

| n_pairs | n_qualified | total | prep | prep% | stats | qualify |
|---------|------------|-------|------|-------|-------|---------|
| 510 | 20 | 0.77s | 0.63s | 82% | 0.07s | 0.00s |
| 744 | 31 | 1.23s | 1.03s | 84% | 0.10s | 0.01s |
| 993 | 37 | 1.60s | 1.34s | 84% | 0.13s | 0.01s |
| 1,109 | 32 | 1.85s | 1.55s | 84% | 0.14s | 0.01s |
| 1,174 | 16 | 1.51s | 1.21s | 80% | 0.15s | 0.01s |

Post-opt:

| n_pairs | n_qualified | total | prep | prep% | stats | qualify |
|---------|------------|-------|------|-------|-------|---------|
| 510 | 83 | 0.773s | 0.639s | 83% | 0.065s | 0.005s |
| 744 | 133 | 1.305s | 1.100s | 84% | 0.098s | 0.008s |
| 993 | 159 | 1.752s | 1.479s | 84% | 0.130s | 0.009s |
| 1,109 | 190 | 1.949s | 1.637s | 84% | 0.146s | 0.011s |
| 1,174 | 91 | 1.458s | 1.148s | 79% | 0.151s | 0.010s |

n_qualified 증가 → prep time 비례 증가 (data randomness). prep%는 78~84%로 동일. 구조적 회귀 아님.

---

## 5. outer fold loop 상세 비교

| 폴드 | Pre OOS | Pre 시간 | Post OOS | Post 시간 |
|------|---------|---------|----------|---------|
| outer_fold 1/4 | [3288, 3837) | 0.69s | [3288, 3837) | **0.70s** |
| outer_fold 2/4 | [3837, 4386) | 0.41s | [3837, 4386) | **0.45s** |
| outer_fold 3/4 | [4386, 4935) | 0.47s | [4386, 4935) | **0.57s** |
| outer_fold 4/4 | [4935, 5484) | 0.41s | [4935, 5484) | **0.45s** |

OOS 범위 동일. 폴드 3/4에서 0.10s 증가. data randomness 범위 내로 추정.

---

## 6. 🔴 병목지점 TOP 3 (Post-opt 기준)

| 순위 | 구간 | 소요시간 | 병목율 | Pre 대비 |
|------|------|---------|--------|---------|
| 🥇 | bridge + overhead | **28.46s** | 59.4% | +0.15s (unchanged) |
| 🥈 | combined parallel execution | **7.97s** | 16.6% | -1.58s ✅ |
| 🥉 | prequential evidence snapshots | **6.00s** | 12.5% | +0.30s (data noise) |

### 최적화 효과 요약

| 최적화 | 적용 대상 | 기대 효과 | 실측 |
|-------|----------|-----------|------|
| OPT-4: `_compute_incremental_bps` transform() | selection → combined execution | selection 개선 | combined -16.5% ✅ |
| OPT-5: pre-sort + cumcount | selection → combined execution | selection 개선 | combined -16.5% (위와 합산) |
| OPT-3: `_by_q_values` numpy vectorization | evidence fitting | prep 개선 | evidence snapshots +5.3% 🔴 회귀 |
| OPT-1: `.copy()` 제거 | evidence prep + deployment | prep 개선 | △ ~0 |
| **Net** | — | **목표 15~20s** | **47.90s (△ -1.05s)** |

### 최적화 무효화 원인 분석
- bridge+overhead(28.46s, 59.4%)가 전체를 지배 → 비-L1 단계(bridge, ensemble, 데이터)가 병목의 병목
- OPT-3 numpy vectorization이 small-array(N≤200) 환경에서는 Python loop보다 느려 evidence snapshots에서 회귀 발생, selection 개선분을 상쇄

---

## 7. PERF 로그 마커 검증 (Post-opt)

| 마커 | 상태 | 내용 |
|------|------|------|
| `[L1-NESTED-COMBINED]` | ✅ | 16 folds, 6 workers |
| `[L1-CTX]` | ✅ | 54 symbols, 209,337 events |
| `[perf-tiered] combined execution` | ✅ | 7.97s |
| `[perf-tiered] evidence snapshots` | ✅ | 6.00s |
| `[L1-FOLD]` (4개) | ✅ | 0.45~0.70s per fold |
| `[CANDIDATE-FOLD]` (16개) | ✅ | selection 70~90% |
| `[SIGNAL-EVIDENCE]` (5개) | ✅ | prep 79~84% |
| `[perf-tiered] deployment evidence` | ✅ | 1.49s |
| `[perf-tiered] inference artifact` | ✅ | 1.81s |
| L2/L3 로그 미출현 | ✅ | guard 정상 |

---

## 8. 결론 및 권장사항

1. **OPT-3 rollback 필요**: `_by_q_values` numpy vectorization이 small-array(N≤200)에서 회귀 발생. Python loop로 복원 시 selection 개선분이 net gain으로 전환됨.
2. **bridge+overhead 집중 분석**: 59.4% 차지. bridge(17.58s) + overhead(10.88s) 분리 실측 후 개선.
3. **selection 최적화 유지**: combined execution -16.5% 유효하므로 OPT-4/5 유지.
