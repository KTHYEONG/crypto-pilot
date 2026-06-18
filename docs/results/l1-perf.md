# L1 PERF Log 실측 결과

> **실행 명령:** `LOG_LEVEL=15 uv run python -m src.execution.opt_main_futures --phase l1 --timeframe 4h --trials 1`
> **실행 일시:** 2026-06-18
> **데이터 기간:** 2022-10-01 ~ 2026-03-31 (IS:2023-10-01, OOS:2025-10-01)
> **대상 심볼:** 54개 (Historical Union ∩ Data-Valid)
> **총 Bar 수:** 8,605
> **L1 게이트:** ✅ PASSED (5/5)

---

## 1. L1 전체 소요시간

| 구간 | 소요시간 | 비율 |
|------|---------|------|
| L1 Pipeline Total | **48.95s** | 100% |

---

## 2. L1 서브페이즈별 소요시간

```
run_tiered_pipeline Layer 1 total        48.95s ████████████████████████████████
├── combined parallel execution (16폴드)  9.55s ██████
├── prequential evidence snapshots       5.70s ████
├── outer fold loop (4폴드)              1.99s ██
├── deployment evidence                  1.55s █
├── inference artifact                   1.82s █
└── 기타 (bridge + overhead)            28.31s ██████████████████
```

### 2.1 outer fold loop 상세

| 폴드 | OOS 범위 | events | 소요시간 |
|------|---------|--------|---------|
| outer_fold 1/4 | [3288, 3837) | 23,358 | 0.69s |
| outer_fold 2/4 | [3837, 4386) | 14,175 | 0.41s |
| outer_fold 3/4 | [4386, 4935) | 16,115 | 0.47s |
| outer_fold 4/4 | [4935, 5484) | 14,415 | 0.41s |

---

## 3. 폴드별 상세 병목 (CANDIDATE-FOLD)

폴드별 `selection` 단계가 전체 소요시간의 **70~90%** 를 차지.

| 폴드 | total | schema | ds_fit | edge | inference | **selection** | selection% |
|------|-------|--------|--------|------|-----------|--------------|------------|
| 0 | 1.28s | 0.10s | 0.06s | 0.02s | 0.03s | **1.05s** | 82% |
| 1 | 1.14s | 0.10s | 0.05s | 0.02s | 0.04s | **0.91s** | 80% |
| 2 | 1.93s | 0.11s | 0.10s | 0.03s | 0.10s | **1.55s** | 80% |
| 3 | 1.92s | 0.04s | 0.19s | 0.04s | 0.10s | **1.51s** | 78% |
| 4 | 1.68s | 0.14s | 0.17s | 0.06s | 0.07s | **1.19s** | 71% |
| 5 | 2.05s | 0.14s | 0.24s | 0.07s | 0.07s | **1.48s** | 72% |
| 6 | 2.14s | 0.08s | 0.31s | 0.08s | 0.08s | **1.55s** | 72% |
| 7 | 2.28s | 0.07s | 0.38s | 0.11s | 0.08s | **1.58s** | 69% |
| 8 | 2.51s | 0.09s | 0.39s | 0.11s | 0.05s | **1.83s** | 73% |
| 9 | 2.23s | 0.10s | 0.44s | 0.13s | 0.03s | **1.48s** | 66% |
| 10 | 2.50s | 0.13s | 0.50s | 0.16s | 0.07s | **1.59s** | 64% |
| 11 | 3.00s | 0.08s | 0.52s | 0.17s | 0.10s | **2.03s** | 68% |
| **12** | **4.12s** | 0.01s | 0.09s | 0.03s | 0.21s | **3.72s** | **90%** |
| **13** | **3.71s** | 0.05s | 0.23s | 0.08s | 0.17s | **3.13s** | **84%** |
| 14 | 3.79s | 0.09s | 0.41s | 0.10s | 0.17s | **2.96s** | 78% |
| 15 | 3.66s | 0.13s | 0.48s | 0.11s | 0.15s | **2.74s** | 75% |

---

## 4. SIGNAL-EVIDENCE 상세

evidence snapshot마다 `prep` 단계가 **78~84%** 차지.

| n_pairs | n_qualified | total | **prep** | prep% | stats | qualify |
|---------|------------|-------|---------|-------|-------|---------|
| 510 | 20 | 0.77s | **0.63s** | 82% | 0.07s | 0.00s |
| 744 | 31 | 1.23s | **1.03s** | 84% | 0.10s | 0.01s |
| 993 | 37 | 1.60s | **1.34s** | 84% | 0.13s | 0.01s |
| 1,109 | 32 | 1.85s | **1.55s** | 84% | 0.14s | 0.01s |
| 1,174 | 16 | 1.51s | **1.21s** | 80% | 0.15s | 0.01s |

---

## 5. 🔴 병목지점 TOP 3

| 순위 | 구간 | 소요시간 | 병목율 | 상세 |
|------|------|---------|--------|------|
| 🥇 | bridge + overhead | **28.31s** | 58% | pre-tiered 단계 (데이터 준비, bridge, ensemble) |
| 🥈 | combined parallel execution | **9.55s** | 20% | 16개 evidence 폴드 병렬 피팅 (6 workers) |
| 🥉 | prequential evidence snapshots | **5.70s** | 12% | 5회 signal evidence + registry 빌드 |

### 5.1 서브-병목

| 구간 | 비고 |
|------|------|
| **selection** | 모든 폴드에서 70~90% 차지. 폴드 12는 3.72s/4.12s=90% |
| **ds_fit** | 후반 폴드에서 0.05s → 0.52s로 증가 (데이터 누적) |
| **prep** (SIGNAL-EVIDENCE) | 전체 evidence 시간의 78~84% |

---

## 6. PERF 로그 마커 검증

| 마커 | 상태 | 내용 |
|------|------|------|
| `[L1-NESTED-COMBINED]` | ✅ | 16 folds, 6 workers |
| `[L1-CTX]` | ✅ | 54 symbols, 209,337 events |
| `[perf-tiered] combined execution` | ✅ | 9.55s |
| `[perf-tiered] evidence snapshots` | ✅ | 5.70s |
| `[L1-FOLD]` (4개) | ✅ | 0.41~0.69s per fold |
| `[CANDIDATE-FOLD]` (16개) | ✅ | selection 70~90% |
| `[SIGNAL-EVIDENCE]` (5개) | ✅ | prep 78~84% |
| `[perf-tiered] deployment evidence` | ✅ | 1.55s |
| `[perf-tiered] inference artifact` | ✅ | 1.82s |
| L2/L3 로그 미출현 | ✅ | guard 정상 |
