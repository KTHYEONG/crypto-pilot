# L0/L1 Discovery Snapshot

- **최신 측정일**: `2026-07-14` (run `--phase l1 --timeframe 4h --date 2026-07-13 --trials 1 --seed 42`, `LOG_LEVEL=DEBUG`)
- 이 문서는 **현재 상태와 최신 관측 데이터만** 담는다. 과거 세션의 방대한 반복 로그는 `docs/decisions/decisions.md`/`decisions_archive.md`에 보존.

## 1. L0 → L1 TF별 배포 현황 (최종)

| TF | L0 gate_passed | L1 n_ready | L1 blockers | 비고 |
| --- | ---: | ---: | --- | --- |
| 2h | ✅ | 17 | none | L1 통과 (17개 승격) |
| **4h** | ✅ | **7** | `fold_ratio:0.250` | `labeled events 없음` 해소, 7개 승격 |
| 6h | ✅ | 16 | none | L1 통과 (16개 승격) |
| 8h | ✅ | 70 | none | L1 통과 (70개 승격) |
| 12h | ✅ | 151 | none | L1 통과 (151개 승격) |
| 1d | ✅ | 153 | none | 1d starvation 해소 및 L1 통과 (153개 승격) |
| 1h | — | — | — | L0 단계 제외 |

배포 가능 TF: **5/6 유효 배포**(4h는 부분) — zero-event 원인 수정 후 실측 성과.

## 2. L1 4h zero-event 근본 원인 해결 및 세부 지표

### 2.1 패치 검증 요약
- **TF-Scaled Warm-Up & Injection (Fix 2/3)**: 전 TF에 대한 `inject_membership_masks_into_maps` 루프 및 `MEMBERSHIP_WARMUP_DAYS`(42일) 배선을 완료하여 기존의 `labeled events 없음` 블로킹을 해소했습니다.
- **Effective Evidence Start (Fix 1)**: `_resolve_effective_evidence_start` 연산을 통해 1d/4h 등 긴 warm-up 요구사항을 가진 TF가 L0-evidence 기간을 전부 잠식당하는 starvation 문제를 차단했습니다.

### 2.2 TF별 L1 verification 세부 지표

#### [TF 2H] - ✅ PASSED
- **Ready Folds**: 2/4 Folds Ready (F0, F3 통과 / F1, F2 블로킹)
  - `Fold #0`: 12 symbols, 113 events, Edge: +326.69 bps
  - `Fold #3`: 5 symbols, 55 events, Edge: +71.54 bps
- **최종 검증**: Cov: 1.000 (>=0.80) | Symbol-Breadth: 14.286 (>=3.00) | probe_lcb_bps: 73.874 (>0.00)
- **승격 결과**: 총 17개 alpha recipe-symbol 페어 Promoted (BELUSDT, CTSIUSDT, ARUSDT 등)

#### [TF 4H] - ❌ BLOCKED
- **Ready Folds**: 1/4 Folds Ready (F0 통과 / F1, F2, F3 블로킹)
  - `Fold #0`: 6 symbols, 34 events, Edge: +133.65 bps
- **최종 검증**: Cov: 1.000 (>=0.80) | Symbol-Breadth: 8.000 (>=3.00) | probe_lcb_bps: 95.796 (>0.00)
- **승격 결과**: 총 7개 승격 (OGNUSDT, FETUSDT 등)되었으나 `fold_ratio:0.250` 블로커로 인해 최종 거부

#### [TF 6H] - ✅ PASSED
- **Ready Folds**: 3/4 Folds Ready (F0, F1, F3 통과 / F2 블로킹)
  - `Fold #0`: 11 symbols, 153 events, Edge: +173.92 bps
  - `Fold #1`: 8 symbols, 117 events, Edge: +37.21 bps
  - `Fold #3`: 4 symbols, 81 events, Edge: +127.57 bps
- **최종 검증**: Cov: 1.000 (>=0.80) | Symbol-Breadth: 16.941 (>=3.00) | probe_lcb_bps: 43.318 (>0.00)
- **승격 결과**: 총 16개 alpha recipe-symbol 페어 Promoted (IOTXUSDT, KAVAUSDT, GALAUSDT 등)

#### [TF 8H] - ✅ PASSED
- **Ready Folds**: 4/4 Folds Ready
  - `Fold #0`: 19 symbols, 297 events, Edge: +161.05 bps
  - `Fold #1`: 7 symbols, 90 events, Edge: +87.03 bps
  - `Fold #2`: 6 symbols, 100 events, Edge: +110.71 bps
  - `Fold #3`: 15 symbols, 328 events, Edge: +21.10 bps
- **최종 검증**: Cov: 1.000 (>=0.80) | Symbol-Breadth: 26.614 (>=3.00) | probe_lcb_bps: 35.215 (>0.00)
- **승격 결과**: 총 70개 alpha recipe-symbol 페어 Promoted (BELUSDT, LQTYUSDT, ZRXUSDT 등)

#### [TF 12H] - ✅ PASSED
- **Ready Folds**: 4/4 Folds Ready
  - `Fold #0`: 33 symbols, 692 events, Edge: +205.91 bps
  - `Fold #1`: 16 symbols, 329 events, Edge: +275.80 bps
  - `Fold #2`: 26 symbols, 735 events, Edge: +265.53 bps
  - `Fold #3`: 45 symbols, 1212 events, Edge: +167.69 bps
- **최종 검증**: Cov: 1.000 (>=0.80) | Symbol-Breadth: 51.064 (>=3.00) | probe_lcb_bps: 170.382 (>0.00)
- **승격 결과**: 총 151개 alpha recipe-symbol 페어 Promoted (ZRXUSDT, PEOPLEUSDT, NEOUSDT 등)

#### [TF 1D] - ✅ PASSED
- **Ready Folds**: 4/4 Folds Ready
  - `Fold #0`: 85 symbols, 3841 events, Edge: +249.75 bps
  - `Fold #1`: 89 symbols, 3593 events, Edge: +360.41 bps
  - `Fold #2`: 80 symbols, 4362 events, Edge: +318.09 bps
  - `Fold #3`: 73 symbols, 3501 events, Edge: +384.43 bps
- **최종 검증**: Cov: 1.000 (>=0.80) | Symbol-Breadth: 97.120 (>=3.00) | probe_lcb_bps: 312.530 (>0.00)
- **승격 결과**: 총 153개 alpha recipe-symbol 페어 Promoted (ZRXUSDT, STXUSDT, BELUSDT 등)

