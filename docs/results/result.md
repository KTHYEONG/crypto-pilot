# L0/L1 최신 파이프라인 결과 (2026-07-16, 2차 레지스트리 어드미션 갭 분석 반영)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python src/domain/futures/strategy/run_l1_cross_tf_diagnosis.py`
- **적용**: (1) 일평균 이벤트 밀도 게이트, (2) 풀링/fold-level 심볼 다양성 하한 측정-후-채택 재보정, (3) 부트스트랩 블록 수 적응형 LCB quantile, (4) pair-level FDR 절차(BY→BH) 측정-후-채택 재보정, (5) missed adaptive LCB quantile 버그 수정 (Tier 2)
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **프로세스 종료**: `exit_code=0`, `reason=l1_mode_done` (4개 Replay 런 수치 완전 일치 — `ablation_restores_control: true`)

## L1 판정 (missed adaptive LCB quantile 버그 수정 후)

| Timeframe | 판정 | Symbol-Breadth | probe_lcb_bps | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **1h** | **✅ PASS** | 33.032(≥5.00) | 36.312 | 회귀 없음, 231개 시그널 승급 |
| **2h** | **✅ PASS** | 51.805(≥5.00) | 82.071 | 회귀 없음, LCB 개선 및 156개 시그널 승급 |
| **4h** | **❌ BLOCKED** | 3.000(≥3.00) | 18.990 | `fold_ratio:0.250` (advisory → BLOCKED), 39개 승급 |
| **6h** | **❌ BLOCKED** | 4.000(≥3.00) | -40.459 | 경제적 무엣지 상태 유지, 0개 승급 |
| **8h** | **❌ BLOCKED** | 11.000(≥2.00) | -147.857 | `fold_ratio:0.250` (BLOCKED), 진짜 마이너스 분기(fold#1) 노출 |
| **12h** | **✅ PASS** | 9.000(≥1.00) | 87.249 | 3/4 fold ready, 11개 시그널 승급 (PEOPLEUSDT LCB+666bps 등) |
| **1d** | **❌ BLOCKED** | 2.000(≥1.00) | 143.346 | `fold_ratio:0.250` (BLOCKED), 구조 게이트 통과했으나 승급은 여전히 0건 |

## L1 QualifiedSignalRegistry Admission Gap 진단 결과 (Tier 3 실측)

- **배경**: 1d에서 `probe_lcb_bps`는 +143.3bps로 강한 양수이나 개별 후보 승급은 0건인 원인 규명을 위해, `no_incremental_edge`로 구조적 탈락한 `mean_gross_bps > 0` 후보군 전수 계측.
- **실측 데이터 요약 (avg_corr: 피어 평균 시계열과의 Pearson 상관계수)**:
  - **1d (정상 작동 🟢)**: `total_rejected` = 96, `avg_corr` = **+0.8215**
    - 동일 버킷 내 피어 시그널 간의 상관관계가 매우 높아, 동일 패밀리의 중복 시그널 유입을 방어하는 중복 억제 필터 본래 기능이 정상 작동 중임.
  - **12h (씻아웃 오작동 🔴)**: `total_rejected` = 65, `avg_corr` = **-0.0065**, `avg_peers` = 1.0
    - 무상관한 별개의 독립 전략들이 단지 동일 심볼/홀딩 기간을 공유한다는 이유만으로 baseline에 묶여 동반 탈락하는 현상 확인.
  - **2h (정상 작동 🟢)**: `total_rejected` = 88, `avg_corr` = **+1.0000**
    - 상관관계 1.0의 완전 중복 시그널 억제.
  - **4h (혼재형 🟡)**: `total_rejected` = 2,373, `avg_corr` = **+0.3761**

## 다음 액션 플랜
1. **피어 수 임계치 도입**: 버킷 내 고유 패밀리 수가 3개 미만(피어 수 1개 이하)인 Thin-breadth 환경의 경우 baseline 감산 필터링을 미적용하고 `absolute` 모드로 강제 폴백(Option A).
2. **패밀리 전용 Baseline 감산**: baseline 비교 대상을 동일 심볼 전체가 아닌 동일 패밀리로 묶어 무상관 전략 간의 오인 사격을 차단(Option B).

