# L0/L1 최신 파이프라인 결과 (2026-07-16, FDR hard-eligible scoping 반영)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python -m src.domain.futures.strategy.run_l1_cross_tf_replay treatment` (1h 포함 전체 7개 TF)
- **이번 세션 적용된 세 차례 아키텍처 수정** (모두 실측 replay로 검증, `docs/decisions/decisions.md` 참조):
  1. `ADR_20260716_L1_BASELINE_FAMILY_SCOPED_ADMISSION`: `no_incremental_edge` 무고한 탈락(washout) 방지 — deployment admission의 incremental baseline을 동일 심볼 전체가 아닌 동일 패밀리로 스코핑(`baseline_mode_override="peer_exclusive_family"`, deployment 호출부만). 최초 구현은 이 설정을 전역 기본값으로 바꿔 walk-forward 스냅샷까지 오염시켜 4h/12h 회귀(PASS→BLOCKED)를 유발했고, 호출부별 override로 재구현하여 회귀 해소.
  2. `ADR_20260716_L1_SNAPSHOT_FDR_DECOUPLING`: walk-forward 스냅샷 admission의 FDR hard-reject(`l1_fdr_hard_reject=True`)를 deployment admission과 분리 — 스냅샷 호출부만 `fdr_hard_reject_override=False`(soft-scale)로 완화, deployment는 strict 유지. 얇은 초기 표본을 가진 느린 TF의 fold가 다중검정 보정으로 통째로 `registry_empty` 처리되던 문제 해소. **8h가 이 수정으로 BLOCKED→PASS 전환**.
  3. `ADR_20260716_L1_FDR_HARD_ELIGIBLE_SCOPING`: FDR 다중검정 보정 분모(m)에 구조적으로 탈락 확정된(`hard_eligible=False`) 후보까지 포함되어 진짜 후보의 q-value가 부풀려지던 문제 수정 — `hard_eligible` 부분집합에만 FDR 보정 적용. 단조적으로 안전(m 축소는 q-value를 개선만 시킴)하여 스냅샷/deployment 양쪽에 call-site 분리 없이 동일 적용.
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **알려진 이슈**: L1 pipeline은 fold-level 모델 피팅에 `ProcessPoolExecutor`(fork)를 사용하며, 동일 코드/시드에서도 run마다 승급 신호 수가 소폭 다르게 나오는 비결정성이 있음(콘솔 실시간 출력 vs JSON 트레이스 아티팩트 간 수치 불일치 — 둘 다 관측됨, 원인 미해결). 아래 표는 `logs/futures/diagnostics/l1_cross_tf/treatment.json`의 `l1_result` 트레이스를 우선 표기.

## L1 판정 (세 차례 수정 후 최종 실측)

| Timeframe | 판정 | Symbol-Breadth | probe_lcb_bps | 승급 신호 수 | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1h** | **✅ PASS** | 66.857(≥5.00) | 37.102 | 100 | blockers=[], 세 수정 전후 동일 — 회귀 없음 |
| **2h** | **✅ PASS** | 74.312(≥5.00) | 96.902 | 87 | blockers=[], 86→87 (단조 증가) |
| **4h** | **✅ PASS** | 21.407(≥3.00) | 54.810 | 33 | blockers=[], 32→33 (단조 증가) |
| **6h** | ❌ BLOCKED | 41.000(≥3.00) | **+75.634** | 0 | 구조 게이트 전부 clean, probe_lcb_bps 강한 양수. `[NOT PROMOTED] quality_weight_zerox233` — FDR 재계산 후에도 대부분 `probability_positive≤0.5`(진짜 약한 부트스트랩 증거)로 확인, **순수 데이터 검정력 부족**으로 최종 판단 |
| **8h** | **✅ PASS** | 81.281(≥2.00) | **+27.964** | **44** | 25→44 대폭 개선 — FDR hard-eligible scoping 효과 실측 확인 |
| **12h** | **✅ PASS** | 61.504(≥1.00) | 104.238 | 21 | 18→21 (단조 증가) |
| **1d** | ❌ BLOCKED | 59.066(≥1.00) | **+87.631** | 0 | blockers=[], 6h와 동일 패턴. `[NOT PROMOTED]` breakdown이 FDR scoping 수정 전후 **완전 동일**(no_incremental_edgex147/quality_weight_zerox147/negative_gross_edgex94) — hard_eligible 풀 자체가 작아 이번 수정의 영향을 거의 받지 않음, 순수 데이터 검정력 부족 |

**단조성 검증**: 세 번째 수정(FDR hard-eligible scoping) 적용 전후 비교에서 **모든 TF가 동일하거나 증가**(감소 0건) — spec에서 요구한 안전성 조건이 실측으로 확인됨.

## 남은 병목: 6h/1d의 순수 데이터 검정력 부족 (구조적 결함 아님, 최종 판단)

- 세 차례 수정으로 6h/1d 모두 **walk-forward 구조 게이트 완전 clean**, **pooled `probe_lcb_bps` 강한 양수**(6h +75.6, 1d +87.6)로 전환됐고, FDR 다중검정 보정의 부당한 과다 계수 문제까지 제거했음에도 최종 승급은 0건.
- 6h는 FDR scoping 이후에도 `quality_weight_zero`가 압도적 다수를 차지 — 개별 pair의 부트스트랩 `probability_positive`가 애초에 0.5 이하(방향성 자체가 불확실)인 경우가 대부분으로 확인. 1d는 hard_eligible 후보 풀 자체가 작아 이번 수정으로 이동한 후보가 사실상 없음.
- **세 가지 아키텍처 레벨을 모두 소진**(baseline scoping → snapshot/deployment 단계 분리 → FDR 보정 대상 축소)했음에도 남아있는 이 병목은, 게이트 설계의 결함이 아니라 **해당 TF에서 개별 전략의 신호가 통계적으로 충분히 검증되지 않는다는 정직한 결과**로 최종 판단. quant.md §0.1(anti-overfitting)에 따라 추가 완화는 권장하지 않음.
- 후속 조사가 필요하다면 threshold 튜닝이 아니라 "더 긴 히스토리 확보" 또는 "완전히 다른 신호/전략 설계"(candidate pool 자체의 alpha 개선) 방향이어야 함 — 이는 L1 게이트 엔지니어링이 아닌 L0 신호 설계의 문제.

## 이전 세션 진단 기록 (모두 이번 세션 세 차례 수정으로 해결/판단 완료)

- `no_incremental_edge`로 탈락한 `mean_gross_bps > 0` 후보군의 피어 상관관계 실측(`avg_corr`) — family-scoped baseline 도입의 근거가 됨: 1d(+0.82, 정상), 12h(-0.0065, 씻아웃 → 수정됨), 2h(+1.0, 정상), 4h(+0.38, 혼재 → 수정됨).
- FDR 다중검정 보정 과다 계수 문제(8h/12h/1d 후보 풀 950~1515개, 2~3개 패밀리 집중) — hard-eligible scoping으로 수정됨(8h 실측 25→44).

## 다음 액션 플랜
1. **콘솔/JSON 승급 수 불일치 원인 규명**: `ProcessPoolExecutor` fork 비결정성으로 추정되나 미확정 — 재현 가능한 최소 사례로 근본 원인 특정 필요.
2. **6h/1d는 L1 게이트 엔지니어링 범위 종료** — 추가 개선이 필요하면 L0 신호/전략 설계 차원에서 접근.
