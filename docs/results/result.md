# L0/L1 최신 파이프라인 결과 (2026-07-16, FDR 스냅샷 디커플링 반영)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python -m src.domain.futures.strategy.run_l1_cross_tf_replay control` (2h/4h/6h/8h/12h/1d, 4h가 대칭 base TF로 필수 포함)
- **이번 세션 적용된 두 차례 아키텍처 수정** (둘 다 실측 control replay로 검증, `docs/decisions/decisions.md` 참조):
  1. `ADR_20260716_L1_BASELINE_FAMILY_SCOPED_ADMISSION`: `no_incremental_edge` 무고한 탈락(washout) 방지 — deployment admission의 incremental baseline을 동일 심볼 전체가 아닌 동일 패밀리로 스코핑(`baseline_mode_override="peer_exclusive_family"`, deployment 호출부만). 최초 구현은 이 설정을 전역 기본값으로 바꿔 walk-forward 스냅샷까지 오염시켜 4h/12h 회귀(PASS→BLOCKED)를 유발했고, 호출부별 override로 재구현하여 회귀 해소.
  2. `ADR_20260716_L1_SNAPSHOT_FDR_DECOUPLING`: walk-forward 스냅샷 admission의 FDR hard-reject(`l1_fdr_hard_reject=True`)를 deployment admission과 분리 — 스냅샷 호출부만 `fdr_hard_reject_override=False`(soft-scale)로 완화, deployment는 strict 유지. 얇은 초기 표본을 가진 느린 TF의 fold가 다중검정 보정으로 통째로 `registry_empty` 처리되던 문제 해소.
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **알려진 이슈**: L1 pipeline은 fold-level 모델 피팅에 `ProcessPoolExecutor`(fork)를 사용하며, 동일 코드/시드에서도 run마다 승급 신호 수가 소폭 다르게 나오는 비결정성이 있음(예: 콘솔 실시간 출력 vs JSON 트레이스 아티팩트 간 수치 불일치 — 둘 다 관측됨, 원인 미해결). 아래 표는 `logs/futures/diagnostics/l1_cross_tf/control.json`의 `l1_result` 트레이스(이번 세션 전체에서 일관되게 사용한 기준)를 우선 표기하고, 콘솔 실시간 출력 값을 괄호로 병기.

## L1 판정 (두 차례 수정 후 최종 실측)

| Timeframe | 판정 | Symbol-Breadth | probe_lcb_bps | 승급 신호 수 (JSON / 콘솔) | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1h** | ⚠️ 미재측정 | — | — | — | 이번 세션 두 수정 이후 재실행 안 됨 (`control` replay는 4h를 base로 하는 2h~1d 6개 TF만 포함; 1h는 `treatment`/`fusion_ablation` label 필요). 최종 재검증 필요. |
| **2h** | **✅ PASS** | 74.312(≥5.00) | 96.902 | 86 / 156 | 두 수정 전후 완전 동일 — 회귀 없음 확인 |
| **4h** | **✅ PASS** | 21.407(≥3.00) | 54.810 | 32 / 37 | blockers=[] (advisory fold_ratio도 통과) |
| **6h** | ❌ BLOCKED | 17.818(≥3.00) | **+13.587** (수정 전 -40.459) | 0 / 0 | 구조 게이트 전부 clean(blockers=[])하고 economics 양전환됐으나 **deployment 단계**에서 0건 승급 — 남은 병목은 deployment FDR/quality_weight의 통계적 검정력 부족(정직한 데이터 부족으로 판단, 추가 완화 보류) |
| **8h** | **✅ PASS** | 81.280(≥2.00) | **+27.384** (수정 전 -147.857) | 25 / 28 | **완전 전환**(BLOCKED→PASS) — 진짜 음의 경제성이 아니라 스냅샷 FDR의 과도한 침묵이 원인이었음이 실측으로 확인됨 |
| **12h** | **✅ PASS** | 61.504(≥1.00) | 104.238 | 18 / 20 | 3/4 fold ready, 두 수정 전후 안정적 |
| **1d** | ❌ BLOCKED | 58.380(≥1.00) | **+88.166** | 0 / 0 | blockers=[] (구조 게이트 전부 clean) — 6h와 동일한 패턴, deployment 단계 병목만 남음 |

## 남은 병목: Deployment-time FDR 통계적 검정력 부족 (6h, 1d)

- 두 차례 수정으로 6h/1d 모두 **walk-forward 구조 게이트가 완전히 clean**해지고 **pooled `probe_lcb_bps`가 강한 양수**로 전환됐음에도, 최종 `QualifiedSignalRegistry` 승급은 여전히 0건.
- 이는 의도적으로 strict 유지한 deployment 단계 FDR(`l1_fdr_hard_reject=True`, `l1_pair_fdr_alpha=0.15`)이 pooled 표본에서도 개별 symbol-strategy pair의 유의성을 인정하지 않는 것으로, 과거 Tier-1 실측(1d/12h `no_incremental_edge`-무관 별도 조사, "q_value>0.15 후보 100%가 정당한 FDR 기각")과 같은 계열의 문제로 잠정 판단.
- **추가 완화는 권장하지 않음** — deployment 단계는 최종 1회 의사결정이므로 다중검정 보정을 약화시키면 quant.md §0.1(anti-overfitting) 위반 소지. 다음 조사가 필요하다면 "더 많은 데이터 확보" 또는 "가설 수 축소(candidate pool 자체를 줄이는 구조적 변경)" 방향이어야 하며, threshold 완화가 아니어야 함.

## 이전 세션 진단 기록 (Tier 3, family-scoped baseline 수정으로 해결됨)

- **배경**: `no_incremental_edge`로 탈락한 `mean_gross_bps > 0` 후보군의 피어 상관관계 실측(`avg_corr`: 피어 평균 시계열과의 Pearson 상관계수) — family-scoped baseline 도입의 근거가 됨.
  - **1d (정상 작동)**: `avg_corr` = +0.8215 — 동일 패밀리 중복 시그널 억제가 정상 작동 중이었음(현재도 유지).
  - **12h (씻아웃 오작동, 수정됨)**: `avg_corr` = -0.0065, `avg_peers` = 1.0 — family-scoped baseline 적용으로 해소.
  - **2h (정상 작동)**: `avg_corr` = +1.0000.
  - **4h (혼재형, 수정됨)**: `avg_corr` = +0.3761.

## 다음 액션 플랜
1. **1h 재검증**: `treatment` 또는 `fusion_ablation` label로 두 수정 반영 후 재실행하여 회귀 여부 확인.
2. **6h/1d deployment 병목 심층 조사**: candidate pool 크기(가설 수) 축소를 통한 FDR 보정 완화 여지 검토, 또는 데이터 확장 가능성 검토 — threshold 튜닝이 아닌 구조적 접근으로 한정.
3. **콘솔/JSON 승급 수 불일치 원인 규명**: `ProcessPoolExecutor` fork 비결정성으로 추정되나 미확정 — 재현 가능한 최소 사례로 근본 원인 특정 필요.
