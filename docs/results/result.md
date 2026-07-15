# L0/L1 최신 파이프라인 결과 (2026-07-15)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python scripts/run_l1_cross_tf_diagnosis.py`
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **결과 파일**: `logs/futures/diagnostics/l1_cross_tf/treatment.json` 및 [diagnosis.json](file:///home/kth/my_coin_traider/logs/futures/diagnostics/l1_cross_tf/diagnosis.json)
- **프로세스 종료**: `exit_code=0`, `reason=l1_mode_done` (4개 Replay 런 모두 완주)

## L1 판정 (Treatment 런 기준)

| Timeframe | 판정 | 관측 결과 | 실제 blocker |
| :--- | :---: | :---: | :--- |
| **1h** (Treatment) | **✅ PASS** | 4/4 folds, 88 valid symbols | 없음 |
| **2h** | **✅ PASS** | 4/4 folds, 74 valid symbols | 없음 |
| **4h** | **⚠️ WARNING** | 4/4 folds, 12 valid symbols | `fold_ratio:0.250` (경제성은 통과하나 Fold 커버리지 부족) |
| **6h** | **❌ BLOCKED** | 0 valid symbols | `sym_count:2.000`, `probe_lcb_bps:-inf`, `fold_ratio:0.000` |
| **8h** | **❌ BLOCKED** | 0 valid symbols | `sym_count:2.000`, `probe_lcb_bps:-35.095`, `fold_ratio:0.250` (음의 net edge) |
| **12h** | **❌ BLOCKED** | 0 valid symbols | `sym_count:2.000`, `probe_lcb_bps:-inf`, `fold_ratio:0.000` |
| **1d** | **❌ BLOCKED** | 0 valid symbols | `sym_count:1.000`, `probe_lcb_bps:-inf`, `fold_ratio:0.000` |

### 해석 및 Cross-TF Causal Diagnostics
- 1h 신규 주입 및 4-Run Replay(아블레이션 포함) 결과, 최종 L1 의사결정 단계(`l1_result`)에서 통제 집단(Control)과 실험 집단(Treatment)의 다이제스트가 **100% 동일하게 수렴**함이 확인되었습니다 (`ablation_restores_control: true`).
- 이로 인해 시간프레임 간 결합 및 배선 연산의 인과관계 무결성이 입증되었습니다.
- 느린 TF(6h~1d)의 차단은 단순 상수 한계가 아닌, Fold #2~#3 기간의 `registry_empty`에 의한 실제 역사적 데이터 밀도의 부재에 기인합니다.

## 구현 반영
- [candidate_contracts.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/candidate_contracts.py)에 dynamic cost 필드 4종 추가 선언.
- [signal_selection.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/tiered_workflow/signal_selection.py)의 `_compute_pooled_probe_lcb` 내부에서 fold 리포트의 dynamic funding/execution cost가 존재할 경우 이를 우선적으로 연동하여 차감하도록 수정함.
- 회귀 검증 및 린트 가트 `🟢 PASS | All checks passed (Cov 28%)` 통과 완료.
