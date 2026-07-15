# L0/L1 최신 파이프라인 결과 (2026-07-16)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python scripts/run_l1_cross_tf_diagnosis.py`
- **적용**: 일평균 이벤트 밀도 검정 (Density-based Gate Policy) 및 동적 웜업 수치 교정
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **결과 파일**: `logs/futures/diagnostics/l1_cross_tf/treatment.json` 및 [diagnosis.json](file:///home/kth/my_coin_traider/logs/futures/diagnostics/l1_cross_tf/diagnosis.json)
- **프로세스 종료**: `exit_code=0`, `reason=l1_mode_done` (4개 Replay 런 완주)

## L1 판정 (Daily Density Scaling 적용 후)

| Timeframe | 판정 | 관측 결과 | 실제 blocker |
| :--- | :---: | :---: | :--- |
| **1h** (Treatment) | **✅ PASS** | 4/4 folds, 88 valid symbols | 없음 |
| **2h** | **✅ PASS** | 4/4 folds, 77 valid symbols | 없음 (기존 74개 대비 +3 증가) |
| **4h** | **✅ PASS (⚠️)** | 4/4 folds, 34 valid symbols | `fold_ratio:0.250` (기존 12개 대비 **+22개, 183% 비약적 증가**) |
| **6h** | **❌ BLOCKED** | 0 valid symbols | `probe_lcb_bps:-40.459`, `fold_ratio:0.000` (**비정상 -inf 및 sym_count 블로커 제거 완료**) |
| **8h** | **❌ BLOCKED** | 0 valid symbols | `sym_count:2.000`, `probe_lcb_bps:-35.095`, `fold_ratio:0.250` (적자 지속) |
| **12h** | **❌ BLOCKED** | 0 valid symbols | `sym_count:1.000`, `probe_lcb_bps:-inf`, `fold_ratio:0.000` |
| **1d** | **❌ BLOCKED** | 0 valid symbols | `sym_count:1.000`, `probe_lcb_bps:-inf`, `fold_ratio:0.000` |

### 주요 개선 효과
- **신호 개수(Breadth) 대폭 증가**: 4h에서 유효 신호 수가 12개에서 **34개**로 183% 증가했고, 2h 역시 74개에서 **77개**로 증가하였습니다.
- **구조적 Blocker 해소**: 6h TF에서 발생하던 비정상적인 `sym_count:2.000` 및 `probe_lcb_bps:-inf` 블로커가 완전히 사라지고, 동적 밀도 스케일링 덕분에 정상적인 통계 수치에 기반한 수익성 필터링(`-40.459 bps`)으로 정상 가동을 입증하였습니다.
- **정합성 유지**: `ablation_restores_control: true` 를 만족하며 최종 L1 판정의 인과 무결성이 변함없이 증명되었습니다.

## 구현 반영
- [contracts.py](file:///home/kth/my_coin_traider/src/domain/futures/alpha_foundry/contracts.py) 및 [cheap_gate.py](file:///home/kth/my_coin_traider/src/domain/futures/alpha_foundry/cheap_gate.py)에서 일평균 밀도 변수 추가 및 `resolve_family_timeframe_gate_policy` 함수 리팩토링.
- [pipeline.py](file:///home/kth/my_coin_traider/src/domain/futures/alpha_foundry/pipeline.py)에 oos_window_days 동적 도출 연계 배선 완료.
