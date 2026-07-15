# L0/L1 최신 실행 결과

## 측정 범위

- **실행일**: 2026-07-15
- **실행군**: `control` 단일 run
- **실행 방식**: heavy 계산 순차 실행, 내부 worker/thread 제한 1, 추가 run 병렬 실행 없음
- **실행 명령**: `scripts/run_l1_cross_tf_replay.py control`
- **기간**: 2023-07-31 ~ 2026-03-31
- **IS/OOS split**: 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106 (integrity 통과)
- **소스 readiness**: 2h/4h/6h/8h/12h/1d 모두 241/241
- **artifact**: `logs/futures/diagnostics/l1_cross_tf/control.json`

## 결과 요약

| TF | L0 canonical recipes | Native/L1 delivery events | L1 결과 | 주요 원인 |
|---|---:|---:|---|---|
| 2h | 6 | 133,740 | **PASS** (`n_valid=74`) | 4/4 outer folds ready |
| 4h | 6 | 62,664 | BLOCKED (`n_valid=0`) | 첫 fold 외 `registry_empty`, symbol breadth 1.0 |
| 6h | 7 | 52,779 | BLOCKED (`n_valid=0`) | 첫 fold `-146.34 bps`, 이후 `registry_empty` |
| 8h | 7 | 44,734 | BLOCKED (`n_valid=0`) | 1/4 fold만 준비, probe LCB `-27.595 bps` |
| 12h | 8 | 39,169 | BLOCKED (`n_valid=0`) | 첫 fold `-11.53 bps`, 이후 `registry_empty` |
| 1d | 6 | 37,004 | BLOCKED (`n_valid=0`) | 모든 fold `registry_empty` |

### 2h 상세

- Outer folds: 4/4 ready
- Fold events: 125, 439, 635, 245
- Fold edge: 160.48, 191.68, 108.26, 61.52 bps
- Gate summary: Cov 1.000, Symbol-Breadth 43.836, probe LCB 63.499 bps
- Promotion: 124개 pair 승격, 74개 valid L1 결과

### 차단 TF 공통 패턴

- L0 canonical recipe와 native delivery event는 존재한다.
- 첫 fold에서 symbol 부족 또는 음수 gross edge가 발생한다.
- 후속 fold는 `empty_opportunities:registry_empty`로 이어져 `fold_ratio=0`이 된다.
- 따라서 현재 결과는 단순한 source-data 부재가 아니라 L1 fold-level readiness/registry 생성 실패로 분류한다.

## L0 probe와 L1 결과의 불일치

- L0 TF probe는 6개 TF 모두 winning cell `0`으로 기록됐다.
- 같은 실행에서 2h L1은 74개 valid를 산출했다.
- 이는 현재 replay에서 L0 probe survivorship 결과가 L1 admission을 hard gate로 차단하지 않거나, 두 단계의 계약이 분리되어 있음을 의미한다.
- 그러므로 이번 결과를 “L0 probe 통과 신호가 L1에서 검증됐다”고 해석하지 않는다.

## Artifact 신뢰성 한계

현재 `control.json`에는 다음 stage가 없다.

- `terminal_event_audit`
- `outer_folds`

또한 replay script는 `run_pipeline()`의 `RunnerResult`를 최종 process exit code로 승격하지 않아, L1 다수 TF가 BLOCKED여도 process `EXIT_CODE=0`으로 끝났다.

따라서 이번 artifact는 TF별 L0/L1 수치 확인에는 사용할 수 있지만, 다음 용도로는 불완전하다.

- cross-TF 최초 divergence 판정
- run 성공/실패 자동 판정
- OOM 또는 외부 signal 원인 확정
- control 대비 treatment 인과 결론

## 결론

1. 현재 control에서 실질적으로 유효한 TF는 2h 하나다.
2. 4h/6h/8h/12h/1d의 실패는 delivery event가 전혀 없어서가 아니라 fold별 registry와 경제성/기호폭 게이트를 통과하지 못해서 발생했다.
3. 1h 추가가 6h/12h에 미치는 인과는 이번 control 단독 실행으로 판단할 수 없다.
4. process exit `0`은 결과 성공을 의미하지 않는다. artifact의 TF별 gate 상태를 별도로 봐야 한다.

## 다음 단계

1. `RunnerResult`를 replay 최종 status에 연결해 BLOCKED/failed를 non-zero로 보존한다.
2. `terminal_event_audit`와 `outer_folds`를 public diagnostic sink로 기록한다.
3. child 종료 signal, 마지막 checkpoint, peak RSS를 보존하는 sequential supervisor를 연결한다.
4. 위 계측이 완성된 뒤 `control_repeat` → `treatment(1h 추가)` → `fusion_ablation`을 각각 순차 실행한다.
5. 네 run 모두 complete trace일 때만 1h의 6h/12h 영향과 최초 divergence stage를 판정한다.
