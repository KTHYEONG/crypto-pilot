# L1→L2 최신 Replay 결과 — 2026-07-16

## 실행 조건과 데이터 무결성

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-16 --seed 42`
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31 ~ 2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 125; L1 admission: 114/125 (`late_start` 11 제외).
- 실행 정책: L2도 L0 gate를 실행해 native TF labeled events를 생성·전달한다. 이전 `missing native labeled events for tf=1h` 오류는 재발하지 않았다.
- L0/L1은 single heavy-process로 실행했다. L2 allocation은 master selection 전 fail-closed되어 실행되지 않았다.

## L1 결과

| Timeframe | Fold readiness | Symbol breadth | Probe LCB (bps) | Final promoted pairs | 판정 |
| :--- | :---: | ---: | ---: | ---: | :---: |
| 1h | 4/4 | 68.554 | +68.850 | 237 | PASS |
| 2h | 4/4 | 86.185 | +106.455 | 164 | PASS |
| 4h | 3/4 | 44.579 | +37.238 | 59 | PASS |
| 6h | 4/4 | 29.755 | +49.121 | 14 | PASS |
| 8h | 4/4 | 105.662 | +80.662 | 186 | PASS |
| 12h | 4/4 | 106.333 | +73.830 | 98 | PASS |
| 1d | 4/4 | 95.006 | +83.277 | 3 | PASS |

- 4h fold #2는 gross edge `-54.21 bps`로 block됐지만 나머지 3개 fold가 통과해 aggregate L1 gate는 PASS다.
- 1d의 승급은 `STGUSDT / btc_regime_pullback`, `ENSUSDT / btc_regime_pullback`, `GALAUSDT / trend_donchian` 3개다. 신호 수 확대를 위한 threshold 완화는 수행하지 않았다.
- L0가 실제로 TF별 native artifact를 생성했다. 예: 1h/2h/6h/8h/12h/1d native panel injection은 각각 4/7/10/11/11/4개였다.

## L2 결과와 차단 사유

```text
TieredPipelineError: no deployable timeframe found for L2 master TF
```

- native TF event handoff는 정상: 모든 L1 TF가 native labeled-event map을 받아 검증을 완료했다.
- 그러나 L2는 시작하지 못했다. `_resolve_l2_master_tf()`가 아직 legacy `_is_deployable_per_tf_result()`만 사용하고, 새 `assess_l1_tf_handoff()`의 master/auxiliary readiness를 연결하지 않았기 때문이다.
- legacy deployability는 `Layer1Result.gate_passed`, non-empty deployment registry, `ready_symbols`를 동시에 요구한다. 이번 run에서 master 후보가 0개로 평가됐다.
- 따라서 L2 CAGR, MDD, turnover, allocation 및 production promotion에 대한 결론은 **미산출**이다.

## Verdict

- **L0→L1→native TF handoff:** PASS.
- **전체 TF L1 robustness gate:** PASS.
- **L2 master selection 및 allocation:** BLOCKED; 성과 수치 해석·production promotion 금지.

## 다음 조치

1. `_resolve_l2_master_tf()`에 `assess_l1_tf_handoff()`를 연결하고 TF별 `ready_symbol_count`, family diversity, evidence-based edge quality, rejection reason을 trace로 기록한다.
2. master-eligible TF만 master 후보로 사용하고, 1d처럼 좁은 TF는 auxiliary sleeve로 유지한다.
3. 동일 cutoff·seed replay를 다시 실행해 master 선택과 L2 gate 결과를 확인한다.
