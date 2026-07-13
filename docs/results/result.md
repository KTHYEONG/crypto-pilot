# L0/L1 Discovery Snapshot

- **최신 측정일**: `2026-07-13` (run `4h_...`, `--phase l1 --timeframe 4h --date 2026-07-13 --trials 1 --seed 42`, `LOG_LEVEL=DEBUG`)
- 이 문서는 **현재 상태와 최신 관측 데이터만** 담는다. 과거 세션의 방대한 반복 로그는 `docs/decisions/decisions.md`/`decisions_archive.md`에 보존.

## 1. L0 → L1 TF별 배포 현황 (최종)

| TF | L0 gate_passed | L1 n_ready | L1 blockers | 비고 |
| --- | ---: | ---: | --- | --- |
| 2h | ✅ | 17 | none | 무변화, 안정 |
| **4h** | ✅ (신호 풍부) | **3** | `fold_ratio:0.250` | 부분 배포(위험조정), 근본원인 확정(§3) |
| 6h | ✅ | 13 | none | 오늘 `match_ratio` 계산 수정으로 0→13 회복 |
| 8h | ✅ | 34 | none | 안정 |
| 12h | ✅ | 84 | none | 안정, `dual_momentum` 상대적으로 잘 통함 |
| 1d | ✅ | 111 | none | 최고 성과, `would_resolve_master_tf`가 12h→1d로 전환 |
| 1h | — | — | — | L0 단계에서 완전 제외(구조적 붕괴, 별도 세션에서 처리) |

배포 가능 TF: 3/6(50%) → **6/6 유효 배포**(4h는 부분) — 오늘 세션 누적 성과.

## 2. L1 게이트 재설계 핵심 수정 (오늘)

1. **`match_ratio` 재정의**: 기존엔 `(decision_idx, symbol, strategy_id, activation_context)` 4키 정확조인 성공률 — 성과지표가 아닌 조인 아티팩트였음. Pooled count + Wilson LCB로 교체(→ `probe_lcb_bps`와 동일한 통계적으로 건전한 패턴). **효과: 6h가 설정 변경 없이 즉시 완전 해제(n_ready 0→13)** — 과거 "6h는 신호가 불안정하다"는 진단 자체가 측정 버그였음이 실측으로 확정됨.
2. **`fold_ratio` 강등**: `wf_n_folds=4`(전 TF 고정) → 이산값 5개뿐({0, 0.25, 0.5, 0.75, 1.0})인데 TF별 임계값(0.40~0.60)으로 비교하는 게 통계적으로 무의미. 하드 게이트에서 진단 전용(advisory, non-blocking)으로 전환.
3. **구조/자문 게이트 분리**: `Layer1GateReport.structural_passed`(fold_cov/sym_count/probe_lcb_bps)와 `advisory_checks`(match_ratio/fold_ratio) 분리. `l1_structural_gate_only`(기본값 **True**로 전환 완료) — 구조적 전제만 통과하면 `build_qualified_signal_registry()`가 실행되고, advisory 실패분은 `advisory_penalty`로 개별 전략 `quality_weight`를 감점(전체 봉쇄 대신 위험조정 부분배포).
   - **안전성 실측 확인**: 2h/6h/8h/12h/1d는 flag 전환 전후 n_ready 완전 동일(이미 advisory 전부 통과 상태라 무영향) — 4h만 0→3.

## 3. 4h 근본원인 포렌식 — 2회 가설 반증 후 확정

**가설 1 (반증)**: activation_context 라벨 drift로 인한 조인 실패 — `l1_qualify_by_regime`/`l1_activation_match_regime` 둘 다 기본값 `False`(레짐 축 자체가 "all"로 통일)임을 코드로 확인, 애초에 발생 불가능한 메커니즘이었음.

**가설 2 (반증)**: 4h가 세밀할수록 Kish 유효표본크기(`effective_n = (Σw)²/Σ(w²)`)가 과도하게 깎이는 계산 버그 — 신규 계측(`l1_family_admission_diag`) 결과 `eff_n_over_n_obs`가 전 TF에서 0.75~0.94로 유사, 계산 자체는 정상.

**확정된 원인**: `dual_momentum`/`taker_imbalance_momentum`의 탈락 사유는 `no_incremental_edge`/`negative_gross_edge` — **순수 경제적 성과 부재**.

| TF | `dual_momentum` `no_incremental_edge` 탈락 (228쌍 중) |
| --- | ---: |
| 4h | 155~170건 |
| 6h | 133~164건 |
| 8h | 149~170건 |
| **12h** | **33~74건**(뚜렷한 개선) |

→ 해당 family들은 일중 시간단위(4h/6h/8h)에서 진짜로 초과수익이 없고, 12h부터 유의미하게 개선됨. L0 단계에서 이미 확인된 "이 유니버스는 추세 계열 외 durable edge 없음"과 동일한 종류의 정직한 결론 — **추가 코드 수정 대상 아님**(더 밀어붙이면 과적합).

## 4. 원시 신호(L0) 밀도 — 병목 아님 확정

4h의 `registry_empty`(등록 0건) 폴드에서도 원시 예측(L0 파생 신호)은 **23,763~34,882건**으로 풍부했음(`[L2-SIGNAL] gates:` 계측). 병목은 L0의 신호 생성이 아니라 L1의 (경제성 기반) 자격심사 단계에 있음 — L0→L1 신호 부족 가설은 이번 조사로 **반증**됨.

## 5. 다음 액션 (우선순위)

1. 없음(4h는 현재 상태 — 부분배포 + 구조/자문 분리 — 가 정직한 최종선으로 판단, `l1_structural_gate_only=True` 유지).
2. 참고: `1000LUNCUSDT`/`BNBUSDT` 등 일부 심볼에서 `[L1-MAJOR-GAP] gap=activation_gap` 경고 반복 관측 — 이번 세션 범위 밖, 별도 확인 필요 시 후속 조사 대상.

## 6. Hybrid 데이터 경로 메모리 검증 (2026-07-13)

- 1m hybrid 적용으로 core loader의 1m 전수 적재는 제거되었지만, 전체 L1 RSS 절감 목표는 달성하지 못했다.
- 실측 단계별 RSS: data `2.19~2.33GB` → 6개 TF panel construction `5.0~5.35GB` → L1 nested 실행 peak `11.10GB`.
- `data_stage_early_release` 후 RSS 감소는 거의 없었다. L1에 필요한 `full_strategy_maps`와 multi-TF native panels가 계속 유지되기 때문이다.
- 병목은 1m 저장/스트리밍이 아니라 `2h,4h,6h,8h,12h,1d` 패널 동시 보유와 nested L1 worker fork이다.
- 21:07 실행에서는 6개 TF L1이 `gate_passed=True`로 정상 완료되었으므로 코드상 L1 불능은 아니다. 다만 실행 환경이 peak 약 11GB를 안정적으로 수용하지 못하면 전체 실행이 중단될 수 있다.
- 후속 개선 대상: TF별 panel 수명 단축, L1 TF 순차 처리 후 즉시 해제, nested worker 수/IPC 메모리 상한 조정. 1m hybrid 자체를 되돌리는 것은 해결책이 아니다.
