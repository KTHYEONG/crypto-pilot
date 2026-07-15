# L0/L1 최신 파이프라인 결과 (2026-07-15)

## 실행 및 데이터 무결성

- 실행: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. timeout 1800 uv run python scripts/run_l1_cross_tf_replay.py control`
- 기간: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- Universe: Pool 377 → Selected 150 → Loaded 106
- L1 admission: 101/106 symbols (5개 `late_start` 제외)
- 결과 파일: `logs/futures/diagnostics/l1_cross_tf/control.json`
- 프로세스 종료: `exit_code=0`, `reason=l1_mode_done`

## L1 판정

| Timeframe | 판정 | 관측 결과 | 실제 blocker |
|---|---|---:|---|
| 2h | PASS | 4/4 folds, LCB 55.999 bps, 74 valid symbols | 없음 |
| 4h | BLOCKED | LCB 18.990 bps, 1/4 folds | `fold_ratio:0.250` |
| 6h | BLOCKED | 0 valid symbols | `registry_empty`, symbol breadth 2 |
| 8h | BLOCKED | LCB -35.095 bps, 1/4 folds | 음의 net edge, `fold_ratio:0.250`, breadth 2 |
| 12h | BLOCKED | 0 valid symbols | `registry_empty`, `fold_ratio:0` |
| 1d | BLOCKED | 0 valid symbols | `registry_empty`, breadth 1 |

### 해석

- 새 pooled LCB는 경제적으로 음수인 fold를 임의로 제거하지 않는다.
- `-inf`는 유효한 경제 데이터가 하나도 없을 때만 발생한다.
- 4h는 경제성(18.990 bps)은 통과하지만 fold coverage가 부족하다.
- 6h/12h/1d는 임계값 문제가 아니라 과거 구간의 opportunity registry 자체가 비어 있다.
- 8h는 비용 반영 후 음의 보수적 LCB이므로 통과시키면 안 된다.
- 현재 결과는 2h만 실거래 후보로 승격 가능하며, 다른 TF를 강제 통과시키는 것은 look-ahead/과적합 위험이 있다.

## 구현 반영

- L0 후보 생성에 `L1CausalFeedback` cutoff 검증 및 feedback multiplier 경로 연결.
- L1 pooled probe LCB를 gross edge에서 execution cost 차감 net edge 기반으로 변경.
- support blocker와 negative economics를 분리해 데이터 부족과 실제 음의 신호를 구분.
- `evidence_policy.py`에 fold assessment 및 pooled evidence 계약 추가.
- 회귀 검증: `lean_check` 전체 대상 `🟢 PASS | All checks passed (Cov 64%)`.

## 제한 및 후속 조치

- 이번 control replay에는 과거 시점의 L1 feedback artifact를 입력하지 않아 causal multiplier는 중립값으로 동작했다. 동일 실행 결과를 feedback으로 재사용하지 않아 누수를 방지했다.
- `capacity_observed`는 아직 실제 체결/유동성 이벤트와 연결되지 않았다.
- funding/slippage는 현재 fold별 동적 관측값이 아닌 보수적 고정 비용 fallback이다.
- 다음 실행에서 fold별 registry backfill, dynamic funding/cost, prior-period feedback replay를 별도 검증해야 한다.
