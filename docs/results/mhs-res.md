# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-12 (15차, `crash_regime_tilt_alpha=0.2` 실전 검증)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`, `eligible_symbols=446`
- **CLI**: `research run portfolio mhs-horizon-diagnostic --crash-regime-tilt-alpha 0.2`
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **성격**: `ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY`(`src/mhs/regime.py`의 `crash_regime_tilt_weights` — BTCUSDT 추세 기반 방향성 틸트를 `slow_momentum` fold 리플레이에만 opt-in 배선)의 실전 동작 검증 실행. **`alpha=0.2`는 메커니즘이 작동하는지 확인하기 위한 임의 시험값이며 검증된 프로덕션 권장값이 아니다** — alpha를 성과 보고 고르는 것 자체가 p-hack이므로(`docs/decisions/task_index.json` ADR 참고) 이 문서의 숫자를 "개선됐다"는 근거로 프로덕션 기본값 결정에 쓰지 말 것.

## 1. Primary metrics (`slow_momentum` == `blend`, `fast_reversal` blend 자본 0%)

상위 books 경로는 이 오버레이가 의도적으로 미배선(fold 리플레이 경로만 배선) — `alpha` 값과 무관하게 항상 동일:

| metric | value |
| :--- | ---: |
| `primary_autocorr_sharpe` | 0.1819 |
| `primary_max_drawdown` | -0.3922 |
| `stress_naive_sharpe` (x3 cost) | -0.0844 |
| `blend.failure` | null |

## 2. Research GO gate

| field | value |
| :--- | :--- |
| `eligible` | `false` |
| `evaluated_folds` | 3 |
| `folds_passed` | 1 |
| `reason_codes` | `CAPITAL_INVARIANT_BREACH`(`fast_reversal` 별개 이슈, 2025-07-14 음의 자기자본, blend 자본 0%라 무관), `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `RELEVANT_EXECUTION_DATA_GAP`(`MISSING_DATA=76`, 정상 스킵), `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` |

## 3. Fold detail — `alpha=0.2` 배선 대상 경로 (baseline 대비 변화 관찰)

| fold | validation | `primary_autocorr_sharpe` | `primary_max_drawdown` | `stress_naive_sharpe` | `failures` |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 0 | 2023 | **+1.2092**(baseline -0.1461) | -0.1284(baseline -0.2009) | **+0.2367**(baseline -0.2093) | `RELEVANT_EXECUTION_DATA_GAP`만 (Sharpe/stress 게이트는 통과) |
| 1 | 2024 | -0.3486(baseline -0.5044) | -0.1859(baseline -0.1825) | -0.2079(baseline -0.3448) | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` |
| 2 | 2025 | +1.4986(baseline +1.6054) | -0.2678(baseline -0.4093) | +0.2023(baseline +0.2646) | (none) |

**관찰**: 오버레이 설계 시 사전등록 실험(2021-2022만 사용, `docs/decisions/task_index.json` ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY)에서 예측한 트레이드오프("alpha를 올릴수록 위기연도 개선·평시연도 소폭 악화")가 실제 fold 리플레이에서도 방향이 일치 — fold0(2023)이 손실→이익 전환(stress도 부호 전환), fold1(2024) 완화, fold2(2025)는 소폭 악화. `folds_passed`는 여전히 1/3(fold0가 Sharpe/stress는 통과했지만 무관한 `RELEVANT_EXECUTION_DATA_GAP`로 인해 "pass"로는 안 잡힘).

## 4. 최근 진단·조사 계보 (코드 변경 없이 원인 규명만 진행된 항목 포함)

- **`OHLCV_IMMEDIATE_TAKER` 결손 가드**(`ADR_20260812_MHS_IMMEDIATE_TAKER_FILL_GUARD`): 예전엔 이 진단 자체가 `CAPITAL_INVARIANT_BREACH`로 미완주였음 — 수정 후 안정적으로 완주.
- **fold Sharpe 편차 근본 원인**(`ADR_20260812_MHS_MOMENTUM_REGIME_DIAGNOSIS`): 2022(LUNA/FTX)가 horizon 무관 전원 음수 — horizon 선택 문제가 아니라 신호의 체계적 붕괴장 취약성. `DiscoveryQualificationResult.yearly_net_t` 진단 필드 신설.
- **전략 유형 자체 재검토**(`ADR_20260812_MHS_MOMENTUM_STRATEGY_REDESIGN_REVIEW`): cross-sectional(완전 시장중립) vs time-series(방향성 허용) 비교 — 완전 시장중립 목표와 붕괴장 생존이 트레이드오프임을 확인. `src/mhs/regime.py` 레짐 프록시 신설.
- **크래시 레짐 방향성 틸트 오버레이**(`ADR_20260812_MHS_CRASH_REGIME_TILT_OVERLAY`, 이번 15차): 위 프록시를 실제 fold 리플레이에 opt-in 연결, `alpha=0.2` 실전 검증 완료(§3).

## 5. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| `alpha` 프로덕션 값 결정 — fold train-only 데이터 기반 사전등록 절차로 별도 리서치 정책 결정 필요 | 미착수 (p-hack 방지를 위해 이번 문서 수치로 확정하지 말 것) |
| `fast_reversal`의 독립적 `CAPITAL_INVARIANT_BREACH`(음의 자기자본, 2025-07-14) 근본 원인 조사 | 미착수 (Research GO엔 무관) |
| 상위 books 진단 경로에도 오버레이 배선 여부 검토(현재 의도적으로 미배선) | 미착수 |
