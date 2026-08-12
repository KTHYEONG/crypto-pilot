# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-12 (12차, 데이터 백필 후 재측정)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`, `eligible_symbols=445`, `run_elapsed_seconds=623.3`
- **CLI**: `research run portfolio mhs-horizon-diagnostic --fold-safe-horizon --discovery-gate`
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **직전 결과와 비교 불가**: 이 실행 직전 futures OHLCV/funding을 650개 심볼에 대해 2020-01-01까지 백필하고, 그 과정에서 드러난 소스 결손(34개 주요 심볼의 2022-02~04 구간 Binance Vision 아카이브 결손 등, 총 75개 심볼)을 정리했다. 유니버스 구성 자체가 바뀌었으므로 이 표의 숫자는 이전 차수(9~11차, 구 데이터 기준)와 직접 비교하지 말 것 — 과거 이력은 git log/task_index.json 참고.

## 1. Primary metrics

| metric | value |
| :--- | ---: |
| `primary_autocorr_sharpe` | 0.0915 |
| `primary_naive_sharpe` | 0.0209 |
| `primary_max_drawdown` | -0.4118 |
| `primary_net_ann` | 0.0013 |
| `primary_geometric_cagr` | -0.0006 |
| `primary_annualized_turnover` | 4.9607 |
| `stress_naive_sharpe` (x3 cost) | -0.1226 |
| `pre_vol_target_reference_naive_sharpe` | -0.0239 |
| `blend.failure` | null |

## 2. Research GO gate

| field | value |
| :--- | :--- |
| `eligible` | `false` |
| `evaluated_folds` | 3 |
| `folds_passed` | 1 |
| `reason_codes` | `CAPITAL_INVARIANT_BREACH`, `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` |

`CAPITAL_INVARIANT_BREACH`는 `fast_reversal` 북(blend 자본 0%, `PHASE_1_BOOK_BLEND_WEIGHTS`) 기인이며 primary 판정과 무관.

## 3. Fold detail

| fold | validation_start | validation_end | `primary_autocorr_sharpe` | `primary_max_drawdown` | `stress_naive_sharpe` | `failures` | `slow_horizon_hours` | `slow_horizon_source` |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: | :--- |
| 0 | 2023-01-08 | 2023-12-31 | -0.4112 | -0.2144 | -0.2822 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` | 168 | `frozen_default` |
| 1 | 2024-01-08 | 2024-12-31 | -0.3568 | -0.1765 | -0.3076 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` | 168 | `frozen_default` |
| 2 | 2025-01-08 | 2025-12-31 | 1.7649 | -0.3646 | 0.3668 | (none) | 168 | `frozen_default` |

3개 fold 전부 `slow_horizon_source=frozen_default` — 2021년 데이터 커버리지 개선(아래 §5) 이후에도 fold-safe-horizon 재선정이 여전히 no-op임을 재확인 (`admission_t=2.0` 미달, `ADR_20260811_MHS_HORIZON_SEARCH_EFFICIENCY` §2.1 결론과 일치).

## 4. Discovery/qualification gate (진단 전용, Research GO 미게이팅)

| family | `selected_horizon` | `admitted` | `discovery_aggregate_net_t` | `qualification_net_t` |
| :--- | :--- | :--- | ---: | ---: |
| reversal (sign=-1) | null | `false` | null | null |
| momentum (sign=+1) | null | `false` | null | null |

## 5. 이번 차수 데이터 변경 이력 (요약)

| 작업 | 대상 | 결과 |
| :--- | ---: | :--- |
| futures OHLCV 1h 백필 (`--start 2020-01-01`) | 650 심볼 | 성공 650/650, +98MB, 24.8분 |
| funding 백필 (`--start 2020-01-01`) | 650 심볼 | 성공 650/650, 25.2분 |
| 내부 결손(internal gap) 탐지 및 최신 연속구간 유지 정리 | 75 심볼 (OHLCV+funding 동일 경계로 정합) | 2021-2025 구간 내 결손 심볼 0건으로 확인 |

상세 근거(2021년 상장 심볼 6개→백필 후 121개, Binance Vision 2022-02-26~04-02 공유 결손 등)는 `ADR_20260811_MHS_2021_DATA_GAP_ROOT_CAUSE_CORRECTION` 및 `docs/decisions/task_index.json` 참고.

## 6. 관찰

- 유니버스를 정확한 상장 이력 기준으로 재구성한 결과, 이전(부분/부정확 데이터 기준) 실측보다 **Research GO 판정이 더 나빠짐** (`primary_autocorr_sharpe` 대폭 하락, MDD 확대, stress Sharpe 부호 전환) — 이전 결과가 불완전한 유니버스 위에서 낙관적으로 측정됐을 가능성을 시사.
- 2021년 데이터 품질 개선에도 fold-safe-horizon은 여전히 168h를 벗어나지 못함 — 병목은 데이터가 아니라 admission 통계/신호 자체(§4의 discovery 미채택과 일관).
- 아직 조사하지 않음: 이번 악화가 (a) 유니버스 확장 자체의 정직한 반영인지, (b) gap-trim(§5, 최신 연속구간 선택) 로직이 의도치 않게 다른 정보를 제거했는지 — 다음 스텝 후보.

## 7. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| §6 악화 원인 분리 검증 (유니버스 확장 vs gap-trim 부작용) | 미착수 |
| 2023/2024 방향성 손실을 고치는 신호 재설계 | 미탐색 (vol-normalization/funding-carry/multi-horizon ensemble 전부 admission floor 근처도 못 감, 구 데이터 기준) |
| fast_reversal의 독립적 `CAPITAL_INVARIANT_BREACH` | 미해결 (blend 자본 0%라 Research GO엔 무관) |
