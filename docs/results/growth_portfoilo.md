## Growth Portfolio — 최신 다전략 라이브러리 실행 결과

실행일: **2026-08-03 17:34:20 (Asia/Seoul 로그 시각)**
명령:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. \
uv run python -m src.cli.main research run portfolio growth --no-log-run
```

`--no-log-run`으로 실행하여 provenance 원장에는 기록하지 않았다. 평가 데이터는
봉인 종료시각 `2025-12-31 23:59:59 UTC`, 1h OHLCV를 완전한 4h bucket으로
리샘플한 결과다. 비용은 기존 `CostModel`(`fee_rate=0.0005`,
`slippage_rate=0.0003`)을 사용했다.

### 1. 평가 데이터와 PIT 유니버스

| 항목 | 값 |
|---|---:|
| 평가 시작일 | `2023-04-01 00:00:00 UTC` |
| 평가 종료 label | `2025-12-31 20:00:00 UTC` |
| 월별 리밸런스 수 | 33 |
| backfill 후보 고유 심볼 | 89 |
| 전체 평가 패널 | 89 symbols |
| dev 파티션 | 74 symbols |
| symbol holdout 파티션 | 15 symbols |
| 실제 매매 패널 (`symbol_scope=dev`) | 74 symbols |
| 전략 family 수 | 4 |
| 전체 가설 수 (`family_size`) | 12 |

1h 연속성 검증에서 다음 12개 심볼은 fail-closed 제외됐다:

```text
AERGOUSDT, AIAUSDT, BNXUSDT, CTKUSDT, CVCUSDT, CVXUSDT,
ICPUSDT, LITUSDT, MAVIAUSDT, PUMPUSDT, SLPUSDT, TLMUSDT
```

가격 보간이나 결측 수익률 0 대체는 하지 않았다.

### 2. 후보 탐색 결과

v1 registry는 다음 4개 전략군과 각 3개 window를 고정한다.

| 전략군 | Window (4h bars) | 데이터 |
|---|---:|---|
| `funding_contrarian_v1` | 42, 84, 168 | OHLCV + settled funding |
| `taker_imbalance_v1` | 42, 84, 168 | OHLCV taker-buy ratio |
| `vol_adjusted_trend_v1` | 42, 84, 180 | OHLCV |
| `donchian_channel_position_v1` | 42, 84, 168 | OHLCV |

Discovery에서 최종 finalist로 선택된 후보:

| 지표 | 값 |
|---|---:|
| selected strategy | `taker_imbalance_v1` |
| selected parameter | **84 bars (14일)** |
| plateau neighbor ratio | **0.785** |
| plateau 기준 | `>= 0.70` → PASS |
| dev qualification gross Sharpe | **1.338** |
| symbol holdout gross Sharpe | **-0.307** |
| qualification net OOS t-stat | **1.893** |
| multiplicity-adjusted t floor (`family_size=12`) | **2.895** |

holdout은 discovery 이후 dev finalist에 대해서만 조회했다. 다른 후보의
holdout 점수로 전략을 재선택하지 않았다.

### 3. 최종 게이트와 포트폴리오

```text
falsification passed=False
binding=multiplicity
plateau=0.785
oos_t=1.893
floor=2.895
dev=1.338
holdout=-0.307
```

| 항목 | 결과 |
|---|---:|
| status | **`NO_ADMISSIBLE_ALPHA`** |
| binding constraint | **`multiplicity`** |
| 실제 적용 risk | 0 |
| 거래 수 | 0 |
| equity bars | 33 |
| 초기 자산 | 10,000 |
| 최종 자산 | **10,000 (flat CASH)** |
| promotion | 실행하지 않음 (`None`) |

Risk solver는 내부적으로 `0.005` 후보를 계산했지만 falsification 실패 이후
사용하지 않았다. 따라서 이번 실행은 수익률을 기록한 운용 결과가 아니라,
`taker_imbalance_v1/84`가 plateau는 통과했으나 12개 가설 보정 OOS 기준과
holdout 재현성을 충족하지 못했다는 결과다.

### 4. 재현 결론

기존 단일 `xs_momentum`의 plateau 실패에서 벗어나, 이번에는 **평탄역을
통과한 finalist가 처음 확인**됐다. 그러나 `1.893 < 2.895`이고 holdout
Sharpe도 음수이므로 gate 완화나 실거래 전환 근거는 없다. 다음 실험은
threshold 조정이 아니라 독립 qualification 표본 확대와 동일 비용 스트레스
하 재검증으로 한정한다. 현재 선택은 계속 CASH다.
