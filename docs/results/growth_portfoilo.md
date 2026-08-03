
# Growth Portfolio CLI 실측 결과 (2026-08-04)

## 실행 조건

```text
명령: uv run python -m src.cli.main research run portfolio growth --no-log-run
종료 코드: 0
HOLDOUT_CUTOFF: 2025-12-31 23:59:59 UTC
평가 시작: 2023-04-01 00:00 UTC
universe_dates: 72
전체 심볼: 89
dev 심볼: 74
holdout 심볼: 15
```

## 최종 판정

```text
status: NO_ADMISSIBLE_ALPHA
selected_strategy: donchian_channel_position_v1
scorecard_reason: multiplicity
trades: 0
position: CASH
```

| 항목 | 실측값 | 판정 |
|---|---:|---|
| plateau score | 0.906 | 통과 후보이나 최종 탈락 |
| dev score | 2.270 | 후보 discovery 지표 |
| OOS t-stat | -0.908 | FAIL |
| multiplicity floor | 2.895 | `-0.908 < 2.895` |
| fold gate | 0 folds | FAIL |
| stress gate | FAIL | FAIL |
| symbol holdout score | -0.349 | FAIL |
| closed trades | 0 (최종 report) | CASH |

## fold 기간 부족의 실제 원인

원자료 전체 기간은 부족하지 않다.

```text
OHLCV parquet: 652 files
전체 원자료 기간: 2020-01-01 00:00 UTC ~ 2026-07-31 01:00 UTC
fold 입력 bar 수: 546
fold 입력 기간: 2024-01-01 00:00 UTC ~ 2024-03-31 20:00 UTC
실제 span: 90일 20시간
필요 조건: 6개월 fold 3개 이상
```

세그먼트 계측 결과:

```text
rolling segment calls: 3
causal_router calls: 3
causal_router reject: 2
segment sizing calls: 1
screen DATA_INVALID: 0
```

따라서 `equity span does not admit an equal-duration fold at 6MS`는 원자료가 90일만 존재해서가 아니라, 3개 rolling deployment 중 2개가 causal router에서 유효 sleeve를 얻지 못해 제거된 결과다. 최종 stitched OOS가 2024-01~03 한 구간만 남아 fold 계산이 fail-closed 되었다.

## 데이터 무결성 관찰

다음 12개 심볼에서 내부 1시간 gap이 탐지되어 해당 심볼은 평가에서 제외됐다.

```text
AERGOUSDT, AIAUSDT, BNXUSDT, CTKUSDT, CVCUSDT, CVXUSDT,
ICPUSDT, LITUSDT, MAVIAUSDT, PUMPUSDT, SLPUSDT, TLMUSDT
```

그러나 제외 후에도 dev 74개 심볼이 남아 universe 최소 크기 20을 충족한다. 현재 수집기는 캐시의 시작·끝 범위가 이미 요청 범위를 포함하면 내부 gap을 자동 복구하지 않으므로, 동일 기간 재수집만으로 이번 fold 실패가 해결되지는 않는다. 이번 실행에서는 원자료 기간 부족이 binding constraint가 아니므로 추가 수집을 수행하지 않았다.

## 결론

이번 실행의 병목은 데이터 종료일이 아니라 `rolling segment 유효성 부족`이다. HOLDOUT 정책상 2026년 데이터는 봉인 평가에 사용할 수 없으며, 데이터 추가보다 causal-router가 3개 이상의 deployment segment를 통과할 수 있는 독립적인 sleeve evidence 확보가 우선이다.
