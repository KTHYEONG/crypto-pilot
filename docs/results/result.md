# 데이터 수집·저장 검증 결과

## 이번 작업에서 완료한 것

- Binance Futures 데이터 lake에 유니버스·신호·자산배분 백테스트용 데이터를 수집했다.
- 유효 Binance symbol 형식만 ingestion plan에 포함했다.
  - 유효 universe: 120개
  - 비정상 Unicode 후보 3개는 외부 데이터가 존재하지 않아 제외했다.
- 수집 dataset:
  - 1h OHLCV
  - 1m OHLCV
  - funding rate
  - premium index 5m
  - mark/index price 1h
  - daily metrics 기반 metrics 5m
- 누락된 `POLUSDT` partition을 추가 수집했다.
- 수집 속도를 개선했다.
  - 월별 Parquet만 읽어 반복적인 대용량 전체 스캔을 줄였다.
  - 최대 4개 partition 병렬 처리
  - Vision 요청 전역 rate limit과 retry/backoff 적용
- 기존 raw 데이터의 `quote_volume` 결측 566개 partition을 `close × volume`으로 보정했다.

## 최종 데이터 상태

| 항목 | 결과 |
|---|---:|
| 저장 partition | 13,167개 |
| 저장 용량 | 약 4.20GB |
| 검증 row | 101,540,621개 |
| Snapshot | `local-1784732400000-36682fb336d1` |
| Manifest hash | `36682fb336d1fac55f64e235890c28ad8a4719890e4ff2d04890a1ef7a8ec0f6` |

## 데이터 무결성 검증

- 누락 파일: 0
- catalog hash 불일치: 0
- catalog row count 불일치: 0
- Parquet schema 오류: 0
- timestamp 역순 또는 중복: 0
- 가격·거래량 비정상값: 0
- 유효 universe의 dataset별 symbol 누락: 0
- `sync=skip` 로컬 snapshot 검증: PASS
- 코드 계약 회귀 검증: PASS, coverage 68%

## 아직 확인하지 않은 것

이번 작업은 데이터 수집과 저장 검증에 한정했다. 다음 결과는 아직 확인하지 않았다.

- L1 신호가 실제 데이터에서 정상 생성되는지
- L1 signal 수, 결측률, signal별 edge와 국면별 안정성
- L2가 L1 신호를 실제 자산배분으로 변환하는지
- L2 정책별 weight, turnover, leverage, cash 비중
- L2 백테스트 수익률, CAGR, Sharpe, MDD, CVaR, 비용·funding 반영 결과
- L3 최근 데이터 최종 점검과 production 차단/통과 판단
- 전체 L1→L2→L3 메인 파이프라인 연결 결과

## 다음에 확인할 순서

1. `src/execution/opt_main_futures.py`에서 `sync=skip`으로 실제 snapshot을 사용한다.
2. L1을 실행해 symbol×timeframe별 signal 생성 수와 결측·look-ahead 여부를 검증한다.
3. 동일 snapshot으로 L2 자산배분 백테스트를 실행하고 비용·funding·turnover를 포함해 평가한다.
4. L3 최근 구간 점검을 실행해 과최적화·데이터 부족·위험 한도를 확인한다.
5. 세 단계 결과가 모두 통과한 경우에만 최신 결과를 별도 백테스트 보고서로 기록한다.
