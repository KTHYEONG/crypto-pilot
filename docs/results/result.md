# 2026-07-25 분기 데이터 파이프라인 실행 결과

## 현재 문제

- 실행 기준일 `2026-07-25`, 확정 cutoff `2026-06-30`으로 `opt_main_futures.py --phase full`을 실행했다.
- `--sync local`과 `--sync auto` 모두 L1/L2/L3 성과 계산 전에 데이터 coverage gate에서 중단됐다.
- Klines 1H와 funding event는 coverage `1.0`이었으나, 필수 `cost_calibration` 데이터가 snapshot에 없어 coverage `0.0`이었다.
- 따라서 CAGR, MDD, Sharpe, CVaR, turnover 등 L2 성과 지표는 아직 산출되지 않았다. 이는 전략 성과 실패가 아니라 필수 거래비용 데이터 부재에 따른 fail-closed 결과다.

## 실행 중 발견·수정한 로직 문제

- `coverage_policy`가 유니버스 시간축과 요구 데이터 시간축의 길이가 다를 때 boolean index 오류를 내던 문제를 수정했다.
- `reconciliation`이 정상 `universe_state` parquet의 `effective_time_ns` 컬럼을 인식하지 못하고 손상 파일로 quarantine하던 문제를 수정했다.
- 관련 단위 테스트와 lean check는 통과했다(coverage 87%).

## 다음 확인 필요

1. `cost_calibration` 파티션의 수집 대상·저장 경로·manifest 등록 여부를 확인한다.
2. `--sync auto --date 2026-07-25` 재실행 후 CORE coverage가 모두 통과하는지 확인한다.
3. coverage 통과 뒤 유니버스 심볼 수와 L1 recipe 활성화 수를 기록한다.
4. L2에서 CAGR, MDD, Sharpe, CVaR95, annual turnover, cost drag를 산출하고 gate verdict와 함께 저장한다.
5. L3 holdout이 `2026-06-30` cutoff 이후 데이터를 사용하지 않는지 manifest/hash로 검증한다.

성과 지표가 위 단계를 통과하기 전까지는 백테스트 성과를 유효한 결과로 보고하지 않는다.
