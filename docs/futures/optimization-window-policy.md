# Futures Optimization Window Policy

본 문서는 futures 최적화/검증 운영에서 사용하는 표준 데이터 기간과 실행 cadence를 고정한다.

## 1. 표준 기간 (Default)

- `fetch warmup`: 12개월
- `IS (in-sample)`: 24개월
- `OOS (out-of-sample)`: 6개월

총 요청 범위는 `OOS 시작일 기준 36개월 이전(fetch_start)`부터 `OOS 종료일`까지다.

## 2. 운영 cadence

- `universe snapshot cadence`: quarterly (3개월) 유지
- `live 재최적화 cadence`: quarterly (3개월) 유지

`OOS=6개월`은 검증 강도 강화를 위한 horizon이며, live 재최적화 주기와 동일해야 할 필요는 없다.

## 3. Champion 승격 기준

- 동일 기간(`IS 24M + OOS 6M`)에서 평가
- `AWF` 및 `stability` 게이트를 통과한 경우에만 승격

## 4. 원칙 및 해석

- 빠른 탐색용 단기 윈도우(예: `IS 15M + OOS 3M`)는 연구/실험 용도로만 사용한다.
- 배포 후보 선정은 반드시 본 표준 기간 정책을 따른다.
- universe 변경 주기(quarterly)와 OOS 길이(6개월)는 서로 독립 변수로 관리한다.

## 5. 적용 대상

- `src/execution/opt_main_futures.py`
- `config/opt_config.py` 내 기간 산출 로직(`get_quarterly_window`)
- futures 최적화/검증 파이프라인 전반
