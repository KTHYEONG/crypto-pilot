# Futures Optimization Window Policy

본 문서는 futures 최적화/검증 운영에서 사용하는 표준 데이터 기간과 실행 cadence를 고정한다.

## 1. 표준 기간 (Default)

- `fetch warmup`: 12개월 (IS 시작 전 지표 계산 전용, 훈련에 포함 안 됨)
- `IS (in-sample)`: 24개월
- `OOS (out-of-sample)`: 6개월 per fold

총 요청 범위는 `OOS 시작일 기준 36개월 이전(fetch_start)`부터 `OOS 종료일`까지다.

## 2. Walk-Forward 구조

- **유형**: Rolling Walk-Forward (고정 IS=24M, quarterly 3M step)
- **OOS 중첩**: 3M step + 6M OOS → 인접 fold 간 50% 중첩 수용.
  크립토 3M은 독립적인 시장 국면을 형성하기에 충분하며, 중첩 편향은 아래 champion 기준으로 흡수한다.
- `universe snapshot cadence`: quarterly (3개월)
- `live 재최적화 cadence`: quarterly (3개월)

## 3. IS↔OOS 경계 처리

- **Purge**: IS 말단 1 decision bar (4h) 버퍼만 적용. 4h TF에서 label horizon ≤ 80h이므로 IS 24M 대비 무시 가능.
- **Embargo**: 크립토 futures는 캘린더 이벤트 기반 leakage가 없으므로 별도 embargo 불요.

## 4. 실행 정밀도 계층

| 구간 | 실행 모드 | 대상 |
|---|---|---|
| IS 최적화 | Coarse OHLC (4h) | 전체 유니버스 (속도 우선) |
| OOS 최종 게이트 | Intrabar 1m | 데이터 보유 주요 심볼 (Top 30 기준) |

1m 데이터는 Binance Vision에서 심볼 상장일부터 수집 가능하다. 현재 11심볼·2023-10 이후 보유이며, 필요 시 확장 수집한다.

## 5. Champion 승격 기준

- 동일 Rolling WF 구조 (IS 24M + OOS 6M, quarterly step)에서 평가
- **≥ 70% fold pass** (예: 12 fold 기준 ≥8/12) — 중첩 50% 보정
- `AWF` 및 `stability` 게이트를 통과한 경우에만 승격

## 6. 원칙 및 해석

- 빠른 탐색용 단기 윈도우(예: `IS 15M + OOS 3M`)는 연구/실험 용도로만 사용한다.
- 배포 후보 선정은 반드시 본 표준 기간 정책을 따른다.
- universe 변경 주기(quarterly)와 OOS 길이(6개월)는 독립이며, 중첩 OOS는 의도적 설계다.

## 7. 적용 대상

- `src/execution/opt_main_futures.py`
- `config/opt_config.py` 내 기간 산출 로직(`get_quarterly_window`)
- futures 최적화/검증 파이프라인 전반
