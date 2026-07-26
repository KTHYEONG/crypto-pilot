## Funding 복구 후 L2 재실행 — 2026-07-26

- funding 검증: **2292개 파티션 / 299021개 이벤트 / invalid 0개**
- 실행: `src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 소요시간: **3분 54.52초**
- 최대 RSS: **996.9 MiB** (1,020,824 KiB)
- Swap/OOM/timeout: **없음**
- 데이터: **120 symbols × 4,380 bars**, integrity `true`

### L2/L3 성과

- L2 verdict: **`FAIL`** (유효한 finite 지표 산출 완료)
- annualized log growth: **0.0581**
- CAGR: **5.98%**
- equity multiple: **0.9686**
- Sharpe: **0.2375**
- max drawdown: **18.01%**
- annual volatility: **14.55%**
- annual turnover: **108.68x**
- cost drag ratio: **23.83%**
- L2 실패 이유: excess growth LCB/확률, outer-fold 양성 수, deflated Sharpe/Sharpe 확률 gate 미충족
- L3 verdict: **`REJECT`** (`l2_not_pass`)

Funding 오염으로 인한 `net_return_le_minus_one`은 제거되었고, 이번 L2 결과는 실제 전략 성과를 평가할 수 있는 정상 finite 결과다. 다만 통계적 유의성과 비용 반영 후 성장 gate를 통과하지 못했으므로 배포하지 않는다.

### 무결성 복구

- 원천 샘플 `[timestamp, 4, -0.00033019]`에서 마지막 열만 rate로 저장하도록 정규화
- 기존 오염값 `4`·`8`을 소수 rate로 변환하지 않고 원천 월 파티션 재수집
- `funding-v3` 검증, LOCAL read-only fail-closed, AUTO 대상 파티션 quarantine/재수집 적용

결과 파일: `logs/futures/compound/20260726_035441/result.json`

## L2 실행 결과 — 2026-07-26

- 실행: `src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 소요시간: **3분 40.35초**
- 최대 RSS: **978.7 MiB** (약 0.98 GB)
- Swap/OOM/timeout: **없음**
- 회귀 테스트: **10 passed**

### 성능 변화

| 항목 | 기존 기준 | 개선 후 |
|---|---:|---:|
| 소요시간 | 18분 09.76초 | 3분 40.35초 |
| 변화 | - | 약 79.8% 단축 |
| 최대 RSS | 약 972.6 MiB | 약 978.7 MiB |

### 판정

- L2 산출물 생성: **완료**
- L2 verdict: **`NO_EVIDENCE`**
- L2 이유: `net_return_le_minus_one`
- L3 verdict: **`REJECT`**
- L3 이유: `max_drawdown_exceeded`, `l2_not_pass`

비정상 수익률이 감지되어 유한값 기반 fail-closed 판정이 내려졌다. 따라서 이번 실행의 0 지표는 투자 성과가 아니며, 유효한 Sharpe·성장확률 성과는 보고하지 않는다.

### 원인 및 후속 조치

로컬 funding parquet에 funding rate로 볼 수 없는 값(`4`, `8` 등)이 남아 일부 구간의 손실이 -100%를 초과했다. funding 캐시/sidecar를 무효화하고 재동기화한 뒤 동일 날짜로 재실행해야 한다.

결과 파일: `logs/futures/compound/20260726_030534/result.json`
