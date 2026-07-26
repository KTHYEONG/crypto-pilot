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
