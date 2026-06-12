---
title: 데이터 수집기 상장일 자동 클리핑 및 무한 백필 루프 방지 (DataLoader Guard)
domain: futures/backtest
type: adr
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/backtest/data_loader.py
  - tests/unit/domain/futures/backtest/test_data_loader.py
last_verified: 2026-06-12
---

## [2026-06-12] 상장일 클리핑 및 24시간 미만 과거 데이터 백필 방어
- **Delta:** `data_loader.py`의 `ensure_ohlcv_data` 및 `ensure_1m_data` 내부의 과거 데이터 갭 수집 조건에 `_load_symbol_sync_profiles` 호출을 추가하여 `req_start`를 `onboard_date`로 자동 보정(클리핑). 추가로 보정 후 시간차(`cache_min_dt - effective_req_start`)가 24시간 미만인 경우 백필을 바이패스(스킵)하도록 방어벽 구축.
- **Rationale:** 신규 상장 코인의 실제 상장 시각(08:00 UTC 등)과 요청일 자정(00:00 UTC)의 수 시간 갭으로 인해 매 백테스팅 기동마다 바이낸스 API에 무한 갭 백필(1 candles 반환)을 재요청하며 API Weight와 리소스를 낭비하던 비효율 현상을 완전 차단.
- **Trade-off:** 상장일 당일의 극히 일부 시간대 갭은 수집에서 스킵되나, 백테스팅 및 전략 구동 정밀도에 미치는 영향은 무시 가능하며 안정성이 대폭 증대됨.
