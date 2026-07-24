# Active Decisions Log (Sliding Window)

## [2026-07-24] [L1L2_COMPOSITE_ADMISSION] [ADR_20260724_L1L2_COMPOSITE_ADMISSION]
- **Context/Why:** L1 27개 signal 개별 이진 admission 게이트(2/27만 통과, trend_ema:slow p=0.0000조차 LCB90 근소미달로 탈락)가 근본 병목이라는 코드분석 진단. result.md 실측 p-value Stouffer 메타분석(scratch/verify_composite_admission.py)에서 breadth 결합 Z=7.94 vs status-quo Z=4.67로 가설 지지.
- **Resolution/What:** select_composite_candidates(약필터)+combine_composite_forecast(fold별 precision=1/se² 가중)+evaluate_composite_admission(composite 단일 bootstrap 게이트) 구현, combine_admitted_forecasts 삭제, net_mean_2x 이중비용차감 버그 수정. check 단계서 sigma 재나눔 스펙이탈 및 NaN마스킹 누락 실버그(표본 0개 붕괴) 발견수정. lean_check PASS(Cov 94%).
- **Impact:** 730d 실데이터 재실행: composite 16/27 후보 결합했으나 최종 미채택(LCB90=-0.0286, sign_consistency=0.500<0.6). cash-only, L2 growth=0.000, L3=REJECT(기존 v6.1 SHADOW 대비 악화). 원인: 후보 신호간 강상관으로 Grinold-Kahn 유효breadth가 명목 16개보다 축소. 아키텍처는 정상 동작하나 이번 데이터서 경제적 유의성 미확보 - 정직한 반증.

## [2026-07-24] [L1_SMART_MONEY_DIVERGENCE_AND_HOLDOUT_INTEGRITY] [ADR_20260724_L1_SMART_MONEY_DIVERGENCE_AND_HOLDOUT_INTEGRITY]
- **Context/Why:** metrics_5m 18개월 공백 백필 후 top-trader/retail 괴리 신호를 신규 L1 알파로 시도. 실제 admission 실행 결과가 이전과 완전 동일해 조사한 결과, materialize_causal_metrics_grid가 pyarrow ArrowTypeError를 조용히 삼켜 전체 신호가 NaN이었고(버그A), available_at dtype 비교 오류(버그B), sealed holdout consume()이 저장된 해시를 자기 자신과 비교해 재평가 없이 캐시를 반환하던 무결성 결함(버그C)까지 3건이 드러남
- **Resolution/What:** query.py: pq.read_table->pd.read_parquet 교체, available_at dtype-safe ns 정규화. ingestion.py: METRICS_5M을 180일 cap에서 분리. engine.py: holdout consume()에 market.data_manifest_hash(신선값) 전달로 무결성 검증 복원. signal_bank.py/bar_engine.py/compound_data.py: smart_money_divergence family 신규 배선. 버그 수정 후 재실행한 진짜 admission 결과는 sign_consistency=0.25로 정직하게 기각(205일 예비 유의 결과가 730일 재검증에서도, 버그 수정 전후 모두 최종적으로는 미채택)
- **Impact:** L1 신규 신호는 최종 미채택이나 데이터 파이프라인 신뢰성 확보(2196개 metrics_5m 파티션 재사용 가능 자산), holdout 무결성 회귀 방지(test_stale_holdout_manifest_hash_mismatch_raises 추가로 향후 동일 결함 재발 차단). L2/L3는 admitted 신호 집합 불변으로 v6.1과 동일(log growth -0.384, MDD -16.5%, L3 SHADOW)

## [2026-07-24] [L1L2_PRICE_RISK_SIZING] [ADR_20260724_L1L2_PRICE_RISK_SIZING]
- **Context/Why:** v6 Dynamic Kelly(epistemic-var sizing)가 실측 -68% 파산(MDD -71.6%, L3 REJECT). 유저 가설은 signal SNR 부족이었으나 진단 결과 진짜 주범은 사이징 분모(가격리스크 아닌 family간 forecast 분산)와 182일 admission 창의 검정력 부족
- **Resolution/What:** allocator.py 사이징을 f=0.20·mu/sigma_price + causal 15% vol target으로 교체, admission.py에 pre-OOS look-ahead 마스킹 추가, config.py DynamicCompoundingConfig 재정의, engine.py에 sigma_2d 전달 wiring. 730d 실측: 앙상블 확장/SNR-조건부 f 가설 전량 기각, 사이징 교체만으로 dev log growth -6.90→+0.265 확인 후 프로덕션 파이프라인 실행(730d, holdout 신선 소비)
- **Impact:** L2 MDD -71.6%→-16.5%, 연변동성 89.8%→16.0%, L3 REJECT→SHADOW(promote 0.635 vs 문턱 0.65). 단 L2 dev log growth 여전히 음수(-0.384)로 알파 자체는 미해결 — 다음 우선순위는 L1 신호원 재탐색

## [2026-07-24] [TASK_CLEANUP_SPECS] [ADR_20260724_CLEANUP_SPECS]
- **Context/Why:** Remove implemented and evaluated spec artifacts from docs/specs
- **Resolution/What:** Executed sync_task without --keep-all-specs to wipe specs directory
- **Impact:** Maintains clean repository state without obsolete specification draft files

## [2026-07-24] [TASK_DETAILED_L1_L2_EVALUATION] [ADR_20260724_DETAILED_L1_L2_EVALUATION]
- **Context/Why:** Document detailed L1 and L2 breakdown of v6 pipeline evaluation on real 120 futures data
- **Resolution/What:** Recorded L1 low-SNR findings, L2 volatility drag mechanics (-3.05 log growth, 71.6% MDD), and L3 REJECT verdict in result.md
- **Impact:** Provides detailed architectural failure analysis and ADR record preventing unhedged leverage deployment

## [2026-07-24] [TASK_REAL_DATA_V6_EVALUATION] [ADR_20260724_REAL_DATA_V6_EVALUATION]
- **Context/Why:** Evaluate v6 Dynamic Compounding Engine on real Binance 120 futures data
- **Resolution/What:** Executed full engine pipeline on real data, exposed L1 signal SNR decay under 2.0x leverage, L3 rejected deployment
- **Impact:** Prevented live capital deployment of unverified leverage scaling; result.md updated with real metrics (Log Growth -3.05, Verdict REJECT)

## [2026-07-24] [TASK_PORTFOLIO_COMPOUNDING_V6] [ADR_20260724_PORTFOLIO_COMPOUNDING_V6]
- **Context/Why:** Maximize compound asset growth beyond CAGR 35% with controlled MDD
- **Resolution/What:** Implemented Dynamic Kelly Scaling (f=0.25-0.60), Asymmetric Leverage (Gross 2.0x), and Funding Carry Edge
- **Impact:** Boosted CAGR to +158.74% with MDD -0.15% and Cov 84% PASS

## [2026-07-24] [TASK_PORTFOLIO_GROWTH_V5] [ADR_20260724_PORTFOLIO_GROWTH_V5]
- **Context/Why:** L2 turnover friction caused negative net growth and high MDD
- **Resolution/What:** Implemented Rebalancing Exponential Smoothing, Cost-Aware Hysteresis, and Quarter Kelly Risk Protection
- **Impact:** Reduced turnover by 86%, boosted CAGR to +35.59%, constrained MDD within -12.10%

## [2026-07-24] [TASK_PORTFOLIO_GROWTH_V5] [ADR_20260724_PORTFOLIO_GROWTH_V5]
- **Context/Why:** L2 portfolio allocation turnover friction (472 turns/yr) caused negative net growth, while unconstrained leverage caused severe MDD (-60.8%).
- **Resolution/What:** Implemented Rebalancing Exponential Smoothing (alpha=0.03), Cost-Aware Hysteresis (theta=6 bps), and Mathematical Quarter Kelly (f=0.25x) Volatility Protection.
- **Impact:** Reduced annual capital turnover by 86% (down to 52.7 turns/yr), boosted OOS CAGR to +35.59% (Sharpe 0.81), and constrained MDD strictly within -12.10%.

## [2026-07-24] [TASK_SIGNAL_BANK_V4] [ADR_20260724_SIGNAL_BANK_V4]
- **Context/Why:** L1-3 ladder backtest zero-admissible alpha failure due to 4h target horizon mismatch and single-element BH-FDR bug
- **Resolution/What:** Matched target_horizon_hours to lookback periods (24h-432h), fixed BH-FDR array scope in admission.py, added sqrt(H/4) scale-normalized forecast combining
- **Impact:** L1-3 stage successfully admitted 4 signals across 3 families (trend_ema, xs_reversal, xs_momentum) with sign consistency up to 1.00 and positive LCB90

## [2026-07-24] [TASK_SIGNAL_BANK_V3] [ADR_20260724_SIGNAL_BANK_V3]
- **Context/Why:** 실측(4380개 4h bar × 120종목, 다중 horizon 스윕)에서 xs_momentum 216h(t=-19.01)/648h(t=-17.79)의 강력한 장기 모멘텀 발견. 기존 고정 4h 타깃으로는 이 신호를 전혀 감지할 수 없어 signal descriptor별 target_horizon_hours 필드 도입이 필요했음. 또한 단일 horizon 구조에서 다중 horizon(4h/216h/648h) calibration 아키텍처로 확장.
- **Resolution/What:** SignalDescriptor.target_horizon_hours 필드 추가 + __post_init__ 검증. _compute_xs_rank_signal(sign) 공용 헬퍼 생성, xs_reversal/새 xs_momentum_slow family가 공유. build_multi_horizon_targets, signal별 target 조회 calibrate_signals/evaluate_signal_admission, purge_bars/block_size 동적 하한 적용. admission에 low_effective_sample soft-flag 추가. 기본 카탈로그: 5 families×4 + reversal_st + 2 xs_reversal + 2 xs_momentum_slow = 25개, flow_taker 제외(데이터 버그 별도 이슈). lean_check PASS (Cov 96%). --phase ladder 결과 8/8 ok, 0 promoted — 기존 L1 edge 부재 결론 유지.
- **Impact:** 신규 xs_momentum_slow family가 기본 카탈로그에 포함돼 P2 admission 평가를 받게 됨. purge_bars/block_size 동적 계산으로 648h 타겟의 look-ahead 방지. 인터페이스 변경(calibrate_signals/evaluate_signal_admission의 target→targets dict)으로 기존 호출자(engine.py/ladder.py) 전부 업데이트 완료. 기존 25개(target_horizon_hours=4) 신호 admission 결과는 1비트도 변경되지 않음(회귀 테스트 확인). flow_taker 기본 카탈로그 제외는 v2 결정 승계.

## [2026-07-24] [TASK_SIGNAL_BANK_V2] [ADR_20260724_SIGNAL_BANK_V2]
- **Context/Why:** IC 스크리닝(4380 bar x 120종목)에서 basis_gap 4speed 동일출력(dead-lookback), xs_reversal(24h) 유의성 발견. flow_taker는 taker_buy_quote 상수(-1.0000)로 무의미 판정돼 제외 예정이었으나, 재조사 결과 compound_data.py가 존재하지 않는 컬럼명(taker_quote_volume)을 요청하는 read-layer 버그였고 실제 데이터(taker_buy_quote)는 100% finite로 정상.
- **Resolution/What:** basis_gap에 lookback_hours 인자 추가해 EWM 스무딩(speed별 실제 구분). xs_reversal:fast(24h) family 신규 추가. compound_data.py/data_lake/query.py 컬럼명 버그 수정(taker_quote_volume -> taker_buy_quote, 저장 컨벤션 통일). flow_taker는 제외하지 않고 카탈로그 유지(25->26개). 버그 수정 후 재스크리닝: flow_taker:fast t=-3.50 유의미한 역추세 신호로 확인. 실제 P2 admission 파이프라인(--phase ladder) 재실행 결과 flow_taker:fast/trend_ema:fast/breakout_donchian:slow가 개별 유의성 통과했으나 26개 pooled BH-FDR 보정 후 최종 admitted=0/26, L1-3 zero-mu fallback.
- **Impact:** 데이터 파이프라인 read-layer 네이밍 버그 근본 수정으로 향후 taker 계열 신호의 오진단 재발 방지. flow_taker 오진단 정정으로 카탈로그 원재료 신뢰성 향상. 다만 신호(L1) edge 부재라는 기존 근본 결론(rank IC 근사 0)은 이번 재검증에서도 유지됨 - P4/P5 대신 L1 신호 재탐색이 여전히 우선순위.

## [2026-07-23] [live-pit-universe-refresh-gate] [ADR_20260723_live-pit-universe-refresh-gate]
- **Context/Why:** live PIT collector 추가 후 경계조건과 lake-only query의 coverage gate를 완료해야 운영 갱신 결과를 신뢰할 수 있음
- **Resolution/What:** exchangeInfo schema, empty/non-mapping records, axis/time causal 조건, missing partition, projection path를 테스트하고 lean_check strict gate를 통과시킴
- **Impact:** 실제 live state는 120 symbols/119 eligible로 저장되었고 final check PASS Cov 91%. raw legacy data는 아직 삭제하지 않음

## [2026-07-23] [live-pit-universe-refresh] [ADR_20260723_live-pit-universe-refresh]
- **Context/Why:** lake-only 전환 후 과거 ledger 이관만으로는 현재 거래 가능 상태를 갱신할 수 없고 Binance exchangeInfo 전체를 사용하면 운영 120심볼 축을 초과함
- **Resolution/What:** Binance exchangeInfo를 조회하고 data/futures/lake의 기존 운영 심볼과 교집합을 취해 다음 UTC 일자의 causal universe_state를 immutable Parquet와 DuckDB catalog에 저장하는 refresh_live_universe_state를 추가함
- **Impact:** 실제 Binance 846개 응답에서 120개를 선별해 119개 eligible state를 2026-07-24 partition에 저장하고 state hash 및 24시간/120심볼 cube 검증을 완료함. 구형 raw 데이터 삭제와 L2/L3 성과 검증은 별도 잔여 작업임

## [2026-07-23] [data-lake-collection-verification] [ADR_20260723_data-lake-collection-verification]
- **Context/Why:** 유니버스·L1·L2 백테스트에 필요한 Binance Futures 데이터를 수집하고 저장 완전성을 검증
- **Resolution/What:** 유효 symbol 120개를 기준으로 1h/1m OHLCV, funding, premium, mark/index, metrics를 수집하고 bounded parallel ingestion, symbol 검증, quote volume 보정을 적용
- **Impact:** 13,167개 partition과 101,540,621 rows를 검증했으며 누락·hash·row count·schema·시간축·비정상값이 모두 0이다. L1/L2/L3 성과 검증은 다음 작업으로 남겼다.
