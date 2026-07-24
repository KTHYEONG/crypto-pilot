# Active Decisions Log (Sliding Window)

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

## [2026-07-23] [TASK_MULTISCALE_DATA_LAKE_CUTOVER] [ADR_20260723_MULTISCALE_DATA_LAKE_CUTOVER]
- **Context/Why:** 최신 구현과 기존 explain 문서가 data snapshot, PIT universe, L1/L2 engine 경계를 일치시키지 못했고 실제 network 수집과 기존 데이터 삭제 시점도 분리되어야 했다.
- **Resolution/What:** 단일 multiscale engine 경로, runner-owned Binance/DuckDB runtime, 승인 기반 network sync, checksum/atomic Parquet snapshot, historical PIT union, sparse L1 event와 signed L2 allocation 계약을 최신 설명 문서에 동기화했다.
- **Impact:** check PASS(Cov 93%) 기준의 현재 로직을 문서화했다. 실제 Binance 다운로드와 data/futures 중복 데이터 삭제는 사용자 승인 전 수행하지 않는다.

## [2026-07-23] [TASK_COMPOUND_MAIN_REAL_DATA_ALIGNMENT] [ADR_20260723_COMPOUND_MAIN_REAL_DATA_ALIGNMENT]
- **Context/Why:** 실제 메인 실행에서 Binance timestamp 정밀도 불일치와 기준일 미전달로 결측·무결성 실패가 발생했고, explain 문서가 최신 compound-only 경로와 불일치했다.
- **Resolution/What:** OHLCV timestamp를 내부 ns로 정규화하고 reference_date를 데이터 로더와 PIT state에 동일하게 전달했다. compound-only 실행 흐름, 18개 recipe, L1 uncertainty, L2 단일 allocator, simulator, L3 결과 및 현재 fallback universe·zero-support 원인을 docs/results/explain.md에 기록했다.
- **Impact:** 메인 실행은 integrity 정상으로 완료되며, 현재 실측은 2종목 fallback과 robust uncertainty shrink로 target weight 0·L2 성장률 0·L3 SHADOW를 명확히 보고한다. check Cov 91% PASS.

## [2026-07-22] [TASK_CAUSAL_ALPHA_ONLINE_GROWTH_ENGINE] [ADR_20260722_CAUSAL_ALPHA_ONLINE_GROWTH_ENGINE]
- **Context/Why:** L2의 zero-overlap fit, direct Kelly-only OOS, legacy fallback 및 production Optuna 경로가 자산증식을 방해하는 구조적 결함을 최신 실측으로 재검증했다.
- **Resolution/What:** Causal 4-fold warm-up, 4-policy shadow posterior, cash abstention, zero leverage support를 active AWF 경로에 연결하고 최신 실행 결과 및 잔여 production migration blocker를 result.md에 기록했다.
- **Impact:** 정상 L2 경로는 fit_bars=0을 제거하고 성장 근거가 없을 때 현금 대기를 수행한다. 다만 crisis legacy replay, Optuna 120회, legacy promotion gate 및 12GB 초과 RSS는 후속 P0로 남는다.
