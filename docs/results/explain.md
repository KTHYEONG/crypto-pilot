# Futures Compound 실행 흐름

이 문서는 `src/execution/opt_main_futures.py` 실행 시 실제 호출되는 모듈과 현재 검증 상태를 설명한다.

## 1. 전체 호출 구조

```text
opt_main_futures.py
  └─ cli()
      └─ run_multiscale_cli()
          ├─ argparse 설정 파싱
          ├─ legacy 옵션 차단
          ├─ build_compound_run_config()
          └─ run_multiscale_compound_main()
              ├─ build_data_lake_runtime()
              ├─ prepare_data_snapshot()
              ├─ build_daily_pit_universe()
              ├─ build_multiscale_market_cube()
              └─ run_multiscale_compound_engine()
                  ├─ build_multiscale_alpha_catalog()
                  ├─ run_l1_multiscale()
                  ├─ simulate_multiscale_portfolio()
                  ├─ evaluate_l2_walk_forward()
                  └─ evaluate_l3_sealed_holdout()
```

실행 명령 예시는 다음과 같다.

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
  --date 2026-07-23 --sync skip
```

현재 CLI는 compound 단일 파이프라인만 허용한다. `--trials`, `--phase`, `--timeframe`, `--symbols`, `--skip-universe` 등 legacy 옵션은 제거 대상으로 차단된다.

## 2. 진입점과 CLI

### `src/execution/opt_main_futures.py`

- 프로젝트 root를 import path에 추가한다.
- `src.application.futures.runner.cli.cli()`를 호출한다.
- 전략·백테스트 로직은 이 파일에 없다.

### `src/application/futures/runner/cli.py`

- 기준일, sync mode, universe refresh, network sync, seed를 파싱한다.
- `build_compound_run_config()`으로 불변 실행 설정을 만든다.
- `run_multiscale_compound_main()`의 `RunnerResult.exit_code`를 프로세스 종료 코드로 반환한다.

## 3. Data Lake와 Snapshot

호출 순서:

```text
build_data_lake_runtime()
  ├─ BinanceQueryClient
  └─ LocalDataCatalog(DuckDB)

prepare_data_snapshot()
  ├─ catalog.load_snapshot()
  ├─ build_ingestion_plan()
  ├─ complete → local snapshot 반환
  └─ incomplete + allow_network_sync → sync_futures_data_lake()
```

- canonical 저장소는 `data/futures/lake`다.
- Parquet partition과 DuckDB manifest를 함께 사용한다.
- partition 저장은 payload 검증, atomic write, hash 기록 순서다.
- Binance Vision 수집은 최대 4개 작업을 병렬 처리하며, 내부 요청 간격·retry/backoff로 요청률을 제한한다.
- 현재 데이터 수집·파일 무결성 검증은 완료됐다.
  - 유효 universe 120개
  - 13,167개 partition
  - 101,540,621 rows
  - hash/schema/row count/time ordering/비정상값 검증 PASS

## 4. Universe 호출

현재 `run_multiscale_compound_main()`은 다음 함수를 직접 호출한다.

```python
universe = build_daily_pit_universe(snapshot=snapshot, config=config.universe)
```

현재 구현의 실제 동작:

- snapshot에 존재하는 모든 partition의 symbol union을 만든다.
- symbol 수가 20개 미만이면 `EmptyPITUniverseError`를 발생시킨다.
- `decision_dates`에는 현재 날짜 하나만 기록한다.

주의: `resolve_compound_universe()`에는 historical 4h ledger와 daily PIT replay 로직이 있지만, 현재 main path의 `build_daily_pit_universe()`가 그 함수를 호출하지 않는다. 따라서 현재 main path의 universe는 완전한 일별 PIT replay라고 볼 수 없다.

## 5. MarketFeatureCube 호출

```python
market = build_multiscale_market_cube(
    snapshot=snapshot,
    universe=universe,
    config=config,
)
```

현재 구현:

- `history_days × 24`개의 UTC 1h execution bar를 만든다.
- `materialize_native_grid()`로 1h OHLCV와 `quote_volume`을 읽는다.
- `funding`은 현재 0 배열로 초기화한다.
- `eligible`, `entry_block`, `exit_required`, `capacity`, `execution_cost_bps`를 기본 배열로 만든다.
- `MarketFeatureCube.data_manifest_hash`에 snapshot hash를 연결한다.

중요한 제한:

- 수집된 1m OHLCV, premium, mark/index, metrics가 현재 cube에 모두 연결되어 있지 않다.
- mark, index, premium, open interest, taker flow 필드는 현재 main cube에 생성되지 않는다.
- 따라서 L1의 해당 conditional recipe가 실제 데이터로 평가되는지는 아직 확인되지 않았다.
- 1m 데이터가 저장되어 있어도 현재 실행 grid 자체는 1h다.

## 6. Engine 내부 호출

`run_multiscale_compound_engine()`의 의도된 순서는 다음과 같다.

```text
build_multiscale_alpha_catalog()
  → run_l1_multiscale()
  → simulate_multiscale_portfolio()
  → slice_execution_ledger()
  → evaluate_l2_walk_forward()
  → slice holdout ledger
  → evaluate_l3_sealed_holdout()
```

### L1: 신호 생성

현재 catalog에는 다음 12개 명시 recipe가 있다.

- trend: 4h, 12h, 1d
- residual momentum: 4h, 12h
- breakout: 4h, 12h
- carry funding event: funding event
- basis reversion: 1h
- taker flow: 15m
- flow/OI confirmation: 1h
- liquidity exhaustion: 15m

`run_l1_multiscale()`는 recipe별 causal forecast와 fold edge evidence를 계산하고, `AlphaEventTape`/forecast tape 형태로 다음 단계에 넘긴다.

하지만 현재 cube가 core 1h 데이터 중심이므로, 실제로 15m·funding·basis·OI recipe에 필요한 필드가 유효하게 연결되는지는 별도 실행 검증이 필요하다.

### L2: 시뮬레이션과 평가

`simulate_multiscale_portfolio()`는 active forecast를 받아 다음을 수행한다.

- forecast 결합
- causal covariance 계산
- growth-optimal weight 계산
- gross/net/per-symbol/capacity/cost 제약 적용
- bar return, funding, fee, slippage, impact 계산
- NAV와 execution ledger 누적

그 다음 `evaluate_l2_walk_forward()`가 ledger에서 다음 지표를 계산한다.

- annualized log growth
- bootstrap growth CI
- equity multiple
- max drawdown
- CVaR95
- annual volatility
- turnover
- integrity/safety 상태

현재 production path에는 Optuna trial이나 champion selection 호출은 없다.

### L3: 최근 구간 최종 점검

L3는 L2 fitting 구간과 분리된 holdout ledger를 대상으로 실행된다.

- holdout manifest의 시작·종료 timestamp 사용
- L2 prior return 일부를 사전 정보로 사용
- sealed holdout store의 consume 계약으로 재사용 방지
- `L3ValidationResult.verdict`와 reject reasons 생성

최종 runner는 L2 integrity failure를 exit code 1로 처리한다. L3가 reject여도 현재 구현은 결과 artifact를 기록한 뒤 `exit_code=0`으로 종료한다.

## 7. 현재 확인된 실행 blocker

데이터 검증 PASS와 전체 L1→L2→L3 실행 PASS는 서로 다른 결과다. 현재 코드 점검에서 다음을 먼저 해결해야 한다.

1. `compound_main.py`는 `run_multiscale_compound_engine()`에 `holdout_manifest`를 전달한다.
2. `engine.py`의 현재 시그니처는 `holdout_store`와 `holdout_id`를 요구한다.
3. 따라서 main path를 실제 실행하면 L1 이전에 인자 불일치가 발생할 가능성이 있다.
4. `build_multiscale_market_feature_cube()`가 수집한 보조 dataset을 실제 field로 연결하지 않는다.
5. 현재 `build_daily_pit_universe()`는 historical daily PIT replay가 아닌 snapshot symbol union이다.

## 8. 앞으로 확인할 순서

1. holdout manifest/store 인자 계약을 main과 engine 사이에서 일치시킨다.
2. `sync=skip`으로 main entrypoint smoke run을 실행해 L1 진입 여부를 확인한다.
3. MarketFeatureCube의 field별 shape·available ratio·timestamp alignment를 출력한다.
4. recipe별 L1 valid event 수, 결측률, causal fold 결과를 기록한다.
5. L2에서 실제 거래 수, cash 비중, weight, turnover, fee/funding/slippage를 확인한다.
6. L2 성장률·MDD·CVaR·integrity 결과를 평가한다.
7. L3 sealed holdout verdict와 reject reason을 확인한다.
8. 위 결과를 모두 확인한 뒤에만 L1/L2/L3 성과 보고서를 작성한다.

---

## 9. 데이터 삭제 대기 항목

현재는 lake 이관과 최신 PIT 상태 갱신만 완료되었으며, 아래 구형 데이터와 로그는 모두 보존 중이다. `data/futures/lake/`와 `logs/futures/compound/`는 현재 운영·검증 산출물이므로 삭제하지 않는다.

### 삭제 대상

| 분류 | 경로 | 상태 |
|---|---|---|
| 원본 OHLCV | `data/futures/ohlcv/` | 삭제 대기 |
| Funding | `data/futures/funding/` | 삭제 대기 |
| 보조 지표 | `data/futures/metrics/` | 삭제 대기 |
| 구형 메타데이터·ledger | `data/futures/metadata/` | 삭제 대기 |
| 구형 유니버스 산출물 | `logs/futures/universe/` | 삭제 대기 |
| 구형 최적화 산출물 | `logs/futures/optimization/` | 삭제 대기 |
| 구형 alpha 산출물 | `logs/futures/alpha_foundry/` | 삭제 대기 |
| 구형 진단 로그 | `logs/futures/diagnostics/` | 삭제 대기 |

### 삭제 전 필수 조건

- [ ] lake와 원본 간 심볼·기간·행 수·스키마 및 migration hash 비교 통과
- [ ] 최신 live PIT 상태와 lake snapshot의 완전성 검증 통과
- [ ] `src/execution/opt_main_futures.py --sync skip` 단일 실행이 raw 데이터 fallback 없이 통과
- [ ] L1/L2/L3 결과와 holdout 검증을 기록
- [ ] 최종 `rg` 실행 참조 감사에서 삭제 대상 경로의 미해결 inbound reference가 0건
- [ ] 명시적 경로만 대상으로 하는 `LegacyRetirementReport` 생성 및 삭제 승인

현재 실행 결과는 L1에서 `no_admissible_alpha`가 발생해 현금 상태로 종료되었으므로, L2/L3 성과 검증은 아직 삭제 전제조건으로 남아 있다.

### 삭제 원칙

- 삭제 완료 항목: 0개
- `data/futures/lake/`는 영구 보존
- broad root나 미확정 glob을 사용하지 않고, 검증된 절대 경로만 이관 후 영구 삭제
- 삭제 뒤 lake snapshot 재검증과 메인 파이프라인 smoke test를 다시 수행
