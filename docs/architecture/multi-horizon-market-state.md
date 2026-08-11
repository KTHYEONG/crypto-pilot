# Multi-Horizon Market State (MHS)

## 목적과 범위

MHS는 여러 보유 horizon의 횡단면 신호를 독립적인 fast/slow 북으로 만들고,
이를 실제 계약수량 기반의 모의 실행 원장으로 검증하는 Phase 1 연구 파이프라인이다.
이 문서는 구현에 필요한 핵심 데이터 흐름과 판정 경계만 정의한다. MHS의 Research GO는
실거래 승인이나 레버리지·사이징 승인과 동일하지 않다.

## 전체 흐름

```text
1h OHLCV + funding + PIT lifecycle
        │
        ├─ trailing 720-bar 유동성 → liquid-half eligibility
        ├─ fast reversal / slow momentum 북 → 고정 50/50 capital blend
        │
        └─ decision 시각별 top-30 실행 roster
                │
                ├─ 5m OHLCV high/low/close → strict 또는 taker proxy fill
                ├─ historical mark cache → causal MTM/funding valuation
                └─ timestamped fill events → simulated inventory ledger
                                      │
                                      └─ Research GO gates/report
```

신호 계산은 1시간봉과 dev partition을 사용한다. 실행 리플레이만 5분봉을 기본으로
사용하며, 1분봉은 비용·체결 민감도 감사 모드다. 실행 roster는 신호 universe를
대체하지 않고, 실제 주문을 재생할 심볼만 줄인다.

## 1. 데이터와 point-in-time 규칙

- 신호 패널: `1h` OHLCV의 `open`, `close`, `quote_vol`.
- funding: 발행 시각 이후의 첫 평가 grid에 causal하게 정렬한다.
- lifecycle: 각 결정 시각에서 관측 가능한 심볼만 사용한다. 신규 상장은 full
  history 조건을 만족하기 전 제외한다.
- 유동성: 각 심볼의 trailing 720개 관측 quote volume을 계산하고, 유효 심볼의
  cross-sectional median 이상인 심볼을 liquid-half로 표시한다.
- 실행 roster: 같은 PIT 상태에서 quote volume 상위 30개를 선택한다. 실제 실행에
  사용된 심볼 union은 결과에 별도로 기록한다.
- 데이터 누락은 조용히 제거하지 않는다. 누락이 eligible 심볼의 보유 inventory나
  active order의 PnL/실행에 영향을 주면 해당 primary window를 무효화한다.

5분봉 실행 데이터는 다음 명령으로 manifest를 먼저 만들고, 검토 후 `--execute`로
수집한다.

```bash
PYTHONPATH=. uv run python -m src.cli.main data collect mhs-execution \
  --timeframe 5m --start 2021-01-01 --end 2025-12-31 --execute
```

수집 구현은 [mhs_execution_collection.py](/home/kth/crypto-pilot/src/application/data/mhs_execution_collection.py)에 있다.

## 2. 신호와 portfolio construction

fast reversal과 slow momentum은 각각 독립적으로 생성한다. 두 신호를 하나의 rank나
TrendScore로 pooling하지 않는다. Phase 1의 결합은 사전등록된 portfolio-level
allocation만 허용한다.

```text
w_blend = 0.5 * w_fast + 0.5 * w_slow
```

합산 후 gross를 재정규화하지 않는다. 서로 상쇄되는 부분은 현금으로 남으며, fast/slow
간 netting은 주문 생성 전에 실제 보유수량을 기준으로 수행한다. phase별 독립 원장 평균은
robustness 진단일 뿐, 실행 가능한 portfolio 수익률이 아니다.

## 3. Historical mark와 실행 리플레이

Mark price는 신호·랭킹·유동성·fill detection을 바꾸지 않는다. 오직 decision notional,
MTM, funding charge의 valuation에 사용한다.

`cache_required` 모드에서는 기존
`data/futures/markPriceKlines/<timeframe>/<symbol>.parquet`만 읽는다. 1시간 mark candle은
마감 후 다음 1시간부터 관측 가능하므로, replay grid에는 한 시간 지연 후 causal하게
forward-fill한다. 5분 grid의 허용 보간 한도는 11개 bar이며, cache gap을 넘어 보간하지
않는다. 유효한 mark가 필요한 시점에 없으면 fallback하지 않고 fail-closed한다.

실행 proxy는 다음 두 경계를 별도 계산한다.

- `OHLCV_IMMEDIATE_TAKER`: 즉시 taker 체결을 적용한다. Research GO의 primary다 --
  참여율(`participation_warnings`)이 분당 거래량의 1e-9 수준으로 무시 가능해
  footprint 회피용 passive 주문 대기의 경제적 근거가 없다
  (`docs/specs/mhs_realistic_execution_primary_swap.md` §0).
- `OHLCV_STRICT_PROXY`: limit intent가 이후 high/low를 관통할 때만 passive fill로 인정하고,
  timeout 시 taker fallback을 발생시킨다. 참고용 patient-reference 지표로만 보고되며
  Research GO를 더 이상 게이팅하지 않는다.

스트레스 bound는 동일한 immediate-taker 체결에 비용을 3배(`SPREAD_AND_COST_X3`,
`maker_fee_bps=6.0, taker_fee_bps=15.0, taker_slippage_bps=9.0`) 가정으로 적용한다.

OHLCV만으로는 partial fill, queue position, post-only rejection, cancel/replace latency,
order-size impact를 복원할 수 없다. 따라서 이 리플레이는 실제 체결 재현이 아니라
사전등록된 proxy 경계다.

## 4. Simulated inventory ledger

모든 PnL과 risk metric의 단일 원천은
[execution.py](/home/kth/crypto-pilot/src/mhs/execution.py)의
`simulated_inventory_ledger`다.

각 timestamp에서 다음 순서를 지킨다.

1. 직전 event 이후 보유 계약수량을 mark price로 MTM한다.
2. 해당 구간 funding을 실제 pre-event quantity와 mark notional에 부과한다.
3. fast/slow intent를 실제 inventory 기준으로 netting한다.
4. timestamp-sorted proxy fill과 fee를 적용해 계약수량·cash를 갱신한다.
5. equity와 fill turnover를 계산한다.

Target weight의 `abs(Δweight)`는 실제 turnover가 아니다. turnover는 각 fill의
`abs(quantity_delta * fill_price) / pre_trade_equity` 합으로 계산한다. equity가 0 이하가
되거나 pre-trade equity가 양수가 아니면 `DataIntegrityError`로 fail-closed한다. 이는
전략이 해당 데이터·비용 조건에서 자본을 보존하지 못했다는 hard failure다.

## 5. Research GO 판정

검증은 dev-only anchored fold에서 수행한다.

- 2021~2022 train → 2023 validation
- 2021~2023 train → 2024 validation
- 2021~2024 train → 2025 validation

각 fold에서 purge/embargo는 168시간이며 validation inventory는 flat으로 시작한다.
Research GO에는 다음이 모두 필요하다.

1. 4.18bp/base와 6.07bp/stress pre-screen 결과 보고
2. immediate-taker simulated-inventory aggregate의 daily autocorrelation-adjusted Sharpe ≥ 0.6
3. cost-stressed (SPREAD_AND_COST_X3) immediate-taker stress Sharpe > 0
4. 양의 primary에 대해 cap30% Sharpe와 net annual return 조건 충족
5. phase degeneracy, relevant missing data, termination, concentration, participation,
   synthetic stress 결과를 모두 보고하고 silent exclusion이 없어야 함

Research GO가 되더라도 Execution/Pilot/Scale GO는 별도다. 이 단계들은 forward
L1/L2/trade, own-order fill, size experiment가 필요하다. historical mark나 OHLCV proxy만으로
레버리지·fractional Kelly·배포 사이징을 결정하지 않는다.

## 6. 현재 구현 및 검증 상태

- 5분봉 실행 데이터: 255개 파일, 약 69.9M rows, 약 2.8GB.
- mark-cache wiring과 제한 구간 `cache_required` replay는 정상 동작한다.
- 전체 2021~2025 strict replay에서는 `pre-trade equity must be positive` invariant가
  깨져 Research GO가 기각됐다.
- 따라서 현재 상태는 **실행·원장·판정 코드 구현 완료, 전략 Research GO 실패**다.
- 최종 symbol/time holdout과 forward execution gate는 architecture freeze 이후에만
  개봉한다.

주요 실행 결과는
[mhs_horizon_diagnostic.json](/home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)에 기록된다.

실행 결과 저장은 요약과 감사 원장을 분리한다. JSON에는 지표·판정·artifact 경로만 남기고,
체결 이벤트와 equity/units/notional 시계열은
`docs/results/mhs_horizon_diagnostic_artifacts/` 아래 zstd 압축 Parquet으로 저장한다.
따라서 요약 파일을 빠르게 읽을 수 있고, 필요할 때만 `pandas.read_parquet()`로 상세 원장을
재현할 수 있다.

## 주요 코드 진입점

| 책임 | 코드 |
|---|---|
| MHS 평가 orchestration | `src/application/research/mhs/evaluation.py` |
| 실행 데이터 계획·수집 | `src/application/data/mhs_execution_collection.py` |
| proxy fill·inventory ledger | `src/mhs/execution.py` |
| mark cache loading/coverage | `src/market_data/services/futures_collection.py` |
| MHS CLI | `src/cli/commands/research/mhs.py` |
