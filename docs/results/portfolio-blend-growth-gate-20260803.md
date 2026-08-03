# Portfolio Blend Growth Gate 실행 결과

실행일: 2026-08-03 (Asia/Seoul)  
실행 경로: `research run portfolio blend`  
데이터: futures 4h OHLCV + funding, sealed end `2025-12-31 23:59:59 UTC`  
비용: 기본 `fee_rate=0.0005`, `slippage_rate=0.0003`  
로그: `--no-log-run`으로 provenance 원장에는 기록하지 않음

## 1. 실행 명령

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.cli.main \
  research run portfolio blend \
  --candidate-kind core5_causal_tournament_v1 \
  --tournament-profile pbgt_discovery_v1 \
  --no-log-run
```

프로파일은 다음과 같이 source-controlled 되어 있다.

- universe: `core5_v1`
- symbols: `BTCUSDT, ETHUSDT, AVAXUSDT, BNBUSDT, DOGEUSDT`
- discovery end: `2024-12-31 23:59:59 UTC`
- qualification interval: `365D`
- stress: fee `1.5x`, slippage `2.0x`, decision delay `+1 bar`

## 2. Tournament 결과

| return source | admitted | discovery 판정 | 이유 |
|---|---:|---|---|
| `donchian_long_only_v1` | 아니오 | observation FAIL, fold FAIL, stress PASS | discovery 구간에서 LCB90/분산 gate 미달 |
| `funding_signed_directional_v1` | 아니오 | feasibility FAIL | `t_stat` 제약 |
| `technical_supertrend_long_v1` | 아니오 | feasibility FAIL | `t_stat` 제약 |
| `technical_parabolic_sar_long_v1` | 아니오 | feasibility FAIL | `mdd_floor` 제약 |
| `technical_keltner_channel_breakout_long_v1` | 아니오 | feasibility FAIL | `mdd_floor` 제약 |

선택된 return source는 빈 tuple이다.

```text
selected = ()
```

따라서 qualification 구간에서 후보를 재선택하거나 gate를 완화하지 않고,
전체 결과를 CASH로 반환했다.

| 항목 | 결과 |
|---|---:|
| base final equity | 10,000.00 |
| stress final equity | 10,000.00 |
| base trades | 0 |
| stress trades | 0 |
| leverage schedule non-zero bars | 0 |
| schedule max leverage | 0.0x |
| schedule hash | `6c33e67d20b2c128391e68b0abf6c1451c5fd6ad15243c65a543d36067ba3e32` |
| observation | `PENDING` |
| fold | `PASS` (빈 ledger의 fail-closed 기본값) |
| stress | `PENDING` |
| promotion | `REJECTED` |

이 결과는 전략 신호가 정상적으로 수익을 냈다는 의미가 아니라, 현재 discovery
증거로는 배포 가능한 전략을 확인하지 못했다는 의미다. 전략이 없을 때 현금으로
남는 것이 구현된 fail-closed 계약이다.

## 3. 기존 고정 Donchian control

동일한 `core5_v1`로 기존 control도 별도 실행했다.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.cli.main \
  research run portfolio blend \
  --candidate-kind fixed_long_only_v1 \
  --universe-id core5_v1 \
  --no-log-run
```

| metric | base | stress |
|---|---:|---:|
| CAGR | 31.15% | 27.73% |
| MDD | -20.24% | -20.46% |
| Sharpe | 1.380 | 1.311 |
| LCB90 CAGR | 15.12% | 13.05% |
| trades | 718 | 702 |
| gate | observation PASS | stress PASS |
| fold | FAIL (`0.4899 > 0.4413`) | — |
| promotion | `REJECTED` | — |

고정 control은 observation과 stress는 통과하지만 fold concentration에서 계속
막힌다. 따라서 새 구현은 이 결과를 억지로 재현하거나 leverage를 키우지 않고,
discovery에서 독립 증거가 없는 경우 CASH로 종료한다.

## 4. 적용된 위험·선정 불변식

1. 운영 CLI는 임의 `--symbols`를 받지 않고 `core5_v1`만 허용한다.
2. leverage schedule은 완성된 과거 marked return만 사용하며, discovery 이후
   qualification 결과가 selection/weight/leverage를 바꾸지 않는다.
3. base와 stress는 동일한 universe, selected source, weight, schedule hash를
   사용한다. stress는 비용과 실행 지연만 변경한다.
4. feasibility 또는 discovery gate 실패 후보는 조용히 제거하지 않고 reason을
   `TournamentCandidateEvidence`에 남긴다.
5. observation/fold/stress 임계값과 `compose_promotion_verdict`는 변경하지
   않았다.

## 5. 검증 결과

- 계약 감사: `lean_check --spec-only` PASS
- Ruff: PASS
- Mypy: PASS (13 source files)
- 대상 테스트: 80 passed
- 전체 pytest: exit code 0

## 결론

이번 실행에서 spec 적용 결과는 **gate PASS 전략 발견이 아니라, 무근거 전략의
자동 배포를 차단하고 CASH로 안전하게 멈추는 것**이다. 기존 control은
`observation/stress PASS + fold FAIL` 상태로 유지되며, 실제 자산증식 후보를
만들려면 2025-12-31 이후의 새 데이터 또는 독립적인 return source 증거가
필요하다. 현재 데이터만으로 qualification 통과를 주장할 수 없다.
