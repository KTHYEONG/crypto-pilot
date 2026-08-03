# Growth Engine v2 — 첫 CLI 실행 결과

실행일: 2026-08-03 (Asia/Seoul)
실행 경로: `research run portfolio growth`
spec: `docs/specs/growth_engine_v2.md` (P1~P3 구현 완료분)
데이터: futures 1h → 4h 리샘플, sealed end `2025-12-31 23:59:59 UTC`
비용: `CostModel` 기본값(`fee_rate=0.0005`, `slippage_rate=0.0003`) 재사용
로그: `--no-log-run`으로 provenance 원장에는 기록하지 않음

## 1. 실행 명령

```bash
uv run python -m src.cli.main research run portfolio growth --no-log-run
```

옵션은 전부 기본값(`--universe-size 20 --max-positions 5 --rebalance-bars 1
--no-trade-band 0.0 --symbol-scope dev`)을 사용했다.

## 2. 구현 중 발견·수정한 결함 2건

CLI를 처음 돌렸을 때 크래시했고, 고친 뒤 재실행에서 추가로 2건을 더 발견해 함께 고쳤다.
세 번째 재실행에서 최종 결과가 확정됐다.

### 2.1 (크래시) 월별 로스터 폭과 전체 유니버스 폭 불일치

`_build_signal_weights`(`src/application/research/growth/evaluation.py`)가
`weights` 프레임은 전체 유니버스 컬럼(74~89종)으로 만들어 놓고, 그 달의
`roster`(부분집합, 예: 14~20종)로 계산한 결과를 컬럼 지정 없이
`weights.loc[bars] = w.to_numpy()`로 대입하려다 `ValueError: shape mismatch:
value array of shape (1,14) could not be broadcast to indexing result of shape
(1,74)`로 죽었다.

```diff
-        weights.loc[bars] = w.fillna(0.0).to_numpy()
+        weights.loc[bars, list(roster)] = w.fillna(0.0).to_numpy()
```

### 2.2 `symbol_scope="dev"`(기본값)에서 홀드아웃 검정이 항상 무의미

`run_growth_engine_evaluation`이 유니버스 스케줄을 `_apply_scope`로 **먼저**
dev 심볼만 남긴 뒤, 그 결과에서 dev/holdout 멤버를 다시 나누고 있었다. 이미
dev만 남은 집합에서 holdout을 추출하니 항상 공집합이었다
(`dev=74 holdout=0`). `evaluate_falsification`의 `symbol_holdout` 검정이
"홀드아웃에서 재현 안 됨"이 아니라 **애초에 측정된 적이 없는데도** FAIL로
잘못 보고될 수 있는 구조였다.

수정: dev/holdout 분리를 `symbol_scope` 필터링 **이전**(`full_schedule`,
2023-04-01 이후 전체)에서 계산하도록 순서를 바꿨다. `symbol_scope`는 이제
"실제 매매 대상"만 제한하고(`schedule = _apply_scope(full_schedule, ...)`),
falsification의 dev/holdout 점수는 항상 두 파티션을 모두 본다.

| | 수정 전 | 수정 후 |
|---|---:|---:|
| dev / holdout 멤버 수 | 74 / **0** | 74 / **15** |
| holdout_score | 0.000 | **0.1492** |

### 2.3 `derive_backfill_candidates`가 봉인 이전(2020~2023) 구간까지 포함

`derive_backfill_candidates(coverage, liquidity, rebalance_dates, ...)`가
`start`(2023-04-01)로 필터링되지 않은 **전체** `rebalance_dates`(2020-01부터
72개월)를 그대로 받고 있었다. `start` 이전 구간(적격 풀이 20개 미만이던
시기)까지 "실제 Top-20 경쟁을 뚫은 심볼"에 섞여 들어갈 위험이 있었다.

```diff
-    backfill_candidates = derive_backfill_candidates(
-        coverage, liquidity, rebalance_dates, request.universe,
-    )
+    dates_from_start = [date for date in rebalance_dates if date >= start]
+    backfill_candidates = derive_backfill_candidates(
+        coverage, liquidity, dates_from_start, request.universe,
+    )
```

수정 전후 모두 **89**로 결과값은 동일했다 — 우연이 아니라, 2023-04 이전에
존재했던 심볼이 BTC/ETH/LINK/ADA/ATOM/BNB/DOGE/DOT/AVAX 9종뿐이고 이들은
2023-04 이후에도 상시 Top-20에 들 만큼 유동적이어서 새로 추가되는 심볼이
없었기 때문이다. 즉 계산 범위 버그는 실재했지만 이 데이터셋에서는 최종
카운트에 영향을 주지 않았다. `docs/specs/growth_engine_v2.md` F1b가 별도
탐색 스크립트(`scratch/test_backfill_candidate_filter.py`, 다른 표본
구간·다른 bar-coverage 엄격도로 계산)로 추정한 **55**는 이번 실측치와
다르며, **89가 봉인 경계와 `min_bar_coverage≥0.99`를 정확히 지킨 신뢰할 수
있는 값**이다.

### 2.4 검증

세 항목 모두 수정 후 `ruff check` / `mypy` / 타깃 pytest(`test_growth_engine_cli.py`,
`tests/unit/research/universe`) 통과, 전체 `pytest` 회귀 없음.

## 3. 유니버스 구축 상세

| 항목 | 값 |
|---|---:|
| 로컬 수집 심볼(1h) | 652 |
| 커버리지 계산 성공(빈 파일·파싱 실패 제외) | 597 |
| 결측 갭으로 제외된 심볼 | 12개 (`AERGOUSDT`, `AIAUSDT`, `BNXUSDT`, `CTKUSDT`, `CVCUSDT`, `CVXUSDT`, `ICPUSDT`, `LITUSDT`, `MAVIAUSDT`, `PUMPUSDT`, `SLPUSDT`, `TLMUSDT` — `load_ohlcv_1h_as`의 1h 연속성 검증에서 fail-closed) |
| 데이터 시작(전 심볼 중 최이른) | 2020-01-01 |
| 데이터 끝(sealed) | 2025-12-31 23:59:59 UTC |
| 전체 캘린더 리밸런스 후보 | 72개월 |
| **`earliest_admissible_start`가 유도한 평가 시작일** | **2023-04-01**(하드코딩 아님 — 적격 풀이 `universe_size=20` 이상으로 유지되는 최초 시점) |
| 실제 평가 리밸런스 개월 수 | 33개월 |
| **`derive_backfill_candidates` — 실제 Top-20에 한 번이라도 뽑힌 고유 심볼** | **89종** |
| 그중 dev 파티션 | 74종 |
| 그중 holdout 파티션(SHA256 사전등록) | **15종** |

홀드아웃 15종: `1000SHIBUSDT`, `AUCTIONUSDT`, `AVAXUSDT`, `BNBUSDT`, `CAKEUSDT`,
`ETHUSDT`, `FLMUSDT`, `KAVAUSDT`, `MKRUSDT`, `NEARUSDT`, `PEOPLEUSDT`,
`SNXUSDT`, `SXPUSDT`, `UNFIUSDT`, `WAVESUSDT`.

첫 리밸런스(2023-04-01) 유니버스: `BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT,
BNBUSDT, LTCUSDT, FILUSDT, DOGEUSDT, FTMUSDT, DYDXUSDT, ADAUSDT, GALAUSDT,
LINKUSDT, ETCUSDT, SXPUSDT, AVAXUSDT, 1000SHIBUSDT, DOTUSDT, SANDUSDT,
SNXUSDT`.

마지막 리밸런스(2025-12-01) 유니버스: `ETHUSDT, BTCUSDT, SOLUSDT, ZECUSDT,
XRPUSDT, DOGEUSDT, BNBUSDT, FILUSDT, SUIUSDT, 1000PEPEUSDT, DASHUSDT,
AVAXUSDT, ADAUSDT, LTCUSDT, LINKUSDT, TNSRUSDT, UNIUSDT, ZENUSDT, TAOUSDT,
ENAUSDT`. 초기 유니버스 대비 절반가량이 교체됐다 — spec F1의 월별 회전율
실측(평균 25.3%)과 방향이 일치한다.

## 4. 신호 선택·평탄역 검정 상세

사전등록된 xs_momentum 가설군(`FAMILY_SIZE=9`, spec F3와 동일 계열) 중
5개 룩백을 dev 파티션(74종)에서만 스크린했다.

| 룩백 | gross Sharpe (dev) |
|---|---:|
| 1d (6봉) | **1.2767** ← 선택 |
| 3d (18봉) | 1.0281 |
| 7d (42봉) | 0.5146 |
| 14d (84봉) | 0.5159 |
| 30d (180봉) | 0.3115 |

`evaluate_parameter_plateau`는 선택값(1d)의 최근접 이웃 2개(3d=1.0281,
7d=0.5146)의 중앙값을 선택값 대비 비율로 본다:

```
neighbor_ratio = median(1.0281, 0.5146) / 1.2767 = 0.7714 / 1.2767 = 0.6042
```

기준 0.70 미달 → **`binding_constraint = "plateau"`로 FAIL**. spec F3에서
14d 룩백에 대해 관측한 것과 같은 패턴(톱니형 곡선, 인접값이 피크의
58~60%대)이 1d 룩백에서도 재현됐다 — 특정 lookback 하나만 우연히 좋게
나오는 현상이 이 신호군 전반의 구조적 특징임을 재확인한다.

## 5. dev/holdout·OOS 검정 상세

| 지표 | 값 |
|---|---:|
| dev_score (gross Sharpe, 74종) | 1.2767 |
| holdout_score (gross Sharpe, 15종) | **0.1492** |
| holdout_retention = holdout/dev | **0.1168** (기준 0.50 미달) |
| oos_t_stat (net 수익률 후반 절반, dev 거래 스트림 기준) | **−1.2769** |

`binding_constraint`는 검정 순서(`plateau` → `multiplicity` →
`symbol_holdout`)상 **plateau에서 이미 확정**되므로 holdout_retention·
multiplicity는 계산은 됐지만 최종 사유에는 등장하지 않는다. 다만
holdout_retention=0.117은 그 자체로 매우 낮아, plateau가 아니었더라도
symbol_holdout에서 확실히 FAIL났을 것이다 — 이중으로 반증된 셈이다.

## 6. 비용·회전율

| 지표 | 값 |
|---|---:|
| 평균 편도 회전율(봉당) | 0.323 |
| 연환산 비용 | 56.6%p |
| 연환산 gross | 8.63%p |
| 연환산 net | **−46.97%p** |

`rebalance_bars=1`(매 4h 리밸런싱) 기본값에서 비용이 gross의 6.6배다 —
spec F4에서 실측한 "회전율이 진짜 파괴자"가 이번 실데이터에서도 동일하게
나타난다. 다만 이번 실행은 P2(`net_construction`)가 구현되어 있어도 CLI
기본값이 `--rebalance-bars 1 --no-trade-band 0.0`이라 F4가 처방한 완화(3봉
리밸런싱, 무거래 밴드)를 아직 적용하지 않은 상태다. 어차피 plateau에서
FAIL이므로 이번 판정에는 영향이 없다.

## 7. 최종 판정

| 항목 | 값 |
|---|---:|
| status | **NO_ADMISSIBLE_ALPHA** |
| 사유(binding_constraint) | **plateau** |
| sizing | infeasible (falsification 실패로 `solve_growth_optimal_risk` 결과 자체가 무의미하므로 CASH 확정) |
| 거래 수 | 0 |
| 최종 자산 | 초기자본(10,000) 그대로, CASH 보유 |

```text
selected = CASH (no_admissible_alpha)
```

이 결과는 "전략이 수익을 내지 못했다"가 아니라, **파라미터 평탄역·홀드아웃
재현성 검정을 실데이터·실코드 경로로 처음 통과시켜 실행해본 결과 여전히
채택 기준을 만족하는 신호가 없다는 것을 확인**했다는 뜻이다. spec
`docs/specs/growth_engine_v2.md` F3(탐색 스크립트 기반 사전 반증)와
독립적으로, 이번엔 실제 프로덕션 코드 경로(P1 유니버스 → P2 net
construction → falsification → sizing)로 재확인됐다.

## 8. 다음 단계

1. `docs/specs/growth_engine_v2.md` F1b의 backfill 후보 수치(55)를 이번
   실측치(89)로 갱신 필요 — `/sync` 시 반영.
2. F4가 제안한 `rebalance_bars=3` + 무거래 밴드를 이번 파이프라인에도
   적용해 비용 절감 효과를 net_construction 계층에서 재검증할 수 있다
   (단, 현재 신호 자체가 plateau FAIL이므로 우선순위는 낮음).
3. 새 사전등록 가설이 필요하다 — xs_momentum 계열은 이번까지 포함해
   실데이터 검증을 마쳤고(다른 룩백에서도 동일 패턴 재현), 재제출 대상이
   아니다.
