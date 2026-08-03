# Growth Engine — CLI 실행 결과 (v2 첫 실행 + v3 재조정 주기 변경 후)

실행일: 2026-08-03 (Asia/Seoul)
실행 경로: `research run portfolio growth`
spec: `docs/specs/growth_engine_v2.md`(P1~P3) + `docs/specs/growth_engine_v3.md`(재조정 주기 기본값 변경)
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

## 8. 다음 단계 (v2 시점 기록)

1. `docs/specs/growth_engine_v2.md` F1b의 backfill 후보 수치(55)를 이번
   실측치(89)로 갱신 필요 — `/sync` 시 반영. **완료(§9 참조).**
2. F4가 제안한 `rebalance_bars=3` + 무거래 밴드를 이번 파이프라인에도
   적용해 비용 절감 효과를 net_construction 계층에서 재검증할 수 있다
   (단, 현재 신호 자체가 plateau FAIL이므로 우선순위는 낮음). **완료(§9
   참조) — spec `growth_engine_v3.md` §1로 CLI 기본값 변경.**
3. 새 사전등록 가설이 필요하다 — xs_momentum 계열은 이번까지 포함해
   실데이터 검증을 마쳤고(다른 룩백에서도 동일 패턴 재현), 재제출 대상이
   아니다. **spec `growth_engine_v3.md` §2에서 20개 신규 가설 스크린
   완료, 전부 미채택(§9 참조).**

---

## 9. v3 재실행 — `rebalance_bars` 기본값 1→3 적용 후

실행일: 2026-08-03. spec `docs/specs/growth_engine_v3.md` §1·§4.1 구현
후 동일 CLI를 옵션 없이(신규 기본값 `--rebalance-bars 3`) 재실행했다.

```bash
uv run python -m src.cli.main research run portfolio growth --no-log-run
```

### 9.1 판정은 불변, CASH 유지

| 항목 | v2(bar=1) | v3(bar=3) |
|---|---:|---:|
| status | NO_ADMISSIBLE_ALPHA | **NO_ADMISSIBLE_ALPHA**(동일) |
| binding_constraint | plateau | **plateau**(동일) |
| plateau ratio | 0.6042 | 0.6042(동일) |
| chosen lookback | 1d(6봉) | 1d(6봉, 동일) |
| oos_t_stat | −1.2769 | **−0.5104** |
| 거래 수 | 0 | 0 |

`plateau`·`multiplicity` 판정은 `dev_schedule`의 **gross**(비용·회전율
반영 전) Sharpe로만 계산되므로(`_gross_sharpe(_signal_pnl(...))`)
`rebalance_bars`·`no_trade_band`를 전혀 보지 않는다 — xs_momentum 1d가
인접 룩백 대비 평탄역 기준 미달인 것은 재조정 주기와 무관한 **신호
자체의 결함**이라 어떤 집행 방식으로도 이 게이트를 통과할 수 없다.

### 9.2 oos_t_stat 개선의 진짜 원인 — 비용 절감이 아니라 타이밍 개선

`rebalance_bars`가 net 비용을 줄여 개선됐을 것이라는 직관과 달리, 동일
`target_weights`에 대해 `compute_net_return_stream`을 bar=1/3/6으로 직접
비교한 결과는 다음과 같았다:

| rebalance_bars | 평균 회전율/봉 | 연환산 비용 | 연환산 gross | 연환산 net(gross−cost) |
|---:|---:|---:|---:|---:|
| 1 | 0.3231 | 56.60%p | 8.63%p | −47.97%p |
| **3(신규 기본값)** | **0.3231**(동일) | **56.60%p**(동일) | **33.97%p** | **−22.63%p** |
| 6(참고, 미채택) | 0.2249 | 39.40%p | 52.79%p | **+13.38%p** |

**bar=1→3 구간에서는 회전율·비용이 전혀 줄지 않았다** —
`realized_weights`를 봉 단위로 직접 비교하면 1,950개 봉에서 실현 비중이
서로 다르지만(재조정 타이밍이 다르므로 당연함), 33개월 전체에 걸친
누적 회전율 "이벤트" 총합이 우연히 같았다(1950.0=1950.0). 대신
**gross가 4배 가까이 개선됐다**(8.63%p→33.97%p) — 매 4h봉마다 신호를
재계산해 즉시 반영하는 대신 3봉(12h) 동안 이전 비중을 유지하는 것이,
노이즈에 휩쓸린 진입·청산("buy high sell low" 휘핑)을 줄여 **같은
거래 수로 더 유리한 타이밍에 체결**되게 한 것으로 해석된다. bar=6은
회전율도 줄고(0.225) gross는 더 오르면서(52.79%p) net이 처음으로
양전환한다(+13.38%p) — 다만 이 수치는 §1의 평탄역 근거로 채택하지 않은
지점이며, 신호 자체가 plateau FAIL이므로 어떤 net 수치도 admission에
영향을 주지 않는다는 점은 동일하다.

### 9.3 결론

- 재조정 주기 기본값 변경(§4.1, 코드 diff 1줄)은 의도대로 **집행 구조를
  개선**했고, 그 메커니즘은 사전에 가정했던 "비용 절감"이 아니라
  **타이밍 개선**이었다 — 재현 시 이 구분을 명확히 해야 한다.
- 신호 채택 여부는 여전히 **아니오**다. `growth_engine_v3.md` §2의
  20개 전략 스크린 결과(전부 미채택, H17만 유망 보류)와 함께 보면,
  "현재 채택 가능한 알파가 없다"는 결론이 재조정 주기 변경과 무관하게
  유지된다.
- 재현 근거: 직접 비교 실행 결과는 세션 로그에만 존재하며 파일로
  영속화하지 않았다(§9.2 표가 그 결과다). 필요시 `compute_net_return_stream`을
  동일 `target_weights`에 bar=1/3/6으로 각각 호출해 재현 가능하다.
