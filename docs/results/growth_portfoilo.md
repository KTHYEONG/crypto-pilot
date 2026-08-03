## Growth Portfolio — 최신 다전략 라이브러리 실행 결과

가장 최신 갱신: **2026-08-03 20:11 — fold-concentration 게이트 추가**(§5). 이 절 이전
내용(§1~§4)은 2026-08-03 19:10 실행 기록으로 보존한다.

실행일: **2026-08-03 19:10:16 (Asia/Seoul 로그 시각)**
명령:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. \
uv run python -m src.cli.main research run portfolio growth --no-log-run
```

`--no-log-run`으로 실행하여 provenance 원장에는 기록하지 않았다. 평가 데이터는
봉인 종료시각 `2025-12-31 23:59:59 UTC`, 1h OHLCV를 완전한 4h bucket으로
리샘플한 결과다. 비용은 기존 `CostModel`(`fee_rate=0.0005`,
`slippage_rate=0.0003`)을 사용했다.

이번 실행은 `docs/specs/growth_engine_gate_diagnostics.md`에서 진단한 두 결함
(`align_funding_bars`의 인덱스 유니온 NaN 오탐, 전역 그리드 기준 causal-alignability
과잉 엄격)을 수정한 이후 첫 재실행이다. 아래 §2에서 그 효과를 직접 비교한다.

### 1. 평가 데이터와 PIT 유니버스

| 항목 | 값 |
|---|---:|
| 평가 시작일 | `2023-04-01 00:00:00 UTC` |
| 평가 종료 label | `2025-12-31 20:00:00 UTC` |
| 월별 리밸런스 수 | 33 |
| backfill 후보 고유 심볼 | 89 |
| 전체 평가 패널 | 89 symbols |
| dev 파티션 | 74 symbols |
| symbol holdout 파티션 | 15 symbols |
| 실제 매매 패널 (`symbol_scope=dev`) | 74 symbols |
| 전략 family 수 | 4 |
| 전체 가설 수 (`family_size`) | 12 |

1h 연속성 검증에서 다음 12개 심볼은 fail-closed 제외됐다:

```text
AERGOUSDT, AIAUSDT, BNXUSDT, CTKUSDT, CVCUSDT, CVXUSDT,
ICPUSDT, LITUSDT, MAVIAUSDT, PUMPUSDT, SLPUSDT, TLMUSDT
```

가격 보간이나 결측 수익률 0 대체는 하지 않았다.

### 2. 후보 탐색 결과 — `funding_contrarian_v1` 최초로 실전 스코어링됨

v1 registry는 다음 4개 전략군과 각 3개 window를 고정한다.

| 전략군 | Window (4h bars) | 데이터 |
|---|---:|---|
| `funding_contrarian_v1` | 42, 84, 168 | OHLCV + settled funding |
| `taker_imbalance_v1` | 42, 84, 168 | OHLCV taker-buy ratio |
| `vol_adjusted_trend_v1` | 42, 84, 180 | OHLCV |
| `donchian_channel_position_v1` | 42, 84, 168 | OHLCV |

이전 실행(2026-08-03 17:34)에서는 `funding_contrarian_v1`의 3개 윈도우 전부가
`status=DATA_INVALID`(`funding for 1000FLOKIUSDT must be finite`)로 즉시
탈락했다 — discovery 단계에서 단 한 번도 스코어링되지 않은 상태였다. 원인은
①`_build_settled_funding`이 심볼별 funding 이벤트를 `pd.DataFrame(dict)`로
합칠 때 서브밀리초 타임스탬프 불일치로 인덱스 유니온 NaN(컬럼당 최대
52.9%)이 발생해 finite 체크를 오탐시킨 버그, ②causal-alignability 체크가
전역 그리드 전체 커버리지를 요구해 49개 discovery 로스터 심볼 중 단 2개
때문에 가족 전체가 죽는 과잉 엄격함이었다(상세 원인·수정 근거는
`docs/specs/growth_engine_gate_diagnostics.md` §2 참고).

수정(`align_funding_bars`의 컬럼별 `dropna` + 심볼별 스케줄 구간 기준
causal-alignability 재정의) 이후 이번 실행에서 `funding_contrarian_v1`은
정상적으로 `SCREENED` 상태로 discovery 점수를 얻었다:

| 윈도우 | dev discovery Sharpe |
|---|---:|
| 42 | -0.748 |
| 84 | 0.372 |
| 168 | **0.609**(family 대표) |

버그가 가려온 전략을 처음으로 정직하게 측정한 결과 실질적인 edge가 없다는
것이 확인됐다 — `taker_imbalance_v1`(최고 2.246)에 크게 못 미쳐 family 선택
경쟁에서 애초에 승산이 없었다. "버그 때문에 놓친 좋은 전략"은 아니었다는
뜻이며, 이는 이번 수정이 gate를 느슨하게 만든 것이 아니라 측정을 정확하게
만들었을 뿐임을 보여주는 증거다.

Discovery에서 최종 finalist로 선택된 후보(이전 실행과 완전히 동일):

| 지표 | 값 |
|---|---:|
| selected strategy | `taker_imbalance_v1` |
| selected parameter | **84 bars (14일)** |
| plateau neighbor ratio | **0.785** |
| plateau 기준 | `>= 0.70` → PASS |
| dev qualification gross Sharpe | **1.338** |
| symbol holdout gross Sharpe | **-0.307** |
| qualification net OOS t-stat | **1.893** |
| multiplicity-adjusted t floor (`family_size=12`) | **2.895** |

holdout은 discovery 이후 dev finalist에 대해서만 조회했다. 다른 후보의
holdout 점수로 전략을 재선택하지 않았다.

### 2-1. 신규 계측: family 내 윈도우 간 상관계수 (측정 전용, 게이트 미반영)

`docs/specs/growth_engine_gate_diagnostics.md` §4에서 지적한 "`family_size=12`
Bonferroni 보정이 상관된 윈도우를 독립 가설처럼 취급할 수 있다"는 우려에 대한
첫 실측 근거다. 각 전략군의 qualification 구간 net-return 상관계수
(측정만 하며 falsification/promotion에는 반영되지 않음):

| 전략군 | 42-84 | 42-168 | 84-168 |
|---|---:|---:|---:|
| `funding_contrarian_v1` | 0.742 | 0.581 | 0.784 |
| `taker_imbalance_v1` | 0.774 | 0.637 | 0.763 |
| `vol_adjusted_trend_v1` | 0.665 | 0.410 | 0.671 |
| `donchian_channel_position_v1` | 0.792 | 0.624 | 0.803 |

4개 전략군 전부 윈도우 간 상관이 0.41~0.80으로 상당히 높다 — 3개 윈도우가
통계적으로 독립적인 시행이 아닐 가능성을 뒷받침한다. `family_size` 보정
단위(현재 12, family 단위로는 4)를 재판단할 근거 자료로 남기며, 이 스펙
사이클에서는 threshold를 바꾸지 않는다.

### 3. 최종 게이트와 포트폴리오

```text
falsification passed=False
binding=multiplicity
plateau=0.785
oos_t=1.893
floor=2.895
dev=1.338
holdout=-0.307
```

| 항목 | 결과 |
|---|---:|
| status | **`NO_ADMISSIBLE_ALPHA`** |
| binding constraint | **`multiplicity`** |
| 실제 적용 risk | 0 |
| 거래 수 | 0 |
| equity bars | 33 |
| 초기 자산 | 10,000 |
| 최종 자산 | **10,000 (flat CASH)** |
| promotion | 실행하지 않음 (`None`) |

Risk solver는 내부적으로 `0.005` 후보를 계산했지만 falsification 실패 이후
사용하지 않았다. 이번 실행의 falsification 수치는 §2에서 수정한 두 결함과
무관하게 이전 실행과 완전히 동일하다 — `funding_contrarian_v1`이 살아났지만
family 선택에서 `taker_imbalance_v1/84`를 이기지 못했기 때문이다.

### 4. 재현 결론

`docs/specs/growth_engine_gate_diagnostics.md`의 예측대로, 버그 수정은 최종
CASH 판정을 뒤집지 않았다 — 정직한 재측정이었지 gate를 완화한 것이 아니다.
`funding_contrarian_v1`을 처음으로 실전 평가했다는 점에서 미탐색 후보 공간을
줄였다는 의미는 있으나, 그 결과 자체는 음성이었다. 다음 실험 우선순위는
threshold 조정이 아니라 ①§2-1의 상관계수 실측치를 근거로 `family_size` 보정
단위를 재판단하는 것, ②신규 전략군을 설계하기 전에 이미 존재하던 4개 전략군을
모두 정당하게 소진했다는 사실을 확인한 상태에서 breadth 확장 여부를 판단하는
것이다. 현재 선택은 계속 CASH다.

### 5. fold-concentration 게이트 추가 (2026-08-03 20:11 재실행)

3라운드에 걸친 신규 전략 탐색(`docs/specs/growth_engine_new_signal_exploration.md`,
`growth_engine_classic_indicator_exploration.md`,
`growth_engine_broad_indicator_sweep.md`, 약 100개 파라미터 조합, plateau 통과 10개
전량 실패)과 그 원인 진단(`growth_engine_regime_split_structural_flaw.md`: discovery
breadth 48.5% vs qualification breadth 16.9%→4-fold 세분 시 20.2%→12.4%로 지속 악화 —
서로 다른 시장 레짐 비교)을 거쳐, 가장 견고하게 구성 가능한 해결책으로
**fold-concentration 게이트**를 구현했다(`docs/specs/growth_engine_fold_concentration_gate.md`).
`src/research/evaluation/reliability.py`에 이미 존재하던 `compute_equal_duration_fold_distribution`
(그동안 growth engine에서는 `n_folds=0`으로 미사용)를 qualification 순수익 스트림에
실전 배선해, `evaluate_falsification`의 게이트 순서를
`plateau → multiplicity → fold_concentration → symbol_holdout`로 확장했다.

재실행 결과:

```text
[EVAL] fold_gate n_folds=3 concentration=0.776 threshold=0.645 gate_pass=False
       median_fold_cagr=0.130 worst_fold_cagr=-0.000
[EVAL] falsification passed=False binding=multiplicity plateau=0.785 oos_t=1.893
       floor=2.895 fold_gate_pass=False dev=1.338 holdout=-0.307
status=NO_ADMISSIBLE_ALPHA selected_strategy=taker_imbalance_v1
```

`binding_constraint`는 여전히 `multiplicity`(오os_t=1.893이 fold_concentration 체크보다
먼저 걸림)라 최종 판정(`NO_ADMISSIBLE_ALPHA`/CASH)은 불변이다. 그러나 새 진단은
**현재 production 최선 후보(`taker_imbalance_v1/84`)조차 qualification 성과의
77.6%가 3개 fold 중 1개에 쏠려 있다**는, 기존 단일분할 `oos_t_stat`만으로는 전혀
드러나지 않던 취약성을 정량적으로 확인했다. 이 게이트는 어떤 후보도 새로 통과시키지
않으며(gate를 느슨하게 만드는 변경이 아님), 향후 어떤 후보가 multiplicity를 근접
통과할 때 "진짜 견고한 신호인지"를 자동으로 검증하는 위험관리 계측이다.
