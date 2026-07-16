# L1→L2 최신 Replay 결과 — 2026-07-16 (master TF handoff 수정 후)

## 실행 조건과 데이터 무결성

- 실행: `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-16 --seed 42`
- 완료 분기 cutoff: `2026-06-30`; horizon: `2023-10-31 ~ 2026-06-30`; IS/OOS split: `2026-01-01`.
- Universe: Pool 414 → Selected 150 → Loaded 125; L1 admission: 114/125 (`late_start` 11 제외).
- 실행 정책: L2도 L0 gate를 실행해 native TF labeled events를 생성·전달한다. `missing native labeled events for tf=1h` 오류는 재발하지 않았다.
- `docs/specs/l1-l2-master-tf-handoff-wiring.md` 구현 반영 후 재실행. 이전 run(동일 cutoff/seed)의 `TieredPipelineError: no deployable timeframe found for L2 master TF` fail-closed는 **재발하지 않았다** — L2가 최종 시뮬레이션까지 완주했다.

## L1 결과 (수정 전과 수치 완전 동일 — 회귀 없음)

| Timeframe | Fold readiness | Symbol breadth | Probe LCB (bps) | Final promoted pairs | 판정 |
| :--- | :---: | ---: | ---: | ---: | :---: |
| 1h | 4/4 | 68.554 | +68.850 | 237 | PASS |
| 2h | 4/4 | 86.185 | +106.455 | 164 | PASS |
| 4h | 3/4 | 44.579 | +37.238 | 59 | PASS |
| 6h | 4/4 | 29.755 | +49.121 | 14 | PASS |
| 8h | 4/4 | 105.662 | +80.662 | 186 | PASS |
| 12h | 4/4 | 106.333 | +73.830 | 98 | PASS |
| 1d | 4/4 | 95.006 | +83.277 | 3 | PASS |

- 4h fold #2는 gross edge `-54.21 bps`로 block됐지만 나머지 3개 fold가 통과해 aggregate L1 gate는 PASS다.
- 1d의 승급은 `STGUSDT / btc_regime_pullback`, `ENSUSDT / btc_regime_pullback`, `GALAUSDT / trend_donchian` 3개다.
- L0가 TF별 native artifact를 생성했다. 1h/2h/6h/8h/12h/1d native panel injection은 각각 4/7/10/11/11/4개.

## L2 결과 — 최초로 산출됨

**Master TF 선정**: `assess_l1_tf_handoff`(breadth + family diversity + finite positive edge) 기준으로 `8h`가 master로 선정됐다(Symbol-Breadth 105.662, 186 promoted pairs — 전체 TF 중 최대 breadth). `_resolve_l2_master_tf_from_prior`가 empty-dict 재계산 없이 Step-A에서 이미 계산된 `selected_timeframe`을 재사용했다.

```text
● [CHAMPION STORE] 신규 챔피언 갱신 (tf=8h, growth_lcb=0.2875)
```

**L2 포트폴리오 스코어카드** (평가 구간 2025-03-20 ~ 2025-12-30):

| 항목 | 값 | 게이트 | 판정 |
| :--- | ---: | :--- | :---: |
| CAGR | +61.2% | ≥30.0% | ✅ |
| Sharpe / Sortino / Calmar | 2.026 / 3.875 / 3.082 | ≥1.0 / ≥1.5 / ≥1.0 | ✅ |
| MDD / CVaR95 | 19.9% / 1.4% | ≤30% / ≤6% | ✅ |
| Fold pass / Trades / Friction | 100.0% / 1115 / 99.4% | ≥60% / ≥30 | ✅ |
| Sharpe Uplift | +0.10 | ≥0.20 | ❌ |
| DSR / PSR | 0.949 / 0.999 | ≥0.60 (diag) | ✅ |

- Leverage(L*) 1.4804 (binding: champion), Relative MDD 1.69x, Turnover 0.113.
- Fold별: #1 Sharpe 4.284/CAGR +134.3%, #2 Sharpe 0.267/CAGR +3.8%, #3 Sharpe 2.172/CAGR +44.7%, #4 Sharpe 2.594/CAGR +91.6% — 전 fold PASS.
- **⚠️ NO-CRISIS-WINDOW**: 이 평가 윈도우는 병목-caliber fold(MDD≥15% & CAGR≤0)를 포함하지 않는다. **위 CAGR/Sharpe 수치를 production 승급 근거로 인용 금지.**
- `[REGIME-L2] proof_failed path=pooled_fallback` — regime 증명 실패로 pooled fallback 사용 중(별개 이슈, 이번 수정 범위 밖, 후속 조사 필요).

## Verdict

- **L0→L1→native TF handoff:** PASS (회귀 없음).
- **전체 TF L1 robustness gate:** PASS (회귀 없음).
- **L2 master selection:** PASS — `assess_l1_tf_handoff` 기반 선정 정상 동작 확인(narrow-breadth TF 배제 로직 실측 검증은 아직 안 됨, 이번 run은 8h가 자연스럽게 최대 breadth였음).
- **L2 allocation 및 스코어카드:** PASS(Uplift 게이트 1개 제외) — **단, NO-CRISIS-WINDOW 캐비아트로 production promotion 근거로 사용 금지.**

## 다음 조치

1. ~~`_resolve_l2_master_tf()`에 `assess_l1_tf_handoff()`를 연결~~ — 완료 (`docs/specs/l1-l2-master-tf-handoff-wiring.md`, 구현·check PASS·실측 replay 확인).
2. 위기장(crisis-caliber fold)을 포함하는 holdout 윈도우로 별도 replay해 Uplift/CAGR을 재검증할 것 — 현재 수치는 우호적 구간 한정.
3. `[REGIME-L2] proof_failed path=pooled_fallback` 원인 규명 — 별도 이슈로 트래킹.
4. `docs/results/next.md` §1(`run_config.timeframe` CLI 기본값 "4h"의 tf-probe 기반 근거화)은 별도 `/spec` 대기 중.
