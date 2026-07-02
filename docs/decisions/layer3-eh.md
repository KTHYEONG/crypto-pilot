---
title: Layer 3 Holdout Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: critical
ai_read_policy: when_related
---
## 2026-07-02 Reversal-Kill Episode 계측 인프라 — Liquidity-Stress 판별력 측정 (measure-first)
- **Delta:** `awf_sim.py`에 `ReversalEpisode`(start_idx/end_idx/realized_price) dataclass + risk_off 연속구간 추출 로직 추가, `Layer2FoldAttribution`/`Layer3Result`/`L3ReversalReplayResult`에 `risk_off_episodes` 필드 배선. 신규 `liquidity_stress_diag.py`의 `compute_liquidity_stress_discriminative_power`가 half-spread z-score로 true-positive(진짜 방어)/false-positive(whipsaw) episode를 구분하는 `stress_gap`을 산출 — baseline window는 `risk_off_mask`로 사전 필터링해 오염 방지, contaminated episode는 그룹 통계에서 제외.
- **Rationale:** reversal-kill이 실제 위기에서 손실을 악화시킨 원인(whipsaw, `layer3-eh.md` 2026-07-01 항목 반증)이 가격-지연 단일축 탐지의 한계인지 계측하려면 episode 단위 타임스탬프가 필요했으나, 기존 fold attribution은 `risk_off_bars`(count)·`risk_off_realized_price`(scalar)로 즉시 집계돼 episode 경계가 소실되고 있었음.
- **Edge Cases:** 판별력 판정은 p<0.05 이진 게이트를 쓰지 않음 — `stress_gap`(방향성)과 `welch_p_value`(참고용)를 함께 반환해 표본부족 시에도 방향성 정보를 보존(L1 admission의 Bayesian 기준 전환 전례, OI/LSR p=0.285 판단보류 전례와 일관). Attribution(`_risk_off_1d[t2]`)과 실제 de-gross 결정(`_risk_off_1d[t-1]`, `rebalance_bars=3` 간격) 사이 최대 3-bar 시차는 알려진 한계로 남김(수정 안 함). **실 half-spread 데이터 기반 측정은 미실행** — 인프라만 구축, 채택 여부는 별도 measure-first 사이클 필요.

## 2026-06-18 L3 scorecard threshold alignment — Calmar removal + absolute gate thresholds
- **Delta:** L3 scorecard now renders `min_trades`, `max_mdd_abs`, `min_sharpe`, `min_sortino`, and `max_cvar95` from `Layer3Result` and drops Calmar from the display. The holdout gate order is now `negative_return` → `mdd_abs` → `cvar_95` → `sharpe_abs` → `sortino_abs`.
- **Rationale:** Calmar was only producing `n/a(loss)` after negative CAGR while the direct gate was already `negative_return`. Absolute thresholds make the replay contract explicit and keep the scorecard aligned with the actual blocker chain.
- **Edge Cases:** `negative_return` remains the first compound-loss blocker. Risk and efficiency thresholds are persisted on the result object so the formatter cannot drift from the gate contract.

## 2026-06-16 L3 빈 holdout 구조적 수정 — IS+OOS 데이터 병합 (PART4)
- **Delta:** `pick_strategy_data_maps`가 `oos_data_maps`를 버리고 IS-only를 반환하던 동작을 IS+OOS `concat+sort+dedup` 병합으로 교체. `full_strategy_maps`를 쓰는 모든 호출부(bridge, END-coverage 필터, `align_data_maps`)가 자동으로 holdout_end까지 데이터를 보게 됨.
- **Rationale:** `aligned.datetimes`가 구조적으로 `holdout_start`에서 끝나, `_resolve_holdout_span`이 항상 `empty_holdout_window`를 raise — "intersection tail truncation(상장폐지 심볼)"이라는 기존 진단은 오진이었고, 실제 원인은 데이터 소스 자체가 IS-only였던 것.
- **Edge Cases:** `keep="first"`로 IS 우선 — 경계 timestamp 중복 시 미래(OOS) 행이 과거를 덮어쓰지 않음. 부작용은 `layer2-eh.md`의 "L2 AWF fold anchoring 복원" 항목 참조(같은 작업에서 발견된 L2 fold 붕괴 regression).

## 2026-06-16 L3 평가체계 lean 보강 (PART2) — Phase D silent fallback 제거 (PART3)
- **Delta:** L3 게이트를 `cagr<0` 단일조건에서 5단계 순차 게이트(`insufficient_trades`→`negative_return`→`sharpe_rel`→`mdd_rel`→`mdd_abs`)로 교체. `total_return`, `equity_multiple`, `sortino`, `n_trades`, `cvar95`, `avg_gross_exposure`를 `Layer3Result`에 추가(L2 헬퍼 재사용, 신규 수학 없음). `except Exception` 발생 시 legacy Phase D fallback으로 조용히 넘어가던 동작을 제거 — 즉시 `RunnerResult(exit_code=1, reason="tiered_pipeline_error:...")`로 실패.
- **Rationale:** L3는 "1회 백테스팅으로 실제 복리자산증식 성과 판단"이 목적이므로 L2(Optuna 검증)와 동일한 수준의 풍부한 진단 지표는 불필요하나, CAGR/MDD/Sharpe/MAR만으론 빈약 — 단일패스 복리(`equity_multiple`)와 거래량 하한이 누락되어 있었음. Phase D fallback은 legacy 경로로, holdout 실패를 가려 "조용한 오류"를 만드는 위험이 있어 제거.
- **Edge Cases:** `max_mdd_abs`(기본 0.35)는 baseline 자체가 붕괴한 경우를 방어하는 절대 캡. `min_trades`(기본 10)는 L3 자체 기준으로 L2의 30보다 완화(단일 holdout 윈도우 특성 고려).

## 2026-06-18 L3 deployment parity 정합화
- **Delta:** `run_l3_holdout`가 선택적으로 `deploy_leverage`를 받아 L2 champion 배치와 동일한 `apply_deployment` 경로로 hybrid holdout의 CAGR/MDD/CVaR/terminal compounding을 계산하도록 변경. `run_tiered_pipeline`는 `l2_params["l2_deploy_leverage"]`를 L3까지 전달한다.
- **Rationale:** L2 승격 파라미터를 L3가 재사용하지 않으면 frozen holdout이 아니라 unit-path replay가 되어, L2/L3 결과 해석이 분리된다. 배치 계약을 L3에 주입해야 holdout 실패가 strategy failure인지 deployment mismatch인지 분리 가능하다.
- **Edge Cases:** `deploy_leverage`가 1.0 이하이거나 비유한값이면 unit path 유지. baseline은 비교용으로만 남기고 동일 배치하지 않는다.

## [2026-07-01] Champion Registry Restructure — BaselineChampionMetrics Split + Validation Package
- **Delta:** (1) `src/domain/futures/optimization/final_evaluator.py` underwent rename conflict resolution: existing `ChampionMetrics` (JSON/guard metrics) renamed to `BaselineChampionMetrics`; V3-renamed `ChampionMetrics` takes the unqualified name. `should_promote_candidate` deprecated; `legacy_should_promote_candidate` retains old logic. (2) `validation/champion_registry.py` created containing both `ChampionMetrics` and `BaselineChampionMetrics`, along with `Layer3Result`, promotion gate, and synthetic crash defense. (3) `validation/gates.py` wraps `ChampionGateConfig`/`evaluate_champion_gates`. (4) `validation/walk_forward.py` wraps layer-3 walk-forward orchestration. (5) `optimization/final_evaluator.py` updated to import `BaselineChampionMetrics` from `validation/champion_registry.py`.
- **Rationale:** The futures-refactor-redesign renamed `ChampionMetricsV3`→`ChampionMetrics` (versionless), creating a duplicate-class conflict with the existing JSON-guard `ChampionMetrics`. Rather than keeping both under the same module, the conflict was resolved by splitting: the guard class becomes `BaselineChampionMetrics` and lives alongside the V3 metrics in a shared `validation/champion_registry.py`. This makes the promotion + guard + baseline relationship explicit in one module.
- **Key Verification:** 191/191 regression tests pass. `final_evaluator.py` imports `BaselineChampionMetrics` from correct `validation/` location. MyPy strict passes. All V3-related test files (`test_champion_promotion_v3.py`, `test_hard_gates_v3.py`, `test_score_v3.py`, `test_v3_score_integration.py`) updated with renamed import paths.

## [2026-07-02] L3 Reversal-Kill Attribution Wiring + Economic Replay Harness
- **Delta:** (Phase 1) `Layer3Result` gained `risk_off_bars`/`risk_off_realized_price`/`risk_on_realized_price`/`reversal_kill_active`, populated in `run_l3_holdout` from `sim.fold_attributions[0]` (already computed by `_run_awf_simulation`, previously discarded) + direct `os.environ` read. `format_layer3_table` renders them via the existing optional-field `hasattr` convention. (Phase 2) `L2ReversalReplayVariant` + `_l2_reversal_replay_variants()` + `_temporary_reversal_env()` moved from `active_pipeline.py` to `pipeline.py` (zero app-layer coupling, safe relocation) to fix a domain→application layering violation and enable reuse. New `run_l3_reversal_economic_replay()` re-runs `run_l3_holdout` across the 8 variants, writing `docs/results/l3_reversal_replay.csv`. Wired into `run_tiered_pipeline` behind `L3_REVERSAL_REPLAY` env (off by default).
- **Rationale:** `_run_awf_simulation` already computed whether/how much the reversal-kill mechanism fired during any holdout, but L3 discarded it — the same "computed but not propagated" bug class as the 2026-07-01 Crisis-Guardrail Fold-MDD fix. Without this, "위기 없는 PASS" 감시(Gate A/B, `next.md` P0)는 여전히 "탐지 로직이 죽지 않았다"만 보장할 뿐, 실제 L3 홀드아웃에서 방어가 작동했는지는 관측 불가능했다.
- **Empirical Finding (실제 파이프라인 재실행, 2026-07-02, `L2_REVERSAL_KILL=1 L3_REVERSAL_REPLAY=1`, 8h tf, 실 BTC 데이터 2025-12-31~2026-06-30, BTC -32.8%/peak-trough -39.5% 실측 위기 구간):** `baseline_off`(reversal-kill 비활성)의 CAGR -4.96%/MDD 23.78%가 나머지 7개 활성 variant 전부보다 우수했다(활성 variant CAGR -5.04%~-5.89%, MDD 24.18%~24.64% — 전부 baseline보다 나쁨). `risk_off_realized_price`(kill-switch 발동 구간의 실현 가격 성과)가 전 variant에서 양수(+5.89%~+10.50%) — kill-switch가 de-gross한 바로 그 구간에서 원 신호가 실제로는 수익 중이었다는 뜻. **`next.md`가 "L\* 흡수를 피하는 유일하게 검증된 방어 레버"로 지목했던 reversal kill-switch가 이 실제 위기 구간의 economic replay에서 방어는커녕 손실을 악화시켰다 — 최초의 실제 crisis-window economic replay 결과가 반증.** SSOT/후속 조치: `docs/results/next.md` §1, §2 P1/P2, §3.
- **Key Verification:** 회귀 스위트 전체 PASS(check 단계 완료). Test scenarios: fold_attribution 배관(P1-S1~S4), env-독립 `reversal_kill_active`(P1-S2), 빈 `fold_attributions` fallback(P1-S3), 8-variant env 스코핑 + 종료 후 env 복원(P2-S5~S6) — 확립된 mocking 경계(`_run_awf_simulation`/`run_l3_holdout` boundary patch, synthetic price path 대신 canned dataclass) 준수.

## 2026-07-02 P1① OI/LSR/funding 조기탐지 후보 5라운드 실측 — 전부 반증 또는 유의성 미달
- **Delta:** 코드 변경 없음(순수 측정). `docs/decisions/universe-eh.md`(같은 날)의 OI/LSR 파이프라인 결함 수정으로 실데이터가 확보된 직후, BTC 단일축~7개 대형심볼(BTC/ETH/SOL/XRP/BNB/ADA/DOGE)에 걸쳐 `Binance Vision` 실데이터로 위기 구간(2025-12-31~2026-06-30, BTC -32.8%) vs 평온 대조 구간(2025-06-30~2025-12-30)을 직접 대조 측정.
- **결과 5종 전부 反證 또는 미확정:** ① LSR z-score(20d) 극단치 — calm 발생빈도(6.5%)가 crisis(5.3%)보다 높음. ② OI 1일 변화율 — corr(당일수익률) crisis 0.041 < calm 0.312(역방향). ③ funding×OI 3일 결합(절대 threshold) — 방향은 맞으나(crisis 기저율 3배, fwd_ret_3d -2.55%(7심볼 합산) vs calm -0.15%) Welch t-test p=0.285로 유의성 미달. ④ OI 14일 장기추세 — calm 기저율(10.9%)이 crisis(7.5%)보다 높고 fwd_ret_5d도 calm이 더 음수(방향 반전, trend-beta의 대리변수일 뿐임을 시사). ⑤ OI/funding 스트레스 멀티심볼 breadth — **calm 구간 최대 breadth(0.857, 7개 중 6개)가 crisis 구간 최대(0.429)를 상회** — 기존에 반증된 가격기반 breadth(`next.md` §1)와 동일한 실패 패턴(고상관 유니버스에서 breadth 레벨이 위기/평온을 구분 못함)을 OI/funding 축에서도 재확인.
- **Rationale/해석:** 관측 가능한 히스토리에 독립적인 "진짜 위기" 국면이 사실상 1회(2026년 1-2월)뿐이라는 근본적 통계적 검정력 한계가 모든 라운드에 공통. funding×OI(③)만 유일하게 방향성 있는 point estimate를 보였으나 유의성 미달(n=45/15, p=0.285)이라 프로덕션 채택 근거로 부족. reversal-kill(가격 단일축) 교체/보강 후보로서 OI/LSR/funding 마이크로구조가 next.md P1①이 기대한 만큼의 즉각적 해법은 아님이 실측으로 확인됨.
- **Edge Cases/한계:** 신규 위기 국면이 재유입될 때(next.md P2 절차 재사용) 표본이 늘어나면 ③(funding×OI)은 재검정 가치가 있음 — 유일하게 완전히 반증되지는 않은 후보. 심볼 이질성 관찰: BNBUSDT는 두 구간 모두 조건 발생 0건, XRPUSDT는 crisis 구간에서도 신호 없음 — 신호가 일부 심볼(SOL/DOGE/ETH/BTC)에 편중돼 심볼 선택 편향 위험 존재, 향후 재시도 시 유니버스 전체로 확장 필요. SSOT: `docs/results/next.md` §1/§2 P1①.

