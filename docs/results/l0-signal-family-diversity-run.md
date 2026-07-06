# L0 Signal Family Diversity — Real Run Report

- Run date: 2026-07-06
- 관련 spec: `docs/specs/l0-l1-signal-family-diversity.md`
- 실행 명령(공통): `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe <TF> --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`

## ⚠️ 정정: `--timeframe 6h/1d` 직접 실행은 아키텍처 오용이었음

최초 시도에서 `--timeframe 6h`, `--timeframe 1d`를 alpha-foundry gate의 native TF로 직접 실행했으나 둘 다 `universe_quality_rejected`로 크래시했다. 이를 "유니버스 계층의 버그"로 잘못 보고했으나, 코드 재추적 결과 **애초에 이 시스템은 4h만을 유일한 native/base TF로 설계**되어 있고, 6h/8h/12h는 `--timeframe 4h` 단일 실행 안에서 `build_multi_tf_panels`(`strategy_runtime/bridge.py:372-`)를 통해 4h 정렬 데이터를 리샘플링한 **HTF(상위 TF) 파생 패널**로만 존재하도록 설계돼 있다. 즉 6h/8h/12h 탐색은 이미 `--timeframe 4h` 단일 실행 안에 내장되어 자동으로 일어나고 있었다 — 별도 CLI 실행이 필요하지도, 지원되지도 않는다. `universe_quality_rejected`는 버그가 아니라 "4h가 유일한 진입점"이라는 설계를 재확인해준 것.

**올바른 실행은 이미 완료되어 있었다**: `--timeframe 4h --alpha-foundry gate` (run_id `4h_1783345440`) 하나로 native 4h L0 평가 + 6h/8h/12h HTF main block 탐색이 전부 포함된다.

## 실행 결과 요약

| TF | 결과 | 비고 |
|---|---|---|
| 4h (native) | ✅ 성공 (run_id `4h_1783345440`) | 오펀 4종 포함 27개 family 전량 L0 평가 확인 |
| 6h/8h/12h (HTF, 4h 실행에 내장) | ✅ 자동 수행됨 | 별도 실행 불필요 — 아래 발견 4 참고 |
| 1d | 미지원 | l1_tfs/HTF 파이프라인에 1d가 등록되어 있지 않아 4h 기반 HTF로도 탐색되지 않음 |

## 발견 1 — 오펀 4종 L0 편입 확인 (spec 목표 달성)

`logs/futures/alpha_foundry/4h_1783345440_evidence.parquet` 실측 확인: family 27종 전량 등장(오펀 `macd_4h`, `supertrend`, `ichimoku_trend`, `positioning_unwind` 포함, 이전 run `4h_1783337608`는 23종만 존재).

| family | variant | mean_net_bps | nw_tstat | block_lcb_bps | reject_reasons |
|---|---|---:|---:|---:|---|
| ichimoku_trend | ichi_9_26 | -12.92 | -0.86 | -28.15 | non_positive_lcb\|weak_tstat\|excess_cost_drag |
| macd_4h | macd_12_26_9 | -8.57 | -0.91 | -17.92 | non_positive_lcb\|weak_tstat\|excess_cost_drag |
| supertrend | st_10_4h | -17.75 | -0.80 | -39.61 | non_positive_lcb\|weak_tstat\|excess_cost_drag |
| positioning_unwind | pu_42_4h | +30.61 | 0.58 | -15.73 | non_positive_lcb\|weak_tstat |

**결론**: 4개 오펀 family 전부 4h에서 LCB 음수로 실측 기각. `positioning_unwind`는 point-estimate는 양수(+30.6bps)지만 표본 불안정(LCB -15.7, tstat 0.58)으로 통계적으로 유의하지 않음 — "제거 대상"으로 잠정 분류하되 다른 TF에서 재평가 여지는 열어둔다(단, 아래 발견 2로 인해 다른 TF에서 동일한 방식의 재평가가 현재 불가능).

L1 승격 후보 3건은 기존 run과 동일(변동 없음): `lsr_oi_regime_filter`(seed), `mtf_breakout_retest`(candidate), `trend_pullback_continuation`(seed).

## 발견 2 — HTF(6h/8h/12h) 패널은 alpha-foundry L0 경제성 게이트를 완전히 우회함 (신규, 핵심 발견)

`strategy_runtime/bridge.py` 재추적 결과, 실행 순서가 다음과 같음을 코드로 확인했다:

1. `panels = build_rule_signal_panels(aligned=aligned, cfg=strategy_cfg.candidate)` (L934) — native 4h, 27개 family 전량
2. `run_alpha_foundry_l0_gate(panels=panels, ..., timeframe=tf)` (L956) — **이 시점의 `panels`는 아직 native 4h뿐**. LCB/tstat/cost-drag/turnover 경제성 게이트가 여기서만 실행됨.
3. `panels = af_result.panels_for_l1` — L0 통과분으로 교체
4. `labeled = label_candidate_events(panels, ...)` → `labeled_all = labeled.copy()`
5. `htf_tfs = tuple(t for t in candidate_cfg.l1_tfs if t != tf)` (L1045) → `build_multi_tf_panels(family_pool=lambda t: resolve_tf_signal_pool(candidate_cfg, t))` (L1051) — 6h/8h/12h HTF 패널을 **여기서 새로** 생성, `family_filter=resolve_tf_signal_pool(...)`로 family만 제한하고 **`run_alpha_foundry_l0_gate`는 다시 호출되지 않음**
6. `labeled_all = pd.concat([labeled_all, htf_labeled])` → L1 fold 단계로 직행

**결론**: L0의 family-level 집계 경제성 스크리닝(LCB/tstat/cost-drag)은 native 4h에만 적용되고, 6h/8h/12h HTF 후보는 이를 완전히 건너뛴 채 `resolve_tf_signal_pool` family 필터만 거쳐 곧장 L1 fold 평가(개별 pair 단위 no_incremental_edge/quality_weight_zero/negative_gross_edge 필터)로 들어간다. main block이 49~98건씩 대량 promote하는 반면 AF-gated block은 3~5건에 그치는 격차의 근본 원인이 바로 이것 — "가족 다양성 부족"이 아니라 **"HTF 후보군 전체가 L0 사전 경제성 필터를 아예 거치지 않는 구조"**다.

## 발견 3 — main block(6h/8h/12h) family별 세부 evidence는 이미 메모리에 존재하나 영속화되지 않음

이번 4h 실행에 내장된 6h/8h/12h main block Top 5 (전체 로그: `run_4h.log`):

| TF | Promoted | Top 5 families |
|---|---:|---|
| 6h | 49 | trend_donchian ×3, trend_pullback_continuation ×1, (5th truncated) |
| 8h | 47 | trend_donchian ×2, trend_pullback_continuation ×2, dual_momentum ×1 |
| 12h | 98 | trend_donchian ×5 |

이번에 per-TF pool을 넓힌 family(`xs_flow`, `xs_oi_skew`, `mtf_breakout_retest`, `lsr_oi_regime_filter` @6h/8h, `vol_term_structure_gate`, `flow_trend_continuation` @12h)는 콘솔 Top 5는 물론 전체 로그 grep에서도 0건 검출됐지만, 코드 추적 결과 이는 "효과 없음"이 아니라 **"측정 불가"**임을 확인했다: `tiered_workflow/pipeline.py:1569` `compute_symbol_strategy_evidence()`가 만드는 `deployment_evidence`(family/strategy_id 포함 pair-level evidence)가 `Layer1Result.deployment_evidence`(`pipeline.py:1653`)로 이미 존재하지만, 콘솔 Top-5 렌더링(`format_layer1_deployment_registry_table`, `tiered_logging.py:569`) 외에는 parquet/csv로 저장되는 경로가 없다.

## 결론 및 다음 액션

1. **4h native L0 평가는 정상 작동 및 목표 달성**: 오펀 4종이 이제 실제로 평가되고, 전부 실측 데이터로 기각됨(추측이 아닌 확인된 사실). L1 승격 3건은 변동 없음.
2. **`--timeframe`을 6h/8h/12h/1d로 바꿔 개별 실행하는 접근은 아키텍처 오용이었다** — 4h가 유일한 base이며 6h/8h/12h는 4h 실행 안에서 자동으로 HTF 파생 평가된다. 이 부분은 이미 올바르게 동작 중이며 추가 실행이 불필요하다.
3. **신규 핵심 발견— HTF 후보는 L0 경제성 게이트를 완전히 건너뛴다**: `run_alpha_foundry_l0_gate`가 native TF `panels`에만 적용되고 HTF 패널 생성(`build_multi_tf_panels`)은 그 이후에 일어나 L0를 다시 통과하지 않는다. main block의 대량 promotion(49~98건)이 AF-gated block(3~5건)과 근본적으로 다른 엄격도의 게이트를 거친다는 뜻 — "family 다양성 부족"보다 이쪽이 실제 구조적 리스크에 가깝다.
4. **HTF family-level evidence는 이미 메모리에 존재, 영속화만 안 됨** — `deployment_evidence`를 parquet으로 dump하는 최소 로깅 추가만으로 6h/8h/12h family 기여도를 직접 검증할 수 있다. 코드 변경이 필요하므로 다음 진행 여부는 사용자 확인 필요.
5. **1d는 4h 기반 HTF 파이프라인(`l1_tfs`)에 등록되어 있지 않아 현재 탐색 대상이 아니다** — 추가하려면 `l1_tfs`에 "1d" 편입 + HTF 리샘플 검증이 필요(별도 `arc` 대상).
