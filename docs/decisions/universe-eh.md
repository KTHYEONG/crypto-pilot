---
title: Futures Universe Ledger Backend Compatibility
domain: futures.universe
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## 2026-06-19 PIT-BREADTH: 풀-윈도우 생존편향 필터 교체 + 용량커버리지 Cap + warm-up 가드
- **Delta:** (C1) `opt_main_futures._resolve_tradeable_scope` 추가 — 3-guard PIT 어드미션(warm-up: `datetimes.min()≤fetch_start`, `min_bars≥1500`, OOS-cov≥0.90). 풀-윈도우 END-coverage(`first≤fetch_start AND last≥holdout_end`) 폐지. `_TIERED_MIN_WINDOW_BARS=1500` 모듈 상수화. (C2) `PITUniverseConfig.k_in=0` 기본값; `capacity_coverage_target=0.90`, `k_max=100` 추가 — 누적 용량 90% prefix 알고리즘. (warm-up guard fix) `datetimes.min()>fetch_start` 심볼 reject: 교집합 start가 밀려 `ValueError: tiered warm-up coverage missing` 유발 차단.
- **Rationale:** END-coverage 필터가 633 온디스크 심볼을 54 "올드가드"로 붕괴 → PIT 설계가 막으려던 생존편향 재주입. 2023-10~2024-09 상장 110종 통째 배제. k_in=50은 교집합(231)·active_mask에 비구속(inert)이었으나 magic number 정당화 불가 → Pareto 용량 커버리지로 대체.
- **Edge Cases:** total capacity=0 → fail-open(`eligible[:k_max]`). fetch_start 이후 상장된 심볼은 warm-up guard로 자동 제외(교집합 보전). OOS 절단 심볼은 90% coverage guard로 제외.

## 2026-06-19 L2-ZERO: PIT cube bypass 해소 + store build/hit mismatch 수정
- **Delta:** `opt_main_futures.py`에 `_resolve_universe_state_cube()` 신규 함수 추가 → `_run_strategy_stage`에서 `universe_result`에서 cube 추출하여 `align_data_maps(state_cube=)` 주입. `pipeline.py` `_is_incomplete_pit_store_run()` 추가 → `load_or_build_universe_snapshot`에서 store hit 시 cube null 체크 후 rebuild. `discover_universe_timeline`에 `l2_start` timeline 경계 강제 로직 추가.
- **Rationale:** P0 - production 경로에서 `state_cube=None` 전달로 인해 L1/L2가 동일 PIT 필터를 소비하지 못함. P0 - store hit 시 decisions empty로 저장/복원되어 selection 정보 소실. P1 - L1/L2가 다른 시작 경계를 가져야 할 때 timeline이 2-way 계산만 함.
- **Edge Cases:** `universe_result is None` → cube=None 유지(기존 fallback 호환). Store hit + cube.parquet 없음 → cube=None fallback → incomplete 감지 → rebuild.

## 2026-06-19 EXACT-FIELDS: execution_pool_score 제거, exact-field only store contract
- **Delta:** `UNIVERSE_DECISION_COLUMNS`에서 `execution_pool_score` 제거. `_selected_frame_columns()`에서 제거. `build_decision_frame()`에서 `execution_pool_score` 쓰기 제거. `materialize_snapshot_from_store()`에서 alias 역매핑 제거. `_symbol_meta_from_decision_row()`에서 `alpha_capacity_score` 단독 사용. 구 cache hit 시 alias-only decisions → `is_exact_selected_feature_schema` False → rebuild. `_universe_metadata_by_symbol()`는 snapshot.selected exact field만 읽음.
- **Rationale:** Store/cache 계층에 `alpha_capacity_score`와 `execution_pool_score`가 동시 존재 → 동일 개념의 2개 truth source. Exact-field only로 단일화하여 cache-hit/fresh-build 간 metadata 불일치 원천 차단.
- **Edge Cases:** 구버전 store run(alias-only) → `build_decision_frame`가 `execution_pool_score` 컬럼을 남겨도 `validate_materializable_pit_store_run`가 detect → rebuild. `pipeline.py:649`에서 `alpha_capacity_score` 우선, 없으면 `execution_pool_score` fallback 유지.

## 2026-06-19 CAPACITY-CLIP: unit-NAV 시뮬레이션에서 portfolio_nav=1.0 → capacity clip 전멸
- **Delta:** `awf_sim.py:_run_awf_simulation`에 `_capacity_clip_enabled` 플래그 추가 (`portfolio_nav is not None`). fit-leg(829) 및 OOS(1025) capacity clip을 `_capacity_clip_enabled` 조건으로 가드.
- **Cause:** `portfolio_nav=None` → `_portfolio_nav=1.0` (unit-NAV). `_min_order_usdt=5.0` → `abs(w)*1.0 < 5.0` → per_symbol cap 10%를 통과한 모든 weight가 zero-out. commit `5f0254f`에서 state_cube와 동시에 추가됨.
- **Rationale:** Unit-NAV 시뮬레이션에서 w는 분수(fraction)이지 USDT 금액이 아님. 최소주문($5)을 weight에 직접 비교하는 것은 차원 오류. 실제 portfolio_nav가 주입될 때만 capacity clip을 활성화.

## 2026-06-19 KELLY-FRICTION: diagonal_kelly_weights 이중 friction filter 제거
- **Delta:** `portfolio_constructor.py:diagonal_kelly_weights`에서 Step 1 friction filter(`mu_bps < effective_hurdle = hurdle * safety_mult / holding_bars`) 제거. `friction_hurdle_bps`, `holding_bars`, `friction_safety_mult` 파라미터와 `hurdle` 변수 삭제. `awf_sim.py` 두 호출부에서 해당 인자 제거.
- **Cause:** `mu_bps` (`signed_net_bps_per_bar`)는 이미 edge computation에서 cost가 차감된 NET 값. `diagonal_kelly_weights`가 이를 다시 `hurdle * safety_mult / holding_bars`와 비교하면 이중과세 발생.
  - state_cube 도입 전(3.8 bps): `hurdle*2.5=9.5` → `gross(20)>9.5` → 통과
  - state_cube 도입 후(12.4 bps): `hurdle*2.5=31.0` → `gross(20)<31.0` → **전량 zero-out**
  - 결과: `trade_count=0`, `Best CAGR: 0.00%`
- **Rationale:** P0 - PIT cube 주입으로 `execution_cost_bps_2d`가 3.8→12.4로 상승하면서 이중 friction filter가 모든 신호를 차단. mu_bps는 이미 NET이므로 friction filter 자체가 개념적으로 불필요.
- **Fix:** Step 1 제거. Kelly 계산은 net edge를 그대로 사용. 모든 200 trial에 적용.
- **Impact:** `_run_awf_simulation`의 fit(801) 및 OOS(987) 경로 모두에 적용. no-trade band(Step 3)는 유지.


## 2026-06-19 META-PARITY: UNIVERSE_DECISION_COLUMNS에 metadata 필드 추가 + full materialization
- **Delta:** `UNIVERSE_DECISION_COLUMNS`에 `vol_30d`, `friction_score`, `alpha_capacity_score`, `diversification_score` 4개 필드 추가. `_symbol_meta_from_decision_row()`에서 해당 필드 복원. `materialize_snapshot_from_store()`가 `decisions.parquet`에서 `SymbolMeta` 전체 필드 재구성. `_selected_meta_to_frame()` 추가 → `build_universe()` output을 decision columns와 일치. `_save_snapshot`에 `decisions=` 파라미터 추가.
- **Rationale:** cold build 시 `SymbolMeta`에 채워진 확장 필드가 cache-hit 시 `0.0` default로 떨어져 L1/L2가 다른 feature vector를 소비. Store schema에 exact field를 포함시켜 build/hit 간 metadata parity 보장.
- **Edge Cases:** 구버전 decisions(필드 누락) → `is_exact_selected_feature_schema` False → `validate_materializable_pit_store_run` False → rebuild 유도.

## 2026-06-19 Phase 4-B/C/E: Stage2-6 Config 및 legacy selection 제거
- **Delta:** `Stage2-6Config` 5종 class config.py에서 제거; `UniverseConfig`에서 `stage2-6/strategy_pool_mode/stage6_is_alpha_rank` 필드 제거. `Stage2Config` → `data_quality._DataQualityConfig` 인라인, `Stage3-5Config` → `filters.py` 로컬 이동. `selection.py` 전체 삭제(`apply_selection_stage` 포함). `pipeline.py` basket_ref/weights → `()`. 레거시 테스트 2종(`test_selection`, `test_strategy_pool_selection`) 삭제.
- **Rationale:** Phase 4-A에서 Stage6 else-branch 제거 완료 후 dead code 정리. `universe_engine` default = `"pit"` (4-A 적용). Stage2-5 config은 필터 유틸리티 함수 로컬 타입으로 유지(test_oi_adv_filter 호환).
- **Edge Cases:** 구버전 `Stage6Config` import하는 외부 테스트 → `@pytest.mark.skip` 처리(4-E); `k_in=50` cap으로 PIT 범위 제한.

## 2026-06-19 Phase 4-A: PIT 단독 경로 확정 + k_in=50 cap
- **Delta:** `build_universe`에서 Stage6 else-branch 완전 제거. `universe_engine` default `"stage6"` → `"pit"`. `PITUniverseConfig.k_in=50` 추가(capacity_usdt 내림차순 top-50). `store.py` empty decisions early return 추가.
- **Rationale:** PIT 경로 shadow validation PASS 후 Stage6 code path 불필요. k_in cap은 411 → 50 symbols로 제한(임시, Phase 4-D 이후 완전 제거 검토).
- **Edge Cases:** ledger `date` vs `datetime` 비교 TypeError 픽스(pipeline.py `_instrument_df_from_ledger`).

## 2026-06-19 Phase 3-3/3-4/3-5: PIT state_cube L1 wiring + lifecycle + capacity
- **Delta:** Phase 3-3: `_run_universe_stage` 7-tuple 반환(`universe_result` 추가). `align_data_maps` 호출에 `state_cube=` 주입 → `active_mask` PIT 반영. Phase 3-4: `SymbolLifecycleRecord` 추가, `promotion_available_at > l2_start` gate로 late-listing 심볼 L2 제외. Phase 3-5: `awf_sim` fit/OOS 양쪽에 `capacity_usdt` clip + 5 USDT min order threshold.
- **Rationale:** PIT state_cube 없이 L1이 stage6 all-True mask 사용 → look-ahead 노출. Lifecycle gate는 mid-window 상장 심볼이 OOS 신호에 참여하는 것을 방지. Capacity clip은 소량 포지션 거래비용 현실화.
- **Edge Cases:** `AlignedMarketData` frozen=True → `dataclasses.replace`; `adv_usdt_2d` shape 동적 체크(`isinstance(np.ndarray)`).

## 2026-06-15 Ledger backend compatibility recovery
- **Delta:** `load_ledger_slice(...)` now dispatches by backend suffix and supports both SQLite and parquet fixtures through the same PIT filter path.
- **Rationale:** universe tests and offline snapshots depend on parquet inputs; the loader must not collapse existing files into silent empty stage0 results.
- **Edge Cases:** missing files may still return empty frames, but readable files that fail backend-specific loading now raise with explicit backend context.

## 2026-06-19 Phase 4-D: UniverseSnapshot legacy panel 6필드 제거 + Stage6 경로 완전 삭제
- **Delta:** `UniverseSnapshot`에서 `training_panel`/`inference_panel`/`live_inference_panel`/`historical_trading_panel`/`inference_panel_quarter_membership`/`stage5_research_panel` 6개 필드 정의 제거. `discover_universe_timeline`의 Stage6 else-branch(230줄) 전체 삭제, dispatch는 PIT 무조건 호출로 단순화, `cfg=None` → `ValueError("universe_engine=pit required; stage6 path removed")` raise. Dead 헬퍼 `_resolve_trading_membership`, `_resolve_inference_membership` 삭제(`_snapshot_quality_symbols`는 `validate_universe_quality`에서 사용 중이므로 리팩터하여 유지). `snapshot_to_payload`/`snapshot_from_payload`에서 panel 직렬화 제거(구버전 payload key는 자동 무시). `store.py` `UniverseSnapshot(...)` panel 대입 제거. `pipeline.py` panel read + `replace(snapshot, ...)` 블록 제거. `strategy_service.py` `run_active_strategy_output_bridge`에서 panel 4개 파라미터 및 `training_panel` filter 제거. `opt_main_futures.py` 호출부 정리 및 `_run_universe_stage` extraction → `universe_result.inference_symbols`.
- **Rationale:** Stage6 panel 필드는 PIT state_cube가 유일 SSOT인 체계에서 불필요한 이중경로. Phase 4-A/4-B/4-C/E에서 Stage6 제거 후 최종 잔여 legacy 필드/경로 정리. `payload.get`-based deserialization은 구버전 스냅샷과의 하위호환 유지.
- **Edge Cases:** `cfg=None` → 명시적 raise, silent fallback 금지. `validate_universe_quality`가 `_snapshot_quality_symbols`에 의존하므로 함수 유지. `n_stageN` int 카운터는 별도 4-F 후보로 제거 대상 아님.

## 2026-06-19 Stage0.empty empty-universe contract: cube 강제 주입
- **Delta:** `build_universe()` stage0.empty 분기에서 `materialize_snapshot_from_store` 호출 시 `cube=None` 대신 empty `UniverseStateCube`(모든 array shape `(0,0)`, `eligible` all `False`)를 명시적으로 생성하여 전달. `validate_materializable_pit_store_run`가 empty-universe를 spec 계약(cube 존재 + eligible all False + zero selected) 하에서 통과시킴.
- **Rationale:** stage0.empty 경로에서 `cube=None` 전달 시 validator가 `cube is None` → `False` 반환 → `ValueError` 발생. 이는 cold build empty-universe 경로가 validated PIT snapshot만 소비한다는 계약을 위반. empty cube 생성으로 일관된 validator 통과 보장.
- **Edge Cases:** `np.empty((0,0), dtype=bool).any()` → `False` (empty array), `selected.empty` → True (zero rows), spec 계약 충족.

## 2026-06-19 Store Consolidation: 단일 Parquet Store 통합 + cube.parquet 영속화
- **Delta:** `snapshots/` flat+nested JSON/Parquet (분기당 7개 파일, 203개) 완전 제거 → `store/v1/runs/` 유일 저장소. `_save_snapshot` flat/nested write 제거. `load_or_build_universe_snapshot` snapshot JSON cache 경로(170줄) 제거 → 40줄 2-tier(store hit→materialize, store miss→build). `write_universe_store_run`에 `snapshot=` 파라미터 추가 → `pit_state_cube`를 `cube.parquet`로 직렬화(numpy tobytes). `load_universe_store_run` 반환값 3→4 튜플 확장(cube 포함). `materialize_snapshot_from_store(cube=)` → snapshot에 `pit_state_cube` 복원. `gc_stale_store_runs()` 신규 함수. `discover_universe_timeline` `cfg=None` → `UniverseConfig()` default (기존 ValueError 대체). `write_universe_store_run` empty-decision short-circuit 제거 → 항상 3파일(manifest+decisions+report) 쓰도록 수정.
- **Rationale:** 3중 JSON/Parquet 중복 및 file proliferation(203→29개) 해소. `pit_state_cube` transient 손실 버그 수정(캐시 적중 시 eligible all-False). `snapshots/` 레거시 호환성 유지 불필요(store가 단일 SSOT). Store run 누적(69→29) 방지 위해 GC 추가.
- **Edge Cases:** 구버전 store run(`cube.parquet` 없음) → `cube=None` fallback(기존 동작 유지). Empty decisions→schema-only DataFrame write로 store 일관성 유지. `load_universe_snapshot` 함수는 dropout computation에서 사용 중이므로 repurpose(store에서 최신 run 로드).
