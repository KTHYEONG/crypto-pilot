---
title: Futures Universe Ledger Backend Compatibility
domain: futures.universe
type: adr
status: active
priority: high
ai_read_policy: when_related
---

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

## 2026-06-19 Store Consolidation: 단일 Parquet Store 통합 + cube.parquet 영속화
- **Delta:** `snapshots/` flat+nested JSON/Parquet (분기당 7개 파일, 203개) 완전 제거 → `store/v1/runs/` 유일 저장소. `_save_snapshot` flat/nested write 제거. `load_or_build_universe_snapshot` snapshot JSON cache 경로(170줄) 제거 → 40줄 2-tier(store hit→materialize, store miss→build). `write_universe_store_run`에 `snapshot=` 파라미터 추가 → `pit_state_cube`를 `cube.parquet`로 직렬화(numpy tobytes). `load_universe_store_run` 반환값 3→4 튜플 확장(cube 포함). `materialize_snapshot_from_store(cube=)` → snapshot에 `pit_state_cube` 복원. `gc_stale_store_runs()` 신규 함수. `discover_universe_timeline` `cfg=None` → `UniverseConfig()` default (기존 ValueError 대체). `write_universe_store_run` empty-decision short-circuit 제거 → 항상 3파일(manifest+decisions+report) 쓰도록 수정.
- **Rationale:** 3중 JSON/Parquet 중복 및 file proliferation(203→29개) 해소. `pit_state_cube` transient 손실 버그 수정(캐시 적중 시 eligible all-False). `snapshots/` 레거시 호환성 유지 불필요(store가 단일 SSOT). Store run 누적(69→29) 방지 위해 GC 추가.
- **Edge Cases:** 구버전 store run(`cube.parquet` 없음) → `cube=None` fallback(기존 동작 유지). Empty decisions→schema-only DataFrame write로 store 일관성 유지. `load_universe_snapshot` 함수는 dropout computation에서 사용 중이므로 repurpose(store에서 최신 run 로드).
