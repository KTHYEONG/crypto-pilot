# 후속 조치 목록

## 1. L2 실행 그리드(`run_config.timeframe`) 근거 기반 선정 — tf-probe 재연결

- **배경**: `docs/specs/l1-l2-master-tf-handoff-wiring.md`는 L1 registry 대표 TF(`l2_master_tf`, `assess_l1_tf_handoff` 기반) 선정만 다룬다. 이와 별개로 `--timeframe` CLI 인자(기본값 `"4h"`, `cli.py:22`)가 L0/L1 데이터 정렬 그리드 및 L2 실행/vol-lookback 그리드(`run_config.timeframe` → `build_l2_simulation_cache`의 `tf` 파라미터)를 결정한다.
- **과거엔 임의가 아니었음**: 구 아키텍처에서는 L0 경제성 게이트가 4h에서만 돌고 HTF(6h/8h/12h)는 `project_htf_panels_to_base`로 4h 그리드에 투영만 되어 게이트를 완전히 우회했다(`--timeframe`을 6h/1d로 직접 실행하는 것은 "아키텍처 오용"으로 문서화됨).
- **현재는 임의임**: 이번 주 "L1→L2 네이티브 TF 핸드오프"(`c2831990`, `c090973c`) 작업으로 L0가 1h/2h/6h/8h/12h/1d 각각에 대해 native panel을 만들고 L1이 TF별로 독립 검증하게 되면서, `run_config.timeframe`이 더 이상 L0/L1 경제성 게이트를 강제하지 않는다. 순수하게 L2 실행 그리드 선택으로 축소되었고, 지금은 CLI에서 안 건드리면 그냥 `"4h"`로 고정되는 **근거 없는 기본값**이다.
- **tf-probe와의 역사적 관계**: `src/domain/futures/strategy/timeframe_probe.py::scan_timeframe_alpha`(HAC t-stat, BH-FDR, fold_sign_consistency, alpha_half_life, net_edge_bps, Hurst/VR)는 2026-06-22 최초 `_resolve_l2_master_tf` 구현에서 tier-3 fallback으로 실제 연결되어 있었으나(`ADR_20260713_L1_DEPLOYMENT_PASS_CONTRACT`에서 제거), **`run_config.timeframe`(실행 그리드) 선택에는 연결된 적이 한 번도 없다**. 지금은 `probe_timeframes.py`라는 완전히 별도 CLI 스크립트로만 존재하고 parquet 산출물만 남길 뿐, 실제 러너의 `--timeframe` 기본값 결정에 자동으로 반영되지 않는다.
- **제안하는 다음 스텝(별도 `/spec` 필요, 이번 스펙과 무관)**:
  1. `scan_timeframe_alpha` + `summarize_timeframe_scan_gate_audit`로 대상 유니버스에 대해 tf_grid 사전 스캔.
  2. FDR-생존 + 비용조정(net_edge_bps) + fold-안정성 기준으로 가장 강건한 TF를 `run_config.timeframe` 기본값 추천 근거로 사용.
  3. 대상 코드: `src/application/futures/runner/cli.py`(`--timeframe` 기본값 및 CLI 해석), `src/application/futures/runner/config.py`(`build_run_config_from_args`) — `pipeline.py`/`active_pipeline.py`의 `l2_master_tf` 관련 코드와는 별개 위치.
- **결정 필요 사항**: 위 스텝을 새 `/spec`으로 착수할지, 이번 우선순위에서 보류할지 — 사용자 판단 대기.

## 2. (이번 spec 완료 후 여기에 추가)

- `docs/specs/l1-l2-master-tf-handoff-wiring.md` 구현(`/implement`) 및 실측 replay 재실행 결과를 여기에 추가할 것.
