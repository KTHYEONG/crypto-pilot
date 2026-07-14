# System Performance Snapshot (2026-07-14)

- **측정**: `LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l1 --timeframe 4h --sync skip`
- **데이터**: 125 symbols loaded, 114 admitted (52 LTF-covered)

## 1. Time Profile

| Stage | Wall Clock | 비고 |
|-------|-----------:|------|
| tf_probe_scoped | 8.1s | |
| bridge_post_align | 0.2s | |
| **bridge_post_rules** | **169.3s** | `build_rule_signal_panels(4h)` + LTF 스트리밍(52 syms × 1m parquet) |
| panel_construction | 19.1s | `build_native_htf_panels` (6 TFs: 2h/4h/6h/8h/12h/1d) |
| l0_phase1_cheap | 6.2s | |
| l0_phase3_canonical | 3.3s | |
| l0_gate_multi_tf_wall | 9.5s | |
| l0_cross_tf_pruning | 4.5s | |

## 2. Memory Profile

| Phase | RSS (MB) | Delta (MB) | 비고 |
|-------|---------:|-----------:|------|
| universe | 420 | — | 414 symbols |
| data | 2,753 | +2,307 | 125 syms loaded |
| bridge_pre_align | 2,230 | — | 114 syms |
| bridge_post_rules | 3,833 | +1,603 | LTF streaming peak |
| htf_panels | 5,152 | +1,319 | 6 TF panels |
| pre_gc | **6,926** | +1,774 | 전체 peak |
| l1_swf (evidence) | 7,074 | — | per-TF L1 worker fork (6 TFs sequential) |

**Peak RSS**: 6.93GB (12GB cap의 **57.8%**)

## 3. Indicator Cache 효과

- `_resample_probe_source_frame` `.copy()` 제거로 RSS ~50MB 절감 (실측 7.0→6.93GB)
- `_SignalIndicatorCache` per-TF wiring: 정확성 104/104 PASS
- `build_rule_signal_panels` 자체는 bridge 169.3s 중 <2% → cache 영향 미미
- **진짜 병목**: `_build_ltf_native_panels_for_l0` (1m parquet load + streaming, ~170s 지배)

## 4. L1 SWF

- 6개 TF 전부 L1 루프 진입 성공
- 2h/4h/6h/8h/12h/1d 각각 evidence+outer fold 실행
- L1 evidence fold avg: schema 0.03s, ds_fit 0.31s, inference 0.01s, selection 0.04s
- 261 pairs promoted (4h 기준, sync skip)

## 5. 결론

1. `.copy()` 제거로 소폭 RAM 개선 (Δ≈50MB)
2. Indicator cache는 정확성 검증 완료, wall-clock 영향 없음 (original bottleneck misidentified)
3. LTF streaming (`_build_ltf_native_panels_for_l0`)이 170s로 bridge 전체의 ~70-80% 차지 → **다음 최적화 대상**
4. L1 SWF 정상 가동, 6개 TF 모두 vaild
