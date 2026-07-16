# L0/L1 최신 파이프라인 결과 (2026-07-16, 심볼 다양성/LCB 재보정 반영)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python -m src.domain.futures.strategy.run_l1_cross_tf_diagnosis`
- **적용**: (1) 일평균 이벤트 밀도 검정 (Density-based Gate Policy), (2) TF별 측정-후-채택 심볼 다양성 하한 재보정, (3) 부트스트랩 블록 수 적응형 LCB quantile
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **결과 파일**: `logs/futures/diagnostics/l1_cross_tf/treatment.json` 및 [diagnosis.json](file:///home/kth/my_coin_traider/logs/futures/diagnostics/l1_cross_tf/diagnosis.json)
- **프로세스 종료**: `exit_code=0`, `reason=l1_mode_done` (4개 Replay 런 전부 동일 결과로 완주 — `ablation_restores_control: true`)

## L1 판정 (심볼 다양성 재보정 + 적응형 LCB quantile 적용 후)

| Timeframe | 판정 | Symbol-Breadth | probe_lcb_bps | 실제 blocker |
| :--- | :---: | :---: | :---: | :--- |
| **1h** | **✅ PASS** | 28.582(≥5.00) | 37.182 | 없음 (회귀 없음) |
| **2h** | **✅ PASS** | 47.222(≥5.00) | 50.699 | 없음 (회귀 없음) |
| **4h** | **✅ PASS (⚠️)** | 3.000(≥3.00) | 18.990 | `fold_ratio:0.250` (advisory, non-blocking — 회귀 없음) |
| **6h** | **❌ BLOCKED** | 3.000(≥3.00) — **PASS** | -40.459 | `probe_lcb_bps:-40.459` — Symbol-Breadth 통과했지만 순수 경제성 적자 (2개 실데이터 fold의 gross edge 자체가 -54bps/-33bps) |
| **8h** | **❌ BLOCKED** | 2.000(≥2.00) — **PASS** | -35.095 | `probe_lcb_bps:-35.095`, `fold_ratio:0.250` — Symbol-Breadth 통과, 소표본 LCB 잔존 |
| **12h** | **❌ BLOCKED** | 1.000(≥1.00) — **PASS** | -inf | `probe_lcb_bps:-inf`, `match_ratio:0.787` — Symbol-Breadth 통과했으나 **fold-level `insufficient_ready_symbols`가 별도로 LUNA2USDT(+229.12bps) fold를 pooled LCB 계산에서 제외** |
| **1d** | **❌ BLOCKED** | 1.000(≥1.00) — **PASS** | -inf | `probe_lcb_bps:-inf` — 동일 원인, JASMYUSDT(+98.50bps) fold가 pooled 계산에서 제외 |

### 이번 변경으로 확인된 것
- **Symbol-Breadth(풀링 단계) 구조적 게이트는 목표대로 전부 해소됨**: 6h/8h/12h/1d 전부 `sym_count` 체크를 통과함 (12h/1d는 기존 3.0 → 측정치 기반 1.0으로 완화, 8h는 3.0 → 2.0). 1h/2h/4h는 값 변화 없이 그대로 PASS — 회귀 없음.
- **그런데도 6h/8h/12h/1d는 여전히 BLOCKED** — 원인이 두 갈래로 분리됨:
  1. **`insufficient_ready_symbols`(fold-level, TF 무관 고정 raw count=2) 잔존**: 12h fold#3(LUNA2USDT, +229.12bps), 1d fold#0(JASMYUSDT, +98.50bps) 모두 이 fold-level 게이트에 걸려 pooled LCB 산출용 evidence pool에서 원천 배제됨 → `probe_lcb_bps=-inf`. 이번 작업은 **pooled 단계**(`l1_min_effective_sym_n`)만 TF-스케일링했고 **fold 단계**(`l1_min_cross_section`)는 손대지 않아 동일 유형의 장벽이 한 겹 더 있었음이 실측으로 드러남.
  2. **`registry_empty`(L0 상류 이슈)**: 6h/8h/12h/1d 전부 4개 분기 중 3개가 "0 symbols loaded" — L1 게이트 조정과 무관한 L0 admission/universe 커버리지 문제.
  3. 6h는 위 두 문제와 별개로, 실데이터가 있는 2개 fold의 gross edge 자체가 마이너스(-54.45bps, -32.96bps)로 실측되어 — 게이트 버그가 아닌 **진짜 경제적 무엣지**로 판정됨.

## 구현 반영
- [config.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/config.py) `_DEFAULT_PER_TF_GATE_OVERRIDES`에 측정 기반 `l1_min_effective_sym_n` 추가(8h=2.0, 12h=1.0, 1d=1.0), `l1_lcb_quantile_*` 4개 필드 신설.
- [evidence_policy.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/tiered_workflow/evidence_policy.py) `_resolve_lcb_quantile` 신설 — 부트스트랩 블록 수(`num_blocks`)가 적을수록 LCB quantile을 0.05→0.20으로 완화(1h/2h/4h는 no-op, 회귀 없음).
- [metrics.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/tiered_workflow/metrics.py) `resolve_num_blocks` 추출(DRY), `moving_block_bootstrap_mean` 리팩토링.
- [signal_selection.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/tiered_workflow/signal_selection.py) `_compute_pooled_probe_lcb`에 quantile 파라미터 배선.
- [src/domain/futures/strategy/calibrate_l1_symbol_breadth_gate.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/calibrate_l1_symbol_breadth_gate.py) 신규 — 측정(p10)-후-채택 분리, `logs/futures/diagnostics/l1_symbol_breadth_calibration.json`에 실측 아티팩트 기록.

## 남은 병목 (다음 스펙 후보)
1. **fold-level 심볼 다양성 게이트(`l1_min_cross_section`, raw count=2, TF 무관)도 TF-스케일링 필요** — pooled 단계와 동일한 논리를 fold 단계에 확장해야 LUNA2USDT/JASMYUSDT급 단일 심볼 초우량 신호가 evidence pool에 진입 가능.
2. **`registry_empty` 원인 규명** — 6h~1d 전부 분기의 75%가 L0 후보 자체를 못 받는 이유는 L1 게이트와 무관한 L0 universe/admission 커버리지 문제로, 별도 조사 필요.
3. **6h는 게이트가 아닌 진짜 무엣지 가능성** — gross edge 실측이 마이너스(-54bps, -33bps)로, 추가 게이트 완화로는 해결되지 않을 수 있음.
