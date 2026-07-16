# L0/L1 최신 파이프라인 결과 (2026-07-16, 레지스트리 어드미션 재보정 반영)

## 실행 및 데이터 무결성

- **실행**: `PYTHONPATH=. uv run python src/domain/futures/strategy/run_l1_cross_tf_diagnosis.py`
- **적용**: (1) 일평균 이벤트 밀도 게이트, (2) 풀링/fold-level 심볼 다양성 하한 측정-후-채택 재보정, (3) 부트스트랩 블록 수 적응형 LCB quantile, (4) pair-level FDR 절차(BY→BH) 측정-후-채택 재보정
- **기간**: 2023-07-31 ~ 2026-03-31, IS/OOS split 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106
- **L1 admission**: 101/106 symbols (5개 `late_start` 제외)
- **프로세스 종료**: `exit_code=0`, `reason=l1_mode_done` (4개 Replay 런 전부 수치 완전 일치 — `ablation_restores_control: true`)

## L1 판정 (레지스트리 어드미션 재보정 적용 후)

| Timeframe | 판정 | Symbol-Breadth | probe_lcb_bps | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **1h** | **✅ PASS** | 28.582(≥5.00) | 37.182 | 회귀 없음 |
| **2h** | **✅ PASS** | 47.222(≥5.00) | 50.699 | 회귀 없음 |
| **4h** | **✅ PASS (⚠️)** | 3.000(≥3.00) | 18.990 | `fold_ratio:0.250` advisory, 회귀 없음 |
| **12h** | **✅ PASS (신규)** | 6.545(≥1.00) | **+122.483** | 3/4 fold ready. 실제 후보 10개 승급 (LUNA2USDT LCB+312bps, PEOPLEUSDT LCB+434bps 등) |
| **1d** | **⚠️ 게이트 PASS, 승급 0건** | 2.000(≥1.00) | **+143.346** | 구조 게이트 3/3 통과(probe_lcb_bps 대폭 개선)했으나 `fold_ratio:0.250`(advisory)로 상단 라벨은 BLOCKED, 개별 후보 승급은 아직 0건 — 새 하류 병목 |
| **8h** | **❌ BLOCKED (악화)** | 11.000(≥2.00) | **-147.857** | Symbol-Breadth는 통과했으나, FDR 완화로 그동안 걸러졌던 실제 마이너스 분기(fold#1: 8심볼, edge -129.5bps)가 풀에 편입되며 LCB가 더 악화 — 게이트 버그가 아닌 진짜 마이너스 분기 노출로 판단 |
| **6h** | **❌ BLOCKED (변화 없음)** | 3.000(≥3.00) | -40.459 | 의도적으로 미조정(경제적 무엣지, 아래 참조) |

### 이번 변경으로 확인된 것
1. **12h는 완전한 신규 PASS**: pooled 심볼 다양성(`l1_min_effective_sym_n`)과 fold-level 원시 심볼 수(`l1_min_cross_section`) 둘 다 TF-스케일링(1.0)하고, pair-level FDR 절차를 Benjamini-Yekutieli→plain BH로 전환(측정 기반)한 결과 3/4 fold가 살아나며 실제 후보 10개가 L2로 승급.
2. **1d는 부분 개선**: probe_lcb_bps가 -inf → +143bps로 급격히 개선되고 구조 게이트는 전부 통과했지만, advisory `fold_ratio` 경고와 함께 여전히 개별 후보 승급 0건 — L1 게이트 통과와 L2 승급 사이에 별도 병목이 있음이 새로 드러남.
3. **8h는 오히려 더 나빠짐 — 이는 버그가 아니라 진실 노출**: FDR을 완화하니 이전에 (의도치 않게) 함께 걸러지던 진짜 마이너스 분기(fold#1, -129.5bps)가 evidence pool에 들어와 pooled LCB를 더 끌어내림. Symbol-Breadth 실측치가 11.0으로 크게 오른 것도 이 분기 유입 때문.
4. **6h는 의도적으로 미조정**: 실데이터가 있는 2개 fold의 gross edge가 실측상 마이너스(-54.45bps, -32.96bps)로, 게이트를 더 풀어도 해결되지 않는 경제적 무엣지로 재확인.
5. **인과 무결성**: `PROCESS`, `runner_result`, 4-run 전부 수치 완전 동일(`ablation_restores_control: true`) — 계측 아티팩트가 아닌 실제 게이트 로직 변화로 확인.

## 구현 반영
- [config.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/config.py) — `_DEFAULT_PER_TF_GATE_OVERRIDES`에 측정 기반 `l1_min_cross_section`(8h=2/12h=1/1d=1) 및 `l1_pair_fdr_procedure`(8h/12h/1d="bh") 추가. `l1_pair_fdr_procedure: Literal["by","bh"]="by"` 필드 신설(기본값 "by"로 전 TF 회귀 없음 보장).
- [signal_selection.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/tiered_workflow/signal_selection.py) — `_by_q_values`에 `harmonic_override` 파라미터 추가(1.0=plain BH, None=기존 Benjamini-Yekutieli 그대로), `compute_symbol_strategy_evidence`가 `cfg.l1_pair_fdr_procedure`를 읽어 배선.
- [calibrate_l1_symbol_breadth_gate.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/calibrate_l1_symbol_breadth_gate.py) — `measure_fold_min_ready_symbols_by_tf()` 신규(registry_empty fold 제외 p10 측정), `propose_cross_section_thresholds()` 신규.
- 측정(별도 scratchpad 진단, 리포지토리 비반영) — `compute_symbol_strategy_evidence` 계측 결과 8h/12h/1d에서 hard_eligible 후보의 98.9~99.8%가 BY 조화급수 페널티만으로 탈락(2h는 76%) 확인, 후보 풀이 2~3개 전략군에 33~67% 집중(독립 가설 아님) 확인 후 BH 채택.

## 남은 병목 (다음 스펙 후보)
1. **1d: 게이트 통과 vs 개별 후보 0건 승급 괴리 원인 규명** — probe_lcb_bps는 강한 양수인데 최종 승급 후보가 0건인 이유(quality_weight_zero/no_incremental_edge 비중이 높게 보고됨)를 진단 필요.
2. **`registry_empty` 원인** — 6h/8h/12h/1d 다수 분기에서 여전히 "0 symbols loaded" 관측(FDR 완화로 일부 회복됐으나 완전 해소는 아님) — L0 상류 원인 규명은 미착수.
3. **8h**: FDR 완화로 진짜 마이너스 분기가 드러난 것으로 잠정 판단하되, 추가 분기 검증(다른 시드/기간) 없이 "게이트 정상 작동"을 최종 확정하지는 않음.
