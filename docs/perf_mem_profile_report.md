# Layer 1 Performance & Memory Profile Report

## 🚨 Top 5 Bottlenecks (개선 완료 포함)

1. **`bridge_run_candidate_strategy`**
   - **기존 (직렬)**: 단일 TF(4h) 기준 **46.7828s** (병목 1위)
   - **현재 (병렬)**: 타임프레임당 평균 **~21.29s** (4h TF: 52.8752s, 12h TF: 37.2970s / 수행량 4배 확대 상태)
2. **`l1_fit_inference_artifact`**
   - **소요 시간**: **20.4892s**
3. **`l1_evidence_ipc_collect`**
   - **소요 시간**: **10.9083s**
4. **`load_futures_data_maps_for_symbols`**
   - **소요 시간**: **10.5708s**
5. **`l1_prequential_evidence_snapshots`**
   - **소요 시간**: **9.7978s**

---

## 🔄 `opt_main_futures.py` 실행 흐름별 소요시간 & RAM 사용량

### 1. Data Ingestion & Setup (총 소요 시간: 약 13.55s)
- `[PERF] step=discover_universe_timeline elapsed=1.3598s`
- `[PERF] step=validate_universe_quality elapsed=0.1808s`
- `[PERF] step=load_futures_data_maps_for_symbols elapsed=10.5708s`
- `[PERF] step=inject_membership_masks_into_maps elapsed=1.4424s`

### 2. Strategy Signal Computation (총 소요 시간: 약 85.37s)
- `[PERF] tiered_align_data_maps n_syms=52 tf=4h took=0.1322s`
- `[PERF] bridge_run_candidate_strategy n_syms=52 took=85.1749s` (4개 타임프레임 누적)
- `[PERF] signal_batch_convert took=0.0707s` (n_raw=146,286 → n_out=1,803 변환)

### 3. Prequential & Walk-Forward Validation (총 소요 시간: 약 34.20s)
- `[PERF] l1_evidence_ipc_collect n=12 took=10.9083s`
- `[PERF] l1_outer_ipc_collect n=4 took=6.5459s`
- `[PERF] l1_wf_summary wall: ev=27.6s out=6.7s total=34.2s`
- `[PERF] l1_prequential_evidence_snapshots took=9.7978s`
- `[PERF] l1_evidence_phase took=20.8815s`

### 4. Promotion & Inference Serializing (총 소요 시간: 약 20.49s)
- `[PERF] l1_fit_inference_artifact took=20.4892s`
- `[PERF] l1_lifecycle n_syms=52 l1_T=3294 took=0.0004s`

### 5. Final Metrics & RAM Footprint (총 누적 시간: 163.64s / Peak RAM: 8,782MB)
- `[PERF] run_tiered_pipeline_l1_total took=163.6427s`
- `[PERF] per_tf_l1 tf=4h aligned=0.0000s folds=0.0002s run_l1=52.8752s total=52.8754s rss=5744MB`
- `[PERF] per_tf_l1 tf=12h aligned=0.0000s folds=0.0003s run_l1=37.2970s total=37.2973s rss=5756MB`
- `[SYS] [MEM] stage=aggregate_l1 rss=5756MB`
- `[SYS] [MEM] stage=l1_gate_complete rss=5756MB`
- `[SYS] [MEM] stage=tiered_pipeline rss=5756MB delta=+396MB peak=8782MB`
- `[SYS] [MEM] stage=strategy rss=4954MB delta=+2535MB peak=8782MB`
