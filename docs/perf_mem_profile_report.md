# L1 Pipeline Performance Profile (2026-06-24)

## Environment
- **Host RAM**: 32 GB / **WSL2 RAM**: 18 GB / **WSL2 Swap**: 20 GB
- **CPU**: 8 physical cores
- **Pipeline**: `--phase l1` / 52 symbols / 8,761 bars (4h base)
- **Command**: `uv run python src/execution/opt_main_futures.py --phase l1 --timeframe 4h --sync skip --date 2026-06-24`

---

## 1. Time Breakdown

### 1.1 Overall Pipeline (L1)

| Phase | Duration | Share |
|---|---|---|
| Universe Discovery | 1.92s | 0.7% |
| Data Load (57 → 52 syms) | **16.90s** | 5.9% |
| Bridge Candidate Strategy | **58.24s** | 20.2% |
| Tiered Pipeline (L1 Nested SWF) | **~211s** | 73.2% |
| **Total** | **288.10s** | 100% |

### 1.2 Data Load Detail (16.90s)

| Sub-operation | Duration |
|---|---|
| `load_futures_data_maps_for_symbols` (57 symbols) | 14.94s |
| `inject_membership_masks_into_maps` | 1.66s |
| `evaluate_data_readiness` | 0.27s |

### 1.3 Bridge Detail (58.24s)

| Phase | Duration | % of Bridge |
|---|---|---|
| **Align** (base TF 4h) | 0.14s | 0.2% |
| **Rules** (base TF rule signal panels) | 3.83s | 6.6% |
| **Events** (base TF candidate_panels_to_events) | 9.39s | 16.1% |
| **Label** (base TF event labeling) | 1.66s | 2.9% |
| **Htf Panels** (build_multi_tf_panels: 3 TFs, THREADED) | **9.94s** | 17.1% |
| **Htf Label** (HTF event labeling) | 4.17s | 7.2% |
| **Htf Events** (HTF candidate_panels_to_events + concat) | 22.74s | 39.0% |
| **Diagnostics** (rule diagnostics) | **8.79s** | 15.1% |
| **Promotions** (variant promotion filter) | 0.34s | 0.6% |
| **Alpha Panel** (zero-weight for signal_only) | 0.12s | 0.2% |
| **Walk Forward** (skipped — signal_only mode) | 0.00s | — |

Total Bridge HTF block (Panels + Label + Events): **22.74s** (39.0%)

### 1.4 Tiered Pipeline Detail (~211s)

| Sub-stage | Duration |
|---|---|
| Tiered data alignment | 0.18s |
| L1 Nested SWF (4 TFs × 4 folds) | ~211s |
| └─ Inner per-TF processing (3 non-base TFs) | Included above |

---

## 2. Memory Profile

### 2.1 Stage-by-Stage RSS (L1)

| Stage | RSS | Delta | Peak | Description |
|---|---|---|---|---|
| Universe | 403MB | +84MB | 405MB | Discovery + quality |
| Data (Parquet load) | 2,403MB | +1,999MB | 2,410MB | Raw OHLCV DataFrames (57 syms) |
| Data Early Release | 2,404MB | +0MB | — | data_maps released (kept filtered) |
| **Bridge** | 5,863MB→**7,282MB** | +3,949MB | **7,282MB** | Aligned + panels + HTF block |
| Tiered Pipeline | 5,783MB→**7,565MB** | +1,782MB | **7,565MB** | Peak: nested SWF + aligned |
| **Strategy End** | **4,834MB** | -2,731MB | — | Post-GC |

### 2.2 Bridge Stage RSS Deltas (top 5)

| Stage | Delta |
|---|---|
| Diagnostics | +4,880 MB |
| Promotions | +4,795 MB |
| Alpha Panel | +4,794 MB |
| Htf Events | +3,941 MB |

---

## 3. Bottleneck Analysis

| Rank | Bottleneck | Time | RSS Impact | Notes |
|---|---|---|---|---|
| #1 | **Tiered Pipeline (L1 Nested SWF)** | ~211s (73%) | Peak 7,565MB | 4 TFs × 4 folds × per-symbol SWF |
| #2 | **Bridge HTF Events** | 22.74s (7.9%) | +3,941MB | Multi-TF panel projection + concat |
| #3 | **Bridge Diagnostics** | 8.79s (3.1%) | +4,880MB | `compute_rule_diagnostics` on 52 syms |
| #4 | **Data Load (Parquet I/O)** | 14.94s (5.2%) | +1,999MB | 57 syms × 6 timeframes × enriched parquet |

HTF Panels (`build_multi_tf_panels`) 9.94s: now **ThreadPoolExecutor(max_workers=2)** applied (3 TFs → ~1.5 TF-equiv). Expected saving vs sequential: ~4-5s.
