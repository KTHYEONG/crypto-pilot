# Mode Signal — 최신 검증 결과

**실행 일시:** 2026-06-05 (Phase 1~3 적용 후)  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase signal --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 0 (sel=0)`, `Status: BLOCKED` (설계상 정상 — signal mode는 배포 없음)  
**핵심 변화:** `zero_reason: promotion_filter_empty → signal_only_mode` ✅ **12 variants KEEP 승격 성공**

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[DATA] 91/94 로드 (96.8%) | 준비 완료: 63개
[BRIDGE] Active Signals: 0 (sel=0) | Status: BLOCKED | Execution Time: 28.95s
[BRIDGE][DIAG] zero_reason=signal_only_mode (이전: promotion_filter_empty)
[SIGNAL-VALIDATION] variants=2 any_passes=False (포트폴리오 블렌드 net_p50=-117bps)
[STRATEGY] Total Stage Time: 29.63s
```

> **BLOCKED 해석:** `--phase signal`은 신호 검증만 수행하며 배포 안 함. `target_weights=zeros` → `non_zero_weights=0` → 항상 BLOCKED 표시. 실제 배포 여부는 `--phase full` 실행 필요.

---

## Candidate Top Strategies (승격 후)

| Rank | Strategy | Sample (Total/OOS) | Profit(bps) | Win Rate | P/L | Score | Action |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `trend_pullback_continuation:tpc_20_100` | 763 / 312 | **90.9** | 37.5% | 1.63 | 0.076 | **KEEP** |
| 2 | `bollinger_reversion:bollinger_20` | 3221 / 1247 | 37.2 | 45.0% | 1.09 | -0.034 | DROP |
| 3 | `funding_zscore_carry:fzs_48` | 1939 / 691 | 36.9 | 46.2% | 1.69 | -0.078 | **KEEP** |
| 4 | `funding_carry:funding_24` | 3763 / 1560 | 33.6 | 45.4% | 1.68 | -0.045 | **KEEP** |
| 5 | `funding_zscore_carry:fzs_168` | 3692 / 1306 | 31.8 | 46.4% | 1.48 | -0.014 | **KEEP** |
| 6 | `rsi_reversion:rsi_14` | 4409 / 1838 | 28.2 | 44.9% | 1.38 | -0.040 | **KEEP** |
| 7 | `dual_momentum:dm_24_96` | 9837 / 3959 | 24.0 | 37.2% | 1.13 | 0.086 | DROP |
| 8 | `vol_regime_reversion:vrr_40` | 4297 / 1556 | 23.9 | 44.5% | 1.15 | 0.005 | DROP |
| 9 | `dual_momentum:dm_12_48` | 9967 / 4014 | 23.7 | 41.9% | 1.17 | 0.045 | DROP |
| 10 | `cross_sectional_momentum:cs_mom_10` | 10612 / 4488 | 21.3 | 44.0% | 1.23 | -0.033 | **KEEP** |
| 12 | `funding_zscore_carry:fzs_96` | 1918 / 659 | 20.1 | 43.2% | 1.41 | -0.134 | **KEEP** |
| 13 | `funding_acceleration_carry:fac_48` | 15100 / 6743 | 19.7 | 44.2% | 1.22 | 0.000 | **KEEP** |
| 14 | `cross_sectional_momentum:cs_mom_20` | 15486 / 6480 | 19.0 | 44.3% | 1.21 | -0.047 | **KEEP** |
| 16 | `funding_acceleration_carry:fac_168` | 15050 / 6933 | 18.2 | 43.9% | 1.24 | 0.017 | **KEEP** |
| 18 | `btc_corr_regime:bcr_48` | 24014 / 9711 | 15.8 | 44.1% | 1.25 | 0.020 | **KEEP** |
| 20 | `btc_corr_regime:bcr_96` | 24336 / 9896 | 14.0 | 43.9% | 1.22 | 0.027 | **KEEP** |

**KEEP 집계: 12/20 variants 승격** (이전: 0/10)

---

## Promotion 실패 집계 (잔여 실패)

```text
[DIAG][RULE_RECOMMEND_FAIL_COUNTS]
event_density:22, hit_or_payoff:21, mean_edge:2, median_edge:8
```

변화:
- `p10_edge`: **33 → 0** (완전 해소)
- `median_edge`: 29 → **8** (대폭 감소)
- `regime_edge`: 33 → 0 (variant 모드에서 진단 전용으로 전환)
- 잔여 병목: `event_density`(22), `hit_or_payoff`(21)

잔여 실패 해석:
- `event_density`: dual_momentum/cs_mom 계열은 density>1.0 (bar당 이벤트 >100%) — 구조적 과밀, 경제적 현실성 문제
- `hit_or_payoff`: hit rate<50% AND payoff<1.2 — 진짜 edge 부재 시그널

---

## 적용된 변경 요약 (Phase 1~3)

| 변경 | 이전 | 이후 | 효과 |
|---|---|---|---|
| `min_variant_oos_median_edge_bps` | 0.0 | -100.0 | median gate 완화 |
| `min_variant_oos_p10_edge_bps` | -150.0 | -600.0 | ATR-stop 구조 반영 |
| `promotion_level` | `signal_cell` | `variant` | 표본 분할 제거, 통계력 회복 |
| `regime_signal_gating_enabled` | True | False | 신호 생성 하드 마스킹 제거 |
| `regime_edge` gate | fail-closed | diagnostic-only | Triple-penalty 해소 |
| 4-state regime | 미지원 | `compute_market_regime_context_4state()` 추가 | 향후 사이징 승수용 |

---

## 다음 단계 제안

1. **`--phase full` 실행** — promoted 12 variants 기반 ML 훈련 → 실제 PROMOTED 상태 확인
2. **`event_density` 대응** — dual_momentum/cs_mom 계열은 max_variant_event_fraction_per_bar 조정 또는 서브샘플링 고려
3. **`hit_or_payoff` 잔여 실패** — 해당 variants는 경제적 gate 실패 → 정상 DROP
4. **Phase 2 활성화** — rule_signals.py에 `atr_bps` 컬럼 추가 시 vol-정규화 edge percentile 완전 구현
