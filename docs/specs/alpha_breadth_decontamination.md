---
title: Alpha 근본 재설계 — IS→OOS 일반화 붕괴 및 신호≪비용 진단
domain: strategy-ml
type: prd
status: proposal
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/calibrator.py
  - src/domain/futures/strategy/labels.py
  - src/domain/futures/portfolio/signal_composer.py
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/execution/opt_main_futures.py
change_triggers:
  - src/domain/futures/strategy/**
  - src/domain/futures/portfolio/**
dependencies:
  documents:
    - docs/specs/master_alpha_universe_architecture.md
    - docs/results/alphas_all.md
last_verified: 2026-05-29
---

# Alpha 근본 재설계 PRD

## 0. TL;DR (2차 검증으로 진단 전환, 2026-05-29)

1차 가설(클리핑 측정오염 / 피처-finite 상류 상한)은 **2차 실측으로 반증**되었다. `dataset.py`의 `np.all→np.any` 피처마스크 완화는 **완전 무효과**(rows=6624, breadth=1.01, SCOREBOARD 전 지표 Phase 0과 동일)였고, SCORE-IC 계측 수정 후에도 0/0/0이 유지됐다.

진짜 근본원인은 **EV-PRECLIP 분포**가 폭로했다: **(1) IS→OOS EV 크기(magnitude) 일반화가 ~15배 붕괴**하고(IS p95=75bps clip천장 → OOS p95=4.9bps, OOS 70% 음수), **(2) 최선의 OOS 신호(p95=4.9bps)조차 비용벽(24bps)의 1/5**이다. `signal_composer`가 클리핑 EV를 **선형으로** 비중화(CS rank 미사용)하므로 ≈0 EV → breadth=1로 직결. 이것은 튜닝이 아니라 **아키텍처 결함**이며, Fundamental Law상 현 OOS IC(~0.004)로는 어떤 실행 재배관도 수익을 못 만든다. **지배적 레버는 IS→OOS 일반화 갭(IC 0.014→0.004, ~3.5× 감쇠) 봉합**이다.

---

## A. 2차 검증 실측 결과 (RC-1/RC-3 수정 후)

명령: `--mode alpha --skip-data-sync --strategy ml_lambdamart_v1` (exit 0). 적용 변경: `dataset.py` finite마스크 `all→any`, SCORE-IC `<5/bar` 가드 수정, EV-PRECLIP 로그, evaluate_alpha `inference_signed_2d` 파라미터. 정적: mypy OK, pytest 31 passed.

| 측정 | 값 | 변화 |
|------|-----|------|
| SCOREBOARD (OOS) | net_ic=0.0025 t=1.07 breadth=1.01 BE=0.0506 | **Phase 0과 동일 (무변화)** |
| `ML_OOS_FILL` | rows=6624 | **동일 → 피처마스크 무효** |
| SCORE-IC | 0/0/0 breadth=3.7 | **수정 후에도 0 → OOS 횡단면 <5/bar 기아** |
| **EV-PRECLIP fold0 (IS)** | neg=43.8% p50=+4.5 p90=75 p95=75bps | IS는 clip천장 포화 |
| **EV-PRECLIP fold1 (IS)** | neg=32.9% p50=+8.5 p90=29.9 p95=33.7bps | IS p50 양수 |
| **EV-PRECLIP vrefit (OOS)** | **neg=70.4% p50=−2.0 p90=4.3 p95=4.9bps** | **OOS 크기 붕괴 + 부호반전** |

핵심 대비: **IS p95=75bps(clip) → OOS p95=4.9bps (~15× 축소)**, OOS 중앙값 −2.0bps(음수), 비용벽 24bps.

---

## B. 근본 진단 (아키텍처 결함 2종 + 측정 한계)

### B-1. IS→OOS EV magnitude 일반화 붕괴 (1차 질병)
calibrator가 IS에서 학습한 EV 크기가 OOS에서 전혀 일반화되지 않음(p95 75→4.9bps, 70% 음수). `ML_EVAL`(IS+OOS) ic=0.0138 t=4.95는 IS 적합에 오염된 지표일 뿐, OOS 진실은 SCOREBOARD t=1.07 / OOS EV 음수다. 전형적 **과적합 + 레짐 시프트**(IS 2023–25 vs OOS 2025-10~2026-03).

### B-2. 신호 ≪ 비용 + 선형 EV 비중화 (구조 결함)
`signal_composer.py:1`("Linear alpha … no CS rank") — 포지션을 클리핑 EV에 **선형 비례**로 산출. OOS p95=4.9bps ≪ 24bps 비용벽이므로 거의 모든 EV가 `max(ev,0)`+비용으로 ≈0 → breadth=1. **rank를 안 쓰고 magnitude에 의존**하는 것이 breadth 붕괴의 직접 기전.

### B-3. 반증된 1차 가설
- `dataset.py` 피처-finite(all→any) 완화: **무효** → binding은 피처 finite가 아니라 `eligible_mask`/OOS 횡단면 희소성.
- SCORE-IC 0/0/0: 수정 후에도 유지 → OOS에서 score∩signed_net_ret co-finite가 bar당 <5 → **OOS rank IC를 아직 측정조차 못함**.

### B-4. Fundamental Law 현실 점검 (냉정한 결론)
crypto 4h, OOS rank IC 프록시 ~0.002–0.004, round-trip 24bps. IR ≈ IC·√BR. IC=0.004에선 BR을 키워도(rank 포트폴리오) post-cost Sharpe가 양수가 안 됨. **실행 재배관(B/D3/D4)만으로는 불가**. IS IC(~0.014) 수준으로 OOS IC를 끌어올리는 **일반화 갭 봉합(D2)**이 유일한 지배 레버.

---

## C. 근본 해결 방향 (아키텍처)

| ID | 방향 | 공격 대상 | 비용 | 비고 |
|----|------|-----------|------|------|
| **D1** | **신뢰가능 OOS edge 측정** | B-3 측정 불능 | ~0.5일 | **전제조건.** OOS rank IC를 dense bar 한정으로 정확 산출 → 진짜 IS→OOS 갭 정량화 |
| **D2** | **일반화 갭 봉합 (1차)** | B-1 과적합/레짐 | 1~2주 | 모델 capacity↓·정규화↑, magnitude→**sign/rank 타깃 전환**, 레짐-강건 샘플가중·refit cadence, purge/embargo 감사 |
| **D3** | **rank 포트폴리오 구성** | B-2 선형 비중화 | ~3일 | 선형 EV-게이트 → **top-k long / bottom-k short (signed score 순위)**. breadth=2k 구조 보장, magnitude 붕괴에 강건 |
| **D4** | **비용벽 인하 (Maker)** | B-2 신호≪비용 | 1~2주 | round-trip 24→~4bps. 한계 OOS edge가 통과하도록 문턱 인하 (alpha0 §3) |

**권장 시퀀스:** D1 → (D3+D4 구조 개선 병행) → **D2 본연구**. **D1 직후 Go/No-Go 게이트**: 정확 측정된 OOS rank IC가 ≈0이면 "현 horizon/universe에 OOS edge 없음"으로 판정하고 **실행 배관이 아니라 신호 연구(D2/feature/regime/horizon)로 전면 전환**.

---

## D. 단계 계획

### Phase A — D1: 신뢰가능 OOS rank-IC 측정 (~0.5일, 위험 low) [선행 필수]
- **목표:** OOS 구간에서 unclipped signed score(또는 EV)의 횡단면 Spearman IC를 dense bar(co-finite≥min) 한정으로 산출. SCORE-IC가 0/0/0인 원인(co-finite<5)을 진단·표면화.
- **수정:** `ml_builder.py` SCORE-IC 블록 — score_grid∩signed_net_ret co-finite per-bar 분포 로그 추가(`p50_cofinite`, `bars_ge5_ratio`); OOS 영역(vrefit 채운 t-범위)으로 시간 슬라이스 후 `rolling_ic`. signed_net_ret이 OOS test 셀에서 finite인지 검증.
- **산출:** `🔬 [OOS-RANKIC] ic=.. t=.. n_bars=.. cofinite_p50=..` 1행. **이 수치가 D 전체 분기를 결정.**

### Phase B — Go/No-Go 판정 ✅ 확정 (2026-05-29)

**실측:** `[OOS-RANKIC] ic=0.0000 t=0.00 n_bars=1417 cofinite_p50=17.0 bars_ge5_ratio=1.000 snr_oos_finite=0.174`  
`[OOS-DIAG] cause=snr_nan_in_oos oos_bars=1417 ge5_bars=1417`

**판정: No-Go — OOS edge ≈ 0**
- 측정 조건 충분(1417 bars × 17 co-finite/bar, bars_ge5_ratio=1.000). 더 이상 아티팩트 아님.
- `ic=0.0000`의 원인: `signed_net_ret`가 OOS에서 82.6% NaN → `rolling_ic` 상수값 가드 발동. 그러나 IC-DECOMP(실제 forward return 기준)도 `c1_raw=0.0018, hit=0.061`로 독립 수렴 → **두 측정 모두 OOS rank edge ≈ 0 확인**.
- **결론: D3(rank portfolio), D4(Maker) 만으로는 수익화 불가. D2(일반화 본연구) 전면 전환.**

추가 확인 필요: `signed_net_ret` OOS 레이블 신뢰성 → D2 착수 전 forward-return 기반 OOS IC 측정 인프라 확보 권장.

### Phase C — (조건부) D3 rank 포트폴리오 (~3일)
- `signal_composer`/`risk_controls`에 top-k long/bottom-k short 모드 추가(signed score 순위 기반, magnitude 무관). breadth=2k 보장. 기존 선형모드와 A/B.

### Phase D — (조건부) D4 Maker 비용 모델 + D2 일반화 본연구
- D4: 동적 비용 floor에 Maker(post-only) 시나리오 추가, BE벽 재계산.
- D2: calibrator_target/타깃을 magnitude→sign-rank로, capacity·정규화 재튜닝, time-decay/refit cadence A/B (clean OOS IC 기준).

---

## E. 결정 확정 (2026-05-29)

1. **시작점 = Phase A(D1 측정) 선행.** OOS rank-IC 정확 산출 → Go/No-Go 후 분기. (맹목 재설계 회피)
2. **한계 edge(0.004~0.01) 시 우선 레버 = D2(일반화 본연구).** magnitude→sign/rank 타깃 전환, capacity↓, 레짐-강건. D3/D4는 구조 보강으로 병행하되, 수익화 지배 레버는 D2.

---

## F. 완료 기준 (Phase A 한정)

- [ ] **AC-1** `🔬 [OOS-RANKIC]` 로그로 OOS 한정 dense-bar rank IC(ic, t, n_bars, cofinite 분포)가 정량 출력된다.
- [ ] **AC-2** SCORE-IC 0/0/0의 원인(co-finite<5 기아 vs signed_net_ret OOS NaN)이 로그로 판별된다.
- [ ] **AC-3** Go/No-Go 판정 결과가 메모리에 기록되어 D2/D3/D4 분기를 확정한다.
- [ ] **AC-4** lint/mypy 무회귀, 신규 테스트 통과.

---

## G. Appendix — 1차 분석 (superseded, 압축 보존)

> 아래는 Phase 0 가설로, 2차 검증(§A)에서 반증/격하되었으나 추적성 위해 보존.

- **(폐기) RC-1 측정오염 주범설:** "C1 게이트가 클리핑 EV 측정 → breadth=1.01". 부분적 사실이나, signed pre-clip로 측정해도 OOS magnitude가 ≈0이라 핵심 해결 못 함. `evaluate_alpha(inference_signed_2d=...)` 파라미터는 구현됨(향후 §C에서 활용).
- **(폐기) 상류 피처-finite 상한설:** `build_long_matrix` `np.all(isfinite55)` → `np.any`로 완화했으나 **무효**. binding은 `eligible_mask`.
- **(유지) 계측 결함:** `rolling_ic`의 `<5/bar` 가드 + `score_grid` NaN-fill → SCORE-IC 0/0/0. Phase A에서 정식 해결.
- **(확정) config:** `label_horizon_bars=12`, `calibrator_target="gross"`, `ev_mode="quantile"`, `alpha_clip_bps=75`, `time_decay_halflife=1080`.
- **이미 적용된 코드 변경(2차):** `dataset.py` any-finite, `ml_builder.py` EV-PRECLIP 로그 2종 + SCORE-IC ≥5필터, `alpha_evaluation.py` inference_signed_2d/inference_stat 패널, `opt_main_futures.py` C1-STAT/C3-EXEC 출력(C1-STAT은 shape가드로 미출력), 테스트 3종.
