# Quant Multiscale Futures Engine Evaluation Report (v6.1 Price-Risk Sizing)

**Date**: 2026-07-24
**Evaluation Horizon**: 730 days (4,380 4-hour bars / 17,520 1-hour bars), 120 Binance Perpetual Futures Symbols (PIT Causal Data)
**Execution Mode**: Full Multiscale Pipeline (`run_multiscale_compound_engine`, direct invocation — CLI `if __name__` guard missing)
**Data Integrity**: PASS (`integrity_ok: true`)
**Prior Baseline**: v6 Dynamic Kelly (182-day eval) — Log Growth **-3.05**, MDD **-71.6%**, L3 **REJECT**

---

## 1. Executive Summary & Level-by-Level Breakdown

| Pipeline Layer | Evaluation Scope & Strategy Config | Core Metric / Finding | Verdict & Status |
| :--- | :--- | :--- | :--- |
| **Level 1 (L1 Alpha Bank)** | 730d admission (n_folds=5, purge/embargo dynamic), 2 admitted signals (`xs_reversal:fast`, `xs_momentum_slow:slow`) | XS rank IC +0.0170, **Newey-West t=+2.71** (182d 창에서는 동일 신호 IC 음수 — 검정력 부족) | PASS (통계적으로 유의, 182d 대비 개선) |
| **Level 2 (L2 Allocation)** | Price-risk Kelly sizing (`f=0.20 · mu/σ_price`), causal 15% vol target, gross cap 1.0x | **Log Growth: -0.384**, MDD **-16.5%**, 연변동성 **16.0%** | 파산 방어 성공, 순수익은 미약한 음수 |
| **Level 3 (L3 Validation)** | Sealed Holdout Gate (180-day), 신선 소비 확인(05:34:05 생성→05:35:54 소비, 캐시 재사용 아님) | Posterior growth prob **0.635**, MDD **-7.0%** | **SHADOW** (REJECT 탈출, promote 문턱 0.65 미달) |

---

## 2. 근본 원인 진단: v6 파산의 주범

유저 가설("신호 SNR 부족")은 부분적으로만 사실이었음. 실측 결과 파산의 진짜 주범은 **L2 사이징 결함**이었고, L1 검정력 부족은 부차적 요인이었다.

### 2.1 사이징 결함 (주범, [HYP-L2-A/B] vs [HYP-L2-C])
- v6/v5 공통 결함: `w ∝ f·mu/se²`에서 `se`는 family 간 forecast 분산(epistemic uncertainty)이며 **가격 변동성이 아님**. 여러 family가 의견 합치 시 se→0 → 웨이트가 캡까지 폭주 (실측 연변동성 89.8%).
- 올바른 Kelly 등가식: mu는 이미 vol-normalized 단위이므로 `w ∝ f·mu/σ_price`.
- **동일 mu, 사이징만 교체한 730d dev 실측**: log growth **-6.90 (v6 재현) / -3.14 (v5 quarter-Kelly, 동일 결함) → +0.265 (price-risk + 15% vol target)**, MDD 100%/98% → 9.7%.
- 프로덕션 전체 파이프라인 실행(비용·funding·5-fold admission 전부 반영) 결과는 dev 스크립트 근사치보다 보수적: log growth -0.384, MDD -16.5% — **파산은 확실히 해소**되었으나 **비용 차감 후 순수익은 아직 미약한 음수**.

### 2.2 L1 검정력 부족 (부차 요인)
- 182일 평가 창에서는 admitted 신호(xs_reversal:fast)의 dev IC가 **음수**(-0.0123, t=-1.71)로 뒤집힘 — 노이즈를 신호로 오인.
- 730일로 확장 시 admitted 2개 신호 IC **+0.0170, Newey-West HAC 보정 t=+2.71**로 유의. → **admission 평가 창은 730일 고정 필수**.
- 겹침 horizon(216h+) 신호의 t-stat은 NW 보정 없이 과대평가됨(t_naive +2.4 → t_NW +0.5 소멸) — HAC 보정 없는 admission gate는 위험.

### 2.3 기각된 대안 가설
| 가설 | 결과 | 판정 |
| :--- | :--- | :--- |
| top-8 q-shrunk 앙상블 | IC -0.0005, t=-0.10 | 기각 (희석) |
| slow-only(≥216h) 앙상블 | IC -0.0036~-0.0063, t<1 | 기각 |
| causal rolling-IC 게이트 | IC +0.010, t=+0.93 | 기각 (개선 없음) |
| SNR-조건부 Kelly f 스케일링 | g -0.06~-0.07 (기준 대비 악화) | 기각 (자유도 추가=과적합) |

프로덕션 결함 추가 발견: `combine_admitted_forecasts`가 fold0 OOS 시작 이전 구간에 미래 β를 그대로 적용하는 **pre-OOS look-ahead 누수**. 마스킹 적용 후 182d slow-ensemble 실측 g +0.597(누수 포함) → -0.118(제거)로 정정 — 기존 앙상블 우위 판단은 이 누수의 산물이었음.

---

## 3. Level 2 (L2) Portfolio Performance Matrix

| Metric Parameter | v6 (Dynamic Kelly, epistemic var, REJECT) | **v6.1 (Price-Risk Sizing)** |
| :--- | :--- | :--- |
| **Net Log Growth Rate ($g$)** | -3.0470 | **-0.3836** |
| **Equity Multiple** | 0.3176 (-68.24%) | **0.8655 (-13.4%)** |
| **Maximum Drawdown (MDD)** | -71.60% | **-16.55%** |
| **Annualized Volatility** | 89.80% | **15.99%** |
| **Daily CVaR (95%)** | -1.75% | **-0.42%** |
| **Sizing formula** | `f·mu/se²_epistemic`, gross 2.0x | `0.20·mu/σ_price + 15% vol target`, gross 1.0x |

## 4. Level 3 (L3) Gate Verdict & Next Action

- **L3 Deployment Verdict**: **SHADOW** (REJECT 탈출; promote_probability 0.65 문턱 대비 0.635로 미달)
- **Holdout 소비 무결성**: `data_manifest_hash`/`strategy_spec_hash` 매칭 확인, 05:34:05 최초 생성 → 05:35:54 최초 소비 — 과거 v6 캐시 재사용 아닌 이번 코드 기준 신선 평가.
- **Architectural Next Steps**:
  1. **L1 알파 원천 재탐색이 최우선**: 현재 admitted 신호(IC +0.017)는 비용을 겨우 상쇄하는 수준. 앙상블/조건화로는 개선 안 됨(§2.3 전량 기각) — 신규 signal family 또는 microstructure/horizon 재설계 필요.
  2. Promote 문턱(0.635 vs 0.65) 근접 — L1 IC를 소폭만 개선해도 SHADOW→PROMOTE 전환 가능성.
  3. CLI 진입점 `run_multiscale_cli`에 `if __name__ == "__main__": main()` 가드 누락 확인 — `python -m` 직접 실행 불가 결함, 별도 수정 필요(범위 외).

---

## 5. Spec/구현 계보

- Spec: `docs/specs/l1l2_price_risk_sizing.md` (+ `_contract.json`) — 730d L1 admission 고정 + L2 가격리스크 사이징 전환.
- 구현: `src/domain/futures/compound/{allocator,admission,config,engine}.py`, 테스트 5개 시나리오(`test_admission.py`, `test_dynamic_compounding.py`, `test_engine.py`) 전량 PASS, Cov 88%.
- 실험 스크립트(비영구): `scratch/verify_l1_ensemble.py`, `verify_l1_nw.py`, `verify_l2_sizing.py`, `verify_730_full.py`, `prep_forecast_cache{,_730}.py`.
