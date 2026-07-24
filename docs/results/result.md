# Quant Multiscale Futures Engine Evaluation Report (v6.2)

**Date**: 2026-07-24
**Evaluation Horizon**: 730 days (4,380 4h bars / 17,520 1h bars), 120 Binance Perpetual Futures Symbols
**Data Integrity**: PASS (`integrity_ok: true`)

---

## 1. L1 Signal Admission — Full Catalog (730d, n_folds=5)

| Signal | Family | β mean | LCB90 | Net Mean (2x cost) | Sign Consistency | p-value | FDR q | Admitted |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| trend_ema:fast | trend_ema | 0.0580 | -130.0609 | -49.8969 | 0.50 | 0.8480 | 1.0000 | False |
| trend_ema:medium | trend_ema | 0.0417 | -54.0965 | 85.6189 | 0.25 | 0.2000 | 0.4611 | False |
| trend_ema:moderate | trend_ema | 0.0412 | -45.8651 | 217.6241 | 1.00 | 0.0020 | 0.0135 | False |
| trend_ema:slow | trend_ema | 0.0685 | -1.3531 | 148.0265 | 1.00 | 0.0000 | 0.0000 | False |
| trend_ema:very_slow | trend_ema | 0.0486 | -17.9337 | 313.4665 | 1.00 | 0.0040 | 0.0216 | False |
| momentum_ts:fast | momentum_ts | 0.1107 | -0.6831 | 1.0801 | 0.75 | 0.2220 | 0.4611 | False |
| momentum_ts:medium | momentum_ts | 0.0618 | -0.8502 | 1.5841 | 0.75 | 0.1700 | 0.4611 | False |
| momentum_ts:moderate | momentum_ts | -0.0006 | -1.0246 | 1.1457 | 0.50 | 0.2680 | 0.4824 | False |
| momentum_ts:slow | momentum_ts | -0.1549 | -1.8261 | 1.7699 | 0.25 | 0.3440 | 0.5559 | False |
| momentum_ts:very_slow | momentum_ts | 0.0886 | -2.9980 | 2.9883 | 0.75 | 0.3500 | 0.5559 | False |
| breakout_donchian:fast | breakout_donchian | 0.0568 | -0.0302 | 0.0297 | 0.75 | 0.2160 | 0.4611 | False |
| breakout_donchian:medium | breakout_donchian | 0.0216 | -0.1688 | -0.0199 | 0.25 | 0.5820 | 0.8730 | False |
| breakout_donchian:moderate | breakout_donchian | -0.1027 | -0.2377 | 0.1733 | 0.50 | 0.2500 | 0.4821 | False |
| breakout_donchian:slow | breakout_donchian | -0.0148 | -1.0698 | -0.5218 | 0.50 | 0.9440 | 1.0000 | False |
| breakout_donchian:very_slow | breakout_donchian | 0.0005 | -2.7109 | -0.9585 | 0.50 | 0.8020 | 1.0000 | False |
| basis_gap:fast | basis_gap | 0.2671 | -0.0307 | 0.0729 | 0.75 | 0.1120 | 0.3780 | False |
| basis_gap:medium | basis_gap | 0.2991 | -0.0892 | 0.0970 | 0.50 | 0.1980 | 0.4611 | False |
| basis_gap:moderate | basis_gap | 0.4239 | -0.0098 | 0.3268 | 0.75 | 0.0400 | 0.1543 | False |
| basis_gap:slow | basis_gap | 0.5186 | -317.3018 | -132.8601 | 0.50 | 0.8440 | 1.0000 | False |
| basis_gap:very_slow | basis_gap | 0.9872 | -743.7990 | -321.4381 | 0.50 | 0.8780 | 1.0000 | False |
| reversal_st:fast | reversal_st | -0.0947 | -1.7768 | -0.6741 | 0.75 | 0.7120 | 1.0000 | False |
| **xs_reversal:fast** | xs_reversal | 0.0011 | **+0.0117** | 0.0276 | 1.00 | 0.0020 | 0.0135 | **True** |
| xs_reversal:medium | xs_reversal | -0.0063 | -0.0936 | -0.0497 | 0.50 | 0.9580 | 1.0000 | False |
| **xs_momentum_slow:slow** | xs_momentum_slow | 0.1330 | **+0.1391** | 0.2705 | 0.75 | 0.0000 | 0.0000 | **True** |
| xs_momentum_slow:very_slow | xs_momentum_slow | 0.1318 | +0.0048 | 0.2737 | 0.75 | 0.0340 | 0.1530 | False |
| smart_money_divergence:fast | smart_money_divergence | 0.0357 | -0.0454 | -0.0115 | 0.25 | 0.6660 | 0.9464 | False |
| smart_money_divergence:medium | smart_money_divergence | 0.0690 | -0.1724 | -0.0611 | 0.25 | 0.8100 | 0.9482 | False |

**Admitted (2/26)**: `xs_reversal:fast`, `xs_momentum_slow:slow`

---

## 1b. Composite Admission (L1→L2 재설계 실측, 2026-07-24)

개별 signal 이진 admission 게이트(위 표)가 근본 병목이라는 진단(SSOT: `docs/decisions/decisions.md` ADR_20260724_L1L2_COMPOSITE_ADMISSION)에 따라, 약필터(sign_consistency≥0.5 & p≤0.5) 통과 후보를 fold별 precision(1/se²) 가중으로 결합한 단일 composite에 대해 admission을 재정의했다. `engine.py`/`ladder.py`는 이미 이 composite 게이트로 전환됐다.

| Metric | Value |
| :--- | ---: |
| 결합 방식 | Precision-weighted (fold별 1/beta_se²) |
| 후보 수 (약필터 통과) | 16 / 27 |
| Composite LCB90 | -0.0286 |
| Composite net_mean_2x | +0.0347 |
| Composite sign_consistency | 0.500 (문턱 0.6 미달) |
| Composite p-value | 0.202 |
| **Composite admitted** | **False** |

**L2/L3 impact**: composite 미채택 → cash-only fallback. L2 log growth **0.000**(무포지션, v6.1 baseline -0.3836 대비 손실은 회피했으나 수익도 없음), L3 verdict **REJECT**(v6.1 SHADOW 대비 악화).

**결론**: composite 아키텍처는 코드적으로 정상 동작(NaN 마스킹 버그 수정 후 실측치 확인)하나, 이번 730d 데이터에서는 admission 문턱을 근소하게 넘지 못했다. 사전 Stouffer 메타분석(Z=7.94, 독립가정)이 과대추정한 원인은 후보 신호(특히 trend_ema 5-speed 등)간 강한 상관 — Grinold-Kahn 유효breadth(`N/(1+(N-1)ρ)`)가 명목 16개보다 크게 축소됨. 현재 실거동은 admitted 신호 카운트(2/26, 위 표)와 무관하게 **cash-only**이며, composite가 문턱을 통과하지 못하는 한 프로덕션 배포 신호셋 교체는 일어나지 않는다. SSOT: 스펙은 `docs/decisions/decisions.md`(sync 시 `docs/specs/`에서 제거됨) 참조.

---

## 2. L1 Cross-Sectional Rank-IC (Newey-West HAC, 신규 신호 candidate screening)

| Signal (candidate) | Window | meanIC @24h | t_NW @24h | meanIC @72h | t_NW @72h |
| :--- | :--- | ---: | ---: | ---: | ---: |
| oi_momentum(72h lb) | 205d | +0.0141 | +2.08 | +0.0155 | +1.51 |
| oi_momentum(72h lb) | **730d** | +0.0041 | +0.94 | +0.0080 | +1.23 |
| oi_confirmed_price_momentum(24h) | 205d | +0.0052 | +0.51 | — | — |
| oi_confirmed_price_momentum(24h) | **730d** | -0.0326 | -5.27 | — | — |
| plain_price_momentum(24h) baseline | 730d | -0.0339 | -6.30 | — | — |
| retail_LSR_contrarian | 205d | +0.0081 | +1.07 | +0.0163 | +1.32 |
| retail_LSR_contrarian | **730d** | -0.0031 | -0.61 | +0.0013 | +0.16 |
| smart_money_divergence(top-trader−retail) | 205d | +0.0063 | +0.93 | +0.0129 | +1.20 |
| smart_money_divergence(top-trader−retail) | **730d** | **-0.0156** | **-3.53** | **-0.0179** | **-2.44** |
| taker_LS_vol_ratio(level EWM6) | 205d/730d | -0.0022/-0.0060 | -0.40/-1.63 | — | — |
| taker_LS_vol_ratio(momentum) | 205d/730d | +0.0024/+0.0012 | +0.46/+0.36 | — | — |
| oi_momentum_extreme_contrarian | 205d/730d | -0.0123/+0.0030 | -2.13/+0.75 | — | — |
| xs_admitted_baseline(reference) | 730d | +0.0168 | +2.69 | — | — |

**smart_money_divergence split-half robustness (730d)**: 전반부(oos0~mid) n=1437 meanIC=-0.0113 t_NW=-1.76 / 후반부(mid~dev_end) n=1437 meanIC=-0.0196 t_NW=-3.26

**Orthogonality**: oi_momentum(72h) vs price_momentum(72h) XS-rank corr = +0.075~+0.076 (양쪽 window 동일)

---

## 3. metrics_5m Data Coverage

| | 백필 전 | 백필 후 |
| :--- | ---: | ---: |
| 파티션 수 (전체 120종목) | 823 | 2,196 |
| 평균 월별 파티션/종목 | 6.8 | 18.1 |
| 저장 용량 | 399 MB | 1.1 GB |
| BTCUSDT 연속 구간 | 2024-07 + 2026-01~07 (공백) | 2024-07~2026-07 (25개월 연속) |
| 체크섬 오류 | — | 0건 |
| null rate (OI / long_short_ratio) | — | 0.0% / 0.01% |

---

## 4. L2 Portfolio Performance

| Metric | Value |
| :--- | ---: |
| Net Log Growth Rate ($g$) | -0.3836 |
| Equity Multiple | 0.8655 |
| Maximum Drawdown | -16.55% |
| Annualized Volatility | 15.99% |
| Daily CVaR (95%) | -0.42% |
| Turnover | 0.0244 |
| Sizing formula | `0.20 · mu/σ_price + 15% vol target`, gross cap 1.0x |

## 5. L3 Sealed Holdout Gate

| Metric | Value |
| :--- | ---: |
| Verdict | SHADOW |
| Posterior Growth Probability | 0.6352 |
| Promote Threshold | 0.65 |
| Holdout Days | 180 |
| Max Drawdown (holdout) | -6.98% |
| Daily CVaR95 (holdout) | -0.65% |

---

## 6. L2 Sizing Comparison (v6 vs v6.1, 730d dev window)

| Sizing | Log Growth | MDD | Ann. Vol | Gross Leverage |
| :--- | ---: | ---: | ---: | ---: |
| v6 dyn-Kelly (epistemic var, 2.0x) | -6.90 | 100.0% | 102.8% | 2.00 |
| v5 quarter-Kelly (epistemic var, 1.0x) | -3.14 | 98.4% | 45.6% | 1.00 |
| price-risk Kelly f=0.20 + 15% vol target (v6.1, production) | -0.384 | 16.5% | 16.0% | 1.00 |
