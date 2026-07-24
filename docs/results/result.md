# L1/L2 Pipeline 실측 결과 (2026-07-24 11:17 fresh run)

## 배경

메인 파이프라인(`compound_main.py`) 매 실행 `no_admissible_alpha` 종료. L1 forecast calibration + L2 convex 최적화 재정의(P1~P3)를 완료하고, v3 아키텍처(다중 horizon term structure)로 4회 실측 라운드 후 현재 신호셋 평가를 최종 확정.

## 최종 아키텍처

- **P1 signal_bank.py**: 1h→4h bar 집계, 25개 신호. 5 families×4 + reversal_st + xs_reversal(fast/medium) + xs_momentum_slow(slow/very_slow). SignalDescriptor별 `target_horizon_hours`(4/216/648).
- **P2 calibration + admission**: 다중 horizon calibration. pooled ridge β, 블록 부트스트랩 LCB90, 2배비용, fold sign consistency, BH-FDR. `low_effective_sample` soft-flag.
- **P3 ladder.py**: L1-0~L1-3 × L2-0~L2-1 8-스테이지.

## --phase ladder 실측 (4380 bars, 120 symbols, seed=42)

### 8-스테이지 결과

| 스테이지 | growth | lcb90 | Sharpe | MDD | turnover | 2x growth | 승격 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| L1-0\|L2-0 | -0.148 | -1.148 | 0.31 | -69% | 186 | -0.368 | ✗ |
| L1-0\|L2-1 | +0.063 | -0.210 | 0.38 | -28% | 208 | -0.183 | ✗ |
| L1-1\|L2-0 | +0.190 | -0.806 | 0.66 | -60% | 356 | -0.232 | ✗ |
| L1-1\|L2-1 | +0.161 | -0.113 | 0.80 | -20% | 346 | -0.247 | ✗ |
| L1-2\|L2-0 | +0.175 | -0.802 | 0.64 | -65% | 675 | -0.624 | ✗ |
| L1-2\|L2-1 | -0.042 | -0.299 | -0.06 | -36% | 635 | -0.793 | ✗ |
| L1-3\|L2-0 | 0.0 | 0.0 | 0.0 | 0% | 0 | 0.0 | ✗ |
| L1-3\|L2-1 | 0.0 | 0.0 | 0.0 | 0% | 0 | 0.0 | ✗ |

8/8 ok, **0 promoted**. L1-3 25개 신호 전량 미채택 → zero-mu fallback.

### P2 admission 상세 (25개 신호 pre-BH-FDR)

| signal_id | beta_mean | lcb90 | net_mean_2x | sign_cons | p | q | admitted |
|---|---|---|---:|---:|---:|---:|:---:|
| trend_ema:fast | 0.0131 | -15.009 | 12.124 | 0.4 | 0.26 | 0.26 | ✗ |
| trend_ema:medium | 0.0046 | -9.470 | 5.486 | 0.4 | 0.29 | 0.29 | ✗ |
| trend_ema:slow | 0.0024 | -10.731 | -2.112 | 0.2 | 0.70 | 0.70 | ✗ |
| trend_ema:very_slow | 0.0012 | -18.208 | -3.964 | 0.2 | 0.68 | 0.68 | ✗ |
| momentum_ts:fast | 0.0437 | -0.431 | -0.165 | 0.2 | 0.74 | 0.74 | ✗ |
| momentum_ts:medium | 0.0668 | -0.299 | -0.127 | 0.4 | 0.83 | 0.83 | ✗ |
| momentum_ts:slow | 0.0609 | -0.156 | -0.050 | 0.4 | 0.61 | 0.61 | ✗ |
| momentum_ts:very_slow | 0.0398 | -0.196 | -0.074 | 0.6 | 0.69 | 0.69 | ✗ |
| breakout_donchian:fast | 0.0240 | -0.002 | 0.009 | 0.6 | 0.06 | 0.06 | ✗ |
| breakout_donchian:medium | 0.0322 | -0.002 | 0.010 | 0.8 | 0.07 | 0.07 | ✗ |
| breakout_donchian:slow | 0.0247 | -0.001 | 0.010 | 0.8 | 0.05 | 0.05 | ✗ |
| breakout_donchian:very_slow | 0.0181 | -0.011 | 0.004 | 0.8 | 0.28 | 0.28 | ✗ |
| carry_funding:fast | 0.0069 | -0.007 | 0.002 | 0.4 | 0.32 | 0.32 | ✗ |
| carry_funding:medium | 0.0072 | -0.007 | 0.005 | 0.4 | 0.22 | 0.22 | ✗ |
| carry_funding:slow | 0.0072 | -0.006 | 0.005 | 0.4 | 0.22 | 0.22 | ✗ |
| carry_funding:very_slow | 0.0061 | -0.007 | 0.006 | 0.6 | 0.23 | 0.23 | ✗ |
| basis_gap:fast | 0.0078 | -0.005 | 0.005 | 0.4 | 0.19 | 0.19 | ✗ |
| basis_gap:medium | 0.0072 | -0.004 | 0.005 | 0.4 | 0.16 | 0.16 | ✗ |
| basis_gap:slow | 0.0071 | -0.004 | 0.005 | 0.4 | 0.17 | 0.17 | ✗ |
| basis_gap:very_slow | 0.0078 | -0.130 | -0.041 | 0.4 | 0.60 | 0.60 | ✗ |
| reversal_st:fast | -0.0162 | -0.272 | -0.093 | 0.2 | 0.61 | 0.61 | ✗ |
| xs_reversal:fast | -0.0814 | -0.007 | -0.005 | 0.6 | 0.94 | 0.94 | ✗ |
| xs_reversal:medium | -0.0268 | -0.004 | -0.002 | 0.2 | 0.64 | 0.64 | ✗ |
| **xs_momentum_slow:slow** | **0.0883** | **0.040** | **0.175** | **0.4** | **0.0000** | **0.0000** | **✗** |
| xs_momentum_slow:very_slow | -0.0035 | -0.498 | -0.070 | 0.4 | 0.53 | 0.53 | ✗ |

### 주요 관찰

- **xs_momentum_slow:slow** (lookback=216h, target=216h): **유일하게 economic gate 통과** (lcb90=0.04>0, net_mean_2x=0.175>0, p=0.0000). 그러나 **sign_consistency=0.4 < 0.6**(5 fold 중 2 fold만 양수)로 rejected. scratch IC t=+19.01의 edge가 P2 admission에서 부분 재현됐으나 gate에서 탈락.
- **xs_momentum_slow:very_slow** (648h): `low_effective_sample` flag (n_effective=20.6<50). beta=-0.0035, 무의미.
- **breakout_donchian:slow/medium**: p=0.05~0.07로 근접했으나 lcb90이 0에 걸쳐 BH-FDR 탈락. sign_consistency 0.8로 가장 안정적.
- **xs_reversal**: scratch t=11.17(8h)~9.14(24h)에도 P2에서는 p=0.64~0.94. 신호 자체가 아니라 OOS 기간·비용 구조에서 edge 소멸로 추정.

## 결론

| 항목 | 판정 |
|---|---|
| 자산배분(L2) 결함 | 수정 완료 (gross cap 버그) |
| 신호(L1) edge | **25개 전부 BH-FDR 탈락** |
| xs_momentum_slow:slow | 경제적 gate 통과 but sign consistency gate 탈락 |

`xs_momentum_slow:slow`가 개별 통계량(p=0.0000)으로는 강력한 증거를 보였으나, 5개 fold 중 2개만 양수인 불안정성이 admission을 막았다. 이는 scratch IC(t=19)와 일관되지만 유효표본(~20개 OOS 구간)의 노이즈가 fold 분할에서 일관된 부호를 보장하지 못한 것으로 해석됨.

## 잔여 기술 부채

- `CompoundEngineResult.handoff` → `AlphaEventTape` 타입. 별도 논의.
- `alpha_catalog.py` 미참조. 삭제 예정.

## 다음 단계

1. P4/P5 (ensemble/robust optimizer) — 신호 edge 부재로 우선순위 낮음.
2. 새 신호군 탐색 — 가장 유력.
3. 현 결론 확정 기록.
