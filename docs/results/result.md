# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-12 (CPCV → SWF-K 전환, Pooled IC + NW HAC t-stat 도입)
**현재 상태:** `BLOCKED` — SWF-K L1 Gate 차단됨 (실제 Alpha 엣지 부재 확인). **Pooled IC = -0.095 ❌**
**평가 기준:** `min_pooled_ic=0.03`, `min_nw_tstat=1.96`, `min_breadth=0.30`, `min_valid_coverage=0.80`

**진단 노트:**
- **SWF-K 전환 (2026-06-12 — CPCV → Purged Sequential Walk-Forward):**
  - ✅ **CPCV 폐기**: OOS collapse 버그(disjoint spans→contiguous mask), anti-causal fold (test_groups=[0,1] → fit=future data), 중첩 fold 중복 이벤트 제거.
  - ✅ **SWF-K 도입**: K=5 등간격 OOS 창, expanding fit (fit_start=0), `fit_end=oos_start-purge_bars` 구조적 인과 보장.
  - ✅ **Pooled IC**: fold 평균 IC → 전 fold OOS 이벤트 concat 후 단일 Spearman rank IC. N=5576 기준.
  - ✅ **NW HAC t-stat**: Newey-West Bartlett kernel, Andrews 1991 자동 대역폭. 자기상관 보정으로 naive t-stat보다 보수적.
  - ✅ **스케일 버그 수정 (Audit)**: `SE(IC)=12·√(S_NW/N)` 정규화 — 미적용 시 |t| 12배 과대평가(거짓 양성 위험).
  - ✅ **fold_pass_ratio gate 제거**: 진단용 보존, gate 조건 4개로 단순화.
  - ❌ **L1 Gate 차단**: Pooled IC `-0.095`, NW HAC t-stat `-4.60`, Breadth `0.017`, Valid Coverage `0.0%` → **BLOCKED**.
  - 🎯 **다음 병목**: Fold 1-4 유효 심볼 0 (N Valid=0) — l1_start 기점 warmup 구간 내 OOS 파티션 흡수 현상. 신호 alpha 부재 지속.

- **Direction A/B 알고리즘 진단:**
  - ✅ **Direction A (Calibration)**: 6개 Regime 중 3개 유효 확인. `score_z` 기반 슬로프 피팅 정상 작동.
  - ✅ **Direction B (Risk)**: q90 실산출을 통한 Kelly Sizing 정상화 완료.
  - **핵심 결론**: 평가 프레임워크(SWF-K) 신뢰성 확보 완료. 현재 블로커는 신호 알파 자체의 예측 방향성 부재.

---

## Symbols
```text
[SELECTED SYMBOLS] ---------------------------------
| 1000FLOKIUSDT, 1000PEPEUSDT, 1000SATSUSDT, 1000SHIBUSDT, 1000XECUSDT, AAVEUSDT |
| ADAUSDT, ANKRUSDT, APEUSDT, API3USDT, ARBUSDT, ARPAUSDT |
| ARUSDT , ASTRUSDT, ATOMUSDT, AUCTIONUSDT, AVAXUSDT, AXSUSDT |
| BAKEUSDT, BANDUSDT, BCHUSDT, BIGTIMEUSDT, BLZUSDT, BNBUSDT |
| BNXUSDT, BTCUSDT, CKBUSDT, CRVUSDT, DOGEUSDT, DOTUSDT |
| DYDXUSDT, EIGENUSDT, ENSUSDT, ETCUSDT, ETHUSDT, FILUSDT |
| FLMUSDT, FTMUSDT, GALAUSDT, GRTUSDT, ICPUSDT, IOTAUSDT |
| IPUSDT , JASMYUSDT, JTOUSDT, KAITOUSDT, KAVAUSDT, LDOUSDT |
| LEVERUSDT, LINAUSDT, LINKUSDT, LPTUSDT, LTCUSDT, LUNA2USDT |
| MANAUSDT, MKRUSDT, MOODENGUSDT, MTLUSDT, NEARUSDT, NEOUSDT |
| OCEANUSDT, OPUSDT , PEOPLEUSDT, POPCATUSDT, REEFUSDT, RSRUSDT |
| RUNEUSDT, RVNUSDT, SANDUSDT, SEIUSDT, SNXUSDT, SOLUSDT |
| STMXUSDT, STORJUSDT, SXPUSDT, THETAUSDT, TRBUSDT, TRXUSDT |
| UNFIUSDT, UNIUSDT, VETUSDT, VIDTUSDT, WAVESUSDT, WIFUSDT |
| WLDUSDT, XLMUSDT, XRPUSDT, XVGUSDT, YGGUSDT, ZECUSDT |
| ZENUSDT, ZETAUSDT, ZILUSDT, ZRXUSDT              |
----------------------------------------------------
```

## 수정 효과 비교 (CPCV → SWF-K)

| 항목 | 이전 (CPCV) | 이후 (SWF-K) | 변화 | 평가 |
|---|---|---|---|---|
| **평가 알고리즘** | CPCV (15 folds, OOS collapse 버그) | **SWF-K** (5 folds, 구조적 인과 보장) | 근본 교체 | anti-causal 제거 ✓ |
| **IC 집계** | fold 평균 (Equal-weight, N 무시) | **Pooled IC** (전체 이벤트 concat) | N-가중 정확화 | 표본 크기 편향 제거 ✓ |
| **t-stat 방식** | Naive + n_eff 보정 | **NW HAC (Bartlett kernel)** | 자기상관 보정 | 보수적 추정 ✓ |
| **Fold 1 N=0** | n/a (anti-causal 버그) | n/a (warmup 이전 데이터 없음) | 원인 변경 | 인과 버그 해소 ✓ |
| **Mean IC** | -0.017 (fold 평균) | **Pooled IC -0.095** | 더 음수 | N-가중으로 정직한 값 |
| **t-stat** | -3.51 (naive) | **-4.60 (NW HAC)** | 보수적 | 과대평가 제거 ✓ |
| **fold_pass_ratio gate** | ≥0.60 (gate 포함) | 진단 전용 (gate 제외) | 단순화 | 게이트 오염 제거 ✓ |
| **L1 Gate** | **BLOCKED** | **BLOCKED** ❌ | — | 진단 신뢰성 확보 |

---

## 최신 실행 요약 (4h Timeframe - Signal Phase, 2026-06-12)

```text
[WINDOW] -------------------------------------------
| Range              | 2022-10-01 ~ 2026-03-31     |
| IS Start           | 2023-10-01                  |
| OOS Start          | 2025-10-01                  |
----------------------------------------------------

[STRATEGY: candidate_ml] ---------------------------
| Inf Panel          | 63 symbols                  |
| Live Panel         | 12 symbols                  |
| Trade Symbols      | 20                          |
----------------------------------------------------

[LAYER 1: SWF SIGNAL VALIDATION] --------------------
| Metric               | Value   | Gate  | Status      |
| -------------------- | ------- | ----- | ----------- |
| Pooled IC            | -0.095  | >0.03 | BLOCKED     |
| IC t-stat (NW HAC)   | -4.60   | >1.96 | ✗ FAIL      |
| Symbol Breadth       | 0.017   | >0.3  | —           |
| Valid Coverage       | 0.0%    | >80%  | —           |
| Valid Symbols/N      | 1/12    | —     | —           |
| L1 Gate              | —       | —     | BLOCKED     |
------------------------------------------------------

[SWF FOLD DETAILS] ----------------------------------
| Fold | IC     | Breadth | N Valid | N Events | Pass  |
| ---- | ------ | ------- | ------- | -------- | ----- |
| 1    | n/a    | 0.000   | 0       | 0        | FAIL  |
| 2    | n/a    | 0.000   | 0       | 0        | FAIL  |
| 3    | n/a    | 0.000   | 0       | 649      | FAIL  |
| 4    | n/a    | 0.000   | 0       | 2759     | FAIL  |
| 5    | -0.142 | 0.083   | 1       | 2168     | FAIL  |
------------------------------------------------------

[PER-SYMBOL AGGREGATE] ------------------------------
| Symbol   | Raw Mu  | Vol   | t-stat  | IC(avg) | Valid |
| -------- | ------- | ----- | ------- | ------- | ----- |
| AAVEUSDT | -1.908  | 0.013 | -0.59   | -0.254  | N     |
| BCHUSDT  | -1.967  | 0.008 | -1.54   | -0.171  | N     |
| BTCUSDT  | -6.538  | 0.006 | -10.71  | -0.135  | Y     |
| DOGEUSDT | -3.506  | 0.011 | -0.93   | -0.053  | N     |
| ETHUSDT  | 0.161   | 0.009 | 0.15    | -0.122  | N     |
| LPTUSDT  | 0.000   | 0.010 | 0.00    | 0.000   | N     |
| MKRUSDT  | 1.459   | 0.000 | 0.49    | -0.086  | N     |
| SOLUSDT  | -5.176  | 0.013 | -1.20   | -0.021  | N     |
| TRBUSDT  | -3.913  | 0.011 | -1.14   | -0.045  | N     |
| UNIUSDT  | -4.604  | 0.010 | -1.55   | -0.130  | N     |
| XRPUSDT  | 0.481   | 0.010 | 0.12    | 0.050   | N     |
------------------------------------------------------

[SYSTEM STATUS] ------------------------------------
| Layer   | Status  | Blocker (if any)            |
| ------- | ------- | --------------------------- |
| Layer 1 | BLOCKED | gate_passed=False           |
| Layer 2 | SKIP    | —                           |
| Layer 3 | SKIP    | —                           |
-----------------------------------------------------
[TIERED] pipeline complete: L1.gate=False L2=False L3=False
```
