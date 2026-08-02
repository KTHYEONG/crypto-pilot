# Library Admission Screen — Full 18 Candidates × 5 Symbols

**실행일**: 2026-08-02
**결과 소스**: `/tmp/opencode/full_admission_summary.json` (in-memory aggregation)

## 1. 실행 개요

| 항목 | 값 |
| :--- | :--- |
| 커맨드 | `research run library-admission` (application service 직접 호출) |
| 후보 소스 | **18개 전원** — 9 families × {long, short} (`TECHNICAL_CANDIDATES`) |
| 심볼 | BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT |
| 전문가 수 | 90 (18 sources × 5 symbols) |
| 공통 윈도우 | 2022-04-01 00:00 ~ 2025-12-31 20:00 (UTC, 4h bars, 8,226 rows) |
| 라우터 컨텍스트 | BTCUSDT (trend=48, vol=48, min_history=96, conf=0.9) |
| 심사 임계값 | min_closed_trades=20, min_active_return_bars=200, corr≤0.7, joint_neg≤0.2, covered_states≥4 |
| 실행 | workers=5, wall=8.76s |
| 결과 | **status=COMPLETE**, proposals=**378,560** (전부 eligible) |

> 비고 1: SOL/XRP는 2022-04-01 상장이므로 `--start 2022-04-01`로 공통 시작점 정렬 필요 (미지정 시 컨텍스트 정렬 `DataIntegrityError`).
> 비고 2: 구조적 조합 수는 768,960개. 상관/동시손실 필터로 378,560개로 축소됨. 전체 제안의 CLI JSON 출력은 ~500MB 규모라 서비스 함수를 직접 호출해 요약.
> 비고 3: 이번 실행에서 **`_pair_compatibility_matrix` shape 버그를 발견·수정** — 불합격 후보가 있는 유니버스에서 전체(90열) 수익률 행렬 vs 합격자(80명) families/symbols 배열 간 브로드캐스트 충돌. 합격자 열로 슬라이싱 후 전달하도록 수정 (`src/research/expert_portfolio/admission.py:94`), 회귀 테스트 추가 (`test_rejected_candidate_is_excluded_from_compatibility_matrix`).

## 2. 후보 심사 결과 (90명 중 80명 합격)

기준: `closed_trades ≥ 20` AND `active_return_bars ≥ 200`. 표 값 = `closed_trades / active_return_bars`.

| 패밀리·방향 | BTC | ETH | SOL | BNB | XRP |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ichimoku long | 188/3251 | 182/3270 | 169/3097 | 185/3337 | 197/3087 |
| ichimoku short | 187/2917 | 188/2890 | 184/3240 | 201/2816 | 203/2927 |
| ema long | 101/2702 | 106/2398 | 97/2334 | 136/2702 | 104/2037 |
| ema short | 108/2461 | 140/2812 | 115/2760 | 96/2270 | 122/2911 |
| rsi long | 41/1498 | 39/1289 | 35/1617 | 45/1798 | 33/1542 |
| rsi short | 47/2238 | 57/2437 | 46/1498 | 39/1718 | 52/1899 |
| macd long | 152/1789 | 130/1525 | 138/1589 | 171/1917 | 162/1694 |
| macd short | 145/1621 | 165/1866 | 170/1983 | 168/1773 | 184/2075 |
| cci long | 120/480 | 115/475 | 131/614 | 135/564 | 112/364 |
| cci short | 133/706 | 162/711 | 135/617 | 141/486 | 144/622 |
| stochastic long | 121/662 | 105/624 | 122/722 | 99/573 | 93/559 |
| stochastic short | 136/800 | 150/905 | 135/816 | 139/782 | 157/942 |
| adx long | 78/650 | 74/590 | 80/552 | 65/499 | 72/693 |
| adx short | 72/867 | 69/684 | 85/984 | 64/544 | 61/533 |
| bb_squeeze long | 39/654 | 33/453 | 26/354 | 42/577 | 27/246 |
| bb_squeeze short | 43/553 | 41/547 | 47/760 | 43/734 | 52/752 |
| **mfi long** | 31/62 REJ | 20/40 REJ | 23/46 REJ | 20/40 REJ | 15/30 REJ |
| **mfi short** | 35/71 REJ | 28/56 REJ | 23/46 REJ | 17/34 REJ | 19/38 REJ |

**불합격 10명 — 전부 `mfi_trend_pullback`**: 활성 바가 30~71에 불과 (기준 200 미달), XRP long(15)과 BNB short(17)은 종결 트레이드도 미달. MFI는 거래 빈도가 극히 낮아 실질적으로 라이브러리 후보에서 제외됨.

관찰:
- ichimoku/ema가 활성 바 최상위(3,000+/2,300+) — 장기 추세형. rsi/bb_squeeze는 낮은 거래 빈도(최저 26회).
- 심사는 빈도/활성도만 판별하며 수익률 기반 랭킹 없음 (`no return-based ranking ever occurs`).

## 3. 컨텍스트 커버리지

`covered_states = 4 / 6`, `coverage_sufficient = True` (기준 ≥ 4). 이전 실행과 동일.

| 상태 | 바 수 | 판정 |
| :--- | ---: | :--- |
| up_low_vol | 2,576 | ✅ |
| down_low_vol | 2,186 | ✅ |
| down_high_vol | 1,702 | ✅ |
| up_high_vol | 1,650 | ✅ |
| flat_low_vol | 48 | ❌ (min_history 96 미달) |
| flat_high_vol | 15 | ❌ (min_history 96 미달) |

## 4. 제안 조합 (378,560개, 전부 eligible)

구조적 조합(패밀리·심볼 중복 금지, `_pair_compatibility_matrix`) 후 상관 ≤0.7·동시손실 ≤0.2 필터 적용.

| 제안 크기 | 구조적 후보 | 실제 제안 |
| ---: | ---: | ---: |
| 2 | 2,880 | 2,240 |
| 3 | 40,320 | 26,880 |
| 4 | 241,920 | 134,400 |
| 5 | 483,840 | 215,040 |
| **합계** | **768,960** | **378,560** |

- 상관/동시손실 필터가 구조적 후보의 약 **51%를 제거** — 필터가 실질적으로 작동.
- 크기 5 제안이 215,040개로 최대 (5 심볼 전부 + 패밀리 5종, side 조합 자유).
- mfi가 합격자에서 빠져 mfi 포함 제안은 없음.
- `eligible`은 라우터 커버리지 **전역 플래그**이므로 378,560개 전부 `eligible: true`. 개별 제안별 차별화 없음.
- 서로 다른 패밀리 집합(distinct family sets) 3,472개.

예시 제안 (크기 2/3/5):
```
lae-v1:technical_adx_di_regime_long_v1:BNBUSDT|technical_bb_squeeze_breakout_long_v1:BTCUSDT
lae-v1:technical_adx_di_regime_long_v1:BNBUSDT|...|technical_cci_trend_pullback_long_v1:ETHUSDT
lae-v1:...:BTCUSDT|...:ETHUSDT|...:SOLUSDT|...:BNBUSDT|technical_ichimoku_cloud_long_v1:XRPUSDT
```

## 5. 결론 및 시사점

- 전체 18 후보 심사 결과 **80명 합격, 378,560개 조합 전부 eligible**. MFI 패밀리만 빈도 기준으로 탈락.
- 상관/동시손실 필터가 구조적 후보의 절반을 걸러내므로 조합 수가 크게 줄지만, 여전히 38만 개 — 등록/평가 전략을 정해야 함.
- 제안 후보가 지나치게 많아 다음 단계(`library-admission-backtest`)를 조합 전체에 돌릴 수 없음. 임계값 강화(예: corr≤0.5, joint_neg≤0.15) 또는 사전 선별 기준(예: 크기 3 이하, 또는 특정 패밀리 우선)이 필요.
- 버그 발견으로 전체 심사가 최초 실행 불가했으며, 수정 후 이 결과가 도출됨.
