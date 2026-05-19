# Futures Backtest Engine Integrity Validation Specification

본 문서는 `my-coin-traider` 프로젝트의 선물(Futures) 백테스팅 엔진(`PortfolioBacktestEngine` / `backtest_target_weights_numba`)이 금융적·수학적으로 무결함을 보장하기 위한 테스트 설계 명세서다.

> **대상 엔진:** `src/domain/futures/backtest_engine.py` — `PortfolioBacktestEngine` (구 `MultiSymbolEngine`)  
> **핵심 Numba 루프:** `src/domain/futures/portfolio/execution_sim.py` — `backtest_target_weights_numba`  
> **Look-ahead 격리 지점:** `src/domain/futures/portfolio/portfolio_constructor.py` — `precompute_rebalance_weights` (`mu_2d[i-1]` 사용)

---

## 1. 개요

백테스팅 엔진은 전략의 성과를 측정하는 '실험실'이다. 실험실의 물리 법칙(수수료, 체결 로직, 증거금 계산)이 현실 시장과 다르면, 아무리 정교한 알파(Alpha)나 HMM 모델도 무의미해진다. 본 명세는 엔진의 수리적 정확성을 검증하는 **9대 핵심 테스트 범주**를 정의한다.

**HMM/Alpha 격리 방법:** `aligned_data["target_weights"]`를 직접 주입하면 `precompute_rebalance_weights`(알파·HMM 의존) 경로를 건너뛴다. 이를 이용해 엔진 수학 로직만 순수하게 검증한다.

---

## 2. 핵심 테스트 범주 (9 Pillars of Integrity)

### Pillar 1: 노출 한도 및 동시 포지션 (Exposure & Concurrent Cap)

엔진이 자산 노출 상한과 동시 포지션 수 제한을 정확히 강제하는지 검증한다.

- **Test Case 1.1: Gross Exposure 캡**
  - **Scenario:** 5개 심볼에 각 weight=0.30 (gross=1.50) 설정, `max_exposure=0.80`.
  - **Verification:** 엔진이 각 weight를 `0.80/1.50` 비율로 스케일다운하여 실제 진입 Notional이 `equity * 0.80` 이하인지 확인.

- **Test Case 1.2: Max Concurrent 포지션 제한**
  - **Scenario:** 10개 심볼에 weight 신호, `max_concurrent=3`.
  - **Verification:** 최대 3개 심볼에만 포지션이 열리고 나머지는 진입 거부(weight=0)되는지 확인. |weight| 내림차순 상위 3개가 선택됨을 검증.

- **Test Case 1.3: Drawdown 기반 포지션 축소 (DD Scaling)**
  - **Scenario:** `dd_scaling_threshold=0.15`. 리밸런싱 시점에 현재 DD가 25%인 상태. weight=0.50 신호 주입.
  - **Math:** `dd_factor = max(0.1, 1.0 - (0.25/0.40)) = 0.375`. 실제 적용 weight ≈ `0.50 * 0.375 = 0.1875`.
  - **Verification:** 엔진이 실제로 축소된 Notional로 진입하는지 확인.

### Pillar 2: 청산 및 파산 방지 (Liquidation & Bankruptcy Protection)

레버리지 포지션에서 자본이 소진될 때 엔진이 강제 청산하는지 검증한다.

- **Test Case 2.1: 역방향 갭에 의한 순자산 소진**
  - **Scenario:** 잔고 $1,000, 레버리지 5배. 포지션 보유 중 역방향 갭으로 `current_equity ≤ 0` 발생.
  - **Verification:** 엔진이 해당 바에서 전 포지션 강제청산 후 루프 종료. `equity_curve` 해당 인덱스 이후가 0 또는 마지막 값으로 고정됨을 확인. `final_balance ≥ 0`.

- **Test Case 2.2: 잔고 부족 시 신규 진입 거부**
  - **Scenario:** `free_margin`이 required margin보다 작은 상태에서 weight 신호 주입.
  - **Verification:** `margin_fail_cnt` 증가, 포지션 미개설.

### Pillar 3: 거래 비용 정밀도 (Fees & Slippage Math)

수수료와 슬리피지가 수학적으로 정확히 차감되는지 검증한다.

- **Test Case 3.1: Taker 왕복 비용 (Round-trip Cost)**
  - **Scenario:** `taker_fee=0.0005`, `slippage_rate=0.0002`. open=$1,000 시 롱 진입, 다음 바 open=$1,100 시 리밸런싱 청산.
  - **Math (멀티엔진 — 항상 Taker, open 체결):**
    - 진입가: `$1,000 * (1 + 0.0002) = $1,000.20`
    - 진입수수료: `qty * $1,000.20 * 0.0005`
    - 청산가: `$1,100 * (1 - 0.0002) = $1,099.78`
    - 청산수수료: `qty * $1,099.78 * 0.0005`
    - Net PnL per unit: `(1099.78 - 1000.20) - (1000.20 + 1099.78) * 0.0005 * qty_scale`
  - **Verification:** `final_balance - initial_balance`가 위 수식과 `abs=1e-4` 이내로 일치.

- **Test Case 3.2: 리밸런싱 회전율(Turnover) 비용**
  - **Scenario:** weight=0.50 → 다음 리밸런싱 바에 weight=0.25로 변경(반절 축소).
  - **Verification:** 포지션 절반이 exit 체결되고, exit fee가 `qty * exit_price * taker_fee`로 정확히 차감됨을 확인.

### Pillar 4: 가격 갭 체결 논리 (Price Gaps & Execution)

캔들 사이 갭 발생 시 체결 가격이 현실적인지 검증한다.

- **Test Case 4.1: 스탑로스 하단 갭 하락 (Gap-down Stop)**
  - **Scenario:** entry_price=$100, stop_price=$90 (ATR stop). 다음 캔들 `open=$80` (갭 하락).
  - **Verification:** 엔진이 $90이 아닌 **`$80 * (1 - slippage_rate)`**에 체결. ($90 체결 시 Look-ahead Bias로 간주).

- **Test Case 4.2: 상향 갭에 의한 자동 리밸런싱 손실**
  - **Scenario:** 숏 포지션 보유 중 상향 갭 발생. stop_price 상단 돌파.
  - **Verification:** `open_price >= stop_price` 조건에서 `open * (1 + slippage_rate)`로 체결됨을 확인.

### Pillar 5: 펀딩비 물리 (Funding Rate Physics)

무기한 선물의 보유 비용이 시간 흐름에 따라 정확히 반영되는지 검증한다.

- **Test Case 5.1: 양수 펀딩비 — 롱 손실**
  - **Scenario:** 가격 변동 0, `funding_rate=+0.0001` 매 바. 롱 포지션 유지 10 바.
  - **Math:** 매 바 `fund_fee = amount * price * 0.0001 * 1(long)`. 10 바 누적 = `10 * fund_fee`.
  - **Verification:** `total_funding_paid ≈ 10 * fund_fee`. 가격 불변 상태에서 balance만 감소.

- **Test Case 5.2: 부호 규약 — 숏 포지션 양수 펀딩비 수취**
  - **Scenario:** 양수 펀딩비(`+0.0001`), 숏 포지션. 숏은 `pos_side=-1` → `fund_fee = notional * (+0.0001) * (-1) = -값`.
  - **Verification:** 엔진은 청산 시 `-fund_fee_stored`를 더하므로, 결과적으로 balance가 **증가**해야 함(펀딩 수취). 반대로 음수 펀딩에서는 숏 balance가 감소해야 함.

### Pillar 6: 미래 참조 오류 방지 (Look-ahead Bias Protection)

T 시점 신호가 T+1 시가(Open)에 실행되는지 검증한다.

- **Test Case 6.1: Signal-Execution Lag**
  - **Scenario:** `target_weights`를 직접 주입. `weights[i]`가 T=i 시점에 결정된 신호.
  - **Verification:** 엔진은 `(i % rebalance_bars) == 0`인 바의 `open_2d[i]`에 체결. `target_weights[i]`는 `precompute_rebalance_weights` 내 `mu_2d[i-1]`로 산출됨을 별도 단위 테스트로 확인.
  - **Anti-pattern 검출:** `weights[i]`가 `close_2d[i]`나 `low_2d[i]`에 영향을 받는다면 Look-ahead. `weights[i+1]` 기준 신호를 주입해 bar i에 체결이 발생하지 않음을 확인.

- **Test Case 6.2: ATR Stop 기준 시점**
  - **Scenario:** 포지션 진입 시 ATR stop은 `atr_2d[prev_i, s]` (진입 전 바의 ATR) 기준임을 확인.
  - **Verification:** `atr_2d[i, s]`와 `atr_2d[i-1, s]`를 다르게 설정 후 실제 stop_price 역산.

### Pillar 7: 자금 보존 항등식 (Conservation of Money)

이것이 엔진 무결성의 가장 강력한 단일 테스트다.

- **Test Case 7.1: 회계 항등식 검증**
  - **Invariant:** `final_balance == initial_balance + Σ(trade.pnl)`
  - **Scenario:** 고정 시드 기반 임의 시나리오 10회 반복 (롱/숏 혼합, 리밸런싱 포함).
  - **Verification:** 모든 케이스에서 `abs(final_balance - (initial_balance + sum(pnl_list))) < 1e-6`. 불일치는 margin 반환 또는 fee 계산 버그를 의미.

- **Test Case 7.2: Equity Curve 일관성**
  - **Verification:** 전 포지션 청산 후 `equity_curve[last_idx] == final_balance`. 포지션 없는 구간의 `equity_curve[i] == balance` (unrealized=0).

### Pillar 8: NaN/Inf 격리 (Numerical Stability)

외부 데이터의 NaN이 잔고를 오염시키지 않는지 검증한다.

- **Test Case 8.1: OHLC NaN 주입**
  - **Scenario:** `open_2d`/`close_2d`에 NaN 주입. 리밸런싱 바에 NaN 가격 심볼 포함.
  - **Verification:** `final_balance`가 NaN/Inf가 아님. NaN 심볼은 진입/청산 스킵(`np.isnan(op)` guard)되며 계정 상태가 오염되지 않음.

- **Test Case 8.2: funding_rate NaN 주입**
  - **Verification:** `fund_fee_stored[s]`에 NaN 누적 없음.

### Pillar 9: 결정론 (Determinism)

- **Test Case 9.1:** 동일 입력으로 엔진을 2회 호출 시 `equity_curve`, `final_balance`가 bit-identical.
- **Rationale:** Numba `@njit(cache=True)` 환경에서 부동소수점 비결정론이 없음을 보장.

---

## 3. Kill Signal 및 Max Hold 검증

- **Test Case KS.1:** `kill_signal[i, s]=1.0`인 바에서 해당 심볼 강제청산이 다음 바 `open`에 실행되는지 확인.
- **Test Case MH.1:** `max_hold_bars=5` 설정, 5바 경과 후 자동 청산 확인.

---

## 4. 구현 가이드

- **파일 위치:** `tests/unit/domain/futures/test_execution_sim_math.py`
- **엔진 격리 방법:** `aligned_data["target_weights"]` 직접 주입 → HMM/alpha 경로 우회
- **Numba 함수 직접 호출:** `from src.domain.futures.portfolio.execution_sim import backtest_target_weights_numba`
- **JIT 워밍업:** 첫 호출 시 컴파일 발생. `conftest.py`의 `session` scope 픽스처에서 1회 워밍업 권장.
- **허용 오차:**
  - 통화 금액: `pytest.approx(expected, abs=1e-4)` (소수점 4자리)
  - 순수 비율/무차원: `pytest.approx(expected, rel=1e-9)`
- **최소 테스트 데이터:** 합성 numpy 배열 사용. 실제 시장 데이터 불필요.

---

## 5. Timeframe 정책 (1h Base → 4h Execution)

- 백테스트 입력의 기준 소스는 **1h OHLCV**이며, `TIMEFRAME=4h` 실행 시 엔진 입력 배열은 내부에서 4개 1h 바를 묶어 집계한다.
- 집계 규칙:
  - `open`: first
  - `high`: max
  - `low`: min
  - `close`: last
  - `volume`: sum
  - `funding_rate_sum`: sum
  - `kill_signal`: max
- `atr`: finite last (윈도우 내 마지막 유한값, 없으면 NaN)
- 집계는 닫힌 4개 1h 바 단위로 수행하며, 미완성 tail bar는 사용하지 않는다(look-ahead 방지).

## 6. 현실성 강화 개선 반영 (7 items)

- `margin_fail_cnt` 전용 테스트를 추가해 dust skip과 분리 검증.
- Funding 부호 규약을 엔진 수식 기준으로 명시(+funding에서 short 수취).
- 고정 시드 랜덤 10회 회계 항등식 검증으로 path 다양성 확대.
- `open/high/low/close` NaN 오염 방지 테스트 강화.
- Funding 시간계약: 1h에서 집계된 `funding_rate_sum`을 4h에서 합산 사용.
- 강제청산 고도화 TODO: maintenance margin / bankruptcy price / liquidation fee 모델 도입.
- `volume_2d` 전달 경로 활성화로 impact-aware slippage 경로 검증 가능화.
