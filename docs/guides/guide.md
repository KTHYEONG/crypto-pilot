# 백테스팅 및 시그널 심사 실행 가이드 (Backtest & Admission Execution Guide)

본 문서는 프로젝트 내에 존재하는 다양한 백테스팅 실행 엔트리포인트(CLI 명령어), 시그널 조합 및 심사(Admission) 로직, 각 백테스트의 목적 및 측정 지표를 안내합니다.

---

## 1. 백테스트 개요 및 선택 매트릭스

통합 CLI 진입점: `python -m src.cli.main` (또는 `uv run python -m src.cli.main`)

| 구분 | 3단계 계층형 CLI 명령어 (`research run ...`) | 호환 어댑터 명령어 (Adapters) | 핵심 측정 목적 및 평가 로직 |
| :--- | :--- | :--- | :--- |
| **단일 베이스라인** | `research run single baseline` | `python -m src.cli.adapters.run_backtest` | 단일 시그널/지표의 PnL, Sharpe/Sortino 비율, 낙폭(MDD) 평가 |
| **단일 기술적 지표** | `research run single technical` | - | 동결 기술적 지표 후보 중 단일 알파 스크린 |
| **현선물 차익거래 (Cash & Carry)** | `research run single carry` | `python -m src.cli.adapters.run_cash_carry_backtest` | 현선물 베이시스 차익 PnL, 펀딩비 수익률, 청산 리스크 측정 |
| **미결제약정 디레버리징 (OI)** | `research run single oi` | - | 선물 미결제약정 급감 및 청산 이벤트 발생 시 알파 성능 측정 |
| **다중 자산 포트폴리오** | `research run portfolio multi` | `python -m src.cli.adapters.run_portfolio_backtest` | 다중 자산/시그널 비중 배분, 리스크 분산(MDD 감소), 회전율 비용 측정 |
| **슬리브 혼합 (Sleeve Blend)** | `research run portfolio blend` | `python -m src.cli.adapters.run_sleeve_blend_backtest` | 이종 전략(방향성 + 델타뉴트럴) 혼합 시 시너지 및 상호 헷지 효과 측정 |
| **성장 엔진 (Growth Engine)** | `research run portfolio growth` | - | 제약조건 기반(Constraint-first) 유니버스 및 포트폴리오 성장 백테스트 수행 |
| **등록 전문가 포트폴리오** | `research run expert eval` | `python -m src.cli.adapters.run_expert_portfolio_backtest` | 검증 완료 및 등록된 전문가 포트폴리오 동적 앙상블 백테스트 |
| **시그널 조합 & 심사 (Admission)** | `research run expert admission` | - | 전문가 시그널들의 상호 상관관계, 동시 손실률, 국면 커버리지 검증 및 통과 조합(`proposal_id`) 도출 |
| **조합 제안 백테스트 (Admission Backtest)** | `research run expert backtest` | - | 심사 통과 조합(`proposal_id`)을 등록 전 가상으로 미리 백테스팅하여 성능 사전 검증 |
| **청산 탐색 (Exit Sweep)** | `research run expert exit-sweep` | - | 기술적 알파 후보에 대한 스탑로스(Fixed Pct / ATR Multiple) 청산 파라미터 그리드 탐색 |
| **입선 파이프라인 (Admission Pipeline)** | `research run expert pipeline --profile technical-5symbol-2022-v1` | - | 후보 발견과 OOS 제안 백테스트를 한 번에 수행 (탐색 워크플로우, 등록·승격 아님) |
| **분기 롤링 백테스트 (Rolling)** | `research run expert rolling --profile technical-5symbol-rolling` | - | 분기별 walk-forward 롤링 재배분 백테스트 수행 |

---

## 2. CLI 명령어 및 상세 설명

### 2.1 시그널 조합 검증 및 심사 (`research run expert admission`)
후보 시그널(Candidate Source)들을 다각도로 조합하고 상관관계 및 동시 손실률 등의 제약조건을 적용하여 합격 조합을 필터링합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run expert admission \
    --candidate-source <소스1> --candidate-source <소스2> \
    --symbols BTCUSDT ETHUSDT --min-experts 2 --max-experts 5 \
    --max-abs-pairwise-log-return-correlation 0.7 \
    --max-joint-negative-return-rate 0.2 \
    --min-context-covered-states 4 \
    --max-combinations 10000 \
    --router-context-symbol BTCUSDT --router-trend-lookback-bars 48 \
    --router-volatility-lookback-bars 48 --router-min-context-history-bars 96
  ```
- **주요 검증 항목**:
  - **시그널 간 상관관계**: 쌍별 로그 수익률 상관계수가 임계값(예: 0.7) 이하인지 확인
  - **동시 손실 비율**: 두 시그널이 동시에 손실을 기록하는 비율이 임계값(예: 0.2) 이하인지 확인
  - **시장 국면 커버리지**: 상승/하강/변동성 등 국면(Context State)을 충분히 커버하는지 확인

---

### 2.2 심사 조합 사전 백테스트 (`research run expert backtest`)
`admission`에서 생성된 합격 조합(`proposal_id`)을 시스템 등록 전 가상 백테스트하여 검증합니다. 스탑로스 엔진 파라미터(`--stop-loss-mode`, `--stop-loss-value`, `--atr-period`, `--trailing-stop`) 설정도 지원합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run expert backtest \
    --proposal-id <생성된_proposal_id> \
    --router-context-symbol BTCUSDT --router-trend-lookback-bars 48 \
    --router-volatility-lookback-bars 48 --router-min-context-history-bars 96
  ```

---

### 2.3 스탑로스 청산 탐색 (`research run expert exit-sweep`)
기술적 시그널 후보군에 대해 고정 비율(Fixed Pct) 및 ATR 승수(ATR Multiple) 방식의 스탑로스 청산 조건들을 탐색하고 성과 변화를 비교합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run expert exit-sweep \
    --candidate-source technical_bollinger_breakout_v1 \
    --symbols BTCUSDT ETHUSDT \
    --timeframes 4h \
    --fixed-pct-values 0.03 0.05 0.08 \
    --atr-multiple-values 1.5 2.5 4.0
  ```

---

### 2.4 라이브러리 입선 파이프라인 (`research run expert pipeline`)
후보 발견(선택 윈도우 2022-04-01 ~ 2024-12-31)과 최대 24개 쇼트리스트의 OOS 백테스트(2025-01-01 ~ 2025-12-31)를 한 번의 실행으로 수행합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run expert pipeline \
    --profile technical-5symbol-2022-v1
  ```
- **주요 동작**:
  - **공통 가용 기간 검증**: 모든 심볼의 연속 1h OHLCV와 펀딩 데이터가 존재하는 최신 시작일로 조정
  - **쇼트리스트**: 쌍별 상관계수/동시 손실률의 분산화 지표만으로 크기 계층별 최대 6개씩, 예산 내에서 순위순으로 결정론적 추림
  - **OOS 성과**: 지정된 평가 기간 데이터로만 자식 제안 백테스트 수행

---

### 2.5 분기 롤링 백테스트 (`research run expert rolling`)
시점 기준(`--as-of`) 데이터 스냅샷을 기반으로 분기별 Walk-Forward 롤링 재배분 백테스트를 수행합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run expert rolling \
    --profile technical-5symbol-rolling \
    --as-of "2026-07-07 20:00+00:00" \
    --mode paper
  ```

---

### 2.6 등록된 전문가 포트폴리오 백테스트 (`research run expert eval`)
심사를 마치고 카탈로그에 최종 등록된 전문가(Expert) 라이브러리를 동적 앙상블하여 백테스트를 실행합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run expert eval --library-id <라이브러리_ID>
  ```
- **호환 어댑터 명령어**:
  ```bash
  uv run python -m src.cli.adapters.run_expert_portfolio_backtest --library-id <라이브러리_ID>
  ```

---

### 2.7 성장 엔진 백테스트 (`research run portfolio growth`)
제약조건 기반 유니버스(PitUniverseSpec) 및 포트폴리오 성장 백테스트를 수행합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run portfolio growth \
    --universe-size 20 \
    --max-positions 5 \
    --symbol-scope dev
  ```

---

### 2.8 단일 시그널 백테스트 (`research run single baseline` / `technical`)
단일 기술적 지표 또는 개별 방향성 알파 시그널의 단독 수익성과 리스크를 측정합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run single baseline --symbol BTCUSDT --start 2022-04-01
  uv run python -m src.cli.main research run single technical --candidate-source technical_bollinger_breakout_v1 --symbol BTCUSDT
  ```
- **호환 어댑터 명령어**:
  ```bash
  uv run python -m src.cli.adapters.run_backtest --symbol BTCUSDT --start 2022-04-01
  ```

---

### 2.9 현선물 차익거래 백테스트 (`research run single carry`)
현물 매수 + 선물 매도(델타 뉴트럴) 포지션을 바탕으로 한 베이시스 차익 및 펀딩비 수취 성과를 측정합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run single carry --symbol BTCUSDT --start 2022-04-01
  ```
- **호환 어댑터 명령어**:
  ```bash
  uv run python -m src.cli.adapters.run_cash_carry_backtest --symbol BTCUSDT --start 2022-04-01
  ```

---

### 2.10 다중 자산 포트폴리오 백테스트 (`research run portfolio multi`)
여러 코인 자산 또는 시그널에 자금을 분산 배분할 때의 성과를 측정합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run portfolio multi --symbols BTCUSDT ETHUSDT --start 2022-04-01
  ```
- **호환 어댑터 명령어**:
  ```bash
  uv run python -m src.cli.adapters.run_portfolio_backtest --symbols BTCUSDT ETHUSDT --start 2022-04-01
  ```

---

### 2.11 슬리브 혼합 백테스트 (`research run portfolio blend`)
방향성 전략 슬리브와 비방향성/차익거래 전략 슬리브 등 복합 구성 성과를 측정합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run portfolio blend --candidate-kind funding_signed_directional_v1
  ```
- **호환 어댑터 명령어**:
  ```bash
  uv run python -m src.cli.adapters.run_sleeve_blend_backtest --candidate-kind funding_signed_directional_v1
  ```

---

### 2.12 미결제약정 디레버리징 백테스트 (`research run single oi`)
선물 시장의 미결제약정(OI) 급감 또는 청산 빔 발생 시 유동성 불균형 이벤트 전략 성과를 측정합니다.

- **명령어 실행**:
  ```bash
  uv run python -m src.cli.main research run single oi --symbol BTCUSDT --start 2022-04-01
  ```

---

## 3. 데이터 수집 및 관리 CLI (`data`)

데이터 수집 및 복구 관련 명령어는 `data collect` 하위에 위치합니다.

- **선물/현물 OHLCV 수집**:
  ```bash
  uv run python -m src.cli.main data collect futures-ohlcv BTCUSDT 1h --start 2022-04-01
  uv run python -m src.cli.main data collect spot-ohlcv BTCUSDT 1h --start 2022-04-01
  ```
- **펀딩비 및 OI 수집**:
  ```bash
  uv run python -m src.cli.main data collect funding BTCUSDT --start 2022-04-01 --end 2025-12-31
  uv run python -m src.cli.main data collect metrics BTCUSDT --start 2022-04-01 --end 2025-12-31
  ```
- **차단/차입 이력 관리 및 Gap 복구**:
  ```bash
  uv run python -m src.cli.main data collect import-borrow BTCUSDT --source path/to/borrow.parquet --source-id op1
  uv run python -m src.cli.main data collect repair-spot-gap BTCUSDT 1h 2024-01-01T00:00:00
  ```

---

## 4. 출처 및 이력 관리 CLI (`provenance`)

전문가 라이브러리 등록 및 백테스트 실행 이력 비교/조회를 담당합니다.

- **전문가 라이브러리 등록**:
  ```bash
  uv run python -m src.cli.main provenance register expert-library --library-id <라이브러리_ID>
  ```
- **실행 결과 비교**:
  ```bash
  uv run python -m src.cli.main provenance compare-runs --last 10 --sort-by metrics.sharpe
  ```

