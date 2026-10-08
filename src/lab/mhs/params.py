"""Exploratory research tunables owned by the lab.

Every constant here is referenced only by ``src.lab`` modules (verified by the
layer import contract). Shared deployable constants stay in
``src.core.params``; derived values import their core base from there.
"""

from __future__ import annotations

import pandas as pd

from src.core.params import (
    COMMITTEE_PURGE_HOURS,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    MOMENTUM_HORIZON_CANDIDATES_HOURS,
    UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
)

# Canonical value of committee_member_set while committee capital is off; equal to the fixed trial-identity baseline so a capital-off run never carries an inert member set into its trial key or procedure digest.
COMMITTEE_MEMBER_SET_INERT: str = "risk_premia"

# Sealed identity of the committee member-admission procedure. Bumped (never edited
# in place) whenever admission semantics change, so run-history trial identities and
# preregistered procedure digests of different admission procedures never collide.
COMMITTEE_ADMISSION_PROCEDURE: str = "boundary_frozen_warmup_excluded_v1"

# 트랜치 평균 창(개수 x 24h 결정 간격)이 flow_momentum 최단 멤버 룩백 168h를
# 넘으면 북이 신호 수명보다 늦게 반응하므로 7을 상한으로 둔다.
COMMITTEE_TRANCHE_COUNT_MAX: int = 7

CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT: int = 60

# The window the CLI defaults (growth_extreme, committee_kelly_sizing, breadth
# 60) were measured on (see pipeline/config.py provenance notes). Any report
# whose window intersects this span is partially in-sample for those defaults;
# the overlap fraction is disclosed observationally, never silently omitted.
DEFAULT_SELECTION_WINDOW: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2025-12-31", tz="UTC"),
)


# 전진 판정 최소 표본: 계절·펀딩 국면이 모두 한 번씩 포함되는 1년(4분기).
# 실측 스트레스 통계로 E2 검정력에 필요한 전진 길이가 1.0~1.4년이라 이보다 짧으면 평가하지 않는다.
FORWARD_MIN_FOLDS: int = 4

DISCOVERY_GATE_TRANCHE_COUNT: int = 8

# 단일 종목 드리프트 상한: 4개 균등 슬롯(0.05 = 레버리지 상한 3.0 / 실행 유니버스 60)이 상한에 걸린 값.
NAME_DRIFT_TRIM_MAX_WEIGHT: float = 0.20

DISCOVERY_REVERSAL_CANDIDATES: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)

DISCOVERY_MOMENTUM_CANDIDATES: tuple[int, ...] = MOMENTUM_HORIZON_CANDIDATES_HOURS

FEATURE_NAME = "multi_horizon_market_state"

PERIODS_PER_YEAR_1H: float = 365.0 * 24.0

WALK_FORWARD_MIN_TRAIN_BARS: int = 2000

GO_PRIMARY_SHARPE_FLOOR: float = 0.6

ARTIFACT_SCHEMA_VERSION: int = 1

ARTIFACT_CATEGORIES: tuple[str, ...] = (
    "fills",
    "units",
    "notional_weights",
    "ledger",
    "times",
)

REBALANCE_DEADBAND_POSITION_FRACTION: float = 0.25

FOLD_BLEND_PARITY_TOLERANCE: float = 0.25

# fold 실현변동성 log-ratio 관측 허용폭(≈1.42x): 관측 전용, reason code 없음.
FOLD_REALIZED_RISK_PARITY_TOLERANCE: float = 0.35

# --- 증거 게이트 보정(I-CALIB) ---------------------------------------------------
# 등록 상수는 원시 지표 임계값이 아니라 선언된 오차율 alpha이며, 임계값은 각 런에서
# 전략 자신의 pooled 수익률 null로부터 파생된다(실측: 오탈락 5.7%/탐지력 99.6%).
# FOLD_GROWTH_CONCENTRATION_MAX_SHARE = 0.5 는 유도 근거 부재(동일분포 null의
# 83 백분위)로 삭제되었고, alpha 파생 임계값이 이를 대체한다.
EVIDENCE_GATE_ALPHA: float = 0.05

NULL_BOOTSTRAP_TRIALS: int = 2000

# I-DETERMINISTIC: 동일 원장 -> 비트 동일 임계값을 보장하는 등록 시드.
NULL_BOOTSTRAP_SEED: int = 20260823

# null 적합에 필요한 최소 유한 pooled 일간 행수(미만 시 fail-closed).
NULL_BOOTSTRAP_MIN_ROWS: int = 250

SIGNAL_EMA_HORIZON_SPAN: float = 1.0

REGIME_CASH_SCALE_FLOOR: float = 0.5

REGIME_CASH_MEDIAN_WINDOW_HOURS: int = 720

REFERENCE_PASS_EQUITY_FLOOR: float = REGIME_CASH_SCALE_FLOOR

PNL_VOL_TARGET_WINDOW_DAYS: int = 21

PNL_VOL_TARGET_SCALE_FLOOR: float = 0.2

PNL_VOL_TARGET_BURN_IN_DAYS: int = 90

PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS: int = 365

# 위원회 Kelly 사이징: z=0 이므로 LCB 페널티가 없다(z/sqrt(window)=0).
# 평균 에지에서 LCB 는 표본 평균과 동일하다. window=42/half-Kelly 0.5 유지.
COMMITTEE_KELLY_WINDOW_DAYS: int = 42

# half-Kelly 상한값. 등록 가능한 최대치이며 half-Kelly 경계로 강제된다.
COMMITTEE_KELLY_FRACTION: float = 0.5

COMMITTEE_KELLY_LCB_Z: float = 0.0

# constant_risk 모드 전용 상수(기존 모드의 PNL_VOL_TARGET_* 는 불변).
# 실측(3m 원장, target=0.40 고정) -- halflife가 유일한 다이얼로는 두 게이트를
# 동시에 통과시키지 못하는 단조 트레이드오프가 실측 확인됨:
#   hl=90  -> fold_blend_parity=0.317(FAIL>0.25) / risk_parity=0.234(PASS) / share=0.526
#   hl=120 -> fold_blend_parity=0.264(FAIL>0.25) / risk_parity=0.309(PASS) / share=0.547
#   hl=150 -> fold_blend_parity=0.230(PASS)      / risk_parity=0.373(FAIL>0.35) / share=0.563
# hl=90을 등록: 이 기능이 겨냥한 1차 목표(FOLD_GROWTH_CONCENTRATION, share 최소화)에
# 가장 근접. FOLD_BLEND_PATH_DIVERGENCE는 미해결 -- ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT
# 후속 과제(fold가 자체 EWMA를 재적합하지 않고 blend의 연속 스케일 궤적을 날짜로
# 슬라이스해 재사용하는 구조적 대안이 유력, 미구현).
CONSTANT_RISK_EWMA_HALFLIFE_DAYS: int = 90

# EWMA sigma 해석에 필요한 최소 유한 행수(fold 워밍업으로 데드존 제거).
CONSTANT_RISK_MIN_PERIODS_DAYS: int = 45

# sigma* = leverage_ceiling * q_p(sigma_book|train): cap 포화 확률 <= p 보장.
CONSTANT_RISK_CAP_BINDING_QUANTILE: float = 0.10

# 고정 목표 위험(단일 상수, fold 경계별로 재적합하지 않음): growth_budget_annual_vol을
# 경계별로 재해석하면 표본 특이적 해가 fold마다 갈라져 위험 등화가 깨진다(실측: fold0-2
# 실현변동성 0.14~0.16 vs fold3 0.29, log-ratio 0.63 -- FOLD_GROWTH_CONCENTRATION 재발).
# 단일 고정값만이 모든 경계에서 동일 목표를 강제한다; _feasible_constant_risk_target의
# leverage_ceiling*q10(sigma_book|train) 클램프는 경계별로 유지된다(실현 가능성 검증).
# 실측(3m 원장, target=0.45/halflife=60d): 배치 실현위험이 0.39~0.51로 근접했으나
# fold3의 잔여 초과 Sharpe가 share=0.518로 남음 -- 0.40으로 하향.
CONSTANT_RISK_TARGET_ANNUAL_VOL: float = 0.40

# Fold panel warm-up: eligibility lookback + 168h slow horizon + one-day boundary buffer, so
# eligibility and signals are fully formed at the first fold decision.
FOLD_PANEL_WARMUP_HOURS: int = UNIVERSE_ELIGIBILITY_LOOKBACK_BARS + 168 + 24

TRAIN_REFERENCE_PREFIX_TARGET_ATOL: float = 1e-12

REBALANCE_TRACKING_ERROR_THRESHOLD: float = 0.20

CAUSAL_BETA_LOOKBACK_BARS: int = 720

CAUSAL_BETA_MIN_PERIODS: int = 360

SIGNAL_REPLAY_WARMUP_DAYS: int = 30

SIGNAL_RETURN_TAIL_DAYS: int = 400

SIGNAL_OVERLAP_TOLERANCE: float = 1e-9

INSTRUMENT_LIFECYCLE_PROCEDURE: str = "pit_registry_settlement_halts_causal_exclusions_exit_deferral"

# --- continuous process backtest -------------------------------------------------
# 한 번의 연속 인과 경로에서 매월 재적합한다(분기 폴드 개별 재생 대체).
PROCESS_REFIT_FREQUENCY: str = "MS"

# 첫 적합은 펀딩/계절 주기 1회(365일)의 학습 표본을 요구한다.
PROCESS_MIN_TRAIN_DAYS: int = 365

# 학습 끝과 적용 시작 사이 퍼지: 최장 멤버 룩백(720h)과 동일.
PROCESS_PURGE_HOURS: int = COMMITTEE_PURGE_HOURS

# 집행 평활 반감기(일): 측정된 가장 느린 멤버 신호 감쇠(flow_imb 8일). 학습 구간에서 고르면
# 같은 구간으로 적합한 가중치 때문에 알파가 부풀어 항상 0(무평활)이 선택된다.
PROCESS_SMOOTHING_HALFLIFE_DAYS: float = 8.0

# rank_weight_book 전체에서 쓰는 최소 종목 수 하한과 동일.
PROCESS_MIN_SYMBOLS: int = 8

# 캔들로 계산 가능한 전체 등록 피처. 동일 북을 만드는 중복(taker_imb_168h≡flow_imb_168h,
# mom_336h≡xs_mom_336h)과 레이크에 없는 컬럼(no_trades)이 필요한 avg_trade_size는 제외.
PROCESS_FEATURE_CANDIDATES: tuple[str, ...] = (
    "mom_168h",
    "rev_24h",
    "taker_imb_24h",
    "amihud",
    "lowvol_168h",
    "hl_range_168h",
    "turnover_chg",
    "flow_imb_168h",
    "flow_imb_720h",
    "xs_mom_336h",
    "xs_mom_720h",
    "xs_idio_mom_336h",
    "mom3_skew_168h",
)

PROCESS_FUNDING_CARRY_CANDIDATES_HOURS: tuple[int, ...] = FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS
