"""MHS input data policy: single source of truth (INV-POLICY-SINGLE-SOURCE).

Every MHS entrypoint (``MhsDiagnosticRequest``, ``MhsRunConfig``,
``LiveStrategyParams``, the live runtime) shares ``MHS_DATA_POLICY_DEFAULT``.
The new default is ``zombie_mask_v1``; artifacts that predate the policy field
keep their legacy interpretation and are never auto-upgraded.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal


class MhsDataPolicy(StrEnum):
    """Registered MHS input-data contracts."""

    LEGACY = "legacy"
    ZOMBIE_MASK_V1 = "zombie_mask_v1"


MHS_DATA_POLICY_DEFAULT: Final[Literal["zombie_mask_v1"]] = "zombie_mask_v1"

# 2026-09-15 mhs_symbol_lifespan_pit_roster: the other 50 previously-listed symbols
# (end-of-life: funding permanently ends before backtest end with zero internal gaps,
# exchangeInfo status=SETTLING) are now handled dynamically by `ledger_terminal_only`
# at finalize time instead of blanket exclusion.
#
# 2026-09-15 후속 재수집 스윕: 이전 Block1(OHLCV 캐시 없음/미미) 21개 심볼을
# ensure_ohlcv_data로 재수집한 결과 전부 복구됨(수집 누락이었음, 소스 공백 아님).
# 19개(ALPHAUSDT/BADGERUSDT/BSWUSDT/FLMUSDT/FTTUSDT/IDEXUSDT/KLAYUSDT/MKRUSDT/
# NULSUSDT/OBOLUSDT/OCEANUSDT/OMGUSDT/SLERFUSDT/STRAXUSDT/TROYUSDT/UNFIUSDT/
# VIDTUSDT/WAVESUSDT/XEMUSDT)는 backtest 구간 내부공백 0건으로 완전히 배제 해제됨
# (말기 종료는 ledger_terminal_only가 처리). CVXUSDT/SLPUSDT는 재수집 후에도
# 2025-06-19~2025-07-23(34일) 펀딩 공백이 재개되는 진짜 불확실성 구간이 드러나
# Block3(MID_LIFE_GAP)로 재분류.
#
# 2026-09-18 원천 재조회: AIA/ICP의 3m 내부 공백은 Binance Vision 월·일별
# klines에도 동일하게 존재해 복구 불가. BNT/BTCST/BDXN은 funding 월별 원천이
# 통째로 비어 있는 장기 공백으로 확인되어 정적 제외한다. 이 목록은 데이터 원천
# 부재를 zero-fill로 숨기지 않고, 인증 백테스트에서 fail-closed하기 위한 것이다.
SOURCE_GAP_EXCLUDED_SYMBOLS = frozenset({
    # Block2: 펀딩 정상, 단일 영구 OHLCV 공백 8-17h, REST 확인으로 복구 불가.
    # 실측 검증 대상(MISSING_HELD_MARK 확장으로 해소되는지 리플레이로 확인 중).
    "AERGOUSDT", "CTKUSDT", "CVCUSDT", "MAVIAUSDT",
    # Block3: 백테스트 중간에 펀딩 공백이 발생했다가 재개되는 진짜 불확실성 구간, 제외 유지
    "LITUSDT", "PUMPUSDT", "CVXUSDT", "SLPUSDT",
    # Block4: BNXUSDT는 조기시작 외에도 자체 영구 OHLCV 공백을 보유, 실측 검증 대상
    "BNXUSDT",
    # 2026-09-18 backfill probe: 원천 아카이브에도 복구 구간이 없는 심볼
    "AIAUSDT", "ICPUSDT", "BNTUSDT", "BTCSTUSDT", "BDXNUSDT",
})

# Frozen-only confirmed unrecoverable execution sources. Binance Vision
# archive and Futures REST re-query both failed to supply the interval.
_FROZEN_SOURCE_GAP_EXCLUSIONS: Final[frozenset[str]] = frozenset({
    # PUMPUSDT: funding gap 2025-06-19 08:00 UTC to 2025-07-10 08:00 UTC,
    # Vision monthly archive and REST re-query both missing (two boundary settlements only).
    "PUMPUSDT",
    # LUNAUSDT: 3-minute archive and REST end at 2022-05-13 06:48 UTC,
    # Vision daily/monthly klines and REST re-query both end with no settlement evidence.
    "LUNAUSDT",
})


def frozen_research_source_gap_exclusions() -> frozenset[str]:
    """Return explicitly evidenced source-unavailable symbols for frozen research only.

    The registry prevents targets in intervals whose required OHLCV or funding
    evidence cannot be recovered from Binance sources; it preserves the wider
    historical census and never fabricates a price, funding rate, or settlement.
    """
    return frozenset({s.strip().upper() for s in _FROZEN_SOURCE_GAP_EXCLUSIONS if s.strip().upper().endswith("USDT")})
