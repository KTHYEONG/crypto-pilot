"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.integrity as integrity


def test_integrity_module_present() -> None:
    assert integrity.__name__ == "src.mhs.evaluation.integrity"
    assert callable(integrity._assert_cache_required_ledger_valid)


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_permanent_funding_gaps() -> None:
    # 2026-09-15 실측(Binance Vision 원본 자체 공백, 재조회로 복구 불가)으로 확인된 심볼:
    # ICPUSDT(펀딩 이력 2022-09-01 시작), AIAUSDT(2025-12-11~2026-01-20 내부 공백),
    # OMNIUSDT(펀딩 이력 2025-09-22 종료, 선물 상장폐지).
    assert {"ICPUSDT", "AIAUSDT", "OMNIUSDT"} <= integrity.SOURCE_GAP_EXCLUDED_SYMBOLS
    assert isinstance(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS, frozenset)


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_permanent_ohlcv_gap() -> None:
    # 2026-09-15 실측: MAVIAUSDT는 2025-03-26 00:00~16:00 구간 3m/1h OHLCV가
    # Vision 월간 아카이브와 REST klines 양쪽 모두에 없다(진짜 소스 공백).
    assert "MAVIAUSDT" in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_bake_delisting() -> None:
    # 2026-09-15 실측: BAKEUSDT 펀딩 이력이 2025-10-03 08:00에 종료되고
    # exchangeInfo status=SETTLING·deliveryDate가 정확히 일치(선물 상장폐지).
    assert "BAKEUSDT" in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_settling_batch() -> None:
    # 2026-09-15 실측: exchangeInfo status=SETTLING으로 확인된 나머지 상장폐지 심볼
    # 전량. 펀딩 이력이 백테스트 구간(2025-12-31) 이전에 종료돼 좀비 꼬리 보유 시
    # MISSING_HELD_FUNDING을 유발한다.
    settling_batch = {
        "1000XUSDT", "AGIXUSDT", "AI16ZUSDT", "ALPACAUSDT", "ALPHAUSDT", "AMBUSDT", "BADGERUSDT", "BALUSDT",
        "BLZUSDT", "BONDUSDT", "BSWUSDT", "COMBOUSDT", "DARUSDT", "DEFIUSDT", "DGBUSDT", "FISUSDT",
        "FLMUSDT", "FTMUSDT", "FTTUSDT", "GLMRUSDT", "HIFIUSDT", "IDEXUSDT", "KDAUSDT", "KEYUSDT",
        "KLAYUSDT", "LEVERUSDT", "LINAUSDT", "LOKAUSDT", "LOOMUSDT", "MDTUSDT", "MEMEFIUSDT", "MILKUSDT",
        "MKRUSDT", "MYROUSDT", "NEIROETHUSDT", "NULSUSDT", "OBOLUSDT", "OCEANUSDT", "OMGUSDT", "ORBSUSDT",
        "PERPUSDT", "PONKEUSDT", "PORT3USDT", "QUICKUSDT", "RADUSDT", "RAYUSDT", "REEFUSDT", "REIUSDT",
        "RENUSDT", "SCUSDT", "SKATEUSDT", "SLERFUSDT", "SNTUSDT", "STMXUSDT", "STPTUSDT", "STRAXUSDT",
        "SWELLUSDT", "TOKENUSDT", "TROYUSDT", "UNFIUSDT", "UXLINKUSDT", "VIDTUSDT", "VOXELUSDT", "WAVESUSDT",
        "XCNUSDT", "XEMUSDT",
    }
    assert len(settling_batch) == 66
    assert settling_batch <= integrity.SOURCE_GAP_EXCLUDED_SYMBOLS
