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
