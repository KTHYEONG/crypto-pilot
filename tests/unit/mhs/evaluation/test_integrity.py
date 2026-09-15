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
    # mhs_symbol_lifespan_pit_roster: ICPUSDT(조기 시작 공백)는 제외 유지.
    # AIAUSDT/OMNIUSDT는 말기(end-of-life)라 ledger_terminal_only가 finalize 시점에
    # 인증하므로 whole-history 배제에서 제거됨.
    assert "ICPUSDT" in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS
    assert {"AIAUSDT", "OMNIUSDT"}.isdisjoint(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS)
    assert isinstance(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS, frozenset)


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_permanent_ohlcv_gap() -> None:
    # 2026-09-15 실측: MAVIAUSDT는 2025-03-26 00:00~16:00 구간 3m/1h OHLCV가
    # Vision 월간 아카이브와 REST klines 양쪽 모두에 없다(진짜 소스 공백).
    assert "MAVIAUSDT" in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_bake_delisting() -> None:
    # mhs_symbol_lifespan_pit_roster: BAKEUSDT(선물 상장폐지, 말기 공백)는
    # ledger_terminal_only가 finalize 시점에 인증하므로 whole-history 배제에서 제거됨.
    assert "BAKEUSDT" not in integrity.SOURCE_GAP_EXCLUDED_SYMBOLS


def test_source_gap_excluded_symbols_covers_2026_09_confirmed_settling_batch() -> None:
    # mhs_symbol_lifespan_pit_roster: 말기(end-of-life) 50개는 ledger_terminal_only가
    # finalize 시점에 인증하므로 whole-history 배제에서 제거됨. Block1(OHLCV 재수집
    # 후보) 잔류분만 제외 목록에 남는다.
    removed_end_of_life = {
        "1000XUSDT", "AGIXUSDT", "AI16ZUSDT", "ALPACAUSDT", "AMBUSDT", "BALUSDT",
        "BLZUSDT", "BONDUSDT", "COMBOUSDT", "DARUSDT", "DEFIUSDT", "DGBUSDT", "FISUSDT",
        "FTMUSDT", "GLMRUSDT", "HIFIUSDT", "KDAUSDT", "KEYUSDT",
        "LEVERUSDT", "LINAUSDT", "LOKAUSDT", "LOOMUSDT", "MDTUSDT", "MEMEFIUSDT", "MILKUSDT",
        "MYROUSDT", "NEIROETHUSDT", "ORBSUSDT",
        "PERPUSDT", "PONKEUSDT", "PORT3USDT", "QUICKUSDT", "RADUSDT", "RAYUSDT", "REEFUSDT", "REIUSDT",
        "RENUSDT", "SCUSDT", "SKATEUSDT", "SNTUSDT", "STMXUSDT", "STPTUSDT",
        "SWELLUSDT", "TOKENUSDT", "UXLINKUSDT", "VOXELUSDT", "XCNUSDT",
    }
    assert removed_end_of_life.isdisjoint(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS)


def test_source_gap_excluded_symbols_after_ohlcv_recollection_sweep() -> None:
    # 2026-09-15 후속 재수집: ensure_ohlcv_data로 재수집한 결과 Block1(OHLCV 캐시
    # 없음/미미) 21개 중 19개는 backtest 구간 내부공백 0건으로 완전히 배제 해제됨
    # (말기 종료는 ledger_terminal_only가 처리). CVXUSDT/SLPUSDT는 재수집 후에도
    # 2025-06-19~2025-07-23 펀딩 공백이 재개되는 진짜 불확실성 구간이 드러나 남는다.
    recovered = {
        "ALPHAUSDT", "BADGERUSDT", "BSWUSDT", "FLMUSDT", "FTTUSDT", "IDEXUSDT",
        "KLAYUSDT", "MKRUSDT", "NULSUSDT", "OBOLUSDT", "OCEANUSDT", "OMGUSDT",
        "SLERFUSDT", "STRAXUSDT", "TROYUSDT", "UNFIUSDT", "VIDTUSDT", "WAVESUSDT", "XEMUSDT",
    }
    assert recovered.isdisjoint(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS)
    assert {"CVXUSDT", "SLPUSDT"} <= integrity.SOURCE_GAP_EXCLUDED_SYMBOLS
    assert len(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS) == 10


def test_funding_gap_terminal_symbols_accepts_gap_with_no_later_fill() -> None:
    from src.mhs.evaluation.integrity import _funding_gap_terminal_symbols

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, ts in rows],
            "symbol": [sym for sym, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": ["timeout_taker"] * len(rows),
            "pre_trade_equity": [1.0] * len(rows),
        })

    def _result(primary_valid, gaps, fills):
        equity = pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity, net_returns=equity * 0.0, simulated_units=None,
            mark_to_market_pnl=equity * 0.0, funding_charge=equity * 0.0, fee_charge=equity * 0.0,
            fill_turnover=equity * 0.0, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            primary_valid=primary_valid, invalid_reasons=() if primary_valid else ("MISSING_DATA",),
            data_gaps=tuple(gaps),
        )
        return StrategyExecutionReplayResult(
            simulated_fills=fills, ledger=ledger, simulated_units=pd.DataFrame(), simulated_notional_weights=pd.DataFrame(),
            fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            submit_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
            fill_count=0, unfilled_count=0, fallback_count=0, all_intent_shortfall_bps=0.0,
            forced_exit_count=0, forced_exit_notional=0.0,
            termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
            unsupported_assumptions=(), elapsed_seconds=0.0, data_gaps=tuple(gaps),
        )

    # Given: HIFIUSDT held-funding-unknown gaps with no fill afterward (delisted, never trades again)
    gaps = [
        _gap("MISSING_HELD_FUNDING", "HIFIUSDT", "2025-10-03 09:00"),
        _gap("MISSING_HELD_FUNDING", "HIFIUSDT", "2025-10-03 10:00"),
        _gap("UNKNOWN_TERMINATION", "HIFIUSDT", "2025-12-31 00:00"),
    ]
    fills = _fills([("HIFIUSDT", "2025-10-03 08:00")])  # only the entry fill, before the gap
    # When
    terminal = _funding_gap_terminal_symbols(gaps, fills)
    # Then
    assert terminal == frozenset({"HIFIUSDT"})

def test_funding_gap_terminal_symbols_excludes_symbol_with_later_fill() -> None:
    from src.mhs.evaluation.integrity import _funding_gap_terminal_symbols

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, ts in rows],
            "symbol": [sym for sym, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": ["timeout_taker"] * len(rows),
            "pre_trade_equity": [1.0] * len(rows),
        })

    def _result(primary_valid, gaps, fills):
        equity = pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity, net_returns=equity * 0.0, simulated_units=None,
            mark_to_market_pnl=equity * 0.0, funding_charge=equity * 0.0, fee_charge=equity * 0.0,
            fill_turnover=equity * 0.0, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            primary_valid=primary_valid, invalid_reasons=() if primary_valid else ("MISSING_DATA",),
            data_gaps=tuple(gaps),
        )
        return StrategyExecutionReplayResult(
            simulated_fills=fills, ledger=ledger, simulated_units=pd.DataFrame(), simulated_notional_weights=pd.DataFrame(),
            fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            submit_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
            fill_count=0, unfilled_count=0, fallback_count=0, all_intent_shortfall_bps=0.0,
            forced_exit_count=0, forced_exit_notional=0.0,
            termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
            unsupported_assumptions=(), elapsed_seconds=0.0, data_gaps=tuple(gaps),
        )

    # Given: AIAUSDT-style gap where funding later resumes and the symbol trades again
    gaps = [
        _gap("MISSING_HELD_FUNDING", "AIAUSDT", "2025-12-11 13:00"),
        _gap("MISSING_HELD_FUNDING", "AIAUSDT", "2025-12-11 14:00"),
    ]
    fills = _fills([("AIAUSDT", "2025-12-11 12:00"), ("AIAUSDT", "2026-01-20 08:00")])  # resumes after the gap
    # When
    terminal = _funding_gap_terminal_symbols(gaps, fills)
    # Then: not terminal -- a later fill proves funding coverage (and trading) resumed
    assert terminal == frozenset()


def test_funding_gap_terminal_symbols_ignores_delist_settlement_as_recovery_evidence() -> None:
    # 2026-09-15 실측(mhs_symbol_lifespan_pit_roster 후속): OMNIUSDT/BAKEUSDT/AIAUSDT가
    # _settle_idle_holdings의 delist_settlement 체결을 "재개"로 오판해 folds_passed가
    # 16/16 -> 14/16으로 퇴행했던 회귀. delist_settlement는 그 자체가 종료 처분이지
    # 정상 거래 재개의 증거가 아니다.
    from src.mhs.evaluation.integrity import _funding_gap_terminal_symbols

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, _, ts in rows],
            "symbol": [sym for sym, _, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": [reason for _, reason, _ in rows],
            "pre_trade_equity": [1.0] * len(rows),
        })

    # Given: OMNIUSDT held funding-unknown, then closed only via delist_settlement afterward
    gaps = [
        _gap("MISSING_HELD_FUNDING", "OMNIUSDT", "2025-09-22 01:00"),
        _gap("MISSING_HELD_FUNDING", "OMNIUSDT", "2025-09-23 01:00"),
    ]
    fills = _fills([
        ("OMNIUSDT", "timeout_taker", "2025-09-22 00:00"),
        ("OMNIUSDT", "delist_settlement", "2025-09-24 01:06"),
    ])
    # When
    terminal = _funding_gap_terminal_symbols(gaps, fills)
    # Then: still terminal -- the delist_settlement fill does not count as resumed trading
    assert terminal == frozenset({"OMNIUSDT"})


def test_funding_gap_terminal_symbols_empty_gaps_and_missing_columns_are_safe() -> None:
    import pandas as pd
    from src.mhs.evaluation.integrity import _funding_gap_terminal_symbols
    # Given: no MISSING_HELD_FUNDING gaps at all
    assert _funding_gap_terminal_symbols((), pd.DataFrame()) == frozenset()
    # Given: a MISSING_HELD_FUNDING gap but an empty (columnless) fills frame -- must not KeyError
    from src.mhs.execution import ExecutionDataGap
    gaps = (ExecutionDataGap(code="MISSING_HELD_FUNDING", symbol="X", timestamp=pd.Timestamp("2025-01-01", tz="UTC")),)
    assert _funding_gap_terminal_symbols(gaps, pd.DataFrame()) == frozenset({"X"})

def test_ledger_terminal_only_accepts_mixed_unknown_termination_and_terminal_funding_gap() -> None:
    from src.mhs.evaluation.integrity import ledger_terminal_only

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, ts in rows],
            "symbol": [sym for sym, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": ["timeout_taker"] * len(rows),
            "pre_trade_equity": [1.0] * len(rows),
        })

    def _result(primary_valid, gaps, fills):
        equity = pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity, net_returns=equity * 0.0, simulated_units=None,
            mark_to_market_pnl=equity * 0.0, funding_charge=equity * 0.0, fee_charge=equity * 0.0,
            fill_turnover=equity * 0.0, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            primary_valid=primary_valid, invalid_reasons=() if primary_valid else ("MISSING_DATA",),
            data_gaps=tuple(gaps),
        )
        return StrategyExecutionReplayResult(
            simulated_fills=fills, ledger=ledger, simulated_units=pd.DataFrame(), simulated_notional_weights=pd.DataFrame(),
            fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            submit_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
            fill_count=0, unfilled_count=0, fallback_count=0, all_intent_shortfall_bps=0.0,
            forced_exit_count=0, forced_exit_notional=0.0,
            termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
            unsupported_assumptions=(), elapsed_seconds=0.0, data_gaps=tuple(gaps),
        )

    # Given: one symbol permanently cut off (no later fill), another simply held to backtest end
    gaps = [
        _gap("MISSING_HELD_FUNDING", "HIFIUSDT", "2025-10-03 09:00"),
        _gap("UNKNOWN_TERMINATION", "HIFIUSDT", "2025-12-31 00:00"),
        _gap("UNKNOWN_TERMINATION", "BTCUSDT", "2025-12-31 00:00"),
    ]
    fills = _fills([("HIFIUSDT", "2025-10-03 08:00")])
    # When / Then
    assert ledger_terminal_only(gaps, fills) is True

def test_ledger_terminal_only_rejects_recovering_funding_gap_and_other_codes() -> None:
    from src.mhs.evaluation.integrity import ledger_terminal_only

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, ts in rows],
            "symbol": [sym for sym, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": ["timeout_taker"] * len(rows),
            "pre_trade_equity": [1.0] * len(rows),
        })

    def _result(primary_valid, gaps, fills):
        equity = pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity, net_returns=equity * 0.0, simulated_units=None,
            mark_to_market_pnl=equity * 0.0, funding_charge=equity * 0.0, fee_charge=equity * 0.0,
            fill_turnover=equity * 0.0, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            primary_valid=primary_valid, invalid_reasons=() if primary_valid else ("MISSING_DATA",),
            data_gaps=tuple(gaps),
        )
        return StrategyExecutionReplayResult(
            simulated_fills=fills, ledger=ledger, simulated_units=pd.DataFrame(), simulated_notional_weights=pd.DataFrame(),
            fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            submit_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
            fill_count=0, unfilled_count=0, fallback_count=0, all_intent_shortfall_bps=0.0,
            forced_exit_count=0, forced_exit_notional=0.0,
            termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
            unsupported_assumptions=(), elapsed_seconds=0.0, data_gaps=tuple(gaps),
        )

    # A recovering MISSING_HELD_FUNDING gap (later fill exists) must still block certification
    recovering = [_gap("MISSING_HELD_FUNDING", "AIAUSDT", "2025-12-11 13:00")]
    recovering_fills = _fills([("AIAUSDT", "2025-12-11 12:00"), ("AIAUSDT", "2026-01-20 08:00")])
    assert ledger_terminal_only(recovering, recovering_fills) is False
    # A non-funding, non-termination gap code must still block certification
    other = [_gap("ZERO_OR_UNKNOWN_VOLUME", "XUSDT", "2025-01-01 00:00")]
    assert ledger_terminal_only(other, _fills([])) is False
    # Empty gaps: preserves the pre-existing quirk (no gaps -> not terminal_only)
    assert ledger_terminal_only((), _fills([])) is False

def test_assert_cache_required_ledger_valid_accepts_terminal_funding_gap() -> None:
    from src.mhs.evaluation.integrity import _assert_cache_required_ledger_valid

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, ts in rows],
            "symbol": [sym for sym, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": ["timeout_taker"] * len(rows),
            "pre_trade_equity": [1.0] * len(rows),
        })

    def _result(primary_valid, gaps, fills):
        equity = pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity, net_returns=equity * 0.0, simulated_units=None,
            mark_to_market_pnl=equity * 0.0, funding_charge=equity * 0.0, fee_charge=equity * 0.0,
            fill_turnover=equity * 0.0, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            primary_valid=primary_valid, invalid_reasons=() if primary_valid else ("MISSING_DATA",),
            data_gaps=tuple(gaps),
        )
        return StrategyExecutionReplayResult(
            simulated_fills=fills, ledger=ledger, simulated_units=pd.DataFrame(), simulated_notional_weights=pd.DataFrame(),
            fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            submit_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
            fill_count=0, unfilled_count=0, fallback_count=0, all_intent_shortfall_bps=0.0,
            forced_exit_count=0, forced_exit_notional=0.0,
            termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
            unsupported_assumptions=(), elapsed_seconds=0.0, data_gaps=tuple(gaps),
        )

    gaps = [
        _gap("MISSING_HELD_FUNDING", "HIFIUSDT", "2025-10-03 09:00"),
        _gap("UNKNOWN_TERMINATION", "HIFIUSDT", "2025-12-31 00:00"),
    ]
    fills = _fills([("HIFIUSDT", "2025-10-03 08:00")])
    primary = _result(False, gaps, fills)
    # When / Then: must not raise
    _assert_cache_required_ledger_valid("slow_momentum", primary)

def test_assert_cache_required_ledger_valid_rejects_recovering_funding_gap() -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _assert_cache_required_ledger_valid

    import pandas as pd
    from src.mhs.execution import ExecutionDataGap
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    def _gap(code, symbol, ts):
        return ExecutionDataGap(code=code, symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"))

    def _fills(rows):
        return pd.DataFrame({
            "timestamp": [pd.Timestamp(ts, tz="UTC") for _, ts in rows],
            "symbol": [sym for sym, _ in rows],
            "quantity_delta": [1.0] * len(rows),
            "fill_price": [1.0] * len(rows),
            "fee_bps": [0.0] * len(rows),
            "reason": ["timeout_taker"] * len(rows),
            "pre_trade_equity": [1.0] * len(rows),
        })

    def _result(primary_valid, gaps, fills):
        equity = pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity, net_returns=equity * 0.0, simulated_units=None,
            mark_to_market_pnl=equity * 0.0, funding_charge=equity * 0.0, fee_charge=equity * 0.0,
            fill_turnover=equity * 0.0, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            primary_valid=primary_valid, invalid_reasons=() if primary_valid else ("MISSING_DATA",),
            data_gaps=tuple(gaps),
        )
        return StrategyExecutionReplayResult(
            simulated_fills=fills, ledger=ledger, simulated_units=pd.DataFrame(), simulated_notional_weights=pd.DataFrame(),
            fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="OHLCV_CLOSE_FALLBACK",
            submit_times=pd.Series(dtype="datetime64[ns, UTC]"), fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
            fill_count=0, unfilled_count=0, fallback_count=0, all_intent_shortfall_bps=0.0,
            forced_exit_count=0, forced_exit_notional=0.0,
            termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
            unsupported_assumptions=(), elapsed_seconds=0.0, data_gaps=tuple(gaps),
        )

    gaps = [_gap("MISSING_HELD_FUNDING", "AIAUSDT", "2025-12-11 13:00")]
    fills = _fills([("AIAUSDT", "2025-12-11 12:00"), ("AIAUSDT", "2026-01-20 08:00")])
    primary = _result(False, gaps, fills)
    with pytest.raises(DataIntegrityError, match="ledger invalid for blend"):
        _assert_cache_required_ledger_valid("blend", primary)

def test_source_gap_excluded_symbols_no_longer_blanket_excludes_resolved_end_of_life_symbols() -> None:
    import src.mhs.evaluation.integrity as integrity
    # 2026-09-15 실측(mhs_symbol_lifespan_pit_roster): 이 심볼들은 ledger_terminal_only가
    # finalize 시점에 인증을 통과시키므로 더 이상 whole-history 배제가 필요 없다.
    resolved = {
        "BAKEUSDT", "HIFIUSDT", "OMNIUSDT", "AIAUSDT", "AGIXUSDT", "ALPACAUSDT", "FTMUSDT",
    }
    assert resolved.isdisjoint(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS)
    # 2026-09-15 후속 재수집 스윕(test_source_gap_excluded_symbols_after_ohlcv_recollection_sweep)
    # 이후 잔여 10개(영구 OHLCV공백/중간공백/조기시작공백)만 그대로 배제된다.
    assert len(integrity.SOURCE_GAP_EXCLUDED_SYMBOLS) == 10
    assert {"LITUSDT", "PUMPUSDT", "ICPUSDT", "BNXUSDT", "MAVIAUSDT"} <= integrity.SOURCE_GAP_EXCLUDED_SYMBOLS

