"""Tax-year summary invariants (spec 16 part 2)."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.tax_ledger import append_tax_records, read_tax_ledger
from src.live.tax_schema import (
    BOUNDARY_MARK_NOT_SUPPLIED,
    BoundaryMark,
    TaxCoverage,
    TaxRecord,
)
from src.live.tax_summary import (
    SYMBOL_DECIMAL_BUCKETS,
    TAX_SUMMARY_KEYS,
    TaxSummaryConfig,
    build_tax_year_summary,
    pre_regime_inventory_symbols,
    regime_boundary_utc,
    summarize_tax_year,
    tax_summary_to_json,
    tax_year_bounds,
    write_tax_summary,
)

CFG = TaxSummaryConfig(
    timezone="Asia/Seoul",
    regime_start=date(2027, 1, 1),
    settlement_asset="USDT",
    reconcile_abs_tolerance=Decimal("0.01"),
    reconcile_per_fill_tolerance=Decimal("0.00000001"),
)
FULL = TaxCoverage(
    genesis_at=pd.Timestamp("2026-11-01T00:00Z"),
    genesis_positions={},
    income_covered_from=pd.Timestamp("2026-08-01T00:00Z"),
    collected_through=pd.Timestamp("2028-01-05T00:00Z"),
    income_gaps=(),
)

_SEQ = [0]


def _trade(rid, *, side="BUY", qty="1", price="100", when="2027-03-01 00:00", source="simulated", mode="paper", vid=None, fee="0", pnl="0", symbol="BTCUSDT", fee_asset="USDT"):
    _SEQ[0] += 1
    return TaxRecord(
        record_id=rid, kind="TRADE", event_time=pd.Timestamp(when, tz="UTC"), symbol=symbol,
        side=side, quantity=Decimal(qty), price=Decimal(price), quote_qty=Decimal(qty) * Decimal(price),
        fee=Decimal(fee), fee_asset=fee_asset, realized_pnl=Decimal(pnl), income_asset="USDT",
        is_maker=False, venue_id=vid if vid is not None else _SEQ[0],
        source=source, mode=mode, position_side="" if source == "simulated" else "BOTH",
    )


def _income(rid, kind, amount, when, vid, itype, asset="USDT", symbol="BTCUSDT"):
    return TaxRecord(
        record_id=rid, kind=kind, event_time=pd.Timestamp(when, tz="UTC"), symbol=symbol,
        side="", quantity=Decimal(0), price=Decimal(0), quote_qty=Decimal(0), fee=Decimal(0),
        fee_asset="", realized_pnl=Decimal(amount), income_asset=asset, is_maker=False,
        venue_id=vid, source="venue", mode="live_testnet", income_type=itype,
    )


def _venue_short_leg():
    return [
        _trade("venue:TRADE:1", side="SELL", price="100", when="2027-02-01 00:00", source="venue", mode="live_testnet", vid=1, fee="0.02", pnl="0"),
        _trade("venue:TRADE:2", side="BUY", price="90", when="2027-03-01 00:00", source="venue", mode="live_testnet", vid=2, fee="0.018", pnl="10"),
        _income("venue:REALIZED_PNL:10", "REALIZED_PNL", "10", "2027-03-01 00:01", 10, "REALIZED_PNL"),
        _income("venue:COMMISSION:11", "COMMISSION", "-0.02", "2027-02-01 00:01", 11, "COMMISSION"),
        _income("venue:COMMISSION:12", "COMMISSION", "-0.018", "2027-03-01 00:02", 12, "COMMISSION"),
    ]


def test_kst_year_boundary():
    recs = [
        _trade("s1", when="2026-12-31T14:59:59Z"),
        _trade("s2", side="SELL", price="110", when="2026-12-31T15:00:00Z"),
    ]
    s26 = build_tax_year_summary(recs, 2026, source="simulated", config=CFG)
    s27 = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    assert s26["per_symbol"]["BTCUSDT"]["n_fills"] == 1
    assert s26["closing_inventory"]["BTCUSDT"]["quantity"] == Decimal(1)
    assert s27["per_symbol"]["BTCUSDT"]["trading_pnl"] == Decimal(10)
    assert s27["opening_inventory"]["BTCUSDT"]["quantity"] == Decimal(1)


def test_year_bounds_are_local_midnight():
    assert tax_year_bounds(2027, "Asia/Seoul") == (
        pd.Timestamp("2026-12-31T15:00Z"), pd.Timestamp("2027-12-31T15:00Z"))
    with pytest.raises(ValueError, match="year"):
        tax_year_bounds(True, "Asia/Seoul")
    with pytest.raises(ValueError, match="timezone"):
        tax_year_bounds(2027, "Mars/Olympus")


def test_venue_short_round_trip_not_double_counted():
    s = build_tax_year_summary(_venue_short_leg(), 2027, source="venue", config=CFG, coverage=FULL)
    e = s["per_symbol"]["BTCUSDT"]
    assert e["trading_pnl"] == e["trading_pnl_short"] == Decimal(10)
    assert e["trade_fees"] == Decimal("0.038")
    assert e["net_pnl"] == Decimal("9.962")
    assert s["closing_inventory"] == {}
    assert s["reconciliation"]["status"] == "reconciled"


def test_simulated_short_round_trip():
    recs = [
        _trade("s1", side="SELL", price="100", when="2027-02-01 00:00"),
        _trade("s2", side="BUY", price="90", when="2027-03-01 00:00"),
    ]
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    e = s["per_symbol"]["BTCUSDT"]
    assert (e["trading_pnl"], e["trade_fees"], e["net_pnl"]) == (Decimal(10), Decimal(0), Decimal(10))
    assert s["reconciliation"]["status"] == "not_applicable"
    assert s["reconciliation"]["coverage"] is None


def test_flip_realizes_both_legs():
    recs = [
        _trade("s1", price="100", when="2027-01-05 00:00"),
        _trade("s2", side="SELL", qty="2", price="110", when="2027-02-05 00:00"),
        _trade("s3", price="105", when="2027-03-05 00:00"),
    ]
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    e = s["per_symbol"]["BTCUSDT"]
    assert e["trading_pnl_long"] == Decimal(10)
    assert e["trading_pnl_short"] == Decimal(5)
    assert e["trading_pnl"] == Decimal(15)
    assert e["closed_quantity"] == Decimal(2)
    assert s["closing_inventory"] == {}


def test_carry_in_across_regime_boundary():
    recs = [
        _trade("s1", price="100", when="2026-12-20 00:00"),
        _trade("s2", side="SELL", price="120", when="2027-01-05 00:00"),
    ]
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    entry = s["opening_inventory"]["BTCUSDT"]
    assert entry["quantity"] == Decimal(1)
    assert entry["avg_entry"] == Decimal(100)
    assert entry["opened_before_regime"] is True
    assert entry["boundary_mark"] is None
    assert entry["boundary_mark_source"] is None
    assert entry["boundary_mark_unavailable_reason"] == BOUNDARY_MARK_NOT_SUPPLIED
    assert s["per_symbol"]["BTCUSDT"]["trading_pnl"] == Decimal(20)
    assert s["closing_inventory"] == {}


def test_prior_year_closing_inventory_with_operator_mark():
    recs = [
        _trade("s1", price="100", when="2026-12-20 00:00"),
        _trade("s2", side="SELL", price="120", when="2027-01-05 00:00"),
    ]
    marks = {"BTCUSDT": BoundaryMark(Decimal("105"), "operator_supplied")}
    s = build_tax_year_summary(recs, 2026, source="simulated", config=CFG, boundary_marks=marks)
    entry = s["closing_inventory"]["BTCUSDT"]
    assert entry["quantity"] == Decimal(1)
    assert entry["avg_entry"] == Decimal(100)
    assert entry["opened_before_regime"] is True
    assert entry["boundary_mark"] == Decimal("105")
    assert entry["boundary_mark_source"] == "operator_supplied"
    assert entry["boundary_mark_unavailable_reason"] is None
    assert s["per_symbol"]["BTCUSDT"]["trading_pnl"] == Decimal(0)


def test_boundary_mark_validation():
    recs = [_trade("s1", price="100", when="2026-12-20 00:00"), _trade("s2", side="SELL", price="120", when="2027-01-05 00:00")]
    with pytest.raises(ValueError, match="pre-regime inventory"):
        build_tax_year_summary(recs, 2027, source="simulated", config=CFG,
                               boundary_marks={"ETHUSDT": BoundaryMark(Decimal("1"), "operator_supplied")})
    for kwargs in [{"price": Decimal("0"), "source": "operator_supplied"},
                   {"price": None, "source": "operator_supplied"},
                   {"price": Decimal("1"), "source": "ohlcv_1h_close", "unavailable_reason": "x"},
                   {"price": Decimal("1"), "source": "guess"}]:
        with pytest.raises(ValueError, match=r"boundary mark|unknown|exclusively"):
            BoundaryMark(**kwargs)


def test_post_regime_position_not_flagged():
    recs = [_trade("s1", price="100", when="2027-03-01 00:00")]
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    entry = s["closing_inventory"]["BTCUSDT"]
    assert entry["opened_before_regime"] is False
    assert entry["boundary_mark"] is None
    assert entry["boundary_mark_source"] is None
    assert entry["boundary_mark_unavailable_reason"] is None
    with pytest.raises(ValueError, match="pre-regime inventory"):
        build_tax_year_summary(recs, 2027, source="simulated", config=CFG,
                               boundary_marks={"BTCUSDT": BoundaryMark(Decimal("1"), "operator_supplied")})


def test_carry_in_mark_survives_post_regime_reopening(tmp_path: Path):
    records = [
        _trade("carry", when="2026-12-20 00:00"),
        _trade("close", side="SELL", price="110", when="2027-02-01 00:00"),
        _trade("reopen", price="120", when="2027-03-01 00:00"),
    ]
    mark = BoundaryMark(Decimal("105"), "operator_supplied")
    result = build_tax_year_summary(
        records, 2027, source="simulated", config=CFG, boundary_marks={"BTCUSDT": mark},
    )
    assert result["opening_inventory"]["BTCUSDT"]["boundary_mark"] == Decimal("105")
    assert result["closing_inventory"]["BTCUSDT"]["opened_before_regime"] is False
    assert result["closing_inventory"]["BTCUSDT"]["boundary_mark"] is None
    assert result["totals"]["trading_pnl"] == Decimal("10")
    append_tax_records(records, tmp_path)
    derived = summarize_tax_year(
        2027, tmp_path, source="simulated", config=CFG,
        derive_boundary_marks=lambda symbols, boundary: {"BTCUSDT": mark},
    )
    assert derived == result


def test_unavailable_derived_mark_reported():
    recs = [_trade("s1", price="100", when="2026-12-20 00:00"), _trade("s2", side="SELL", price="120", when="2027-01-05 00:00")]
    plain = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    marks = {"BTCUSDT": BoundaryMark(None, "ohlcv_1h_close", "ohlcv_bar_missing")}
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG, boundary_marks=marks)
    entry = s["opening_inventory"]["BTCUSDT"]
    assert entry["boundary_mark"] is None
    assert entry["boundary_mark_source"] == "ohlcv_1h_close"
    assert entry["boundary_mark_unavailable_reason"] == "ohlcv_bar_missing"
    assert s["per_symbol"] == plain["per_symbol"]
    assert s["totals"] == plain["totals"]


def test_regime_boundary_instant():
    assert regime_boundary_utc(CFG) == pd.Timestamp("2026-12-31T15:00Z")
    assert pre_regime_inventory_symbols({"opening_inventory": {}, "closing_inventory": {}}) == frozenset()


def test_summarize_derives_marks_only_for_pre_regime(tmp_path: Path):
    from src.live.tax_ledger import append_tax_records as _append
    recs = [
        _trade("s1", price="100", when="2026-12-20 00:00", symbol="BTCUSDT"),
        _trade("s2", price="50", when="2027-02-01 00:00", symbol="ETHUSDT"),
    ]
    _append(recs, tmp_path)
    calls: list = []

    def _derive(symbols, boundary):
        calls.append((symbols, boundary))
        return {s: BoundaryMark(Decimal("98"), "ohlcv_1h_close") for s in symbols}

    s = summarize_tax_year(2027, tmp_path, source="simulated", config=CFG, derive_boundary_marks=_derive)
    assert calls == [(frozenset({"BTCUSDT"}), pd.Timestamp("2026-12-31T15:00Z"))]
    assert s["opening_inventory"]["BTCUSDT"]["boundary_mark"] == Decimal("98")
    assert s["closing_inventory"]["ETHUSDT"]["boundary_mark"] is None
    from src.live.tax_ledger import append_tax_records as _append2
    flat_dir = tmp_path / "flat"
    _append2([_trade("f1", price="50", when="2027-02-01 00:00", symbol="ETHUSDT")], flat_dir)
    flat_calls: list = []
    s2 = summarize_tax_year(2027, flat_dir, source="simulated", config=CFG,
                            derive_boundary_marks=lambda syms, b: flat_calls.append((syms, b)) or {})
    assert flat_calls == []
    with pytest.raises(ValueError, match="not both"):
        summarize_tax_year(2027, tmp_path, source="simulated", config=CFG,
                           boundary_marks={"BTCUSDT": BoundaryMark(Decimal("1"), "operator_supplied")},
                           derive_boundary_marks=_derive)
    with pytest.raises(ValueError, match="exactly"):
        summarize_tax_year(2027, tmp_path, source="simulated", config=CFG,
                           derive_boundary_marks=lambda syms, b: {})


def test_venue_facts_separated_probe_d():
    recs = [
        _trade("venue:TRADE:1", price="100", when="2027-02-01 00:00", source="venue", mode="live_testnet", vid=1, fee="0.0001", fee_asset="BNB", pnl="0"),
        _trade("venue:TRADE:2", side="SELL", price="110", when="2027-03-01 00:00", source="venue", mode="live_testnet", vid=2, fee="0.044", pnl="10"),
        _income("venue:REALIZED_PNL:3", "REALIZED_PNL", "10", "2027-03-01 00:01", 3, "REALIZED_PNL"),
        _income("venue:COMMISSION:4", "COMMISSION", "-0.0001", "2027-02-01 00:01", 4, "COMMISSION", asset="BNB"),
        _income("venue:COMMISSION:5", "COMMISSION", "-0.044", "2027-03-01 00:02", 5, "COMMISSION"),
        _income("venue:FUNDING_FEE:6", "FUNDING_FEE", "-0.3", "2027-04-01 00:00", 6, "FUNDING_FEE"),
        _income("venue:TRANSFER:7", "TRANSFER", "10000", "2027-05-01 00:00", 7, "TRANSFER", symbol=""),
    ]
    s = build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)
    e = s["per_symbol"]["BTCUSDT"]
    assert e["trading_pnl"] == Decimal(10)
    assert e["funding"] == Decimal("-0.3")
    assert e["trade_fees"] == Decimal("0.044")
    assert e["other_asset_fees"] == {"BNB": Decimal("0.0001")}
    assert s["transfers"] == {"TRANSFER": {"USDT": Decimal("10000")}}
    assert e["net_pnl"] == Decimal("9.656")
    assert s["reconciliation"]["status"] == "reconciled"


def test_missing_commission_raises_under_full_coverage():
    recs = [
        _trade("venue:TRADE:1", price="100", when="2027-02-01 00:00", source="venue", mode="live_testnet", vid=1, fee="0.0001", fee_asset="BNB", pnl="0"),
        _trade("venue:TRADE:2", side="SELL", price="110", when="2027-03-01 00:00", source="venue", mode="live_testnet", vid=2, fee="0.044", pnl="10"),
        _income("venue:REALIZED_PNL:3", "REALIZED_PNL", "10", "2027-03-01 00:01", 3, "REALIZED_PNL"),
        _income("venue:COMMISSION:4", "COMMISSION", "-0.0001", "2027-02-01 00:01", 4, "COMMISSION", asset="BNB"),
        _income("venue:FUNDING_FEE:6", "FUNDING_FEE", "-0.3", "2027-04-01 00:00", 6, "FUNDING_FEE"),
    ]
    with pytest.raises(DataIntegrityError, match="commission_mismatch"):
        build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)


def test_fold_vs_venue_mismatch_raises():
    recs = [
        _trade("venue:TRADE:2", side="SELL", price="90", when="2027-03-01 00:00", source="venue", mode="live_testnet", vid=2, fee="0", pnl="10"),
        _income("venue:REALIZED_PNL:3", "REALIZED_PNL", "10", "2027-03-01 00:01", 3, "REALIZED_PNL"),
        _income("venue:COMMISSION:5", "COMMISSION", "0", "2027-03-01 00:02", 5, "COMMISSION"),
    ]
    with pytest.raises(DataIntegrityError, match="fold_pnl_mismatch"):
        build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)


def test_mismatch_under_incomplete_is_reported():
    recs = [
        _trade("venue:TRADE:2", side="SELL", price="90", when="2027-03-01 00:00", source="venue", mode="live_testnet", vid=2, fee="0", pnl="10"),
        _income("venue:REALIZED_PNL:3", "REALIZED_PNL", "10", "2027-03-01 00:01", 3, "REALIZED_PNL"),
        _income("venue:COMMISSION:5", "COMMISSION", "0", "2027-03-01 00:02", 5, "COMMISSION"),
    ]
    import dataclasses
    gap_cov = dataclasses.replace(FULL, income_gaps=((pd.Timestamp("2027-06-01T00:00Z"), pd.Timestamp("2027-06-02T00:00Z")),))
    s = build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=gap_cov)
    assert s["reconciliation"]["status"] == "incomplete"
    codes = {i["code"] for i in s["reconciliation"]["issues"]}
    assert {"income_gap", "fold_pnl_mismatch"} <= codes
    assert s["per_symbol"]["BTCUSDT"]["trading_pnl_unattributed"] == Decimal(10)


def test_coverage_blockers_mark_incomplete():
    import dataclasses
    base = _venue_short_leg()
    cases = [
        (None, "coverage_unknown"),
        (dataclasses.replace(FULL, genesis_at=None), "genesis_missing"),
        (dataclasses.replace(FULL, genesis_positions={"BTCUSDT": Decimal(1)}), "genesis_not_flat"),
        (dataclasses.replace(FULL, genesis_at=pd.Timestamp("2027-02-01T00:00Z")), "period_precedes_genesis"),
        (dataclasses.replace(FULL, income_covered_from=None), "income_coverage_unknown"),
        (dataclasses.replace(FULL, income_covered_from=pd.Timestamp("2026-12-01T00:00Z")), "income_coverage_late"),
        (dataclasses.replace(FULL, collected_through=pd.Timestamp("2027-12-31T14:00Z")), "collection_behind_period_end"),
    ]
    for cov, code in cases:
        s = build_tax_year_summary(base, 2027, source="venue", config=CFG, coverage=cov)
        assert s["reconciliation"]["status"] == "incomplete", code
        assert code in {i["code"] for i in s["reconciliation"]["issues"]}, code


def test_pre_genesis_history_discarded_at_genesis():
    recs = [
        _trade("venue:TRADE:0", price="100", when="2026-10-25 00:00", source="venue", mode="live_testnet", vid=0, fee="0", pnl="0"),
        *_venue_short_leg(),
    ]
    s27 = build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)
    assert s27["opening_inventory"] == {}
    assert s27["reconciliation"]["status"] == "reconciled"
    s26 = build_tax_year_summary(recs, 2026, source="venue", config=CFG, coverage=FULL)
    assert "pre_genesis_inventory_discarded" in {i["code"] for i in s26["reconciliation"]["issues"]}
    assert s26["reconciliation"]["status"] == "incomplete"


def test_delivery_settlement_closes_position():
    recs = [
        _trade("venue:TRADE:1", qty="2", price="100", when="2027-02-01 00:00", source="venue", mode="live_testnet", vid=1, fee="0", pnl="0"),
        _income("venue:DELIVERED_SETTELMENT:9", "REALIZED_PNL", "12.5", "2027-06-01 00:00", 9, "DELIVERED_SETTELMENT"),
        _income("venue:COMMISSION:11", "COMMISSION", "0", "2027-02-01 00:01", 11, "COMMISSION"),
    ]
    s = build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)
    e = s["per_symbol"]["BTCUSDT"]
    assert e["trading_pnl"] == Decimal("12.5")
    assert e["delivery_settlement_pnl"] == Decimal("12.5")
    assert s["closing_inventory"] == {}
    assert s["other_income"] == {}
    assert s["reconciliation"]["status"] == "reconciled"


def test_unknown_income_itemized_outside_pnl():
    recs = [
        _income("venue:INSURANCE_CLEAR:1", "UNCLASSIFIED", "3", "2027-02-01 00:00", 1, "INSURANCE_CLEAR", symbol=""),
        _income("venue:COMMISSION_REBATE:2", "UNCLASSIFIED", "1", "2027-02-01 00:01", 2, "COMMISSION_REBATE", symbol=""),
    ]
    s = build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)
    assert s["other_income"] == {"INSURANCE_CLEAR": {"USDT": Decimal(3)}, "COMMISSION_REBATE": {"USDT": Decimal(1)}}
    assert s["totals"]["trading_pnl"] == Decimal(0)


def test_empty_ledger_stable_schema():
    for source, coverage, status in [("venue", None, "incomplete"), ("simulated", None, "not_applicable")]:
        s = build_tax_year_summary([], 2027, source=source, config=CFG, coverage=coverage)
        assert set(s) == set(TAX_SUMMARY_KEYS)
        assert s["per_symbol"] == {}
        assert s["totals"]["n_fills"] == 0
        assert all(s["totals"][k] == Decimal(0) for k in SYMBOL_DECIMAL_BUCKETS)
        assert s["mode"] is None
        assert s["reconciliation"]["status"] == status
    nonempty = build_tax_year_summary([_trade("s1")], 2027, source="simulated", config=CFG)
    assert set(nonempty) == set(TAX_SUMMARY_KEYS)


def test_exact_decimal_accumulation():
    recs = [_trade(f"s{i}", qty="0.1", price="0.1", when="2027-03-01 00:00", fee="0.1", vid=i) for i in range(1, 11)]
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    e = s["per_symbol"]["BTCUSDT"]
    assert e["trade_fees"] == Decimal("1.0")
    assert e["buy_notional"] == Decimal("0.10")
    assert e["buy_quantity"] == Decimal("1.0")


def test_conservation_per_symbol_and_totals():
    import dataclasses

    sim_funding = dataclasses.replace(
        _trade("simf", when="2027-03-01 00:00", symbol="BTCUSDT"),
        record_id="sim:F1", kind="FUNDING_FEE", side="", quantity=Decimal(0),
        price=Decimal(0), quote_qty=Decimal(0), fee=Decimal(0),
        realized_pnl=Decimal("-0.2"), venue_id=0,
    )
    recs = [
        _trade("s1", price="100", when="2027-01-05 00:00", symbol="BTCUSDT"),
        _trade("s2", side="SELL", price="110", when="2027-02-05 00:00", symbol="BTCUSDT", fee="0.05"),
        _trade("s3", price="50", when="2027-01-06 00:00", symbol="ETHUSDT", fee="0.01", fee_asset="BNB"),
        sim_funding,
    ]
    s = build_tax_year_summary(recs, 2027, source="simulated", config=CFG)
    for key in SYMBOL_DECIMAL_BUCKETS:
        assert s["totals"][key] == sum(e[key] for e in s["per_symbol"].values())
    for e in s["per_symbol"].values():
        assert e["trading_pnl"] == e["trading_pnl_long"] + e["trading_pnl_short"] + e["trading_pnl_unattributed"]
        assert e["net_pnl"] == e["trading_pnl"] + e["funding"] - e["trade_fees"]


def test_future_records_do_not_alter_year():
    base = [_trade("s1", price="100", when="2027-01-05 00:00"), _trade("s2", side="SELL", price="120", when="2027-02-05 00:00")]
    extra = [*base, _trade("s9", price="999", when="2027-12-31T15:00:00Z")]
    assert build_tax_year_summary(base, 2027, source="simulated", config=CFG) == build_tax_year_summary(extra, 2027, source="simulated", config=CFG)


def test_input_order_independence():
    recs = [_trade("s1", price="100", when="2027-01-05 00:00"), _trade("s2", side="SELL", price="120", when="2027-02-05 00:00")]
    assert build_tax_year_summary(recs, 2027, source="simulated", config=CFG) == build_tax_year_summary(list(reversed(recs)), 2027, source="simulated", config=CFG)


def test_ledger_purity_fails_closed():
    venue = _trade("v1", when="2027-01-05 00:00", source="venue", mode="live_testnet", vid=1)
    sim = _trade("s1", when="2027-01-06 00:00")
    with pytest.raises(DataIntegrityError):
        build_tax_year_summary([venue, sim], 2027, source="venue", config=CFG, coverage=FULL)
    other_mode = _trade("v2", when="2028-01-05 00:00", source="venue", mode="live_mainnet", vid=2)
    with pytest.raises(DataIntegrityError):
        build_tax_year_summary([venue, other_mode], 2027, source="venue", config=CFG, coverage=FULL)
    bad_funding = _income("venue:FUNDING_FEE:1", "FUNDING_FEE", "1", "2028-06-01 00:00", 1, "FUNDING_FEE", asset="BUSD")
    with pytest.raises(DataIntegrityError):
        build_tax_year_summary([venue, bad_funding], 2027, source="venue", config=CFG, coverage=FULL)
    bad_trade = _trade("v3", when="2028-06-01 00:00", source="venue", mode="live_testnet", vid=3)
    object.__setattr__(bad_trade, "income_asset", "BNB")
    with pytest.raises(DataIntegrityError):
        build_tax_year_summary([venue, bad_trade], 2027, source="venue", config=CFG, coverage=FULL)
    bad_kind = _trade("s9", when="2028-01-01 00:00")
    object.__setattr__(bad_kind, "kind", "COMMISSION")
    with pytest.raises(DataIntegrityError):
        build_tax_year_summary([sim, bad_kind], 2027, source="simulated", config=CFG)


def test_only_moving_average_and_valid_arguments():
    recs = [_trade("s1")]
    with pytest.raises(ValueError, match="cost basis"):
        build_tax_year_summary(recs, 2027, source="simulated", config=CFG, cost_basis="fifo")
    with pytest.raises(ValueError, match="source"):
        build_tax_year_summary(recs, 2027, source="paper", config=CFG)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coverage"):
        build_tax_year_summary(recs, 2027, source="simulated", config=CFG, coverage=FULL)


def test_no_tax_computation_keys():
    s = build_tax_year_summary(_venue_short_leg(), 2027, source="venue", config=CFG, coverage=FULL)
    blob = json.dumps(tax_summary_to_json(s))
    for banned in ("tax_rate", "deduction", "taxable_income", "income_classification"):
        assert banned not in blob
    assert "income_type" not in s


def test_json_is_lossless():
    s = build_tax_year_summary([_trade("s1", fee="0.00000001", vid=1)], 2027, source="simulated", config=CFG)
    text = tax_summary_to_json(s)
    assert '"0.00000001"' in text
    back = json.loads(text, parse_float=Decimal)
    assert back["per_symbol"]["BTCUSDT"]["trade_fees"] == "0.00000001"
    with pytest.raises(TypeError):
        tax_summary_to_json({"bad": object()})


def test_ledger_loader_path(tmp_path: Path):
    recs = [_trade("s1", price="100", when="2026-12-20 00:00"), _trade("s2", side="SELL", price="120", when="2027-01-05 00:00")]
    append_tax_records(recs, tmp_path)
    assert summarize_tax_year(2027, tmp_path, source="simulated", config=CFG) == build_tax_year_summary(
        read_tax_ledger(tmp_path), 2027, source="simulated", config=CFG)
    out = write_tax_summary(summarize_tax_year(2027, tmp_path, source="simulated", config=CFG), tmp_path / "out" / "s.json")
    assert out.exists()
    assert json.loads(out.read_text())["year"] == 2027


def test_summary_config_validation() -> None:
    import dataclasses

    with pytest.raises(ValueError, match="timezone"):
        TaxSummaryConfig(timezone="Mars/Olympus", regime_start=date(2027, 1, 1), settlement_asset="USDT",
                         reconcile_abs_tolerance=Decimal("0.01"), reconcile_per_fill_tolerance=Decimal("0.00000001"))
    with pytest.raises(ValueError, match="settlement_asset"):
        dataclasses.replace(CFG, settlement_asset="")
    with pytest.raises(ValueError, match="settlement_asset"):
        dataclasses.replace(CFG, settlement_asset="usdt")
    with pytest.raises(ValueError, match="tolerance"):
        dataclasses.replace(CFG, reconcile_abs_tolerance=Decimal("-1"))
    assert TaxSummaryConfig.from_settings(__import__("src.live.settings", fromlist=["LiveSettings"]).LiveSettings()).timezone == "Asia/Seoul"


def test_source_for_mode() -> None:
    from src.live.settings import ExecutionMode
    from src.live.tax_summary import tax_source_for_mode

    assert tax_source_for_mode(ExecutionMode.SHADOW) == "simulated"
    assert tax_source_for_mode(ExecutionMode.PAPER) == "simulated"
    assert tax_source_for_mode(ExecutionMode.LIVE_TESTNET) == "venue"
    assert tax_source_for_mode(ExecutionMode.LIVE_MAINNET) == "venue"


def test_year_must_be_int_not_bool() -> None:
    with pytest.raises(ValueError, match="year"):
        build_tax_year_summary([_trade("s1")], True, source="simulated", config=CFG)


def test_income_only_symbol_reconciles() -> None:
    recs = [
        _trade("venue:TRADE:1", price="100", when="2027-02-01 00:00", source="venue", mode="live_testnet", vid=1, fee="0", pnl="0"),
        _income("venue:COMMISSION:9", "COMMISSION", "0", "2027-02-01 00:01", 9, "COMMISSION", symbol="ETHUSDT"),
    ]
    s = build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)
    assert s["reconciliation"]["status"] == "reconciled"
    assert s["reconciliation"]["per_symbol"]["ETHUSDT"]["income_realized_pnl"] == Decimal(0)


def test_income_pnl_mismatch_raises() -> None:
    recs = [
        _trade("venue:TRADE:1", side="SELL", price="100", when="2027-02-01 00:00", source="venue", mode="live_testnet", vid=1, fee="0", pnl="0"),
        _trade("venue:TRADE:2", side="BUY", price="90", when="2027-03-01 00:00", source="venue", mode="live_testnet", vid=2, fee="0", pnl="10"),
        _income("venue:REALIZED_PNL:3", "REALIZED_PNL", "12", "2027-03-01 00:01", 3, "REALIZED_PNL"),
        _income("venue:COMMISSION:4", "COMMISSION", "0", "2027-02-01 00:01", 4, "COMMISSION"),
    ]
    with pytest.raises(DataIntegrityError, match="income_pnl_mismatch"):
        build_tax_year_summary(recs, 2027, source="venue", config=CFG, coverage=FULL)


def test_json_rejects_naive_datetime() -> None:
    from datetime import datetime

    with pytest.raises(TypeError, match=r"tz-aware pd\.Timestamp"):
        tax_summary_to_json({"when": datetime(2027, 1, 1, 0, 0, 0)})


def test_write_summary_tolerates_unfsyncable_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as _os

    summary = build_tax_year_summary([_trade("s1")], 2027, source="simulated", config=CFG)
    real_open = _os.open

    def _boom(path, flags, *args, **kwargs):
        if isinstance(path, str) and path == str((tmp_path / "out").resolve()):
            raise OSError(38, "Function not implemented")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(_os, "open", _boom)
    out = write_tax_summary(summary, tmp_path / "out" / "s.json")
    assert out.exists()


def test_operator_mark_without_price_rejected() -> None:
    with pytest.raises(ValueError, match="operator"):
        BoundaryMark(None, "operator_supplied", "ohlcv_bar_missing")


def test_future_settlement_does_not_change_historical_summary():
    records = [
        _trade("v1", when="2027-02-01 00:00", source="venue", mode="live_testnet"),
    ]
    future = _income("future", "REALIZED_PNL", "10", "2027-12-31T15:00:00Z", 9,
                     "DELIVERED_SETTELMENT", symbol="ETHUSDT")
    before = build_tax_year_summary(records, 2027, source="venue", config=CFG, coverage=FULL)
    after = build_tax_year_summary([*records, future], 2027, source="venue", config=CFG, coverage=FULL)
    assert after == before


def test_pre_genesis_realizations_are_preserved():
    records = [
        _trade("buy", when="2026-10-01 00:00", source="venue", mode="live_testnet", fee="0.02"),
        _trade("sell", side="SELL", price="110", when="2026-10-02 00:00", source="venue",
               mode="live_testnet", pnl="10", fee="0.022"),
        _income("pnl", "REALIZED_PNL", "10", "2026-10-02 00:01", 10, "REALIZED_PNL"),
    ]
    result = build_tax_year_summary(records, 2026, source="venue", config=CFG, coverage=FULL)
    assert result["totals"]["n_fills"] == 2
    assert result["totals"]["trading_pnl"] == Decimal("10")
    assert result["totals"]["closed_quantity"] == Decimal("1")
    assert result["totals"]["net_pnl"] == Decimal("9.958")
    assert result["closing_inventory"] == {}
    assert result["reconciliation"]["per_symbol"]["BTCUSDT"]["fold_trading_pnl"] == Decimal("10")


def test_future_genesis_cannot_erase_past_inventory():
    from dataclasses import replace

    records = [_trade("carry", when="2026-12-20 00:00", source="venue", mode="live_testnet")]
    coverage = replace(FULL, genesis_at=pd.Timestamp("2028-02-01T00:00Z"))
    result = build_tax_year_summary(records, 2027, source="venue", config=CFG, coverage=coverage)
    assert result["opening_inventory"]["BTCUSDT"]["quantity"] == Decimal("1")
    assert result["closing_inventory"]["BTCUSDT"]["quantity"] == Decimal("1")


def test_inventory_before_and_after_genesis_reset():
    from dataclasses import replace

    records = [_trade("carry", when="2026-12-20 00:00", source="venue", mode="live_testnet")]
    coverage = replace(FULL, genesis_at=pd.Timestamp("2027-02-01T00:00Z"))
    result = build_tax_year_summary(records, 2027, source="venue", config=CFG, coverage=coverage)
    assert result["opening_inventory"]["BTCUSDT"]["quantity"] == Decimal("1")
    assert result["closing_inventory"] == {}
    assert "pre_genesis_inventory_discarded" in {item["code"] for item in result["reconciliation"]["issues"]}


def test_genesis_at_year_boundary_preserves_prior_closing_inventory():
    from dataclasses import replace

    records = [_trade("carry", when="2026-12-20 00:00", source="venue", mode="live_testnet")]
    coverage = replace(FULL, genesis_at=pd.Timestamp("2026-12-31T15:00Z"))
    previous = build_tax_year_summary(records, 2026, source="venue", config=CFG, coverage=coverage)
    following = build_tax_year_summary(records, 2027, source="venue", config=CFG, coverage=coverage)
    assert previous["closing_inventory"]["BTCUSDT"]["quantity"] == Decimal("1")
    assert following["opening_inventory"] == {}
    assert following["closing_inventory"] == {}


@pytest.mark.parametrize("amounts", [("0",), ("-0.2", "0.2")])
def test_zero_net_funding_retains_symbol(amounts):
    from dataclasses import replace

    records = [replace(_income(f"f{i}", "FUNDING_FEE", amount, "2027-02-01 00:00", i,
                               "FUNDING_FEE"), source="simulated", mode="paper", income_type="")
               for i, amount in enumerate(amounts)]
    result = build_tax_year_summary(records, 2027, source="simulated", config=CFG)
    assert set(result["per_symbol"]) == {"BTCUSDT"}
    assert result["per_symbol"]["BTCUSDT"]["funding"] == Decimal("0")
    assert result["totals"]["net_pnl"] == Decimal("0")


def test_missing_genesis_reports_all_known_coverage_failures():
    coverage = TaxCoverage(None, {"BTCUSDT": Decimal("1")}, None, None, ())
    result = build_tax_year_summary([], 2027, source="venue", config=CFG, coverage=coverage)
    assert result["reconciliation"]["status"] == "incomplete"
    assert {issue["code"] for issue in result["reconciliation"]["issues"]} == {
        "genesis_missing", "genesis_not_flat", "income_coverage_unknown", "collection_behind_period_end",
    }
