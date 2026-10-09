"""Invariant scenarios for account persistence and statistics."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.strategy_account import (
    run_account_replay,
)
from tests.unit.application._strategy_account_helpers import (
    _GOLDEN_ACCOUNT_CASES,
    _PINNED_NOW,
    _PRE_REFACTOR,
    _account_request,
    _install_strategy_account_fakes,
    _run_dirs,
    _write_same_book_reference,
)


def test_account_payload_golden_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth/taker run persists the exact account.json key sets and sign conventions."""
    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"kind": "mhs", "strategy_id": "x", "base_cagr": 0.1}) + "\n", encoding="utf-8")
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/old", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path))
    assert len(seen["replays"]) == 3
    (run_dir,) = _run_dirs(tmp_path)
    payload = json.loads((run_dir / "account.json").read_text(encoding="utf-8"))
    assert payload.pop("statistics_limitations") == ["ACCOUNT_FUNDING_ATTRIBUTION_UNAVAILABLE"]
    assert payload.pop("stress_execution")["taker_fee_bps"] == 18.0
    assert set(payload) == {"strategy_id", "capital", "execution", "policy", "venue_captured_at", "venue_path", "evaluation_start", "evaluation_end", "cagr", "mdd", "daily_mdd", "final_equity", "liquidated_at", "mean_exposure", "min_exposure", "last_exposure", "skipped_orders", "untraded_fraction", "initial_margin_breaches", "fee_paid", "impact_paid", "funding_paid", "fallback_ladder_symbols", "missing_filter_symbols", "moment_source", "entry_anchor", "unit_reference", "venue_rules_applied_retroactively", "reconciliation", "created_at", "statistics", "design_data_cutoff"}
    assert set(payload["execution"]) == {"mode", "maker_fee_bps", "taker_fee_bps", "passive_window_bars", "maker_fill_fraction"}
    assert set(payload["policy"]) == {"kind", "exposure_max", "exposure_step", "mean_haircut", "prior_days", "min_moment_days", "shock_per_unit", "margin_reserve", "initial_margin_cap", "impact_y"}
    assert set(payload["unit_reference"]) == {"capital", "cagr", "mdd", "daily_mdd", "maker_fill_fraction"}
    assert set(payload["reconciliation"]) == {"status", "fixed_exposure", "capital", "order_filters", "impact_y", "cagr", "mdd", "mdd_convention", "mdd_definition", "cagr_tolerance", "mdd_tolerance", "reference_canonical", "cagr_gap", "mdd_gap"}
    assert set(payload["reconciliation"]["reference_canonical"]) == {"strategy_id", "run_dir", "evaluation_start", "evaluation_end", "base_cagr", "base_max_drawdown", "name_clip", "exposure_multiplier"}
    assert payload["mdd"] == pytest.approx(-0.05)
    assert payload["unit_reference"]["mdd"] == pytest.approx(-0.05)
    assert payload["reconciliation"]["mdd"] == pytest.approx(0.05)
    assert payload["reconciliation"]["mdd_convention"] == "magnitude"
    assert "error_type" not in json.dumps(payload)
    daily = pd.read_parquet(run_dir / "account_daily.parquet")
    assert list(daily.columns) == ["equity", "exposure"]
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    assert last["kind"] == "mhs_frozen_account"
    assert last["base_max_drawdown"] == pytest.approx(0.05)

def test_account_run_directory_suffix_on_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated timestamp resolves to a fresh suffixed run directory."""
    from src.application.strategy_account_persist import _resolve_account_destination

    now_fixed = pd.Timestamp("2025-03-03 12:00:00", tz="UTC")
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: now_fixed))
    root = tmp_path / "runs"
    kwargs = {"runs_root": root, "start": pd.Timestamp("2025-01-01", tz="UTC"), "end": pd.Timestamp("2025-02-01", tz="UTC"), "policy": "growth", "capital": 2100.0}
    first = _resolve_account_destination(**kwargs)
    second = _resolve_account_destination(**kwargs)
    assert first.name.endswith("Z")
    assert second.name == f"{first.name}-2"

def test_account_maker_run_directory_suffixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A maker run directory carries the maker infix after the capital."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    run_account_replay(_account_request(tmp_path, execution="maker"))
    (run_dir,) = _run_dirs(tmp_path)
    assert "_account_growth_2100_maker_" in run_dir.name

def test_account_catalog_append_confined_to_runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly one account row lands in the catalog derived from runs_root."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    run_account_replay(_account_request(tmp_path))
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["kind"] == "mhs_frozen_account"
    assert list(tmp_path.rglob("index.jsonl")) == [tmp_path / "index.jsonl"]

def test_account_unit_returns_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Export writes stamped parquet and leaves the payload unchanged."""
    import pyarrow.parquet as pq

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    dest = tmp_path / "nested" / "u.parquet"
    report = run_account_replay(_account_request(tmp_path, export_unit_returns=dest))
    unit_equity = seen["equities"][0]
    table = pq.read_table(dest)
    metadata = table.schema.metadata
    assert metadata[b"strategy_id"] == b"flow_mom_top20"
    assert metadata[b"execution"] == b"taker"
    assert metadata[b"evaluation_start"] == b"2025-01-01T00:00:00+00:00"
    assert metadata[b"evaluation_end"] == b"2025-02-01T00:00:00+00:00"
    assert metadata[b"run_dir"] == str(report.run_dir).encode()
    frame = table.to_pandas()
    expected = unit_equity.pct_change().iloc[1:]
    assert list(frame.columns) == ["unit_return"]
    assert frame.index.name == "entry_day"
    np.testing.assert_allclose(frame["unit_return"].to_numpy(), expected.to_numpy())
    assert (report.run_dir / "account.json").is_file()
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()
    plain = run_account_replay(_account_request(tmp_path, runs_root=tmp_path / "plain" / "runs")).payload
    exported = {key: value for key, value in report.payload.items() if key != "created_at"}
    unexported = {key: value for key, value in plain.items() if key != "created_at"}
    assert exported == unexported

@pytest.mark.parametrize("case", sorted(_GOLDEN_ACCOUNT_CASES))
def test_account_payload_matches_pre_refactor_capture(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Success-path account.json is byte-identical to the pre-refactor CLI output (created_at aside)."""
    reference, overrides = _GOLDEN_ACCOUNT_CASES[case]
    _install_strategy_account_fakes(monkeypatch, tmp_path, account_intraday=0.2)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: _PINNED_NOW))
    if reference is not None:
        _write_same_book_reference(tmp_path, tmp_path / "index.jsonl", run_dir="runs/old", base_cagr=reference[0], base_mdd=reference[1])
    report = run_account_replay(_account_request(tmp_path, **overrides))
    text = (report.run_dir / "account.json").read_text(encoding="utf-8")
    text = text.replace(f'\n  "created_at": "{_PINNED_NOW.isoformat()}",', "").replace(str(tmp_path), "<TMP>")
    data = json.loads(text)
    statistics = data.pop("statistics")
    assert data.pop("design_data_cutoff") == "2026-07-01T00:00:00+00:00"
    assert set(statistics) == {"base", "stress", "unit_reference"}
    assert statistics["base"]["in_sample_days"] >= 0
    assert statistics["stress"]["in_sample_days"] == statistics["base"]["in_sample_days"]
    assert data.pop("statistics_limitations") == ["ACCOUNT_FUNDING_ATTRIBUTION_UNAVAILABLE"]
    assert data.pop("stress_execution")["taker_fee_bps"] == 18.0
    assert json.dumps(data, indent=2, sort_keys=True) == _PRE_REFACTOR["account"][case]
    assert "error_type" not in text

def test_account_headline_failure_leaves_no_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Headlines are derived before the run directory exists, so a degenerate ledger leaves nothing behind."""
    import src.application.strategy_account_persist as persist_mod

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    real = persist_mod._account_headlines

    def _fail_account(equity: pd.Series, capital: float) -> tuple[float, float, float]:
        if capital != 100000.0:
            raise IndexError("empty account ledger")
        return real(equity, capital)

    monkeypatch.setattr(persist_mod, "_account_headlines", _fail_account)
    with pytest.raises(IndexError, match=r"empty account ledger"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []
    assert not (tmp_path / "index.jsonl").exists()

def test_account_run_dir_creation_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unusable runs_root surfaces as AccountReplayError, never a raw OSError traceback."""
    from src.application.strategy_account import AccountReplayError

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(AccountReplayError, match=r"^account replay failed: ") as raised:
        run_account_replay(_account_request(tmp_path, runs_root=blocker / "runs"))
    assert isinstance(raised.value.__cause__, OSError)
    assert not (tmp_path / "index.jsonl").exists()

def test_account_statistics_rejects_degenerate_ledger(tmp_path: Path) -> None:
    """A single-row ledger has no finite daily return for statistics."""
    import types

    import src.application.strategy_account_persist as persist_mod
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    dates = pd.DatetimeIndex([], tz="UTC")
    ledger = types.SimpleNamespace(daily_equity=pd.Series([], index=dates, dtype="float64"))
    request = _account_request(tmp_path)
    with pytest.raises(persist_mod.AccountReplayError, match="no finite daily returns"):
        persist_mod._account_path_statistics(
            unit=ledger, result=ledger, stress=ledger, request=request, strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT,
        )

@pytest.mark.parametrize("values", [[2100, float("nan")], [0, 2100]])
def test_account_statistics_rejects_corrupt_equity(tmp_path: Path, values) -> None:
    import src.application.strategy_account_persist as persist_mod
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    ledger = types.SimpleNamespace(daily_equity=pd.Series(values, index=pd.date_range("2025-01-01", periods=2, tz="UTC")))
    with pytest.raises(persist_mod.AccountReplayError):
        persist_mod._account_path_statistics(unit=ledger, result=ledger, stress=ledger, request=_account_request(tmp_path), strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)

def test_account_statistics_preserves_initial_cost_and_liquidation(tmp_path: Path) -> None:
    import src.application.strategy_account_persist as persist_mod
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    dates = pd.date_range("2025-01-01", periods=3, tz="UTC")
    unit = types.SimpleNamespace(daily_equity=pd.Series([100000, 100000, 100000], index=dates))
    account = types.SimpleNamespace(daily_equity=pd.Series([1890, 0, 0], index=dates))
    payload = persist_mod._account_path_statistics(unit=unit, result=account, stress=account, request=_account_request(tmp_path), strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    assert payload["base"]["cagr"] == -1
    assert payload["base"]["max_drawdown"] == 1
    assert payload["base"]["in_sample_days"] == 3
    assert payload["base"]["funding"] is None
    assert payload["stress"]["max_drawdown"] == 1

def test_account_statistics_uses_observed_funding(tmp_path: Path) -> None:
    import src.application.strategy_account_persist as persist_mod
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    dates = pd.date_range("2025-01-01", periods=3, tz="UTC")
    funding = pd.DataFrame({"AAA": [0, -10, -5]}, index=dates)
    ledger = types.SimpleNamespace(daily_equity=pd.Series([2100, 2110, 2115], index=dates), funding_by_symbol_daily=funding)
    payload = persist_mod._account_path_statistics(unit=ledger, result=ledger, stress=ledger, request=_account_request(tmp_path), strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    assert payload["base"]["funding"]["total_contribution"] == pytest.approx(-15 / 2100)
    assert payload["base"]["years"][0]["funding_share"] == pytest.approx(15 / 2100)
    ledger.funding_by_symbol_daily = funding.iloc[1:]
    with pytest.raises(persist_mod.AccountReplayError, match="index mismatch"):
        persist_mod._account_path_statistics(unit=ledger, result=ledger, stress=ledger, request=_account_request(tmp_path), strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)

def test_account_statistics_wraps_integrity_failure(tmp_path: Path) -> None:
    """A non-datetime equity index fails statistics inside the account wrapper."""
    import types

    import src.application.strategy_account_persist as persist_mod
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    ledger = types.SimpleNamespace(
        daily_equity=pd.Series([2100.0, 2200.0, 2150.0], dtype="float64"),
    )
    request = _account_request(tmp_path)
    with pytest.raises(persist_mod.AccountReplayError, match="statistics"):
        persist_mod._account_path_statistics(
            unit=ledger, result=ledger, stress=ledger, request=request, strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT,
        )

def test_persist_wraps_statistics_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Statistics rejections propagate through persistence as AccountReplayError."""
    import types

    import src.application.strategy_account_persist as persist_mod
    from src.common.errors import DataIntegrityError

    dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    ledger = types.SimpleNamespace(daily_equity=pd.Series([2100.0, 2200.0, 2150.0], index=dates))
    request = _account_request(tmp_path)
    monkeypatch.setattr(persist_mod, "_account_headlines", lambda equity, capital: (0.0, 0.0, 0.0))
    monkeypatch.setattr(
        persist_mod, "_account_path_statistics",
        lambda **kwargs: (_ for _ in ()).throw(persist_mod.AccountReplayError("statistics boom")),
    )
    with pytest.raises(persist_mod.AccountReplayError, match="statistics boom"):
        persist_mod._persist_account_run(
            request=request, strategy=object(), policy=object(), unit=ledger,
            result=ledger, stress=ledger, reconciliation={},
            rules_captured_at="t", venue_path="v",
        )
    monkeypatch.setattr(
        persist_mod, "_account_path_statistics",
        lambda **kwargs: (_ for _ in ()).throw(DataIntegrityError("statistics broken")),
    )
    with pytest.raises(persist_mod.AccountReplayError, match="statistics broken"):
        persist_mod._persist_account_run(
            request=request, strategy=object(), policy=object(), unit=ledger,
            result=ledger, stress=ledger, reconciliation={},
            rules_captured_at="t", venue_path="v",
        )

def test_account_statistics_rejects_misaligned_stress_and_unknown_funding(tmp_path: Path) -> None:
    import src.application.strategy_account_persist as persist_mod
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    dates = pd.date_range("2025-01-01", periods=2, tz="UTC")
    base = types.SimpleNamespace(daily_equity=pd.Series([2100, 2110], index=dates))
    stress = types.SimpleNamespace(daily_equity=pd.Series([2100, 2110], index=dates + pd.Timedelta(days=1)))
    with pytest.raises(persist_mod.AccountReplayError, match="indexes differ"):
        persist_mod._account_path_statistics(unit=base, result=base, stress=stress, request=_account_request(tmp_path), strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
    base.funding_by_symbol_daily = pd.DataFrame({"AAA": [0, float("nan")]}, index=dates)
    with pytest.raises(persist_mod.AccountReplayError, match="non-finite funding"):
        persist_mod._account_path_statistics(unit=base, result=base, stress=base, request=_account_request(tmp_path), strategy=FLOW_MOM_TOP20_ACCOUNT_UNIT)
