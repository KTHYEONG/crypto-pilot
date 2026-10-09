"""Invariant scenarios for the account replay service."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.application.strategy_account import (
    strategy_execution_specs,
    reconcile_unit_reference,
    run_account_replay,
)
from tests.unit.application._strategy_account_helpers import (
    _account_request,
    _install_strategy_account_fakes,
    _run_dirs,
    _write_catalog_index,
    _write_same_book_reference,
)


def test_strategy_specs_use_submit_anchor() -> None:
    """Both strategy cost cases cross from the last pre-submission mark at 6 and 18 bps."""
    base, stress = strategy_execution_specs()
    assert base.decision_anchor == "submit_bar"
    assert stress.decision_anchor == "submit_bar"
    assert base.one_way_taker_bps() == 6.0
    assert stress.one_way_taker_bps() == 18.0

def test_account_unit_reference_replays_before_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default request replays unit reference first with fixed/1.0 then the growth account."""
    from src.strategy.sizing import account_growth_policy
    from src.core.params import ACCOUNT_DEFAULT_CAPITAL_USDT, ACCOUNT_TAKER_FEE_BPS, ACCOUNT_UNIT_REFERENCE_CAPITAL

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    run_account_replay(_account_request(tmp_path))
    assert len(seen["replays"]) == 3
    unit_replay, main, stress = seen["replays"]
    assert stress["fee"] == 18.0
    assert stress["policy"] == main["policy"]
    pd.testing.assert_series_equal(stress["unit_equity"], main["unit_equity"])
    assert unit_replay["policy"].kind == "fixed"
    assert unit_replay["policy"].exposure_max == 1.0
    assert unit_replay["policy"].impact_y == 0.0
    assert unit_replay["capital"] == ACCOUNT_UNIT_REFERENCE_CAPITAL
    assert unit_replay["filters"] is False
    assert unit_replay["unit_equity"] is None
    assert main["capital"] == ACCOUNT_DEFAULT_CAPITAL_USDT
    assert main["policy"] == account_growth_policy(impact_y=main["policy"].impact_y)
    assert main["fee"] == ACCOUNT_TAKER_FEE_BPS
    assert main["filters"] is True
    pd.testing.assert_series_equal(main["unit_equity"], seen["equities"][0])

def test_account_candidate_is_unlevered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The account book is the registered unlevered clip unit book; both replays use its anchors."""
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT
    from src.core.params import STRATEGY_NAME_CLIP

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    run_account_replay(_account_request(tmp_path))
    assert seen["request"].strategy is FLOW_MOM_TOP20_ACCOUNT_UNIT
    assert seen["request"].strategy.exposure_multiplier == 1.0
    assert seen["request"].strategy.name_clip == STRATEGY_NAME_CLIP
    assert seen["request"].execution_bound == "OHLCV_IMMEDIATE_TAKER"
    for replay in seen["replays"]:
        pd.testing.assert_index_equal(replay["anchor_times"], seen["anchors"])

def test_account_maker_threads_identical_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maker execution passes identical maker controls to both ledgers."""
    from src.core.params import ACCOUNT_MAKER_FEE_BPS, ACCOUNT_PASSIVE_WINDOW_BARS

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    run_account_replay(_account_request(tmp_path, execution="maker"))
    assert len(seen["replays"]) == 3
    for replay in seen["replays"]:
        assert replay["execution"] == "maker"
        assert replay["maker_fee_bps"] == ACCOUNT_MAKER_FEE_BPS
        assert replay["passive_window_bars"] == ACCOUNT_PASSIVE_WINDOW_BARS

def test_account_default_execution_stays_taker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Taker replays carry no maker kwargs and the run dir has no maker infix."""
    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    run_account_replay(_account_request(tmp_path))
    for replay in seen["replays"]:
        assert replay["execution"] == "taker"
        assert "maker_fee_bps" not in replay
        assert "passive_window_bars" not in replay
    (run_dir,) = _run_dirs(tmp_path)
    assert "_maker_" not in run_dir.name

def test_account_fixed_moment_source_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed policy discloses no moment source but still replays the unit reference."""
    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    report = run_account_replay(_account_request(tmp_path, policy="fixed", fixed_exposure=2.5))
    assert seen["replays"][0]["policy"].exposure_max == 1.0
    assert seen["replays"][1]["unit_equity"] is not None
    assert report.payload["moment_source"] == "none"

def test_account_execution_disclosed_in_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maker mode is disclosed in account.json and the catalog row."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    report = run_account_replay(_account_request(tmp_path, execution="maker"))
    assert report.payload["execution"]["mode"] == "maker"
    assert report.payload["execution"]["maker_fill_fraction"] == pytest.approx(0.9)
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["execution"] == "maker"

def test_account_headline_drawdown_is_intraday_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Headline drawdown is the 3m close path with opposite signs in payload and catalog."""
    _install_strategy_account_fakes(monkeypatch, tmp_path, account_intraday=0.2)
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["mdd"] == pytest.approx(-0.2)
    assert report.payload["unit_reference"]["mdd"] == pytest.approx(-0.05)
    assert report.payload["reconciliation"]["mdd"] == pytest.approx(0.05)
    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["base_max_drawdown"] == pytest.approx(0.2)

def test_account_unit_reference_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed unit reference replay raises with the cause chained and no run dir."""
    from src.application.strategy_account import AccountReplayError

    _install_strategy_account_fakes(monkeypatch, tmp_path, unit_fail=True)
    with pytest.raises(AccountReplayError, match=r"unit reference"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []

def test_account_unit_reference_liquidation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A liquidated unit reference fails the run with no run dir."""
    from src.application.strategy_account import AccountReplayError

    _install_strategy_account_fakes(monkeypatch, tmp_path, unit_liquidated=True)
    with pytest.raises(AccountReplayError, match=r"unit reference liquidated at"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []

def test_account_replay_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An account replay failure raises without persisting."""
    import src.engine.account_ledger as ledger_mod
    from src.application.strategy_account import AccountReplayError
    from src.common.errors import DataIntegrityError

    _install_strategy_account_fakes(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("replay boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _boom)
    with pytest.raises(AccountReplayError, match=r"account replay failed"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []

def test_account_source_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source assembly failure raises before any replay."""
    import src.engine.strategy_backtest as run_mod
    from src.application.strategy_account import AccountReplayError
    from src.common.errors import DataIntegrityError

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)

    def _boom(request: object) -> tuple:
        raise DataIntegrityError("source boom")

    monkeypatch.setattr(run_mod, "build_request_targets", _boom)
    with pytest.raises(AccountReplayError, match=r"account replay failed"):
        run_account_replay(_account_request(tmp_path))
    assert "replays" not in seen

def test_account_missing_venue_snapshot_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty venue root fails before the candidate build."""
    import src.engine.strategy_backtest as run_mod
    from src.application.strategy_account import AccountReplayError

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path, stub_venue=False)
    called = []
    monkeypatch.setattr(run_mod, "build_request_targets", lambda req: (called.append(req), (_ for _ in ()).throw(AssertionError("must not build"))))
    with pytest.raises(AccountReplayError, match=r"data collect venue-rules"):
        run_account_replay(_account_request(tmp_path))
    assert called == []

def test_account_invalid_request_rejected_before_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid controls raise ValueError without touching venue or candidate seams."""
    import src.market_data.binance.venue_rules as venue_mod
    import src.engine.strategy_backtest as run_mod

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(venue_mod, "latest_venue_rule_snapshot", lambda root: (_ for _ in ()).throw(AssertionError("no io")))
    monkeypatch.setattr(run_mod, "build_request_targets", lambda req: (_ for _ in ()).throw(AssertionError("no build")))
    bad_variants = [
        {"source_start": pd.Timestamp("2025-01-01")},
        {"policy": "turbo"},
        {"execution": "limit"},
        {"policy": "fixed"},
        {"fixed_exposure": 0},
        {"capital": float("inf")},
    ]
    for overrides in bad_variants:
        with pytest.raises(ValueError, match=r".+"):
            run_account_replay(_account_request(tmp_path, **overrides))

def test_reconcile_missing_reference() -> None:
    """None reference yields a missing_reference record with null gaps."""
    rec = reconcile_unit_reference(None, unit_cagr=0.1, unit_mdd=0.05)
    assert rec["status"] == "missing_reference"
    assert rec["reference_canonical"] is None
    assert rec["cagr_gap"] is None
    assert rec["mdd_gap"] is None
    assert rec["fixed_exposure"] == 1.0

def test_reconcile_boundary_inclusive() -> None:
    """Gaps exactly at tolerance are ok; twice the CAGR tolerance is mismatch."""
    from src.core.params import ACCOUNT_RECON_CAGR_TOLERANCE

    ref = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_cagr": 0.0, "base_max_drawdown": 0.0, "name_clip": 0.05, "exposure_multiplier": 1.0}
    ok = reconcile_unit_reference(ref, unit_cagr=ACCOUNT_RECON_CAGR_TOLERANCE, unit_mdd=0.0)
    assert ok["status"] == "ok"
    bad = reconcile_unit_reference(ref, unit_cagr=2 * ACCOUNT_RECON_CAGR_TOLERANCE, unit_mdd=0.0)
    assert bad["status"] == "mismatch"
    assert bad["cagr_gap"] == pytest.approx(2 * ACCOUNT_RECON_CAGR_TOLERANCE)

def test_reconcile_missing_base_metric_is_mismatch() -> None:
    """A None base CAGR yields mismatch with a null gap."""
    ref = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_cagr": None, "base_max_drawdown": 0.05, "name_clip": 0.05, "exposure_multiplier": 1.0}
    rec = reconcile_unit_reference(ref, unit_cagr=0.1, unit_mdd=0.05)
    assert rec["status"] == "mismatch"
    assert rec["cagr_gap"] is None

def test_reconcile_drawdown_sign_agnostic() -> None:
    """Negative and positive base drawdowns give identical gaps."""
    base = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "name_clip": 0.05, "exposure_multiplier": 1.0}
    pos = reconcile_unit_reference({**base, "base_cagr": 0.1, "base_max_drawdown": 0.05}, unit_cagr=0.1, unit_mdd=0.05)
    neg = reconcile_unit_reference({**base, "base_cagr": 0.1, "base_max_drawdown": -0.05}, unit_cagr=0.1, unit_mdd=0.05)
    assert pos["mdd_gap"] == neg["mdd_gap"] == pytest.approx(0.0)

def test_reconcile_pure_and_non_mutating() -> None:
    """Repeated calls agree and the input mapping is unchanged."""
    ref = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_cagr": 0.1, "base_max_drawdown": 0.05, "name_clip": 0.05, "exposure_multiplier": 1.0}
    before = dict(ref)
    first = reconcile_unit_reference(ref, unit_cagr=0.1, unit_mdd=0.05)
    second = reconcile_unit_reference(ref, unit_cagr=0.1, unit_mdd=0.05)
    assert first == second
    assert ref == before

def test_reconcile_non_numeric_base_metric_raises() -> None:
    """Unconvertible base metrics raise typed errors."""
    base = {"strategy_id": "s", "run_dir": "r", "evaluation_start": "a", "evaluation_end": "b", "base_max_drawdown": 0.05, "name_clip": 0.05, "exposure_multiplier": 1.0}
    with pytest.raises(ValueError, match=r".*"):
        reconcile_unit_reference({**base, "base_cagr": "abc"}, unit_cagr=0.1, unit_mdd=0.05)
    with pytest.raises(TypeError):
        reconcile_unit_reference({**base, "base_cagr": [1]}, unit_cagr=0.1, unit_mdd=0.05)
    with pytest.raises(OverflowError):
        reconcile_unit_reference({**base, "base_cagr": 10**400}, unit_cagr=0.1, unit_mdd=0.05)

def test_account_reference_lookup_failure_is_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing catalog lookup is disclosed with error_type and the run persists."""
    import src.application.strategy_account as app_mod

    _install_strategy_account_fakes(monkeypatch, tmp_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise OSError("catalog boom")

    monkeypatch.setattr(app_mod, "_latest_same_book_reference", _boom)
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"] == {"status": "failed", "error": "catalog boom", "error_type": "OSError"}
    assert (report.run_dir / "account_daily.parquet").exists()

def test_account_corrupt_catalog_line_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid JSON in the catalog discloses failed with JSONDecodeError."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    (tmp_path / "index.jsonl").write_text("not json\n", encoding="utf-8")
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "failed"
    assert report.payload["reconciliation"]["error_type"] == "JSONDecodeError"
    assert (report.run_dir / "account.json").is_file()

def test_account_non_object_catalog_row_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-object catalog row discloses failed with ValueError."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    (tmp_path / "index.jsonl").write_text("[1, 2]\n", encoding="utf-8")
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "failed"
    assert report.payload["reconciliation"]["error_type"] == "ValueError"

def test_account_non_numeric_reference_metric_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric base metric discloses failed with ValueError."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/ref", base_cagr="abc", base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "failed"
    assert report.payload["reconciliation"]["error_type"] == "ValueError"

def test_account_unexpected_error_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError from the lookup propagates with no artifacts."""
    import src.application.strategy_account as app_mod

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_latest_same_book_reference", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []
    assert not (tmp_path / "index.jsonl").exists()

def test_account_non_object_result_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A non-object result.json discloses failed while the run succeeds."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/good", base_cagr=(105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0, base_mdd=0.05)
    bad_row = {"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/bad", "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00", "base_cagr": 0.1, "base_max_drawdown": 0.05}
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(bad_row, sort_keys=True) + "\n")
    bad_dir = tmp_path / "runs" / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "result.json").write_text("[]", encoding="utf-8")
    with caplog.at_level("WARNING", logger="src.application.strategy_account"):
        report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"] == {"status": "failed", "error": "reference result is not a JSON object", "error_type": "ValueError"}
    assert (report.run_dir / "account_daily.parquet").exists()
    assert "status=failed" in caplog.text

def test_account_success_schema_has_no_error_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ok/mismatch/missing_reference payloads never carry error_type."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "missing_reference"
    assert "error_type" not in json.dumps(report.payload)

def test_account_same_execution_reference_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Maker and taker runs each reconcile against their own execution canonical."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/taker", base_cagr=unit_cagr, base_mdd=0.05)
    _write_same_book_reference(tmp_path, index, run_dir="runs/maker", execution="maker", base_cagr=unit_cagr, base_mdd=0.05)
    maker = run_account_replay(_account_request(tmp_path, execution="maker", runs_root=tmp_path / "m" / "runs"))
    assert maker.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/maker"
    taker = run_account_replay(_account_request(tmp_path))
    assert taker.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/taker"

def test_account_missing_same_execution_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A maker run without a maker canonical warns with missing_reference."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/taker", base_cagr=(105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0, base_mdd=0.05)
    with caplog.at_level("WARNING", logger="src.application.strategy_account"):
        report = run_account_replay(_account_request(tmp_path, execution="maker"))
    assert report.payload["reconciliation"]["status"] == "missing_reference"
    assert "status=missing_reference" in caplog.text

def test_account_unclipped_primary_never_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unclipped primary row never reconciles the clipped book."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    _write_same_book_reference(tmp_path, index, run_dir="runs/unclipped", execution="maker", name_clip=None, exposure_multiplier=1.0, base_cagr=0.1, base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path, execution="maker"))
    assert report.payload["reconciliation"]["status"] == "missing_reference"

def test_account_gap_beyond_tolerance_is_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A same-book gap beyond tolerance is a disclosed mismatch."""
    from src.core.params import ACCOUNT_RECON_CAGR_TOLERANCE

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/ref", base_cagr=unit_cagr + 2.0 * ACCOUNT_RECON_CAGR_TOLERANCE, base_mdd=0.05)
    with caplog.at_level("WARNING", logger="src.application.strategy_account"):
        report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "mismatch"
    assert report.payload["reconciliation"]["cagr_gap"] == pytest.approx(-2.0 * ACCOUNT_RECON_CAGR_TOLERANCE)
    assert "status=mismatch" in caplog.text

def test_account_window_or_execution_mismatch_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows from another window never reconcile this run."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/other-window", execution="maker", base_cagr=unit_cagr, base_mdd=0.05, evaluation_start="2024-01-01T00:00:00+00:00", evaluation_end="2024-02-01T00:00:00+00:00")
    _write_same_book_reference(tmp_path, index, run_dir="runs/taker", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path, execution="maker"))
    assert report.payload["reconciliation"]["status"] == "missing_reference"

def test_account_latest_matching_row_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two same-book rows reconcile against the later catalog entry."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/first", base_cagr=unit_cagr, base_mdd=0.05)
    _write_same_book_reference(tmp_path, index, run_dir="runs/second", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/second"

def test_account_legacy_index_row_matches_canonical_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-book reconciliation finds pre-rename index rows via legacy id resolution."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = _write_catalog_index(tmp_path, [])
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/legacy", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path))
    reconciliation = report.payload["reconciliation"]
    assert reconciliation["status"] == "ok"
    assert reconciliation["reference_canonical"]["run_dir"] == "runs/legacy"

def test_account_same_book_skips_unreadable_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrupt and mismatched rows never break reconciliation; the good row wins."""
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    index = tmp_path / "index.jsonl"
    lines = [
        "",
        json.dumps({"kind": "mhs", "strategy_id": "x"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "other_book"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": 123, "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00"}),
        json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/gone", "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00"}),
    ]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad_json = tmp_path / "runs" / "broken"
    bad_json.mkdir(parents=True)
    (bad_json / "result.json").write_text("not json", encoding="utf-8")
    index.write_text(index.read_text(encoding="utf-8") + json.dumps({"kind": "mhs_frozen", "strategy_id": "frozen_mhs_top20_v2", "run_dir": "runs/broken", "evaluation_start": "2025-01-01T00:00:00+00:00", "evaluation_end": "2025-02-01T00:00:00+00:00"}) + "\n", encoding="utf-8")
    unit_cagr = (105000.0 / 100000.0) ** (365.0 / 3.0) - 1.0
    _write_same_book_reference(tmp_path, index, run_dir="runs/levered", base_cagr=unit_cagr, base_mdd=0.05, exposure_multiplier=2.5)
    _write_same_book_reference(tmp_path, index, run_dir="runs/good", base_cagr=unit_cagr, base_mdd=0.05)
    report = run_account_replay(_account_request(tmp_path))
    assert report.payload["reconciliation"]["status"] == "ok"
    assert report.payload["reconciliation"]["reference_canonical"]["run_dir"] == "runs/good"

def test_validate_rejects_non_request_and_bad_roots(tmp_path: Path) -> None:
    """Non-request, unordered and non-Path inputs raise before any I/O."""
    from src.application.strategy_account import validate_account_replay_request

    with pytest.raises(ValueError, match=r"AccountReplayRequest"):
        validate_account_replay_request("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"source_start < evaluation"):
        run_account_replay(_account_request(tmp_path, source_start=pd.Timestamp("2025-03-01", tz="UTC")))
    with pytest.raises(ValueError, match=r"runs_root must be a Path"):
        run_account_replay(_account_request(tmp_path, runs_root="x"))  # type: ignore[arg-type]

def test_invalid_strategy_request_construction_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad cost pair fails the strategy request build without any replay."""
    import src.application.strategy_account as app_mod
    from src.application.strategy_account import AccountReplayError

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "strategy_execution_specs", lambda: (None, None))
    with pytest.raises(AccountReplayError, match=r"invalid account replay request"):
        run_account_replay(_account_request(tmp_path))
    assert "replays" not in seen

def test_account_second_replay_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the account replay failing still leaves no run dir."""
    import src.engine.account_ledger as ledger_mod
    from src.application.strategy_account import AccountReplayError
    from src.common.errors import DataIntegrityError

    seen = _install_strategy_account_fakes(monkeypatch, tmp_path)
    real = ledger_mod.replay_account

    def _fail_second(*args: object, **kwargs: object) -> object:
        if len(seen.get("replays", [])) == 0:
            return real(*args, **kwargs)
        raise DataIntegrityError("account boom")

    monkeypatch.setattr(ledger_mod, "replay_account", _fail_second)
    with pytest.raises(AccountReplayError, match=r"account replay failed: account boom"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []

def test_account_persistence_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An artifact write failure surfaces as a failed run."""
    from src.application.strategy_account import AccountReplayError

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(AccountReplayError, match=r"account replay failed: disk full"):
        run_account_replay(_account_request(tmp_path))

@pytest.mark.parametrize(("execution", "passive"), [("taker", False), ("maker", False), ("maker", True)])
def test_real_account_stress_replay_compares_same_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution: str, passive: bool) -> None:
    import src.engine.account_ledger as ledger_mod
    import src.engine.account_sources as sources_mod
    import src.market_data.binance.venue_rules as venue_mod
    from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules

    replay = ledger_mod.replay_account
    _install_strategy_account_fakes(monkeypatch, tmp_path)
    rules = VenueRuleSnapshot(
        captured_at=pd.Timestamp("2025-01-01", tz="UTC"),
        symbols={"AAA": VenueSymbolRules(symbol="AAA", brackets=(VenueBracket(0.0, 1e12, 0.01, 0.0, 10),), step_size=0.001, min_notional=5.0)},
    )
    monkeypatch.setattr(venue_mod, "load_venue_rule_snapshot", lambda path: rules)
    monkeypatch.setattr(ledger_mod, "replay_account", replay)
    assemble = sources_mod.assemble_account_inputs

    def _parts(candidate, context):
        parts = list(assemble(candidate, context))
        if passive:
            marks = parts[1]
            parts[1] = ledger_mod.AccountMarkPanels(close=marks.close, low=marks.close * 0.99, high=marks.close * 1.01)
        return tuple(parts)

    monkeypatch.setattr(sources_mod, "assemble_account_inputs", _parts)
    report = run_account_replay(_account_request(tmp_path, policy="fixed", fixed_exposure=1.0, execution=execution, impact_y=0.0, apply_order_filters=False))
    base = report.payload["statistics"]["base"]
    stress = report.payload["statistics"]["stress"]
    assert base["in_sample_days"] == stress["in_sample_days"] == 3
    assert report.payload["statistics_limitations"] == []
    assert report.payload["stress_execution"]["taker_fee_bps"] == 18.0
    assert report.stress_result.capital == report.result.capital == 2100.0
    assert (report.run_dir / "account_stress_daily.parquet").exists()
    persisted = json.loads((report.run_dir / "account.json").read_text())
    assert persisted["statistics"]["stress"] == stress
    if passive:
        pd.testing.assert_series_equal(report.result.daily_equity, report.stress_result.daily_equity)
        assert base["cagr"] == stress["cagr"]
    else:
        assert stress["cagr"] < base["cagr"]
        assert report.stress_result.fee_paid > report.result.fee_paid

def test_stress_replay_failure_leaves_no_completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.strategy_account as app_mod
    import src.engine.account_ledger as ledger_mod
    from src.common.errors import DataIntegrityError

    _install_strategy_account_fakes(monkeypatch, tmp_path)
    replay = ledger_mod.replay_account

    def _fail_stress(*args, **kwargs):
        if kwargs["taker_fee_bps"] == 18.0:
            raise DataIntegrityError("stress source broken")
        return replay(*args, **kwargs)

    monkeypatch.setattr(ledger_mod, "replay_account", _fail_stress)
    with pytest.raises(app_mod.AccountReplayError, match="stress source broken"):
        run_account_replay(_account_request(tmp_path))
    assert _run_dirs(tmp_path) == []
