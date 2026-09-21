"""Invariant scenarios for the frozen-MHS research payload."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionReplayWindow
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2, FrozenMhsCandidate
from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod, evaluate_frozen_mhs_research
from src.mhs.frozen_research_run import FrozenMhsBacktestRequest, FrozenMhsBacktestRun
from src.mhs.types import ExecutionSpec

from src.mhs.frozen_research_report import (
    frozen_mhs_backtest_payload,
    frozen_mhs_daily_frame,
    persist_frozen_mhs_backtest,
)

_SYMBOLS = ("AAA", "BBB")
_DAY1 = pd.Timestamp("2021-06-01", tz="UTC")


def _specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar")
    stress = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0, decision_anchor="submit_bar")
    return base, stress


def _candidate(labels: list[pd.Timestamp]) -> FrozenMhsCandidate:
    weights = pd.DataFrame(
        {"AAA": [0.05] * len(labels), "BBB": [-0.05] * len(labels)},
        index=pd.DatetimeIndex(labels, tz="UTC"), dtype="float64",
    )
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    return FrozenMhsCandidate(target_weights=weights, signal_available_at=avail, strategy=FROZEN_MHS_TOP20_V2)


def _frames(grid: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    cols = list(_SYMBOLS)
    return {
        "highs": pd.DataFrame(101.0, index=grid, columns=cols, dtype="float64"),
        "lows": pd.DataFrame(99.0, index=grid, columns=cols, dtype="float64"),
        "closes": pd.DataFrame(100.0, index=grid, columns=cols, dtype="float64"),
        "marks": pd.DataFrame(100.0, index=grid, columns=cols, dtype="float64"),
        "bar_funding": pd.DataFrame(0.0, index=grid, columns=cols, dtype="float64"),
        "quote_volumes": pd.DataFrame(1000.0, index=grid, columns=cols, dtype="float64"),
        "funding_known": pd.DataFrame(True, index=grid, columns=cols),
    }


def _window(grid: pd.DatetimeIndex, candidate: FrozenMhsCandidate, labels: list[pd.Timestamp]) -> ExecutionReplayWindow:
    weights = candidate.target_weights.loc[labels].copy()
    avail = pd.DatetimeIndex(
        [candidate.signal_available_at[candidate.target_weights.index.get_loc(label)] for label in labels], tz="UTC"
    )
    params: dict[str, object] = {
        "window_start": grid[0], "window_end": grid[-1], "columns": _SYMBOLS, "symbols": _SYMBOLS,
        "minute_grid": grid, "target_weights": weights, "signal_available_at": avail,
        "bar_available_at": grid + pd.Timedelta(minutes=3),
    }
    params.update(_frames(grid))
    return ExecutionReplayWindow(**params)  # type: ignore[arg-type]


def _windows(candidate: FrozenMhsCandidate, labels: list[pd.Timestamp]) -> list[ExecutionReplayWindow]:
    first = _window(pd.date_range(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=1), freq="3min", tz="UTC"), candidate, labels[:1])
    second = _window(pd.date_range(labels[1] - pd.Timedelta(hours=2), labels[2] + pd.Timedelta(hours=2), freq="3min", tz="UTC"), candidate, labels[1:])
    return [first, second]


def _run() -> FrozenMhsBacktestRun:
    labels = [_DAY1 + pd.Timedelta(days=i) for i in (1, 2, 3)]
    candidate = _candidate(labels)
    base_spec, stress_spec = _specs()
    probe_periods = (
        FrozenMhsReportPeriod(label="probe", start=pd.Timestamp("2022-01-01", tz="UTC"), end=pd.Timestamp("2022-01-02", tz="UTC")),
    )
    probe = evaluate_frozen_mhs_research(
        candidate, iter(_windows(candidate, labels)), initial_equity=100000.0,
        base_spec=base_spec, stress_spec=stress_spec, report_periods=probe_periods,
    )
    covered = probe.base_daily.returns.index
    periods = (
        FrozenMhsReportPeriod(label="P1", start=covered[0], end=covered[1]),
        FrozenMhsReportPeriod(label="P9", start=pd.Timestamp("2022-01-01", tz="UTC"), end=pd.Timestamp("2022-01-10", tz="UTC")),
    )
    evidence = evaluate_frozen_mhs_research(
        candidate, iter(_windows(candidate, labels)), initial_equity=100000.0,
        base_spec=base_spec, stress_spec=stress_spec, report_periods=periods,
    )
    request = FrozenMhsBacktestRequest(
        source_start=_DAY1, evaluation_start=labels[0], evaluation_end=labels[-1] + pd.Timedelta(days=1),
        strategy=FROZEN_MHS_TOP20_V2, initial_equity=100000.0,
        base_spec=base_spec, stress_spec=stress_spec, report_periods=periods,
    )
    return FrozenMhsBacktestRun(
        request=request, candidate=candidate, evidence=evidence,
        execution_start=labels[0], execution_end=labels[-1] + pd.Timedelta(days=1),
        source_symbols=_SYMBOLS,
    )


def test_payload_preserves_strategy_provenance() -> None:
    """Completed Top-20 evidence serializes strategy, costs, validity, and research-only limits."""
    run = _run()
    payload = frozen_mhs_backtest_payload(run)
    assert payload["strategy_id"] == "frozen_mhs_top20_v2"
    assert payload["breadth"] == 20
    assert [member["name"] for member in payload["members"]] == [  # type: ignore[index]
        "flow_imb_168h", "flow_imb_720h", "xs_mom_336h", "xs_idio_mom_336h", "mom3_skew_168h"
    ]
    assert payload["canonical_symbols"] == 2
    assert payload["base_one_way_taker_bps"] == 6.0
    assert payload["stress_one_way_taker_bps"] == 18.0
    assert payload["base_valid"] is True
    assert payload["research_only"] is True
    assert payload["source_gap_blocked_decisions"] == run.source_gap_blocked_decisions
    assert "NO_DEPLOYMENT_VERDICT" in payload["limitations"]  # type: ignore[operator]
    assert payload["report_periods"]["P1"]["status"] == "complete"  # type: ignore[index]
    assert payload["report_periods"]["P1"]["base_cagr"] is not None  # type: ignore[index]
    json.dumps(payload)


def test_partial_period_unavailable_rather_than_zero() -> None:
    """Incomplete daily coverage serializes as null metrics with explicit coverage."""
    run = _run()
    payload = frozen_mhs_backtest_payload(run)
    partial = payload["report_periods"]["P9"]  # type: ignore[index]
    assert partial["status"] == "unavailable"
    assert partial["base_coverage"] == 0.0
    assert "base_cagr" not in partial


def test_payload_has_no_deployment_verdict() -> None:
    """No go, live, or deploy decision field appears at any payload level."""
    run = _run()
    payload = frozen_mhs_backtest_payload(run)
    assert "go" not in payload
    assert not any("deploy" in key or "live" in key for key in payload)
    assert not any("signal" in key or "fills" in key for key in payload)


def test_fresh_atomic_output_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing output or failed atomic replace leaves no false completed envelope."""
    import src.mhs.frozen_research_report as report_mod

    run = _run()
    output = tmp_path / "result.json"
    assert persist_frozen_mhs_backtest(run, output) == output
    assert json.loads(output.read_text(encoding="utf-8"))["strategy_id"] == "frozen_mhs_top20_v2"
    with pytest.raises(DataIntegrityError, match=r"fresh"):
        persist_frozen_mhs_backtest(run, output)
    second_dir = tmp_path / "second_run"
    second_dir.mkdir()
    second = second_dir / "second.json"
    monkeypatch.setattr(report_mod.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        persist_frozen_mhs_backtest(run, second)
    assert not second.exists()
    assert not (second_dir / "second.json.tmp").exists()
    assert not (second_dir / "daily.parquet").exists()


def test_payload_rejects_incomplete_evidence() -> None:
    """An invalid ledger or provenance mismatch cannot be represented as completed."""
    run = _run()
    bad_ledger = dataclasses.replace(run.evidence.base.ledger, primary_valid=False)
    bad_base = dataclasses.replace(run.evidence.base, ledger=bad_ledger)
    bad_evidence = dataclasses.replace(run.evidence, base=bad_base)
    bad_run = dataclasses.replace(run, evidence=bad_evidence)
    with pytest.raises(DataIntegrityError, match=r"completed valid"):
        frozen_mhs_backtest_payload(bad_run)
    narrow = dataclasses.replace(
        run, candidate=dataclasses.replace(run.candidate, target_weights=run.candidate.target_weights[["AAA"]])
    )
    with pytest.raises(DataIntegrityError, match=r"census"):
        frozen_mhs_backtest_payload(narrow)


def test_jsonable_scalars_and_rejection() -> None:
    """Ledger scalars map to JSON values; non-finite becomes null and unknown raises."""
    import numpy as np

    from src.mhs.frozen_research_report import _jsonable

    assert _jsonable(None) is None
    assert _jsonable(True) is True
    assert _jsonable(5) == 5
    assert _jsonable(np.int64(3)) == 3
    assert _jsonable(2.5) == 2.5
    assert _jsonable(float("nan")) is None
    assert _jsonable("x") == "x"
    assert _jsonable(pd.Timestamp("2021-06-02", tz="UTC")) == "2021-06-02T00:00:00+00:00"
    with pytest.raises(DataIntegrityError, match=r"no JSON representation"):
        _jsonable(object())


def test_payload_rejects_strategy_mismatch_and_missing_period() -> None:
    """A foreign strategy or a metrics row outside the request cannot be published."""
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP40_CONTROL_V2

    run = _run()
    foreign = dataclasses.replace(run, request=dataclasses.replace(run.request, strategy=FROZEN_MHS_TOP40_CONTROL_V2))
    with pytest.raises(DataIntegrityError, match=r"must match the request strategy"):
        frozen_mhs_backtest_payload(foreign)
    ghost_periods = (
        *run.request.report_periods,
        FrozenMhsReportPeriod(label="ghost", start=pd.Timestamp("2023-01-01", tz="UTC"), end=pd.Timestamp("2023-01-02", tz="UTC")),
    )
    ghost = dataclasses.replace(run, request=dataclasses.replace(run.request, report_periods=ghost_periods))
    with pytest.raises(DataIntegrityError, match=r"no metrics row"):
        frozen_mhs_backtest_payload(ghost)


def test_persist_rejects_unsafe_output(tmp_path: Path) -> None:
    """A non-Path or non-JSON destination is refused before any serialization."""
    run = _run()
    with pytest.raises(DataIntegrityError, match=r"must be a Path"):
        persist_frozen_mhs_backtest(run, "result.json")  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match=r"must be a JSON path"):
        persist_frozen_mhs_backtest(run, tmp_path / "result.parquet")


def test_payload_records_integrity_exclusions() -> None:
    run = _run()
    tagged = dataclasses.replace(run, source_gap_excluded_symbols=("LUNAUSDT", "PUMPUSDT"))
    payload = frozen_mhs_backtest_payload(tagged)
    assert payload["source_gap_excluded_symbols"] == ["LUNAUSDT", "PUMPUSDT"]
    assert payload["source_gap_excluded_count"] == 2
    assert payload["canonical_symbols"] == 2
    assert "approval" not in payload
    assert "deployment_verdict" not in payload
    assert "go_live" not in payload
    assert payload["research_only"] is True


def test_registry_is_single_source_gap_view() -> None:
    import argparse

    from src.cli.commands.backtest import add_backtest_commands
    from src.mhs.data_policy import (
        MHS_DATA_POLICY_DEFAULT,
        SOURCE_GAP_EXCLUDED_SYMBOLS,
        source_gap_excluded_symbols,
    )

    resolved = source_gap_excluded_symbols()
    assert isinstance(resolved, frozenset)
    assert source_gap_excluded_symbols() is not resolved
    assert set(resolved) == set(SOURCE_GAP_EXCLUDED_SYMBOLS)
    assert "PUMPUSDT" in SOURCE_GAP_EXCLUDED_SYMBOLS
    assert "LUNAUSDT" in SOURCE_GAP_EXCLUDED_SYMBOLS
    assert MHS_DATA_POLICY_DEFAULT == "zombie_mask_v1"
    parser = argparse.ArgumentParser()
    add_backtest_commands(parser)
    assert "exclud" not in parser.format_help().lower()


def _growth_run() -> FrozenMhsBacktestRun:
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_GROWTH_V2

    run = _run()
    return dataclasses.replace(
        run,
        request=dataclasses.replace(run.request, strategy=FROZEN_MHS_TOP20_GROWTH_V2),
        candidate=dataclasses.replace(run.candidate, strategy=FROZEN_MHS_TOP20_GROWTH_V2),
    )


def test_payload_states_policy() -> None:
    """Growth and primary payloads state their registered exposure policy explicitly."""
    from src.mhs.params import FROZEN_GROWTH_EXPOSURE_MULTIPLIER, FROZEN_GROWTH_NAME_CLIP

    growth = frozen_mhs_backtest_payload(_growth_run())
    assert growth["exposure_multiplier"] == FROZEN_GROWTH_EXPOSURE_MULTIPLIER
    assert growth["name_clip"] == FROZEN_GROWTH_NAME_CLIP
    assert growth["daily_artifact"] == "daily.parquet"
    primary = frozen_mhs_backtest_payload(_run())
    assert primary["exposure_multiplier"] == 1.0
    assert primary["name_clip"] is None
    assert primary["daily_artifact"] == "daily.parquet"
    json.dumps(growth)


def test_daily_frame_matches_ledger_aggregates() -> None:
    """Daily returns are exact and intraday lows, turnover, and funding reconcile to the 3m ledger."""
    import numpy as np

    run = _run()
    frame = frozen_mhs_daily_frame(run)
    assert list(frame.columns) == [
        "base_return", "stress_return",
        "base_equity_close", "stress_equity_close",
        "base_equity_low", "stress_equity_low",
        "base_turnover", "stress_turnover",
        "base_funding", "stress_funding", "target_gross", "max_name_weight",
    ]
    assert all(str(dtype) == "float64" for dtype in frame.dtypes)
    pd.testing.assert_series_equal(frame["base_return"], run.evidence.base_daily.returns, check_names=False)
    pd.testing.assert_series_equal(frame["stress_return"], run.evidence.stress_daily.returns, check_names=False)
    assert bool((frame["base_equity_low"] <= frame["base_equity_close"]).all())
    assert bool((frame["stress_equity_low"] <= frame["stress_equity_close"]).all())
    days = frame.index
    for prefix, replay in (("base", run.evidence.base), ("stress", run.evidence.stress)):
        turnover = replay.ledger.fill_turnover
        funding = replay.ledger.funding_charge
        span = turnover.index.normalize().isin(days)
        assert frame[f"{prefix}_turnover"].sum() == pytest.approx(float(turnover.loc[span].sum()), rel=1e-12)
        assert frame[f"{prefix}_funding"].sum() == pytest.approx(float(funding.loc[span].sum()), rel=1e-12)
    assert bool(np.allclose(frame["target_gross"].to_numpy(), 0.1))


def test_daily_frame_rejects_mismatched_and_nonfinite() -> None:
    """Disagreeing paired daily indexes or non-finite ledger values fail closed."""
    run = _run()
    shifted = run.evidence.stress_daily.returns.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    bad_daily = dataclasses.replace(run.evidence.stress_daily, returns=shifted)
    bad_evidence = dataclasses.replace(run.evidence, stress_daily=bad_daily)
    with pytest.raises(DataIntegrityError, match="indexes disagree"):
        frozen_mhs_daily_frame(dataclasses.replace(run, evidence=bad_evidence))
    bad_equity = run.evidence.base.ledger.equity.copy()
    bad_equity.iloc[-1] = float("inf")
    bad_ledger = dataclasses.replace(run.evidence.base.ledger, equity=bad_equity)
    bad_base = dataclasses.replace(run.evidence.base, ledger=bad_ledger)
    with pytest.raises(DataIntegrityError, match="finite"):
        frozen_mhs_daily_frame(dataclasses.replace(run, evidence=dataclasses.replace(run.evidence, base=bad_base)))


def test_persist_writes_daily_artifact_with_envelope(tmp_path: Path) -> None:
    """A fresh output persists daily.parquet beside result.json with the payload reference."""
    run = _run()
    output = tmp_path / "result.json"
    assert persist_frozen_mhs_backtest(run, output) == output
    daily = pd.read_parquet(tmp_path / "daily.parquet")
    assert list(daily.columns) == [
        "base_return", "stress_return",
        "base_equity_close", "stress_equity_close",
        "base_equity_low", "stress_equity_low",
        "base_turnover", "stress_turnover",
        "base_funding", "stress_funding", "target_gross", "max_name_weight",
    ]
    assert json.loads(output.read_text(encoding="utf-8"))["daily_artifact"] == "daily.parquet"


def test_persist_failure_leaves_no_partial_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing JSON write removes the already-written daily artifact and every temp file."""
    import src.mhs.frozen_research_report as report_mod

    run = _run()
    output = tmp_path / "result.json"
    monkeypatch.setattr(report_mod.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        persist_frozen_mhs_backtest(run, output)
    assert not output.exists()
    assert not (tmp_path / "daily.parquet").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_persist_rejects_occupied_daily_artifact(tmp_path: Path) -> None:
    """A pre-existing daily.parquet is refused before any serialization."""
    run = _run()
    (tmp_path / "daily.parquet").write_text("occupied", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="daily artifact"):
        persist_frozen_mhs_backtest(run, tmp_path / "result.json")


def test_payload_states_execution_provenance() -> None:
    """A maker run serializes the crossing model, anchor, and maker fee."""
    import dataclasses

    run = _run()
    maker_run = dataclasses.replace(
        run, request=dataclasses.replace(run.request, execution_bound="OHLCV_STRICT_PROXY"),
    )
    payload = frozen_mhs_backtest_payload(maker_run)
    assert payload["execution_bound"] == "OHLCV_STRICT_PROXY"
    assert payload["decision_anchor"] == "submit_bar"
    assert payload["maker_fee_bps"] == maker_run.request.base_spec.maker_fee_bps
    json.dumps(payload)


def test_daily_frame_reports_max_name_weight() -> None:
    """The largest single-name weight is reported per entry day and zero otherwise."""
    import numpy as np

    run = _run()
    frame = frozen_mhs_daily_frame(run)
    assert frame["max_name_weight"].max() == pytest.approx(0.05)
    assert bool((frame["max_name_weight"] <= frame["target_gross"]).all())
    assert bool(np.allclose(frame["target_gross"].to_numpy(), 0.1))
    assert bool(np.allclose(frame["max_name_weight"].to_numpy(), 0.05))
