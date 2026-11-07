"""Spec 38 part 6: the committed release record loads, digests verify, acceptance is strict."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

import src.strategy.release as release_mod
from src.common.errors import DataIntegrityError
from src.strategy.release import (
    EvaluationCriteria,
    ImplementationAmendment,
    append_implementation_amendment,
    criteria_digest,
    effective_spec_digest,
    load_release,
    record_acceptance,
    release_path,
    strategy_spec_digest,
)
from src.strategy.targets import FLOW_MOM_TOP20, FLOW_MOM_TOP20_ACCOUNT_UNIT

_IMPLEMENTATION_KEYS = {"implementation", "shared_parameters"}


def test_committed_release_loads_and_digests_verify() -> None:
    """The shipped flow_mom_top20 record is self-consistent."""
    release = load_release("flow_mom_top20")
    assert release.strategy_id == "flow_mom_top20"
    assert release.legacy_ids == ("frozen_mhs_top20_v2",)
    assert release.criteria_digest == criteria_digest(release.criteria)
    assert release.criteria_digest == criteria_digest(EvaluationCriteria())
    assert effective_spec_digest(release) == strategy_spec_digest(FLOW_MOM_TOP20, dict(release.sizing))
    assert release.verdict in (None, "accept")
    assert (release.verdict is None) == (release.evaluation_digest is None)


def test_criteria_change_clears_verdict(tmp_path) -> None:
    """A criteria edit changes the digest: the stored verdict no longer verifies."""
    root = tmp_path / "root"
    target = root / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    shutil.copy(release_path("flow_mom_top20"), target / "flow_mom_top20.json")
    raw = json.loads((target / "flow_mom_top20.json").read_text(encoding="utf-8"))
    raw["criteria"]["dsr_min"] = 0.5
    (target / "flow_mom_top20.json").write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="criteria digest mismatch"):
        load_release("flow_mom_top20", root=root)


def test_accept_requires_matching_digests(tmp_path) -> None:
    """Acceptance writes only for matching spec digests; mismatches fail closed."""
    root = tmp_path / "root"
    target = root / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    shutil.copy(release_path("flow_mom_top20"), target / "flow_mom_top20.json")
    release = load_release("flow_mom_top20", root=root)
    with pytest.raises(DataIntegrityError, match="matching spec digest"):
        record_acceptance("flow_mom_top20", spec_digest="wrong", evaluation_digest="eval-1", root=root)
    accepted = record_acceptance(
        "flow_mom_top20", spec_digest=release.spec_digest, evaluation_digest="eval-1", root=root,
    )
    assert accepted.verdict == "accept"
    assert accepted.evaluation_digest == "eval-1"


def test_release_guards_reject_bad_records(tmp_path) -> None:
    """Missing, corrupt, and inconsistent release records fail closed."""
    import pytest

    from src.common.errors import DataIntegrityError
    from src.strategy.release import release_path

    with pytest.raises(DataIntegrityError):
        load_release("flow_mom_top20", root=tmp_path / "empty")
    with pytest.raises(DataIntegrityError):
        release_path("")
    target = tmp_path / "root" / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    dest = target / "flow_mom_top20.json"
    shutil.copy(release_path("flow_mom_top20"), dest)
    from src.strategy.release import _parse_criteria

    with pytest.raises(DataIntegrityError):
        _parse_criteria([])
    with pytest.raises(DataIntegrityError):
        _parse_criteria({"dsr_min": "high"})
    dest.write_text("{broken", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="corrupt"):
        load_release("flow_mom_top20", root=tmp_path / "root")
    dest.write_text("[1]", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="corrupt"):
        load_release("flow_mom_top20", root=tmp_path / "root")
    shutil.copy(release_path("flow_mom_top20"), dest)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    del raw["design_data_cutoff"]
    dest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="invalid"):
        load_release("flow_mom_top20", root=tmp_path / "root")
    shutil.copy(release_path("flow_mom_top20"), dest)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    raw["verdict"] = "maybe"
    dest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="verdict invalid"):
        load_release("flow_mom_top20", root=tmp_path / "root")


def test_acceptance_validates_digests(tmp_path, monkeypatch) -> None:
    """Acceptance needs a non-empty evaluation digest and a readable record."""
    import pytest

    from src.common.errors import DataIntegrityError

    root = tmp_path / "root"
    target = root / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    shutil.copy(release_path("flow_mom_top20"), target / "flow_mom_top20.json")
    release = load_release("flow_mom_top20", root=root)
    with pytest.raises(DataIntegrityError, match="criteria digest"):
        record_acceptance("flow_mom_top20", spec_digest=release.spec_digest, evaluation_digest="eval",
                          expected_criteria_digest="stale", root=root)
    with pytest.raises(DataIntegrityError, match="non-empty"):
        record_acceptance(
            "flow_mom_top20", spec_digest=release.spec_digest, evaluation_digest="", root=root,
        )
    (target / "flow_mom_top20.json").unlink()
    with pytest.raises(DataIntegrityError, match=r"missing|unreadable"):
        record_acceptance(
            "flow_mom_top20", spec_digest=release.spec_digest,
            evaluation_digest="eval-1", root=root,
        )
    shutil.copy(release_path("flow_mom_top20"), target / "flow_mom_top20.json")
    release = load_release("flow_mom_top20", root=root)
    (target / "flow_mom_top20.json").unlink()
    import src.strategy.release as release_mod

    monkeypatch.setattr(release_mod, "load_release", lambda *args, **kwargs: release)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        record_acceptance(
            "flow_mom_top20", spec_digest=release.spec_digest,
            evaluation_digest="eval-1", root=root,
        )


def test_criteria_reject_nonfinite_and_coerced_thresholds() -> None:
    import dataclasses

    for overrides in ({"dsr_min": float("nan")}, {"min_holdout_days": 1},
                      {"max_top5_funding_dependence": "false"},
                      {"growth_horizon_years": 0.5}, {"growth_horizon_years": 11.0},
                      {"growth_horizon_years": float("nan")}, {"overbet_probe_scale": 0.0},
                      {"overbet_probe_scale": 1.0}, {"overbet_probe_scale": float("inf")}):
        with pytest.raises(DataIntegrityError):
            dataclasses.replace(EvaluationCriteria(), **overrides)


def test_criteria_digest_covers_withdrawn_seat_fraction() -> None:
    """The same criteria with a different fraction digests differently."""
    import dataclasses

    assert criteria_digest(EvaluationCriteria()) != criteria_digest(
        dataclasses.replace(EvaluationCriteria(), max_withdrawn_seat_fraction=0.002)
    )
    for bad in (float("nan"), -0.001, 0.05, True):
        with pytest.raises(DataIntegrityError):
            dataclasses.replace(EvaluationCriteria(), max_withdrawn_seat_fraction=bad)


@pytest.mark.parametrize("overrides", [{"growth_horizon_years": 4.0}, {"overbet_probe_scale": 0.5}])
def test_criteria_digest_covers_growth_policy(overrides) -> None:
    import dataclasses

    assert criteria_digest(EvaluationCriteria()) != criteria_digest(
        dataclasses.replace(EvaluationCriteria(), **overrides)
    )


def test_release_without_seat_fraction_key_fails_closed(tmp_path) -> None:
    """A release lacking the fraction key raises instead of taking a silent default."""
    raw = json.loads(release_path("flow_mom_top20").read_text(encoding="utf-8"))
    del raw["criteria"]["max_withdrawn_seat_fraction"]
    target = tmp_path / "root" / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    (target / "flow_mom_top20.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="max_withdrawn_seat_fraction"):
        load_release("flow_mom_top20", root=tmp_path / "root")


@pytest.mark.parametrize("retired", ["risk_envelope", "criteria.risk_envelope", "criteria.holdout_drawdown_max_quantile"])
def test_release_envelope_mismatch_fails_closed(tmp_path, retired) -> None:
    """A release carrying any retired envelope key fails to load."""
    raw = json.loads(release_path("flow_mom_top20").read_text(encoding="utf-8"))
    if retired.startswith("criteria."):
        raw["criteria"][retired.removeprefix("criteria.")] = "growth"
    else:
        raw[retired] = "growth"
    raw["criteria_digest"] = "stale"
    target = tmp_path / "root" / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    (target / "flow_mom_top20.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="retired key"):
        load_release("flow_mom_top20", root=tmp_path / "root")


def test_spec_digest_covers_exposure_cap() -> None:
    """The sizing exposure cap participates in the spec digest."""
    from src.core.params import ACCOUNT_EXPOSURE_CAP

    release = load_release("flow_mom_top20")
    assert release.sizing.get("exposure_cap") == ACCOUNT_EXPOSURE_CAP
    altered = dict(release.sizing)
    altered["exposure_cap"] = 3.0
    assert strategy_spec_digest(FLOW_MOM_TOP20, altered) != release.spec_digest


def test_signal_digest_ignores_sizing_and_params() -> None:
    """Two sizings differing only in exposure_cap share the signal digest but not the spec digest."""
    import dataclasses

    from src.strategy.release import strategy_signal_digest
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    assert strategy_signal_digest(FLOW_MOM_TOP20) == strategy_signal_digest(FLOW_MOM_TOP20_ACCOUNT_UNIT)
    assert strategy_signal_digest(FLOW_MOM_TOP20) == strategy_signal_digest(
        dataclasses.replace(FLOW_MOM_TOP20, exposure_multiplier=3.0)
    )
    assert strategy_spec_digest(FLOW_MOM_TOP20, {"exposure_cap": 1.0}) != strategy_spec_digest(
        FLOW_MOM_TOP20, {"exposure_cap": 3.0})


def test_signal_digest_breaks_on_member_change() -> None:
    """Flipping one member sign changes the signal digest."""
    import dataclasses

    from src.strategy.release import strategy_signal_digest
    from src.strategy.targets import FeatureMember

    flipped = dataclasses.replace(
        FLOW_MOM_TOP20,
        members=tuple(
            FeatureMember(name=m.name, sign=-m.sign) if position == 0 else m
            for position, m in enumerate(FLOW_MOM_TOP20.members)
        ),
    )
    assert strategy_signal_digest(flipped) != strategy_signal_digest(FLOW_MOM_TOP20)


def test_signal_digest_keeps_foreign_strategy_identity() -> None:
    """A non-flow_mom spec keeps its own id and clip in the signal digest."""
    import dataclasses

    from src.strategy.release import strategy_signal_digest

    foreign = dataclasses.replace(FLOW_MOM_TOP20, strategy_id="other", name_clip=0.02)
    assert strategy_signal_digest(foreign) != strategy_signal_digest(FLOW_MOM_TOP20)


def test_digest_covers_participation_basis() -> None:
    """The committed release carries the ADV basis and a matching digest."""
    release = load_release("flow_mom_top20")
    assert release.criteria.participation_basis == "adv30_median_prior_day"
    assert release.criteria_digest == criteria_digest(release.criteria)
    with pytest.raises(DataIntegrityError):
        EvaluationCriteria(participation_basis="trailing_24h")


def test_release_target_equals_default_seed() -> None:
    """The committed release targets the declared retail seed."""
    from src.core.params import ACCOUNT_DEFAULT_CAPITAL_USDT

    release = load_release("flow_mom_top20")
    assert ACCOUNT_DEFAULT_CAPITAL_USDT == 1000.0
    assert release.target_capital_usdt == ACCOUNT_DEFAULT_CAPITAL_USDT == 1000.0


def _write_identity_unit_run(run_dir, *, start: str = "2025-01-01", n: int = 100) -> None:
    """Unit run satisfying every book-identity clause of the release."""
    import numpy as np
    import pandas as pd

    from src.strategy.targets import FLOW_MOM_TOP20

    run_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    pd.DataFrame(
        {"base_return": np.full(n, 0.001), "stress_return": np.full(n, 0.0008)},
        index=index,
    ).to_parquet(run_dir / "daily.parquet")
    (run_dir / "result.json").write_text(
        json.dumps({
            "strategy_id": "flow_mom_top20",
            "breadth": FLOW_MOM_TOP20.breadth,
            "members": [{"name": m.name, "sign": m.sign} for m in FLOW_MOM_TOP20.members],
            "min_rank_symbols": FLOW_MOM_TOP20.min_rank_symbols,
            "design_data_cutoff": pd.Timestamp(FLOW_MOM_TOP20.design_data_cutoff).tz_convert("UTC").isoformat(),
            "name_clip": FLOW_MOM_TOP20_ACCOUNT_UNIT.name_clip,
            "exposure_multiplier": 1.0,
            "evaluation_start": index[0].isoformat(),
            "evaluation_end": (index[-1] + pd.Timedelta(days=1)).isoformat(),
            "ledger_certified": True,
            "base_valid": True,
            "stress_valid": True,
            "source_gap_excluded_count": 0,
            "base_source_gaps": 0,
            "stress_source_gaps": 0,
            "roster_seat_days": 40000,
            "participation_scale": 100000.0,
            "participation_basis": "adv30_median_prior_day",
            "data_availability_withdrawals": [],
            "limitations": [],
            "report_periods": {"evaluation": {"base_cagr": 0.3}},
        }),
        encoding="utf-8",
    )


def _write_identity_neighbor(run_dir, *, spec, start: str = "2025-01-01", n: int = 100) -> None:
    """Neighbor run carrying the declared breadth/members with certified ledger."""
    import numpy as np
    import pandas as pd

    run_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    pd.DataFrame({"stress_return": np.full(n, 0.0008)}, index=index).to_parquet(run_dir / "daily.parquet")
    (run_dir / "result.json").write_text(
        json.dumps({
            "breadth": spec.breadth,
            "members": [{"name": m.name, "sign": m.sign} for m in spec.members],
            "ledger_certified": True,
            "source_gap_excluded_count": 0,
            "base_source_gaps": 0,
            "stress_source_gaps": 0,
            "roster_seat_days": 40000,
            "data_availability_withdrawals": [],
        }),
        encoding="utf-8",
    )


def _write_identity_account(run_dir, *, capital: float, start: str = "2025-01-01", n: int = 100) -> None:
    """Account run whose window matches the identity unit run."""
    import numpy as np
    import pandas as pd

    run_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(start=start, periods=n, freq="D", tz="UTC")
    equity = capital * np.cumprod(np.full(n, 1.001))
    pd.DataFrame({"equity": equity, "exposure": np.full(n, 1.2)}, index=index).to_parquet(
        run_dir / "account_daily.parquet"
    )
    pd.DataFrame({"equity": equity * 0.999, "exposure": np.full(n, 1.1)}, index=index).to_parquet(
        run_dir / "account_stress_daily.parquet"
    )
    (run_dir / "account.json").write_text(
        json.dumps({
            "capital": capital,
            "execution": {"mode": "maker"},
            "evaluation_start": index[0].isoformat(),
            "evaluation_end": (index[-1] + pd.Timedelta(days=1)).isoformat(),
            "cagr": 0.3,
            "mdd": -0.1,
            "mean_exposure": 1.2,
            "liquidated_at": None,
            "initial_margin_breaches": 0,
            "stress_execution": {"liquidated_at": None},
        }),
        encoding="utf-8",
    )


def test_account_run_at_other_capital_is_rejected(tmp_path) -> None:
    """An account run at 2100 USDT fails the 1000 USDT release identity check."""
    import src.cli.commands.evaluate as evaluate_mod
    from src.cli.commands.backtest import _neighbor_specs
    from src.strategy.targets import FLOW_MOM_TOP20

    unit = tmp_path / "unit"
    _write_identity_unit_run(unit)
    neighbors = tuple(
        tmp_path / f"neighbor{position}" for position in range(len(_neighbor_specs(FLOW_MOM_TOP20)))
    )
    for path, spec in zip(neighbors, _neighbor_specs(FLOW_MOM_TOP20), strict=True):
        _write_identity_neighbor(path, spec=spec)
    matching = tmp_path / "account_1000"
    _write_identity_account(matching, capital=1000.0)
    matched = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=matching,
        neighbor_runs=neighbors,
    )
    assert matched.book_identity_ok is True
    other = tmp_path / "account_2100"
    _write_identity_account(other, capital=2100.0)
    mismatched = evaluate_mod.build_evaluation_inputs(
        strategy_id="flow_mom_top20", unit_run=unit, account_run=other,
        neighbor_runs=neighbors,
    )
    assert mismatched.book_identity_ok is False


def _amendment(from_digest: str, to_digest: str, **overrides) -> ImplementationAmendment:
    fields = {
        "from_spec_digest": from_digest,
        "to_spec_digest": to_digest,
        "baseline_commit": "a1b2c3d",
        "candidate_commit": None,
        "evidence_sha256": "e" * 64,
        "window_start": "2026-01-01T00:00:00+00:00",
        "window_end": "2026-11-01T00:00:00+00:00",
        "decision_rows": 100,
        "max_abs_diff": 1e-16,
        "recorded_at": "2026-11-07T12:00:00+00:00",
    }
    fields.update(overrides)
    return ImplementationAmendment(**fields)


def _root_copy(tmp_path) -> Path:
    root = tmp_path / "root"
    target = root / "crypto-pilot" / "releases"
    target.mkdir(parents=True)
    raw = json.loads(release_path("flow_mom_top20").read_text(encoding="utf-8"))
    raw.pop("implementation_amendments", None)  # model an unamended release; chains are built per test
    (target / "flow_mom_top20.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _unamended_release():
    return dataclasses.replace(load_release("flow_mom_top20"), implementation_amendments=())




def test_effective_digest_follows_chain() -> None:
    """Chained amendments resolve to the last ``to``; a release without amendments keeps its digest."""
    release = _unamended_release()
    chained = dataclasses.replace(
        release,
        implementation_amendments=(_amendment(release.spec_digest, "1" * 64), _amendment("1" * 64, "2" * 64)),
    )
    assert effective_spec_digest(chained) == "2" * 64
    assert effective_spec_digest(release) == release.spec_digest


@pytest.mark.parametrize(
    "chain",
    [
        (_amendment("9" * 64, "1" * 64),),
        (_amendment("SPEC", "1" * 64), _amendment("9" * 64, "2" * 64)),
        (_amendment("SPEC", "1" * 64), _amendment("1" * 64, "SPEC")),
    ],
    ids=["wrong_first_from", "gap", "repeated_to"],
)
def test_broken_chain_raises(chain) -> None:
    release = _unamended_release()
    chain = tuple(
        dataclasses.replace(a, from_spec_digest=release.spec_digest if a.from_spec_digest == "SPEC" else a.from_spec_digest,
                            to_spec_digest=release.spec_digest if a.to_spec_digest == "SPEC" else a.to_spec_digest)
        for a in chain
    )
    broken = dataclasses.replace(release, implementation_amendments=chain)
    with pytest.raises(DataIntegrityError, match="chain"):
        effective_spec_digest(broken)


@pytest.mark.parametrize(
    "overrides",
    [
        {"from_spec_digest": ""}, {"to_spec_digest": ""}, {"baseline_commit": ""},
        {"candidate_commit": ""}, {"candidate_commit": 7}, {"evidence_sha256": ""},
        {"window_start": ""}, {"window_end": ""}, {"recorded_at": ""},
        {"decision_rows": -1}, {"decision_rows": True}, {"max_abs_diff": float("nan")},
        {"max_abs_diff": -1.0}, {"max_abs_diff": True},
    ],
)
def test_amendment_rejects_malformed_fields(overrides) -> None:
    with pytest.raises(DataIntegrityError):
        _amendment("1" * 64, "2" * 64, **overrides)
    with pytest.raises(DataIntegrityError, match="must change"):
        _amendment("1" * 64, "1" * 64)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.__setitem__("implementation_amendments", "not-a-list"),
        lambda raw: raw.__setitem__("implementation_amendments", [{"from_spec_digest": "1"}]),
        lambda raw: raw.__setitem__("implementation_amendments", [dict(
            dataclasses.asdict(_amendment("SPEC", "1" * 64)), extra=1)]),
    ],
    ids=["not_list", "missing_keys", "extra_key"],
)
def test_release_rejects_malformed_amendment_records(tmp_path, mutate) -> None:
    root = _root_copy(tmp_path)
    path = release_path("flow_mom_top20", root=root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="amendment"):
        load_release("flow_mom_top20", root=root)


def test_append_refuses_stale_from(tmp_path) -> None:
    """An amendment that does not start at the effective digest is refused and the file is untouched."""
    root = _root_copy(tmp_path)
    path = release_path("flow_mom_top20", root=root)
    before = path.read_bytes()
    with pytest.raises(DataIntegrityError, match="effective spec digest"):
        append_implementation_amendment("flow_mom_top20", _amendment("9" * 64, "1" * 64), root=root)
    assert path.read_bytes() == before


def test_append_refuses_repeated_to(tmp_path) -> None:
    """A second amendment that returns to an earlier digest is refused as a repeated chain link."""
    root = _root_copy(tmp_path)
    release = load_release("flow_mom_top20", root=root)
    append_implementation_amendment(
        "flow_mom_top20", _amendment(release.spec_digest, "1" * 64), root=root,
    )
    with pytest.raises(DataIntegrityError, match="repeats"):
        append_implementation_amendment("flow_mom_top20", _amendment("1" * 64, release.spec_digest), root=root)


def test_append_preserves_other_keys(tmp_path) -> None:
    """Appending writes one list entry; criteria, verdict and evaluation digest stay byte-equal when parsed."""
    root = _root_copy(tmp_path)
    path = release_path("flow_mom_top20", root=root)
    before = json.loads(path.read_text(encoding="utf-8"))
    release = load_release("flow_mom_top20", root=root)
    append_implementation_amendment(
        "flow_mom_top20", _amendment(release.spec_digest, "1" * 64), root=root,
    )
    after = json.loads(path.read_text(encoding="utf-8"))
    recorded = after.pop("implementation_amendments")
    assert recorded[0]["to_spec_digest"] == "1" * 64
    assert after == before
    assert not path.with_name(f"{path.name}.tmp").exists()
    assert effective_spec_digest(load_release("flow_mom_top20", root=root)) == "1" * 64


def test_append_fails_closed_when_record_vanishes(tmp_path, monkeypatch) -> None:
    """A record removed after the chain check is reported as unreadable, never silently recreated."""
    root = _root_copy(tmp_path)
    path = release_path("flow_mom_top20", root=root)
    release = load_release("flow_mom_top20", root=root)
    path.unlink()
    monkeypatch.setattr(release_mod, "load_release", lambda *args, **kwargs: release)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        append_implementation_amendment("flow_mom_top20", _amendment(release.spec_digest, "1" * 64), root=root)
    assert not path.exists()
