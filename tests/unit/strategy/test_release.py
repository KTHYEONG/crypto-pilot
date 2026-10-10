"""Spec 38 part 6: the committed release record loads, digests verify, acceptance is strict."""

from __future__ import annotations

import json
import shutil

import pytest

from src.common.errors import DataIntegrityError
from src.strategy.release import (
    EvaluationCriteria,
    criteria_digest,
    load_release,
    record_acceptance,
    release_path,
    strategy_spec_digest,
)
from src.strategy.targets import FLOW_MOM_TOP20


def test_committed_release_loads_and_digests_verify() -> None:
    """The shipped flow_mom_top20 record is self-consistent."""
    release = load_release("flow_mom_top20")
    assert release.strategy_id == "flow_mom_top20"
    assert release.legacy_ids == ("frozen_mhs_top20_v2",)
    assert release.criteria_digest == criteria_digest(release.criteria)
    assert release.criteria_digest == criteria_digest(EvaluationCriteria())
    assert release.spec_digest == strategy_spec_digest(FLOW_MOM_TOP20, dict(release.sizing))
    assert release.verdict is None
    assert release.evaluation_digest is None


def test_criteria_change_clears_verdict(tmp_path) -> None:
    """A criteria edit changes the digest: the stored verdict no longer verifies."""
    root = tmp_path / "root"
    target = root / "src" / "strategy" / "releases"
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
    target = root / "src" / "strategy" / "releases"
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
    target = tmp_path / "root" / "src" / "strategy" / "releases"
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
    target = root / "src" / "strategy" / "releases"
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
                      {"max_top5_funding_dependence": "false"}):
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


def test_release_without_seat_fraction_key_fails_closed(tmp_path) -> None:
    """A release lacking the fraction key raises instead of taking a silent default."""
    raw = json.loads(release_path("flow_mom_top20").read_text(encoding="utf-8"))
    del raw["criteria"]["max_withdrawn_seat_fraction"]
    target = tmp_path / "root" / "src" / "strategy" / "releases"
    target.mkdir(parents=True)
    (target / "flow_mom_top20.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="max_withdrawn_seat_fraction"):
        load_release("flow_mom_top20", root=tmp_path / "root")


def test_release_envelope_mismatch_fails_closed(tmp_path) -> None:
    """A release whose envelope differs from the registered sizing cap fails to load."""
    raw = json.loads(release_path("flow_mom_top20").read_text(encoding="utf-8"))
    raw["risk_envelope"] = "growth"
    target = tmp_path / "root" / "src" / "strategy" / "releases"
    target.mkdir(parents=True)
    (target / "flow_mom_top20.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="risk envelope"):
        load_release("flow_mom_top20", root=tmp_path / "root")


def test_spec_digest_covers_exposure_cap() -> None:
    """The sizing exposure cap participates in the spec digest."""
    from src.core.params import ACCOUNT_EXPOSURE_CAP

    release = load_release("flow_mom_top20")
    assert release.sizing.get("exposure_cap") == ACCOUNT_EXPOSURE_CAP
    altered = dict(release.sizing)
    altered["exposure_cap"] = 10.0
    assert strategy_spec_digest(FLOW_MOM_TOP20, altered) != release.spec_digest
