"""Spec 38 part 6: append-only trial ledger counts every evaluated configuration."""

from __future__ import annotations

import pandas as pd

from src.evaluation.trials import TrialRecord, append_trial, trial_population


def _record(digest: str, sharpe: float | None) -> TrialRecord:
    return TrialRecord(
        family="flow_mom", spec_digest=digest,
        window_start=pd.Timestamp("2021-04-01", tz="UTC"),
        window_end=pd.Timestamp("2026-04-01", tz="UTC"),
        daily_sharpe=sharpe, n_obs=1095,
        recorded_at=pd.Timestamp("2026-10-08", tz="UTC"), source="cli",
    )


def test_idempotent_append_by_digest_and_window(tmp_path) -> None:
    """Same digest+window twice: one record, second append reports False."""
    path = tmp_path / "flow_mom.trials.jsonl"
    assert append_trial(_record("abc", 0.05), path=path) is True
    assert append_trial(_record("abc", 0.05), path=path) is False
    population = trial_population("flow_mom", path=path)
    assert population.n_trials == 96 + 1


def test_failed_runs_count_toward_trials(tmp_path) -> None:
    """A failed run carries daily_sharpe=None yet increments n_trials."""
    path = tmp_path / "flow_mom.trials.jsonl"
    append_trial(_record("failed-run", None), path=path)
    population = trial_population("flow_mom", path=path)
    assert population.n_trials == 96 + 1
    assert population.sharpes == ()


def test_documented_prior_search_counted(tmp_path) -> None:
    """The five-feature search lineage counts 96 prior trials (ADR-02)."""
    path = tmp_path / "flow_mom.trials.jsonl"
    path.write_text(
        '{"daily_sharpe": null, "family": "flow_mom", "n_obs": 2007, '
        '"recorded_at": "2026-10-08T00:00:00+00:00", "source": "documented_prior_search", '
        '"spec_digest": "documented_prior_search", '
        '"window_end": "2026-07-01T00:00:00+00:00", "window_start": "2021-01-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    population = trial_population("flow_mom", path=path)
    assert population.prior_trials == 96
    assert population.n_trials >= 96


def test_invalidation_preserves_history_and_allows_real_retry(tmp_path):
    import dataclasses
    import json
    from src.evaluation.trials import trial_record_digest

    path = tmp_path / "flow_mom.trials.jsonl"
    append_trial(_record("same-spec", 4.9), path=path)
    original = path.read_bytes()
    row = json.loads(original)
    path.with_name(path.name + ".invalidations.jsonl").write_text(
        json.dumps({"record_sha256": trial_record_digest(row), "reason": "synthetic fixture"}) + "\n", encoding="utf-8")
    assert trial_population("flow_mom", path=path).n_trials == 96
    assert path.read_bytes() == original
    actual = dataclasses.replace(_record("same-spec", 0.1), window_end=pd.Timestamp("2026-05-01", tz="UTC"))
    append_trial(actual, path=path)
    population = trial_population("flow_mom", path=path)
    assert population.n_trials == 97
    assert population.sharpes == (0.1,)


def test_corrupt_invalidation_fails_closed(tmp_path):
    import pytest
    from src.common.errors import DataIntegrityError

    path = tmp_path / "flow_mom.trials.jsonl"
    path.with_name(path.name + ".invalidations.jsonl").write_text("{broken", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="invalidation journal corrupt"):
        trial_population("flow_mom", path=path)


def test_trial_ledger_guards_reject_bad_records(tmp_path) -> None:
    """Corrupt records and ledgers fail closed instead of silently counting."""
    import pytest

    from src.common.errors import DataIntegrityError

    path = tmp_path / "flow_mom.trials.jsonl"
    base = _record("ok", 0.05)
    import dataclasses

    with pytest.raises(DataIntegrityError):
        append_trial(dataclasses.replace(base, family=""), path=path)
    with pytest.raises(DataIntegrityError):
        append_trial(dataclasses.replace(base, spec_digest=""), path=path)
    with pytest.raises(DataIntegrityError):
        append_trial(dataclasses.replace(base, source=""), path=path)
    with pytest.raises(DataIntegrityError):
        append_trial(
            dataclasses.replace(
                base,
                window_start=pd.Timestamp("2026-04-01", tz="UTC"),
                window_end=pd.Timestamp("2021-04-01", tz="UTC"),
            ),
            path=path,
        )
    with pytest.raises(DataIntegrityError):
        append_trial(dataclasses.replace(base, n_obs=-1), path=path)
    with pytest.raises(DataIntegrityError):
        append_trial(dataclasses.replace(base, daily_sharpe=float("inf")), path=path)
    with pytest.raises(DataIntegrityError):
        append_trial(
            dataclasses.replace(base, window_start=pd.Timestamp("2021-04-01")), path=path
        )
    with pytest.raises(DataIntegrityError):
        append_trial(dataclasses.replace(base, window_start="2021-04-01"), path=path)
    with pytest.raises(DataIntegrityError):
        trial_population("", path=path)

    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        append_trial(base, path=path)
    with pytest.raises(DataIntegrityError):
        trial_population("flow_mom", path=path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        append_trial(base, path=path)
    path.write_text('{"family": "flow_mom"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        trial_population("flow_mom", path=path)
    path.write_text(
        '{"family": "flow_mom", "spec_digest": "x", "daily_sharpe": "nan"}\n', encoding="utf-8"
    )
    with pytest.raises(DataIntegrityError):
        trial_population("flow_mom", path=path)


def test_trial_ledger_skips_blanks_and_foreign_families(tmp_path) -> None:
    """Blank lines and other families never pollute the population."""
    path = tmp_path / "flow_mom.trials.jsonl"
    path.write_text("\n   \n", encoding="utf-8")
    append_trial(_record("aaa", 0.05), path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(
            '{"family": "other", "spec_digest": "zzz", "daily_sharpe": 0.9, '
            '"window_start": "2021-01-01T00:00:00+00:00", "window_end": "2026-07-01T00:00:00+00:00", '
            '"n_obs": 100, "recorded_at": "2026-10-08T00:00:00+00:00", "source": "cli"}\n'
        )
        handle.write("   \n")
    population = trial_population("flow_mom", path=path)
    assert population.n_trials == 96 + 1
    assert population.sharpes == (0.05,)
