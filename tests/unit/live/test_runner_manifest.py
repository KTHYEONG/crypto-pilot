"""Run-manifest compatibility across the strategy rename (spec 38 part 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.live.test_runner import _seed_run_cycle_artifact


def _manifest_path(tmp_path: Path, record_run_id: str) -> Path:
    return tmp_path / "data" / "state" / "runs" / record_run_id / "run_manifest.json"


def test_legacy_paper_manifest_accepted_and_upgraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-rename paper manifest (legacy id, current everything else) upgrades in place."""
    import src.live.runner as runner_mod

    record_run_id = "frozen_top20_v2_repeg_fillcap_20261002"
    settings, _, _, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    manifest = runner_mod._current_run_manifest(settings, now)
    manifest["strategy_id"] = "frozen_mhs_top20_v2"
    path = _manifest_path(tmp_path, record_run_id)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    runner_mod._assert_run_manifest_compatible(settings)

    upgraded = json.loads(path.read_text(encoding="utf-8"))
    assert upgraded["strategy_id"] == "flow_mom_top20"
    for key, value in manifest.items():
        if key == "strategy_id":
            continue
        assert upgraded[key] == value


def test_different_strategy_still_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest naming another strategy fails closed with a strategy_id mismatch."""
    import src.live.runner as runner_mod
    from src.common.errors import DataIntegrityError

    record_run_id = "flow_mom_top40_control_probe"
    settings, _, _, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    manifest = runner_mod._current_run_manifest(settings, now)
    manifest["strategy_id"] = "flow_mom_top40_control"
    path = _manifest_path(tmp_path, record_run_id)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match=r"run manifest mismatch key=strategy_id"):
        runner_mod._assert_run_manifest_compatible(settings)


def test_unresolvable_stored_id_still_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-string or unknown stored id never resolves; the manifest mismatches."""
    import src.live.runner as runner_mod
    from src.common.errors import DataIntegrityError

    record_run_id = "unresolvable_probe"
    settings, _, _, now = _seed_run_cycle_artifact(tmp_path, monkeypatch, record_run_id)
    path = _manifest_path(tmp_path, record_run_id)
    for stored in (123, "no_such_strategy"):
        manifest = runner_mod._current_run_manifest(settings, now)
        manifest["strategy_id"] = stored
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        with pytest.raises(DataIntegrityError, match=r"run manifest mismatch key=strategy_id"):
            runner_mod._assert_run_manifest_compatible(settings)
