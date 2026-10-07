"""Per-run acceptance artifacts remain isolated even when the CLI fails."""

from pathlib import Path
import subprocess

import pytest

from tests.integration.mhs import test_mhs_growth_budget_acceptance as acceptance


@pytest.mark.parametrize("failure", [None, "cli", "assertion"])
def test_cli_run_cleanup_preserves_other_runs(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(acceptance, "DATA_DIR", tmp_path)
    other = tmp_path / "research" / "mhs" / "existing"
    other.mkdir(parents=True)
    retained = other / "report.json"
    retained.write_text("retained", encoding="utf-8")
    run_dirs: list[Path] = []

    def fake_run(cmd, **kwargs):
        assert "env" not in kwargs
        assert kwargs["check"] is True
        run_id = cmd[cmd.index("--run-id") + 1]
        assert run_id.startswith("pytest-growth-acceptance-")
        run_dir = other.parent / run_id
        run_dirs.append(run_dir)
        run_dir.mkdir()
        (run_dir / "mhs_horizon_diagnostic.json").write_text("{}", encoding="utf-8")
        if failure == "cli":
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)

    def invoke():
        with acceptance._run_cli([]) as report_path:
            assert report_path.exists()
            if failure == "assertion":
                raise AssertionError("failed numeric pin")

    if failure == "cli":
        with pytest.raises(subprocess.CalledProcessError):
            invoke()
    elif failure == "assertion":
        with pytest.raises(AssertionError, match="failed numeric pin"):
            invoke()
    else:
        invoke()
        invoke()
        assert run_dirs[0] != run_dirs[1]
    assert all(not path.exists() for path in run_dirs)
    assert retained.read_text(encoding="utf-8") == "retained"
