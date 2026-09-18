"""Documentation retirement and sealed-delivery confidentiality contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path

_APPROVED_SEALED_NAMES = frozenset(
    {
        "strategy_params.json.enc",
        "strategy_bootstrap.parquet.enc",
        "deployed_target_weights.parquet.enc",
    }
)
_PLAINTEXT_STRATEGY_NAMES = frozenset(
    {
        "strategy_params.json",
        "strategy_bootstrap.parquet",
        "deployed_target_weights.parquet",
    }
)
_RETIRED_PATHS = (    Path("docs/results/mhs_backtest"),
    Path("docs/results/mhs_run_history"),
    Path("docs/results/mhs_process_backtest.json"),
    Path("docs/results/mhs_process_3m_backtest.failure.json"),
    Path("docs/results/mhs_horizon_diagnostic_artifacts"),
)
# Legacy run-history compatibility readers owned by earlier result-tree specs;
# they never select a documentation destination for new output.
_KNOWN_READONLY_COMPAT = frozenset(
    {
        (
            "src/mhs/run_history.py",
            '_DEFAULT_HISTORY_DIR = Path("docs") / "results" / "mhs_run_history"',
        ),
        (
            "src/mhs/run_history.py",
            'if str(history_dir).endswith("docs/results/mhs_run_history"):',
        ),
        (
            "src/mhs/run_history.py",
            'if text.endswith("docs/results/mhs_run_history"):',
        ),
    }
)


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], text=True)  # noqa: S603, S607
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_documentation_has_no_generated_result_payloads() -> None:
    tracked = _tracked_files()
    results_payloads = [path for path in tracked if path.startswith("docs/results/")]
    assert results_payloads == [], f"payloads under docs/results/: {results_payloads}"
    binary_payloads = [
        path
        for path in tracked
        if path.startswith("docs/") and path.endswith((".parquet", ".jsonl", ".log"))
    ]
    assert binary_payloads == [], f"generated payloads under docs/: {binary_payloads}"


def test_only_sealed_delivery_artifacts_may_be_tracked() -> None:
    tracked = _tracked_files()
    delivered = [path for path in tracked if path.startswith("deploy/mhs/")]
    assert delivered, "delivery boundary deploy/mhs/ must carry the sealed artifacts"
    violations = [path for path in delivered if Path(path).name not in _APPROVED_SEALED_NAMES]
    assert violations == [], f"unapproved files under deploy/mhs/: {violations}"


def test_deployment_directory_has_no_plaintext_artifacts() -> None:
    tracked = _tracked_files()
    violations = [
        path
        for path in tracked
        if path.startswith("deploy/mhs/") and Path(path).name in _PLAINTEXT_STRATEGY_NAMES
    ]
    assert violations == [], f"plaintext strategy artifacts tracked: {violations}"
    for name in _PLAINTEXT_STRATEGY_NAMES:
        assert not Path("deploy/mhs", name).exists(), f"plaintext artifact present: {name}"


def test_legacy_generated_result_paths_are_absent() -> None:
    missing_ok = [path for path in _RETIRED_PATHS if not path.exists()]
    assert len(missing_ok) == len(_RETIRED_PATHS), (
        f"retired paths still present: {[str(p) for p in _RETIRED_PATHS if p.exists()]}"
    )
    assert not Path("docs/results").exists() or not any(Path("docs/results").iterdir())


def test_no_production_writer_targets_docs_results() -> None:
    violations: list[str] = []
    for source in sorted(Path("src").rglob("*.py")):
        rel = source.as_posix()
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if "docs" not in line or "results" not in line:
                continue
            if (rel, line.strip()) in _KNOWN_READONLY_COMPAT:
                continue
            violations.append(f"{rel}:{lineno}:{line.strip()}")
    assert violations == [], f"production docs/results writers: {violations}"


def test_gitignore_holds_retirement_boundaries() -> None:
    content = Path(".gitignore").read_text(encoding="utf-8")
    assert "docs/results/" in content.splitlines()
    assert "data/research/" in content
    assert "data/backtests/" in content
    assert "deploy/mhs/*.enc" in content
    assert "!docs/results/" not in content
    assert "docs/results/mhs_horizon_diagnostic_artifacts" not in content


def test_dockerignore_ships_only_sealed_delivery() -> None:
    content = Path(".dockerignore").read_text(encoding="utf-8")
    assert "docs/results" not in content
    assert "deploy/mhs/*" in content
    assert "!deploy/mhs/*.enc" in content
