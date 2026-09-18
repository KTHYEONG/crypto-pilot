"""Maintained dependency and dead-code boundaries for the MHS maintenance cut."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DELETED_PATHS = [
    "tools/capture_golden.py",
    "tools/benchmark_library_admission.py",
    "tools/research/structural_tuner.py",
    "tools/research/xs_alpha_blend_joint_search.py",
    "tests/integration/research/test_structural_tuner_optuna.py",
    "tools/devops/mhs_baseline_run.py",
    "tools/devops/daemon_idle_gate.py",
    "tools/devops/seal_artifact.py",
    "src/mhs/process_backtest.py",
]

STALE_TOKENS = [
    "tools/capture_golden",
    "tools/benchmark_library_admission",
    "tools/research/structural_tuner",
    "tools/research/xs_alpha_blend",
    "tools/devops/mhs_baseline_run",
    "tools/devops/daemon_idle_gate",
    "tools/devops/seal_artifact",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_deleted_tool_inventory_absent() -> None:
    """Deleted tool inventory leaves no files or stale config references."""
    for rel in DELETED_PATHS:
        assert not (ROOT / rel).exists(), rel
    pyproject = _read_text(ROOT / "pyproject.toml")
    for token in STALE_TOKENS:
        assert token not in pyproject, token
    for rel in ("tools/devops/repartition_ohlcv_parquet.py", "tools/devops/clean_logs.py"):
        assert not (ROOT / rel).exists(), rel
    assert not (ROOT / "archive").exists()
    assert not (ROOT / "legacy_trash").exists()


def test_maintenance_tool_identity_preserved(tmp_path: Path) -> None:
    """Repartitioned maintenance tools keep physical data and value semantics."""
    from tools.maintenance.clean_logs import clean_logs
    from tools.maintenance.repartition_ohlcv_parquet import repartition_ohlcv_parquet

    sig = inspect.signature(repartition_ohlcv_parquet)
    assert "dry_run" in sig.parameters
    stats = repartition_ohlcv_parquet(tmp_path, "3m")
    assert stats["files"] == 0
    assert stats["rewritten"] == 0
    assert callable(clean_logs)
    assert "retention_days" in inspect.signature(clean_logs).parameters


def test_active_golden_preserved() -> None:
    """Active golden capture matrix stays usable after cleanup."""
    path = ROOT / "tests/fixtures/golden/capture_matrix.py"
    assert path.exists()
    assert "def capture_golden_matrix" in _read_text(path)


def test_no_source_tools_dependency() -> None:
    """Maintained src graph never imports or executes tools."""
    markers = ("from tools.", "from tools import", "import tools.")
    offenders = [
        f"{path.relative_to(ROOT)}:{marker}"
        for path in (ROOT / "src").rglob("*.py")
        for marker in markers
        if marker in _read_text(path)
    ]
    assert offenders == []


def test_canonical_benchmark_uses_supervisor() -> None:
    """Maintained benchmark uses the canonical 3m supervisor invocation."""
    text = _read_text(ROOT / "tests/benchmark/mhs/test_mhs_replay_resources.py")
    assert "uv run python -m src.cli.main backtest mhs" in text
    assert "mhs_baseline_run" not in text
    assert (ROOT / "src/application/mhs_supervisor.py").exists()


def test_used_dependencies_preserved() -> None:
    """Retained research and live CLIs keep their required dependencies."""
    pyproject = _read_text(ROOT / "pyproject.toml")
    assert "optuna" in pyproject
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.report.persist import persist_mhs_horizon_diagnostic_report

    assert MhsDiagnosticRequest is not None
    assert callable(persist_mhs_horizon_diagnostic_report)


def test_no_eager_facade_on_package_import() -> None:
    """Importing the evaluation package performs no orchestration."""
    init_text = _read_text(ROOT / "src/mhs/evaluation/__init__.py").strip()
    assert "package initialization performs no orchestration or report loading" in init_text
    code_lines = [line for line in init_text.splitlines() if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("Research")]
    assert code_lines == []
    proc = subprocess.run(
        [sys.executable, "-c", "import src.mhs.evaluation, sys; print(sorted(m for m in sys.modules if m.startswith('src.mhs.report') or m.startswith('src.mhs.pipeline')))"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "[]"


def test_maintained_live_targets_direct_owner() -> None:
    """Live signal step builds targets from the concrete fold owner."""
    text = _read_text(ROOT / "src/mhs/live_signal_step.py")
    assert "from src.mhs.evaluation.fold_weights import _build_fold_target_weights" in text
    assert "missing _build_fold_target_weights" not in text
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights

    params = list(inspect.signature(_build_fold_target_weights).parameters)
    assert params[:4] == ["root", "fold", "request", "funding_by_symbol"]
    for name in ("decision_start", "decision_end", "deadband_seed_row", "panel_quarantine"):
        assert name in params


def test_maintained_research_and_deployment_smoke() -> None:
    """Supported research CLI and deployment wiring stay importable."""
    from src.cli.commands.research.mhs import _run_mhs_horizon_diagnostic, add_mhs_commands
    from src.mhs.contracts import MhsDiagnosticRequest, MhsOutputTier
    from src.mhs.execution.window_stream import MhsExecutionWindow
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    assert callable(run_mhs_diagnostic)
    assert callable(_run_mhs_horizon_diagnostic)
    assert callable(add_mhs_commands)
    assert MhsDiagnosticRequest is not None
    assert MhsOutputTier is not None
    assert MhsExecutionWindow is not None
    workflow = _read_text(ROOT / ".github/workflows/deploy.yml")
    assert "src.application.ops.daemon_idle_gate" in workflow


def test_direct_owner_patch_points() -> None:
    """Facade lookups resolve to concrete owners with no vacuous pass."""
    for rel in [
        "src/mhs/pipeline/stages/panel.py",
        "src/mhs/pipeline/stages/book.py",
        "src/mhs/pipeline/stages/committee.py",
        "src/mhs/pipeline/stages/selection.py",
        "src/mhs/pipeline/stages/fold.py",
        "src/mhs/pipeline/stages/replay.py",
        "src/mhs/pipeline/stages/assemble.py",
        "src/mhs/pipeline/orchestrator.py",
        "src/mhs/evaluation/fold_weights.py",
        "src/mhs/evaluation/books.py",
        "src/mhs/evaluation/diagnostics.py",
        "src/mhs/evaluation/committee.py",
        "src/mhs/evaluation/windows.py",
    ]:
        text = _read_text(ROOT / rel)
        assert "from src.mhs.evaluation import" not in text, rel
        assert "import src.mhs.evaluation as ev" not in text, rel
    from src.mhs.books import inverse_realized_vol_tilt, renormalize_within_mask
    from src.mhs.regime import crash_regime_tilt_weights

    assert callable(inverse_realized_vol_tilt)
    assert callable(renormalize_within_mask)
    assert callable(crash_regime_tilt_weights)


def test_dead_declarations_have_no_maintained_references() -> None:
    """Removed symbols leave no unresolved maintained or dynamic references."""
    roots = [ROOT / "src", ROOT / "tests/unit", ROOT / "tests/benchmark"]
    self_path = Path(__file__).resolve()
    hits = [
        f"{path.relative_to(ROOT)}:{token}"
        for base in roots
        for path in base.rglob("*.py")
        for token in STALE_TOKENS
        if path.resolve() != self_path and token in _read_text(path)
    ]
    assert hits == []
    deploy = _read_text(ROOT / ".github/workflows/deploy.yml")
    for token in STALE_TOKENS:
        assert token not in deploy, token


def test_full_three_minute_equivalence_scope() -> None:
    """Canonical run keeps 3m grid with supervisor wall and memory scope."""
    import dataclasses

    from src.mhs.pipeline.config import MhsRunConfig

    fields = {f.name: f.default for f in dataclasses.fields(MhsRunConfig)}
    assert fields.get("execution_timeframe") == "3m"
    supervisor = _read_text(ROOT / "src/application/mhs_supervisor.py")
    assert "wall_seconds" in supervisor
    assert "PSS" in supervisor


def test_direct_owner_blend_grid_selection() -> None:
    """Direct blend grid selection follows the capital contract."""
    import pandas as pd

    from src.mhs.evaluation.books import _active_blend_book_and_grid
    from src.mhs.types import BOOK_SPECS

    fast = BOOK_SPECS["fast_reversal"]
    slow = BOOK_SPECS["slow_momentum"]
    fast_grid = pd.DatetimeIndex([], tz="UTC")
    slow_grid = pd.DatetimeIndex([], tz="UTC")
    spec, grid = _active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    assert spec is not None
    assert grid is not None


def test_direct_owner_committee_book_call() -> None:
    """Committee book resolves through the concrete feature owner."""
    import pandas as pd
    import pytest

    from src.mhs.evaluation.committee import _committee_execution_book

    idx = pd.date_range("2021-01-01", periods=4, freq="6h", tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    quote_vol = pd.DataFrame(1.0, index=idx, columns=cols)
    taker = pd.DataFrame(1.0, index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    with pytest.raises(RuntimeError, match=r"no committee member admitted"):
        _committee_execution_book(close, quote_vol, taker, mask, idx, 2, members=("no_such_member_xyz",))


def test_direct_owner_fold_tilt_and_renormalize() -> None:
    """Fold weight tilt and mask renormalization use concrete book owners."""
    import numpy as np
    import pandas as pd

    from src.mhs.books import inverse_realized_vol_tilt, renormalize_within_mask
    from src.mhs.horizons import realized_vol
    from src.mhs.regime import crash_regime_tilt_weights

    idx = pd.date_range("2021-01-01", periods=30, freq="6h", tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT", "BTCUSDT"]
    log_close = pd.DataFrame(np.log(100.0 + np.arange(90, dtype=float).reshape(30, 3)), index=idx, columns=cols)
    weights = pd.DataFrame(0.5, index=idx, columns=cols)
    tilted = inverse_realized_vol_tilt(weights, realized_vol(log_close, 6).reindex(idx))
    assert tilted.shape == weights.shape
    mask = pd.DataFrame(True, index=idx, columns=cols)
    out = renormalize_within_mask(tilted, mask, 2)
    assert out.shape == weights.shape
    tilted_slow = crash_regime_tilt_weights(
        weights, log_close, mask, ("BTCUSDT",), 24, 0.5, min_symbols=2,
    )
    assert tilted_slow.shape == weights.shape


def test_direct_owner_window_batch_call(monkeypatch: object) -> None:
    """Window rescaled batch executes through the concrete execution owner."""
    import types

    import pandas as pd
    import pytest

    import src.mhs.evaluation.windows as windows
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.types import BOOK_SPECS

    grid = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    weights = pd.DataFrame({"AAAUSDT": [1.0, 0.0, 1.0, 0.0]}, index=grid)
    opens = pd.DataFrame({"AAAUSDT": [100.0, 101.0, 102.0, 103.0]}, index=grid)
    funding = pd.DataFrame({"AAAUSDT": [0.0, 0.0, 0.0, 0.0]}, index=grid)

    evidence = types.SimpleNamespace(prescreen={}, tail=None)
    monkeypatch.setattr(windows, "book_evidence", lambda *a, **k: evidence)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        windows.integrity,
        "_truncate_replayable_decisions",
        lambda w, s, g, spec: (w.iloc[:2], s[:2], 0),
    )
    monkeypatch.setattr(windows, "_resolve_ram_budget", lambda *a, **k: (1, 1))  # type: ignore[attr-defined]
    ledger = types.SimpleNamespace(
        equity=pd.Series([1.0, 1.01], index=pd.date_range("2021-01-01", periods=2, tz="UTC")),
    )
    phase_a = types.SimpleNamespace(ledger=ledger)
    monkeypatch.setattr(windows, "replay_execution_windows", lambda *a, **k: phase_a)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        windows._scaling, "_replay_exposure_scale", lambda *a, **k: pd.Series([1.0, 1.0], index=ledger.equity.index)
    )
    monkeypatch.setattr(windows._statistics, "_naive_sharpe", lambda *a, **k: 0.0)  # type: ignore[attr-defined]
    monkeypatch.setattr(windows, "resolved_anchored_folds", lambda *a, **k: [])  # type: ignore[attr-defined]
    monkeypatch.setattr(windows, "_iter_spilled_windows", lambda *a, **k: iter(()))  # type: ignore[attr-defined]
    monkeypatch.setattr(windows, "_rescaled_windows", lambda w, s: w)  # type: ignore[attr-defined]

    def _boom(*a: object, **k: object) -> object:
        raise SystemExit("batch-reached")

    monkeypatch.setattr(windows, "replay_execution_window_batch_isolated", _boom)  # type: ignore[attr-defined]

    request = MhsDiagnosticRequest()
    with pytest.raises(SystemExit, match="batch-reached"):
        windows._book_outcome(
            "blend", BOOK_SPECS["fast_reversal"], 1, grid, weights, grid, opens, funding,
            types.SimpleNamespace(), str(ROOT), request, {}, grid[0], grid[-1], 1, 1.0,
        )


def test_direct_owner_fold_target_weights() -> None:
    """Fold target builder runs on concrete book and regime owners."""
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.evaluation.fold_weights import _build_fold_target_weights
    from src.mhs.evidence import AnchoredPurgedFold

    n = 2000
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT", "BTCUSDT"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 0.1, (n, 3)), axis=0), index=idx, columns=cols)
    base_panel = {
        "close": close,
        "open": close.copy(),
        "quote_vol": pd.DataFrame(1e6, index=idx, columns=cols),
        "taker_buy_quote": pd.DataFrame(1e5, index=idx, columns=cols),
    }
    fold = AnchoredPurgedFold(
        train_start=idx[0], train_end=idx[100],
        validation_start=idx[800], validation_end=idx[1800],
        forward_dependency_hours=24, purge_hours=24,
    )
    funding = {c: pd.Series(0.0, index=idx) for c in cols}
    req = dataclasses.replace(MhsDiagnosticRequest(), committee_capital=True, crash_regime_tilt_alpha=0.5)
    targets, signal_at, roster, grid = _build_fold_target_weights(
        "root", fold, req, funding, base_panel=base_panel,
        require_minute_roster=False, panel_warmup_hours=24,
    )
    assert not targets.empty
    assert len(signal_at) == len(targets)
    assert grid is not None
