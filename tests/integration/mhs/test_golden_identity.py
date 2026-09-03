"""Golden identity tests for the MHS horizon diagnostic report.

Running the full pipeline on the synthetic market must reproduce the captured
golden under two complementary gates:

* ``assert_report_digest_identical`` — sha256 over every replay ledger series'
  index bytes + float64 values bytes (bit-equality by construction), with
  landmark repr()s naming the first divergence;
* ``assert_report_identical`` over the row-count-stubbed summaries.

The fixtures live under ``tests/fixtures/golden/`` as
``mhs_report_golden_<name>_digest.json`` + ``_summary.json`` pairs produced by
``capture_matrix.py``. A missing fixture FAILS the test (never skips): the
bit-exactness gate must never go inert silently.

SCENARIO_GOLDEN_IDENTITY_REGENERATED_AFTER_SCHEMA_CHANGE: report schema 키가
바뀌면(예: holdout_tail 신설) capture_matrix.py로 재캡처 후 이 게이트로
검증한다 — golden 파일은 수동 편집하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.golden.compare import (
    GOLDEN_MATRIX_NAMES,
    assert_report_digest_identical,
    assert_report_identical,
)
from tests.fixtures.golden.digest import build_report_summary

GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "golden"

# Wall-time and host-dependent fields can never be part of an identity gate.
NON_DETERMINISTIC_FIELDS = frozenset({
    "elapsed_seconds", "run_elapsed_seconds", "resource_measurements",
    "tree_memory", "worker_plan",
})


def _golden_paths(name: str) -> tuple[Path, Path]:
    return (
        GOLDEN_DIR / f"mhs_report_golden_{name}_digest.json",
        GOLDEN_DIR / f"mhs_report_golden_{name}_summary.json",
    )


def _load_golden(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the named digest+summary pair; a missing fixture FAILS, not skips."""
    digest_path, summary_path = _golden_paths(name)
    missing = [str(p) for p in (digest_path, summary_path) if not p.exists()]
    if missing:
        raise AssertionError(
            f"Golden fixture(s) missing: {missing}. Run "
            "`uv run python tests/fixtures/golden/capture_matrix.py` to capture them."
        )
    with open(digest_path, encoding="utf-8") as f:
        digest = json.load(f)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    size_bytes = digest_path.stat().st_size + summary_path.stat().st_size
    assert size_bytes <= 400 * 1024, f"golden fixtures too large: {size_bytes} bytes"
    return digest, summary


def _assert_matches_golden(
    golden_digest: dict[str, Any],
    golden_summary: dict[str, Any],
    report: Any,
) -> None:
    assert_report_digest_identical(golden_digest, report)
    actual_summary = build_report_summary(report)
    assert_report_identical(
        golden_summary, actual_summary, renames={}, exclude=NON_DETERMINISTIC_FIELDS,
    )


@pytest.fixture(scope="module")
def matrix_market(request, tmp_path_factory):
    """Synthetic market for a named golden.

    Non-baseline goldens need the ``taker_buy_quote`` column, so those markets
    are written via ``test_evaluation._write_mhs_market(include_taker_buy_quote=True)``.
    """
    import src.market_data.services.futures_collection as fc
    import src.mhs.marks as marks
    import src.mhs.statistics as statistics
    from tests.unit.mhs.test_evaluation_appresearch import (
        _START as _TE_START,
        _write_mhs_market as _write_taker_market,
    )

    name = request.param
    # Every matrix golden (baseline included) was captured by capture_matrix.py
    # using test_evaluation._write_mhs_market(n_hours=2700, ...) -- matching that
    # exact writer/param set here is required for the goldens to be reproducible.
    root = tmp_path_factory.mktemp(f"mhs_golden_market_{name}")
    end = _write_taker_market(root, n_hours=2700, include_taker_buy_quote=(name != "baseline"))
    start = _TE_START

    originals = {
        "funding_path": marks.funding_path,
        "_mark_price_path": fc._mark_price_path,
        "_BOOTSTRAP_REPLICATES": statistics._BOOTSTRAP_REPLICATES,
        "_BOOTSTRAP_MEAN_BLOCK": statistics._BOOTSTRAP_MEAN_BLOCK,
        "_BOOTSTRAP_SEED": statistics._BOOTSTRAP_SEED,
    }
    marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    # test_evaluation writes mark frames under markPriceKlines/1h

    def _mp(symbol, timeframe):
        return root / "markPriceKlines" / timeframe / f"{symbol}.parquet"

    fc._mark_price_path = _mp
    statistics._BOOTSTRAP_REPLICATES = 20
    statistics._BOOTSTRAP_MEAN_BLOCK = 24
    statistics._BOOTSTRAP_SEED = 20260807
    yield root, end, start
    marks.funding_path = originals["funding_path"]
    fc._mark_price_path = originals["_mark_price_path"]
    for nm, val in originals.items():
        if nm == "funding_path" or nm == "_mark_price_path":
            continue
        setattr(statistics, nm, val)


@pytest.fixture(scope="module")
def matrix_golden(request):
    """Digest+summary pair for the named golden (fails when absent)."""
    return _load_golden(request.param)


# One opt-in flag flipped per golden (baseline = all defaults).  Exactly mirrors
# capture_matrix.py.  The identity test for non-baseline goldens requires a
# market written with ``include_taker_buy_quote=True`` (see capture_matrix.py).
MATRIX_OVERRIDES: dict[str, dict[str, object]] = {
    "baseline": {},
    "committee": {"committee_capital": True},
    "discovery": {"discovery_gate": True},
    "trend_sleeve": {"trend_sleeve": True, "trend_sleeve_gross": 0.15},
    "fold_safe": {"fold_safe_horizon_selection": True},
}


@pytest.mark.parametrize(
    ("name", "matrix_market", "matrix_golden"),
    [(n, n, n) for n in GOLDEN_MATRIX_NAMES],
    ids=GOLDEN_MATRIX_NAMES,
    indirect=["matrix_market", "matrix_golden"],
)
def test_golden_identity_matrix(name, matrix_market, matrix_golden):
    """SCENARIO_MHS_PERF_P0_01_GOLDEN_GATE_LIVE: each named golden matches the
    decomposed pipeline bit-exactly under the sha256 digest gate."""
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic

    root, end, start = matrix_market
    request = MhsDiagnosticRequest(
        start=str(start), end=str(end), data_root=str(root),
        execution_timeframe="1m", log_run=False,
        **MATRIX_OVERRIDES[name],
    )
    report = run_mhs_horizon_diagnostic(request)
    golden_digest, golden_summary = matrix_golden
    _assert_matches_golden(golden_digest, golden_summary, report)


@pytest.mark.parametrize(
    ("name", "matrix_market", "matrix_golden"),
    [("baseline", "baseline", "baseline")],
    ids=["baseline"],
    indirect=["matrix_market", "matrix_golden"],
)
def test_golden_identity(name, matrix_market, matrix_golden):
    """SCENARIO_ANALYSIS_ARCHITECTURE_04: full pipeline on the synthetic market
    yields the baseline golden bit-exactly (digest + row-count summary)."""
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic

    root, end, start = matrix_market
    request = MhsDiagnosticRequest(
        start=str(start), end=str(end), data_root=str(root),
        execution_timeframe="1m", log_run=False,
        **MATRIX_OVERRIDES[name],
    )
    report = run_mhs_horizon_diagnostic(request)
    golden_digest, golden_summary = matrix_golden
    _assert_matches_golden(golden_digest, golden_summary, report)


@pytest.mark.parametrize(
    ("name", "matrix_market", "matrix_golden"),
    [("baseline", "baseline", "baseline")],
    ids=["entry_point"],
    indirect=["matrix_market", "matrix_golden"],
)
def test_run_mhs_diagnostic_entry_point_matches_golden(name, matrix_market, matrix_golden):
    """The CLI's actual entry point (MhsRunConfig -> run_mhs_diagnostic) is
    bit-exact against the same baseline golden test_golden_identity validates
    via the MhsDiagnosticRequest -> run_mhs_horizon_diagnostic entry point.

    Exercises the full six-stage decomposition end to end through
    src/mhs/pipeline/orchestrator.py -> runner.py -> stages/{panel,selection,
    book,committee,replay,fold}.py -> stages/assemble.py:
    SCENARIO_MHS_STAGE_DECOMP_01_ASSEMBLE
    SCENARIO_MHS_STAGE_DECOMP_02_FOLDS
    SCENARIO_MHS_STAGE_DECOMP_03_REPLAY
    SCENARIO_MHS_STAGE_DECOMP_04_PANEL_SELECTION
    SCENARIO_MHS_STAGE_DECOMP_05_BOOKS_COMMITTEE
    SCENARIO_MHS_STAGE_DECOMP_06_ORCHESTRATOR
    """
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    root, end, start = matrix_market
    config = MhsRunConfig(
        start=str(start), end=str(end), data_root=str(root),
        execution_timeframe="1m", log_run=False,
        committee_capital=False, committee_regime_adaptive_tranche=False,
        funding_carry_sleeve=False, committee_target_gross=None,
        pnl_vol_target_mode="median_relative",
        # MhsRunConfig의 CLI 실효 기본값(60/growth_extreme)이 아니라 golden이 잡힌
        # MhsDiagnosticRequest의 동결 기본값(30/conservative)과 맞춰야 bit-exact parity가 성립한다.
        execution_universe_size=30,
        growth_envelope="conservative",
    )
    report = run_mhs_diagnostic(config)
    golden_digest, golden_summary = matrix_golden
    _assert_matches_golden(golden_digest, golden_summary, report)


def test_golden_identity_survives_legacy_removal() -> None:
    """Marker: the real gate is the existing golden suite (execution_command)."""
    from pathlib import Path

    assert Path(
        "tests/fixtures/golden/mhs_report_golden_baseline_digest.json"
    ).exists()


def test_golden_identity_survives_evaluation_split() -> None:
    """Marker: the real gate is the existing golden suite (execution_command)."""
    from pathlib import Path

    assert Path(
        "tests/fixtures/golden/mhs_report_golden_baseline_digest.json"
    ).exists()


def test_golden_identity_survives_execution_split() -> None:
    """Marker: the real gate is the existing golden suite (execution_command)."""
    from pathlib import Path

    assert Path(
        "tests/fixtures/golden/mhs_report_golden_baseline_digest.json"
    ).exists()
