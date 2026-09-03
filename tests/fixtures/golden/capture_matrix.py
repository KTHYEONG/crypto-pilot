"""Capture the five named MHS golden report fixtures (digest + summary).

Writes ``mhs_report_golden_{name}_digest.json`` and
``mhs_report_golden_{name}_summary.json`` for each matrix name into
``tests/fixtures/golden/``. The former monolithic full-payload golden measured
1085.61 MB -- over GitHub's 100 MB per-file limit -- and its deletion left all
identity tests skipping; the sha256 digest (~300 KB) plus the row-count
summary (<100 KB) restore the gate at >= 2700x smaller size with no loss of
strength (sha256 over float64 bytes is bit-equality by construction).

Each capture uses
``tests/unit/application/research/mhs/test_evaluation.py::_write_mhs_market(
..., include_taker_buy_quote=True)`` -- empirically required so that
``committee_capital=True`` (and the fold-safe discovery scan) can load the
``taker_buy_quote`` column without raising ``ArrowInvalid`` at
``load_base_panel``.  Exactly one opt-in flag is flipped per golden; everything
else stays at its default.

Run with::

    uv run python tests/fixtures/golden/capture_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


import src.mhs.marks as marks
import src.market_data.services.futures_collection as fc
from src.mhs import statistics as _statistics
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
from tests.fixtures.golden.compare import GOLDEN_MATRIX_NAMES, assert_report_digest_identical
from tests.fixtures.golden.digest import build_report_digest, build_report_summary
from tests.unit.mhs.test_evaluation_appresearch import (
    _START,
    _write_mhs_market,
)

GOLDEN_DIR = Path(__file__).resolve().parent

# (name, kwargs-flipped-on-MhsDiagnosticRequest) -- exactly one opt-in per golden.
MATRIX: tuple[tuple[str, dict[str, object]], ...] = (
    ("baseline", {}),
    ("committee", {"committee_capital": True}),
    ("discovery", {"discovery_gate": True}),
    ("trend_sleeve", {"trend_sleeve": True, "trend_sleeve_gross": 0.15}),
    ("fold_safe", {"fold_safe_horizon_selection": True}),
)

assert tuple(name for name, _ in MATRIX) == GOLDEN_MATRIX_NAMES


def golden_digest_path(name: str) -> Path:
    return GOLDEN_DIR / f"mhs_report_golden_{name}_digest.json"


def golden_summary_path(name: str) -> Path:
    return GOLDEN_DIR / f"mhs_report_golden_{name}_summary.json"


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def capture_golden_matrix(out_dir: Path) -> dict[str, Path]:
    """Generate and write all named golden digest+summary files; return paths."""
    import tempfile

    written: dict[str, Path] = {}
    for name, overrides in MATRIX:
        out_path = golden_digest_path(name)
        summary_path = golden_summary_path(name)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Non-baseline goldens require the taker_buy_quote column so the
            # committee / fold-safe code paths can load the panel.
            end = _write_mhs_market(root, n_hours=2700, include_taker_buy_quote=(name != "baseline"))
            # Redirect the funding/mark-price resolvers to the synthetic root;
            # without this, _load_funding_series reads the real production data
            # directory and "no dev symbol has funding coverage" is raised.
            orig_funding_path = marks.funding_path
            orig_mark_price_path = fc._mark_price_path
            orig_replicates = _statistics._BOOTSTRAP_REPLICATES
            orig_block = _statistics._BOOTSTRAP_MEAN_BLOCK
            orig_seed = _statistics._BOOTSTRAP_SEED
            marks.funding_path = lambda sym, _root=root: _root / "funding" / f"{sym}.parquet"
            fc._mark_price_path = (
                lambda symbol, timeframe, _root=root: _root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
            )
            _statistics._BOOTSTRAP_REPLICATES = 20
            _statistics._BOOTSTRAP_MEAN_BLOCK = 24
            _statistics._BOOTSTRAP_SEED = 20260807
            try:
                request = MhsDiagnosticRequest(
                    start=str(_START),
                    end=str(end),
                    data_root=str(root),
                    execution_timeframe="1m",
                    log_run=False,
                    **overrides,  # type: ignore[arg-type]
                )
                report = run_mhs_horizon_diagnostic(request)
                # Self-check the gate before writing: the fresh report must be
                # identical to its own freshly built digest.
                assert_report_digest_identical(build_report_digest(report), report)
                digest = build_report_digest(report)
                summary = build_report_summary(report)
            finally:
                marks.funding_path = orig_funding_path
                fc._mark_price_path = orig_mark_price_path
                _statistics._BOOTSTRAP_REPLICATES = orig_replicates
                _statistics._BOOTSTRAP_MEAN_BLOCK = orig_block
                _statistics._BOOTSTRAP_SEED = orig_seed
            _write_json(out_path, digest)
            _write_json(summary_path, summary)
            written[name] = out_path
    return written


if __name__ == "__main__":
    result = capture_golden_matrix(GOLDEN_DIR)
    for name, path in result.items():
        sys.stdout.write(f"captured {name}: {path}\n")
