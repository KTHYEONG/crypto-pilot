"""Capture the five named MHS golden report fixtures.

Writes ``mhs_report_golden.json`` (baseline, left untouched if already present)
plus ``mhs_report_golden_{committee,discovery,trend_sleeve,fold_safe}.json`` into
``tests/fixtures/golden/``.

Each non-baseline capture uses
``tests/unit/application/research/mhs/test_evaluation.py::_write_mhs_market(
..., include_taker_buy_quote=True)`` -- empirically required so that
``committee_capital=True`` (and the fold-safe discovery scan) can load the
``taker_buy_quote`` column without raising ``ArrowInvalid`` at
``load_base_panel``.  Exactly one opt-in flag is flipped per golden; everything
else stays at its default, matching the empirically-verified feasibility run
(status=COMPLETE, blend.primary non-None, folds=4 for committee_capital=True).

Run with::

    uv run python tests/fixtures/golden/capture_matrix.py
"""

from __future__ import annotations

from pathlib import Path


import src.application.research.mhs.marks as marks
import src.market_data.services.futures_collection as fc
from src.application.research.mhs import statistics as _statistics
from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs.evaluation import run_mhs_horizon_diagnostic
from tests.unit.application.research.mhs.test_evaluation import (
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


def capture_golden_matrix(out_dir: Path) -> dict[str, Path]:
    """Generate and write all five named golden JSON files; return name->path."""
    import tempfile

    written: dict[str, Path] = {}
    for name, overrides in MATRIX:
        out_path = out_dir / f"mhs_report_golden_{name}.json"
        if name == "baseline" and out_path.exists():
            # The baseline golden is the canonical, previously-captured fixture;
            # never overwrite it (keeps the historical byte reference intact).
            written[name] = out_path
            continue
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
                    **overrides,
                )
                report = run_mhs_horizon_diagnostic(request)
            finally:
                marks.funding_path = orig_funding_path
                fc._mark_price_path = orig_mark_price_path
                _statistics._BOOTSTRAP_REPLICATES = orig_reps if "orig_reps" in locals() else orig_replicates
                _statistics._BOOTSTRAP_MEAN_BLOCK = orig_block
                _statistics._BOOTSTRAP_SEED = orig_seed
            payload = report.to_payload()
            with open(out_path, "w", encoding="utf-8") as f:
                import json

                json.dump(payload, f)
            written[name] = out_path
    return written


if __name__ == "__main__":
    result = capture_golden_matrix(GOLDEN_DIR)
    for name, path in result.items():
        print(f"captured {name}: {path}")  # noqa: T201
