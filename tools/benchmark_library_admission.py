"""Sealed library-admission execution benchmark.

Runs a representative sealed fixture three times at one worker and at every
effective process-worker count, then emits median wall seconds, peak RSS bytes,
and the assembled panel bytes as deterministic JSON. The tool never mutates the
ledger, registry, or catalog, and it never asserts an absolute wall-time
threshold: the operator uses the recorded scaling to set the operational
``HARDWARE_MAX_WORKERS`` cap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import time

import numpy as np
from src.application.research.expert.admission import (
    _assemble_panel,
    _materialize_definitions,
    _run_symbol_tasks,
    run_technical_library_admission,
)
from src.research.expert_portfolio.admission_types import (
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionRequest,
)
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.provenance.code_manifest import TECHNICAL_CODE_UNITS, compute_code_hash

from src.core.settings import HARDWARE_MAX_WORKERS, effective_worker_count
from src.research.evaluation.policy import resolve_evaluation_end

_ITERATIONS = 3


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in kilobytes; scaled to bytes for the report.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _measure_panel_bytes(request: TechnicalLibraryAdmissionRequest) -> int:
    end = resolve_evaluation_end(request.end, unseal_holdout=False)
    definitions = _materialize_definitions(
        request, compute_code_hash(TECHNICAL_CODE_UNITS),
    )
    sources = tuple(sorted(request.candidate_sources))
    evidence = _run_symbol_tasks(request, end, 1, sources)
    panel, _ = _assemble_panel(evidence, definitions, request.symbols)
    return int(panel.memory_usage(deep=True).sum())


def _benchmark_request(args: argparse.Namespace) -> TechnicalLibraryAdmissionRequest:
    sources = tuple(sorted(set(args.candidate_source or [
        "technical_macd_histogram_regime_long_v1",
    ])))
    symbols = tuple(sorted(set(args.symbols or ["BTCUSDT"])))
    return TechnicalLibraryAdmissionRequest(
        candidate_sources=sources,
        symbols=symbols,
        router=ContextualRouterSpec(
            context_symbol=symbols[0],
            trend_lookback_bars=60,
            volatility_lookback_bars=20,
            min_context_history_bars=30,
        ),
        admission=LibraryAdmissionConfig(
            min_experts=2,
            max_experts=4,
            min_closed_trades=1,
            min_active_return_bars=1,
            max_abs_pairwise_log_return_correlation=0.8,
            max_joint_negative_return_rate=0.5,
            min_context_covered_states=1,
            max_combinations=500,
            max_workers=args.max_workers,
        ),
        start=args.start,
        end=args.end,
    )


def main() -> None:
    """Benchmark sequential vs process scaling and emit a JSON measurement record."""
    parser = argparse.ArgumentParser(
        prog="benchmark_library_admission",
        description="Measure library-admission sequential/process scaling",
    )
    parser.add_argument("--candidate-source", action="append", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    request = _benchmark_request(args)
    effective = effective_worker_count(
        len(request.symbols), requested=args.max_workers,
        hardware_cap=HARDWARE_MAX_WORKERS,
    )
    worker_counts = sorted({1, *range(2, effective + 1)})
    panel_bytes = _measure_panel_bytes(request)

    records: list[dict[str, object]] = []
    for workers in worker_counts:
        run_request = dataclasses.replace(
            request,
            admission=dataclasses.replace(request.admission, max_workers=workers),
        )
        wall_times: list[float] = []
        rss_values: list[int] = []
        for _ in range(_ITERATIONS):
            started = time.perf_counter()
            run_technical_library_admission(run_request)
            wall_times.append(time.perf_counter() - started)
            rss_values.append(_peak_rss_bytes())
        records.append({
            "effective_workers": workers,
            "backend": "process",
            "median_wall_seconds": float(np.median(wall_times)),
            "peak_rss_bytes": int(np.max(rss_values)),
            "panel_bytes": panel_bytes,
        })

    print(
        json.dumps(
            {"backend": "process", "records": records},
            sort_keys=True, indent=2,
        )
    )


assert main.__name__ == "main"


if __name__ == "__main__":
    main()
