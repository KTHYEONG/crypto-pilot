#!/usr/bin/env python3
"""Timeframe Alpha Probe CLI runner."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def main() -> None:
    """Run timeframe alpha probe and persist manifest."""
    parser = argparse.ArgumentParser(description="Timeframe Alpha Probe")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to probe")
    parser.add_argument(
        "--tf-grid",
        nargs="+",
        default=["15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h"],
        help="Timeframe grid to evaluate",
    )
    parser.add_argument("--max-workers", type=int, default=12, help="Max parallel workers")
    parser.add_argument(
        "--output",
        default="docs/results/tf_probe_manifest.parquet",
        help="Output parquet path for manifest",
    )
    parser.add_argument(
        "--tf",
        default="4h",
        help="Base timeframe for data loading",
    )
    parser.add_argument(
        "--fetch-start",
        default="2022-01-01",
        help="Data fetch start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start",
        default="2022-06-01",
        help="IS start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--is-end",
        default="2024-06-01",
        help="IS end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2025-01-01",
        help="OOS end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--round-trip-cost-bps",
        type=float,
        default=6.0,
        help="Round-trip transaction cost in bps",
    )
    parser.add_argument(
        "--fdr-q",
        type=float,
        default=0.10,
        help="BH-FDR target false-discovery rate",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    try:
        from src.domain.futures.optimization.opt_data_utils import (
            load_futures_data_maps_for_symbols,
        )
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.timeframe_probe import (
            probe_timeframe_alpha,
            select_tf_family_cells,
        )
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Symbol resolution (--symbols required; no auto-discovery in probe mode)
    # ---------------------------------------------------------------------------
    if not args.symbols:
        _logger.error("--symbols is required. Pass at least one symbol (e.g. BTCUSDT).")
        sys.exit(1)
    symbols = args.symbols

    # ---------------------------------------------------------------------------
    # Data loading (IS only for probe)
    # ---------------------------------------------------------------------------
    _logger.info(
        "Loading data for %d symbols [%s → %s]", len(symbols), args.fetch_start, args.is_end
    )
    try:
        data_maps, _oos_maps, valid_symbols = load_futures_data_maps_for_symbols(
            symbols=symbols,
            tf=args.tf,
            fetch_start=args.fetch_start,
            start=args.start,
            is_end=args.is_end,
            end=args.end,
            skip_metrics=True,
            scope_name="tf_probe",
        )
    except Exception as exc:
        _logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    if not valid_symbols:
        _logger.error("No valid symbols after data loading.")
        sys.exit(1)

    _logger.info("Valid symbols after loading: %d", len(valid_symbols))

    # ---------------------------------------------------------------------------
    # Base config
    # ---------------------------------------------------------------------------
    base_cfg = CandidateStrategyConfig(timeframe=args.tf)

    # ---------------------------------------------------------------------------
    # Run probe
    # ---------------------------------------------------------------------------
    _logger.info("Starting timeframe probe over tf_grid=%s", args.tf_grid)
    try:
        manifest = probe_timeframe_alpha(
            data_maps=data_maps,
            symbols=valid_symbols,
            base_cfg=base_cfg,
            tf_grid=args.tf_grid,
            round_trip_cost_bps=args.round_trip_cost_bps,
            fdr_q=args.fdr_q,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        _logger.error("probe_timeframe_alpha failed: %s", exc)
        sys.exit(1)

    _logger.info(
        "Probe complete: %d cells, %d tfs",
        len(manifest.cells),
        len(manifest.tf_grid),
    )

    # ---------------------------------------------------------------------------
    # Persist manifest as parquet
    # ---------------------------------------------------------------------------
    import dataclasses

    import pandas as pd

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest.cells:
        rows = [dataclasses.asdict(c) for c in manifest.cells]
        df_out = pd.DataFrame(rows)
        df_out.to_parquet(output_path, index=False)
        _logger.info("Manifest saved: %s (%d rows)", output_path, len(df_out))

        # Summary log
        promoted = select_tf_family_cells(manifest)
        _logger.info(
            "Promotable cells (tstat>=2, FDR, net_edge>=0, fold_consistency>=0.75): %d",
            len(promoted),
        )
        for cell in promoted[:20]:
            _logger.info(
                "  %s | %s:%s | tf=%s | ic_tstat=%.2f | net_edge=%.1fbps | hurst=%.2f",
                cell.symbol,
                cell.family,
                cell.variant,
                cell.tf,
                cell.ic_tstat_hac,
                cell.net_edge_bps,
                cell.hurst,
            )
    else:
        _logger.warning("No cells produced by probe — check data and config.")

    _logger.info("TF probe complete. See %s", args.output)


if __name__ == "__main__":
    main()
