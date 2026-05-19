import argparse
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Project Root Setup
import sys  # noqa: E402

project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.domain.futures.universe.pipeline import build_universe  # noqa: E402
from src.domain.futures.universe.sync_utils import run_historical_sync  # noqa: E402

_LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
logger = logging.getLogger("UniverseSimulator")

def get_simulation_dates() -> tuple[date, date]:
    """Dynamically determine start and end dates for the simulation."""
    today = date.today()
    current_month = today.month
    if 1 <= current_month <= 3:
        end_date = date(today.year, 1, 1)
    elif 4 <= current_month <= 6:
        end_date = date(today.year, 4, 1)
    elif 7 <= current_month <= 9:
        end_date = date(today.year, 7, 1)
    else:
        end_date = date(today.year, 10, 1)
    return date(2020, 4, 1), end_date

def run_quarterly_simulation(
    start_date: date, end_date: date
) -> list[dict[str, Any]]:
    """Phase B: Simulate quarterly universe evolution."""
    logger.info("Starting Quarterly Simulation...")
    quarterly_dates: list[date] = []
    current = start_date
    while current <= end_date:
        quarterly_dates.append(current)
        if current.month > 9:
            current = current.replace(year=current.year + 1, month=(current.month + 3) % 12 or 12)
        else:
            current = current.replace(month=current.month + 3)

    results_dir = Path("logs/futures/universe/snapshots")
    results_dir.mkdir(parents=True, exist_ok=True)
    previous_selected: set[str] = set()
    evolution_data: list[dict[str, Any]] = []

    for as_of in quarterly_dates:
        logger.info(f"Building universe as_of {as_of}")
        try:
            snapshot, selected_df, _ = build_universe(
                as_of=as_of,
                tf="4h",
                ledger_path=Path("data/futures/universe_ledger.parquet"),
                snapshot_root=results_dir
            )
            current_selected = (
                set(selected_df['symbol'].tolist()) if not selected_df.empty else set()
            )
            new_entries = current_selected - previous_selected
            dropouts = previous_selected - current_selected
            evolution_data.append({
                "date": as_of.isoformat(),
                "total": len(current_selected),
                "new_entries": sorted(list(new_entries)),
                "dropouts": sorted(list(dropouts))
            })
            previous_selected = current_selected
        except Exception as e:
            logger.error(f"Simulation failed for {as_of}: {e}")
    return evolution_data

def generate_report(evolution_data: list[dict[str, Any]]) -> None:
    """Generate the markdown report in logs/futures/universe/."""
    log_dir = Path("logs/futures/universe")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "universe_evolution_report.md"
    with open(report_path, "w") as f:
        f.write("# Binance Futures Universe Evolution Report (Real Data)\n\n")
        f.write(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        window_str = f"{evolution_data[0]['date']} to {evolution_data[-1]['date']}"
        f.write(f"**Analysis Window:** {window_str}\n\n")
        f.write("| Quarter | Active Symbols | New Entries | Dropouts |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for entry in evolution_data:
            if entry['new_entries']:
                suffix = "..." if len(entry['new_entries']) > 5 else ""
                new_text = ", ".join(entry['new_entries'][:5]) + suffix
            else:
                new_text = "(no change)" if entry['total'] > 0 else ""
            if entry['dropouts']:
                suffix = "..." if len(entry['dropouts']) > 5 else ""
                drop_text = ", ".join(entry['dropouts'][:5]) + suffix
            else:
                drop_text = "(no change)" if entry['total'] > 0 else ""
            f.write(f"| {entry['date']} | {entry['total']} | {new_text} | {drop_text} |\n")
    logger.info(f"✅ Report successfully generated at: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    start_date, end_date = get_simulation_dates()
    if not args.skip_data:
        run_historical_sync(start_date, end_date, limit=args.limit, force=args.force)

    evolution = run_quarterly_simulation(start_date, end_date)
    if evolution:
        generate_report(evolution)
