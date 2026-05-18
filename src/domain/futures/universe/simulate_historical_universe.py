import logging
import argparse
from datetime import date, datetime
from pathlib import Path

# Project Root Setup
project_root = Path(__file__).resolve().parents[4]
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.domain.futures.universe.pipeline import build_universe
from src.domain.futures.universe.sync_utils import run_historical_sync

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniverseSimulator")

def get_simulation_dates():
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
    return date(2021, 1, 1), end_date

def run_quarterly_simulation(start_date: date, end_date: date):
    """Phase B: Simulate quarterly universe evolution."""
    logger.info("Starting Quarterly Simulation...")
    quarterly_dates = []
    current = start_date
    while current <= end_date:
        quarterly_dates.append(current)
        if current.month > 9:
            current = current.replace(year=current.year + 1, month=(current.month + 3) % 12 or 12)
        else:
            current = current.replace(month=current.month + 3)
            
    results_dir = Path("logs/futures/universe/snapshots")
    results_dir.mkdir(parents=True, exist_ok=True)
    previous_selected = set()
    evolution_data = []
    
    for as_of in quarterly_dates:
        logger.info(f"Building universe as_of {as_of}")
        try:
            snapshot, selected_df, _ = build_universe(
                as_of=as_of,
                tf="4h",
                ledger_path=Path("data/futures/universe_ledger.parquet"),
                snapshot_root=results_dir
            )
            current_selected = set(selected_df['symbol'].tolist()) if not selected_df.empty else set()
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

def generate_report(evolution_data):
    """Generate the markdown report in logs/futures/universe/."""
    log_dir = Path("logs/futures/universe")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "universe_evolution_report.md"
    with open(report_path, "w") as f:
        f.write("# Binance Futures Universe Evolution Report (Real Data)\n\n")
        f.write(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis Window:** {evolution_data[0]['date']} to {evolution_data[-1]['date']}\n\n")
        f.write("| Quarter | Active Symbols | New Entries | Dropouts |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for entry in evolution_data:
            new_text = ", ".join(entry['new_entries'][:5]) + ("..." if len(entry['new_entries']) > 5 else "")
            drop_text = ", ".join(entry['dropouts'][:5]) + ("..." if len(entry['dropouts']) > 5 else "")
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
