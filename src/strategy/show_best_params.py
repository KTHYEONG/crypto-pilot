import os
import sys
import optuna
import sqlite3
from pathlib import Path

def get_project_root():
    """
    Get the project root directory.
    Assumes this script is located in src/strategy/
    """
    try:
        # Go up 2 levels from src/strategy/
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path(os.getcwd())

def show_study_details(storage_url, study_name, strategy_label):
    """
    Load a study and print its best parameters.
    """
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy_label}")
    print(f"Study:    {study_name}")
    print(f"{'='*70}")

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Error loading study '{study_name}': {e}")
        return

    if not study.trials:
        print("  - No trials found in this study.")
        return

    try:
        best_trial = study.best_trial
    except ValueError:
        print("  - No successful trials found (cannot determine best parameters).")
        return

    print(f"\n  [Best Value (Score)]: {best_trial.value}")
    print(f"  [Best Trial Number]:  {best_trial.number}")
    print(f"  [Date Completed]:     {best_trial.datetime_complete}")
    
    print("\n  [Best Parameters]:")
    print(f"  {'-'*30}")
    
    # Sort parameters alphabetically for better readability
    sorted_params = dict(sorted(best_trial.params.items()))
    for key, value in sorted_params.items():
        print(f"    {key:<25}: {value}")
    print(f"  {'-'*30}")

def inspect_database(db_path, specific_study_name=None):
    """
    Connect to the SQLite DB and inspect studies.
    """
    if not db_path.exists():
        print(f"\n[!] Database not found: {db_path}")
        return

    storage_url = f"sqlite:///{db_path}"
    print(f"\nConnecting to database: {db_path.name} ...")

    try:
        # Get all study summaries
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        
        if not summaries:
            print("  - No studies found in this database.")
            return

        found_specific = False
        for summary in summaries:
            if specific_study_name and summary.study_name == specific_study_name:
                show_study_details(storage_url, summary.study_name, f"Target Study ({summary.study_name})")
                found_specific = True
            elif not specific_study_name:
                # If no specific name requested, show all
                show_study_details(storage_url, summary.study_name, f"Found Study ({summary.study_name})")

        if specific_study_name and not found_specific:
            print(f"\n[!] Target study '{specific_study_name}' not found in {db_path.name}.")
            print("    Available studies:")
            for s in summaries:
                print(f"     - {s.study_name}")

    except Exception as e:
        print(f"Error accessing database {db_path.name}: {e}")

def main():
    root_dir = get_project_root()
    
    # 1. Futures Strategy Database
    futures_db = root_dir / "futures_strategy.db"
    # Based on verify_futures.py, the deploy function uses 'futures_strategy' as study name
    inspect_database(futures_db, specific_study_name="futures_strategy")

    # 2. Spot Strategy Database
    spot_db = root_dir / "spot_strategy.db"
    # Based on verify_spot.py patterns, assuming 'spot_strategy' or we list all
    # We will try to find 'spot_strategy' essentially, or all if we are unsure.
    # Let's list all for spot to be safe, as naming might vary (e.g. 'spot_strategy_v1')
    inspect_database(spot_db)

if __name__ == "__main__":
    main()
