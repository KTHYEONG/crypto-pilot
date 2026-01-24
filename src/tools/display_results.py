import optuna
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config.settings import SPOT_STRATEGY_DB, FUTURES_STRATEGY_DB

def display_study_results(db_path, title):
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print(f"📂 Path: {db_path}")
    print(f"{'='*80}")

    if not os.path.exists(db_path):
        print(f"❌ Database file not found at {db_path}")
        return

    storage_url = f"sqlite:///{db_path}"
    
    try:
        study_summaries = optuna.get_all_study_summaries(storage=storage_url)
        if not study_summaries:
            print("❓ No studies found in this database.")
            return

        for summary in study_summaries:
            study_name = summary.study_name
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            
            try:
                best_trial = study.best_trial
                print(f"\n🔹 Study Name: {study_name}")
                print(f"   🏆 Best Score: {best_trial.value:.4f}")
                print(f"   🕒 Best Trial Number: {best_trial.number}")
                print(f"   📅 Trial Finished: {best_trial.datetime_complete}")
                
                print("\n   ✨ Best Parameters:")
                for param_name, param_value in best_trial.params.items():
                    print(f"     - {param_name:<25}: {param_value}")

                if best_trial.user_attrs:
                    print("\n   📊 Performance Metrics (User Attrs):")
                    for attr_name, attr_value in best_trial.user_attrs.items():
                        # Round if float
                        if isinstance(attr_value, float):
                            print(f"     - {attr_name:<25}: {attr_value:.4f}")
                        else:
                            print(f"     - {attr_name:<25}: {attr_value}")
                
                print(f"\n   - Total Trials: {len(study.trials)}")
                print(f"   {'-'*40}")

            except ValueError:
                print(f"\n🔹 Study Name: {study_name}")
                print("   ⚠️ No completed trials found in this study.")
            except Exception as e:
                print(f"\n🔹 Study Name: {study_name}")
                print(f"   ❌ Error loading best trial: {e}")

    except Exception as e:
        print(f"❌ Error accessing database: {e}")

if __name__ == "__main__":
    # Check Spot Strategy Results
    display_study_results(SPOT_STRATEGY_DB, "SPOT STRATEGY OPTIMIZATION RESULTS")
    
    # Check Futures Strategy Results
    display_study_results(FUTURES_STRATEGY_DB, "FUTURES STRATEGY OPTIMIZATION RESULTS")
    
    print(f"\n{'='*80}")
    print(f"✅ Report Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
