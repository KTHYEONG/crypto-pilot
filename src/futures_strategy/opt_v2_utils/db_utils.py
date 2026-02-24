"""
최적화 도구인 Optuna의 학습 결과(Study)를 원격 DB(MySQL)에서 로컬 DB(SQLite)로 복사하고 관리하는 기능을 담당함.
최적화된 파라미터를 실제 트레이딩 시스템에서 사용할 수 있도록 데이터베이스를 실시간으로 동기화함.
"""
import os
import optuna
import logging

_logger: logging.Logger = logging.getLogger("opt_v2")

def save_study_to_sqlite(study: optuna.Study, project_root: str) -> bool:
    """
    Export ONLY the best trial from the current study to local SQLite for production.
    This avoids the bottleneck of copying thousands of trials.
    """
    study_name: str = study.study_name
    sqlite_path: str = os.path.join(project_root, "futures_strategy.db")
    sqlite_storage_url: str = f"sqlite:///{sqlite_path}"
    
    _logger.info(f"  💾 Exporting BEST trial of '{study_name}' to local SQLite...")
    
    try:
        # 1. Delete existing local study to ensure fresh best trial
        try:
            optuna.delete_study(study_name=study_name, storage=sqlite_storage_url)
        except (KeyError, Exception):
            pass
            
        # 2. Create new local study
        local_study: optuna.Study = optuna.create_study(
            study_name=study_name,
            storage=sqlite_storage_url,
            direction=study.direction,
            load_if_exists=False
        )
        
        # 3. Add only the best trial
        local_study.add_trial(study.best_trial)
        
        _logger.info("✅ SQLite persistence (Best Trial Only) complete.")
        return True
        
    except Exception as e:
        _logger.error(f"❌ Failed to persist best trial to SQLite: {e}")
        return False
