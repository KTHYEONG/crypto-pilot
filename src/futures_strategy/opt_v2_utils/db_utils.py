"""
최적화 도구인 Optuna의 학습 결과(Study)를 원격 DB(MySQL)에서 로컬 DB(SQLite)로 복사하고 관리하는 기능을 담당함.
최적화된 파라미터를 실제 트레이딩 시스템에서 사용할 수 있도록 데이터베이스를 실시간으로 동기화함.
"""
import os
import optuna
import logging

_logger: logging.Logger = logging.getLogger("opt_v2")

def save_study_to_sqlite(study_name: str, source_url: str, project_root: str) -> bool:
    """
    Export optimized Optuna study from MySQL/Remote to local SQLite for production.
    Returns True if successful, False otherwise.
    """
    sqlite_path: str = os.path.join(project_root, "futures_strategy.db")
    sqlite_storage_url: str = f"sqlite:///{sqlite_path}"
    
    _logger.info(f"  💾 Exporting optimized study '{study_name}' to local SQLite...")
    _logger.info(f"     Path: {sqlite_path}")
    
    try:
        try:
            optuna.delete_study(study_name=study_name, storage=sqlite_storage_url)
            _logger.debug("Existing SQLite study deleted.")
        except KeyError:
            pass
            
        optuna.copy_study(
            from_study_name=study_name,
            from_storage=source_url,
            to_storage=sqlite_storage_url,
            to_study_name=study_name
        )
        _logger.info("✅ SQLite persistence complete.")
        return True
        
    except Exception as e:
        _logger.error(f"❌ Failed to persist study to SQLite: {e}")
        return False
