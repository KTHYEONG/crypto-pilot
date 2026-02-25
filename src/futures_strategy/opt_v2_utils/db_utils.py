"""
최적화 도구인 Optuna의 학습 결과(Study)를 원격 DB(MySQL)에서 로컬 DB(SQLite)로 복사하고 관리하는 기능을 담당함.
최적화된 파라미터를 실제 트레이딩 시스템에서 사용할 수 있도록 데이터베이스를 실시간으로 동기화함.
"""
import os
import optuna
import logging
from typing import Optional

_logger: logging.Logger = logging.getLogger("opt_v2")


def fast_reset_study(
    study_name: str,
    db_user: str,
    db_pass: str,
    db_host: str,
    db_port: str,
    db_name: str,
) -> bool:
    """
    Bypass Optuna's slow ORM delete_study() by directly executing raw SQL against MySQL.

    Optuna's delete_study() uses SQLAlchemy ORM cascade, which performs row-by-row
    DELETE across 6+ child tables — extremely slow when trial count is large (100+).
    This function directly looks up study_id and runs targeted DELETE statements,
    reducing deletion time from ~30s to under 1s regardless of trial count.

    Optuna MySQL schema (as of optuna >= 3.x):
        studies
        └─ study_directions        (FK: study_id)
        └─ study_user_attributes   (FK: study_id)
        └─ study_system_attributes (FK: study_id)
        └─ trials
           └─ trial_params             (FK: trial_id)
           └─ trial_user_attributes    (FK: trial_id)
           └─ trial_system_attributes  (FK: trial_id)
           └─ trial_intermediate_values(FK: trial_id)
           └─ trial_heartbeats         (FK: trial_id)
    """
    try:
        import pymysql
    except ImportError:
        _logger.warning("pymysql not installed; falling back to optuna.delete_study()")
        return False

    conn: Optional[pymysql.connections.Connection] = None
    try:
        conn = pymysql.connect(
            host=db_host,
            port=int(db_port),
            user=db_user,
            password=db_pass,
            database=db_name,
            charset="utf8mb4",
            connect_timeout=10,
            autocommit=False,
        )
        cursor = conn.cursor()

        # 1. Resolve study_id
        cursor.execute(
            "SELECT study_id FROM studies WHERE study_name = %s LIMIT 1",
            (study_name,),
        )
        row = cursor.fetchone()
        if row is None:
            _logger.debug("fast_reset_study: study '%s' not found; nothing to delete.", study_name)
            return True  # nothing to do

        study_id: int = int(row[0])

        # 2. Collect all trial_ids belonging to this study
        cursor.execute(
            "SELECT trial_id FROM trials WHERE study_id = %s",
            (study_id,),
        )
        trial_ids = [r[0] for r in cursor.fetchall()]

        # 3. Delete child tables in dependency order (no FK constraint violation)
        if trial_ids:
            fmt = ",".join(["%s"] * len(trial_ids))
            child_tables = [
                "trial_heartbeats",
                "trial_intermediate_values",
                "trial_system_attributes",
                "trial_user_attributes",
                "trial_params",
            ]
            for table in child_tables:
                try:
                    cursor.execute(
                        f"DELETE FROM {table} WHERE trial_id IN ({fmt})",  # noqa: S608
                        trial_ids,
                    )
                except Exception:  # table may not exist in older optuna versions
                    pass

        # 4. Delete trials
        cursor.execute(
            "DELETE FROM trials WHERE study_id = %s",
            (study_id,),
        )

        # 5. Delete study-level attributes and the study row itself
        for table in (
            "study_system_attributes",
            "study_user_attributes",
            "study_directions",
        ):
            try:
                cursor.execute(
                    f"DELETE FROM {table} WHERE study_id = %s",  # noqa: S608
                    (study_id,),
                )
            except Exception:
                pass

        cursor.execute(
            "DELETE FROM studies WHERE study_id = %s",
            (study_id,),
        )

        conn.commit()
        _logger.info(
            "fast_reset_study: deleted study '%s' (id=%d, trials=%d) via direct SQL.",
            study_name,
            study_id,
            len(trial_ids),
        )
        return True

    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        _logger.warning(
            "fast_reset_study: direct SQL deletion failed (%s); caller should fallback.", exc
        )
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


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
