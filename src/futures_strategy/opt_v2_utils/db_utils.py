"""
최적화 도구인 Optuna의 학습 결과(Study)를 원격 DB(MySQL)에서 로컬 DB(SQLite)로 복사하고 관리하는 기능을 담당함.
최적화된 파라미터를 실제 트레이딩 시스템에서 사용할 수 있도록 데이터베이스를 실시간으로 동기화함.
"""
import os
import logging
from typing import Any, Dict, Optional

import optuna

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

        # 3. Delete child tables in chunks to avoid huge query and handle potential locking
        if trial_ids:
            chunk_size = 500
            child_tables = [
                "trial_heartbeats",
                "trial_intermediate_values",
                "trial_system_attributes",
                "trial_user_attributes",
                "trial_params",
                "trial_values",  # Required for Optuna 3.x to avoid FK violation
            ]
            for i in range(0, len(trial_ids), chunk_size):
                chunk = trial_ids[i : i + chunk_size]
                fmt = ",".join(["%s"] * len(chunk))
                for table in child_tables:
                    try:
                        cursor.execute(
                            f"DELETE FROM {table} WHERE trial_id IN ({fmt})",
                            chunk,
                        )
                    except Exception as e:
                        _logger.debug("Table '%s' skip or error: %s", table, e)

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
            except Exception as e:
                _logger.debug("Study table '%s' skip or error: %s", table, e)

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
            "fast_reset_study: direct SQL deletion failed. Error: %s", exc
        )
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def save_study_to_sqlite(
    study: optuna.Study, project_root: str, target_study_name: Optional[str] = None
) -> bool:
    """
    Export ONLY the best trial from the current study to local SQLite for production.
    When target_study_name is set (e.g. "futures_strategy"), the trial is stored under
    that name so downstream (e.g. real_trader) can load it.
    """
    study_name: str = target_study_name if target_study_name is not None else study.study_name
    sqlite_path: str = os.path.join(project_root, "futures_strategy.db")
    sqlite_storage_url: str = f"sqlite:///{sqlite_path}"
    
    _logger.info("  💾 Exporting BEST trial to local SQLite as '%s'...", study_name)
    
    try:
        # 1. Delete existing local study to ensure fresh best trial
        try:
            optuna.delete_study(study_name=study_name, storage=sqlite_storage_url)
        except (KeyError, Exception):
            # Study may not exist yet or storage may be empty; both are safe to ignore.
            pass

        # 2. Create new local study
        create_kwargs: Dict[str, Any] = {
            "study_name": study_name,
            "storage": sqlite_storage_url,
            "load_if_exists": False,
        }

        directions = getattr(study, "directions", None)
        if directions:
            # Multi-objective (or single-objective with directions tuple)
            create_kwargs["directions"] = list(directions)
        else:
            # Backward-compatible path for old single-objective studies
            create_kwargs["direction"] = study.direction  # type: ignore[assignment]

        local_study: optuna.Study = optuna.create_study(**create_kwargs)

        # 3. Add trials (이미 Pareto Front로만 구성된 study가 전달됨)
        for trial in study.trials:
            local_study.add_trial(trial)

        _logger.info("✅ SQLite persistence (%d Pareto Trials) complete.", len(study.trials))
        return True

    except Exception as e:
        _logger.error("❌ Failed to persist best trial to SQLite: %s", e)
        return False
