"""Database and Locking utilities for Optimization.

Combines Optuna Study management (MySQL/SQLite) and Windows-specific I/O locking.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Any

import optuna

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

_logger: logging.Logger = logging.getLogger("opt_futures")

# --- DB Utilities (from db_utils.py) ---

def fast_reset_study(
    study_name: str,
    db_user: str,
    db_pass: str,
    db_host: str,
    db_port: str,
    db_name: str,
) -> bool:
    """Bypass Optuna's slow ORM delete_study() by directly executing raw SQL against MySQL."""
    try:
        import pymysql
    except ImportError:
        _logger.warning("pymysql not installed; falling back to optuna.delete_study()")
        return False

    conn: pymysql.connections.Connection | None = None
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

        # 3. Delete child tables in chunks
        if trial_ids:
            chunk_size = 500
            child_tables = [
                "trial_heartbeats",
                "trial_intermediate_values",
                "trial_system_attributes",
                "trial_user_attributes",
                "trial_params",
                "trial_values",
            ]
            for i in range(0, len(trial_ids), chunk_size):
                chunk = trial_ids[i : i + chunk_size]
                fmt = ",".join(["%s"] * len(chunk))
                for table in child_tables:
                    try:
                        cursor.execute(
                            f"DELETE FROM {table} WHERE trial_id IN ({fmt})",  # noqa: S608
                            chunk,
                        )
                    except Exception as e:
                        _logger.debug("Table '%s' skip or error: %s", table, e)

        # 4. Delete trials
        cursor.execute(
            "DELETE FROM trials WHERE study_id = %s",
            (study_id,),
        )

        # 5. Delete study-level attributes
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
                ...
        _logger.warning("fast_reset_study: direct SQL deletion failed. Error: %s", exc)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                ...


def save_study_to_sqlite(
    study: optuna.Study, project_root: str, target_study_name: str | None = None
) -> bool:
    """Export ONLY the best trial from the current study to local SQLite for production."""
    study_name: str = target_study_name if target_study_name is not None else study.study_name
    sqlite_path: str = os.path.join(project_root, "futures_strategy.db")
    sqlite_storage_url: str = f"sqlite:///{sqlite_path}"

    _logger.info("  💾 Exporting BEST trial to local SQLite as '%s'...", study_name)

    try:
        try:
            optuna.delete_study(study_name=study_name, storage=sqlite_storage_url)
        except (KeyError, Exception):
            ...

        create_kwargs: dict[str, Any] = {
            "study_name": study_name,
            "storage": sqlite_storage_url,
            "load_if_exists": False,
        }

        directions = getattr(study, "directions", None)
        if directions:
            create_kwargs["directions"] = list(directions)
        else:
            create_kwargs["direction"] = study.direction  # type: ignore[assignment]

        local_study: optuna.Study = optuna.create_study(**create_kwargs)

        for trial in study.trials:
            local_study.add_trial(trial)

        _logger.info("✅ SQLite persistence (%d Pareto Trials) complete.", len(study.trials))
        return True

    except Exception as e:
        _logger.error("❌ Failed to persist best trial to SQLite: %s", e)
        return False

# --- Locking Utilities (from win_journal_lock.py) ---

_INFINITE: int = 0xFFFFFFFF
_WAIT_OBJECT_0: int = 0

def _mutex_name_from_path(file_path: str) -> str:
    """Derive a short, safe system-wide mutex name from the journal file path."""
    normalized: str = str(file_path).replace("\\", "/").lower()
    h: str = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"Global\\optuna_journal_{h}"


if sys.platform == "win32":

    class WindowsNamedMutexJournalLock:
        """Process-spanning lock for Journal file I/O on Windows."""

        def __init__(self, file_path: str) -> None:
            """Initialize the mutex for the given file path."""
            self._mutex_name: str = _mutex_name_from_path(file_path)
            self._handle: wintypes.HANDLE | None = self._open_or_create_mutex()

        def _open_or_create_mutex(self) -> wintypes.HANDLE | None:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle: wintypes.HANDLE = kernel32.CreateMutexW(
                None,
                False,
                self._mutex_name,  # type: ignore[arg-type]
            )
            if not handle:
                err: int = ctypes.get_last_error()
                _logger.error("CreateMutexW failed for %s, error=%s", self._mutex_name, err)
                raise OSError(err, f"CreateMutexW failed: {self._mutex_name}")
            return handle

        def acquire(self) -> bool:
            """Acquire the Windows mutex."""
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if self._handle is None:
                self._handle = self._open_or_create_mutex()
            ret: int = kernel32.WaitForSingleObject(
                self._handle,
                _INFINITE,  # type: ignore[arg-type]
            )
            if ret != _WAIT_OBJECT_0:
                err = ctypes.get_last_error()
                raise OSError(err, f"WaitForSingleObject failed: {ret}")
            return True

        def release(self) -> None:
            """Release the Windows mutex."""
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if self._handle is None:
                raise RuntimeError("Error: did not possess lock")
            if not kernel32.ReleaseMutex(self._handle):  # type: ignore[arg-type]
                err = ctypes.get_last_error()
                raise OSError(err, "ReleaseMutex failed")

        def __del__(self) -> None:
            """Ensure handle is closed on deletion."""
            if getattr(self, "_handle", None) is not None:
                try:
                    ctypes.windll.kernel32.CloseHandle(self._handle)  # type: ignore[attr-defined]
                except Exception:
                    ...
                self._handle = None

        def __getstate__(self) -> dict[str, Any]:
            """Return state for pickling (mutex name only)."""
            return {"_mutex_name": self._mutex_name}

        def __setstate__(self, state: dict[str, Any]) -> None:
            """Restore state after pickling and re-open mutex."""
            self._mutex_name = state["_mutex_name"]
            self._handle = self._open_or_create_mutex()
