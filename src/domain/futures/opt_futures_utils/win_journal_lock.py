"""
Windows Named Mutex lock for Optuna Journal storage.
Cross-process safe: avoids PermissionError [WinError 32] and 'did not possess lock'
when using JournalFileBackend with n_jobs > 1 on Windows.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

_logger: logging.Logger = logging.getLogger(__name__)

# Constants used only on Windows
_INFINITE: int = 0xFFFFFFFF
_WAIT_OBJECT_0: int = 0


def _mutex_name_from_path(file_path: str) -> str:
    """Derive a short, safe system-wide mutex name from the journal file path."""
    normalized: str = str(file_path).replace("\\", "/").lower()
    h: str = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"Global\\optuna_journal_{h}"


if sys.platform == "win32":

    class WindowsNamedMutexJournalLock:
        """
        Process-spanning lock for Journal file I/O on Windows.
        Uses a named mutex so all processes (main + n_jobs workers) share the same lock
        without file rename; picklable so workers get a handle to the same mutex by name.
        """

        def __init__(self, file_path: str) -> None:
            self._mutex_name: str = _mutex_name_from_path(file_path)
            self._handle: wintypes.HANDLE | None = self._open_or_create_mutex()

        def _open_or_create_mutex(self) -> wintypes.HANDLE | None:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # CreateMutexW(lpMutexAttributes, bInitialOwner, lpName)
            # bInitialOwner=False: do not take ownership on create; use WaitForSingleObject to acquire
            handle: wintypes.HANDLE = kernel32.CreateMutexW(
                None, False, self._mutex_name  # type: ignore[arg-type]
            )
            if not handle:
                err: int = ctypes.get_last_error()
                _logger.error("CreateMutexW failed for %s, error=%s", self._mutex_name, err)
                raise OSError(err, f"CreateMutexW failed: {self._mutex_name}")
            return handle

        def acquire(self) -> bool:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if self._handle is None:
                self._handle = self._open_or_create_mutex()
            ret: int = kernel32.WaitForSingleObject(
                self._handle, _INFINITE  # type: ignore[arg-type]
            )
            if ret != _WAIT_OBJECT_0:
                err = ctypes.get_last_error()
                raise OSError(err, f"WaitForSingleObject failed: {ret}")
            return True

        def release(self) -> None:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if self._handle is None:
                raise RuntimeError("Error: did not possess lock")
            if not kernel32.ReleaseMutex(self._handle):  # type: ignore[arg-type]
                err = ctypes.get_last_error()
                raise OSError(err, "ReleaseMutex failed")

        def __del__(self) -> None:
            if getattr(self, "_handle", None) is not None:
                try:
                    ctypes.windll.kernel32.CloseHandle(self._handle)  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._handle = None

        def __getstate__(self) -> dict[str, Any]:
            return {"_mutex_name": self._mutex_name}

        def __setstate__(self, state: dict[str, Any]) -> None:
            self._mutex_name = state["_mutex_name"]
            self._handle = self._open_or_create_mutex()
