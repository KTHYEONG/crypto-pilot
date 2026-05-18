"""pytest root conftest — project root를 sys.path에 추가."""
from __future__ import annotations

import sys
from pathlib import Path

# project root: tests/ 의 상위 디렉토리
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
