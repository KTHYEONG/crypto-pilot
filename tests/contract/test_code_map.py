from __future__ import annotations

import json
from pathlib import Path


def _code_map() -> dict[str, object]:
    path = Path("docs/code_map.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_code_map_canonical_sources_exist() -> None:
    """RF-MAP-01: every canonical source referenced by the code map exists."""
    stale = [source for source in _code_map() if not Path(source).exists()]
    assert not stale, f"code_map.json references missing sources: {stale}"


def test_code_map_linked_tests_exist() -> None:
    """RF-MAP-01: every test path linked from the code map exists."""
    missing: list[str] = []
    for source, entry in _code_map().items():
        testing = entry.get("testing") if isinstance(entry, dict) else None
        if testing is None:
            continue
        linked = [testing] if isinstance(testing, str) else testing
        missing.extend(
            f"{source} -> {test_path}"
            for test_path in linked
            if not Path(str(test_path)).exists()
        )
    assert not missing, f"code_map.json links missing tests: {missing}"


def test_code_map_has_no_stale_legacy_paths() -> None:
    """RF-MAP-01: no entry references the removed legacy layout."""
    banned_fragments = ("src/engine.py", "tests/test_baseline.py", "src/reliability_gate.py")
    offenders = [
        source for source in _code_map() if any(frag in source for frag in banned_fragments)
    ]
    assert not offenders, f"code_map.json contains stale legacy paths: {offenders}"
