from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).parents[2] / "tools" / "agent_skills" / "lean_check.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("lean_check", _SCRIPT_PATH)
assert _MODULE_SPEC is not None
assert _MODULE_SPEC.loader is not None
_lean_check = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_lean_check)


def test_feature_named_cli_test_matches_imported_source() -> None:
    source = "src/mhs/contracts.py"
    test_file = "tests/integration/mhs/test_golden_identity.py"

    assert _lean_check._test_references_source(test_file, source)


def test_unrelated_test_does_not_match_source() -> None:
    assert not _lean_check._test_references_source(
        "tests/unit/quant/evaluation/test_promotion.py",
        "src/mhs/evaluation/folds.py",
    )


def test_find_test_files_includes_semantic_source_test() -> None:
    files = _lean_check._find_test_files(["src/mhs/contracts.py"])

    assert "tests/unit/mhs/test_contracts.py" in files


def test_import_index_contains_semantic_reference() -> None:
    tf = "tests/unit/mhs/test_contracts_appresearch.py"

    mods = _lean_check._imported_source_modules(tf)

    assert "src.mhs.contracts" in mods


def test_semantic_match_builds_index_once() -> None:
    _lean_check._imported_source_modules.cache_clear()
    tf = "tests/unit/mhs/test_contracts_appresearch.py"
    src = "src/mhs/contracts.py"

    assert _lean_check._test_references_source(tf, src)
    _lean_check._test_references_source(tf, src)

    idx = _lean_check._imported_source_modules.cache_info()
    assert idx.misses == 1  # test file parsed exactly once, not once per pair
