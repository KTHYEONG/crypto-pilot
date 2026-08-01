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
    source = "src/cli/run_backtest.py"
    test_file = "tests/integration/cli/test_candidate_promotion_cli.py"

    assert _lean_check._test_references_source(test_file, source)
    assert _lean_check._source_has_matching_test(source, [test_file])


def test_unrelated_test_does_not_match_source() -> None:
    assert not _lean_check._test_references_source(
        "tests/unit/research/evaluation/test_promotion.py",
        "src/cli/run_backtest.py",
    )


def test_find_test_files_includes_semantic_source_test() -> None:
    files = _lean_check._find_test_files(["src/cli/run_backtest.py"])

    assert "tests/integration/cli/test_candidate_promotion_cli.py" in files
