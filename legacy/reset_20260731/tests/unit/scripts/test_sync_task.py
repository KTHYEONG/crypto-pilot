# tests/unit/scripts/test_sync_task.py
from __future__ import annotations

import json
import os
from scripts.sync_task import (
    _clean_specs,
    _update_decisions_json,
    _update_index,
)


def test_update_decisions_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("docs/decisions", exist_ok=True)

    adr_id = _update_decisions_json(
        task="TASK_TEST",
        title="Test Task",
        why="Why test",
        what="What test",
        impact="Impact test",
        domain="signal",
        failed_hypothesis="Bad hypo",
        failure_reason="Failed test",
    )

    assert adr_id.startswith("ADR_")
    tasks_path = tmp_path / "docs" / "decisions" / "task_index.json"
    anti_path = tmp_path / "docs" / "decisions" / "anti_patterns.json"

    assert tasks_path.exists()
    assert anti_path.exists()

    tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))
    assert tasks_data["tasks"][0]["task_id"] == "TASK_TEST"

    anti_data = json.loads(anti_path.read_text(encoding="utf-8"))
    assert anti_data[0]["failed_hypothesis"] == "Bad hypo"


def test_update_index_code_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("docs", exist_ok=True)

    _update_index(source_file="src/foo.py", test_file="tests/test_foo.py", doc_file="docs/foo.md")

    code_map_path = tmp_path / "docs" / "code_map.json"
    assert code_map_path.exists()

    data = json.loads(code_map_path.read_text(encoding="utf-8"))
    assert "src/foo.py" in data
    assert data["src/foo.py"]["testing"] == "tests/test_foo.py"


def test_clean_specs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("docs/specs", exist_ok=True)
    spec_file = tmp_path / "docs" / "specs" / "feature.md"
    spec_file.write_text("spec content", encoding="utf-8")

    cleaned = _clean_specs()
    assert cleaned == 1
    assert not spec_file.exists()
