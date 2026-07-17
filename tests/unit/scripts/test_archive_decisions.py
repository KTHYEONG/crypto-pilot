# tests/unit/scripts/test_archive_decisions.py
from __future__ import annotations

import pytest

from scripts.archive_decisions import archive_decisions


def test_archive_decisions_happy_path(tmp_path):
    # Arrange
    decisions_file = tmp_path / "decisions.md"
    archive_file = tmp_path / "decisions_archive.md"

    # decisions.md에 18개의 ADR 엔트리 생성 (최신이 위로)
    decisions_content = "# Active Decisions\n\n"
    for i in range(18, 0, -1):
        decisions_content += (
            f"## [2026-07-10] [TASK_{i}] [ADR_20260710_TASK_{i}]\n- Context {i}\n- Resolution {i}\n- Impact {i}\n\n"
        )
    decisions_file.write_text(decisions_content, encoding="utf-8")

    # Act
    code = archive_decisions(
        decisions_path=str(decisions_file),
        archive_path=str(archive_file),
        max_entries=15,
    )

    # Assert
    assert code == 0
    # decisions.md에는 최신 15개(TASK_18 ~ TASK_4)만 남아야 함
    decisions_text = decisions_file.read_text(encoding="utf-8")
    assert "[TASK_18]" in decisions_text
    assert "[TASK_4]" in decisions_text
    assert "[TASK_3]" not in decisions_text  # 아카이브로 넘어가야 함

    # decisions_archive.md에는 나머지 3개(TASK_3 ~ TASK_1)가 들어가야 함 (순서 유지)
    archive_text = archive_file.read_text(encoding="utf-8")
    assert "[TASK_3]" in archive_text
    assert "[TASK_1]" in archive_text


def test_archive_decisions_limit_01_malformed(tmp_path):
    # Arrange
    decisions_file = tmp_path / "decisions.md"
    decisions_file.write_text("# Active Decisions\n\nNo structured headers here.\n", encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match="No valid ADR entries found"):
        archive_decisions(
            decisions_path=str(decisions_file),
            archive_path=str(tmp_path / "archive.md"),
            max_entries=15,
        )


def test_archive_decisions_limit_02_duplicate_prevention(tmp_path):
    # Arrange
    decisions_file = tmp_path / "decisions.md"
    archive_file = tmp_path / "decisions_archive.md"

    # 이미 아카이브에 이관된 엔트리가 있음
    archive_file.write_text(
        "# Archive\n\n## [2026-07-10] [TASK_1] [ADR_20260710_TASK_1]\n- Content 1\n",
        encoding="utf-8",
    )

    # decisions.md에 동일한 [ADR_20260710_TASK_1]을 포함하여 max_entries를 넘겨 이관 대상이 되게 함
    decisions_content = "# Active Decisions\n\n"
    for i in range(17, 0, -1):
        decisions_content += f"## [2026-07-10] [TASK_{i}] [ADR_20260710_TASK_{i}]\n- Content {i}\n\n"
    decisions_file.write_text(decisions_content, encoding="utf-8")

    # Act
    code = archive_decisions(
        decisions_path=str(decisions_file),
        archive_path=str(archive_file),
        max_entries=15,
    )

    # Assert
    assert code == 0
    # archive에 TASK_1이 두 번 기재되지 않고 유일해야 함
    archive_text = archive_file.read_text(encoding="utf-8")
    assert archive_text.count("[ADR_20260710_TASK_1]") == 1


def test_archive_decisions_file_not_found():
    with pytest.raises(FileNotFoundError, match="Decisions log not found"):
        archive_decisions(decisions_path="non_existent_file.md")


def test_archive_decisions_no_need_to_archive(tmp_path):
    # Arrange
    decisions_file = tmp_path / "decisions.md"
    decisions_content = "# Active Decisions\n\n## [2026-07-10] [TASK_1] [ADR_20260710_TASK_1]\n- Content\n"
    decisions_file.write_text(decisions_content, encoding="utf-8")

    # Act
    code = archive_decisions(
        decisions_path=str(decisions_file),
        archive_path=str(tmp_path / "archive.md"),
        max_entries=15,
    )

    # Assert
    assert code == 0
