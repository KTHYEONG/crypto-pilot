# tests/unit/scripts/test_update_index.py
from __future__ import annotations

import json

import pytest

from scripts.update_index import update_index


def test_update_index_happy_path(tmp_path):
    # Arrange
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")

    # 소스 파일이 물리적으로 존재하는지 검증하므로, 임시 가상 파일 생성
    source_file = tmp_path / "foo.py"
    source_file.touch()

    # Act
    code = update_index(
        index_path=str(index_file),
        source_file=str(source_file),
        test_file="tests/test_foo.py",
        doc_file="docs/architecture/layer1.md",
    )

    # Assert
    assert code == 0
    data = json.loads(index_file.read_text(encoding="utf-8"))
    assert str(source_file) in data
    entry = data[str(source_file)]
    assert entry["testing"] == "tests/test_foo.py"
    assert entry["architecture"] == "docs/architecture/layer1.md"


def test_update_index_limit_03_missing_source(tmp_path):
    # Arrange
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Source file does not exist"):
        update_index(
            index_path=str(index_file),
            source_file="non_existent_file.py",
        )


def test_update_index_limit_04_broken_json_recovery(tmp_path):
    # Arrange
    index_file = tmp_path / "index.json"
    # 깨진 JSON 데이터 작성
    index_file.write_text('{"mappings": [', encoding="utf-8")

    source_file = tmp_path / "foo.py"
    source_file.touch()

    # Act
    code = update_index(
        index_path=str(index_file),
        source_file=str(source_file),
    )

    # Assert
    assert code == 0
    # 백업 파일(.bak)이 존재해야 함
    backup_file = tmp_path / "index.json.bak"
    assert backup_file.exists()
    assert backup_file.read_text(encoding="utf-8") == '{"mappings": ['

    # index.json은 정상 복구되어 데이터가 삽입되어야 함
    data = json.loads(index_file.read_text(encoding="utf-8"))
    assert str(source_file) in data
    assert "architecture" not in data[str(source_file)]


def test_update_index_existing_mapping(tmp_path):
    # Arrange
    index_file = tmp_path / "index.json"
    source_file = tmp_path / "foo.py"
    source_file.touch()

    index_file.write_text(
        json.dumps({str(source_file): {"testing": "old_test.py", "architecture": "old_doc.md"}}),
        encoding="utf-8",
    )

    # Act
    code = update_index(
        index_path=str(index_file),
        source_file=str(source_file),
        test_file="new_test.py",
        doc_file="new_doc.md",
    )

    # Assert
    assert code == 0
    data = json.loads(index_file.read_text(encoding="utf-8"))
    assert str(source_file) in data
    entry = data[str(source_file)]
    # 기존 testing이 문자열이고 새로 추가하는 것과 다르면 list로 병합됨
    assert entry["testing"] == ["old_test.py", "new_test.py"]
    assert entry["architecture"] == "new_doc.md"


def test_update_index_new_mapping_with_only_source(tmp_path):
    # Arrange
    index_file = tmp_path / "index.json"
    index_file.write_text("{}", encoding="utf-8")

    source_file = tmp_path / "foo.py"
    source_file.touch()

    # Act
    code = update_index(
        index_path=str(index_file),
        source_file=str(source_file),
    )

    # Assert
    assert code == 0
    data = json.loads(index_file.read_text(encoding="utf-8"))
    assert str(source_file) in data
    entry = data[str(source_file)]
    assert "testing" not in entry
    assert "architecture" not in entry
