from __future__ import annotations

from pathlib import Path

import pytest

from src.research.provenance.code_manifest import (
    CANONICAL_CARRY_CODE_UNITS,
    compute_code_hash,
)


def test_code_hash_default_is_64_hex_chars() -> None:
    digest = compute_code_hash()
    assert len(digest) == 64
    int(digest, 16)


def test_code_hash_is_ordered_by_logical_unit_and_excludes_facades(
    temporary_source_units: dict[str, Path],
) -> None:
    ordered = dict(temporary_source_units)
    reversed_units = dict(reversed(list(temporary_source_units.items())))
    assert compute_code_hash(ordered) == compute_code_hash(reversed_units)


def test_code_hash_changes_when_canonical_unit_bytes_change(
    temporary_source_units: dict[str, Path],
) -> None:
    before = compute_code_hash(temporary_source_units)
    unit_a = temporary_source_units["unit.a"]
    unit_a.write_text("# changed\nVALUE = 2\n", encoding="utf-8")
    after = compute_code_hash(temporary_source_units)
    assert before != after


def test_code_hash_excludes_facade_modules() -> None:
    assert all("src/core" not in str(p) for p in CANONICAL_CARRY_CODE_UNITS.values())
    assert all("src/data" not in str(p) for p in CANONICAL_CARRY_CODE_UNITS.values())
    assert all("src/engine" not in str(p) for p in CANONICAL_CARRY_CODE_UNITS.values())
    assert all("src/strategy" not in str(p) for p in CANONICAL_CARRY_CODE_UNITS.values())
    assert all("src/validation" not in str(p) for p in CANONICAL_CARRY_CODE_UNITS.values())
    for path in CANONICAL_CARRY_CODE_UNITS.values():
        assert Path(path).exists()


def test_code_hash_raises_file_not_found_for_missing_source() -> None:
    with pytest.raises(FileNotFoundError):
        compute_code_hash({"missing.unit": Path("does/not/exist.py")})
