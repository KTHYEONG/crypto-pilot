"""Unit isolation for src.mhs.report.artifacts payload normalization."""

from __future__ import annotations

from enum import StrEnum


class _SampleTier(StrEnum):
    SEALED = "sealed"


def test_jsonable_normalizes_str_enum_to_plain_string() -> None:
    from src.mhs.report.artifacts import _jsonable

    assert _jsonable(_SampleTier.SEALED) == "sealed"
    assert type(_jsonable(_SampleTier.SEALED)) is str


def test_jsonable_passes_json_primitives_through() -> None:
    from src.mhs.report.artifacts import _jsonable

    assert _jsonable({"a": (1, 2.5, None, True, "x")}) == {"a": [1, 2.5, None, True, "x"]}
