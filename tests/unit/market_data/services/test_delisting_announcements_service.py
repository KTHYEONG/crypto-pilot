"""Invariant scenarios for the delisting notice collector service (spec 43 part 1)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.core.delisting_announcements import DelistingNotice
from src.market_data.services.delisting_announcements import (
    CmsHttpError,
    collect_delisting_notices,
)

_NOW = pd.Timestamp("2024-05-02T00:00:00Z")
_TITLE = "Binance Futures Will Delist AAAUSDT"
_BODY = "The AAAUSDT perpetual contract will be settled and removed."


def _article(code: str, release_ms: int = 1_714_598_400_000) -> dict[str, Any]:
    return {"code": code, "title": _TITLE, "releaseDate": release_ms}


def _collect(
    articles: list[dict[str, Any]],
    bodies: dict[str, str] | Callable[[str], dict[str, Any]],
    existing: tuple[DelistingNotice, ...] = (),
) -> tuple[list[dict[str, Any]], Any, list[float], list[str]]:
    total = len(articles)
    pages = [articles[index:index + 50] for index in range(0, max(total, 1), 50)] or [[]]
    sleeps: list[float] = []
    detail_calls: list[str] = []

    def fetch_page(page_no: int) -> dict[str, Any]:
        batch = pages[page_no - 1] if page_no - 1 < len(pages) else []
        return {"data": {"catalogs": [{"articles": batch, "total": total}]}}

    def fetch_detail(code: str) -> dict[str, Any]:
        detail_calls.append(code)
        if callable(bodies):
            return bodies(code)
        return {"data": {"body": bodies[code]}}

    rows, report = collect_delisting_notices(
        existing=existing,
        fetch_page=fetch_page,
        fetch_detail=fetch_detail,
        sleep=sleeps.append,
        now=_NOW,
    )
    return rows, report, sleeps, detail_calls


def test_skips_known_codes() -> None:
    existing = (
        DelistingNotice(code="c1", title=_TITLE, release_at=pd.Timestamp("2024-05-01T00:00:00Z"),
                        kind="delist", symbols=("AAAUSDT",)),
        DelistingNotice(code="c2", title=_TITLE, release_at=pd.Timestamp("2024-05-01T00:00:01Z"),
                        kind="delist", symbols=("AAAUSDT",)),
    )
    rows, report, _, detail_calls = _collect([_article("c1"), _article("c2")], {"c1": _BODY, "c2": _BODY}, existing)
    assert detail_calls == []
    assert rows == []
    assert report.skipped_existing == ("c1", "c2")


def test_429_backoff() -> None:
    attempts = {"count": 0}

    def fetch_detail(code: str) -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise CmsHttpError(429, "rate limited")
        return {"data": {"body": _BODY}}

    rows, report, recorded_sleeps, _ = _collect([_article("c1")], fetch_detail)
    assert attempts["count"] == 3
    assert 10.0 in recorded_sleeps
    assert 20.0 in recorded_sleeps
    assert [row["code"] for row in rows] == ["c1"]
    assert report.fetched == ("c1",)


def test_persistent_failure_is_reported_not_written() -> None:
    def fetch_detail(code: str) -> dict[str, Any]:
        raise CmsHttpError(500, "boom")

    rows, report, _, _ = _collect([_article("c1"), _article("c2")], fetch_detail)
    assert rows == []
    assert report.unfetched == ("c1", "c2")
    assert report.fetched == ()


def test_empty_body_is_unfetched() -> None:
    rows, report, _, _ = _collect([_article("c1")], {"c1": ""})
    assert rows == []
    assert report.unfetched == ("c1",)


def test_spacing_before_every_request() -> None:
    events: list[tuple[str, float | str]] = []

    def fetch_page(page_no: int) -> dict[str, Any]:
        events.append(("fetch", f"page:{page_no}"))
        return {"data": {"catalogs": [{"articles": [_article("c1")], "total": 1}]}}

    def fetch_detail(code: str) -> dict[str, Any]:
        events.append(("fetch", code))
        return {"data": {"body": _BODY}}

    def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))

    collect_delisting_notices(
        existing=(), fetch_page=fetch_page, fetch_detail=fetch_detail, sleep=sleep, now=_NOW,
    )
    slept = 0.0
    for kind, value in events:
        if kind == "sleep":
            assert isinstance(value, float)
            slept += value
        else:
            assert slept >= 1.5, f"request {value} without 1.5 s spacing"
            slept = 0.0


def test_rows_are_canonical_and_sorted() -> None:
    rows, report, _, _ = _collect(
        [_article("c2", 1_714_598_401_000), _article("c1", 1_714_598_400_000)],
        {"c1": _BODY, "c2": _BODY},
    )
    assert [row["code"] for row in rows] == ["c1", "c2"]
    assert set(rows[0]) == {"code", "title", "release_ms", "kind", "symbols", "collected_at"}
    assert rows[0]["kind"] == "delist"
    assert rows[0]["symbols"] == ["AAAUSDT"]
    assert rows[0]["collected_at"] == "2024-05-02T00:00:00Z"
    assert report.total_remote == 2


def test_malformed_article_fails_closed() -> None:
    import pytest

    with pytest.raises(DataIntegrityError):
        _collect([{"title": _TITLE}], {"x": _BODY})


@pytest.mark.parametrize("body", [None, 123, {}, {"type": "paragraph"}, "<p> </p>", "{bad", ""])
def test_missing_visible_body_is_unfetched(body: object) -> None:
    rows, report, _, _ = _collect([_article("c1")], lambda code: {"data": {"body": body}})
    assert rows == []
    assert report.unfetched == ("c1",)


def test_missing_body_key_is_reported() -> None:
    rows, report, _, _ = _collect([_article("c1")], lambda code: {"data": {}})
    assert rows == []
    assert report.unfetched == ("c1",)


def test_cms_tree_ignores_metadata_and_decodes_html_entities() -> None:
    import json

    body = json.dumps({"type": "paragraph", "children": [
        {"type": "text", "text": "<p>USD&#x24C8;-M CVC perpetual contract</p>"},
        {"type": "link", "href": "https://example.test/FAKEUSDT", "children": [
            {"text": " will be delisted."},
        ]},
    ]})
    rows, _, _, _ = _collect([{"code": "c1", "title": "Update", "releaseDate": 1000}], {"c1": body})
    assert rows[0]["kind"] == "delist"
    assert rows[0]["symbols"] == ["CVCUSDT"]


def test_overlapping_pages_fetch_each_code_once_and_report_shortfall(caplog) -> None:
    pages = [[_article("c1")], [_article("c1"), _article("c2")], []]
    calls: list[str] = []

    def fetch_detail(code: str) -> dict[str, Any]:
        calls.append(code)
        return {"data": {"body": _BODY}}

    rows, report = collect_delisting_notices(
        existing=(), fetch_page=lambda page: {"data": {"catalogs": [{
            "articles": pages[page - 1], "total": 3,
        }]}}, fetch_detail=fetch_detail, sleep=lambda seconds: None, now=_NOW,
    )
    assert calls == ["c1", "c2"]
    assert len(rows) == 2
    assert (report.listed_remote, report.total_remote) == (2, 3)
    assert "INCONSISTENT_TOTAL" in caplog.text


def test_repeated_page_terminates_without_duplicate_rows() -> None:
    rows, report = collect_delisting_notices(
        existing=(), fetch_page=lambda page: {"data": {"catalogs": [{
            "articles": [_article("c1")], "total": 3,
        }]}}, fetch_detail=lambda code: {"data": {"body": _BODY}},
        sleep=lambda seconds: None, now=_NOW,
    )
    assert len(rows) == report.listed_remote == 1


@pytest.mark.parametrize(("total", "articles"), [(None, []), (True, []), (-1, []), (1, {})])
def test_invalid_catalog_fails_closed(total: object, articles: object) -> None:
    with pytest.raises(DataIntegrityError):
        collect_delisting_notices(
            existing=(), fetch_page=lambda page: {"data": {"catalogs": [{
                "articles": articles, "total": total,
            }]}}, fetch_detail=lambda code: {}, sleep=lambda seconds: None, now=_NOW,
        )


@pytest.mark.parametrize("now", [pd.NaT, pd.Timestamp("2024-01-01")])
def test_invalid_collection_time_rejected(now: pd.Timestamp) -> None:
    with pytest.raises(DataIntegrityError):
        collect_delisting_notices(
            existing=(), fetch_page=lambda page: {}, fetch_detail=lambda code: {},
            sleep=lambda seconds: None, now=now,
        )


@pytest.mark.parametrize("overrides", [{"title": ""}, {"releaseDate": -1}, {"releaseDate": True}])
def test_invalid_article_metadata_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(DataIntegrityError):
        _collect([{**_article("c1"), **overrides}], {"c1": _BODY})
