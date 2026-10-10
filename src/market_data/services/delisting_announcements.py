"""Fetch missing Binance delisting CMS notices into canonical evidence rows."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from html.parser import HTMLParser
from typing import Any, Final
from urllib.error import HTTPError

import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.delisting_announcements import (
    DelistingNotice,
    classify_notice,
    extract_contract_symbols,
)

_logger = logging.getLogger(__name__)

_REQUEST_SPACING_S: Final[float] = 1.5
_MAX_RETRIES: Final[int] = 4
_BACKOFF_UNIT_S: Final[float] = 10.0
class _BodyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class CmsHttpError(Exception):
    """HTTP failure of the CMS transport, carrying the response status."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"CMS request failed with status {status}")
        self.status = status


@dataclass(frozen=True, slots=True)
class CollectionReport:
    fetched: tuple[str, ...] = ()
    skipped_existing: tuple[str, ...] = ()
    unfetched: tuple[str, ...] = ()
    total_remote: int = 0
    listed_remote: int = 0


def _detail_text(body: object) -> str:
    if body is None:
        return ""
    if isinstance(body, dict | list):
        parts: list[str] = []

        def _walk(node: object) -> None:
            if isinstance(node, str):
                if node.strip():
                    parts.append(node)
            elif isinstance(node, dict):
                for key in ("text", "child", "children"):
                    if key in node:
                        _walk(node[key])
            elif isinstance(node, list):
                for value in node:
                    _walk(value)

        _walk(body)
        return _detail_text("\n".join(parts))
    if not isinstance(body, str):
        return ""
    text = body.strip()
    if text[:1] in ("{", "["):
        try:
            return _detail_text(json.loads(text))
        except json.JSONDecodeError:
            return ""
    parser = _BodyParser()
    parser.feed(body)
    return " ".join(parser.parts).strip()


def _request_with_backoff(
    operation: Callable[[], Any], sleep: Callable[[float], None], *, label: str,
) -> Any:
    attempt = 0
    while True:
        sleep(_REQUEST_SPACING_S)
        try:
            return operation()
        except (CmsHttpError, HTTPError) as exc:
            status = exc.status if isinstance(exc, CmsHttpError) else exc.code
            if status != 429 or attempt >= _MAX_RETRIES:
                raise
            attempt += 1
            wait = _BACKOFF_UNIT_S * attempt
            _logger.warning("[DATA] stage=collect_delisting_announcements label=%s status=429 retry=%d wait_s=%s", label, attempt, wait)
            sleep(wait)


def _list_articles(
    fetch_page: Callable[[int], dict[str, Any]], sleep: Callable[[float], None],
) -> tuple[list[dict[str, Any]], int]:
    listed: list[dict[str, Any]] = []
    total = 0
    page = 1
    seen: set[str] = set()
    while True:
        payload = _request_with_backoff(partial(fetch_page, page), sleep, label=f"list:{page}")
        try:
            catalog = payload["data"]["catalogs"][0]
            articles = catalog.get("articles", [])
            total = catalog["total"]
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                raise ValueError("invalid total")
            if not isinstance(articles, list):
                raise ValueError("invalid articles")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise DataIntegrityError(f"delisting CMS list page {page} has no article catalog") from exc
        if not articles:
            break
        before = len(listed)
        for article in articles:
            if not isinstance(article, dict) or not isinstance(article.get("code"), str) or not article["code"]:
                raise DataIntegrityError("delisting CMS article without code")
            if article["code"] not in seen:
                seen.add(article["code"])
                listed.append(article)
        if len(listed) == before:
            break
        if len(listed) >= total:
            break
        page += 1
    return listed, total


def _fetch_body(
    code: str, fetch_detail: Callable[[str], dict[str, Any]], sleep: Callable[[float], None],
) -> str | None:
    try:
        payload = _request_with_backoff(lambda: fetch_detail(code), sleep, label=f"detail:{code}")
    except CmsHttpError as exc:
        _logger.warning("[DATA] stage=collect_delisting_announcements code=%s status=%s", code, exc.status)
        return None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[DATA] stage=collect_delisting_announcements code=%s error=%s", code, exc)
        return None
    try:
        body = payload["data"]["body"] if isinstance(payload.get("data"), dict) else None
    except (AttributeError, TypeError, KeyError):
        return None
    text = _detail_text(body)
    return text or None


def _iso_z(moment: pd.Timestamp) -> str:
    as_utc: datetime = moment.tz_convert("UTC").to_pydatetime().astimezone(UTC)
    return as_utc.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ReextractReport:
    reextracted: tuple[str, ...] = ()
    replaced: tuple[str, ...] = ()
    unfetched: tuple[str, ...] = ()


def reextract_delisting_notices(
    *,
    existing: Sequence[DelistingNotice],
    fetch_detail: Callable[[str], dict[str, Any]],
    sleep: Callable[[float], None],
    now: pd.Timestamp,
) -> tuple[list[dict[str, Any]], ReextractReport]:
    """Re-derive symbols for committed ``delist`` rows from live bodies; replace only on symbol change."""
    if pd.isna(now) or now.tzinfo is None:
        raise DataIntegrityError("now must be a timezone-aware timestamp")
    replacements: list[dict[str, Any]] = []
    reextracted: list[str] = []
    replaced: list[str] = []
    unfetched: list[str] = []
    for notice in existing:
        if notice.kind != "delist":
            continue
        text = _fetch_body(notice.code, fetch_detail, sleep)
        if text is None:
            unfetched.append(notice.code)
            continue
        reextracted.append(notice.code)
        symbols = list(extract_contract_symbols(notice.title, text))
        if symbols != list(notice.symbols):
            replacements.append({
                "code": notice.code,
                "title": notice.title,
                "release_ms": notice.release_ms if notice.release_ms is not None else int(notice.release_at.value // 1_000_000),
                "kind": notice.kind,
                "symbols": symbols,
                "collected_at": _iso_z(now),
            })
            replaced.append(notice.code)
    replacements.sort(key=lambda row: (row["release_ms"], row["code"]))
    return replacements, ReextractReport(
        reextracted=tuple(reextracted),
        replaced=tuple(replaced),
        unfetched=tuple(unfetched),
    )


def collect_delisting_notices(
    *,
    existing: Sequence[DelistingNotice],
    fetch_page: Callable[[int], dict[str, Any]],
    fetch_detail: Callable[[str], dict[str, Any]],
    sleep: Callable[[float], None],
    now: pd.Timestamp,
) -> tuple[list[dict[str, Any]], CollectionReport]:
    """Fetch missing CMS notices and return canonical evidence rows plus a report; never mutates ``existing``."""
    if pd.isna(now) or now.tzinfo is None:
        raise DataIntegrityError("now must be a timezone-aware timestamp")
    known = {notice.code for notice in existing}
    articles, total = _list_articles(fetch_page, sleep)
    rows: list[dict[str, Any]] = []
    fetched: list[str] = []
    skipped: list[str] = []
    unfetched: list[str] = []
    for article in articles:
        code = article["code"]
        if code in known:
            skipped.append(code)
            continue
        title = article.get("title")
        release_ms = article.get("releaseDate")
        if not isinstance(title, str) or not title.strip():
            raise DataIntegrityError(f"delisting CMS article {code} without title")
        if isinstance(release_ms, bool) or not isinstance(release_ms, int) or release_ms < 0:
            raise DataIntegrityError(f"delisting CMS article {code} without releaseDate")
        text = _fetch_body(code, fetch_detail, sleep)
        if text is None:
            unfetched.append(code)
            continue
        fetched.append(code)
        rows.append({
            "code": code,
            "title": title,
            "release_ms": release_ms,
            "kind": classify_notice(title, text),
            "symbols": list(extract_contract_symbols(title, text)),
            "collected_at": _iso_z(now),
        })
    rows.sort(key=lambda row: (row["release_ms"], row["code"]))
    if len(articles) < total:
        _logger.warning(
            "[DATA] stage=collect_delisting_announcements status=INCONSISTENT_TOTAL listed=%d total=%d",
            len(articles), total,
        )
    return rows, CollectionReport(
        fetched=tuple(fetched),
        skipped_existing=tuple(skipped),
        unfetched=tuple(unfetched),
        total_remote=total,
        listed_remote=len(articles),
    )
