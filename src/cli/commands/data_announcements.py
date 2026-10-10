"""CLI handler: collect Binance delisting CMS notices into committed evidence."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.core import delisting_announcements as evidence
from src.core.delisting_announcements import DelistingNotice
from src.market_data.services import delisting_announcements as notices

# Shares the data command logger so the operator log stream and its filters stay unchanged.
_logger = logging.getLogger("src.cli.commands.data")

_CMS_BASE = "https://www.binance.com/bapi/composite/v1/public/cms/article"
# The CMS edge answers 429 to non-browser agents regardless of request rate.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


def _get_json(url: str, *, label: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": _BROWSER_USER_AGENT})  # noqa: S310 - fixed Binance CMS host
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed Binance CMS host
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return payload
    except urllib.error.HTTPError as exc:
        raise notices.CmsHttpError(int(exc.code), f"CMS {label} failed with status {exc.code}") from exc
    except OSError as exc:
        raise notices.CmsHttpError(-1, f"CMS {label} unreachable: {exc}") from exc


def _fetch_page(page_no: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"type": 1, "catalogId": 161, "pageNo": page_no, "pageSize": 50})
    return _get_json(f"{_CMS_BASE}/list/query?{query}", label=f"list:{page_no}")


def _fetch_detail(code: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"articleCode": code})
    return _get_json(f"{_CMS_BASE}/detail/query?{query}", label=f"detail:{code}")


def _load_existing(committed_path: Path) -> tuple[DelistingNotice, ...]:
    try:
        return tuple(evidence.load_delisting_notices())
    except DataIntegrityError:
        if committed_path.exists():
            raise
        _logger.warning(
            "[DATA] stage=collect_delisting_announcements status=NO_COMMITTED_EVIDENCE bootstrapping from empty",
        )
        return ()


def _collect_new(existing: tuple[DelistingNotice, ...]) -> list[dict[str, Any]]:
    rows, report = notices.collect_delisting_notices(
        existing=existing,
        fetch_page=_fetch_page,
        fetch_detail=_fetch_detail,
        sleep=time.sleep,
        now=pd.Timestamp.now(tz="UTC"),
    )
    _logger.info(
        "[DATA] stage=collect_delisting_announcements fetched=%d skipped=%d unfetched=%d total_remote=%d",
        len(report.fetched), len(report.skipped_existing), len(report.unfetched), report.total_remote,
    )
    print(json.dumps(asdict(report), sort_keys=True))  # noqa: T201 - report on stdout is the CLI contract
    return list(rows)


def _reextract(existing: tuple[DelistingNotice, ...]) -> list[dict[str, Any]]:
    replacements, report = notices.reextract_delisting_notices(
        existing=existing,
        fetch_detail=_fetch_detail,
        sleep=time.sleep,
        now=pd.Timestamp.now(tz="UTC"),
    )
    existing_by_code = {notice.code: notice for notice in existing}
    for row in replacements:
        _logger.info(
            "[DATA] stage=collect_delisting_announcements status=REEXTRACTED code=%s old_symbols=%s new_symbols=%s",
            row["code"], list(existing_by_code[row["code"]].symbols), row["symbols"],
        )
    _logger.info(
        "[DATA] stage=collect_delisting_announcements reextracted=%d replaced=%d unfetched=%d",
        len(report.reextracted), len(report.replaced), len(report.unfetched),
    )
    print(json.dumps(asdict(report), sort_keys=True))  # noqa: T201 - reextract report on stdout is the CLI contract
    return list(replacements)


def _merge_committed(
    committed_path: Path, replacements: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if committed_path.exists():
        for line in committed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                merged[record["code"]] = record
    for row in replacements:
        merged[row["code"]] = row
    for row in rows:
        if row["code"] in merged:
            raise DataIntegrityError(f"delisting evidence duplicate code {row['code']!r}")
        merged[row["code"]] = row
    return sorted(merged.values(), key=lambda row: (row["release_ms"], row["code"]))


def _write_validated(committed_path: Path, ordered: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in ordered)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(committed_path.parent), prefix=".delisting_", suffix=".tmp")
    try:
        with open(tmp_fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        evidence.load_delisting_notices(Path(tmp_name))
        Path(tmp_name).replace(committed_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def collect_delisting_announcements(args: argparse.Namespace) -> None:
    committed_path = evidence.default_delisting_evidence_path()
    existing = _load_existing(committed_path)
    collector: Callable[[tuple[DelistingNotice, ...]], list[dict[str, Any]]] = _reextract if args.reextract else _collect_new
    collected = collector(existing)
    if not args.write:
        return
    replacements, rows = (collected, []) if args.reextract else ([], collected)
    ordered = _merge_committed(committed_path, replacements, rows)
    _write_validated(committed_path, ordered)
    _logger.info("[DATA] stage=collect_delisting_announcements status=WRITTEN records=%d", len(ordered))
