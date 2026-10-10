"""Binance delisting announcement evidence: committed notices and pure resolution."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Literal, cast

import pandas as pd

from src.common.errors import DataIntegrityError

NoticeKind = Literal["delist", "postponed", "other"]

_EVIDENCE_FIELDS: Final[tuple[str, ...]] = ("code", "title", "release_ms", "kind", "symbols", "collected_at")
_VALID_KINDS: Final[frozenset[str]] = frozenset({"delist", "postponed", "other"})
_RESOLVE_WINDOW: Final[pd.Timedelta] = pd.Timedelta(days=21)

_ACTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"delist|remov|settl|terminat", re.IGNORECASE)
_CONTEXT_PATTERN: Final[re.Pattern[str]] = re.compile(r"perpetual|futures?|contract|coin-m|usd[\s\-s\u24c8]*-m", re.IGNORECASE)
_DIRECT_SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z0-9]{2,15}USD[TC])\b")
_SPACED_SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b([A-Z0-9]{2,15})\s+USDT\b(?!-)")
_AND: Final[str] = r"[Aa][Nn][Dd]"
_PERP: Final[str] = r"[Pp][Ee][Rr][Pp][Ee][Tt][Uu][Aa][Ll]"
_CONTRACT_WORD: Final[str] = r"[Cc][Oo][Nn][Tt][Rr][Aa][Cc][Tt][Ss]?"
_MARG_SUFFIX: Final[str] = r"[Mm](?:[Aa][Rr][Gg][Ii][Nn][Ee][Dd])?"
_MARGIN_TAG: Final[str] = rf"(?:USDT-{_MARG_SUFFIX}|USD[S\u24c8]-{_MARG_SUFFIX})"
_TICK: Final[str] = r"(?:[A-Z0-9]{2,15}USD[TC]|[A-Z0-9]{2,15})"
_SEP: Final[str] = rf"(?:\s*,\s*(?:{_AND}\s+)?|\s*&\s*|\s+{_AND}\s+)"
_TICKER_LIST: Final[str] = rf"{_TICK}(?:{_SEP}{_TICK})*"
_TICKER_LIST_BEFORE_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\b({_TICKER_LIST})\s+{_MARGIN_TAG}\s+(?:{_PERP}\s+)?{_CONTRACT_WORD}\b"
)
_TICKER_LIST_AFTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\b{_MARGIN_TAG}\s+({_TICKER_LIST})\s+{_PERP}\s+{_CONTRACT_WORD}\b"
)
_TICKER_LIST_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    _SEP
)
_FULL_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]{2,15}USD[TC]\Z")
_TICKER_STOP_WORDS: Final[frozenset[str]] = frozenset({
    "USDT", "USDC", "USD", "BUSD", "USD\u24c8", "M", "MARGINED",
    "PERPETUAL", "CONTRACT", "CONTRACTS", "FUTURES", "BINANCE",
    "WILL", "DELIST", "AND", "THE", "ON", "OF", "COIN",
})
_SENTENCE_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[.!?;]+")
_NON_TICKER_WORDS: Final[frozenset[str]] = frozenset({
    "A", "AN", "THE", "AND", "FOR", "ALL", "ANY", "ARE", "BUT", "NOT", "YOU", "YOUR",
    "WILL", "WITH", "FROM", "THIS", "THAT", "HAVE", "HAS", "PLEASE", "NOTE", "TIME",
    "DATE", "MARGIN", "MARGINED", "SETTLEMENT", "SETTLE", "FUTURES", "FUTURE", "PERPETUAL",
    "CONTRACT", "CONTRACTS", "BINANCE", "DELIST", "DELISTING", "REMOVAL", "TRADING",
    "OPEN", "CLOSE", "AFTER", "BEFORE", "DURING", "UNTIL", "WHEN", "THEN", "THAN",
    "INTO", "OVER", "SUCH", "EACH", "OTHER", "MORE", "ONLY", "ALSO", "MAY", "USD",
    "USDT", "USDC", "BUSD", "USDⓈ", "M", "COIN",
    "IN", "ON", "OF", "TO", "BY", "AS", "AT", "OR", "SO", "UP",
    "OUT", "OFF", "PER", "VIA", "NEW", "END",
})


@dataclass(frozen=True, slots=True)
class DelistingNotice:
    code: str
    title: str
    release_at: pd.Timestamp
    kind: NoticeKind
    symbols: tuple[str, ...]
    release_ms: int | None = None


def default_delisting_evidence_path() -> Path:
    """Return the committed evidence path (``src/core/policy/delisting_announcements.jsonl``)."""
    return Path(__file__).resolve().parent / "policy" / "delisting_announcements.jsonl"


def _release_at(release_ms: int) -> pd.Timestamp:
    whole_seconds = (int(release_ms) + 999) // 1000
    try:
        return pd.Timestamp(whole_seconds, unit="s", tz="UTC")
    except (ValueError, OverflowError) as exc:
        raise DataIntegrityError("release_ms is outside the supported timestamp range") from exc


def _parse_row(record: object, line_no: int, source: str) -> DelistingNotice:
    if not isinstance(record, dict):
        raise DataIntegrityError(f"{source} line {line_no}: record must be a JSON object")
    extra = sorted(set(record) - set(_EVIDENCE_FIELDS))
    missing = sorted(set(_EVIDENCE_FIELDS) - set(record))
    if extra or missing:
        raise DataIntegrityError(
            f"{source} line {line_no}: missing or extra field (missing={missing}, extra={extra})",
        )
    code = record["code"]
    title = record["title"]
    if not isinstance(code, str) or not code.strip():
        raise DataIntegrityError(f"{source} line {line_no}: code must be a non-empty string")
    if not isinstance(title, str) or not title.strip():
        raise DataIntegrityError(f"{source} line {line_no}: title must be a non-empty string")
    release_ms = record["release_ms"]
    if isinstance(release_ms, bool) or not isinstance(release_ms, int) or release_ms < 0:
        raise DataIntegrityError(f"{source} line {line_no}: release_ms must be a non-negative int")
    kind = record["kind"]
    if not isinstance(kind, str) or kind not in _VALID_KINDS:
        raise DataIntegrityError(f"{source} line {line_no}: unknown kind {kind!r}")
    symbols = record["symbols"]
    if not isinstance(symbols, list) or any(
        not isinstance(item, str) or not item or item != item.upper() for item in symbols
    ):
        raise DataIntegrityError(f"{source} line {line_no}: symbols must be an upper-case list")
    if tuple(symbols) != tuple(sorted(set(symbols))):
        raise DataIntegrityError(f"{source} line {line_no}: symbols must be sorted and de-duplicated")
    if "USDT" in symbols or "USDC" in symbols:
        raise DataIntegrityError(f"{source} line {line_no}: symbols must not contain bare quote assets")
    collected_at = record["collected_at"]
    if not isinstance(collected_at, str) or not collected_at.strip():
        raise DataIntegrityError(f"{source} line {line_no}: collected_at must be a non-empty string")
    try:
        collected = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
        if collected.utcoffset() != timedelta(0):
            raise ValueError("collected_at must be UTC")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"{source} line {line_no}: collected_at is not a TZ-aware timestamp") from exc
    return DelistingNotice(
        code=code,
        title=title,
        release_at=_release_at(release_ms),
        kind=cast(NoticeKind, kind),
        symbols=tuple(symbols),
        release_ms=release_ms,
    )


def parse_delisting_evidence(raw: bytes, *, source: str) -> tuple[DelistingNotice, ...]:
    """Parse and validate evidence bytes, enforcing append-only order by ``(release_ms, code)``."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(f"{source}: file must be UTF-8") from exc
    notices: list[DelistingNotice] = []
    order: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"{source} line {line_no}: malformed JSON") from exc
        notice = _parse_row(record, line_no, source)
        release_ms = record["release_ms"]
        order.append((release_ms, notice.code))
        notices.append(notice)
    seen: set[str] = set()
    for notice in notices:
        if notice.code in seen:
            raise DataIntegrityError(f"{source}: duplicate code {notice.code!r}")
        seen.add(notice.code)
    if order != sorted(order):
        raise DataIntegrityError(f"{source}: lines must be sorted by (release_ms, code)")
    return tuple(notices)


def load_delisting_notices(path: Path | None = None) -> tuple[DelistingNotice, ...]:
    """Parse the committed evidence file; raises DataIntegrityError on malformed, duplicate, or unsorted lines."""
    target = default_delisting_evidence_path() if path is None else path
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise DataIntegrityError(f"delisting evidence unreadable: {target}") from exc
    return parse_delisting_evidence(raw, source=str(target))


def classify_notice(title: str, body_text: str) -> NoticeKind:
    """Notice kind from title and body text."""
    lowered_title = title.lower()
    if "postpon" in lowered_title:
        return "postponed"
    head = body_text[:2000].lower()
    combined = f"{lowered_title}\n{head}"
    if _ACTION_PATTERN.search(combined) and _CONTEXT_PATTERN.search(combined):
        return "delist"
    return "other"


def _base_tickers(text: str) -> list[str]:
    found: list[str] = []
    for match in _SPACED_SYMBOL_PATTERN.finditer(text):
        ticker = match.group(1)
        if ticker not in _NON_TICKER_WORDS:
            found.append(f"{ticker}USDT")
    return found


def _ticker_list_symbols(text: str) -> list[str]:
    found: list[str] = []
    for pattern in (_TICKER_LIST_BEFORE_PATTERN, _TICKER_LIST_AFTER_PATTERN):
        for match in pattern.finditer(text):
            for raw in _TICKER_LIST_SPLIT_PATTERN.split(match.group(1)):
                ticker = raw.strip().strip(",& ")
                if not ticker or ticker in _TICKER_STOP_WORDS:
                    continue
                if _FULL_SYMBOL_RE.match(ticker):
                    found.append(ticker)
                    continue
                found.append(f"{ticker}USDT")
    return found


def extract_contract_symbols(title: str, body_text: str) -> tuple[str, ...]:
    """Contract symbols a notice names, sorted and de-duplicated."""
    combined = f"{title}\n{body_text}"
    symbols = set(_DIRECT_SYMBOL_PATTERN.findall(combined))
    if classify_notice(title, body_text) == "delist":
        symbols.update(_base_tickers(title))
        symbols.update(_ticker_list_symbols(title))
        for sentence in _SENTENCE_SPLIT_PATTERN.split(body_text):
            lowered = sentence.lower()
            if ("settle" in lowered or "delist" in lowered) and "contract" in lowered:
                symbols.update(_base_tickers(sentence))
                symbols.update(_ticker_list_symbols(sentence))
    symbols.discard("USDT")
    symbols.discard("USDC")
    return tuple(sorted(symbols))


def resolve_announcement(
    symbol: str, last_trade_at: pd.Timestamp, notices: Sequence[DelistingNotice],
) -> DelistingNotice | None:
    """Earliest qualifying notice that announced the delisting of ``symbol``, or None."""
    if pd.isna(last_trade_at) or last_trade_at.tzinfo is None:
        raise DataIntegrityError("last_trade_at must be a timezone-aware timestamp")
    earliest = last_trade_at - _RESOLVE_WINDOW
    candidates = [
        notice
        for notice in notices
        if notice.kind == "delist" and symbol in notice.symbols and earliest <= notice.release_at <= last_trade_at
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda notice: (notice.release_at, notice.code))
