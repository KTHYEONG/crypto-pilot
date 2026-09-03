"""Tagged log emission and optional JSONL sidecar streams for MHS stages.

``StageTelemetry`` owns both structured log lines and the JSONL sidecar
files.  It absorbs the ``_StageRecorder`` resource-measurement collection
from ``resources.py`` without changing the ``MhsResourceMeasurement`` records.

All telemetry is observational only (I-OBSERVE): exceptions are swallowed,
and reports produced with sidecars ON and OFF satisfy I-IDENTITY.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path

import psutil

from src.common.logging import LOG_DIR, setup_logger

_MAX_SEQUENCE_ITEMS = 5


class Tag(StrEnum):
    """Closed 4-member tag set for MHS telemetry (rules/logging.md §3)."""

    SYS = "SYS"
    DATA = "DATA"
    ALGO = "ALGO"
    EVAL = "EVAL"


# Closed set of the 7 sidecar stream names. One stream per stage; a stage
# may not write to another stage's stream.
MHS_DEBUG_STREAMS: tuple[str, ...] = (
    "panel",
    "horizon_selection",
    "book_weights",
    "committee_weights",
    "diagnostics",
    "replay",
    "folds",
)


class _ResourceMeasurement:
    """Lightweight resource sample (replaces MhsResourceMeasurement import)."""

    __slots__ = (
        "active_symbols",
        "elapsed_ms",
        "fill_count",
        "grid_bars",
        "n_symbols",
        "peak_rss_bytes",
        "rss_bytes",
        "stage",
        "window_end",
        "window_start",
    )

    def __init__(
        self,
        stage: str,
        elapsed_ms: int,
        rss_bytes: int,
        *,
        grid_bars: int | None = None,
        n_symbols: int | None = None,
        fill_count: int | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
        active_symbols: int | None = None,
        peak_rss_bytes: int | None = None,
    ) -> None:
        self.stage = stage
        self.elapsed_ms = elapsed_ms
        self.rss_bytes = rss_bytes
        self.grid_bars = grid_bars
        self.n_symbols = n_symbols
        self.fill_count = fill_count
        self.window_start = window_start
        self.window_end = window_end
        self.active_symbols = active_symbols
        self.peak_rss_bytes = peak_rss_bytes

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "stage": self.stage,
            "elapsed_ms": self.elapsed_ms,
            "rss_bytes": self.rss_bytes,
        }
        for k in ("grid_bars", "n_symbols", "fill_count", "window_start",
                   "window_end", "active_symbols", "peak_rss_bytes"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def _current_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return -1


def _format_value(v: object) -> str:
    """Format a single value for log emission."""
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        if len(v) > _MAX_SEQUENCE_ITEMS:
            items = ", ".join(str(x) for x in v[:_MAX_SEQUENCE_ITEMS])
            return f"({items}) truncated={len(v) - _MAX_SEQUENCE_ITEMS}"
        return "(" + ", ".join(str(x) for x in v) + ")"
    return str(v)


class StageTelemetry:
    """Tagged log lines + optional JSONL sidecars.  Computation is never affected (I-OBSERVE)."""

    def __init__(
        self,
        *,
        log_run: bool = True,
        debug_streams: bool = False,
        streams_root: Path | None = None,
    ) -> None:
        self._logger = setup_logger("MhsTelemetry")
        self._log_run = log_run
        self._debug_streams = debug_streams
        self._streams_root = streams_root or (LOG_DIR / "mhs")
        self._records: list[_ResourceMeasurement] = []
        self._last = time.perf_counter()
        self._peak_rss = -1
        if self._debug_streams:
            self._streams_root.mkdir(parents=True, exist_ok=True)

    @property
    def records(self) -> tuple[_ResourceMeasurement, ...]:
        return tuple(self._records)

    def log(self, tag: Tag, stage: str, **fields: object) -> None:
        """Emit ``[TAG] stage=<stage> k=v ...`` with floats at 3dp."""
        if not isinstance(tag, Tag):
            raise ValueError(f"tag must be a Tag member, got {tag!r}")
        parts = [f"[{tag}]", f"stage={stage}"]
        for k, v in fields.items():
            parts.append(f"{k}={_format_value(v)}")
        self._logger.info(" ".join(parts))

    def stream(self, name: str, rows: Iterable[Mapping[str, object]]) -> None:
        """Append full-precision JSONL rows to ``logs/mhs/<name>.jsonl``.

        No-op unless sidecars are enabled.  Any exception is swallowed (I-OBSERVE).
        """
        if not self._debug_streams:
            return
        try:
            path = self._streams_root / f"{name}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        except Exception:  # noqa: BLE001 - observational; never break the caller
            return

    def record(
        self,
        stage: str,
        *,
        grid_bars: int | None = None,
        n_symbols: int | None = None,
        fill_count: int | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
        active_symbols: int | None = None,
    ) -> _ResourceMeasurement:
        """Record a resource measurement and emit a [SYS] log line."""
        now = time.perf_counter()
        elapsed_ms = int((now - self._last) * 1000)
        self._last = now
        rss = _current_rss_bytes()
        self._peak_rss = max(self._peak_rss, rss)
        m = _ResourceMeasurement(
            stage=stage,
            elapsed_ms=elapsed_ms,
            rss_bytes=rss,
            grid_bars=grid_bars,
            n_symbols=n_symbols,
            fill_count=fill_count,
            window_start=window_start,
            window_end=window_end,
            active_symbols=active_symbols,
            peak_rss_bytes=self._peak_rss,
        )
        self._records.append(m)
        if self._log_run:
            self._logger.info(
                "[SYS] stage=%s rss=%d elapsed_ms=%d",
                stage, rss, elapsed_ms,
            )
        return m

    def absorb(self, records: tuple[_ResourceMeasurement, ...]) -> None:
        """Merge frozen records (e.g. from a book subprocess) into this recorder."""
        if not records:
            return
        self._records.extend(records)
        self._peak_rss = max(self._peak_rss, max(getattr(r, "peak_rss_bytes", 0) or 0 for r in records))
        self._last = time.perf_counter()
