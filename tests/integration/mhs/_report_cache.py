"""Shared MHS diagnostic report cache for the integration suite.

Process-local memo of full ``run_mhs_diagnostic`` executions keyed by
``(patch profile, request digest, market fingerprint)``. A pytest-xdist worker
is a separate process, so every consumer of a key must run on one worker via
``xdist_group`` + ``--dist loadgroup``; without that the cache stays correct
but shares less.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.mhs.contracts import MhsDiagnosticRequest, MhsHorizonDiagnosticReport

MHS_SYNTHETIC_DEFAULT_GROUP: Final[str] = "mhs_synthetic_default"
MHS_GOLDEN_BASELINE_GROUP: Final[str] = "mhs_golden_baseline"
MHS_LATE_MARKET_GROUP: Final[str] = "mhs_late_market"
GOLDEN_SHARED_NAMES: Final[frozenset[str]] = frozenset({"baseline"})


@dataclass(frozen=True, slots=True)
class DiagnosticRunProfile:
    """Every module-global patch that can change a cached diagnostic's report.

    The cache owns applying these around the run, so a profile plus the request and the
    market bytes fully determine the report; resource-only patches (virtual-memory pin,
    fork-worker cap) are deliberately excluded because they alter only fields no shared
    consumer asserts on (``worker_plan``, ``resource_measurements`` values).

    Attributes:
        bootstrap_replicates: Value for ``src.mhs.statistics._BOOTSTRAP_REPLICATES``.
        bootstrap_mean_block: Value for ``src.mhs.statistics._BOOTSTRAP_MEAN_BLOCK``.
        bootstrap_seed: Value for ``src.mhs.statistics._BOOTSTRAP_SEED``.
        trials_attempted_pin: Return value pinned on
            ``src.mhs.pipeline.stages.fold.derive_trials_attempted``; ``None`` leaves it untouched.
        observe_wiring: Install the pass-through wiring spies and capture observations.
    """

    bootstrap_replicates: int
    bootstrap_mean_block: int
    bootstrap_seed: int
    trials_attempted_pin: tuple[int, str] | None
    observe_wiring: bool


SYNTHETIC_DEFAULT_PROFILE: Final[DiagnosticRunProfile] = DiagnosticRunProfile(20, 24, 20260807, None, True)
GOLDEN_PROFILE: Final[DiagnosticRunProfile] = DiagnosticRunProfile(20, 24, 20260807, (80, "constant_plus_ledger"), False)


@dataclass(frozen=True, slots=True)
class DiagnosticRunSpec:
    """One requested diagnostic execution: the request plus its patch profile.

    ``request.data_root`` names the market directory the run reads and the directory the
    path patches point at; it is excluded from the cache key (the market fingerprint
    replaces it) so byte-identical markets written by different modules share one run.
    """

    request: MhsDiagnosticRequest
    profile: DiagnosticRunProfile


@dataclass(frozen=True, slots=True)
class DiagnosticRunKey:
    """Cache identity: ``(profile, request digest, market fingerprint)``."""

    profile: DiagnosticRunProfile
    request_digest: str
    market_fingerprint: str


def request_digest(request: MhsDiagnosticRequest) -> str:
    """Hex sha256 of every request field except ``data_root``.

    Fields are serialised as sorted ``(name, repr(value))`` pairs; all 54 fields are scalars
    at HEAD, so ``repr`` is process-stable. ``data_root`` is excluded because the market
    fingerprint pins market content and reports carry no ``data_root``-derived value.
    """
    pairs = sorted(
        (f.name, repr(getattr(request, f.name)))
        for f in dataclasses.fields(request)
        if f.name != "data_root"
    )
    h = hashlib.sha256()
    for name, value_repr in pairs:
        name_b = name.encode("utf-8")
        h.update(len(name_b).to_bytes(4, "big"))
        h.update(name_b)
        val_b = value_repr.encode("utf-8")
        h.update(len(val_b).to_bytes(8, "big"))
        h.update(val_b)
    return h.hexdigest()


def market_fingerprint(root: Path) -> str:
    """Hex sha256 over the sorted relative POSIX paths and bytes of every file under ``root``.

    Raises:
        FileNotFoundError: ``root`` does not exist (a cache key is never built for an
            absent market).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"market root does not exist: {root}")
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    h = hashlib.sha256()
    for path in files:
        rel_b = path.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel_b).to_bytes(4, "big"))
        h.update(rel_b)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(len(chunk).to_bytes(8, "big"))
                h.update(chunk)
    return h.hexdigest()


class _DigestFeed:
    def __init__(self) -> None:
        self.h = hashlib.sha256()

    def feed(self, tag: str, data: bytes) -> None:
        tag_b = tag.encode("utf-8")
        self.h.update(len(tag_b).to_bytes(4, "big"))
        self.h.update(tag_b)
        self.h.update(len(data).to_bytes(8, "big"))
        self.h.update(data)


def _feed_value(feed: _DigestFeed, obj: object, _depth: int = 0) -> None:
    if obj is None:
        feed.feed("none", b"")
        return
    if isinstance(obj, bool):
        feed.feed("bool", repr(obj).encode())
        return
    if isinstance(obj, int):
        feed.feed("int", repr(obj).encode())
        return
    if isinstance(obj, float):
        feed.feed("float", repr(obj).encode())
        return
    if isinstance(obj, str):
        feed.feed("str", obj.encode("utf-8"))
        return
    if isinstance(obj, bytes):
        feed.feed("bytes", obj)
        return
    if isinstance(obj, pd.DataFrame):
        feed.feed("df.type", b"DataFrame")
        feed.feed("df.shape", repr(obj.shape).encode())
        feed.feed("df.dtypes", repr([str(t) for t in obj.dtypes]).encode())
        _feed_value(feed, obj.columns, _depth + 1)
        _feed_value(feed, obj.index, _depth + 1)
        for position in range(len(obj.columns)):
            feed.feed(
                "df.col",
                np.ascontiguousarray(pd.util.hash_pandas_object(obj.iloc[:, position], index=False).to_numpy()).tobytes(),
            )
        return
    if isinstance(obj, pd.Series):
        feed.feed("series.type", b"Series")
        feed.feed("series.shape", repr(obj.shape).encode())
        feed.feed("series.dtype", str(obj.dtype).encode())
        feed.feed("series.name", repr(obj.name).encode())
        _feed_value(feed, obj.index, _depth + 1)
        feed.feed(
            "series",
            np.ascontiguousarray(pd.util.hash_pandas_object(obj, index=False).to_numpy()).tobytes(),
        )
        return
    if isinstance(obj, pd.Index):
        feed.feed("index.type", type(obj).__qualname__.encode())
        feed.feed("index.shape", repr(obj.shape).encode())
        feed.feed("index.dtype", str(obj.dtype).encode())
        _feed_value(feed, tuple(obj.names), _depth + 1)
        feed.feed("index.freq", repr(getattr(obj, "freqstr", None)).encode())
        if isinstance(obj, pd.MultiIndex):
            _feed_value(feed, tuple(obj.levels), _depth + 1)
            _feed_value(feed, tuple(obj.codes), _depth + 1)
        if isinstance(obj, pd.CategoricalIndex):
            _feed_value(feed, obj.categories, _depth + 1)
            _feed_value(feed, obj.ordered, _depth + 1)
        feed.feed(
            "index",
            np.ascontiguousarray(pd.util.hash_pandas_object(obj, index=False).to_numpy()).tobytes(),
        )
        return
    if isinstance(obj, np.ndarray):
        feed.feed("ndarray.dtype", str(obj.dtype).encode())
        feed.feed("ndarray.shape", repr(obj.shape).encode())
        if obj.dtype == object:
            feed.feed("ndarray.object", repr(obj.tolist()).encode("utf-8"))
        else:
            feed.feed("ndarray", np.ascontiguousarray(obj).tobytes())
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        feed.feed("dataclass.type", f"{type(obj).__module__}.{type(obj).__qualname__}".encode())
        for f in dataclasses.fields(obj):
            feed.feed("field.name", f.name.encode())
            _feed_value(feed, getattr(obj, f.name), _depth + 1)
        return
    if isinstance(obj, Mapping):
        feed.feed("mapping.type", type(obj).__qualname__.encode())
        feed.feed("mapping.len", repr(len(obj)).encode())
        for key in sorted(obj.keys(), key=repr):
            _feed_value(feed, key, _depth + 1)
            _feed_value(feed, obj[key], _depth + 1)
        return
    if isinstance(obj, (set, frozenset)):
        feed.feed("set.type", type(obj).__qualname__.encode())
        members = sorted(obj, key=repr)
        feed.feed("set.len", repr(len(members)).encode())
        for member in members:
            _feed_value(feed, member, _depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        feed.feed("seq.type", type(obj).__qualname__.encode())
        feed.feed("seq.len", repr(len(obj)).encode())
        for item in obj:
            _feed_value(feed, item, _depth + 1)
        return
    if isinstance(obj, os.PathLike):
        feed.feed("pathlike", os.fspath(obj).encode("utf-8"))
        return
    feed.feed(f"repr:{type(obj).__module__}.{type(obj).__qualname__}", repr(obj).encode("utf-8"))


def report_state_digest(report: object) -> str:
    """Hex sha256 of the report's full reachable value state, insensitive to pandas caches.

    Traverses dataclass fields (in declaration order, tagged by qualified type name),
    mappings (keys sorted by ``repr``), sets/frozensets (members sorted by ``repr``),
    lists/tuples (in order), pandas ``DataFrame``/``Series``/``Index`` (type, shape, dtypes,
    column labels and name, then ``pd.util.hash_pandas_object(..., index=False)`` bytes for
    the index and for each column/series), numpy arrays (dtype, shape and bytes; object
    arrays via ``repr(tolist())``) and any other leaf via ``repr``. Every fed chunk is
    length-prefixed and tagged so adjacent values cannot alias.

    Used as the cache mutation guard: equal before and after any read-only consumer
    (``to_payload``, ``persist_mhs_horizon_diagnostic_report``, golden digest/summary),
    different after any in-place value change of a reachable frame, series, array or
    container.
    """
    feed = _DigestFeed()
    _feed_value(feed, report)
    return feed.h.hexdigest()


def _mentions_value(obj: object, needle: str, _depth: int = 0) -> bool:
    if not needle:
        return False
    if isinstance(obj, str):
        return needle in obj
    if isinstance(obj, os.PathLike):
        return needle in os.fspath(obj)
    if isinstance(obj, pd.DataFrame):
        return _mentions_value(obj.columns, needle, _depth + 1) or _mentions_value(obj.index, needle, _depth + 1)
    if isinstance(obj, pd.Series):
        return _mentions_value(obj.name, needle, _depth + 1) or _mentions_value(obj.index, needle, _depth + 1)
    if isinstance(obj, pd.Index):
        if _mentions_value(tuple(obj.names), needle, _depth + 1):
            return True
        if isinstance(obj, pd.MultiIndex):
            return any(_mentions_value(level, needle, _depth + 1) for level in obj.levels)
        if isinstance(obj, pd.CategoricalIndex):
            return _mentions_value(obj.categories, needle, _depth + 1)
        if pd.api.types.is_object_dtype(obj.dtype) or pd.api.types.is_string_dtype(obj.dtype):
            return any(_mentions_value(label, needle, _depth + 1) for label in obj)
        return False
    if isinstance(obj, np.ndarray):
        return False
    if obj is None or isinstance(obj, (bool, int, float, bytes)):
        if isinstance(obj, bytes):
            return needle in obj.decode("utf-8", errors="ignore")
        return False
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return any(_mentions_value(getattr(obj, f.name), needle, _depth + 1) for f in dataclasses.fields(obj))
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if _mentions_value(key, needle, _depth + 1):
                return True
            if _mentions_value(value, needle, _depth + 1):
                return True
        return False
    if isinstance(obj, (set, frozenset)):
        return any(_mentions_value(m, needle, _depth + 1) for m in obj)
    if isinstance(obj, (list, tuple)):
        return any(_mentions_value(v, needle, _depth + 1) for v in obj)
    return False


def report_mentions(report: object, needle: str) -> bool:
    """Whether any string-like leaf reachable from ``report`` contains ``needle``.

    Uses the ``report_state_digest`` traversal; inspects ``str``/``os.PathLike`` leaves,
    mapping keys, dataclass string fields, and pandas ``name``/column/index labels (not the
    values of object-dtype columns). Guards the ``data_root``-free key: the HEAD probe found no
    ``data_root`` in ``to_payload()`` or ``build_report_summary``.
    """
    if not needle:
        return False
    return _mentions_value(report, needle)


@dataclass(frozen=True, slots=True)
class WiringObservations:
    """Immutable captures from the pass-through spies of exactly one pipeline execution.

    Attributes:
        ema_spans: ``spec.band.name`` -> every ``ema_span`` passed to
            ``src.mhs.evaluation.books._book_weights`` (parent and fork children), in arrival order.
        regime_callers: Immediate caller function names of
            ``src.mhs.scaling._regime_cash_scale`` and ``src.mhs.scaling.regime_cash_scale_1h``.
        deadband_callers: Immediate caller function names of
            ``src.mhs.scaling._apply_rebalance_deadband``.
        replay_entry: ``{"log_close_released": bool, "signal_empty": bool}`` recorded at the
            last entry into ``src.mhs.pipeline.stages.replay.run_replays``; empty when never entered.
    """

    ema_spans: Mapping[str, tuple[int | None, ...]]
    regime_callers: tuple[str, ...]
    deadband_callers: tuple[str, ...]
    replay_entry: Mapping[str, bool]

    def calibration_captures(self) -> Mapping[str, object]:
        """Read-only view with the HEAD ``calibrated_report`` capture keys.

        Returns:
            Mapping with keys ``"ema_spans"``, ``"regime_callers"``, ``"deadband_callers"``
            whose values support the operations the existing assertions use (key lookup,
            truthiness, iteration, ``.count``).
        """
        return MappingProxyType({
            "ema_spans": self.ema_spans,
            "regime_callers": self.regime_callers,
            "deadband_callers": self.deadband_callers,
        })


@dataclass(frozen=True, slots=True)
class CachedDiagnostic:
    """One cached diagnostic execution.

    Attributes:
        report: The report returned by ``run_mhs_diagnostic``; shared by reference, never copied.
        observations: Wiring captures when the profile observed wiring, else ``None``.
        state_digest: ``report_state_digest(report)`` taken right after the run.
    """

    report: MhsHorizonDiagnosticReport
    observations: WiringObservations | None
    state_digest: str


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Monotonic per-process counters for run-count verification."""

    hits: int
    misses: int
    evictions: int


@contextmanager
def _observation_events() -> Iterator[tuple[Callable[[tuple[Any, ...]], None], list[tuple[Any, ...]]]]:
    import multiprocessing
    from threading import Thread

    events: list[tuple[Any, ...]] = []
    try:
        manager = multiprocessing.Manager()
    except (EOFError, OSError, PermissionError):
        manager = None
    if manager is not None:
        try:
            shared = manager.list()
            yield shared.append, events
            events.extend(shared)
        finally:
            manager.shutdown()
        return

    queue = multiprocessing.Queue()

    def drain() -> None:
        while (event := queue.get()) is not None:
            events.append(event)

    reader = Thread(target=drain, daemon=True)
    reader.start()
    try:
        yield queue.put, events
    finally:
        # Joined fork workers have flushed their feeders before this sentinel.
        queue.put(None)
        reader.join()
        queue.close()
        queue.join_thread()


class DiagnosticReportCache:
    """Process-local memo of full MHS diagnostics shared by several integration tests.

    A pytest-xdist worker is a separate process, so sharing requires every consumer of a
    key to run on one worker (``xdist_group`` + ``--dist loadgroup``); without that the
    cache is still correct, merely less effective.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[DiagnosticRunProfile, str], tuple[str, CachedDiagnostic]] = {}
        self._failures: dict[DiagnosticRunKey, BaseException] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def stats(self) -> CacheStats:
        """Current hit/miss/eviction counters."""
        return CacheStats(self._hits, self._misses, self._evictions)

    def clear(self) -> None:
        """Drop every entry and memoised failure (session teardown; releases report memory)."""
        self._entries.clear()
        self._failures.clear()

    @contextmanager
    def lease(self, spec: DiagnosticRunSpec, *, consumer: str) -> Iterator[CachedDiagnostic]:
        """Yield the cached diagnostic for ``spec``, executing it once on first use.

        On a miss the run executes ``src.mhs.pipeline.orchestrator.run_mhs_diagnostic``
        (resolved at call time) with these module globals set for exactly the duration of
        the run and restored in ``finally``: ``src.mhs.marks.funding_path`` ->
        ``<data_root>/funding/<SYM>.parquet``,
        ``src.market_data.services.futures_collection._mark_price_path`` ->
        ``<data_root>/markPriceKlines/<tf>/<SYM>.parquet``, the three
        ``src.mhs.statistics._BOOTSTRAP_*`` constants, the optional
        ``derive_trials_attempted`` pin, and the wiring spies when observed.

        Args:
            spec: Request plus patch profile; ``spec.request.data_root`` must be set.
            consumer: Pytest node id of the leasing test, used in guard messages.

        Yields:
            The shared ``CachedDiagnostic``.

        Raises:
            AssertionError: The entry's state digest differs from its post-run digest at
                lease entry or at lease exit (mutation by ``consumer`` or by an earlier
                lessee); the entry is evicted first so the next lease recomputes.
            AssertionError: ``report_mentions(report, spec.request.data_root)`` is True
                (cross-directory sharing would be unsound); nothing is cached.
            RuntimeError: An earlier execution of the same key raised; chained to the
                original exception (memoised failure, no re-execution).
            ValueError: ``spec.request.data_root`` is ``None``.
        """
        data_root = spec.request.data_root
        if data_root is None:
            raise ValueError("DiagnosticRunSpec.request.data_root must be set")
        root_str = str(data_root)
        fingerprint = market_fingerprint(Path(root_str))
        digest = request_digest(spec.request)
        key = DiagnosticRunKey(spec.profile, digest, fingerprint)
        pair = (spec.profile, digest)
        if key in self._failures:
            original = self._failures[key]
            raise RuntimeError(f"cached diagnostic failure for {key}") from original
        existing = self._entries.get(pair)
        if existing is not None:
            old_fingerprint, entry = existing
            if old_fingerprint != fingerprint:
                del self._entries[pair]
                self._evictions += 1
            else:
                if report_state_digest(entry.report) != entry.state_digest:
                    del self._entries[pair]
                    self._evictions += 1
                    raise AssertionError(f"MHS report cache: entry mutated before lease by {consumer}")
                if report_mentions(entry.report, root_str):
                    del self._entries[pair]
                    self._evictions += 1
                    raise AssertionError(f"MHS report cache: report mentions data_root for {consumer}")
                self._hits += 1
                try:
                    yield entry
                finally:
                    if report_state_digest(entry.report) != entry.state_digest:
                        self._entries.pop(pair, None)
                        self._evictions += 1
                        raise AssertionError(f"MHS report cache: entry mutated by {consumer}")
                return
        self._misses += 1
        try:
            report, observations = self._execute(spec)
        except BaseException as exc:
            self._failures[key] = exc
            raise
        if report_mentions(report, root_str):
            raise AssertionError(f"MHS report cache: report mentions data_root for {consumer}")
        state = report_state_digest(report)
        entry = CachedDiagnostic(report, observations, state)
        self._entries[pair] = (fingerprint, entry)
        try:
            yield entry
        finally:
            if report_state_digest(entry.report) != entry.state_digest:
                self._entries.pop(pair, None)
                self._evictions += 1
                raise AssertionError(f"MHS report cache: entry mutated by {consumer}")

    def _execute(self, spec: DiagnosticRunSpec) -> tuple[MhsHorizonDiagnosticReport, WiringObservations | None]:
        import src.market_data.services.futures_collection as fc
        import src.mhs.marks as marks
        import src.mhs.statistics as statistics

        root_str = str(spec.request.data_root)
        targets = [
            (marks, "funding_path", lambda sym: Path(root_str) / "funding" / f"{sym}.parquet"),
            (fc, "_mark_price_path", lambda symbol, timeframe: Path(root_str) / "markPriceKlines" / timeframe / f"{symbol}.parquet"),
            (statistics, "_BOOTSTRAP_REPLICATES", spec.profile.bootstrap_replicates),
            (statistics, "_BOOTSTRAP_MEAN_BLOCK", spec.profile.bootstrap_mean_block),
            (statistics, "_BOOTSTRAP_SEED", spec.profile.bootstrap_seed),
        ]
        if spec.profile.trials_attempted_pin is not None:
            import src.mhs.pipeline.stages.fold as fold_stage

            pin = spec.profile.trials_attempted_pin
            targets.append((fold_stage, "derive_trials_attempted", lambda *args, **kwargs: pin))
        for module, name, _value in targets:
            getattr(module, name)
        with ExitStack() as patches:
            for module, name, value in targets:
                patches.enter_context(patch.object(module, name, value))
            if not spec.profile.observe_wiring:
                from src.mhs.pipeline import orchestrator

                return orchestrator.run_mhs_diagnostic(spec.request), None
            return self._execute_observed(spec)

    def _execute_observed(self, spec: DiagnosticRunSpec) -> tuple[Any, WiringObservations | None]:
        from src.mhs import scaling
        from src.mhs.evaluation import books as eval_books
        from src.mhs.pipeline.stages import replay as replay_stage

        _ = eval_books._book_weights
        _ = scaling._regime_cash_scale
        _ = scaling.regime_cash_scale_1h
        _ = scaling._apply_rebalance_deadband
        _ = replay_stage.run_replays
        real_book_weights = eval_books._book_weights
        real_regime = scaling._regime_cash_scale
        real_helper = scaling.regime_cash_scale_1h
        real_deadband = scaling._apply_rebalance_deadband
        real_replays = replay_stage.run_replays
        replay_state: dict[str, object] = {}

        def _book_weights(log_close: pd.DataFrame, eligible: pd.DataFrame, spec: Any, step_grid: Any, ema_span: int | None = None) -> Any:
            record(("ema", spec.band.name, ema_span))
            return real_book_weights(log_close, eligible, spec, step_grid, ema_span=ema_span)

        def _regime(*args: Any, **kwargs: Any) -> Any:
            caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
            record(("regime", caller))
            return real_regime(*args, **kwargs)

        def _helper(*args: Any, **kwargs: Any) -> Any:
            caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
            record(("regime", caller))
            return real_helper(*args, **kwargs)

        def _deadband(*args: Any, **kwargs: Any) -> Any:
            caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
            record(("deadband", caller))
            return real_deadband(*args, **kwargs)

        def _spy_replays(ctx: Any, telemetry: Any) -> Any:
            replay_state["log_close_released"] = "log_close" not in vars(ctx)
            replay_state["signal_empty"] = bool(ctx.signal_48h.empty)
            return real_replays(ctx, telemetry)

        with _observation_events() as (record, events), ExitStack() as patches:
            for module, name, spy in (
                (eval_books, "_book_weights", _book_weights),
                (scaling, "_regime_cash_scale", _regime),
                (scaling, "regime_cash_scale_1h", _helper),
                (scaling, "_apply_rebalance_deadband", _deadband),
                (replay_stage, "run_replays", _spy_replays),
            ):
                patches.enter_context(patch.object(module, name, spy))
            from src.mhs.pipeline import orchestrator

            report = orchestrator.run_mhs_diagnostic(spec.request)
        ema_spans: dict[str, list[int | None]] = {}
        regime_callers: list[str] = []
        deadband_callers: list[str] = []
        for kind, value, *extra in events:
            if kind == "ema":
                ema_spans.setdefault(value, []).append(extra[0])
            elif kind == "regime":
                regime_callers.append(value)
            elif kind == "deadband":
                deadband_callers.append(value)
        entry_map: Mapping[str, bool] = MappingProxyType(
            {k: bool(v) for k, v in replay_state.items() if k in ("log_close_released", "signal_empty")}
        )
        observations = WiringObservations(
            ema_spans=MappingProxyType({k: tuple(v) for k, v in ema_spans.items()}),
            regime_callers=tuple(regime_callers),
            deadband_callers=tuple(deadband_callers),
            replay_entry=entry_map,
        )
        return report, observations


class CollectedItem(Protocol):
    """Structural view of ``pytest.Item`` used by collection guards (fakeable in tests)."""

    nodeid: str
    path: Path
    fixturenames: Sequence[str]

    def iter_markers(self, name: str | None = None) -> Iterator[pytest.Mark]: ...


_SHARED_FIXTURES: Final[frozenset[str]] = frozenset(
    {"canonical_report_run", "report", "calibrated_report", "annualization_report"}
)


def report_cache_group_violations(items: Iterable[CollectedItem], suite_root: Path) -> list[str]:
    """List consumers of a shared diagnostic whose ``xdist_group`` would split the share.

    Only items whose ``path`` lies under ``suite_root`` are checked (a sub-directory
    ``conftest.py`` collection hook receives every session item). Expected group per item:
    ``MHS_SYNTHETIC_DEFAULT_GROUP`` when its fixture closure contains any of
    ``canonical_report_run``, ``report``, ``calibrated_report``, ``annualization_report``;
    ``MHS_LATE_MARKET_GROUP`` when it contains ``late_market_report``;
    ``MHS_GOLDEN_BASELINE_GROUP`` when ``item.callspec.params["matrix_market"]`` is in
    ``GOLDEN_SHARED_NAMES`` (``callspec`` read with ``getattr``, absent on unparametrised
    items); otherwise no group.

    Returns:
        One ``"<nodeid>: expected <expected|none>, found <sorted names|none>"`` line per item
        whose set of ``xdist_group`` names (all markers, class and module included) differs
        from the expected singleton (or from empty); also one line per item matching two
        rules. Empty when consistent.
    """
    suite_root = Path(suite_root)
    violations: list[str] = []
    for item in items:
        path = Path(item.path)
        if path != suite_root and suite_root not in path.parents:
            continue
        fixtures = set(item.fixturenames)
        matched: list[str] = []
        if fixtures & _SHARED_FIXTURES:
            matched.append(MHS_SYNTHETIC_DEFAULT_GROUP)
        if "late_market_report" in fixtures:
            matched.append(MHS_LATE_MARKET_GROUP)
        callspec = getattr(item, "callspec", None)
        params = getattr(callspec, "params", None) if callspec is not None else None
        matrix_name = params.get("matrix_market") if isinstance(params, dict) else None
        if matrix_name in GOLDEN_SHARED_NAMES:
            matched.append(MHS_GOLDEN_BASELINE_GROUP)
        expected = matched[0] if len(matched) == 1 else None
        names = sorted(
            {
                mark.args[0]
                for mark in item.iter_markers(name="xdist_group")
                if mark.args and isinstance(mark.args[0], str)
            }
        )
        expected_set = {expected} if expected is not None else set()
        if len(matched) > 1 or set(names) != expected_set:
            expected_str = expected if expected is not None else "none"
            found_str = ",".join(names) if names else "none"
            violations.append(f"{item.nodeid}: expected {expected_str}, found {found_str}")
    return violations
