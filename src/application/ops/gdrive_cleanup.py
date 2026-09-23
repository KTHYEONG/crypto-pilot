"""One-time, evidence-gated removal of superseded crypto-pilot objects from ``gdrive:quant-lake``.

Why: ``rclone copy`` never propagates deletions, so renamed or retired local data lingers
remotely and creates same-name duplicates. Each rule deletes an object only when a verifiable
equal-or-better copy exists, or the data is re-downloadable from Binance Vision. Dry-run is
the default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

DRIVE_REMOTE: str = "gdrive:quant-lake"
LIVE_DATA: str = "live/crypto-pilot/data"
RESEARCH_FUTURES: str = "research/crypto-pilot-full/futures"
LEGACY_ARCHIVE: str = "live/crypto-pilot/data/archive/legacy_horizon_v2_20260902_20260920"
OCI_SNAPSHOT: str = "_backup/oci-server"
KRX_DATA: str = "live/krx-alpha/data"
LEGACY_FUTURES_SUBDIRS: tuple[str, ...] = ("ohlcv/1h", "markPriceKlines/1h", "funding")
LIVE_STATE_KEEP: frozenset[str] = frozenset({"live_daemon_heartbeat.json", "live_daemon_last_run.json", "non_crypto_symbols.json"})
VISION_BUCKET_LISTING_URL: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
REPORT_DIR: Path = Path("logs/ops")

CleanupRule = Literal["futures_legacy", "legacy_state", "plaintext_run_artifact", "manual_debug_backup", "oci_server_snapshot"]

RULE_ORDER: tuple[CleanupRule, ...] = (
    "futures_legacy",
    "legacy_state",
    "plaintext_run_artifact",
    "manual_debug_backup",
    "oci_server_snapshot",
)

_FUTURES_DATASETS: dict[str, str] = {
    "ohlcv/1h": "klines",
    "markPriceKlines/1h": "markPriceKlines",
    "funding": "fundingRate",
}

_FUTURES_PREFIX: str = LIVE_DATA + "/futures/"
_STATE_PREFIX: str = LIVE_DATA + "/state/"
_RUNS_PREFIX: str = LIVE_DATA + "/state/runs/"
_ARCHIVE_STATE_PREFIX: str = LEGACY_ARCHIVE + "/state/"

_RULE_RMDIR_ROOTS: dict[CleanupRule, str] = {
    "futures_legacy": LIVE_DATA + "/futures",
    "legacy_state": LIVE_DATA + "/state",
    "plaintext_run_artifact": LIVE_DATA + "/state/runs",
    "manual_debug_backup": LIVE_DATA + "/state/runs",
    "oci_server_snapshot": OCI_SNAPSHOT,
}

_LISTED_ROOTS: tuple[str, ...] = (LIVE_DATA, RESEARCH_FUTURES, OCI_SNAPSHOT, KRX_DATA)

_BOUND_SUBPROCESS_RUN = subprocess.run


@dataclass(frozen=True, slots=True)
class RemoteObject:
    path: str  # relative to DRIVE_REMOTE, posix
    size: int


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    rule: CleanupRule
    obj: RemoteObject
    evidence: str  # human-readable proof, e.g. "research:<path> size=<n>" or "vision:<symbol>"


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    candidates: tuple[CleanupCandidate, ...]
    kept: tuple[tuple[RemoteObject, str], ...]  # (object, reason it is not deletable)


def _active_runner(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Honor a test-faked transport when the caller kept the default binding."""
    if runner is _BOUND_SUBPROCESS_RUN:
        return subprocess.run
    return runner


def _futures_subdir(path: str) -> str | None:
    """Return the legacy futures subdir containing ``path``, else None."""
    if not path.startswith(_FUTURES_PREFIX):
        return None
    rest = path[len(_FUTURES_PREFIX):]
    for sub in LEGACY_FUTURES_SUBDIRS:
        if rest.startswith(sub + "/") and len(rest) > len(sub) + 1:
            return sub
    return None


def _is_legacy_state(path: str) -> bool:
    """Whether ``path`` falls under the legacy-state rule scope."""
    if not path.startswith(_STATE_PREFIX) or path.startswith(_RUNS_PREFIX):
        return False
    return path.rsplit("/", 1)[-1] not in LIVE_STATE_KEEP


def _manual_backup_date(dirname: str) -> str | None:
    """Return the YYYYMMDD stamp when ``dirname`` is a manual debug backup dir."""
    prefix = "_manual_debug_backup_"
    if not dirname.startswith(prefix):
        return None
    stamp = dirname[len(prefix):]
    if len(stamp) == 8 and stamp.isdigit():
        return stamp
    return None


def _plaintext_run_parts(path: str) -> tuple[str, str] | None:
    """Split a ``runs/<run>/<name>.parquet`` path, excluding backup dirs and sealed files."""
    if not path.startswith(_RUNS_PREFIX):
        return None
    rest = path[len(_RUNS_PREFIX):]
    parts = rest.split("/")
    if len(parts) != 2 or not parts[0] or _manual_backup_date(parts[0]) is not None:
        return None
    name = parts[1]
    if not name.endswith(".parquet") or name.endswith(".parquet.enc"):
        return None
    return (parts[0], name)


def _scope_rule_of(path: str) -> CleanupRule | None:
    """Attribute an in-scope path to its rule for per-rule kept accounting."""
    if _futures_subdir(path) is not None:
        return "futures_legacy"
    if path.startswith(OCI_SNAPSHOT + "/"):
        return "oci_server_snapshot"
    if _is_legacy_state(path):
        return "legacy_state"
    if _plaintext_run_parts(path) is not None:
        return "plaintext_run_artifact"
    if path.startswith(_RUNS_PREFIX):
        rest = path[len(_RUNS_PREFIX):]
        dirname = rest.split("/", 1)[0]
        if _manual_backup_date(dirname) is not None:
            return "manual_debug_backup"
    return None


def build_cleanup_plan(
    listings: Mapping[str, Sequence[RemoteObject]],
    vision_has_symbol: Callable[[str, str], bool],
) -> CleanupPlan:
    """Classify remote objects into evidence-backed deletion candidates and kept objects.

    Pure function: all remote state arrives via ``listings`` (keyed by the listed root:
    LIVE_DATA, RESEARCH_FUTURES, OCI_SNAPSHOT) and ``vision_has_symbol(dataset, symbol)``.

    Args:
        listings: Recursive file listings per root; paths are relative to DRIVE_REMOTE.
        vision_has_symbol: Whether Binance Vision publishes ``dataset`` ("klines", "markPriceKlines",
            "fundingRate") for ``symbol``.

    Returns:
        Plan whose candidates each carry the evidence that justified deletion; every other object
        under a rule's scope appears in ``kept`` with a reason.

    Raises:
        ValueError: a required root is missing from ``listings``.
    """
    missing = [root for root in (LIVE_DATA, RESEARCH_FUTURES, OCI_SNAPSHOT) if root not in listings]
    if missing:
        raise ValueError(f"missing required listing root(s): {', '.join(missing)}")
    live = list(listings[LIVE_DATA])
    research = list(listings[RESEARCH_FUTURES])
    oci = list(listings[OCI_SNAPSHOT])
    krx = list(listings.get(KRX_DATA, ()))

    research_size: dict[str, int] = {}
    research_prefix = RESEARCH_FUTURES + "/"
    for obj in research:
        if not obj.path.startswith(research_prefix):
            continue
        rel = obj.path[len(research_prefix):]
        if obj.size > research_size.get(rel, -1):
            research_size[rel] = obj.size

    archive_size: dict[str, int] = {}
    for obj in live:
        if not obj.path.startswith(_ARCHIVE_STATE_PREFIX):
            continue
        rel = obj.path[len(_ARCHIVE_STATE_PREFIX):]
        if obj.size > archive_size.get(rel, -1):
            archive_size[rel] = obj.size

    live_paths: set[str] = set()
    live_size: dict[str, int] = {}
    for obj in live:
        live_paths.add(obj.path)
        if obj.size > live_size.get(obj.path, -1):
            live_size[obj.path] = obj.size

    live_by_base: dict[str, list[tuple[str, int]]] = {}
    live_prefix = LIVE_DATA + "/"
    for obj in live:
        if not obj.path.startswith(live_prefix):
            continue
        live_by_base.setdefault(obj.path.rsplit("/", 1)[-1], []).append((obj.path, obj.size))
    krx_by_base: dict[str, list[tuple[str, int]]] = {}
    krx_prefix = KRX_DATA + "/"
    for obj in krx:
        if not obj.path.startswith(krx_prefix):
            continue
        krx_by_base.setdefault(obj.path.rsplit("/", 1)[-1], []).append((obj.path, obj.size))

    runs_entries: list[tuple[str, str, str, int]] = []
    for obj in live:
        if not obj.path.startswith(_RUNS_PREFIX):
            continue
        rest = obj.path[len(_RUNS_PREFIX):]
        if "/" not in rest:
            continue
        dirname, basename = rest.split("/", 1)
        if "/" in basename:
            continue
        runs_entries.append((dirname, basename, obj.path, obj.size))

    candidates: list[CleanupCandidate] = []
    kept: list[tuple[RemoteObject, str]] = []

    for obj in live:
        sub = _futures_subdir(obj.path)
        if sub is not None:
            rest = obj.path[len(_FUTURES_PREFIX):]
            basename = rest.rsplit("/", 1)[-1]
            research_hit = research_size.get(rest)
            if research_hit is not None and research_hit > 0:
                candidates.append(
                    CleanupCandidate("futures_legacy", obj, f"research:{RESEARCH_FUTURES}/{rest} size={research_hit}")
                )
            elif basename.endswith(".coverage.json"):
                candidates.append(CleanupCandidate("futures_legacy", obj, "regenerable:coverage-metadata"))
            elif basename.endswith(".parquet"):
                symbol = basename[: -len(".parquet")]
                dataset = _FUTURES_DATASETS[sub]
                if symbol and vision_has_symbol(dataset, symbol):
                    candidates.append(CleanupCandidate("futures_legacy", obj, f"vision:{symbol}"))
                else:
                    kept.append((obj, "no_equivalent_copy"))
            else:
                kept.append((obj, "no_equivalent_copy"))
            continue
        if _is_legacy_state(obj.path):
            rel = obj.path[len(_STATE_PREFIX):]
            counterpart = f"{LEGACY_ARCHIVE}/state/{rel}"
            asize = archive_size.get(rel)
            if asize is None:
                kept.append((obj, "no_archive_counterpart"))
                continue
            basename = obj.path.rsplit("/", 1)[-1]
            orderbook = "/live_orderbook/" in obj.path or basename.startswith("live_orderbook")
            if (orderbook and asize == obj.size) or (not orderbook and asize >= obj.size):
                candidates.append(CleanupCandidate("legacy_state", obj, f"archive:{counterpart} size={asize}"))
            else:
                kept.append((obj, "archive_size_mismatch"))
            continue
        parts = _plaintext_run_parts(obj.path)
        if parts is not None:
            run, name = parts
            sibling = f"{LIVE_DATA}/state/runs/{run}/{name}.enc"
            if sibling in live_paths:
                candidates.append(
                    CleanupCandidate("plaintext_run_artifact", obj, f"sealed-sibling:{sibling} size={live_size[sibling]}")
                )
            else:
                kept.append((obj, "no_sealed_sibling"))
            continue
        if obj.path.startswith(_RUNS_PREFIX):
            rest = obj.path[len(_RUNS_PREFIX):]
            dirname = rest.split("/", 1)[0]
            stamp = _manual_backup_date(dirname)
            if stamp is not None and "/" in rest:
                basename = rest.rsplit("/", 1)[-1]
                match: tuple[str, int] | None = None
                for other_dir, other_base, other_path, other_size in runs_entries:
                    if other_dir == dirname or not other_dir.endswith("_" + stamp):
                        continue
                    if other_base in (basename, basename + ".enc") and (match is None or other_path < match[0]):
                        match = (other_path, other_size)
                if match is not None:
                    candidates.append(
                        CleanupCandidate("manual_debug_backup", obj, f"run-counterpart:{match[0]} size={match[1]}")
                    )
                else:
                    kept.append((obj, "no_run_counterpart"))
            continue

    oci_prefix = OCI_SNAPSHOT + "/"
    for obj in oci:
        if not obj.path.startswith(oci_prefix):
            continue
        rest = obj.path[len(oci_prefix):]
        basename = obj.path.rsplit("/", 1)[-1]
        if rest.startswith("crypto-pilot/"):
            hits = sorted(live_by_base.get(basename, ()))
            if hits:
                live_path, live_n = hits[0]
                candidates.append(
                    CleanupCandidate(
                        "oci_server_snapshot",
                        obj,
                        f"live:{live_path} size={live_n} snapshot_size={obj.size}",
                    )
                )
            else:
                kept.append((obj, "no_live_counterpart"))
        elif rest.startswith("krx-alpha/"):
            hits = sorted(krx_by_base.get(basename, ()))
            if hits:
                live_path, live_n = hits[0]
                candidates.append(
                    CleanupCandidate(
                        "oci_server_snapshot",
                        obj,
                        f"live-krx:{live_path} size={live_n} snapshot_size={obj.size}",
                    )
                )
            else:
                kept.append((obj, "no_live_counterpart"))
        else:
            kept.append((obj, "no_live_counterpart"))

    candidates.sort(key=lambda c: (c.obj.path, c.rule, c.evidence))
    kept.sort(key=lambda item: (item[0].path, item[1]))
    return CleanupPlan(candidates=tuple(candidates), kept=tuple(kept))


def list_remote(rclone: str, root: str, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[RemoteObject]:
    """Recursive file listing of ``DRIVE_REMOTE/root`` via ``rclone lsjson -R --fast-list --files-only``.

    Returns:
        Objects with paths relative to DRIVE_REMOTE; an absent root (rclone exit 3) yields [].

    Raises:
        RuntimeError: any other non-zero exit or unparseable JSON (fail-closed: no plan without full listings).
    """
    active = _active_runner(runner)
    completed = active(
        [rclone, "lsjson", "-R", "--fast-list", "--files-only", f"{DRIVE_REMOTE}/{root}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 3:
        return []
    if completed.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed for {root}: rc={completed.returncode} err={completed.stderr.strip()}")
    try:
        entries = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rclone lsjson unparseable for {root}: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"rclone lsjson unexpected shape for {root}")
    objects: list[RemoteObject] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("IsDir"):
            continue
        rel = entry.get("Path")
        size = entry.get("Size", 0)
        if not isinstance(rel, str) or not rel or not isinstance(size, int):
            raise RuntimeError(f"rclone lsjson unexpected entry for {root}: {entry!r}")
        objects.append(RemoteObject(path=f"{root}/{rel}", size=size))
    return objects


def _default_opener(url: str) -> bytes:  # pragma: no cover - live network is untestable here
    """Fetch a Vision bucket listing URL; any transport problem fails closed."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - public Binance Vision listing
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"vision listing HTTP {getattr(response, 'status', '?')} for {url}")
            body = response.read()
            if not isinstance(body, bytes):
                raise RuntimeError(f"vision listing unexpected body for {url}")
            return body
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"vision listing failed for {url}: {exc}") from exc


def vision_symbol_probe(*, opener: Callable[[str], bytes] = _default_opener) -> Callable[[str, str], bool]:
    """Return a memoized Vision availability check using the public S3 listing (prefix ``data/futures/um/{monthly,daily}/<dataset>/<symbol>/``).

    Raises:
        RuntimeError: network/HTTP failure (fail-closed: callers must not treat errors as "available").
    """
    cache: dict[tuple[str, str], bool] = {}

    def has_symbol(dataset: str, symbol: str) -> bool:
        key = (dataset, symbol)
        if key in cache:
            return cache[key]
        for scope in ("monthly", "daily"):
            # 심볼명에 비ASCII 문자(예: CJK 표기 페어)가 섞여 있어 URL은 반드시 퍼센트 인코딩해야 한다.
            encoded_prefix = urllib.parse.quote(f"data/futures/um/{scope}/{dataset}/{symbol}/", safe="/")
            url = f"{VISION_BUCKET_LISTING_URL}?list-type=2&prefix={encoded_prefix}"
            try:
                payload = opener(url)
            except Exception as exc:
                raise RuntimeError(f"vision probe failed for {dataset}/{symbol}: {exc}") from exc
            if b"<Key>" in payload:
                cache[key] = True
                return True
        cache[key] = False
        return False

    return has_symbol


def apply_plan(plan: CleanupPlan, rclone: str, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, int]:
    """Delete every candidate in one batch (Drive trash), then ``rclone rmdirs --leave-root`` on each touched rule root.

    One ``rclone delete --files-from`` call replaces a per-file ``deletefile`` subprocess: thousands of
    individual rclone invocations each pay Drive API round-trip latency and can blow past any reasonable
    wall-clock budget (measured: 2617 candidates did not finish a per-file sweep in 30 minutes). Outcome
    is verified by re-listing the touched roots afterward rather than trusted from the batch's exit code,
    since ``rclone delete`` continues past individual failures.

    Returns:
        Counts ``{"deleted": n, "failed": m, "bytes": b}``.

    Raises:
        RuntimeError: after the batch attempt, when re-listing shows any candidate still present.
    """
    active = _active_runner(runner)
    if not plan.candidates:
        return {"deleted": 0, "failed": 0, "bytes": 0}
    with tempfile.TemporaryDirectory(prefix="gdrive-cleanup-delete-") as listdir:
        files_from = Path(listdir) / "candidates.txt"
        files_from.write_text("".join(f"{c.obj.path}\n" for c in plan.candidates), encoding="utf-8")
        active(
            [rclone, "delete", DRIVE_REMOTE, "--files-from", str(files_from), "--fast-list", "--checkers", "8"],
            capture_output=True,
            text=True,
            check=False,
        )
    still_present: set[str] = set()
    for root in sorted({_RULE_RMDIR_ROOTS[candidate.rule] for candidate in plan.candidates}):
        still_present.update(obj.path for obj in list_remote(rclone, root, runner=runner))
    deleted = 0
    failed = 0
    freed = 0
    for candidate in plan.candidates:
        if candidate.obj.path in still_present:
            failed += 1
        else:
            deleted += 1
            freed += candidate.obj.size
    for root in sorted({_RULE_RMDIR_ROOTS[candidate.rule] for candidate in plan.candidates}):
        active(
            [rclone, "rmdirs", "--leave-root", f"{DRIVE_REMOTE}/{root}"],
            capture_output=True,
            text=True,
            check=False,
        )
    counts = {"deleted": deleted, "failed": failed, "bytes": freed}
    if failed:
        raise RuntimeError(f"gdrive cleanup partial failure: {counts}")
    return counts


def empty_trash(rclone: str, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> None:
    """Permanently empty Drive trash (``rclone cleanup gdrive:``). Irreversible; only reachable via an explicit CLI flag.

    Raises:
        RuntimeError: non-zero rclone exit.
    """
    active = _active_runner(runner)
    completed = active([rclone, "cleanup", "gdrive:"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"rclone cleanup failed: rc={completed.returncode} err={completed.stderr.strip()}")


def _rule_summary(plan: CleanupPlan) -> dict[CleanupRule, dict[str, int]]:
    """Count candidates, bytes, and kept objects per rule in RULE_ORDER."""
    summary: dict[CleanupRule, dict[str, int]] = {rule: {"candidates": 0, "bytes": 0, "kept": 0} for rule in RULE_ORDER}
    for candidate in plan.candidates:
        summary[candidate.rule]["candidates"] += 1
        summary[candidate.rule]["bytes"] += candidate.obj.size
    for obj, _reason in plan.kept:
        rule = _scope_rule_of(obj.path)
        if rule is not None:
            summary[rule]["kept"] += 1
    return summary


def _write_report(
    mode: str,
    plan: CleanupPlan | None,
    counts: dict[str, int] | None,
    *,
    status: str = "ok",
    error: str | None = None,
) -> Path:
    """Persist the cleanup report and return its path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"gdrive_cleanup_{stamp}.json"
    payload: dict[str, object] = {"mode": mode, "status": status, "created_utc": stamp}
    if plan is not None:
        payload["rules"] = dict(_rule_summary(plan))
        payload["candidates"] = [
            {"rule": c.rule, "path": c.obj.path, "size": c.obj.size, "evidence": c.evidence}
            for c in plan.candidates
        ]
        payload["kept"] = [{"path": obj.path, "size": obj.size, "reason": reason} for obj, reason in plan.kept]
    if counts is not None:
        payload["counts"] = counts
    if error is not None:
        payload["error"] = error
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: dry-run by default; ``--apply`` deletes; ``--empty-trash`` purges trash (mutually exclusive with ``--apply``). Writes a JSON report to REPORT_DIR/gdrive_cleanup_<UTC ts>.json and returns 0 on success."""
    parser = argparse.ArgumentParser(description="Evidence-gated one-time Drive cleanup (dry-run by default).")
    parser.add_argument("--apply", action="store_true", default=False, help="Delete evidence-backed candidates")
    parser.add_argument("--empty-trash", action="store_true", default=False, help="Permanently empty Drive trash")
    parser.add_argument(
        "--i-understand-irreversible",
        action="store_true",
        default=False,
        help="Explicit acknowledgement for --empty-trash",
    )
    parser.add_argument("--rclone", default="rclone", help="rclone binary")
    args = parser.parse_args(argv)
    if args.apply and args.empty_trash:
        parser.error("--apply and --empty-trash are mutually exclusive")
    if args.empty_trash:
        if not args.i_understand_irreversible:
            parser.error("--empty-trash requires --i-understand-irreversible")
        try:
            empty_trash(args.rclone)
        except RuntimeError as exc:
            print(f"[SYS] stage=gdrive_cleanup status=failed error={exc}")  # noqa: T201 - ops CLI contract output
            return 1
        _write_report("empty-trash", None, {"deleted": 0, "failed": 0, "bytes": 0})
        print("[SYS] stage=gdrive_cleanup status=ok mode=empty-trash")  # noqa: T201 - ops CLI contract output
        return 0
    try:
        listings = {root: list_remote(args.rclone, root) for root in _LISTED_ROOTS}
        plan = build_cleanup_plan(listings, vision_symbol_probe())
        if args.apply:
            listings = {root: list_remote(args.rclone, root) for root in _LISTED_ROOTS}
            plan = build_cleanup_plan(listings, vision_symbol_probe())
    except (RuntimeError, ValueError) as exc:
        print(f"[SYS] stage=gdrive_cleanup status=failed error={exc}")  # noqa: T201 - ops CLI contract output
        return 1
    if args.apply:
        try:
            counts = apply_plan(plan, args.rclone)
        except RuntimeError as exc:
            _write_report("apply", plan, None, status="failed", error=str(exc))
            for rule in RULE_ORDER:
                entry = _rule_summary(plan)[rule]
                print(  # noqa: T201 - ops CLI contract output
                    f"[SYS] stage=gdrive_cleanup rule={rule} "
                    f"candidates={entry['candidates']} bytes={entry['bytes']} kept={entry['kept']}"
                )
            print(f"[SYS] stage=gdrive_cleanup status=failed error={exc}")  # noqa: T201 - ops CLI contract output
            return 1
        _write_report("apply", plan, counts)
        for rule in RULE_ORDER:
            entry = _rule_summary(plan)[rule]
            print(  # noqa: T201 - ops CLI contract output
                f"[SYS] stage=gdrive_cleanup rule={rule} "
                f"candidates={entry['candidates']} bytes={entry['bytes']} kept={entry['kept']}"
            )
        print(  # noqa: T201 - ops CLI contract output
            f"[SYS] stage=gdrive_cleanup status=ok mode=apply "
            f"deleted={counts['deleted']} failed={counts['failed']} bytes={counts['bytes']}"
        )
        return 0
    _write_report("dry-run", plan, None)
    for rule in RULE_ORDER:
        entry = _rule_summary(plan)[rule]
        print(  # noqa: T201 - ops CLI contract output
            f"[SYS] stage=gdrive_cleanup rule={rule} "
            f"candidates={entry['candidates']} bytes={entry['bytes']} kept={entry['kept']}"
        )
    print("[SYS] stage=gdrive_cleanup status=ok mode=dry-run")  # noqa: T201 - ops CLI contract output
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
