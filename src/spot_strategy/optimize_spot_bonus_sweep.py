from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import quote_plus

from tqdm import tqdm

_logger = logging.getLogger(__name__)

ProfileValue = Union[str, float]
Profile = Dict[str, ProfileValue]
Snapshot = Dict[str, Union[str, float, int]]
_OPTIMIZE_SPOT_SCRIPT = Path(__file__).resolve().with_name("optimize_spot.py")
_STATUS_PATTERN = re.compile(r"\[STATUS\] .* \| Trial (\d+)/(\d+) \| Best: (.*)")
_SNAPSHOT_PREFIX = "[SNAPSHOT]"


PROFILE_PRESETS = {
    # Simplified profiles: only growth/risk/tail coefficients differ.
    # A = Moderate growth, balanced risk/tail
    # B = Higher growth, balanced risk/tail
    # C = Maximum growth, reduced risk/tail
    "A": {  # Moderate Growth
        "db_name": "trading_optuna_spot_bonus_a",
        "spot_growth": 42.0,
        "spot_risk": 5.0,
        "spot_tail": 5.0,
    },
    "B": {  # Balanced Growth
        "db_name": "trading_optuna_spot_bonus_b",
        "spot_growth": 52.0,
        "spot_risk": 5.0,
        "spot_tail": 5.0,
    },
    "C": {  # Maximum Growth
        "db_name": "trading_optuna_spot_bonus_c",
        "spot_growth": 65.0,
        "spot_risk": 4.0,
        "spot_tail": 4.0,
    },
}


def build_env(
    base_env: Mapping[str, str], profile: Profile, preload: bool = False
) -> Dict[str, str]:
    env = dict(base_env)
    env["DB_NAME"] = str(profile["db_name"])
    env["SPOT_GROWTH_BONUS_COEF"] = str(profile["spot_growth"])
    env["SPOT_RISK_DRAG_COEF"] = str(profile["spot_risk"])
    env["SPOT_TAIL_DRAG_COEF"] = str(profile["spot_tail"])
    env["SPOT_BONUS_PROFILE"] = str(profile.get("profile_key", ""))
    env["SPOT_SWEEP_CHILD"] = "1"
    env["OPTUNA_NO_PROGRESS"] = "1"  # Force no progress bar in child

    # Safe API defaults for parallel sweep: reduce 429 bursts while keeping throughput.
    env.setdefault("SPOT_GAP_FILL_MAX_RANGES", "3")
    env.setdefault("UPBIT_OHLCV_LOOP_SLEEP_SEC", "0.45")
    env.setdefault("UPBIT_OHLCV_RETRY_BASE_SEC", "1.2")
    env.setdefault("UPBIT_OHLCV_RETRY_MAX_SEC", "20.0")
    env.setdefault("UPBIT_OHLCV_MAX_RETRIES", "8")

    # Run gap-fill only in preload; workers must not perform extra gap-fill.
    env["SPOT_GAP_FILL_ENABLE"] = "1" if preload else "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _has_jobs_arg(args: List[str]) -> bool:
    for a in args:
        if a == "--jobs" or a.startswith("--jobs="):
            return True
    return False


def _with_child_jobs(forwarded_args: List[str], child_jobs: int) -> List[str]:
    if _has_jobs_arg(forwarded_args):
        return list(forwarded_args)
    return list(forwarded_args) + ["--jobs", str(max(1, int(child_jobs)))]


def _run_child_with_snapshot(
    cmd: List[str],
    env: Dict[str, str],
    timeout_sec: int,
    run_id: str,
    pbar: Optional[tqdm] = None,
    stream_prefix: str = "",
) -> Tuple[int, Optional[Snapshot]]:
    snapshot: Optional[Snapshot] = None
    start_ts = time.time()
    timeout_limit_sec = max(1, int(timeout_sec))

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if pbar:
            pbar.set_description(f"[{stream_prefix}] Starting...")

        if process.stdout is None:
            return 1, snapshot

        for line in process.stdout:
            # Parse snapshot
            if line.startswith(_SNAPSHOT_PREFIX):
                payload = line[len(_SNAPSHOT_PREFIX) :].strip()
                if payload:
                    try:
                        parsed = json.loads(payload)
                        if isinstance(parsed, dict):
                            snapshot = parsed
                    except json.JSONDecodeError:
                        pass
                continue

            # Parse status for progress bar
            if "[STATUS]" in line:
                match = _STATUS_PATTERN.search(line)
                if match and pbar:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    best_val = match.group(3)

                    if pbar.total != total:
                        pbar.total = total
                        pbar.refresh()

                    pbar.n = current
                    pbar.set_description(f"[{stream_prefix}] Best: {best_val}")
                    pbar.refresh()

            # Check timeout only if enabled (matching futures behavior)
            if timeout_limit_sec > 0 and (time.time() - start_ts) > timeout_limit_sec:
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                return 124, snapshot

        return int(process.wait()), snapshot

    except Exception:
        return 1, snapshot


def run_profile(
    profile_key: str,
    profile: Profile,
    forwarded_args: List[str],
    child_jobs: int,
    retry_count: int,
    retry_backoff_sec: float,
    timeout_sec: int,
    run_id: str,
    pbar: Optional[tqdm] = None,
) -> Tuple[int, Dict[str, Snapshot]]:
    timeout_limit_sec = max(1, int(timeout_sec))
    base_cmd = [sys.executable, str(_OPTIMIZE_SPOT_SCRIPT)] + _with_child_jobs(
        forwarded_args, child_jobs
    )
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=False)

    attempts = max(1, int(retry_count) + 1)
    snapshots: Dict[str, Snapshot] = {}
    stream_prefix = f"{profile_key}:single"

    for attempt in range(1, attempts + 1):
        try:
            code, snap = _run_child_with_snapshot(
                base_cmd,
                env=env,
                timeout_sec=timeout_limit_sec,
                run_id=run_id,
                pbar=pbar,
                stream_prefix=stream_prefix,
            )
            if code == 0:
                if snap:
                    snapshots["single"] = snap
                return 0, snapshots

            if attempt < attempts:
                wait_s = max(0.1, float(retry_backoff_sec)) * attempt
                msg = f"[{stream_prefix}] attempt {attempt}/{attempts} failed (code={code}). retrying in {wait_s:.1f}s..."
                if pbar:
                    pbar.write(msg)
                time.sleep(wait_s)
                continue

            return code, snapshots

        except Exception as e:
            code = 124
            if attempt < attempts:
                wait_s = max(0.1, float(retry_backoff_sec)) * attempt
                msg = f"[{stream_prefix}] attempt {attempt}/{attempts} error ({e}). retrying in {wait_s:.1f}s..."
                if pbar:
                    pbar.write(msg)
                time.sleep(wait_s)
                continue
            return code, snapshots

    return 1, snapshots


def run_preload(profile_key: str, profile: Profile, forwarded_args: List[str]) -> int:
    cmd = [sys.executable, str(_OPTIMIZE_SPOT_SCRIPT), "--prepare-data-only"] + list(
        forwarded_args
    )
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=True)

    print("\n" + "=" * 72)
    print(f"[PRELOAD] DB={profile['db_name']} (cache warmup only)")
    print(f"[PRELOAD] RUN: {' '.join(cmd)}")
    print("=" * 72)
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _run_pre_sweep_cleanup(
    run_list: List[Tuple[str, Profile]],
    study_suffix: str = "spot_unified_strategy",
) -> None:
    """Connect to each profile DB and remove orphan studies + reclaim InnoDB space.

    Requires ``DB_USER``, ``DB_PASS``, ``DB_HOST``, ``DB_PORT`` env vars (loaded from .env).
    Skips silently if credentials are missing.
    """
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")

    if not all([db_user, db_pass]):
        _logger.warning("[CLEANUP] Missing DB credentials — skipping pre-sweep cleanup")
        return

    try:
        import optuna  # noqa: PLC0415

        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.optimization.opt_utils import (  # noqa: PLC0415
            cleanup_orphan_studies,
            get_db_size_mb,
            reclaim_mysql_space,
        )
    except Exception as exc:
        _logger.warning("[CLEANUP] Import failed — skipping pre-sweep cleanup: %s", exc)
        return

    print("\n" + "=" * 72)
    print("[PRE-SWEEP CLEANUP] Removing orphan studies and reclaiming InnoDB space...")
    print("=" * 72)

    for key, profile in run_list:
        db_name = str(profile["db_name"])
        storage_url = f"mysql+pymysql://{db_user}:{quote_plus(str(db_pass))}@{db_host}:{db_port}/{db_name}"
        try:
            storage = optuna.storages.RDBStorage(
                url=storage_url, engine_kwargs={"pool_pre_ping": True}
            )
        except Exception as exc:
            _logger.warning("[CLEANUP] %s: cannot connect to %s: %s", key, db_name, exc)
            continue

        size_before = get_db_size_mb(storage_url, db_name)
        print(f"[CLEANUP] {key}: {db_name} ({size_before:.2f} MB)")

        orphan_keep = int(os.getenv("OPTUNA_ORPHAN_KEEP_RECENT", "0"))
        cleanup_orphan_studies(storage, study_suffix, "__", keep_recent=orphan_keep)

        reclaim_mysql_space(storage_url)
        size_after = get_db_size_mb(storage_url, db_name)
        print(
            f"[CLEANUP] {key}: {db_name} → {size_after:.2f} MB (freed ~{max(0.0, size_before - size_after):.2f} MB)"
        )

    print("=" * 72 + "\n")


def _snapshot_to_columns(snapshot: Snapshot) -> Dict[str, str]:
    return {
        "ret": f"{float(snapshot.get('ret', 0.0)):.2f}%",
        "mdd": f"{float(snapshot.get('mdd', 0.0)):.2f}%",
        "pf": f"{float(snapshot.get('pf', 0.0)):.2f}",
        "win": f"{float(snapshot.get('win_rate', 0.0)):.1f}%",
        "trades": str(int(snapshot.get("trades", 0))),
        "score": f"{float(snapshot.get('score', 0.0)):.2f}",
    }


def _build_summary_rows(
    key: str,
    db_name: str,
    code: int,
    runtime_snapshots: Dict[str, Snapshot],
) -> List[Dict[str, str]]:
    status = "OK" if code == 0 else f"FAIL({code})"
    if not runtime_snapshots:
        return [
            {
                "profile": key,
                "db": db_name,
                "status": status,
                "ret": "N/A",
                "mdd": "N/A",
                "pf": "N/A",
                "win": "N/A",
                "trades": "N/A",
                "score": "N/A",
            }
        ]

    single = runtime_snapshots.get("single")
    if single is None:
        single = next(iter(runtime_snapshots.values()))
    cols = _snapshot_to_columns(single)
    return [{"profile": key, "db": db_name, "status": status, **cols}]


def main() -> int:
    if sys.platform == "win32":
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if callable(stdout_reconfigure):
            stdout_reconfigure(encoding="utf-8")
        if callable(stderr_reconfigure):
            stderr_reconfigure(encoding="utf-8")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Run spot optimization with bonus coefficient sweep across 3 DBs. "
        "Quick smoke test: --profiles A --max-concurrent 1 and forward e.g. --trials 50 to reduce runtime."
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="A,B,C",
        help="Comma-separated profile keys (default: A,B,C). Use a single key (e.g. A) for faster runs.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max number of profiles to run concurrently (default: 3)",
    )
    parser.add_argument(
        "--child-jobs",
        type=int,
        default=3,
        help="Per-profile --jobs for optimize_spot child process when --jobs is not explicitly forwarded (default: 3)",
    )
    parser.add_argument(
        "--skip-preload",
        dest="skip_preload",
        action="store_true",
        help="Skip one-time serial data preload step before profile runs (default: preload runs).",
    )
    parser.add_argument(
        "--preload",
        dest="skip_preload",
        action="store_false",
        help="Run one-time serial data preload before profiles (default).",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Retry count per profile on non-zero exit/timeout (default: 0)",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=10.0,
        help="Base backoff seconds for profile retry (default: 10.0)",
    )
    parser.add_argument(
        "--profile-timeout-sec",
        type=int,
        default=5400,
        help="Timeout seconds per profile child process (default: 5400)",
    )
    parser.add_argument(
        "--cleanup-db",
        dest="cleanup_db",
        action="store_true",
        default=False,
        help=(
            "Run pre-sweep DB maintenance: delete orphan studies and OPTIMIZE TABLE "
            "on each profile DB before starting optimization (default: disabled)."
        ),
    )
    parser.set_defaults(skip_preload=False)
    args, forwarded_args = parser.parse_known_args()

    requested = [p.strip().upper() for p in args.profiles.split(",") if p.strip()]
    run_list: List[Tuple[str, Profile]] = []
    for key in requested:
        if key not in PROFILE_PRESETS:
            print(
                f"Unknown profile key: {key}. Available: {', '.join(PROFILE_PRESETS)}"
            )
            return 2
        run_list.append((key, PROFILE_PRESETS[key]))
    if not run_list:
        print("No profiles selected. Use --profiles with at least one of: A,B,C")
        return 2

    max_concurrent = max(1, min(args.max_concurrent, len(run_list)))

    if args.cleanup_db:
        _run_pre_sweep_cleanup(run_list, study_suffix="spot_unified_strategy")

    print(f"Starting sweep for profiles: {', '.join(k for k, _ in run_list)}")
    print("-" * 60)
    debug_run_id = f"sweep_{int(time.time() * 1000)}"

    summary_map: Dict[str, Dict[str, Union[int, Dict[str, Snapshot]]]] = {}

    if not args.skip_preload:
        preload_key, preload_profile = run_list[0]
        preload_code = run_preload(preload_key, preload_profile, forwarded_args)
        if preload_code != 0:
            print(f"\n[PRELOAD] failed with exit code={preload_code}. stopping sweep.")
            return preload_code

    # Create progress bars
    pbars = {}
    for i, (key, _) in enumerate(run_list):
        pbars[key] = tqdm(total=100, desc=f"[{key}] Waiting...", position=i, leave=True)

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {}
        for key, profile in run_list:
            fut = pool.submit(
                run_profile,
                key,
                profile,
                forwarded_args,
                args.child_jobs,
                args.retry,
                args.retry_backoff_sec,
                args.profile_timeout_sec,
                debug_run_id,
                pbars[key],
            )
            futures[fut] = (key, profile)

        for fut in as_completed(futures):
            key, profile = futures[fut]
            try:
                code, runtime_snapshots = fut.result()
            except Exception as e:
                code = 1
                runtime_snapshots = {}
                pbars[key].write(f"[{key}] Exception: {e}")

            summary_map[key] = {"code": int(code), "snapshots": runtime_snapshots}
            status = "OK" if code == 0 else f"FAIL({code})"
            pbars[key].set_description(f"[{key}] Done ({status})")
            pbars[key].close()

    print("\n" + "-" * 72)
    print("Sweep summary")
    summary = []
    for key, profile in run_list:
        if key in summary_map:
            entry = summary_map[key]
            raw_code = entry.get("code", 1)
            code = int(raw_code) if isinstance(raw_code, int) else 1
            runtime_snapshots_raw = entry.get("snapshots", {})
            runtime_snapshots = (
                runtime_snapshots_raw if isinstance(runtime_snapshots_raw, dict) else {}
            )
            summary.extend(
                _build_summary_rows(
                    key=key,
                    db_name=str(profile["db_name"]),
                    code=code,
                    runtime_snapshots=runtime_snapshots,
                )
            )

    headers = ["profile", "db", "status", "ret", "mdd", "pf", "win", "trades", "score"]
    widths = {h: len(h) for h in headers}
    for row in summary:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    header_line = " | ".join(h.upper().ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in summary:
        print(" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))
    print("-" * 72)

    final_code = (
        0
        if summary
        and all(
            str(r.get("status", "")).startswith("OK")
            for r in summary
            if str(r.get("profile", ""))
        )
        else 1
    )
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
