from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import json
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote_plus

from tqdm import tqdm

_logger = logging.getLogger(__name__)


PROFILE_PRESETS = {
    "A": {
        "db_name": "trading_optuna_bonus_a",
        "fut_growth": 30.0,
        "fut_risk": 8.0,
        "fut_tail": 12.0,
        "spot_growth": 18.0,
        "spot_risk": 10.0,
        "spot_tail": 10.0,
    },
    "B": {
        "db_name": "trading_optuna_bonus_b",
        "fut_growth": 24.0,
        "fut_risk": 10.0,
        "fut_tail": 14.0,
        "spot_growth": 14.0,
        "spot_risk": 12.0,
        "spot_tail": 12.0,
    },
    "C": {
        "db_name": "trading_optuna_bonus_c",
        "fut_growth": 36.0,
        "fut_risk": 6.0,
        "fut_tail": 10.0,
        "spot_growth": 22.0,
        "spot_risk": 8.0,
        "spot_tail": 8.0,
    },
}


ProfileValue = Union[str, float]
Profile = Dict[str, ProfileValue]


def build_env(base_env: Dict[str, str], profile: Profile, preload: bool = False) -> Dict[str, str]:
    env = base_env.copy()
    env["DB_NAME"] = str(profile["db_name"])
    env["FUT_GROWTH_BONUS_COEF"] = str(profile["fut_growth"])
    env["FUT_RISK_DRAG_COEF"] = str(profile["fut_risk"])
    env["FUT_TAIL_DRAG_COEF"] = str(profile["fut_tail"])
    env["SPOT_GROWTH_BONUS_COEF"] = str(profile["spot_growth"])
    env["SPOT_RISK_DRAG_COEF"] = str(profile["spot_risk"])
    env["SPOT_TAIL_DRAG_COEF"] = str(profile["spot_tail"])
    env["FUTURES_BONUS_PROFILE"] = str(profile.get("profile_key", ""))
    env["FUTURES_SWEEP_CHILD"] = "1"
    # Keep child output parseable for status monitoring.
    env["OPTUNA_NO_PROGRESS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _has_arg(args: List[str], flag: str) -> bool:
    for arg in args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def _with_child_jobs(forwarded_args: List[str], child_jobs: int) -> List[str]:
    cmd_args = list(forwarded_args)
    if not _has_arg(cmd_args, "--jobs"):
        cmd_args += ["--jobs", str(max(1, int(child_jobs)))]
    if not _has_arg(cmd_args, "--no-progress"):
        cmd_args.append("--no-progress")
    return cmd_args


def _run_child_with_status(
    cmd: List[str],
    env: Dict[str, str],
    timeout_sec: int,
    pbar: Optional[tqdm] = None,
    stream_prefix: str = "",
) -> int:
    start_ts = time.time()

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

        status_pattern = re.compile(r"\[STATUS\] .* \| Trial (\d+)/(\d+) \| Best: (.*)")
        if pbar:
            pbar.set_description(f"[{stream_prefix}] Starting...")

        if process.stdout is None:
            return 1

        last_line: Optional[str] = None
        error_lines: List[str] = []
        all_lines: List[str] = []

        for line in process.stdout:
            stripped = line.rstrip("\n")
            last_line = stripped
            all_lines.append(stripped)
            
            if any(keyword in stripped for keyword in ["[ERROR]", "Traceback", "Exception", "Error:", "failed", "Failed"]):
                error_lines.append(stripped)
            
            if "[STATUS]" in line:
                match = status_pattern.search(line)
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

            # Timeout is opt-in to keep default behavior backward compatible.
            if int(timeout_sec) > 0 and (time.time() - start_ts) > int(timeout_sec):
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                return 124

        code = int(process.wait())

        if code != 0:
            # region agent log
            try:
                log_path = Path("debug-2e3aee.log")
                payload = {
                    "sessionId": "2e3aee",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "optimize_futures_bonus_sweep.py:_run_child_with_status",
                    "message": "child_exit_nonzero",
                    "data": {
                        "stream_prefix": stream_prefix,
                        "exit_code": code,
                        "last_line": last_line,
                        "error_lines": error_lines[-10:] if error_lines else [],
                        "last_20_lines": all_lines[-20:] if all_lines else [],
                    },
                    "timestamp": int(time.time() * 1000),
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # endregion

        return code
    except Exception:
        return 1


def run_profile(
    profile_key: str,
    profile: Profile,
    forwarded_args: List[str],
    child_jobs: int,
    retry_count: int,
    retry_backoff_sec: float,
    timeout_sec: int,
    pbar: Optional[tqdm] = None,
) -> int:
    optimize_script = Path(__file__).resolve().with_name("optimize_futures.py")
    base_cmd = [sys.executable, str(optimize_script)] + _with_child_jobs(forwarded_args, child_jobs)
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=False)
    attempts = max(1, int(retry_count) + 1)
    stream_prefix = f"{profile_key}:single"

    # region agent log
    try:
        log_path = Path("debug-2e3aee.log")
        payload = {
            "sessionId": "2e3aee",
            "runId": "pre-fix",
            "hypothesisId": "H2",
            "location": "optimize_futures_bonus_sweep.py:run_profile",
            "message": "profile_env_snapshot",
            "data": {
                "profile_key": profile_key,
                "db_name": env.get("DB_NAME"),
                "futures_bonus_profile": env.get("FUTURES_BONUS_PROFILE"),
                "futures_sweep_child": env.get("FUTURES_SWEEP_CHILD"),
            },
            "timestamp": int(time.time() * 1000),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion

    for attempt in range(1, attempts + 1):
        code = _run_child_with_status(
            cmd=list(base_cmd),
            env=env,
            timeout_sec=int(timeout_sec),
            pbar=pbar,
            stream_prefix=stream_prefix,
        )
        if code == 0:
            return 0

        if attempt < attempts:
            wait_s = max(0.1, float(retry_backoff_sec)) * attempt
            msg = (
                f"[{stream_prefix}] attempt {attempt}/{attempts} failed (code={code}). "
                f"retrying in {wait_s:.1f}s..."
            )
            if pbar:
                pbar.write(msg)
            time.sleep(wait_s)
            continue
        return code

    return 1


def run_preload(
    profile_key: str,
    profile: Profile,
    forwarded_args: List[str],
    child_jobs: int,
) -> int:
    optimize_script = Path(__file__).resolve().with_name("optimize_futures.py")
    cmd = (
        [sys.executable, str(optimize_script), "--prepare-data-only"]
        + _with_child_jobs(forwarded_args, child_jobs)
    )
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=True)

    print("\n" + "=" * 72)
    print(f"[PRELOAD] DB={profile['db_name']} (cache warmup only)")
    print(f"[PRELOAD] RUN: {' '.join(cmd)}")
    print("=" * 72)
    return int(subprocess.run(cmd, env=env).returncode)


def _run_pre_sweep_cleanup(
    run_list: List[Tuple[str, Profile]],
    study_suffix: str = "futures_unified_strategy",
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
            storage = optuna.storages.RDBStorage(url=storage_url, engine_kwargs={"pool_pre_ping": True})
        except Exception as exc:
            _logger.warning("[CLEANUP] %s: cannot connect to %s: %s", key, db_name, exc)
            continue

        size_before = get_db_size_mb(storage_url, db_name)
        print(f"[CLEANUP] {key}: {db_name} ({size_before:.2f} MB)")

        orphan_keep = int(os.getenv("OPTUNA_ORPHAN_KEEP_RECENT", "0"))
        cleanup_orphan_studies(storage, study_suffix, "__", keep_recent=orphan_keep)

        reclaim_mysql_space(storage_url)
        size_after = get_db_size_mb(storage_url, db_name)
        print(f"[CLEANUP] {key}: {db_name} → {size_after:.2f} MB (freed ~{max(0.0, size_before - size_after):.2f} MB)")

    print("=" * 72 + "\n")


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run futures optimization with bonus coefficient sweep across 3 DBs."
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="A,B,C",
        help="Comma-separated profile keys (default: A,B,C)",
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
        help="Per-profile --jobs for optimize_futures child process when --jobs is not explicitly forwarded (default: 3)",
    )
    parser.add_argument(
        "--skip-preload",
        dest="skip_preload",
        action="store_true",
        help="Skip one-time serial data preload step before profile runs.",
    )
    parser.add_argument(
        "--preload",
        dest="skip_preload",
        action="store_false",
        help="Enable one-time serial data preload before running profiles (default).",
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
        default=0,
        help="Timeout seconds per profile child process (default: 0=disabled)",
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
            print(f"Unknown profile key: {key}. Available: {', '.join(PROFILE_PRESETS)}")
            return 2
        run_list.append((key, PROFILE_PRESETS[key]))
    if not run_list:
        print("No profiles selected. Use --profiles with at least one of: A,B,C")
        return 2

    max_concurrent = max(1, min(args.max_concurrent, len(run_list)))

    if args.cleanup_db:
        _run_pre_sweep_cleanup(run_list, study_suffix="futures_unified_strategy")

    print(f"Starting sweep for profiles: {', '.join(k for k, _ in run_list)}")
    print("-" * 60)

    summary_map: Dict[str, int] = {}

    if not args.skip_preload:
        preload_key, preload_profile = run_list[0]
        preload_code = run_preload(preload_key, preload_profile, forwarded_args, args.child_jobs)
        if preload_code != 0:
            print(f"\n[PRELOAD] failed with exit code={preload_code}. stopping sweep.")
            return preload_code

    # Create progress bars and pass one bar per profile worker.
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
                pbars[key],
            )
            futures[fut] = (key, profile)

        for fut in as_completed(futures):
            key, profile = futures[fut]
            try:
                code = fut.result()
            except Exception as e:
                code = 1
                pbars[key].write(f"[{key}] Exception: {e}")
            
            summary_map[key] = code
            status = "OK" if code == 0 else f"FAIL({code})"
            pbars[key].set_description(f"[{key}] Done ({status})")
            pbars[key].close()

    print("\n" + "-" * 72)
    print("Sweep summary")
    summary = []
    for key, profile in run_list:
        if key in summary_map:
            summary.append((key, profile["db_name"], summary_map[key]))

    for key, db_name, code in summary:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"- {key}: {db_name} -> {status}")
    print("-" * 72)

    final_code = 0 if summary and all(code == 0 for _, _, code in summary) else 1
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
