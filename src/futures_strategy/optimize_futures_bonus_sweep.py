import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm


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
    log_file: Optional[str] = None,
    stream_prefix: str = "",
) -> int:
    start_ts = time.time()
    log_f = None
    if log_file:
        try:
            log_f = open(log_file, "a", encoding="utf-8")
        except Exception:
            log_f = None

    try:
        if log_f:
            log_f.write(f"\n[{stream_prefix}] START: {' '.join(cmd)}\n")
            log_f.flush()

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
            if log_f:
                log_f.write(f"\n[{stream_prefix}] ERROR: child stdout is None\n")
            return 1

        for line in process.stdout:
            if log_f:
                log_f.write(line)
                log_f.flush()

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
                if log_f:
                    log_f.write(f"\n[{stream_prefix}] KILLED (timeout)\n")
                return 124

        code = int(process.wait())
        if log_f:
            log_f.write(f"\n[{stream_prefix}] FINISHED (code={code})\n")
        return code
    except Exception as e:
        if log_f:
            log_f.write(f"\n[{stream_prefix}] EXCEPTION: {e}\n")
        return 1
    finally:
        if log_f:
            log_f.close()


def run_profile(
    profile_key: str,
    profile: Profile,
    forwarded_args: List[str],
    child_jobs: int,
    retry_count: int,
    retry_backoff_sec: float,
    timeout_sec: int,
    pbar: Optional[tqdm] = None,
    log_file: Optional[str] = None,
) -> int:
    optimize_script = Path(__file__).resolve().with_name("optimize_futures.py")
    base_cmd = [sys.executable, str(optimize_script)] + _with_child_jobs(forwarded_args, child_jobs)
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=False)
    attempts = max(1, int(retry_count) + 1)
    stream_prefix = f"{profile_key}:single"

    for attempt in range(1, attempts + 1):
        code = _run_child_with_status(
            cmd=list(base_cmd),
            env=env,
            timeout_sec=int(timeout_sec),
            pbar=pbar,
            log_file=log_file,
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
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{msg}\n")
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
    
    # Setup logs dir
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    print(f"Starting sweep for profiles: {', '.join(k for k, _ in run_list)}")
    print(f"Logs will be written to: {logs_dir.absolute()}")
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
            log_file = logs_dir / f"sweep_{key.lower()}.log"
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
                str(log_file),
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
