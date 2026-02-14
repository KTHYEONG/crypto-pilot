import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

ProfileValue = Union[str, float]
Profile = Dict[str, ProfileValue]
Snapshot = Dict[str, Union[str, float, int]]


PROFILE_PRESETS = {
    "A": {
        "db_name": "trading_optuna_spot_bonus_a",
        "spot_growth": 20.0,
        "spot_risk": 9.0,
        "spot_tail": 9.0,
    },
    "B": {
        "db_name": "trading_optuna_spot_bonus_b",
        "spot_growth": 18.0,
        "spot_risk": 10.0,
        "spot_tail": 10.0,
    },
    "C": {
        "db_name": "trading_optuna_spot_bonus_c",
        "spot_growth": 26.0,
        "spot_risk": 6.0,
        "spot_tail": 6.0,
    },
    "D": {
        "db_name": "trading_optuna_spot_bonus_d",
        "spot_growth": 30.0,
        "spot_risk": 5.0,
        "spot_tail": 5.0,
    },
    "E": {
        "db_name": "trading_optuna_spot_bonus_e",
        "spot_growth": 34.0,
        "spot_risk": 4.0,
        "spot_tail": 4.0,
    },
}


def build_env(base_env: Dict[str, str], profile: Profile, preload: bool = False) -> Dict[str, str]:
    env = base_env.copy()
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
    pbar: Optional[tqdm] = None,
    log_file: Optional[str] = None,
    stream_prefix: str = "",
) -> Tuple[int, Optional[Snapshot]]:
    snapshot: Optional[Snapshot] = None
    start_ts = time.time()
    
    log_f = None
    if log_file:
        try:
            log_f = open(log_file, "a", encoding="utf-8")
        except Exception:
            pass

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

        for line in process.stdout:
            line_stripped = line.strip()
            
            # Write to log file
            if log_f:
                log_f.write(line)
                log_f.flush()
                
            # Parse snapshot
            if line.startswith("[SNAPSHOT]"):
                payload = line[len("[SNAPSHOT]") :].strip()
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        snapshot = parsed
                except json.JSONDecodeError:
                    pass
                continue
                
            # Parse status for progress bar
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

            # Check timeout
            if (time.time() - start_ts) > max(1, int(timeout_sec)):
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                if log_f:
                    log_f.write(f"\n[{stream_prefix}] KILLED (timeout)\n")
                return 124, snapshot

        code = int(process.wait())
        if log_f:
            log_f.write(f"\n[{stream_prefix}] FINISHED (code={code})\n")
        
        return code, snapshot
        
    except Exception as e:
        if log_f:
            log_f.write(f"\n[{stream_prefix}] EXCEPTION: {e}\n")
        return 1, snapshot
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
) -> Tuple[int, Dict[str, Snapshot]]:
    optimize_script = Path(__file__).resolve().with_name("optimize_spot.py")
    base_cmd = [sys.executable, str(optimize_script)] + _with_child_jobs(forwarded_args, child_jobs)
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=False)

    attempts = max(1, int(retry_count) + 1)
    snapshots: Dict[str, Snapshot] = {}
    stream_prefix = f"{profile_key}:single"

    for attempt in range(1, attempts + 1):
        try:
            code, snap = _run_child_with_snapshot(
                list(base_cmd),
                env=env,
                timeout_sec=max(1, int(timeout_sec)),
                pbar=pbar,
                log_file=log_file,
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
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n{msg}\n")
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
    optimize_script = Path(__file__).resolve().with_name("optimize_spot.py")
    cmd = [sys.executable, str(optimize_script), "--prepare-data-only"] + list(forwarded_args)
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile, preload=True)

    print("\n" + "=" * 72)
    print(f"[PRELOAD] DB={profile['db_name']} (cache warmup only)")
    print(f"[PRELOAD] RUN: {' '.join(cmd)}")
    print("=" * 72)
    result = subprocess.run(cmd, env=env)
    return result.returncode


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
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run spot optimization with bonus coefficient sweep across 3 DBs."
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
        help="Per-profile --jobs for optimize_spot child process when --jobs is not explicitly forwarded (default: 3)",
    )
    parser.add_argument(
        "--skip-preload",
        dest="skip_preload",
        action="store_true",
        help="Skip one-time serial data preload step before profile runs (default: enabled).",
    )
    parser.add_argument(
        "--preload",
        dest="skip_preload",
        action="store_false",
        help="Enable one-time serial data preload before running profiles.",
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
    parser.set_defaults(skip_preload=True)
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
            log_file = logs_dir / f"spot_sweep_{key.lower()}.log"
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
                str(log_file)
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
                
            summary_map[key] = {"code": int(code), "snapshots": dict(runtime_snapshots)}
            status = "OK" if code == 0 else f"FAIL({code})"
            pbars[key].set_description(f"[{key}] Done ({status})")
            pbars[key].close()

    print("\n" + "-" * 72)
    print("Sweep summary")
    summary = []
    for key, profile in run_list:
        if key in summary_map:
            entry = summary_map[key]
            code = int(entry.get("code", 1))
            runtime_snapshots = dict(entry.get("snapshots", {}))
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
        if summary and all(str(r.get("status", "")).startswith("OK") for r in summary if str(r.get("profile", "")))
        else 1
    )
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
