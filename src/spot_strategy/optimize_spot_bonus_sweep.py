import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

ProfileValue = Union[str, float]
Profile = Dict[str, ProfileValue]


PROFILE_PRESETS = {
    "A": {
        "db_name": "trading_optuna_spot_bonus_a",
        "spot_growth": 18.0,
        "spot_risk": 10.0,
        "spot_tail": 10.0,
    },
    "B": {
        "db_name": "trading_optuna_spot_bonus_b",
        "spot_growth": 14.0,
        "spot_risk": 12.0,
        "spot_tail": 12.0,
    },
    "C": {
        "db_name": "trading_optuna_spot_bonus_c",
        "spot_growth": 22.0,
        "spot_risk": 8.0,
        "spot_tail": 8.0,
    },
}


def build_env(base_env: Dict[str, str], profile: Profile) -> Dict[str, str]:
    env = base_env.copy()
    env["DB_NAME"] = str(profile["db_name"])
    env["SPOT_GROWTH_BONUS_COEF"] = str(profile["spot_growth"])
    env["SPOT_RISK_DRAG_COEF"] = str(profile["spot_risk"])
    env["SPOT_TAIL_DRAG_COEF"] = str(profile["spot_tail"])
    env["SPOT_BONUS_PROFILE"] = str(profile.get("profile_key", ""))
    env["SPOT_SWEEP_CHILD"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_profile(profile_key: str, profile: Profile, forwarded_args: List[str]) -> int:
    optimize_script = Path(__file__).resolve().with_name("optimize_spot.py")
    cmd = [sys.executable, str(optimize_script)] + forwarded_args
    env_profile = dict(profile)
    env_profile["profile_key"] = profile_key
    env = build_env(os.environ, env_profile)

    print("\n" + "=" * 72)
    print(f"[{profile_key}] DB={profile['db_name']}")
    print(
        f"[{profile_key}] SPOT bonus=(growth:{profile['spot_growth']}, "
        f"risk:{profile['spot_risk']}, tail:{profile['spot_tail']})"
    )
    print(f"[{profile_key}] RUN: {' '.join(cmd)}")
    print("=" * 72)

    result = subprocess.run(cmd, env=env)
    return result.returncode


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
        default=1,
        help="Max number of profiles to run concurrently (default: 1)",
    )
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
    summary_map: Dict[str, int] = {}
    if max_concurrent == 1:
        for key, profile in run_list:
            code = run_profile(key, profile, forwarded_args)
            summary_map[key] = code
            if code != 0:
                print(f"\n[{key}] failed with exit code={code}. stopping sweep.")
                break
    else:
        print(
            f"\nRunning sweep in parallel (max_concurrent={max_concurrent}, "
            f"profiles={','.join(k for k, _ in run_list)})"
        )
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(run_profile, key, profile, forwarded_args): (key, profile)
                for key, profile in run_list
            }
            for fut in as_completed(futures):
                key, profile = futures[fut]
                code = fut.result()
                summary_map[key] = code
                status = "OK" if code == 0 else f"FAIL({code})"
                print(f"[{key}] finished -> {status}")

    summary = []
    for key, profile in run_list:
        if key in summary_map:
            summary.append((key, profile["db_name"], summary_map[key]))

    print("\n" + "-" * 72)
    print("Sweep summary")
    for key, db_name, code in summary:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"- {key}: {db_name} -> {status}")
    print("-" * 72)

    final_code = 0 if summary and all(code == 0 for _, _, code in summary) else 1
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
