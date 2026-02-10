import argparse
import os
import subprocess
import sys
from pathlib import Path


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


def build_env(base_env, profile):
    env = base_env.copy()
    env["DB_NAME"] = profile["db_name"]
    env["FUT_GROWTH_BONUS_COEF"] = str(profile["fut_growth"])
    env["FUT_RISK_DRAG_COEF"] = str(profile["fut_risk"])
    env["FUT_TAIL_DRAG_COEF"] = str(profile["fut_tail"])
    env["SPOT_GROWTH_BONUS_COEF"] = str(profile["spot_growth"])
    env["SPOT_RISK_DRAG_COEF"] = str(profile["spot_risk"])
    env["SPOT_TAIL_DRAG_COEF"] = str(profile["spot_tail"])
    return env


def run_profile(profile_key, profile, forwarded_args):
    optimize_script = Path(__file__).resolve().with_name("optimize_futures.py")
    cmd = [sys.executable, str(optimize_script)] + forwarded_args
    env = build_env(os.environ, profile)

    print("\n" + "=" * 72)
    print(f"[{profile_key}] DB={profile['db_name']}")
    print(
        f"[{profile_key}] FUT bonus=(growth:{profile['fut_growth']}, "
        f"risk:{profile['fut_risk']}, tail:{profile['fut_tail']})"
    )
    print(
        f"[{profile_key}] SPOT bonus=(growth:{profile['spot_growth']}, "
        f"risk:{profile['spot_risk']}, tail:{profile['spot_tail']})"
    )
    print(f"[{profile_key}] RUN: {' '.join(cmd)}")
    print("=" * 72)

    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run futures optimization with bonus coefficient sweep across 3 DBs."
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="A,B,C",
        help="Comma-separated profile keys (default: A,B,C)",
    )
    args, forwarded_args = parser.parse_known_args()

    requested = [p.strip().upper() for p in args.profiles.split(",") if p.strip()]
    run_list = []
    for key in requested:
        if key not in PROFILE_PRESETS:
            print(f"Unknown profile key: {key}. Available: {', '.join(PROFILE_PRESETS)}")
            return 2
        run_list.append((key, PROFILE_PRESETS[key]))

    summary = []
    for key, profile in run_list:
        code = run_profile(key, profile, forwarded_args)
        summary.append((key, profile["db_name"], code))
        if code != 0:
            print(f"\n[{key}] failed with exit code={code}. stopping sweep.")
            break

    print("\n" + "-" * 72)
    print("Sweep summary")
    for key, db_name, code in summary:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"- {key}: {db_name} -> {status}")
    print("-" * 72)

    final_code = 0 if all(code == 0 for _, _, code in summary) else 1
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
