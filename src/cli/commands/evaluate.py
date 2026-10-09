"""Local strategy evaluation standard: judge the book that trades."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import BACKTESTS_DIR
from src.strategy.features import FEATURE_REGISTRY
from src.strategy.targets import (
    FLOW_MOM_TOP20,
    FLOW_MOM_TOP20_GROWTH,
    FLOW_MOM_TOP40_CONTROL,
    resolve_strategy_id,
)

logger = logging.getLogger("EvaluateCli")

_EXIT_BY_VERDICT = {"accept": 0, "inconclusive": 2, "reject": 3, "invalid": 4}


def add_evaluate_commands(evaluate_parser: argparse.ArgumentParser) -> None:
    """Register the local evaluation standard leaves (never reads paper/live ledgers)."""
    sub = evaluate_parser.add_subparsers(dest="command", required=True)
    strategy = sub.add_parser(
        "strategy",
        help="Judge a locally researched strategy on historical data only.",
        description="Judge the same book that trades: unit book for edge, deployed sizing book for risk.",
    )
    strategy.add_argument("--strategy", required=True, help="Canonical release strategy id.")
    strategy.add_argument("--unit-run", required=True, help="Finished `backtest strategy` run directory.")
    strategy.add_argument("--stress-from-unit", action="store_true", default=False)
    strategy.add_argument("--account-run", required=True, help="Finished `backtest account` run directory.")
    strategy.add_argument("--neighbors", nargs="*", default=[], help="Finished neighbor `backtest strategy` run directories.")
    strategy.add_argument("--holdout-run", default=None, help="Finished holdout `backtest strategy` run directory (one look).")
    strategy.add_argument("--accept", action="store_true", default=False)
    strategy.set_defaults(handler=run_evaluate_strategy_command)


def _family_of(strategy_id: str) -> str:
    if strategy_id.startswith("flow_mom"):
        return "flow_mom"
    return strategy_id


def _utc_series(values: pd.Series, label: str) -> pd.Series:
    if not isinstance(values, pd.Series) or values.empty:
        raise DataIntegrityError(f"{label} must be a non-empty Series")
    series = pd.Series(values.to_numpy(dtype="float64"), index=values.index, dtype="float64")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataIntegrityError(f"{label} must carry a DatetimeIndex")
    index = series.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    series.index = index.tz_convert("UTC")
    return series


def _read_unit_run(run_dir: Path) -> tuple[dict[str, Any], pd.Series, pd.Series, pd.Series, dict[str, float]]:
    payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(run_dir / "daily.parquet")
    base = _utc_series(pd.Series(daily["base_return"].to_numpy(dtype="float64"), index=daily.index), "base_returns")
    stress = _utc_series(pd.Series(daily["stress_return"].to_numpy(dtype="float64"), index=daily.index), "stress_returns")
    if "fill_participation" in daily.columns:
        participation = _utc_series(
            pd.Series(daily["fill_participation"].to_numpy(dtype="float64"), index=daily.index), "participation"
        )
    else:
        participation = pd.Series(dtype="float64")
    funding_by_symbol: dict[str, float] = {}
    try:
        evaluation = payload.get("report_periods", {}).get("evaluation", {})
        per_symbol = evaluation.get("funding_by_symbol", {}).get("stress", {})
        if isinstance(per_symbol, dict):
            funding_by_symbol = {str(k): float(v) for k, v in per_symbol.items() if v is not None}
    except (AttributeError, TypeError, ValueError):
        funding_by_symbol = {}
    if not funding_by_symbol:
        statistics = payload.get("statistics", {}).get("stress", {})
        raw = statistics.get("funding_by_symbol") if isinstance(statistics, dict) else None
        if isinstance(raw, dict):
            funding_by_symbol = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    return payload, base, stress, participation, funding_by_symbol


def _read_neighbor_stress(run_dir: Path) -> pd.Series:
    daily = pd.read_parquet(Path(run_dir) / "daily.parquet")
    return _utc_series(pd.Series(daily["stress_return"].to_numpy(dtype="float64"), index=daily.index), "neighbor")


def _read_account_run(run_dir: Path) -> tuple[pd.Series, float]:
    frame = pd.read_parquet(Path(run_dir) / "account_daily.parquet")
    equity = pd.Series(frame["equity"].to_numpy(dtype="float64"), index=frame.index, dtype="float64")
    equity.index = equity.index.tz_localize("UTC") if equity.index.tz is None else equity.index.tz_convert("UTC")
    payload = json.loads((Path(run_dir) / "account.json").read_text(encoding="utf-8"))
    capital = float(payload["capital"])
    previous = equity.shift(1).fillna(capital)
    deployed = equity / previous - 1.0
    deployed = _utc_series(deployed, "deployed_returns")
    leverage = 1.0
    if "exposure" in frame.columns:
        leverage = float(pd.Series(frame["exposure"].to_numpy(dtype="float64")).abs().max())
    return deployed, leverage


def _causality_ok(strategy_id: str) -> bool:
    canonical = resolve_strategy_id(strategy_id)
    specs = {
        "flow_mom_top20": FLOW_MOM_TOP20,
        "flow_mom_top40_control": FLOW_MOM_TOP40_CONTROL,
        "flow_mom_top20_growth": FLOW_MOM_TOP20_GROWTH,
    }
    spec = specs.get(canonical)
    if spec is None:
        return canonical == "flow_mom_top20"
    registered = {feature.name for feature in FEATURE_REGISTRY}
    if not all(m.name in registered for m in spec.members):
        return False
    from src.strategy.targets import build_strategy_targets

    index = pd.date_range("2021-01-01", periods=2400, freq="h", tz="UTC")
    symbols = tuple(f"S{i:02}" for i in range(40))
    rng = np.random.default_rng(20261008)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.002, (2400, 40)), axis=0)), index=index, columns=symbols)
    volume = pd.DataFrame(rng.uniform(100_000, 300_000, (2400, 40)), index=index, columns=symbols)
    panels = {"close": close, "quote_vol": volume, "taker_buy_quote": volume * rng.uniform(0.3, 0.7, (2400, 40))}
    available = pd.DataFrame({symbol: index + pd.Timedelta(hours=1) for symbol in symbols}, index=index)
    daily_close = close.resample("1D").last()
    daily_volume = volume.resample("1D").sum()
    original = build_strategy_targets(panels, available, daily_close, daily_volume, symbols, market_close=close, strategy=spec)
    release_time = index[94 * 24 + spec.release_hour_utc]
    perturbed = {key: value.copy() for key, value in panels.items()}
    for value in perturbed.values():
        value.loc[value.index >= release_time] *= 7
    changed = build_strategy_targets(
        perturbed, available, daily_close, daily_volume, symbols, market_close=perturbed["close"], strategy=spec,
    )
    prefix = original.target_weights.index <= release_time.normalize() + pd.Timedelta(days=1)
    before = original.target_weights.loc[prefix]
    return bool(before.abs().to_numpy().sum() > 0 and before.equals(changed.target_weights.loc[prefix]))


def _declared_book(payload: dict[str, Any], release_id: str) -> bool:
    spec = {"flow_mom_top20": FLOW_MOM_TOP20, "flow_mom_top20_growth": FLOW_MOM_TOP20_GROWTH,
            "flow_mom_top40_control": FLOW_MOM_TOP40_CONTROL}.get(release_id)
    if spec is None:
        return False
    members = [{"name": member.name, "sign": member.sign} for member in spec.members]
    return bool(payload.get("breadth") == spec.breadth and payload.get("members") == members
                and payload.get("min_rank_symbols") == spec.min_rank_symbols
                and payload.get("design_data_cutoff") == spec.design_data_cutoff.isoformat())


def build_evaluation_inputs(
    *,
    strategy_id: str,
    unit_run: Path,
    account_run: Path,
    neighbor_runs: tuple[Path, ...] = (),
    holdout_run: Path | None = None,
) -> Any:
    """Assemble pure evaluation inputs from finished run directories (no live I/O)."""
    from src.evaluation.standard import EvaluationInputs
    from src.evaluation.trials import trial_population
    from src.strategy.release import load_release, releases_dir

    release = load_release(strategy_id)
    unit_payload, base, stress, participation, funding_by_symbol = _read_unit_run(Path(unit_run))
    deployed, leverage = _read_account_run(Path(account_run))
    account_payload = json.loads((Path(account_run) / "account.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(Path(unit_run) / "daily.parquet")
    columns = [column for column in daily if column.startswith("stress_funding_income_")]
    funding_daily = daily[columns].rename(columns=lambda column: column.removeprefix("stress_funding_income_")) if columns else None
    neighbors = tuple(_read_neighbor_stress(p) for p in neighbor_runs)
    holdout_returns: pd.Series | None = None
    holdout_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if holdout_run is not None:
        holdout_payload = json.loads((Path(holdout_run) / "result.json").read_text(encoding="utf-8"))
        holdout_window = (
            pd.Timestamp(str(holdout_payload["evaluation_start"])).tz_convert("UTC"),
            pd.Timestamp(str(holdout_payload["evaluation_end"])).tz_convert("UTC"),
        )
        from src.evaluation.holdout import holdout_look_recorded
        if not holdout_look_recorded(release.strategy_id, release.spec_digest, holdout_window):
            raise DataIntegrityError("holdout run has no matching one-look journal entry")
        holdout_daily = pd.read_parquet(Path(holdout_run) / "daily.parquet")
        holdout_returns = _utc_series(
            pd.Series(holdout_daily["base_return"].to_numpy(dtype="float64"), index=holdout_daily.index),
            "holdout_returns",
        )
    discovery = (
        pd.Timestamp(str(unit_payload["evaluation_start"])).tz_convert("UTC"),
        pd.Timestamp(str(unit_payload["evaluation_end"])).tz_convert("UTC"),
    )
    import numpy as np

    base_arr = base.to_numpy(dtype="float64")
    skew = float(pd.Series(base_arr).skew())
    kurtosis = float(pd.Series(base_arr).kurt() + 3.0)
    population = trial_population(
        _family_of(release.strategy_id),
        path=releases_dir() / f"{_family_of(release.strategy_id)}.trials.jsonl",
        candidate_sharpe=float(base_arr.mean() / np.std(base_arr, ddof=1)) if np.std(base_arr, ddof=1) > 1e-12 else 0.0,
        candidate_n_obs=int(base.size),
        candidate_skew=skew if skew == skew else 0.0,
        candidate_kurtosis=kurtosis if kurtosis == kurtosis else 3.0,
    )
    limitations = unit_payload.get("limitations", [])
    lake_ok = bool(
        int(unit_payload.get("source_gap_excluded_count", 0)) == 0
        and "DATA_AVAILABILITY_SELECTION" not in limitations
    )
    ledger_ok = unit_payload.get("ledger_certified") is True
    identity_ok = (
        _declared_book(unit_payload, release.strategy_id)
        and
        unit_payload.get("strategy_id") == release.strategy_id
        and unit_payload.get("name_clip") == release.sizing.get("name_clip")
        and unit_payload.get("exposure_multiplier") == 1.0
        and account_payload.get("capital") == release.target_capital_usdt
        and isinstance(account_payload.get("execution"), dict)
        and account_payload["execution"].get("mode") == "maker"
        and account_payload.get("evaluation_start") == unit_payload.get("evaluation_start")
        and account_payload.get("evaluation_end") == unit_payload.get("evaluation_end")
    )
    if holdout_run is not None:
        identity_ok = identity_ok and _declared_book(holdout_payload, release.strategy_id) and holdout_payload.get("ledger_certified") is True
    from src.cli.commands.backtest import _neighbor_specs
    expected = _neighbor_specs(FLOW_MOM_TOP20)
    observed = []
    for path in neighbor_runs:
        neighbor_payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
        observed.append((neighbor_payload.get("breadth"), neighbor_payload.get("members")))
        identity_ok = identity_ok and neighbor_payload.get("ledger_certified") is True
    declared = [(neighbor.breadth, [{"name": m.name, "sign": m.sign} for m in neighbor.members]) for neighbor in expected]
    if len(observed) != len(declared) or any(observed.count(item) != 1 for item in declared):
        identity_ok = False
    return EvaluationInputs(
        strategy_id=release.strategy_id,
        spec_digest=release.spec_digest,
        discovery=discovery,
        holdout=holdout_window,
        design_data_cutoff=release.design_data_cutoff,
        base_returns=base,
        stress_returns=stress,
        funding_by_symbol=funding_by_symbol,
        participation=participation,
        ledger_certified=ledger_ok,
        lake_coverage_ok=lake_ok,
        causality_ok=_causality_ok(release.strategy_id),
        deployed_returns=deployed,
        deployed_max_leverage=leverage,
        neighbors=neighbors,
        trial_population=population,
        holdout_returns=holdout_returns,
        funding_income_daily=funding_daily,
        book_identity_ok=identity_ok,
    )


def run_evaluate_strategy_command(args: argparse.Namespace) -> None:
    """Print the verdict table, persist the evaluation envelope, and map verdicts to exits."""
    from src.evaluation.standard import evaluate_strategy
    from src.strategy.release import criteria_digest, load_release, record_acceptance

    strategy_id = str(args.strategy)
    release = load_release(strategy_id)
    inputs = build_evaluation_inputs(
        strategy_id=strategy_id,
        unit_run=Path(args.unit_run),
        account_run=Path(args.account_run),
        neighbor_runs=tuple(Path(p) for p in (args.neighbors or [])),
        holdout_run=Path(args.holdout_run) if args.holdout_run else None,
    )
    evaluation = evaluate_strategy(inputs, release.criteria)
    for check in evaluation.checks:
        print(f"{check.group:10s} {check.code:24s} value={check.value} threshold={check.threshold} " f"passed={check.passed} :: {check.reason}")  # noqa: T201
    print(f"verdict={evaluation.verdict} digest={evaluation.digest}")  # noqa: T201
    out_dir = BACKTESTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{evaluation.strategy_id}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "strategy_id": evaluation.strategy_id,
                "spec_digest": evaluation.spec_digest,
                "criteria_digest": evaluation.criteria_digest,
                "verdict": str(evaluation.verdict),
                "digest": evaluation.digest,
                "n_trials": evaluation.n_trials,
                "checks": [
                    {
                        "code": c.code,
                        "group": c.group,
                        "passed": c.passed,
                        "value": c.value,
                        "threshold": c.threshold,
                        "reason": c.reason,
                    }
                    for c in evaluation.checks
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("[EVAL] evaluate strategy=%s verdict=%s digest=%s path=%s", evaluation.strategy_id, evaluation.verdict, evaluation.digest, out_path)
    if args.accept:
        if str(evaluation.verdict) != "accept":
            raise SystemExit(f"--accept refused: verdict is {evaluation.verdict}")
        if evaluation.spec_digest != release.spec_digest or evaluation.criteria_digest != criteria_digest(release.criteria):
            raise SystemExit("--accept refused: spec or criteria digest mismatch")
        record_acceptance(strategy_id, spec_digest=evaluation.spec_digest, evaluation_digest=evaluation.digest,
                          expected_criteria_digest=evaluation.criteria_digest)
    raise SystemExit(_EXIT_BY_VERDICT[str(evaluation.verdict)])
