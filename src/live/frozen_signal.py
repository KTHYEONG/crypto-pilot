"""One decision day of the live frozen path: levered runner inputs from the frozen book."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.live.deployed_weights import append_weight_row, decision_ohlcv_close_path, load_weights_frame
from src.live.errors import ArtifactSealError, CausalityViolation
from src.live.frozen_book import (
    build_live_frozen_book,
    crypto_census,
    extend_unit_history,
    unit_proxy_returns,
)
from src.live.ledger import load_ledger
from src.market_data.binance.venue_rules import (
    VenueRuleSnapshot,
    load_venue_rule_snapshot,
    venue_rule_snapshot_paths,
)
from src.market_data.storage.loaders import load_funding_rates
from src.market_data.storage.ohlcv import is_temp_artifact
from src.mhs.account_policy import (
    account_growth_policy,
    bayesian_unit_moments,
    build_venue_ladders,
    choose_exposure,
)
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2
from src.mhs.params import (
    ACCOUNT_MIN_MOMENT_DAYS,
    ACCOUNT_PRIOR_DAYS,
    LIVE_FROZEN_WARMUP_DAYS,
    LIVE_UNIT_PROXY_COST_BPS,
)

FROZEN_SIGNAL_REPORT_NAME: str = "frozen_signal_report.json"


@dataclass(frozen=True, slots=True)
class FrozenStepReport:
    """Outcome of one frozen signal step: sizing inputs, exposure, and write status."""

    decision_day: pd.Timestamp
    exposure: float
    equity_usdt: float
    gross_weight: float
    names: int
    unit_observations: int
    posterior_mean: float | None
    posterior_sigma: float | None
    unit_history_end: pd.Timestamp
    venue_snapshot: str
    written: bool


def _read_unit_series(path: Path, label: str, *, artifact_key: SecretStr | None = None) -> pd.Series:
    """Read a ``unit_return`` parquet column as a UTC daily return series.

    ``path`` ending in ``.enc`` is opened as an AES-256-GCM envelope (``artifact_key``
    required); this is how the calibrated bootstrap ships publicly without handing the
    historical performance series to anyone who clones the repository.
    """
    try:
        if str(path).endswith(".enc"):
            if artifact_key is None:
                raise ArtifactSealError(f"sealed artifact requires a key: {path}")
            from src.live.crypto import derive_key, read_sealed_parquet

            frame = read_sealed_parquet(path, derive_key(artifact_key))
        else:
            frame = pd.read_parquet(path)
        values = frame["unit_return"]
    except ArtifactSealError:
        raise
    except Exception as exc:
        raise DataIntegrityError(f"{label} unit returns unreadable: {path}: {exc}") from exc
    index = pd.DatetimeIndex(pd.to_datetime(values.index, utc=True))
    return pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"),
        index=index, name="unit_return",
    ).sort_index()


def _load_live_funding(data_root: Path, census: tuple[str, ...]) -> dict[str, pd.Series]:
    """Funding series present under ``data_root/funding``; absent symbols carry no funding."""
    funding: dict[str, pd.Series] = {}
    for symbol in census:
        path = Path(data_root) / "funding" / f"{symbol}.parquet"
        if path.exists():
            funding[symbol] = load_funding_rates(str(path))
    return funding


def _resolve_venue(venue_dir: Path, fallback: Path) -> tuple[str, VenueRuleSnapshot]:
    """Newest snapshot under ``venue_dir``, or the fallback file when the directory is empty."""
    candidates = venue_rule_snapshot_paths(Path(venue_dir))
    if candidates:
        chosen = candidates[-1]
    else:
        chosen = Path(fallback)
        if not chosen.exists():
            raise DataIntegrityError("no venue snapshot at all")
    return chosen.name, load_venue_rule_snapshot(chosen)


def run_frozen_signal_step(
    decision_day: pd.Timestamp,
    *,
    now: pd.Timestamp,
    data_root: Path,
    weights_path: Path,
    unit_bootstrap_path: Path,
    unit_forward_path: Path,
    venue_rules_dir: Path,
    fallback_venue_path: Path,
    ledger_path: Path,
    seed_equity_usdt: float,
    account_equity_usdt: float | None = None,
    non_crypto: frozenset[str],
    artifact_key: SecretStr | None = None,
) -> FrozenStepReport:
    """Compute and persist one decision day's levered frozen target row for the runner.

    Rebuilds the frozen book from the trailing 1h window, extends the unit-return history
    with the live proxy (append-only), sizes exposure with the account policy's posterior
    half-Kelly inside venue margin and impact limits, and appends the levered weight row
    and its snapshot-close sizing row -- sealed (AES-256-GCM) when ``artifact_key`` is set,
    plaintext otherwise. Orders are never placed here.

    ``account_equity_usdt`` is the LIVE account's margin equity; when given it is the sizing equity for exposure. PAPER/SHADOW pass None and size from the virtual ledger (cash plus positions at snapshot closes), or from ``seed_equity_usdt`` before the first fill.

    Raises:
        CausalityViolation: ``now`` precedes the decision day's release hour.
        DataIntegrityError: no book row for ``decision_day``, a unit-history gap, a held
            symbol without a snapshot close, no venue snapshot at all, or non-finite sizing.
    """
    day = pd.Timestamp(decision_day)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    day = day.normalize()
    at = pd.Timestamp(now)
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    release = day + pd.Timedelta(hours=int(FROZEN_MHS_TOP20_V2.release_hour_utc))
    if at < release:
        raise CausalityViolation(f"frozen signal for {day.date()} not released until {release}")
    hourly_dir = Path(data_root) / "ohlcv" / "1h"
    census = crypto_census(
        [path.stem for path in hourly_dir.glob("*.parquet") if not is_temp_artifact(path.name)],
        non_crypto,
    )
    bootstrap = _read_unit_series(Path(unit_bootstrap_path), "bootstrap", artifact_key=artifact_key)
    forward_path = Path(unit_forward_path)
    forward = _read_unit_series(forward_path, "forward") if forward_path.exists() else pd.Series(
        dtype="float64", index=pd.DatetimeIndex([], tz="UTC"), name="unit_return",
    )
    first_needed = (forward.index[-1] - pd.Timedelta(days=2)) if len(forward) else (
        bootstrap.index[-1] - pd.Timedelta(days=1)
    )
    panel_start = min(day, first_needed) - pd.Timedelta(days=int(LIVE_FROZEN_WARMUP_DAYS))
    panel_end = day + pd.Timedelta(days=1, hours=int(FROZEN_MHS_TOP20_V2.snapshot_hour_utc) + 1)
    book = build_live_frozen_book(Path(data_root), census, panel_start=panel_start, panel_end=panel_end)
    if day not in book.unit_weights.index:
        raise DataIntegrityError(f"no book row for {day.date()}")
    # 북은 창 안에 봉이 있는 심볼로 census를 좁히므로, 이후 정렬 기준은 북의 컬럼이다.
    census = tuple(str(symbol) for symbol in book.unit_weights.columns)
    proxy = unit_proxy_returns(
        book, _load_live_funding(Path(data_root), census), cost_bps=LIVE_UNIT_PROXY_COST_BPS,
    )
    history = extend_unit_history(bootstrap, forward, proxy)
    past = history[history.index <= day]
    moments = bayesian_unit_moments(
        len(past), float(past.sum()), float((past**2).sum()),
        prior_days=ACCOUNT_PRIOR_DAYS, min_moment_days=ACCOUNT_MIN_MOMENT_DAYS,
    )
    state = load_ledger(Path(ledger_path))
    positions = {str(symbol): float(quantity) for symbol, quantity in state.positions.items() if quantity != 0}
    unit_row = book.unit_weights.loc[day]
    snap_row = book.snapshot_closes.loc[day]
    needed = {str(symbol) for symbol in book.unit_weights.columns if float(unit_row[str(symbol)]) != 0.0} | set(positions)
    for symbol in needed:
        if symbol not in snap_row.index or not math.isfinite(float(snap_row[symbol])):
            raise DataIntegrityError(f"held symbol without a snapshot close: {symbol}")
    if account_equity_usdt is not None:
        if not math.isfinite(float(account_equity_usdt)) or float(account_equity_usdt) <= 0.0:
            raise DataIntegrityError(f"account equity not finite and positive: {account_equity_usdt!r}")
        equity = float(account_equity_usdt)
    elif state.cash_usdt is None:
        equity = float(seed_equity_usdt)
    else:
        equity = float(state.cash_usdt) + sum(
            quantity * float(snap_row[symbol]) for symbol, quantity in positions.items()
        )
    venue_name, rules = _resolve_venue(Path(venue_rules_dir), Path(fallback_venue_path))
    ladders, _ = build_venue_ladders(list(census), rules)
    census_list = list(census)
    unit_values = unit_row[census_list].to_numpy(dtype="float64")
    held_notional = np.array(
        [positions.get(symbol, 0.0) * float(snap_row[symbol]) if positions.get(symbol, 0.0) != 0.0 else 0.0
         for symbol in census_list],
        dtype="float64",
    )
    exposure = choose_exposure(
        unit_values, equity, held_notional,
        book.adv.loc[day, census_list].to_numpy(dtype="float64"),
        book.daily_sigma.loc[day, census_list].to_numpy(dtype="float64"),
        ladders, account_growth_policy(), moments,
    )
    if not (math.isfinite(equity) and math.isfinite(exposure)):
        raise DataIntegrityError(f"non-finite sizing for {day.date()}")
    levered = unit_row[census_list] * float(exposure)
    gross = float(np.abs(levered.to_numpy(dtype="float64")).sum())
    names = int((levered.to_numpy(dtype="float64") != 0.0).sum())
    report = FrozenStepReport(
        decision_day=day, exposure=float(exposure), equity_usdt=float(equity),
        gross_weight=gross, names=names, unit_observations=len(past),
        posterior_mean=None if moments is None else float(moments.mean),
        posterior_sigma=None if moments is None else float(moments.sigma),
        unit_history_end=history.index[-1], venue_snapshot=venue_name, written=True,
    )
    existing = load_weights_frame(Path(weights_path), artifact_key=artifact_key)
    if not existing.empty and day in existing.index:
        return replace(report, written=False)
    forward_new = history[history.index > bootstrap.index[-1]]
    forward_path.parent.mkdir(parents=True, exist_ok=True)
    forward_tmp = forward_path.with_suffix(forward_path.suffix + ".tmp")
    forward_new.to_frame("unit_return").to_parquet(forward_tmp, index=True)
    os.replace(forward_tmp, forward_path)
    # 행 병합은 컬럼 합집합이라, 오늘 census에 없는 과거 심볼이 NaN으로 남으면 러너 사이징이 깨진다.
    # 오늘 census 밖 = 보유 목표 0이므로 명시적 0.0으로 채운다.
    prior_columns = [str(c) for c in existing.columns] if not existing.empty else []
    levered = levered.reindex(sorted(set(prior_columns) | set(census_list)), fill_value=0.0)
    append_weight_row(Path(weights_path), day, levered, artifact_key=artifact_key)
    sizing = snap_row[[symbol for symbol in census_list if float(unit_row[symbol]) != 0.0 or symbol in positions]]
    append_weight_row(decision_ohlcv_close_path(Path(weights_path)), day, sizing, artifact_key=artifact_key)
    report_path = Path(weights_path).parent / FROZEN_SIGNAL_REPORT_NAME
    payload = {
        "decision_day": report.decision_day.isoformat(),
        "exposure": report.exposure,
        "equity_usdt": report.equity_usdt,
        "gross_weight": report.gross_weight,
        "names": report.names,
        "unit_observations": report.unit_observations,
        "posterior_mean": report.posterior_mean,
        "posterior_sigma": report.posterior_sigma,
        "unit_history_end": report.unit_history_end.isoformat(),
        "venue_snapshot": report.venue_snapshot,
        "written": report.written,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(report_tmp, report_path)
    return report
