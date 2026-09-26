"""One decision day of the live frozen path: levered runner inputs from the frozen book."""

from __future__ import annotations

import json
import logging
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
    classify_snapshot_gaps,
    crypto_census,
    extend_unit_history,
    snapshot_gap_blocked_decisions,
    unit_proxy_returns,
)
from src.live.ledger import load_ledger
from src.live.venue_listing import (
    SettlementEvidence,
    VenueListingSnapshot,
    delisting_blocked_decisions,
    latest_venue_listing,
    load_venue_listing_history,
    settlement_evidence_from_bars,
)
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

_logger = logging.getLogger(__name__)


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
    venue_snapshot_age_days: float = 0.0
    decision_bar_missing: int = 0
    venue_gap_excluded: tuple[str, ...] = ()


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


def _resolve_venue(
    venue_dir: Path, fallback: Path, *, decision_day: pd.Timestamp | None = None,
    max_age: pd.Timedelta | None = None,
) -> tuple[str, VenueRuleSnapshot, float]:
    """Newest snapshot under ``venue_dir`` (or the fallback), rejected when older than ``max_age``.

    Age is ``decision_day - snapshot.captured_at``. The deploy fallback is dated too, so an empty
    directory cannot run forever on a stale ladder. ``None`` for both new arguments preserves
    the legacy two-tuple behavior contract with a reported age of 0.0.

    Raises:
        DataIntegrityError: no snapshot at all, or the chosen snapshot is older than ``max_age``.
    """
    candidates = venue_rule_snapshot_paths(Path(venue_dir))
    if candidates:
        chosen = candidates[-1]
    else:
        chosen = Path(fallback)
        if not chosen.exists():
            raise DataIntegrityError("no venue snapshot at all")
    snapshot = load_venue_rule_snapshot(chosen)
    age_days = 0.0
    if decision_day is not None:
        day = pd.Timestamp(decision_day)
        day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
        age = day - snapshot.captured_at.tz_convert("UTC")
        age_days = float(age.total_seconds()) / 86400.0
        if max_age is not None and age > max_age:
            max_days = float(pd.Timedelta(max_age).total_seconds()) / 86400.0
            raise DataIntegrityError(
                f"venue snapshot age {age_days:.1f} days exceeds max_age {max_days:.1f} days "
                f"(snapshot={chosen.name})",
            )
    return chosen.name, snapshot, age_days


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
    listing_root: Path | None = None,
    delisting_block_lead: pd.Timedelta | None = None,
    listing_max_age: pd.Timedelta | None = None,
    settlement_min_flat_bars: int | None = None,
    settlement_price_rtol: float | None = None,
    decision_bar_max_missing_fraction: float | None = None,
    venue_max_age: pd.Timedelta | None = None,
) -> FrozenStepReport:
    """Compute and persist one decision day's levered frozen target row for the runner.

    Rebuilds the frozen book from the trailing 1h window, extends the unit-return history
    with the live proxy (append-only), sizes exposure with the account policy's posterior
    half-Kelly inside venue margin and impact limits, and appends the levered weight row
    and its snapshot-close sizing row -- sealed (AES-256-GCM) when ``artifact_key`` is set,
    plaintext otherwise. Orders are never placed here.

    ``account_equity_usdt`` is the LIVE account's margin equity; when given it is the sizing equity for exposure. PAPER/SHADOW pass None and size from the virtual ledger (cash plus positions at snapshot closes), or from ``seed_equity_usdt`` before the first fill.

    When ``listing_root`` is given, announced delistings are withdrawn point-in-time from the book
    (see ``delisting_blocked_decisions``). Held or previously booked symbols past delivery are
    valued at their venue-evidenced settlement price for both the unit proxy and sizing equity.
    Newly scored forward unit returns are persisted as soon as the history extends, before any
    later check can fail, so a HALT later in the step never re-anchors the next attempt on a stale
    window.

    When ``decision_bar_max_missing_fraction`` is given, the decision-day snapshot bar is
    gated before the book: census symbols lacking the raw bar are classified into
    ``refresh_incomplete`` (tail not yet refreshed) versus ``venue_gap`` (a later bar evidences
    a permanent hole). A held symbol or a nonzero symbol of the latest deployed row that
    lacks the bar fails closed (``decision_bar_missing required=<csv> kind=...``); systemic
    incompleteness past the fraction fails closed (``decision_bar_missing census=<n>/<m>``).
    Otherwise non-held venue-gap symbols are excluded from the ranking with target weight 0
    (their snapshot close stays NaN, never filled) and reported in ``venue_gap_excluded``.
    ``venue_snapshot_age_days`` is always reported; sizing halts only past ``venue_max_age``.
    ``None`` for either new argument preserves today's behavior.

    Raises:
        CausalityViolation: ``now`` precedes the decision day's release hour.
        DataIntegrityError: no book row for ``decision_day``, a unit-history gap, a held
            symbol without a snapshot close, no venue snapshot at all, or non-finite sizing;
            plus a stale or missing listing snapshot, or a held symbol past
            delivery whose settlement price is not yet evidenced ("delisted holding without settlement
            evidence: <symbols>"); plus a stale venue ladder past ``venue_max_age``, or a
            missing decision bar per the gate above.
    """
    day = pd.Timestamp(decision_day)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    day = day.normalize()
    at = pd.Timestamp(now)
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    release = day + pd.Timedelta(hours=int(FROZEN_MHS_TOP20_V2.release_hour_utc))
    if at < release:
        raise CausalityViolation(f"frozen signal for {day.date()} not released until {release}")
    if account_equity_usdt is not None and (
        not math.isfinite(float(account_equity_usdt)) or float(account_equity_usdt) <= 0.0
    ):
        raise DataIntegrityError(f"account equity not finite and positive: {account_equity_usdt!r}")
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
    listing_history: tuple[VenueListingSnapshot, ...] = ()
    if listing_root is not None:
        listing_history = load_venue_listing_history(Path(listing_root), through_day=day)
        if listing_max_age is not None:
            latest_venue_listing(Path(listing_root), now=at, max_age=listing_max_age)
    ledger_state = load_ledger(Path(ledger_path))
    ledger_positions = {
        str(symbol): float(quantity)
        for symbol, quantity in ledger_state.positions.items() if quantity != 0
    }
    snapshot_hour = int(FROZEN_MHS_TOP20_V2.snapshot_hour_utc)
    gap_refresh: tuple[str, ...] = ()
    gap_venue: tuple[str, ...] = ()
    if decision_bar_max_missing_fraction is not None:
        trading_census = census
        if listing_root is not None and len(listing_history) > 0:
            in_force = listing_history[-1]
            trading_names: list[str] = []
            for name in census:
                entry = in_force.entries.get(name)
                # exchangeInfo에서 사라진(purge된) 심볼은 거래 불가: 디스크에 남은 1h 파일 때문에
                # refresh_incomplete로 집계되어 결정봉 결손 비율을 부풀리지 않도록 census에서 뺀다.
                if entry is not None and entry.status == "TRADING":
                    trading_names.append(name)
            trading_census = tuple(trading_names)
        snapshot_bar = day + pd.Timedelta(hours=snapshot_hour)
        gap_report = classify_snapshot_gaps(Path(data_root), trading_census, snapshot_bar=snapshot_bar)
        gap_refresh = gap_report.refresh_incomplete
        gap_venue = gap_report.venue_gap
        deployed_nonzero: set[str] = set()
        deployed_frame = load_weights_frame(Path(weights_path), artifact_key=artifact_key)
        if not deployed_frame.empty:
            latest_row = deployed_frame.iloc[-1]
            for column in deployed_frame.columns:
                try:
                    deployed_weight = float(latest_row[column])
                except (TypeError, ValueError):  # pragma: no cover - deployed rows are numeric by construction
                    continue
                if deployed_weight != 0.0 and math.isfinite(deployed_weight):
                    deployed_nonzero.add(str(column))
        required_names = set(ledger_positions) | deployed_nonzero
        gap_names = set(gap_refresh) | set(gap_venue)
        required_missing = sorted(required_names & gap_names)
        if required_missing:
            refresh_set = set(gap_refresh)
            kind = "refresh_incomplete" if any(name in refresh_set for name in required_missing) else "venue_gap"
            raise DataIntegrityError(
                f"decision_bar_missing required={','.join(required_missing)} kind={kind}",
            )
        if len(trading_census) and (
            len(gap_refresh) / len(trading_census) > float(decision_bar_max_missing_fraction)
        ):
            raise DataIntegrityError(
                f"decision_bar_missing census={len(gap_refresh)}/{len(trading_census)}",
            )
        for excluded in gap_venue:
            _logger.info(
                "[ALGO] venue_gap_excluded symbol=%s snapshot_bar=%s",
                excluded, snapshot_bar.isoformat(),
            )
    blocked_callable = None
    if listing_root is not None or decision_bar_max_missing_fraction is not None:
        block_lead = delisting_block_lead if delisting_block_lead is not None else pd.Timedelta(hours=48)
        holding_end_offset = pd.Timedelta(
            days=1, hours=int(FROZEN_MHS_TOP20_V2.snapshot_hour_utc),
        )
        gap_active = decision_bar_max_missing_fraction is not None

        def _blocked(
            decision_index: pd.DatetimeIndex, census_order: tuple[str, ...],
        ) -> pd.DataFrame:
            frames: list[pd.DataFrame] = []
            if listing_root is not None:
                frames.append(
                    delisting_blocked_decisions(
                        listing_history, decision_index, census_order,
                        holding_end_offset=holding_end_offset, lead=block_lead,
                    )
                )
            if gap_active:
                frames.append(
                    snapshot_gap_blocked_decisions(
                        Path(data_root), decision_index, census_order,
                        snapshot_hour=snapshot_hour,
                    )
                )
            combined = frames[0]
            for extra in frames[1:]:
                combined = combined | extra
            return combined

        blocked_callable = _blocked
    book = build_live_frozen_book(
        Path(data_root), census, panel_start=panel_start, panel_end=panel_end,
        blocked_decisions=blocked_callable,
    )
    if day not in book.unit_weights.index:
        raise DataIntegrityError(f"no book row for {day.date()}")
    # 북은 창 안에 봉이 있는 심볼로 census를 좁히므로, 이후 정렬 기준은 북의 컬럼이다.
    census = tuple(str(symbol) for symbol in book.unit_weights.columns)
    settlements: dict[str, SettlementEvidence] = {}
    if listing_root is not None and len(listing_history) > 0:
        in_force = listing_history[-1]
        min_flat = settlement_min_flat_bars if settlement_min_flat_bars is not None else 3
        price_rtol = settlement_price_rtol if settlement_price_rtol is not None else 1e-9
        booked_nonzero = set(
            book.unit_weights.columns[(book.unit_weights != 0.0).any(axis=0).to_numpy(dtype=bool)],
        )
        for symbol in sorted(set(ledger_positions) | {str(s) for s in booked_nonzero}):
            entry = in_force.entries.get(symbol)
            if entry is None or entry.delivery_time is None:
                continue
            if not (at > entry.delivery_time):
                continue
            evidence = settlement_evidence_from_bars(
                Path(data_root) / "ohlcv" / "1h" / f"{symbol}.parquet",
                symbol=symbol, delivery_time=entry.delivery_time,
                min_flat_bars=min_flat, price_rtol=price_rtol,
            )
            if evidence is not None:
                settlements[symbol] = evidence
            elif symbol in ledger_positions:
                raise DataIntegrityError(
                    f"delisted holding without settlement evidence: {symbol}",
                )
    proxy = unit_proxy_returns(
        book, _load_live_funding(Path(data_root), census), cost_bps=LIVE_UNIT_PROXY_COST_BPS,
        settlements=settlements,
    )
    history = extend_unit_history(bootstrap, forward, proxy)
    # HALT 이후 재시도가 낡은 창으로 되돌아가지 않도록, 이후 점검이 실패하기 전에 forward를 먼저 영속화한다.
    forward_new = history[history.index > bootstrap.index[-1]]
    forward_path.parent.mkdir(parents=True, exist_ok=True)
    forward_tmp = forward_path.with_suffix(forward_path.suffix + ".tmp")
    forward_new.to_frame("unit_return").to_parquet(forward_tmp, index=True)
    os.replace(forward_tmp, forward_path)
    past = history[history.index <= day]
    moments = bayesian_unit_moments(
        len(past), float(past.sum()), float((past**2).sum()),
        prior_days=ACCOUNT_PRIOR_DAYS, min_moment_days=ACCOUNT_MIN_MOMENT_DAYS,
    )
    state = ledger_state
    positions = dict(ledger_positions)
    unit_row = book.unit_weights.loc[day]
    snap_row = book.snapshot_closes.loc[day]
    if settlements:
        snap_row = snap_row.copy()
        for settled_symbol, evidence in settlements.items():
            if settled_symbol in snap_row.index and not math.isfinite(float(snap_row[settled_symbol])):
                snap_row[settled_symbol] = float(evidence.price)
    needed = {str(symbol) for symbol in book.unit_weights.columns if float(unit_row[str(symbol)]) != 0.0} | set(positions)
    for symbol in needed:
        if symbol not in snap_row.index or not math.isfinite(float(snap_row[symbol])):
            raise DataIntegrityError(f"held symbol without a snapshot close: {symbol}")
    if account_equity_usdt is not None:
        equity = float(account_equity_usdt)
    elif state.cash_usdt is None:
        equity = float(seed_equity_usdt)
    else:
        equity = float(state.cash_usdt) + sum(
            quantity * float(snap_row[symbol]) for symbol, quantity in positions.items()
        )
    venue_name, rules, venue_age_days = _resolve_venue(
        Path(venue_rules_dir), Path(fallback_venue_path), decision_day=day, max_age=venue_max_age,
    )
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
        venue_snapshot_age_days=float(venue_age_days),
        decision_bar_missing=len(gap_refresh),
        venue_gap_excluded=tuple(sorted(gap_venue)),
    )
    existing = load_weights_frame(Path(weights_path), artifact_key=artifact_key)
    if not existing.empty and day in existing.index:
        return replace(report, written=False)
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
        "venue_snapshot_age_days": report.venue_snapshot_age_days,
        "decision_bar_missing": report.decision_bar_missing,
        "venue_gap_excluded": sorted(report.venue_gap_excluded),
        "written": report.written,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(report_tmp, report_path)
    return report
