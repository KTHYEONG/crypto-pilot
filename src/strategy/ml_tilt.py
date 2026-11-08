"""Flow-momentum ML tilt (candidate C1): LightGBM scores blended into the clipped C0 book.

C1 = clip(0.75 * C0 + 0.25 * ML, name_clip) with each row's gross restored to C0's row gross, where
C0 is the clipped production flow_mom_top20 book and ML is a top-20 rank book of EMA-smoothed
LightGBM scores. The model learns from per-member percentile ranks and forward volatility-scaled
returns, retrained on a fixed schedule with a purged expanding window. Parameters live in the
private strategy store ``ml_tilt`` section. This module is outside the strategy release digest set
(``src/strategy/release.py``), so none of its definitions change the deployed flow_mom_top20 digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.durable_io import durable_write_text
from src.common.errors import DataIntegrityError
from src.strategy.books import clip_names_preserving_gross, rank_weight_book
from src.strategy.features import FEATURE_REGISTRY, MARKET_CLOSE_PANEL
from src.strategy.store import MlTiltValues
from src.strategy.targets import StrategySpec, compute_source_readiness
from src.strategy.universe import build_pit_roster

MEMBER_RANK_PREFIX: Final = "r_"
_REQUIRED_PANELS: Final = ("close", "quote_vol", "taker_buy_quote")
_VOL_WINDOW_BARS: Final = 168
_VOL_MIN_PERIODS: Final = 84
_HOURS_PER_DAY: Final = 24
_DAY: Final = pd.Timedelta(days=1)
EMA_WARMUP_HALFLIVES: Final = 10


@dataclass(frozen=True, slots=True)
class MlTiltModel:
    """One boundary-trained LightGBM model with its digest-verified serialized text."""

    boundary: pd.Timestamp
    train_end: pd.Timestamp
    n_rows: int
    feature_names: tuple[str, ...]
    values_digest: str
    lightgbm_version: str
    model_text: str
    sha256: str


def member_rank_features(
    hourly_panels: Mapping[str, pd.DataFrame],
    hourly_available_at: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_quote_volume: pd.DataFrame,
    census: tuple[str, ...],
    *,
    market_close: pd.DataFrame,
    strategy: StrategySpec,
    values: MlTiltValues,
    blocked_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-member percentile-rank features over the PIT roster, indexed by (decision, symbol).

    Raises:
        DataIntegrityError: Source planes or market census are misaligned.
        ValueError: The strategy references an unregistered feature.
    """
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    if any(m.name not in registry for m in strategy.members):
        raise ValueError("strategy references an unregistered feature")
    if any(p not in hourly_panels for p in _REQUIRED_PANELS):
        raise DataIntegrityError("hourly_panels must contain close, quote_vol, and taker_buy_quote")
    first = hourly_panels["close"]
    if any(
        not hourly_panels[p].index.equals(first.index) or list(hourly_panels[p].columns) != list(first.columns)
        for p in _REQUIRED_PANELS
    ):
        raise DataIntegrityError("hourly panels must share an identical index and column order")
    if not hourly_available_at.index.equals(first.index) or list(hourly_available_at.columns) != list(first.columns):
        raise DataIntegrityError("hourly_available_at must align with the hourly panel")
    if not market_close.index.equals(first.index) or list(market_close.columns) != list(census):
        raise DataIntegrityError("market_close must share the hourly index and census columns")

    roster = build_pit_roster(
        daily_close,
        daily_quote_volume,
        census,
        breadth=values.feature_breadth,
        blocked_decisions=blocked_decisions,
    )
    decisions = roster.index[:-1]
    roster_dec = roster.loc[decisions].to_numpy(dtype=bool)
    panels = {**{p: hourly_panels[p] for p in _REQUIRED_PANELS}, MARKET_CLOSE_PANEL: market_close}

    readiness = compute_source_readiness(hourly_available_at, decisions, strategy)
    bar_positions = np.asarray(readiness.bar_positions, dtype=np.int64)
    ready = np.asarray(readiness.ready, dtype=bool)
    census_hourly = first.columns.get_indexer(census)
    census_ready = np.zeros((len(decisions), len(census)), dtype=bool)
    valid_cols = census_hourly >= 0
    census_ready[:, valid_cols] = ready[:, census_hourly[valid_cols]]
    hit = np.flatnonzero(bar_positions >= 0)
    rows_i, rows_j = np.nonzero(roster_dec)

    columns: dict[str, np.ndarray] = {}
    for member in strategy.members:
        feature = registry[member.name].builder(panels)
        aligned = feature.reindex(index=first.index, columns=list(census)).to_numpy(dtype="float64")
        snaps = np.full((len(decisions), len(census)), np.nan, dtype=np.float64)
        snaps[hit] = np.where(census_ready[hit], aligned[bar_positions[hit]], np.nan)
        masked = np.where(roster_dec, snaps, np.nan)
        ranks = pd.DataFrame(masked).rank(axis=1, pct=True).to_numpy(dtype="float64") - 0.5
        columns[f"{MEMBER_RANK_PREFIX}{member.name}"] = ranks[rows_i, rows_j]

    index = pd.MultiIndex.from_arrays(
        [decisions[rows_i], pd.Index([census[j] for j in rows_j])],
        names=["decision", "symbol"],
    )
    return pd.DataFrame(columns, index=index, dtype="float64")


def forward_labels(
    entry_marks: pd.DataFrame,
    funding_by_symbol: Mapping[str, pd.Series],
    hourly_close: pd.DataFrame,
    features: pd.DataFrame,
    *,
    values: MlTiltValues,
    entry_hour_utc: int,
    snapshot_hour_utc: int,
) -> pd.Series:
    """Return demeaned, volatility-scaled forward labels per (decision, symbol); NaN when unobservable.

    ``entry_marks`` row ``t`` must hold the close of the 1h bar labelled ``t - 1h``. The volatility floor
    is the ``vol_floor_quantile`` over the rows passed in, so callers must pass training rows only.
    """
    decisions = pd.DatetimeIndex(features.index.get_level_values("decision"))
    symbols = pd.Index(features.index.get_level_values("symbol"))
    entries = decisions + _DAY + pd.Timedelta(hours=entry_hour_utc)
    exits = entries + pd.Timedelta(days=values.horizon_days)
    start_mark = _positive(_lookup(entry_marks, entries, symbols))
    end_mark = _positive(_lookup(entry_marks, exits, symbols))
    gross_move = end_mark / start_mark - 1.0
    funding = _window_sums(funding_by_symbol, symbols, entries, exits)
    raw = pd.Series(gross_move - funding, index=features.index)
    centered = raw - raw.groupby(level="decision").transform("mean")

    log_close = np.log(hourly_close)
    vol = log_close.diff().rolling(_VOL_WINDOW_BARS, min_periods=_VOL_MIN_PERIODS).std()
    vol168 = _lookup(vol, decisions + pd.Timedelta(hours=snapshot_hour_utc), symbols)
    observed = vol168[np.isfinite(vol168)]
    if observed.size == 0:
        return pd.Series(np.nan, index=features.index, name="label", dtype="float64")
    floor = float(np.quantile(observed, values.vol_floor_quantile))
    denom = np.maximum(vol168, floor) * np.sqrt(_HOURS_PER_DAY * values.horizon_days)
    scaled = np.where(denom > 0.0, centered.to_numpy(dtype="float64") / denom, np.nan)
    clipped = np.clip(scaled, -values.label_clip, values.label_clip)
    return pd.Series(clipped, index=features.index, name="label", dtype="float64")


def ema_warmup_start(values: MlTiltValues, first_decision: pd.Timestamp) -> pd.Timestamp:
    """Return the first decision whose score feeds the EMA of ``first_decision``, clamped to the schedule anchor."""
    lookback = pd.Timedelta(days=math.ceil(values.smooth_halflife_days * EMA_WARMUP_HALFLIVES))
    return max(values.schedule_anchor, pd.Timestamp(first_decision) - lookback)


def ema_warmup_days(values: MlTiltValues) -> int:
    """Calendar days the EMA warm-up reaches before its first kept decision."""
    return math.ceil(values.smooth_halflife_days * EMA_WARMUP_HALFLIVES)


def label_data_end(values: MlTiltValues, boundary: pd.Timestamp, *, entry_hour_utc: int) -> pd.Timestamp:
    """Exclusive source end a ``boundary`` model needs: its latest label exit plus one day of refresh slack.

    The latest training decision is ``boundary - purge_days - 1d``; its label exits at ``decision + 1d +
    entry_hour + horizon_days``. The extra day keeps a lagging per-symbol refresh from freezing a model fit on
    partially published bars, while still letting the model train before the boundary itself.
    """
    cutoff = boundary - pd.Timedelta(days=values.purge_days)
    return cutoff + pd.Timedelta(days=values.horizon_days, hours=entry_hour_utc) + _DAY


def next_boundary(values: MlTiltValues, after: pd.Timestamp) -> pd.Timestamp:
    """First scheduled boundary strictly after ``after`` (the anchor when ``after`` precedes it)."""
    if after < values.schedule_anchor:
        return values.schedule_anchor
    step = pd.Timedelta(days=values.retrain_days)
    count = int((after - values.schedule_anchor) // step) + 1
    return values.schedule_anchor + count * step


def boundary_schedule(values: MlTiltValues, until: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    """Return retrain boundaries ``anchor + k * retrain_days`` that are not after ``until``."""
    if until < values.schedule_anchor:
        return ()
    step = pd.Timedelta(days=values.retrain_days)
    count = int((until - values.schedule_anchor) // step)
    return tuple(values.schedule_anchor + k * step for k in range(count + 1))


def train_boundary_model(
    features: pd.DataFrame,
    labels: pd.Series,
    boundary: pd.Timestamp,
    *,
    values: MlTiltValues,
) -> MlTiltModel:
    """Fit a LightGBM regressor on decisions purged before ``boundary``.

    ``train_end`` is the exclusive decision bound ``boundary - purge_days``.

    Raises:
        DataIntegrityError: Features and labels are misaligned or too few training rows exist.
    """
    if not labels.index.equals(features.index):
        raise DataIntegrityError("labels must share the feature index exactly")
    cutoff = boundary - pd.Timedelta(days=values.purge_days)
    decisions = features.index.get_level_values("decision")
    target = labels.to_numpy(dtype="float64")
    keep = np.asarray(decisions < cutoff, dtype=bool) & np.isfinite(target)
    n_rows = int(keep.sum())
    if n_rows < values.min_train_rows:
        raise DataIntegrityError(f"ml tilt training rows below ml_tilt.min_train_rows: {n_rows}")
    params = {
        "objective": "regression",
        "num_leaves": values.num_leaves,
        "min_data_in_leaf": values.min_child_samples,
        "learning_rate": values.learning_rate,
        "bagging_fraction": values.subsample,
        "bagging_freq": 1,
        "feature_fraction": values.colsample_bytree,
        "lambda_l2": values.reg_lambda,
        "seed": values.seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(features.to_numpy(dtype="float64")[keep], label=target[keep]),
        num_boost_round=values.n_trees,
    )
    feature_names = tuple(str(c) for c in features.columns)
    model_text = booster.model_to_string()
    digest = values_digest(values)
    return MlTiltModel(
        boundary=boundary,
        train_end=cutoff,
        n_rows=n_rows,
        feature_names=feature_names,
        values_digest=digest,
        lightgbm_version=lgb.__version__,
        model_text=model_text,
        sha256=_model_sha(model_text, boundary, feature_names, digest),
    )


def save_model(model: MlTiltModel, directory: Path) -> Path:
    """Write the model as ``<boundary:%Y%m%d>.json`` durably and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.boundary:%Y%m%d}.json"
    payload = {
        "boundary": model.boundary.isoformat(),
        "train_end": model.train_end.isoformat(),
        "n_rows": model.n_rows,
        "feature_names": list(model.feature_names),
        "values_digest": model.values_digest,
        "lightgbm_version": model.lightgbm_version,
        "model_text": model.model_text,
        "sha256": model.sha256,
    }
    durable_write_text(path, json.dumps(payload, sort_keys=True) + "\n")
    return path


@dataclass(frozen=True, slots=True)
class BoundaryTraining:
    """Outcome of one scheduled boundary: ``trained`` (written), ``exists`` (kept), or ``awaiting_data``.

    ``awaiting_data`` means the observed source ends before ``label_data_end``; nothing is written, so a later
    run with complete data trains it instead of freezing a model fitted on a truncated history.
    """

    boundary: pd.Timestamp
    sha256: str
    n_rows: int
    status: str


def train_missing_boundaries(
    features: pd.DataFrame,
    *,
    entry_marks: pd.DataFrame,
    funding_by_symbol: Mapping[str, pd.Series],
    hourly_close: pd.DataFrame,
    until: pd.Timestamp,
    out_dir: Path,
    values: MlTiltValues,
    strategy: StrategySpec,
) -> tuple[BoundaryTraining, ...]:
    """Train and save each scheduled boundary up to ``until`` that has no valid model in ``out_dir``.

    Labels are rebuilt per boundary from decisions before ``boundary - purge_days`` only, so the
    volatility floor never sees rows the boundary model could not have observed.

    Raises:
        DataIntegrityError: An existing model file is invalid (left untouched) or a boundary has too few rows.
    """
    existing = {m.boundary: m for m in load_model_set(out_dir, values=values)}
    decisions = pd.DatetimeIndex(features.index.get_level_values("decision"))
    observed = hourly_close.dropna(how="all").index
    data_end = observed.max() + pd.Timedelta(hours=1) if len(observed) else None
    rows: list[BoundaryTraining] = []
    for boundary in boundary_schedule(values, until):
        if boundary in existing:
            kept = existing[boundary]
            rows.append(BoundaryTraining(boundary, kept.sha256, kept.n_rows, "exists"))
            continue
        if data_end is None or data_end < label_data_end(values, boundary, entry_hour_utc=strategy.entry_hour_utc):
            rows.append(BoundaryTraining(boundary, "", 0, "awaiting_data"))
            continue
        cutoff = boundary - pd.Timedelta(days=values.purge_days)
        past = features.loc[decisions < cutoff]
        labels = forward_labels(
            entry_marks,
            funding_by_symbol,
            hourly_close,
            past,
            values=values,
            entry_hour_utc=strategy.entry_hour_utc,
            snapshot_hour_utc=strategy.snapshot_hour_utc,
        )
        model = train_boundary_model(past, labels, boundary, values=values)
        save_model(model, out_dir)
        rows.append(BoundaryTraining(boundary, model.sha256, model.n_rows, "trained"))
    return tuple(rows)


def ml_tilt_signal_digest(base_signal_digest: str, models: Sequence[MlTiltModel], values_digest: str) -> str:
    """Digest identifying the ML variant: base signal, this module's source, values, and model sha256s."""
    payload = {
        "base_signal_digest": base_signal_digest,
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "values_digest": values_digest,
        "model_sha256": [m.sha256 for m in sorted(models, key=lambda m: m.boundary)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_model_set(directory: Path, *, values: MlTiltValues) -> tuple[MlTiltModel, ...]:
    """Load and digest-verify every saved model, ordered by boundary; empty when the directory is absent."""
    if not directory.is_dir():
        return ()
    models = [_load_model(path, values) for path in sorted(directory.glob("*.json"))]
    boundaries = [m.boundary for m in models]
    if len(set(boundaries)) != len(boundaries):
        raise DataIntegrityError("ml tilt model set contains a duplicate boundary")
    return tuple(sorted(models, key=lambda m: m.boundary))


def ml_scores(features: pd.DataFrame, models: Sequence[MlTiltModel], *, values: MlTiltValues) -> pd.DataFrame:
    """Return raw decision x symbol predictions, each scored by the latest model at or before the decision.

    Raises:
        DataIntegrityError: A decision lacks its required scheduled model (fails closed, no fallback).
    """
    ordered = sorted(models, key=lambda m: m.boundary)
    by_boundary = {m.boundary: m for m in ordered}
    row_decisions = pd.DatetimeIndex(features.index.get_level_values("decision"))
    owner: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision in row_decisions.unique():
        schedule = boundary_schedule(values, decision)
        if not schedule:
            raise DataIntegrityError("ml tilt decision precedes the schedule anchor")
        eligible = [m.boundary for m in ordered if m.boundary <= decision]
        if not eligible or eligible[-1] != schedule[-1]:
            raise DataIntegrityError("ml tilt required boundary model is missing")
        owner[decision] = eligible[-1]
    row_owner = pd.DatetimeIndex([owner[d] for d in row_decisions])
    predictions = np.full(len(features), np.nan, dtype=np.float64)
    matrix = features.to_numpy(dtype="float64")
    for boundary in row_owner.unique():
        model = by_boundary[boundary]
        if tuple(str(c) for c in features.columns) != model.feature_names:
            raise DataIntegrityError("ml tilt features do not match the model feature set")
        selected = np.asarray(row_owner == boundary, dtype=bool)
        booster = lgb.Booster(model_str=model.model_text)
        predictions[selected] = booster.predict(matrix[selected])
    predictions[np.isnan(matrix).all(axis=1)] = np.nan
    return pd.Series(predictions, index=features.index, dtype="float64").unstack(level="symbol")


def apply_ml_tilt(
    base_weights: pd.DataFrame,
    scores: pd.DataFrame,
    trade_roster: pd.DataFrame,
    *,
    values: MlTiltValues,
    name_clip: float,
    min_rank_symbols: int,
) -> pd.DataFrame:
    """Blend the EMA-smoothed ML rank book into C0 entry weights, preserving C0 row gross.

    The entry offset is the single entry hour shared by ``base_weights``; each decision maps to
    ``decision + 1 day + entry hour``. Scores after a decision never influence its weights.

    Raises:
        DataIntegrityError: ``base_weights`` entries do not share one entry hour, scores are unordered,
            or a held base row has fewer than ``min_rank_symbols`` scored trade-roster names.
    """
    hours = pd.unique(base_weights.index.hour)
    if len(hours) != 1:
        raise DataIntegrityError("base weights must share one entry hour")
    entry_offset = _DAY + pd.Timedelta(hours=int(hours[0]))
    columns = base_weights.columns
    if not scores.index.is_monotonic_increasing or scores.index.has_duplicates:
        raise DataIntegrityError("ml tilt scores must be strictly increasing by decision")
    raw = scores.reindex(columns=columns).astype("float64")
    trade = trade_roster.reindex(index=raw.index, columns=columns, fill_value=False).astype(bool)
    held = np.abs(base_weights.to_numpy(dtype="float64")).sum(axis=1) > 0.0
    scored_names = (raw.notna() & trade).sum(axis=1)
    decisions = pd.DatetimeIndex(base_weights.index - entry_offset)
    scored = scored_names.reindex(decisions).fillna(0).to_numpy() >= min_rank_symbols
    if bool((held & ~scored).any()):
        raise DataIntegrityError("ml tilt scores are missing for a held base row")

    spread = raw.std(axis=1)
    z = raw.sub(raw.mean(axis=1), axis=0).div(spread.where(spread > 0.0), axis=0)
    smoothed = _truncated_ema(z, values).where(z.notna())
    ml_book = rank_weight_book(smoothed, trade & smoothed.notna(), 1, min_rank_symbols)
    ml_book.index = ml_book.index + entry_offset
    ml_book = ml_book.reindex(index=base_weights.index, columns=columns).fillna(0.0)

    base = base_weights.to_numpy(dtype="float64")
    c0_gross = np.abs(base).sum(axis=1)
    ml_scaled = ml_book.to_numpy(dtype="float64") * c0_gross[:, None]
    blended = (1.0 - values.tilt) * base + values.tilt * ml_scaled
    clipped = clip_names_preserving_gross(pd.DataFrame(blended, index=base_weights.index, columns=columns), name_clip)
    clipped_values = clipped.to_numpy(dtype="float64")
    clipped_gross = np.abs(clipped_values).sum(axis=1)
    scale = np.divide(c0_gross, clipped_gross, out=np.zeros_like(c0_gross), where=clipped_gross > 0.0)
    return pd.DataFrame(clipped_values * scale[:, None], index=base_weights.index, columns=columns)


def _truncated_ema(z: pd.DataFrame, values: MlTiltValues) -> pd.DataFrame:
    """Exponential smoothing over the last ``ema_warmup_days`` rows only (finite kernel).

    A finite kernel makes each smoothed row a function of a fixed trailing window, so a recomputation
    window that starts anywhere at least that far back (live cycles, backtests) yields identical rows.
    """
    if z.empty:
        return z
    grid = pd.date_range(z.index.min(), z.index.max(), freq="D")
    daily = z.reindex(grid)
    decay = 0.5 ** (1.0 / values.smooth_halflife_days)
    numerator = pd.DataFrame(0.0, index=grid, columns=z.columns)
    denominator = pd.DataFrame(0.0, index=grid, columns=z.columns)
    for lag in range(ema_warmup_days(values) + 1):
        shifted = daily.shift(lag)
        weight = decay**lag
        numerator += shifted.fillna(0.0) * weight
        denominator += shifted.notna() * weight
    return numerator.div(denominator.where(denominator > 0.0)).reindex(z.index)


def _lookup(frame: pd.DataFrame, times: pd.DatetimeIndex, symbols: pd.Index) -> np.ndarray:
    rows = frame.index.get_indexer(times)
    cols = frame.columns.get_indexer(symbols)
    table = frame.to_numpy(dtype="float64")
    out = np.full(len(times), np.nan, dtype=np.float64)
    ok = (rows >= 0) & (cols >= 0)
    out[ok] = table[rows[ok], cols[ok]]
    return out


def _positive(marks: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(marks) & (marks > 0.0), marks, np.nan)


def _window_sums(
    funding_by_symbol: Mapping[str, pd.Series],
    symbols: pd.Index,
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
) -> np.ndarray:
    out = np.full(len(symbols), np.nan, dtype=np.float64)
    for symbol in symbols.unique():
        series = funding_by_symbol.get(symbol)
        if series is None:
            continue
        ordered = series.sort_index()
        values = ordered.to_numpy(dtype="float64")
        bad = ~np.isfinite(values)
        cumulative = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, values))])
        bad_count = np.concatenate([[0], np.cumsum(bad)])
        mask = np.asarray(symbols == symbol, dtype=bool)
        lo = ordered.index.searchsorted(starts[mask], side="right")
        hi = ordered.index.searchsorted(ends[mask], side="right")
        broken = (bad_count[hi] - bad_count[lo]) > 0
        if len(ordered):
            broken |= np.asarray(ends[mask] > ordered.index[-1], dtype=bool)
        else:
            broken[:] = True
        out[mask] = np.where(broken, np.nan, cumulative[hi] - cumulative[lo])
    return out


def values_digest(values: MlTiltValues) -> str:
    payload = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in asdict(values).items()
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _model_sha(model_text: str, boundary: pd.Timestamp, feature_names: tuple[str, ...], digest: str) -> str:
    payload = {
        "model_text": model_text,
        "boundary": boundary.isoformat(),
        "feature_names": list(feature_names),
        "values_digest": digest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_model(path: Path, values: MlTiltValues) -> MlTiltModel:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        model = MlTiltModel(
            boundary=pd.Timestamp(raw["boundary"]),
            train_end=pd.Timestamp(raw["train_end"]),
            n_rows=int(raw["n_rows"]),
            feature_names=tuple(str(c) for c in raw["feature_names"]),
            values_digest=str(raw["values_digest"]),
            lightgbm_version=str(raw["lightgbm_version"]),
            model_text=str(raw["model_text"]),
            sha256=str(raw["sha256"]),
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
        raise DataIntegrityError(f"ml tilt model unreadable: {path.name}") from exc
    if model.values_digest != values_digest(values):
        raise DataIntegrityError(f"ml tilt model was trained under different values: {path.name}")
    expected = _model_sha(model.model_text, model.boundary, model.feature_names, model.values_digest)
    if model.sha256 != expected:
        raise DataIntegrityError(f"ml tilt model digest mismatch: {path.name}")
    return model
