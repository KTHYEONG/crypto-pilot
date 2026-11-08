from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.strategy.books import clip_names_preserving_gross, rank_weight_book
from src.strategy.ml_tilt import (
    MEMBER_RANK_PREFIX,
    apply_ml_tilt,
    boundary_schedule,
    forward_labels,
    load_model_set,
    member_rank_features,
    ml_scores,
    save_model,
    train_boundary_model,
)
from src.strategy.store import EXAMPLE_STORE_ROOT, MlTiltValues, ml_tilt_values
from src.strategy.targets import FLOW_MOM_TOP20

_SYMBOLS = tuple(f"S{i:02d}" for i in range(10))
_N_DAYS = 100
_TOTAL_BARS = _N_DAYS * 24
_ENTRY_HOUR = FLOW_MOM_TOP20.entry_hour_utc
_SNAPSHOT_HOUR = FLOW_MOM_TOP20.snapshot_hour_utc
_DIGEST_FILES = (
    "src/strategy/features.py",
    "src/strategy/targets.py",
    "src/strategy/books.py",
    "src/strategy/universe.py",
    "src/strategy/sizing.py",
    "src/core/params.py",
)
_REPO = Path(__file__).resolve().parents[3]


def _values(**overrides: object) -> MlTiltValues:
    base = ml_tilt_values(EXAMPLE_STORE_ROOT)
    assert base is not None
    return dataclasses.replace(base, **overrides)


def _hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=_TOTAL_BARS, freq="h", tz="UTC")


def _panels(seed: int = 7) -> dict[str, pd.DataFrame]:
    idx = _hourly_index()
    rng = np.random.default_rng(seed)
    walks = np.cumsum(rng.normal(0.0, 0.002, (_TOTAL_BARS, len(_SYMBOLS))), axis=0)
    close = pd.DataFrame(np.exp(walks) * 100.0, index=idx, columns=list(_SYMBOLS))
    drift = np.linspace(0.0, 1.0, _TOTAL_BARS)[:, None] * np.arange(len(_SYMBOLS))[None, :]
    qv = pd.DataFrame(200_000.0 + drift * 1_000.0, index=idx, columns=list(_SYMBOLS))
    share = 0.5 + 0.02 * np.sin(np.arange(_TOTAL_BARS)[:, None] / 24.0 + np.arange(len(_SYMBOLS))[None, :])
    taker = pd.DataFrame(qv.to_numpy() * share, index=idx, columns=list(_SYMBOLS))
    available = pd.DataFrame(
        np.broadcast_to((idx + pd.Timedelta(hours=1)).to_numpy()[:, None], (_TOTAL_BARS, len(_SYMBOLS))),
        index=idx, columns=list(_SYMBOLS),
    )
    return {"close": close, "quote_vol": qv, "taker_buy_quote": taker, "available_at": available}


def _daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2021-01-01", periods=_N_DAYS, freq="D", tz="UTC")
    close = pd.DataFrame(100.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    qv = pd.DataFrame(5_000_000.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    return close, qv


def _funding(seed: int = 3) -> dict[str, pd.Series]:
    idx = pd.date_range("2021-01-01", periods=_TOTAL_BARS // 8, freq="8h", tz="UTC")
    rng = np.random.default_rng(seed)
    return {s: pd.Series(rng.normal(0.0, 1e-4, len(idx)), index=idx) for s in _SYMBOLS}


def _features(panels: dict[str, pd.DataFrame], values: MlTiltValues) -> pd.DataFrame:
    daily_close, daily_qv = _daily()
    return member_rank_features(
        {k: panels[k] for k in ("close", "quote_vol", "taker_buy_quote")},
        panels["available_at"], daily_close, daily_qv, _SYMBOLS,
        market_close=panels["close"], strategy=FLOW_MOM_TOP20, values=values,
    )


def _labels(panels: dict[str, pd.DataFrame], features: pd.DataFrame, values: MlTiltValues) -> pd.Series:
    close = panels["close"]
    return forward_labels(
        close.shift(1), _funding(), close, features, values=values,
        entry_hour_utc=_ENTRY_HOUR, snapshot_hour_utc=_SNAPSHOT_HOUR,
    )


@pytest.fixture(scope="module")
def train_values() -> MlTiltValues:
    return _values(
        schedule_anchor=pd.Timestamp("2021-03-31", tz="UTC"),
        retrain_days=5,
        min_train_rows=20,
        n_trees=5,
        feature_breadth=48,
        vol_floor_quantile=0.0,
    )


@pytest.fixture(scope="module")
def dataset(train_values: MlTiltValues) -> tuple[pd.DataFrame, pd.Series]:
    panels = _panels()
    features = _features(panels, train_values)
    return features, _labels(panels, features, train_values)


def test_member_features_have_one_rank_column_per_member(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, _ = dataset
    assert list(features.columns) == [f"{MEMBER_RANK_PREFIX}{m.name}" for m in FLOW_MOM_TOP20.members]
    assert features.index.names == ["decision", "symbol"]
    assert features.stack().between(-0.5, 0.5).all()


def test_member_features_mask_unready_symbol_for_that_decision(train_values: MlTiltValues) -> None:
    panels = _panels()
    decision = pd.Timestamp("2021-03-31", tz="UTC")
    late = decision + pd.Timedelta(hours=_SNAPSHOT_HOUR)
    panels["available_at"].loc[late, "S00"] = decision + pd.Timedelta(days=1, hours=2)
    features = _features(panels, train_values)
    assert features.loc[(decision, "S00")].isna().all()
    assert features.loc[decision].drop(index="S00").notna().any(axis=None)


def test_member_features_are_causal_under_future_perturbation(train_values: MlTiltValues) -> None:
    decision = pd.Timestamp("2021-03-31", tz="UTC")
    base_panels = _panels()
    original = _features(base_panels, train_values)
    perturbed = {k: v.copy() for k, v in base_panels.items()}
    cutoff = decision + pd.Timedelta(hours=_SNAPSHOT_HOUR)
    future = perturbed["close"].index > cutoff
    rng = np.random.default_rng(11)
    for key in ("close", "quote_vol", "taker_buy_quote"):
        block = perturbed[key].to_numpy(dtype="float64", copy=True)
        block[future] = rng.uniform(1.0, 500.0, size=(int(future.sum()), len(_SYMBOLS)))
        perturbed[key] = pd.DataFrame(block, index=perturbed[key].index, columns=perturbed[key].columns)
    shifted = _features(perturbed, train_values)
    pd.testing.assert_frame_equal(original.loc[original.index.get_level_values("decision") <= decision],
                                  shifted.loc[shifted.index.get_level_values("decision") <= decision])


def test_labels_do_not_use_later_rows_for_vol_floor(train_values: MlTiltValues, dataset: tuple) -> None:
    values = dataclasses.replace(train_values, vol_floor_quantile=0.5)
    panels = _panels()
    features, _ = dataset
    decisions = features.index.get_level_values("decision")
    last_early = pd.Timestamp("2021-03-31", tz="UTC")
    early = features.loc[decisions <= last_early]
    alone = _labels(panels, early, values)
    rng = np.random.default_rng(5)
    close = panels["close"].copy()
    late = close.index >= last_early + pd.Timedelta(days=1 + values.horizon_days, hours=_ENTRY_HOUR + 1)
    close.loc[late] = close.loc[late] * np.exp(rng.normal(0.0, 0.5, (int(late.sum()), len(_SYMBOLS))))
    distorted = dict(panels, close=close)
    pd.testing.assert_series_equal(_labels(distorted, early, values), alone)
    combined = _labels(distorted, features, values)
    assert not np.allclose(combined.loc[early.index].to_numpy(), alone.to_numpy(), equal_nan=True)


def test_labels_nan_when_horizon_past_last_bar(train_values: MlTiltValues, dataset: tuple) -> None:
    panels = _panels()
    features, _ = dataset
    last_decision = pd.Timestamp("2021-04-09", tz="UTC")
    last = features.loc[features.index.get_level_values("decision") == last_decision]
    assert len(last) > 0
    tail = _labels(panels, last, train_values)
    assert tail.isna().all()
    assert not (tail == 0.0).any()


def test_boundary_schedule_steps_from_anchor() -> None:
    values = _values(schedule_anchor=pd.Timestamp("2021-01-10", tz="UTC"), retrain_days=5)
    assert boundary_schedule(values, pd.Timestamp("2021-01-09", tz="UTC")) == ()
    assert boundary_schedule(values, pd.Timestamp("2021-01-10", tz="UTC")) == (pd.Timestamp("2021-01-10", tz="UTC"),)
    assert boundary_schedule(values, pd.Timestamp("2021-01-21", tz="UTC")) == (
        pd.Timestamp("2021-01-10", tz="UTC"),
        pd.Timestamp("2021-01-15", tz="UTC"),
        pd.Timestamp("2021-01-20", tz="UTC"),
    )


def test_training_excludes_purged_recent_decisions(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    boundary = pd.Timestamp("2021-04-07", tz="UTC")
    model = train_boundary_model(features, labels, boundary, values=train_values)
    decisions = features.index.get_level_values("decision")
    cutoff = boundary - pd.Timedelta(days=train_values.purge_days)
    expected = int(((decisions < cutoff) & np.isfinite(labels.to_numpy())).sum())
    assert model.n_rows == expected
    assert model.train_end == cutoff
    assert model.lightgbm_version == lgb.__version__


def test_training_is_deterministic(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    boundary = pd.Timestamp("2021-04-07", tz="UTC")
    first = train_boundary_model(features, labels, boundary, values=train_values)
    second = train_boundary_model(features, labels, boundary, values=train_values)
    assert first.sha256 == second.sha256
    assert first.feature_names == tuple(features.columns)


def test_training_rejects_too_few_rows(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    strict = dataclasses.replace(train_values, min_train_rows=10_000)
    with pytest.raises(DataIntegrityError):
        train_boundary_model(features, labels, pd.Timestamp("2021-04-07", tz="UTC"), values=strict)


def test_training_rejects_misaligned_labels(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    with pytest.raises(DataIntegrityError):
        train_boundary_model(features, labels.iloc[1:], pd.Timestamp("2021-04-07", tz="UTC"), values=train_values)


def test_model_roundtrip_and_tamper_detection(
    tmp_path: Path, train_values: MlTiltValues, dataset: tuple,
) -> None:
    features, labels = dataset
    model = train_boundary_model(features, labels, pd.Timestamp("2021-04-07", tz="UTC"), values=train_values)
    path = save_model(model, tmp_path / "models")
    assert path.name == "20210407.json"
    (loaded,) = load_model_set(tmp_path / "models", values=train_values)
    assert loaded == model
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model_text"] = raw["model_text"].replace("tree", "trEe", 1)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_model_set(tmp_path / "models", values=train_values)


def test_load_rejects_model_trained_under_other_values(
    tmp_path: Path, train_values: MlTiltValues, dataset: tuple,
) -> None:
    features, labels = dataset
    model = train_boundary_model(features, labels, pd.Timestamp("2021-04-07", tz="UTC"), values=train_values)
    save_model(model, tmp_path)
    other = dataclasses.replace(train_values, seed=train_values.seed + 1)
    with pytest.raises(DataIntegrityError, match="different values"):
        load_model_set(tmp_path, values=other)


def test_load_rejects_unreadable_and_duplicate_models(
    tmp_path: Path, train_values: MlTiltValues, dataset: tuple,
) -> None:
    assert load_model_set(tmp_path / "absent", values=train_values) == ()
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="unreadable"):
        load_model_set(tmp_path, values=train_values)
    (tmp_path / "bad.json").unlink()
    features, labels = dataset
    model = train_boundary_model(features, labels, pd.Timestamp("2021-04-07", tz="UTC"), values=train_values)
    save_model(model, tmp_path)
    payload = json.loads((tmp_path / "20210407.json").read_text(encoding="utf-8"))
    (tmp_path / "copy.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="duplicate"):
        load_model_set(tmp_path, values=train_values)


def test_schedule_ordering_and_missing_boundary_fail_closed(
    train_values: MlTiltValues, dataset: tuple, tmp_path: Path,
) -> None:
    features, labels = dataset
    values = dataclasses.replace(train_values, schedule_anchor=pd.Timestamp("2021-04-01", tz="UTC"))
    first = train_boundary_model(features, labels, pd.Timestamp("2021-04-01", tz="UTC"), values=values)
    second = train_boundary_model(features, labels, pd.Timestamp("2021-04-06", tz="UTC"), values=values)
    scored = features.loc[features.index.get_level_values("decision") >= pd.Timestamp("2021-04-01", tz="UTC")]
    with pytest.raises(DataIntegrityError, match="required boundary"):
        ml_scores(scored, [first], values=values)
    with pytest.raises(DataIntegrityError, match="required boundary"):
        ml_scores(scored, [second], values=values)
    early = features.loc[features.index.get_level_values("decision") < pd.Timestamp("2021-04-01", tz="UTC")]
    with pytest.raises(DataIntegrityError, match="precedes"):
        ml_scores(early, [first], values=values)


def test_scores_use_latest_eligible_model(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    values = dataclasses.replace(train_values, schedule_anchor=pd.Timestamp("2021-04-01", tz="UTC"), retrain_days=5)
    first = train_boundary_model(features, labels, pd.Timestamp("2021-04-01", tz="UTC"), values=values)
    second = train_boundary_model(features, labels, pd.Timestamp("2021-04-06", tz="UTC"), values=values)
    scored = features.loc[features.index.get_level_values("decision") >= pd.Timestamp("2021-04-01", tz="UTC")]
    scores = ml_scores(scored, [second, first], values=values)
    switch = pd.Timestamp("2021-04-06", tz="UTC")
    before_switch = scored.loc[scored.index.get_level_values("decision") < switch]
    expected = lgb.Booster(model_str=first.model_text).predict(before_switch.to_numpy(dtype="float64"))
    observed = scores.loc[before_switch.index.get_level_values("decision").unique()]
    observed_rows = observed.stack()
    np.testing.assert_allclose(observed_rows.loc[before_switch.index].to_numpy(), expected, rtol=0, atol=0)


def test_scores_reject_feature_mismatch(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    values = dataclasses.replace(train_values, schedule_anchor=pd.Timestamp("2021-04-01", tz="UTC"))
    model = train_boundary_model(features, labels, pd.Timestamp("2021-04-01", tz="UTC"), values=values)
    decisions = features.index.get_level_values("decision")
    scored = features.loc[(decisions >= pd.Timestamp("2021-04-01", tz="UTC")) & (decisions < pd.Timestamp("2021-04-06", tz="UTC"))]
    with pytest.raises(DataIntegrityError, match="feature set"):
        ml_scores(scored.iloc[:, :1], [model], values=values)


def _base_book(n_decisions: int = 30, seed: int = 3) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    rng = np.random.default_rng(seed)
    decisions = pd.date_range("2021-04-01", periods=n_decisions, freq="D", tz="UTC")
    entries = decisions + pd.Timedelta(days=1)
    signal = pd.DataFrame(rng.normal(size=(n_decisions, len(_SYMBOLS))), index=entries, columns=list(_SYMBOLS))
    eligible = pd.DataFrame(True, index=entries, columns=list(_SYMBOLS))
    base = clip_names_preserving_gross(rank_weight_book(signal, eligible, 1, 4), 0.2)
    base.iloc[5] = 0.0
    return base, decisions


def _scores_and_roster(decisions: pd.DatetimeIndex, seed: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    scores = pd.DataFrame(rng.normal(size=(len(decisions), len(_SYMBOLS))), index=decisions, columns=list(_SYMBOLS))
    trade = pd.DataFrame(rng.uniform(size=scores.shape) > 0.2, index=decisions, columns=list(_SYMBOLS))
    return scores, trade


def test_tilt_preserves_c0_gross_and_neutrality() -> None:
    base, decisions = _base_book()
    scores, trade = _scores_and_roster(decisions)
    out = apply_ml_tilt(base, scores, trade, values=_values(), name_clip=0.2, min_rank_symbols=4)
    assert out.index.equals(base.index)
    assert list(out.columns) == list(base.columns)
    nonzero = base.abs().sum(axis=1) > 0
    np.testing.assert_allclose(out.abs().sum(axis=1)[nonzero], base.abs().sum(axis=1)[nonzero], rtol=1e-12)
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-12)


def test_tilt_keeps_zero_c0_rows_zero() -> None:
    base, decisions = _base_book()
    scores, trade = _scores_and_roster(decisions)
    out = apply_ml_tilt(base, scores, trade, values=_values(), name_clip=0.2, min_rank_symbols=4)
    assert (out.iloc[5] == 0.0).all()


def test_tilt_fails_closed_when_held_row_unscored() -> None:
    base, decisions = _base_book()
    scores, trade = _scores_and_roster(decisions)
    with pytest.raises(DataIntegrityError, match="scores are missing"):
        apply_ml_tilt(base, scores.drop(index=decisions[10]), trade, values=_values(), name_clip=0.2, min_rank_symbols=4)
    out = apply_ml_tilt(base, scores.drop(index=decisions[5]), trade, values=_values(), name_clip=0.2, min_rank_symbols=4)
    assert (out.iloc[5] == 0.0).all()


def test_tilt_rejects_mixed_entry_hours() -> None:
    base, decisions = _base_book()
    mixed = base.copy()
    mixed.index = mixed.index.where(np.arange(len(mixed)) % 2 == 0, mixed.index + pd.Timedelta(hours=1))
    scores, trade = _scores_and_roster(decisions)
    with pytest.raises(DataIntegrityError, match="entry hour"):
        apply_ml_tilt(mixed, scores, trade, values=_values(), name_clip=0.2, min_rank_symbols=4)


def test_tilt_ema_is_causal_for_later_scores() -> None:
    base, decisions = _base_book()
    scores, trade = _scores_and_roster(decisions)
    cut = decisions[15]
    changed = scores.copy()
    changed.loc[changed.index > cut] = np.random.default_rng(21).normal(size=(int((changed.index > cut).sum()), len(_SYMBOLS)))
    values = _values()
    first = apply_ml_tilt(base, scores, trade, values=values, name_clip=0.2, min_rank_symbols=4)
    second = apply_ml_tilt(base, changed, trade, values=values, name_clip=0.2, min_rank_symbols=4)
    entry_cut = cut + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(first.loc[first.index <= entry_cut], second.loc[second.index <= entry_cut])
    assert not first.loc[first.index > entry_cut].equals(second.loc[second.index > entry_cut])


@pytest.mark.parametrize("path", _DIGEST_FILES)
def test_release_digest_files_untouched(path: str) -> None:
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "show", f"HEAD:{path}"],  # noqa: S607
            cwd=_REPO, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git history unavailable: {exc}")
    assert hashlib.sha256((_REPO / path).read_bytes()).hexdigest() == hashlib.sha256(head).hexdigest()


def _member_inputs(train_values: MlTiltValues) -> tuple:
    panels = _panels()
    daily_close, daily_qv = _daily()
    return panels, daily_close, daily_qv


def test_member_features_reject_unregistered_member(train_values: MlTiltValues) -> None:
    from src.strategy.targets import FeatureMember

    panels, daily_close, daily_qv = _member_inputs(train_values)
    strategy = dataclasses.replace(FLOW_MOM_TOP20, members=(FeatureMember("not_registered", 1),))
    with pytest.raises(ValueError, match="unregistered"):
        member_rank_features(
            {k: panels[k] for k in ("close", "quote_vol", "taker_buy_quote")},
            panels["available_at"], daily_close, daily_qv, _SYMBOLS,
            market_close=panels["close"], strategy=strategy, values=train_values,
        )


@pytest.mark.parametrize("case", ["missing_panel", "misaligned_panel", "misaligned_available", "misaligned_market"])
def test_member_features_reject_misaligned_inputs(train_values: MlTiltValues, case: str) -> None:
    panels, daily_close, daily_qv = _member_inputs(train_values)
    hourly = {k: panels[k] for k in ("close", "quote_vol", "taker_buy_quote")}
    available = panels["available_at"]
    market = panels["close"]
    reversed_cols = list(_SYMBOLS)[::-1]
    if case == "missing_panel":
        del hourly["taker_buy_quote"]
    elif case == "misaligned_panel":
        hourly["quote_vol"] = hourly["quote_vol"][reversed_cols]
    elif case == "misaligned_available":
        available = available[reversed_cols]
    else:
        market = market[reversed_cols]
    with pytest.raises(DataIntegrityError):
        member_rank_features(
            hourly, available, daily_close, daily_qv, _SYMBOLS,
            market_close=market, strategy=FLOW_MOM_TOP20, values=train_values,
        )


def test_labels_nan_without_observable_volatility(train_values: MlTiltValues) -> None:
    panels = _panels()
    early = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-01-02", tz="UTC"), s) for s in _SYMBOLS], names=["decision", "symbol"],
    )
    features = pd.DataFrame({"r_mom_168h": 0.1}, index=early)
    labels = _labels(panels, features, train_values)
    assert labels.isna().all()


def test_labels_nan_for_symbols_without_funding(train_values: MlTiltValues, dataset: tuple) -> None:
    panels = _panels()
    features, _ = dataset
    close = panels["close"]
    funding = _funding()
    del funding["S00"]
    funding["S01"] = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    labels = forward_labels(
        close.shift(1), funding, close, features, values=train_values,
        entry_hour_utc=_ENTRY_HOUR, snapshot_hour_utc=_SNAPSHOT_HOUR,
    )
    symbols = features.index.get_level_values("symbol")
    assert labels[symbols == "S00"].isna().all()
    assert labels[symbols == "S01"].isna().all()
    assert labels[~symbols.isin(["S00", "S01"])].notna().any()


def test_labels_nan_when_funding_ends_before_horizon(train_values: MlTiltValues, dataset: tuple) -> None:
    panels = _panels()
    features, _ = dataset
    close = panels["close"]
    funding = _funding()
    cut = pd.Timestamp("2021-04-02", tz="UTC")
    funding["S00"] = funding["S00"].loc[funding["S00"].index < cut]
    labels = forward_labels(
        close.shift(1), funding, close, features, values=train_values,
        entry_hour_utc=_ENTRY_HOUR, snapshot_hour_utc=_SNAPSHOT_HOUR,
    )
    idx = features.index
    after = (idx.get_level_values("symbol") == "S00") & (idx.get_level_values("decision") >= cut)
    assert labels[after].isna().all()
    assert labels[(idx.get_level_values("symbol") != "S00") & (idx.get_level_values("decision") >= cut)].notna().any()


def test_labels_nan_for_nonpositive_marks_without_contaminating_row(train_values: MlTiltValues, dataset: tuple) -> None:
    panels = _panels()
    features, _ = dataset
    close = panels["close"]
    marks = close.shift(1)
    decision = pd.Timestamp("2021-03-30", tz="UTC")
    marks.loc[decision + pd.Timedelta(days=1, hours=_ENTRY_HOUR), "S01"] = 0.0
    labels = forward_labels(
        marks, _funding(), close, features, values=train_values,
        entry_hour_utc=_ENTRY_HOUR, snapshot_hour_utc=_SNAPSHOT_HOUR,
    )
    row = labels.loc[decision]
    assert np.isnan(row["S01"])
    others = row.drop(index="S01")
    assert others.notna().all()
    assert (others.abs() < train_values.label_clip).any()


def test_scores_nan_for_rows_without_any_feature(train_values: MlTiltValues, dataset: tuple) -> None:
    features, labels = dataset
    values = dataclasses.replace(train_values, schedule_anchor=pd.Timestamp("2021-04-01", tz="UTC"))
    model = train_boundary_model(features, labels, pd.Timestamp("2021-04-01", tz="UTC"), values=values)
    decisions = features.index.get_level_values("decision")
    scored = features.loc[(decisions >= pd.Timestamp("2021-04-01", tz="UTC")) & (decisions < pd.Timestamp("2021-04-04", tz="UTC"))].copy()
    target = (pd.Timestamp("2021-04-02", tz="UTC"), "S03")
    scored.loc[target] = np.nan
    out = ml_scores(scored, [model], values=values)
    assert np.isnan(out.loc[target[0], target[1]])
    assert out.loc[target[0]].drop(index=target[1]).notna().all()


def test_tilt_rejects_unordered_scores() -> None:
    base, decisions = _base_book()
    scores, trade = _scores_and_roster(decisions)
    with pytest.raises(DataIntegrityError, match="strictly increasing"):
        apply_ml_tilt(base, scores.iloc[::-1], trade, values=_values(), name_clip=0.2, min_rank_symbols=4)


def test_tilt_fails_closed_when_held_row_scores_all_nan() -> None:
    base, decisions = _base_book()
    scores, trade = _scores_and_roster(decisions)
    scores.loc[decisions[10]] = np.nan
    with pytest.raises(DataIntegrityError, match="scores are missing"):
        apply_ml_tilt(base, scores, trade, values=_values(), name_clip=0.2, min_rank_symbols=4)


def test_tilt_rows_independent_of_scoring_window_start_with_near_ties() -> None:
    """With the finite EMA kernel, rows at least ``ema_warmup_days`` past the window start are window-free."""
    from src.strategy.ml_tilt import ema_warmup_days

    values = _values(smooth_halflife_days=2.0)
    span = ema_warmup_days(values)
    base, decisions = _base_book(n_decisions=3 * span + 10)
    base.iloc[5] = base.iloc[4]
    scores, trade = _scores_and_roster(decisions)
    offset = 7
    scores.iloc[:, 1] = scores.iloc[:, 0] + 1e-9
    scores.iloc[:offset, 1] = scores.iloc[:offset, 0] - 5.0
    trade[:] = True
    full = apply_ml_tilt(base, scores, trade, values=values, name_clip=0.2, min_rank_symbols=4)
    late = apply_ml_tilt(base.iloc[offset:], scores.iloc[offset:], trade, values=values, name_clip=0.2, min_rank_symbols=4)
    common = base.index[offset + span:]
    pd.testing.assert_frame_equal(full.loc[common], late.loc[common], check_exact=False, atol=1e-12, rtol=0.0)


def test_tilt_all_zero_base_without_scores_stays_zero() -> None:
    base, decisions = _base_book()
    base.iloc[:] = 0.0
    empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=base.columns, dtype="float64")
    _, trade = _scores_and_roster(decisions)
    out = apply_ml_tilt(base, empty, trade, values=_values(), name_clip=0.2, min_rank_symbols=4)
    assert (out == 0.0).all(axis=None)


def test_next_boundary_and_label_data_end() -> None:
    from src.strategy.ml_tilt import label_data_end, next_boundary

    values = _values(schedule_anchor=pd.Timestamp("2022-07-01", tz="UTC"), retrain_days=91, purge_days=5, horizon_days=3)
    anchor = values.schedule_anchor
    assert next_boundary(values, anchor - pd.Timedelta(days=3)) == anchor
    assert next_boundary(values, anchor) == anchor + pd.Timedelta(days=91)
    assert next_boundary(values, anchor + pd.Timedelta(days=90)) == anchor + pd.Timedelta(days=91)
    boundary = pd.Timestamp("2026-12-25", tz="UTC")
    assert label_data_end(values, boundary, entry_hour_utc=0) == pd.Timestamp("2026-12-24", tz="UTC")
