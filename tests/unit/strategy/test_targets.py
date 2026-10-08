from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.strategy.books import clip_names_preserving_gross, rank_weight_book
from src.strategy.features import FEATURE_REGISTRY
from src.strategy.targets import (
    FLOW_MOM_TOP20_GROWTH,
    FLOW_MOM_TOP20,
    FLOW_MOM_TOP40_CONTROL,
    FeatureMember,
    StrategyTargets,
    StrategySpec,
    build_strategy_targets,
)

_SYMBOLS = tuple(f"S{i:02d}" for i in range(10))
_MEMBERS = ("flow_imb_168h", "flow_imb_720h", "xs_mom_336h", "xs_idio_mom_336h", "mom3_skew_168h")
_N_DAYS = 100


def _daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2021-01-01", periods=_N_DAYS, freq="D", tz="UTC")
    close = pd.DataFrame(100.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    qv = pd.DataFrame(5_000_000.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    return close, qv


def _hourly(n_bars: int | None = None) -> dict[str, pd.DataFrame]:
    total = _N_DAYS * 24 if n_bars is None else n_bars
    idx = pd.date_range("2021-01-01", periods=total, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    walks = np.cumsum(rng.normal(0.0, 0.002, (total, len(_SYMBOLS))), axis=0)
    close = pd.DataFrame(np.exp(walks) * 100.0, index=idx, columns=list(_SYMBOLS))
    drift = np.linspace(0.0, 1.0, total)[:, None] * np.arange(len(_SYMBOLS))[None, :]
    qv = pd.DataFrame(200_000.0 + drift * 1_000.0, index=idx, columns=list(_SYMBOLS))
    share = 0.5 + 0.02 * np.sin(np.arange(total)[:, None] / 24.0 + np.arange(len(_SYMBOLS))[None, :])
    taker = pd.DataFrame(qv.to_numpy() * share, index=idx, columns=list(_SYMBOLS))
    available = pd.DataFrame(
        np.broadcast_to((idx + pd.Timedelta(hours=1)).to_numpy()[:, None], (total, len(_SYMBOLS))),
        index=idx, columns=list(_SYMBOLS),
    )
    return {"close": close, "quote_vol": qv, "taker_buy_quote": taker, "available_at": available}


def _build(strategy: StrategySpec = FLOW_MOM_TOP20) -> StrategyTargets:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    return build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], strategy=strategy
    )


def test_default_strategy_is_explicit_top20() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    candidate = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    assert candidate.strategy.strategy_id == "flow_mom_top20"
    assert candidate.strategy.breadth == 20
    assert candidate.breadth == 20


def test_top40_control_shares_feature_policy() -> None:
    assert FLOW_MOM_TOP40_CONTROL.breadth == 40
    assert [m.name for m in FLOW_MOM_TOP40_CONTROL.members] == [m.name for m in FLOW_MOM_TOP20.members]
    assert [m.sign for m in FLOW_MOM_TOP40_CONTROL.members] == [m.sign for m in FLOW_MOM_TOP20.members]
    assert FLOW_MOM_TOP40_CONTROL.strategy_id != FLOW_MOM_TOP20.strategy_id
    daily_close, daily_qv = _daily()
    panels = _hourly()
    top20 = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], strategy=FLOW_MOM_TOP20
    )
    top40 = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], strategy=FLOW_MOM_TOP40_CONTROL
    )
    assert top40.breadth == 40
    assert list(top40.target_weights.columns) == list(top20.target_weights.columns)


def test_custom_breadth_reaches_pit_roster() -> None:
    custom = dataclasses.replace(FLOW_MOM_TOP20, strategy_id="custom_b12", breadth=12)
    candidate = _build(custom)
    assert candidate.breadth == 12
    assert candidate.strategy.breadth == 12
    gross = candidate.target_weights.abs().sum(axis=1)
    assert bool((gross <= 1.0 + 1e-9).all())


def test_invalid_member_definitions_fail() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(
            FLOW_MOM_TOP20,
            members=(FeatureMember(name="flow_imb_168h", sign=1), FeatureMember(name="flow_imb_168h", sign=1)),
        )
    unknown = dataclasses.replace(
        FLOW_MOM_TOP20, members=(FeatureMember(name="no_such_feature", sign=1),)
    )
    with pytest.raises(ValueError, match="unregistered"):
        build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], strategy=unknown)
    with pytest.raises(ValueError, match="strictly earlier"):
        StrategySpec(
            strategy_id="bad-clock", breadth=20, members=FLOW_MOM_TOP20.members,
            min_rank_symbols=8, snapshot_hour_utc=23, release_hour_utc=23, entry_hour_utc=0,
            design_data_cutoff=FLOW_MOM_TOP20.design_data_cutoff,
        )
    with pytest.raises(ValueError, match="min_rank_symbols"):
        StrategySpec(
            strategy_id="bad-pop", breadth=20, members=FLOW_MOM_TOP20.members,
            min_rank_symbols=1, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
            design_data_cutoff=FLOW_MOM_TOP20.design_data_cutoff,
        )


def test_design_data_cutoff_rejects_naive_and_nat() -> None:
    """The design cutoff is a required UTC instant: NaT, naive, and offset zones fail."""
    base = {
        "strategy_id": "bad-cutoff", "breadth": 20, "members": FLOW_MOM_TOP20.members,
        "min_rank_symbols": 8, "snapshot_hour_utc": 22, "release_hour_utc": 23, "entry_hour_utc": 0,
    }
    with pytest.raises(ValueError, match="design_data_cutoff"):
        StrategySpec(**base, design_data_cutoff=pd.NaT)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="design_data_cutoff"):
        StrategySpec(**base, design_data_cutoff=pd.Timestamp("2026-07-01"))
    with pytest.raises(ValueError, match="design_data_cutoff"):
        StrategySpec(**base, design_data_cutoff=pd.Timestamp("2026-07-01", tz="Asia/Seoul"))
    assert dataclasses.replace(FLOW_MOM_TOP20).design_data_cutoff == pd.Timestamp("2026-07-01T00:00:00Z")


def test_future_perturbation_invariance_remains_exact() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    hacked = {k: v.copy() for k, v in panels.items()}
    release = daily_close.index[94] + pd.Timedelta(hours=23)
    later = hacked["close"].index[hacked["close"].index > release]
    hacked["close"].loc[later] *= 7.0
    hacked["quote_vol"].loc[later] *= 7.0
    hacked["taker_buy_quote"].loc[later] *= 7.0
    rebuilt = build_strategy_targets(hacked, hacked["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=hacked["close"])
    cutoff = release - pd.Timedelta(hours=23) + pd.Timedelta(days=1)
    early_labels = base.target_weights.index[base.target_weights.index <= cutoff]
    pd.testing.assert_frame_equal(rebuilt.target_weights.loc[early_labels], base.target_weights.loc[early_labels])


def test_insufficient_ranked_population_becomes_no_trade() -> None:
    tiny = dataclasses.replace(FLOW_MOM_TOP20, strategy_id="tiny-pop", min_rank_symbols=9)
    daily_close, daily_qv = _daily()
    panels = _hourly()
    candidate = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], strategy=tiny
    )
    row = candidate.target_weights.iloc[50]
    assert bool((row.to_numpy() == 0.0).all())
    assert bool(np.isfinite(candidate.target_weights.to_numpy()).all())


def test_target_ignores_unclosed_2300_bar() -> None:
    """Changing a 23:00-open candle leaves the next-midnight target unchanged."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    hacked = {k: v.copy() for k, v in panels.items()}
    day = daily_close.index[94]
    bar = day + pd.Timedelta(hours=23)
    hacked["close"].loc[bar] *= 25.0
    hacked["quote_vol"].loc[bar] *= 25.0
    hacked["taker_buy_quote"].loc[bar] *= 25.0
    rebuilt = build_strategy_targets(hacked, hacked["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=hacked["close"])
    entry = day + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(rebuilt.target_weights.loc[[entry]], base.target_weights.loc[[entry]])


def test_target_moves_with_completed_2200_bar() -> None:
    """Changing the 22:00-open candle may change decisions from its release onward."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    hacked = {k: v.copy() for k, v in panels.items()}
    day = daily_close.index[94]
    bar = day + pd.Timedelta(hours=22)
    hacked["close"].loc[bar] *= 50.0
    rebuilt = build_strategy_targets(hacked, hacked["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=hacked["close"])
    entry = day + pd.Timedelta(days=1)
    assert not np.allclose(
        rebuilt.target_weights.loc[entry].to_numpy(), base.target_weights.loc[entry].to_numpy()
    )
    early = daily_close.index[1:10]
    pd.testing.assert_frame_equal(
        rebuilt.target_weights.loc[early], base.target_weights.loc[early]
    )


def test_late_historical_publication_fails_closed() -> None:
    """A late source bar invalidates the dependent feature snapshot."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    delayed = panels["available_at"].copy()
    decision = daily_close.index[94]
    delayed.loc[decision + pd.Timedelta(hours=21), :] = decision + pd.Timedelta(hours=24)
    rebuilt = build_strategy_targets(panels, delayed, daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    entry = decision + pd.Timedelta(days=1)
    assert bool((base.target_weights.loc[entry] != 0.0).any())
    assert bool((rebuilt.target_weights.loc[entry] == 0.0).all())


def test_target_equals_equal_weight_member_average() -> None:
    """Output equals the equal-weight average of the five native rank books."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    candidate = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    decisions = daily_close.index[:-1]
    hourly_index = panels["close"].index
    books = []
    for name in _MEMBERS:
        feature = registry[name].builder(panels).reindex(columns=list(_SYMBOLS))
        snaps = pd.DataFrame(float("nan"), index=decisions, columns=list(_SYMBOLS), dtype="float64")
        for stamp in decisions:
            snaps.loc[stamp] = feature.loc[stamp + pd.Timedelta(hours=22)].to_numpy()
        assert hourly_index.equals(panels["quote_vol"].index)
        mask = snaps.notna()
        mask[:] = True
        from src.strategy.universe import build_pit_roster as _roster

        roster = _roster(daily_close, daily_qv, _SYMBOLS, breadth=20).loc[decisions]
        books.append(rank_weight_book(snaps, roster & snaps.notna(), 1, 8))
    expected = (books[0] + books[1] + books[2] + books[3] + books[4]) / 5.0
    expected.index = candidate.target_weights.index
    pd.testing.assert_frame_equal(candidate.target_weights, expected, check_dtype=True)
    assert str(candidate.target_weights.dtypes.unique()) == "[dtype('float64')]"


def test_ensemble_gross_never_rescaled_upward() -> None:
    """Averaging keeps gross at or below one without renormalization."""
    candidate = _build()
    gross = candidate.target_weights.abs().sum(axis=1)
    assert bool((gross <= 1.0 + 1e-9).all())
    assert float(gross.max()) < 1.0
    assert bool(np.isfinite(candidate.target_weights.to_numpy()).all())
    assert (candidate.target_weights.sum(axis=1).abs().max()) < 1e-9


def test_missing_hourly_archive_for_selected_symbol_fails_closed() -> None:
    """A rostered symbol without hourly history fails instead of substitution."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    panels["close"] = panels["close"].drop(columns=[_SYMBOLS[0]])
    panels["quote_vol"] = panels["quote_vol"].drop(columns=[_SYMBOLS[0]])
    panels["taker_buy_quote"] = panels["taker_buy_quote"].drop(columns=[_SYMBOLS[0]])
    with pytest.raises(DataIntegrityError, match="hourly source archive"):
        build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])


def test_warmup_and_sparse_rows_emit_zero_without_fill() -> None:
    """Decisions without hourly coverage emit finite zero rows."""
    daily_close, daily_qv = _daily()
    panels = _hourly(n_bars=200)
    candidate = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    tail = candidate.target_weights.iloc[50:]
    assert bool((tail.to_numpy() == 0.0).all())
    assert bool(np.isfinite(candidate.target_weights.to_numpy()).all())


def test_entry_labels_and_availability_align_to_decision_clock() -> None:
    """Entry D+1 00:00 pairs with availability D 23:00 on the prior daily bar."""
    candidate = _build()
    daily_close, _ = _daily()
    assert bool((candidate.target_weights.index == daily_close.index[1:]).all())
    assert bool((candidate.signal_available_at == daily_close.index[:-1] + pd.Timedelta(hours=23)).all())
    assert list(candidate.target_weights.columns) == list(_SYMBOLS)
    assert candidate.breadth == 20
    outside = candidate.target_weights.loc[:, []]
    assert outside.shape[1] == 0


def test_rejects_invalid_panels_grid_and_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    with pytest.raises(DataIntegrityError, match="must contain"):
        build_strategy_targets({"close": panels["close"]}, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    naive = {k: v.copy() for k, v in panels.items()}
    for v in naive.values():
        v.index = v.index.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_strategy_targets(naive, naive["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=naive["close"])
    eastern = {k: v.copy() for k, v in panels.items()}
    from datetime import timedelta, timezone

    for v in eastern.values():
        v.index = v.index.tz_convert(timezone(timedelta(hours=-5)))
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_strategy_targets(eastern, eastern["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=eastern["close"])
    offminute = {k: v.copy() for k, v in panels.items()}
    for v in offminute.values():
        v.index = v.index + pd.Timedelta(minutes=30)
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_strategy_targets(offminute, offminute["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=offminute["close"])
    duped = {k: v.copy() for k, v in panels.items()}
    for k, v in duped.items():
        duped[k] = pd.concat([v.iloc[[0]], v])
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_strategy_targets(duped, duped["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=duped["close"])
    gapped = {k: v.drop(v.index[100]) for k, v in panels.items()}
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_strategy_targets(gapped, gapped["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=gapped["close"])
    misaligned = {k: v.copy() for k, v in panels.items()}
    misaligned["quote_vol"] = misaligned["quote_vol"].rename(columns={_SYMBOLS[0]: "ZZZ"})
    with pytest.raises(DataIntegrityError, match="identical index"):
        build_strategy_targets(misaligned, misaligned["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=misaligned["close"])
    availability_misaligned = panels["available_at"].iloc[:-1]
    with pytest.raises(DataIntegrityError, match="align with the hourly panel"):
        build_strategy_targets(panels, availability_misaligned, daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    availability_missing = panels["available_at"].copy()
    availability_missing.iloc[0, 0] = pd.NaT
    with pytest.raises(DataIntegrityError, match="missing publication"):
        build_strategy_targets(panels, availability_missing, daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    availability_early = panels["available_at"].copy()
    availability_early.iloc[0, 0] = panels["close"].index[0] - pd.Timedelta(hours=1)
    with pytest.raises(DataIntegrityError, match="precede the bar"):
        build_strategy_targets(panels, availability_early, daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    availability_object = panels["available_at"].astype(object)
    with pytest.raises(DataIntegrityError, match="timezone-aware"):
        build_strategy_targets(panels, availability_object, daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    import src.strategy.targets as candidate_mod

    monkeypatch.setattr(candidate_mod, "FEATURE_REGISTRY", ())
    with pytest.raises(ValueError, match="unregistered"):
        build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])


def test_candidate_row_mismatch_fails_closed() -> None:
    candidate = _build()
    with pytest.raises(DataIntegrityError, match="matching rows"):
        StrategyTargets(
            target_weights=candidate.target_weights.iloc[:-1],
            signal_available_at=candidate.signal_available_at,
            strategy=FLOW_MOM_TOP20,
        )


def test_strategy_member_validation_branches() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        FeatureMember(name="", sign=1)
    with pytest.raises(ValueError, match="sign"):
        FeatureMember(name="flow_imb_168h", sign=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strategy_id"):
        StrategySpec(
            strategy_id="", breadth=20, members=FLOW_MOM_TOP20.members,
            min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
            design_data_cutoff=FLOW_MOM_TOP20.design_data_cutoff,
        )
    for bad_breadth in (0, -3, True):
        with pytest.raises(ValueError, match="breadth"):
            StrategySpec(
                strategy_id="bad", breadth=bad_breadth, members=FLOW_MOM_TOP20.members,  # type: ignore[arg-type]
                min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
                design_data_cutoff=FLOW_MOM_TOP20.design_data_cutoff,
            )
    with pytest.raises(ValueError, match="non-empty tuple"):
        StrategySpec(
            strategy_id="bad", breadth=20, members=(),  # type: ignore[arg-type]
            min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
            design_data_cutoff=FLOW_MOM_TOP20.design_data_cutoff,
        )
    with pytest.raises(ValueError, match="integer hour"):
        StrategySpec(
            strategy_id="bad-sign", breadth=20, members=FLOW_MOM_TOP20.members,
            min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=24, entry_hour_utc=0,
            design_data_cutoff=FLOW_MOM_TOP20.design_data_cutoff,
        )
    daily_close, daily_qv = _daily()
    panels = _hourly()
    with pytest.raises(ValueError, match="StrategySpec"):
        build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], strategy=20)  # type: ignore[arg-type]


def _blocked(idx: pd.DatetimeIndex, symbols: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(False, index=idx, columns=list(symbols), dtype=bool)


def test_blocked_symbol_never_targetable() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    blocked = _blocked(daily_close.index, _SYMBOLS)
    blocked[_SYMBOLS[0]] = True
    candidate = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], blocked_decisions=blocked
    )
    col = candidate.target_weights[_SYMBOLS[0]].to_numpy()
    assert bool(np.isfinite(col).all())
    assert bool((col == 0.0).all())
    from src.strategy.universe import build_pit_roster

    roster = build_pit_roster(daily_close, daily_qv, _SYMBOLS, breadth=20, blocked_decisions=blocked)
    assert not bool(roster[_SYMBOLS[0]].any())


def test_blocking_retains_census_provenance() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    blocked = _blocked(daily_close.index, _SYMBOLS)
    blocked.iloc[90, 1] = True
    candidate = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], blocked_decisions=blocked
    )
    assert list(candidate.target_weights.columns) == list(_SYMBOLS)


def test_blocked_cell_zeroes_only_that_decision() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    plain = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    day = daily_close.index[90]
    entry = day + pd.Timedelta(days=1)
    victim = str(plain.target_weights.loc[entry].abs().idxmax())
    blocked = _blocked(daily_close.index, _SYMBOLS)
    blocked.loc[day, victim] = True
    candidate = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], blocked_decisions=blocked
    )
    assert float(candidate.target_weights.loc[entry, victim]) == 0.0
    row = candidate.target_weights.loc[entry].to_numpy(dtype="float64")
    assert abs(float(row.sum())) < 1e-9
    other = entry + pd.Timedelta(days=1)
    pd.testing.assert_series_equal(candidate.target_weights.loc[other], plain.target_weights.loc[other])


def test_unknown_block_fails_closed() -> None:
    from src.strategy.universe import build_pit_roster

    daily_close, daily_qv = _daily()
    panels = _hourly()
    bad_cols = _blocked(daily_close.index, _SYMBOLS).rename(columns={_SYMBOLS[0]: "NOPEUSDT"})
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_pit_roster(daily_close, daily_qv, _SYMBOLS, breadth=20, blocked_decisions=bad_cols)
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_strategy_targets(
            panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], blocked_decisions=bad_cols
        )
    bad_idx = _blocked(daily_close.index[1:], _SYMBOLS)
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_strategy_targets(
            panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"], blocked_decisions=bad_idx
        )


def test_no_block_preserves_behavior() -> None:
    from src.strategy.universe import build_pit_roster

    daily_close, daily_qv = _daily()
    panels = _hourly()
    base_roster = build_pit_roster(daily_close, daily_qv, _SYMBOLS, breadth=20)
    empty_roster = build_pit_roster(
        daily_close, daily_qv, _SYMBOLS, breadth=20, blocked_decisions=_blocked(daily_close.index, _SYMBOLS)
    )
    pd.testing.assert_frame_equal(empty_roster, base_roster)
    base = build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"])
    empty = build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=panels["close"],
        blocked_decisions=_blocked(daily_close.index, _SYMBOLS),
    )
    pd.testing.assert_frame_equal(empty.target_weights, base.target_weights)


def test_breadth_applies_after_block() -> None:
    from src.strategy.universe import build_pit_roster

    daily_close, daily_qv = _daily()
    daily_qv = daily_qv.copy()
    daily_qv[_SYMBOLS[0]] = daily_qv[_SYMBOLS[0]] * 10.0
    blocked = _blocked(daily_close.index, _SYMBOLS)
    blocked[_SYMBOLS[0]] = True
    roster = build_pit_roster(daily_close, daily_qv, _SYMBOLS, breadth=3, blocked_decisions=blocked)
    assert not bool(roster[_SYMBOLS[0]].any())
    row_sums = roster.sum(axis=1).to_numpy()
    assert bool((row_sums <= 3).all())
    assert bool((row_sums[90:] == 3).all())


def test_market_plane_is_mandatory() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    with pytest.raises(TypeError):
        build_strategy_targets(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)  # type: ignore[call-arg]


def test_market_plane_census_order_enforced() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    permuted = panels["close"][list(reversed(_SYMBOLS))]
    with pytest.raises(DataIntegrityError, match="census_symbols"):
        build_strategy_targets(
            panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=permuted
        )
    shifted = panels["close"].copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(DataIntegrityError, match="index"):
        build_strategy_targets(
            panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, market_close=shifted
        )


def test_default_strategy_identity_bumped() -> None:
    candidate = _build()
    assert candidate.strategy.strategy_id == "flow_mom_top20"


def _build_with(strategy: StrategySpec) -> StrategyTargets:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    return build_strategy_targets(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS,
        market_close=panels["close"], strategy=strategy,
    )


def test_default_policy_is_identity() -> None:
    default = _build()
    explicit = _build_with(
        dataclasses.replace(FLOW_MOM_TOP20, exposure_multiplier=1.0, name_clip=None)
    )
    pd.testing.assert_frame_equal(explicit.target_weights, default.target_weights)


def test_multiplier_scales_every_row() -> None:
    default = _build()
    levered = _build_with(
        dataclasses.replace(FLOW_MOM_TOP20, exposure_multiplier=2.5, name_clip=None)
    )
    pd.testing.assert_frame_equal(levered.target_weights, default.target_weights * 2.5)


def test_growth_spec_applies_clip_before_scaling() -> None:
    default = _build()
    growth = _build_with(FLOW_MOM_TOP20_GROWTH)
    expected = clip_names_preserving_gross(default.target_weights, 0.05) * 2.5
    pd.testing.assert_frame_equal(growth.target_weights, expected)
    default_gross = default.target_weights.abs().sum(axis=1).to_numpy()
    growth_gross = growth.target_weights.abs().sum(axis=1).to_numpy()
    assert np.allclose(growth_gross, 2.5 * default_gross, rtol=1e-12, atol=0.0)


def test_invalid_policy_fields_rejected() -> None:
    for bad_multiplier in (0, 0.0, -2.0, float("nan"), True):
        with pytest.raises(ValueError, match="exposure_multiplier"):
            dataclasses.replace(FLOW_MOM_TOP20, exposure_multiplier=bad_multiplier)
    for bad_clip in (0.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="name_clip"):
            dataclasses.replace(FLOW_MOM_TOP20, name_clip=bad_clip)


def test_policy_does_not_move_the_clock() -> None:
    default = _build()
    growth = _build_with(FLOW_MOM_TOP20_GROWTH)
    assert growth.target_weights.index.equals(default.target_weights.index)
    assert growth.signal_available_at.equals(default.signal_available_at)


def test_legacy_strategy_ids_resolve_to_canonical() -> None:
    """Each pre-rename id maps to its canonical id; canonical ids map to themselves."""
    from src.strategy.targets import LEGACY_STRATEGY_IDS, resolve_strategy_id

    assert LEGACY_STRATEGY_IDS == {
        "frozen_mhs_top20_v2": "flow_mom_top20",
        "frozen_mhs_top40_control_v2": "flow_mom_top40_control",
        "frozen_mhs_top20_growth_v2": "flow_mom_top20_growth",
    }
    for legacy, canonical in LEGACY_STRATEGY_IDS.items():
        assert resolve_strategy_id(legacy) == canonical
    for canonical in ("flow_mom_top20", "flow_mom_top40_control", "flow_mom_top20_growth"):
        assert resolve_strategy_id(canonical) == canonical
    assert resolve_strategy_id("flow_mom_b60_control") == "flow_mom_b60_control"


def test_unknown_strategy_id_fails_closed() -> None:
    """An id no writer ever produced raises instead of silently matching nothing."""
    from src.strategy.targets import resolve_strategy_id

    with pytest.raises(DataIntegrityError, match="unknown strategy id"):
        resolve_strategy_id("no_such_strategy")
    with pytest.raises(DataIntegrityError, match="strategy id must be a non-empty string"):
        resolve_strategy_id("")
    for invalid in ("flow_mom_b0_control", "flow_mom_b60_control\n", "flow_mom_b\u0660_control"):
        with pytest.raises(DataIntegrityError, match="unknown strategy id"):
            resolve_strategy_id(invalid)


def test_legacy_alias_table_is_immutable() -> None:
    """Persisted evidence identity cannot be redirected by mutating the alias table."""
    from src.strategy.targets import LEGACY_STRATEGY_IDS

    with pytest.raises(TypeError):
        LEGACY_STRATEGY_IDS["frozen_mhs_top20_v2"] = "flow_mom_top40_control"  # type: ignore[index]


def test_renamed_strategy_definitions_unchanged() -> None:
    """Members, signs, breadth, hours, exposure multiplier and name clip equal pre-rename values."""
    from src.strategy.targets import FLOW_MOM_TOP20_ACCOUNT_UNIT

    expected_members = (
        ("flow_imb_168h", 1),
        ("flow_imb_720h", 1),
        ("xs_mom_336h", 1),
        ("xs_idio_mom_336h", 1),
        ("mom3_skew_168h", 1),
    )
    for spec, strategy_id, breadth, exposure_multiplier, name_clip in (
        (FLOW_MOM_TOP20, "flow_mom_top20", 20, 1.0, None),
        (FLOW_MOM_TOP40_CONTROL, "flow_mom_top40_control", 40, 1.0, None),
        (FLOW_MOM_TOP20_GROWTH, "flow_mom_top20_growth", 20, 2.5, 0.05),
        (FLOW_MOM_TOP20_ACCOUNT_UNIT, "flow_mom_top20", 20, 1.0, 0.05),
    ):
        assert spec.strategy_id == strategy_id
        assert spec.breadth == breadth
        assert [(m.name, m.sign) for m in spec.members] == list(expected_members)
        assert spec.min_rank_symbols == 8
        assert (spec.snapshot_hour_utc, spec.release_hour_utc, spec.entry_hour_utc) == (22, 23, 0)
        assert spec.exposure_multiplier == exposure_multiplier
        assert spec.name_clip == name_clip
