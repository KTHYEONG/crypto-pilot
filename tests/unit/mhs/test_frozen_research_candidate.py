from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.books import rank_weight_book
from src.mhs.features import FEATURE_REGISTRY
from src.mhs.frozen_research_candidate import (
    FROZEN_MHS_TOP20_V1,
    FROZEN_MHS_TOP40_CONTROL_V1,
    FrozenFeatureMember,
    FrozenMhsCandidate,
    FrozenMhsStrategySpec,
    build_frozen_mhs_candidate,
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


def _build(strategy: FrozenMhsStrategySpec = FROZEN_MHS_TOP20_V1) -> FrozenMhsCandidate:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    return build_frozen_mhs_candidate(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, strategy=strategy
    )


def test_default_strategy_is_explicit_top20() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    candidate = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
    assert candidate.strategy.strategy_id == "frozen_mhs_top20_v1"
    assert candidate.strategy.breadth == 20
    assert candidate.breadth == 20


def test_top40_control_shares_feature_policy() -> None:
    assert FROZEN_MHS_TOP40_CONTROL_V1.breadth == 40
    assert [m.name for m in FROZEN_MHS_TOP40_CONTROL_V1.members] == [m.name for m in FROZEN_MHS_TOP20_V1.members]
    assert [m.sign for m in FROZEN_MHS_TOP40_CONTROL_V1.members] == [m.sign for m in FROZEN_MHS_TOP20_V1.members]
    assert FROZEN_MHS_TOP40_CONTROL_V1.strategy_id != FROZEN_MHS_TOP20_V1.strategy_id
    daily_close, daily_qv = _daily()
    panels = _hourly()
    top20 = build_frozen_mhs_candidate(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, strategy=FROZEN_MHS_TOP20_V1
    )
    top40 = build_frozen_mhs_candidate(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, strategy=FROZEN_MHS_TOP40_CONTROL_V1
    )
    assert top40.breadth == 40
    assert list(top40.target_weights.columns) == list(top20.target_weights.columns)


def test_custom_breadth_reaches_pit_roster() -> None:
    custom = dataclasses.replace(FROZEN_MHS_TOP20_V1, strategy_id="custom_b12", breadth=12)
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
            FROZEN_MHS_TOP20_V1,
            members=(FrozenFeatureMember(name="flow_imb_168h", sign=1), FrozenFeatureMember(name="flow_imb_168h", sign=1)),
        )
    unknown = dataclasses.replace(
        FROZEN_MHS_TOP20_V1, members=(FrozenFeatureMember(name="no_such_feature", sign=1),)
    )
    with pytest.raises(ValueError, match="unregistered"):
        build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, strategy=unknown)
    with pytest.raises(ValueError, match="strictly earlier"):
        FrozenMhsStrategySpec(
            strategy_id="bad-clock", breadth=20, members=FROZEN_MHS_TOP20_V1.members,
            min_rank_symbols=8, snapshot_hour_utc=23, release_hour_utc=23, entry_hour_utc=0,
        )
    with pytest.raises(ValueError, match="min_rank_symbols"):
        FrozenMhsStrategySpec(
            strategy_id="bad-pop", breadth=20, members=FROZEN_MHS_TOP20_V1.members,
            min_rank_symbols=1, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
        )


def test_future_perturbation_invariance_remains_exact() -> None:
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
    hacked = {k: v.copy() for k, v in panels.items()}
    release = daily_close.index[94] + pd.Timedelta(hours=23)
    later = hacked["close"].index[hacked["close"].index > release]
    hacked["close"].loc[later] *= 7.0
    hacked["quote_vol"].loc[later] *= 7.0
    hacked["taker_buy_quote"].loc[later] *= 7.0
    rebuilt = build_frozen_mhs_candidate(hacked, hacked["available_at"], daily_close, daily_qv, _SYMBOLS)
    cutoff = release - pd.Timedelta(hours=23) + pd.Timedelta(days=1)
    early_labels = base.target_weights.index[base.target_weights.index <= cutoff]
    pd.testing.assert_frame_equal(rebuilt.target_weights.loc[early_labels], base.target_weights.loc[early_labels])


def test_insufficient_ranked_population_becomes_no_trade() -> None:
    tiny = dataclasses.replace(FROZEN_MHS_TOP20_V1, strategy_id="tiny-pop", min_rank_symbols=9)
    daily_close, daily_qv = _daily()
    panels = _hourly()
    candidate = build_frozen_mhs_candidate(
        panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, strategy=tiny
    )
    row = candidate.target_weights.iloc[50]
    assert bool((row.to_numpy() == 0.0).all())
    assert bool(np.isfinite(candidate.target_weights.to_numpy()).all())


def test_target_ignores_unclosed_2300_bar() -> None:
    """Changing a 23:00-open candle leaves the next-midnight target unchanged."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
    hacked = {k: v.copy() for k, v in panels.items()}
    day = daily_close.index[94]
    bar = day + pd.Timedelta(hours=23)
    hacked["close"].loc[bar] *= 25.0
    hacked["quote_vol"].loc[bar] *= 25.0
    hacked["taker_buy_quote"].loc[bar] *= 25.0
    rebuilt = build_frozen_mhs_candidate(hacked, hacked["available_at"], daily_close, daily_qv, _SYMBOLS)
    entry = day + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(rebuilt.target_weights.loc[[entry]], base.target_weights.loc[[entry]])


def test_target_moves_with_completed_2200_bar() -> None:
    """Changing the 22:00-open candle may change decisions from its release onward."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    base = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
    hacked = {k: v.copy() for k, v in panels.items()}
    day = daily_close.index[94]
    bar = day + pd.Timedelta(hours=22)
    hacked["close"].loc[bar] *= 50.0
    rebuilt = build_frozen_mhs_candidate(hacked, hacked["available_at"], daily_close, daily_qv, _SYMBOLS)
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
    base = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
    delayed = panels["available_at"].copy()
    decision = daily_close.index[94]
    delayed.loc[decision + pd.Timedelta(hours=21), :] = decision + pd.Timedelta(hours=24)
    rebuilt = build_frozen_mhs_candidate(panels, delayed, daily_close, daily_qv, _SYMBOLS)
    entry = decision + pd.Timedelta(days=1)
    assert bool((base.target_weights.loc[entry] != 0.0).any())
    assert bool((rebuilt.target_weights.loc[entry] == 0.0).all())


def test_target_equals_equal_weight_member_average() -> None:
    """Output equals the equal-weight average of the five native rank books."""
    daily_close, daily_qv = _daily()
    panels = _hourly()
    candidate = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
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
        from src.mhs.frozen_research_universe import build_frozen_pit_roster as _roster

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
        build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)


def test_warmup_and_sparse_rows_emit_zero_without_fill() -> None:
    """Decisions without hourly coverage emit finite zero rows."""
    daily_close, daily_qv = _daily()
    panels = _hourly(n_bars=200)
    candidate = build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
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
        build_frozen_mhs_candidate({"close": panels["close"]}, panels["available_at"], daily_close, daily_qv, _SYMBOLS)
    naive = {k: v.copy() for k, v in panels.items()}
    for v in naive.values():
        v.index = v.index.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_frozen_mhs_candidate(naive, naive["available_at"], daily_close, daily_qv, _SYMBOLS)
    eastern = {k: v.copy() for k, v in panels.items()}
    from datetime import timedelta, timezone

    for v in eastern.values():
        v.index = v.index.tz_convert(timezone(timedelta(hours=-5)))
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_frozen_mhs_candidate(eastern, eastern["available_at"], daily_close, daily_qv, _SYMBOLS)
    offminute = {k: v.copy() for k, v in panels.items()}
    for v in offminute.values():
        v.index = v.index + pd.Timedelta(minutes=30)
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_frozen_mhs_candidate(offminute, offminute["available_at"], daily_close, daily_qv, _SYMBOLS)
    duped = {k: v.copy() for k, v in panels.items()}
    for k, v in duped.items():
        duped[k] = pd.concat([v.iloc[[0]], v])
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_frozen_mhs_candidate(duped, duped["available_at"], daily_close, daily_qv, _SYMBOLS)
    gapped = {k: v.drop(v.index[100]) for k, v in panels.items()}
    with pytest.raises(DataIntegrityError, match="1h grid"):
        build_frozen_mhs_candidate(gapped, gapped["available_at"], daily_close, daily_qv, _SYMBOLS)
    misaligned = {k: v.copy() for k, v in panels.items()}
    misaligned["quote_vol"] = misaligned["quote_vol"].rename(columns={_SYMBOLS[0]: "ZZZ"})
    with pytest.raises(DataIntegrityError, match="identical index"):
        build_frozen_mhs_candidate(misaligned, misaligned["available_at"], daily_close, daily_qv, _SYMBOLS)
    availability_misaligned = panels["available_at"].iloc[:-1]
    with pytest.raises(DataIntegrityError, match="align with the hourly panel"):
        build_frozen_mhs_candidate(panels, availability_misaligned, daily_close, daily_qv, _SYMBOLS)
    availability_missing = panels["available_at"].copy()
    availability_missing.iloc[0, 0] = pd.NaT
    with pytest.raises(DataIntegrityError, match="missing publication"):
        build_frozen_mhs_candidate(panels, availability_missing, daily_close, daily_qv, _SYMBOLS)
    availability_early = panels["available_at"].copy()
    availability_early.iloc[0, 0] = panels["close"].index[0] - pd.Timedelta(hours=1)
    with pytest.raises(DataIntegrityError, match="precede the bar"):
        build_frozen_mhs_candidate(panels, availability_early, daily_close, daily_qv, _SYMBOLS)
    availability_object = panels["available_at"].astype(object)
    with pytest.raises(DataIntegrityError, match="timezone-aware"):
        build_frozen_mhs_candidate(panels, availability_object, daily_close, daily_qv, _SYMBOLS)
    import src.mhs.frozen_research_candidate as candidate_mod

    monkeypatch.setattr(candidate_mod, "FEATURE_REGISTRY", ())
    with pytest.raises(ValueError, match="unregistered"):
        build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS)


def test_candidate_row_mismatch_fails_closed() -> None:
    candidate = _build()
    with pytest.raises(DataIntegrityError, match="matching rows"):
        FrozenMhsCandidate(
            target_weights=candidate.target_weights.iloc[:-1],
            signal_available_at=candidate.signal_available_at,
            strategy=FROZEN_MHS_TOP20_V1,
        )


def test_strategy_member_validation_branches() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        FrozenFeatureMember(name="", sign=1)
    with pytest.raises(ValueError, match="sign"):
        FrozenFeatureMember(name="flow_imb_168h", sign=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strategy_id"):
        FrozenMhsStrategySpec(
            strategy_id="", breadth=20, members=FROZEN_MHS_TOP20_V1.members,
            min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
        )
    for bad_breadth in (0, -3, True):
        with pytest.raises(ValueError, match="breadth"):
            FrozenMhsStrategySpec(
                strategy_id="bad", breadth=bad_breadth, members=FROZEN_MHS_TOP20_V1.members,  # type: ignore[arg-type]
                min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
            )
    with pytest.raises(ValueError, match="non-empty tuple"):
        FrozenMhsStrategySpec(
            strategy_id="bad", breadth=20, members=(),  # type: ignore[arg-type]
            min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=23, entry_hour_utc=0,
        )
    with pytest.raises(ValueError, match="integer hour"):
        FrozenMhsStrategySpec(
            strategy_id="bad-sign", breadth=20, members=FROZEN_MHS_TOP20_V1.members,
            min_rank_symbols=8, snapshot_hour_utc=22, release_hour_utc=24, entry_hour_utc=0,
        )
    daily_close, daily_qv = _daily()
    panels = _hourly()
    with pytest.raises(ValueError, match="FrozenMhsStrategySpec"):
        build_frozen_mhs_candidate(panels, panels["available_at"], daily_close, daily_qv, _SYMBOLS, strategy=20)  # type: ignore[arg-type]
