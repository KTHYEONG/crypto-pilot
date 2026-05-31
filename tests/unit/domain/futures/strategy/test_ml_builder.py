from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LabelPanel, LongMatrixDataset
from src.domain.futures.strategy.ml_builder import (
    _fit_predict_fold_dual_side,
    _predict_quantiles_with_fallback,
    _rank_score,
    _resolve_side_targets,
)


def _dataset(rows: int, groups: int, features: int = 3) -> LongMatrixDataset:
    rng = np.random.default_rng(7)
    group_size = rows // groups
    x = rng.normal(size=(rows, features)).astype(np.float32)
    y_rank = np.tile(np.arange(group_size, dtype=np.int32), groups)
    y_ev = rng.normal(scale=1e-3, size=rows).astype(np.float32)
    group = np.full((groups,), group_size, dtype=np.int32)
    sample_weight = np.ones((rows,), dtype=np.float32)
    index_map = np.column_stack(
        [np.arange(rows, dtype=np.int64), np.zeros((rows,), dtype=np.int64)]
    )
    return LongMatrixDataset(
        X=x,
        y_rank=y_rank,
        y_ev=y_ev,
        group=group,
        sample_weight=sample_weight,
        index_map=index_map,
        feature_names=tuple(f"f{i}" for i in range(features)),
    )


def test_rank_score_returns_nan_when_ranker_disabled() -> None:
    ds = _dataset(rows=12, groups=3)
    score = _rank_score(None, ds)
    assert score.shape == (ds.X.shape[0],)
    assert np.all(np.isnan(score))


def test_predict_quantiles_with_fallback_degenerate() -> None:
    ds = _dataset(rows=10, groups=2)
    score = np.linspace(-0.2, 0.2, 10, dtype=np.float32)
    ev = np.maximum(score, 0.0).astype(np.float32)
    q = _predict_quantiles_with_fallback(models=object(), dataset=ds, rank_score=score, ev_pred=ev)
    assert np.array_equal(q.q10, ev)
    assert np.array_equal(q.q50, ev)
    assert np.array_equal(q.q90, ev)


def test_fit_predict_fold_dual_side_signed_split(monkeypatch: MonkeyPatch) -> None:
    train_long = _dataset(rows=20, groups=4)
    valid_long = _dataset(rows=10, groups=2)
    test_long = _dataset(rows=10, groups=2)
    train_short = _dataset(rows=20, groups=4)
    valid_short = _dataset(rows=10, groups=2)
    test_short = _dataset(rows=10, groups=2)
    cfg = StrategyMLConfig(ranker_enabled=True)

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_ranker",
        lambda **_: SimpleNamespace(model=object(), models=None),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_rank_score",
        lambda _model, ds: np.linspace(-0.4, 0.4, ds.X.shape[0], dtype=np.float32),
    )

    out = _fit_predict_fold_dual_side(
        train_long=train_long,
        valid_long=valid_long,
        test_long=test_long,
        train_short=train_short,
        valid_short=valid_short,
        test_short=test_short,
        ml_cfg=cfg,
    )

    assert out.ev_test_long.shape == (test_long.X.shape[0],)
    assert out.ev_test_short.shape == (test_short.X.shape[0],)
    assert np.all(out.ev_test_long >= 0.0)
    assert np.all(out.ev_test_short >= 0.0)
    assert np.all(out.conf_test_long == 1.0)
    assert np.all(out.conf_test_short == 1.0)


def test_resolve_side_targets_cs_residual_uses_residual_rank_and_relevance() -> None:
    labels = LabelPanel(
        long_net_ret=np.zeros((2, 3), dtype=np.float32),
        short_net_ret=np.zeros((2, 3), dtype=np.float32),
        signed_net_ret=np.zeros((2, 3), dtype=np.float32),
        exec_net_ret=np.zeros((2, 3), dtype=np.float32),
        relevance=np.array([[0, 2, 4], [1, 3, 4]], dtype=np.int32),
        sample_weight=np.ones((2, 3), dtype=np.float32),
        eligible_mask=np.ones((2, 3), dtype=bool),
        rank_target=np.array([[0.1, -0.2, 0.3], [0.0, 0.4, -0.5]], dtype=np.float32),
    )
    ml_cfg = StrategyMLConfig(rank_target_mode="cs_residual")
    long_rank, short_rank, long_rel, short_rel, _long_mag, _short_mag = _resolve_side_targets(
        labels, ml_cfg
    )

    assert long_rank is not None
    assert short_rank is not None
    assert labels.rank_target is not None
    assert np.allclose(long_rank, labels.rank_target)
    assert np.allclose(short_rank, -labels.rank_target)
    assert np.array_equal(long_rel, labels.relevance)
    assert np.array_equal(short_rel, 4 - labels.relevance)
