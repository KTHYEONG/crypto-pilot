from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from src.domain.futures.strategy.config import StrategyConfig, StrategyMLConfig
from src.domain.futures.strategy.contracts import (
    FeaturePanel,
    FoldSpec,
    LabelPanel,
    LongMatrixDataset,
)
from src.domain.futures.strategy.ml_builder import build_ml_strategy_alpha


def _dataset(
    *,
    x: np.ndarray,
    group: np.ndarray,
    index_map: np.ndarray,
    feature_names: tuple[str, ...],
) -> LongMatrixDataset:
    return LongMatrixDataset(
        X=x.astype(np.float32),
        y_rank=np.arange(x.shape[0], dtype=np.int32),
        y_ev=np.linspace(-1e-3, 1e-3, x.shape[0], dtype=np.float32),
        group=group.astype(np.int32),
        sample_weight=np.ones((x.shape[0],), dtype=np.float32),
        index_map=index_map.astype(np.int64),
        feature_names=feature_names,
    )


def test_build_ml_strategy_alpha_emits_orchestration_tags(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    datetimes = np.array(
        [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-01T04:00:00"),
            np.datetime64("2024-01-01T08:00:00"),
        ],
        dtype="datetime64[ns]",
    )
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
    feature_names = ("f0", "f1")
    fp = FeaturePanel(
        datetimes=datetimes,
        symbols=symbols,
        values=np.ones((3, 5, 2), dtype=np.float32),
        feature_names=feature_names,
        valid_mask=np.ones((3, 5), dtype=bool),
    )
    lp = LabelPanel(
        long_net_ret=np.zeros((3, 5), dtype=np.float32),
        short_net_ret=np.zeros((3, 5), dtype=np.float32),
        signed_net_ret=np.zeros((3, 5), dtype=np.float32),
        relevance=np.full((3, 5), 2, dtype=np.int32),
        sample_weight=np.ones((3, 5), dtype=np.float32),
        eligible_mask=np.ones((3, 5), dtype=bool),
    )

    ds_train = _dataset(
        x=np.ones((10, 2), dtype=np.float32),
        group=np.array([5, 5], dtype=np.int32),
        index_map=np.array(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]
        ),
        feature_names=feature_names,
    )
    ds_valid = _dataset(
        x=np.ones((5, 2), dtype=np.float32),
        group=np.array([5], dtype=np.int32),
        index_map=np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]),
        feature_names=feature_names,
    )
    ds_test = _dataset(
        x=np.ones((5, 2), dtype=np.float32),
        group=np.array([5], dtype=np.int32),
        index_map=np.array([[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]]),
        feature_names=feature_names,
    )

    fold = FoldSpec(
        fold_id=0,
        train_start=0,
        train_end=1,
        valid_start=1,
        valid_end=2,
        test_start=2,
        test_end=3,
        purge_bars=1,
        embargo_bars=1,
    )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.align_data_maps",
        lambda data_maps, symbols, tf: SimpleNamespace(symbols=list(symbols)),
    )
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_feature_panel", lambda *_: fp)
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_label_panel", lambda *_: lp)
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.make_walk_forward_folds",
        lambda *_: [fold],
    )

    def _build_long_matrix(*, split: str, **kwargs: object) -> LongMatrixDataset:
        if split == "train":
            return ds_train
        if split == "valid":
            return ds_valid
        return ds_test

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_long_matrix",
        _build_long_matrix,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_ranker",
        lambda **_: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_rank_score",
        lambda _model, ds: np.linspace(-0.5, 0.5, ds.X.shape[0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_quantile_calibrators",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda *_: np.array([0.20, -0.10, 0.30, -0.40, 0.15], dtype=np.float32),
    )

    cfg = StrategyConfig(
        name="ml_lambdamart_v1",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    caplog.set_level("INFO")

    panel = build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)

    assert "[ML-FOLD]" in caplog.text
    assert "[ML-RANKER]" in caplog.text
    assert "[ML-CALIB]" in caplog.text
    assert "[ML-OOS]" in caplog.text
    assert float(panel["alpha_long"].sum()) > 0.0
    assert float(panel["alpha_short"].sum()) > 0.0
