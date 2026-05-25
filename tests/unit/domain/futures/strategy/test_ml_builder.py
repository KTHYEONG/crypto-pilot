from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from numpy.typing import NDArray

from src.domain.futures.strategy.config import StrategyConfig, StrategyMLConfig
from src.domain.futures.strategy.contracts import (
    FeaturePanel,
    FoldSpec,
    LabelPanel,
    LongMatrixDataset,
)
from src.domain.futures.strategy.ml_builder import (
    build_ml_strategy_alpha,
    build_ml_strategy_alpha_anchored,
)


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
        exec_net_ret=np.zeros((3, 5), dtype=np.float32),
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
    assert float(panel.loc[(datetimes[2], "BTCUSDT"), "alpha_long"]) >= 0.0
    assert float(panel.loc[(datetimes[2], "XRPUSDT"), "alpha_short"]) >= 0.0


def test_build_ml_strategy_alpha_filters_nonfinite_and_clips_test_outlier(
    monkeypatch: MonkeyPatch,
) -> None:
    datetimes = np.array(
        [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-01T04:00:00"),
            np.datetime64("2024-01-01T08:00:00"),
            np.datetime64("2024-01-01T12:00:00"),
        ],
        dtype="datetime64[ns]",
    )
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
    feature_names = tuple(f"f{i}" for i in range(20))
    values: NDArray[np.float32] = np.ones((4, 5, 20), dtype=np.float32)
    values[2, 0, 0] = np.nan
    values[3, 4, 1] = 1_000_000.0
    fp = FeaturePanel(
        datetimes=datetimes,
        symbols=symbols,
        values=values,
        feature_names=feature_names,
        valid_mask=np.ones((4, 5), dtype=bool),
    )
    lp = LabelPanel(
        long_net_ret=np.zeros((4, 5), dtype=np.float32),
        short_net_ret=np.zeros((4, 5), dtype=np.float32),
        signed_net_ret=np.tile(
            np.array([-2e-3, -1e-3, 0.0, 1e-3, 2e-3], dtype=np.float32),
            (4, 1),
        ),
        exec_net_ret=np.tile(
            np.array([-2e-3, -1e-3, 0.0, 1e-3, 2e-3], dtype=np.float32),
            (4, 1),
        ),
        relevance=np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int32), (4, 1)),
        sample_weight=np.ones((4, 5), dtype=np.float32),
        eligible_mask=np.ones((4, 5), dtype=bool),
    )
    fold = FoldSpec(
        fold_id=0,
        train_start=0,
        train_end=1,
        valid_start=1,
        valid_end=2,
        test_start=2,
        test_end=4,
        purge_bars=1,
        embargo_bars=1,
    )
    captured: dict[str, LongMatrixDataset] = {}

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

    def _fit_model(
        *,
        train: LongMatrixDataset,
        valid: LongMatrixDataset,
        cfg: StrategyMLConfig,
    ) -> SimpleNamespace:
        del cfg
        captured["train"] = train
        captured["valid"] = valid
        return SimpleNamespace(model=object())

    def _predict_score(_model: object, ds: LongMatrixDataset) -> NDArray[np.float32]:
        captured[f"score_input_{len(captured)}"] = ds
        return np.asarray(np.linspace(-0.5, 0.5, ds.X.shape[0], dtype=np.float32))

    def _fit_quantile_models(
        *,
        train: LongMatrixDataset,
        valid: LongMatrixDataset,
        rank_score_train: np.ndarray,
        rank_score_valid: np.ndarray,
        cfg: StrategyMLConfig,
    ) -> SimpleNamespace:
        del rank_score_train, rank_score_valid, cfg
        captured["fit_train"] = train
        captured["fit_valid"] = valid
        return SimpleNamespace()

    def _predict_conservative_ev(
        _models: object,
        dataset: LongMatrixDataset,
        _rank_score: NDArray[np.float32],
        _cfg: StrategyMLConfig,
    ) -> NDArray[np.float32]:
        captured["test"] = dataset
        return np.asarray(np.array([0.20, -0.10, 0.30, -0.40, 0.15] * 2, dtype=np.float32))

    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.fit_ranker", _fit_model)
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_rank_score",
        _predict_score,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_quantile_calibrators",
        _fit_quantile_models,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        _predict_conservative_ev,
    )

    cfg = StrategyConfig(
        name="ml_lambdamart_v1",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    panel = build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)

    for split in ("train", "valid", "test", "fit_train", "fit_valid"):
        assert split in captured
        assert np.all(np.isfinite(captured[split].X))
    assert captured["test"].X.shape == (10, 20)
    assert float(np.max(captured["test"].X)) == 1.0
    assert float(np.min(captured["test"].X)) == 1.0
    assert np.all(np.isfinite(panel[["alpha_long", "alpha_short"]].to_numpy()))
    assert float(panel["alpha_long"].sum()) > 0.0
    assert float(panel["alpha_short"].sum()) > 0.0


def test_build_ml_strategy_alpha_anchored_does_not_center_ev_by_group(
    monkeypatch: MonkeyPatch,
) -> None:
    datetimes = np.array(
        [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(4 * i, "h") for i in range(40)],
        dtype="datetime64[ns]",
    )
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
    feature_names = ("f0", "f1")
    fp = FeaturePanel(
        datetimes=datetimes,
        symbols=symbols,
        values=np.ones((40, 5, 2), dtype=np.float32),
        feature_names=feature_names,
        valid_mask=np.ones((40, 5), dtype=bool),
    )
    lp = LabelPanel(
        long_net_ret=np.zeros((40, 5), dtype=np.float32),
        short_net_ret=np.zeros((40, 5), dtype=np.float32),
        signed_net_ret=np.zeros((40, 5), dtype=np.float32),
        exec_net_ret=np.zeros((40, 5), dtype=np.float32),
        relevance=np.full((40, 5), 2, dtype=np.int32),
        sample_weight=np.ones((40, 5), dtype=np.float32),
        eligible_mask=np.ones((40, 5), dtype=bool),
    )
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.align_data_maps",
        lambda data_maps, symbols, tf: SimpleNamespace(symbols=list(symbols)),
    )
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_feature_panel", lambda *_: fp)
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_label_panel", lambda *_: lp)
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

    expected_ev = np.array([0.20, -0.10, 0.30, -0.40, 0.15], dtype=np.float32)
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda *_: expected_ev.copy(),
    )

    def _infer_fold_alpha(
        *,
        fold: FoldSpec,
        test: LongMatrixDataset,
        ev_test: np.ndarray,
        t_size: int,
        n_size: int,
    ) -> SimpleNamespace:
        del fold, test, t_size, n_size
        captured["ev_test"] = ev_test.copy()
        return SimpleNamespace(ev_grid=np.zeros((40, 5), dtype=np.float32))

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.infer_fold_alpha",
        _infer_fold_alpha,
    )

    cfg = StrategyConfig(
        name="ml_lambdamart_v1",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    build_ml_strategy_alpha_anchored(
        data_maps={},
        symbols=list(symbols),
        tf="4h",
        cfg=cfg,
        anchor_end_idx=35,
        target_start=35,
        target_end=36,
    )

    np.testing.assert_allclose(captured["ev_test"], expected_ev, rtol=0.0, atol=0.0)


def test_build_ml_strategy_alpha_virtual_refit_uses_own_train_normalization(
    monkeypatch: MonkeyPatch,
) -> None:
    datetimes = np.array(
        [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(4 * i, "h") for i in range(5)],
        dtype="datetime64[ns]",
    )
    symbols = ("BTCUSDT", "ETHUSDT")
    feature_names = ("f0",)
    values = np.array(
        [
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[3.0], [3.0]],
            [[4.0], [4.0]],
            [[5.0], [5.0]],
        ],
        dtype=np.float32,
    )
    fp = FeaturePanel(
        datetimes=datetimes,
        symbols=symbols,
        values=values,
        feature_names=feature_names,
        valid_mask=np.ones((5, 2), dtype=bool),
    )
    lp = LabelPanel(
        long_net_ret=np.zeros((5, 2), dtype=np.float32),
        short_net_ret=np.zeros((5, 2), dtype=np.float32),
        signed_net_ret=np.ones((5, 2), dtype=np.float32) * 1e-3,
        exec_net_ret=np.ones((5, 2), dtype=np.float32) * 1e-3,
        relevance=np.full((5, 2), 2, dtype=np.int32),
        sample_weight=np.ones((5, 2), dtype=np.float32),
        eligible_mask=np.ones((5, 2), dtype=bool),
        dynamic_cost_bps_2d=np.zeros((5, 2), dtype=np.float32),
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
    bound_markers: list[float] = []
    captured_split_marker: dict[tuple[int, str], float] = {}

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

    def _fit_robust_bounds(train_values: np.ndarray, clip_quantile: float) -> dict[str, float]:
        del clip_quantile
        marker = float(np.mean(train_values))
        bound_markers.append(marker)
        return {"marker": marker}

    def _apply_robust_bounds(
        all_values: np.ndarray,
        bounds: dict[str, float],
    ) -> NDArray[np.float64]:
        filled: NDArray[np.float64] = np.asarray(
            np.full_like(all_values, float(bounds["marker"]), dtype=np.float64),
            dtype=np.float64,
        )
        return filled

    def _build_long_matrix(
        *,
        features: FeaturePanel,
        labels: LabelPanel,
        start: int,
        end: int,
        fold: FoldSpec,
        split: str,
        min_group_size: int,
    ) -> LongMatrixDataset:
        del labels, min_group_size
        captured_split_marker[(fold.fold_id, split)] = float(features.values[start, 0, 0])
        rows = max(1, (end - start) * len(symbols))
        index_map: list[list[int]] = []
        for t in range(start, max(start + 1, end)):
            index_map.extend([[t, s_idx] for s_idx in range(len(symbols))])
        x: NDArray[np.float32] = np.ones((rows, 1), dtype=np.float32)
        return _dataset(
            x=x,
            group=np.array([len(index_map)], dtype=np.int32),
            index_map=np.asarray(index_map[:rows], dtype=np.int64),
            feature_names=feature_names,
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_robust_bounds",
        _fit_robust_bounds,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.apply_robust_bounds",
        _apply_robust_bounds,
    )
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
        lambda _model, ds: np.zeros(ds.X.shape[0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_quantile_calibrators",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda calib, ds, score, cfg: np.ones(ds.X.shape[0], dtype=np.float32) * 5e-3,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.infer_fold_alpha",
        lambda **_: SimpleNamespace(ev_grid=np.ones((5, 2), dtype=np.float32) * 5e-3),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.assemble_alpha_panel",
        lambda **_: pd.DataFrame(
            {
                "alpha_long": np.ones(10, dtype=np.float32) * 1e-3,
                "alpha_short": np.ones(10, dtype=np.float32) * 1e-3,
            },
            index=pd.MultiIndex.from_product([datetimes, symbols], names=["datetime", "symbol"]),
        ),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_quality_report",
        lambda **_: {
            "feature_finite_ratio": 1.0,
            "label_valid_ratio": 1.0,
            "ranker_valid_ndcg_at_5": 1.0,
            "spearman_rank_ic": 0.1,
            "ic_icir": 0.1,
            "ic_t_stat": 2.5,
            "ic_hit_ratio": 0.5,
            "ic_n_obs": 1,
            "alpha_p95_bps": 30.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_quality_gate",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.ml_alpha_metrics",
        lambda *_: {
            "long_nz": 1.0,
            "short_nz": 1.0,
            "long_p95_bps": 10.0,
            "short_p95_bps": 10.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_ic_gate",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        lambda **_: {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        lambda **_: {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        lambda **_: {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        lambda **_: {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        },
    )

    cfg = StrategyConfig(
        name="ml_lambdamart_v1",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)

    assert len(bound_markers) == 2
    assert bound_markers[0] == 1.0
    assert bound_markers[1] == 1.5
    assert captured_split_marker[(1, "train")] == 1.5
    assert captured_split_marker[(1, "valid")] == 1.5
    assert captured_split_marker[(1, "test")] == 1.5


def test_build_ml_strategy_alpha_selects_best_horizon_and_records_metadata(
    monkeypatch: MonkeyPatch,
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
        exec_net_ret=np.zeros((3, 5), dtype=np.float32),
        relevance=np.full((3, 5), 2, dtype=np.int32),
        sample_weight=np.ones((3, 5), dtype=np.float32),
        eligible_mask=np.ones((3, 5), dtype=bool),
        dynamic_cost_bps_2d=np.zeros((3, 5), dtype=np.float32),
    )
    ds = _dataset(
        x=np.ones((10, 2), dtype=np.float32),
        group=np.array([5, 5], dtype=np.int32),
        index_map=np.array([[2, i % 5] for i in range(10)], dtype=np.int64),
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
    state: dict[str, int] = {"horizon": 0}

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.align_data_maps",
        lambda data_maps, symbols, tf: SimpleNamespace(symbols=list(symbols)),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_feature_panel",
        lambda *_: fp,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_label_panel",
        lambda aligned, ml_cfg: (
            state.__setitem__("horizon", int(ml_cfg.label_horizon_bars)) or lp
        ),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.make_walk_forward_folds",
        lambda *_: [fold],
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_long_matrix",
        lambda **_: ds,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_ranker",
        lambda **_: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_rank_score",
        lambda _model, ds: np.zeros(ds.X.shape[0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_quantile_calibrators",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda *_: np.asarray(
            [5e-3 if i % 2 == 0 else -5e-3 for i in range(ds.X.shape[0])],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_quality_report",
        lambda **_: {
            "feature_finite_ratio": 1.0,
            "label_valid_ratio": 1.0,
            "ranker_valid_ndcg_at_5": 1.0,
            "spearman_rank_ic": 0.1,
            "ic_icir": 0.1,
            "ic_t_stat": 2.5,
            "ic_hit_ratio": 0.5,
            "ic_n_obs": 1,
            "alpha_p95_bps": 10.0,
            "in_fold_valid_alpha_p95_bps": 40.0 if state["horizon"] == 12 else 28.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_quality_gate",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.ml_alpha_metrics",
        lambda *_: {
            "long_nz": 1.0,
            "short_nz": 1.0,
            "long_p95_bps": 10.0,
            "short_p95_bps": 10.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_ic_gate",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        lambda **_: {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        },
    )

    cfg = StrategyConfig(
        name="ml_lambdamart_v1",
        ml=StrategyMLConfig(
            min_group_size=2,
            train_months=1,
            valid_months=1,
            test_months=1,
            horizon_experiment_enabled=True,
            horizon_candidates=(6, 12),
        ),
    )
    panel = build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)
    assert panel.attrs["selected_horizon"] == 12
    assert len(panel.attrs["horizon_experiment"]) == 2
    assert panel.attrs["baseline_harness"]["mode"] == "horizon_experiment"
