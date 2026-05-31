from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
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
    _fit_predict_fold_dual_side,
    _rank_score,
    _resolve_side_targets,
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
        name="lambdamart",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    caplog.set_level("INFO")

    panel = build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)

    assert "ML-PARALLEL: Completed all" in caplog.text
    assert "ML_EVAL" in caplog.text
    assert "ML_COST" in caplog.text
    assert float(panel["alpha_long"].sum()) > 0.0
    assert float(panel["alpha_short"].sum()) > 0.0
    assert float(panel.loc[(datetimes[2], "BTCUSDT"), "alpha_long"]) >= 0.0
    assert float(panel.loc[(datetimes[2], "XRPUSDT"), "alpha_short"]) >= 0.0
    q = panel.attrs["quality_report"]
    assert 0.0 <= float(q.get("xs_long_preservation_ratio", 0.0)) <= 1.0
    assert 0.0 <= float(q.get("xs_short_preservation_ratio", 0.0)) <= 1.0


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
        name="lambdamart",
        ml=StrategyMLConfig(
            min_group_size=2,
            train_months=1,
            valid_months=1,
            test_months=1,
            ranker_enabled=True,  # explicit: test verifies ranker call path
        ),
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


def test_build_ml_strategy_alpha_anchored_preserves_short_side_opportunity(
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
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda *_: np.array([0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32),
    )

    cfg = StrategyConfig(
        name="lambdamart",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    panel = build_ml_strategy_alpha_anchored(
        data_maps={},
        symbols=list(symbols),
        tf="4h",
        cfg=cfg,
        anchor_end_idx=35,
        target_start=35,
        target_end=36,
    )
    target = panel.loc[(datetimes[35], slice(None)), :]
    assert float(target["alpha_long"].sum()) > 0.0
    assert float(target["alpha_short"].sum()) > 0.0


def test_build_ml_strategy_alpha_anchored_test_alpha_independent_of_future_labels(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test 구간 alpha가 미래 실현수익 label과 무관함을 검증 (leakage 회귀 테스트).

    cost_clearance_target 계열은 LabelDiagnostics로 격리(LabelPanel에서 삭제)됨.
    레이블이 변경되어도 test 구간 alpha가 동일해야 한다.
    """
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

    def _make_label_panel(signed_val: float) -> LabelPanel:
        # cost_clearance_target 계열은 LabelDiagnostics로 격리됨 — LabelPanel에 없음
        signed = np.full((40, 5), signed_val, dtype=np.float32)
        return LabelPanel(
            long_net_ret=np.zeros((40, 5), dtype=np.float32),
            short_net_ret=np.zeros((40, 5), dtype=np.float32),
            signed_net_ret=signed,
            exec_net_ret=np.zeros((40, 5), dtype=np.float32),
            relevance=np.full((40, 5), 2, dtype=np.int32),
            relevance_long=np.full((40, 5), 2, dtype=np.int32),
            relevance_short=np.full((40, 5), 2, dtype=np.int32),
            magnitude_target_long=np.ones((40, 5), dtype=np.float32) * 0.01,
            magnitude_target_short=np.ones((40, 5), dtype=np.float32) * 0.02,
            sample_weight=np.ones((40, 5), dtype=np.float32),
            eligible_mask=np.ones((40, 5), dtype=bool),
        )

    label_state: dict[str, LabelPanel] = {"value": _make_label_panel(1.0)}

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.align_data_maps",
        lambda data_maps, symbols, tf: SimpleNamespace(symbols=list(symbols)),
    )
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_feature_panel", lambda *_: fp)
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_label_panel",
        lambda *_: label_state["value"],
    )

    def _build_long_matrix_stub(
        *,
        features: FeaturePanel,
        labels: LabelPanel,
        start: int,
        end: int,
        fold: FoldSpec,
        split: str,
        min_group_size: int,
        rank_target_override: np.ndarray | None = None,
        relevance_override: np.ndarray | None = None,
        ev_target_override: np.ndarray | None = None,
    ) -> LongMatrixDataset:
        del (
            features,
            labels,
            fold,
            min_group_size,
            rank_target_override,
            relevance_override,
            ev_target_override,
        )
        rows = (end - start) * len(symbols)
        idx = [[t, s] for t in range(start, end) for s in range(len(symbols))]
        return _dataset(
            x=np.ones((rows, 2), dtype=np.float32),
            group=np.array([len(symbols)] * (end - start), dtype=np.int32),
            index_map=np.asarray(idx, dtype=np.int64),
            feature_names=feature_names,
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.build_long_matrix",
        _build_long_matrix_stub,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_ranker",
        lambda **_: SimpleNamespace(model=object()),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_rank_score",
        lambda _m, ds: np.linspace(-0.5, 0.5, ds.X.shape[0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_quantile_calibrators",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda _m, ds, _s, _c: np.ones(ds.X.shape[0], dtype=np.float32) * np.float32(0.05),
    )

    cfg = StrategyConfig(
        name="lambdamart",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )

    # Arrange: run with signed_net_ret = all-ones (label variant A)
    label_state["value"] = _make_label_panel(1.0)
    panel_cleared = build_ml_strategy_alpha_anchored(
        data_maps={},
        symbols=list(symbols),
        tf="4h",
        cfg=cfg,
        anchor_end_idx=35,
        target_start=35,
        target_end=36,
    )

    # Act: run with signed_net_ret = all-zeros (label variant B)
    label_state["value"] = _make_label_panel(0.0)
    panel_blocked = build_ml_strategy_alpha_anchored(
        data_maps={},
        symbols=list(symbols),
        tf="4h",
        cfg=cfg,
        anchor_end_idx=35,
        target_start=35,
        target_end=36,
    )

    # Assert: test window alpha must be identical regardless of cost_clearance_target
    target_ts = datetimes[35]
    long_cleared = panel_cleared.loc[(target_ts, slice(None)), "alpha_long"].to_numpy()
    long_blocked = panel_blocked.loc[(target_ts, slice(None)), "alpha_long"].to_numpy()
    np.testing.assert_array_equal(
        long_cleared,
        long_blocked,
        err_msg="alpha_long must be deterministic (test window is isolated from train labels)",
    )
    short_cleared = panel_cleared.loc[(target_ts, slice(None)), "alpha_short"].to_numpy()
    short_blocked = panel_blocked.loc[(target_ts, slice(None)), "alpha_short"].to_numpy()
    np.testing.assert_array_equal(
        short_cleared,
        short_blocked,
        err_msg="alpha_short must be deterministic (test window is isolated from train labels)",
    )


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
    assert np.allclose(long_rank, labels.rank_target)
    assert np.allclose(short_rank, -labels.rank_target)
    assert np.array_equal(long_rel, labels.relevance)
    assert np.array_equal(short_rel, 4 - labels.relevance)


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
        rank_target_override: np.ndarray | None = None,
        relevance_override: np.ndarray | None = None,
        ev_target_override: np.ndarray | None = None,
    ) -> LongMatrixDataset:
        del labels, min_group_size, rank_target_override, relevance_override, ev_target_override
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
        name="lambdamart",
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
            "alpha_p95_bps": 10.0 if state["horizon"] == 12 else 14.0,
            "alpha_active_p95_bps": 35.0 if state["horizon"] == 12 else 18.0,
            "alpha_long_tradable_nz": 0.20 if state["horizon"] == 12 else 0.05,
            "alpha_short_tradable_nz": 0.15 if state["horizon"] == 12 else 0.05,
            "in_fold_valid_alpha_p95_bps": 20.0 if state["horizon"] == 12 else 40.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.side_alpha_tail_metrics",
        lambda alpha_long, alpha_short, *, cost_floor: {
            "alpha_full_matrix_p95_bps": 10.0 if state["horizon"] == 12 else 14.0,
            "alpha_active_p95_bps": 35.0 if state["horizon"] == 12 else 18.0,
            "alpha_long_active_p95_bps": 30.0 if state["horizon"] == 12 else 16.0,
            "alpha_short_active_p95_bps": 35.0 if state["horizon"] == 12 else 18.0,
            "alpha_long_tradable_nz": 0.20 if state["horizon"] == 12 else 0.05,
            "alpha_short_tradable_nz": 0.15 if state["horizon"] == 12 else 0.05,
            "alpha_long_active_count": 10.0,
            "alpha_short_active_count": 10.0,
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
        name="lambdamart",
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
    assert "active_p95_bps" in panel.attrs["horizon_experiment"][0]
    assert "full_matrix_p95_bps" in panel.attrs["horizon_experiment"][0]
    assert "tradable_density" in panel.attrs["horizon_experiment"][0]
    assert panel.attrs["baseline_harness"]["mode"] == "horizon_experiment"


def test_build_ml_strategy_alpha_gate_uses_oos_alpha_p95_not_in_fold(
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
    called: dict[str, float] = {}
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
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_long_matrix", lambda **_: ds)
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
        lambda *_: np.array([0.02] * 10, dtype=np.float32),
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
            "alpha_p95_bps": 7.0,
            "in_fold_valid_alpha_p95_bps": 77.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_quality_gate",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_ic_gate",
        lambda *_, **__: True,
    )

    def _capture_alpha_gate(**kwargs: float) -> dict[str, object]:
        called["alpha_p95_bps"] = float(kwargs["alpha_p95_bps"])
        called["active_alpha_p95_bps"] = float(kwargs["active_alpha_p95_bps"])
        called["min_tradable_long_nz"] = float(kwargs["min_tradable_long_nz"])
        called["min_tradable_short_nz"] = float(kwargs["min_tradable_short_nz"])
        return {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        }

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        _capture_alpha_gate,
    )

    cfg = StrategyConfig(
        name="lambdamart",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)
    assert called["alpha_p95_bps"] == pytest.approx(7.0)
    assert called["active_alpha_p95_bps"] > 0.0
    assert called["min_tradable_long_nz"] == pytest.approx(cfg.ml.alpha_gate_min_tradable_long_nz)
    assert called["min_tradable_short_nz"] == pytest.approx(
        cfg.ml.alpha_gate_min_tradable_short_nz
    )


def test_build_ml_strategy_alpha_anchored_gate_uses_oos_alpha_p95_not_in_fold(
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
    called: dict[str, float] = {}

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
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda *_: np.ones(5, dtype=np.float32) * np.float32(0.02),
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
            "alpha_p95_bps": 6.0,
            "in_fold_valid_alpha_p95_bps": 66.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_quality_gate",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_ic_gate",
        lambda *_, **__: True,
    )

    def _capture_alpha_gate(**kwargs: float) -> dict[str, object]:
        called["alpha_p95_bps"] = float(kwargs["alpha_p95_bps"])
        called["active_alpha_p95_bps"] = float(kwargs["active_alpha_p95_bps"])
        return {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
        }

    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        _capture_alpha_gate,
    )

    cfg = StrategyConfig(
        name="lambdamart",
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
    assert called["alpha_p95_bps"] == pytest.approx(6.0)
    assert called["active_alpha_p95_bps"] > 0.0


def test_build_ml_strategy_alpha_logs_cost_wall_gate_metric_source(
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
    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.build_long_matrix", lambda **_: ds)
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
        lambda *_: np.array([0.02] * 10, dtype=np.float32),
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
            "alpha_p95_bps": 8.0,
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_quality_gate",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.passes_ic_gate",
        lambda *_, **__: True,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.alpha_gate_diagnostics",
        lambda **_: {
            "alpha_gate_pass": True,
            "alpha_gate_fail_reasons": [],
            "alpha_gate_floor_bps": 0.0,
            "alpha_gate_metric_bps": 30.0,
            "alpha_gate_metric_source": "active_alpha_p95_bps",
        },
    )

    cfg = StrategyConfig(
        name="lambdamart",
        ml=StrategyMLConfig(min_group_size=2, train_months=1, valid_months=1, test_months=1),
    )
    caplog.set_level("INFO")
    build_ml_strategy_alpha(data_maps={}, symbols=list(symbols), tf="4h", cfg=cfg)
    assert "ML_COST: gate=30.0bps floor=24.0bps pass=true" in caplog.text


# ---------------------------------------------------------------------------
# T-B: ranker ablation (_rank_score, ranker_enabled)
# ---------------------------------------------------------------------------


def test_rank_score_returns_nans_when_fit_result_is_none() -> None:
    """_rank_score(None, dataset) must return all-NaN float32 array of correct length."""
    # Arrange
    feature_names: tuple[str, ...] = ("f0", "f1")
    ds = LongMatrixDataset(
        X=np.ones((8, 2), dtype=np.float32),
        y_rank=np.arange(8, dtype=np.int32),
        y_ev=np.zeros(8, dtype=np.float32),
        group=np.array([4, 4], dtype=np.int32),
        sample_weight=np.ones(8, dtype=np.float32),
        index_map=np.zeros((8, 2), dtype=np.int64),
        feature_names=feature_names,
    )

    # Act
    scores = _rank_score(None, ds)

    # Assert
    assert scores.shape == (8,)
    assert scores.dtype == np.float32
    assert np.all(np.isnan(scores))


def test_ranker_enabled_false_skips_fit_ranker(monkeypatch: MonkeyPatch) -> None:
    """When ranker_enabled=False, fit_ranker must NOT be called."""
    # Arrange
    feature_names: tuple[str, ...] = ("f0", "f1")
    fit_ranker_calls: list[int] = []

    def _mock_fit_ranker(**_kwargs: object) -> object:
        fit_ranker_calls.append(1)
        return object()  # should never be reached

    monkeypatch.setattr("src.domain.futures.strategy.ml_builder.fit_ranker", _mock_fit_ranker)
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.fit_quantile_calibrators",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ml_builder.predict_conservative_ev",
        lambda *_: np.zeros(5, dtype=np.float32),
    )

    def _make_ds(n: int) -> LongMatrixDataset:
        return LongMatrixDataset(
            X=np.ones((n, 2), dtype=np.float32),
            y_rank=np.arange(n, dtype=np.int32),
            y_ev=np.zeros(n, dtype=np.float32),
            group=np.full(max(1, n // 5), 5, dtype=np.int32),
            sample_weight=np.ones(n, dtype=np.float32),
            index_map=np.zeros((n, 2), dtype=np.int64),
            feature_names=feature_names,
        )

    cfg = StrategyMLConfig(ranker_enabled=False, min_group_size=2)

    # Act
    _fit_predict_fold_dual_side(
        train_long=_make_ds(10),
        valid_long=_make_ds(5),
        test_long=_make_ds(5),
        train_short=_make_ds(10),
        valid_short=_make_ds(5),
        test_short=_make_ds(5),
        ml_cfg=cfg,
    )

    # Assert — fit_ranker was never called
    assert len(fit_ranker_calls) == 0


def test_emit_rank_sized_alpha_preserves_breadth_and_positive_presv() -> None:
    """rank_sized emission이 EV-clip보다 더 넓은 breadth와 양수 presv를 보장하는지 검증."""
    from src.domain.futures.strategy.ml_builder import _emit_rank_sized_alpha

    rng = np.random.default_rng(42)
    T, N = 20, 12
    rank_score_long = rng.standard_normal((T, N)).astype(np.float32)
    rank_score_short = rng.standard_normal((T, N)).astype(np.float32)
    eligible = np.ones((T, N), dtype=bool)

    # Act
    al, as_ = _emit_rank_sized_alpha(
        rank_score_long,
        rank_score_short,
        eligible,
        select_q=0.40,
        weight_k=3.0,
        clip_lim=1.0,
    )

    # Assert shape
    assert al.shape == (T, N)
    assert as_.shape == (T, N)
    assert al.dtype == np.float32
    assert as_.dtype == np.float32

    # breadth: 각 timestep에서 비영(non-zero) 심볼 수 ≥ ceil(N*select_q)
    min_keep = int(np.ceil(N * 0.40))
    long_nz = (al > 0).sum(axis=1)
    short_nz = (as_ > 0).sum(axis=1)
    assert (long_nz >= min_keep).all(), f"long breadth 부족: min={long_nz.min()}"
    assert (short_nz >= min_keep).all(), f"short breadth 부족: min={short_nz.min()}"

    # 모든 값이 [0, clip_lim] 범위 내
    assert al.min() >= 0.0
    assert al.max() <= 1.0 + 1e-6
    assert as_.min() >= 0.0
    assert as_.max() <= 1.0 + 1e-6


def test_emit_rank_sized_alpha_skips_sparse_timestep() -> None:
    """n_elig < 2인 timestep에서 출력이 0임을 검증."""
    from src.domain.futures.strategy.ml_builder import _emit_rank_sized_alpha

    T, N = 3, 5
    rank_score = np.ones((T, N), dtype=np.float32)
    eligible = np.zeros((T, N), dtype=bool)
    eligible[0, :1] = True  # t=0: n_elig=1 (<2 → skip)
    eligible[1, :3] = True  # t=1: n_elig=3 (emit)
    eligible[2, :] = True   # t=2: n_elig=5 (emit)

    al, as_ = _emit_rank_sized_alpha(
        rank_score, rank_score, eligible,
        select_q=0.40, weight_k=3.0, clip_lim=1.0,
    )

    assert (al[0] == 0.0).all(), "n_elig<2 timestep은 0이어야 함"
    assert (as_[0] == 0.0).all()
    assert (al[1] > 0).any(), "t=1 emit 미발생"
    assert (al[2] > 0).any(), "t=2 emit 미발생"
