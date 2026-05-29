from __future__ import annotations

import typing
import numpy as np
from _pytest.monkeypatch import MonkeyPatch
from numpy.typing import NDArray

from src.domain.futures.strategy.calibrator import (
    compute_conservative_ev,
    fit_quantile_calibrators,
    predict_conservative_ev,
    predict_ev_quantiles,
)
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LongMatrixDataset


def _dataset(rows: int, groups: int, features: int = 5) -> LongMatrixDataset:
    rng = np.random.default_rng(7)
    group_size = rows // groups
    x = rng.normal(size=(rows, features)).astype(np.float32)
    y_rank = np.tile(np.arange(group_size, dtype=np.int32), groups)
    y_ev = rng.normal(scale=1e-3, size=rows).astype(np.float32)
    group: NDArray[np.int32] = np.full((groups,), group_size, dtype=np.int32)
    sample_weight: NDArray[np.float32] = np.ones((rows,), dtype=np.float32)
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


def test_fit_quantile_calibrators_and_predict_ev_clip() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    test = _dataset(rows=36, groups=4)
    cfg = StrategyMLConfig(calibrator_n_estimators=20, early_stopping_rounds=10, n_jobs=1)

    score_train = np.linspace(-1.0, 1.0, train.X.shape[0], dtype=np.float32)
    score_valid = np.linspace(-0.8, 0.8, valid.X.shape[0], dtype=np.float32)
    score_test = np.linspace(-0.5, 0.5, test.X.shape[0], dtype=np.float32)

    fit_result = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=score_train,
        rank_score_valid=score_valid,
        cfg=cfg,
    )
    ev = predict_conservative_ev(models=fit_result, dataset=test, rank_score=score_test, cfg=cfg)

    clip = cfg.alpha_clip_bps / 10000.0
    assert ev.shape == (test.X.shape[0],)
    assert ev.dtype == np.float32
    assert np.all(np.isfinite(ev))
    assert np.max(np.abs(ev)) <= clip + 1e-12


def test_predict_conservative_ev_empty_returns_empty() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    cfg = StrategyMLConfig(calibrator_n_estimators=10, early_stopping_rounds=5, n_jobs=1)

    score_train = np.zeros((train.X.shape[0],), dtype=np.float32)
    score_valid = np.zeros((valid.X.shape[0],), dtype=np.float32)
    fit_result = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=score_train,
        rank_score_valid=score_valid,
        cfg=cfg,
    )

    empty = LongMatrixDataset(
        X=np.zeros((0, train.X.shape[1]), dtype=np.float32),
        y_rank=np.zeros((0,), dtype=np.int32),
        y_ev=np.zeros((0,), dtype=np.float32),
        group=np.zeros((0,), dtype=np.int32),
        sample_weight=np.zeros((0,), dtype=np.float32),
        index_map=np.zeros((0, 2), dtype=np.int64),
        feature_names=train.feature_names,
    )
    pred = predict_conservative_ev(
        models=fit_result,
        dataset=empty,
        rank_score=np.zeros((0,), dtype=np.float32),
        cfg=cfg,
    )

    assert pred.shape == (0,)


def test_predict_ev_quantiles_and_compute_conservative_ev() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    test = _dataset(rows=36, groups=4)
    cfg = StrategyMLConfig(calibrator_n_estimators=10, early_stopping_rounds=5, n_jobs=1)
    fit_result = fit_quantile_calibrators(train=train, valid=valid, cfg=cfg)
    score_test = np.linspace(-0.5, 0.5, test.X.shape[0], dtype=np.float32)

    quantiles = predict_ev_quantiles(models=fit_result, dataset=test, rank_score=score_test)
    ev = compute_conservative_ev(quantiles.q10, quantiles.q50, quantiles.q90, cfg)

    assert quantiles.q10.shape == (test.X.shape[0],)
    assert quantiles.q50.shape == (test.X.shape[0],)
    assert quantiles.q90.shape == (test.X.shape[0],)
    assert ev.shape == (test.X.shape[0],)


def test_fit_quantile_calibrators_uses_raw_y_ev(monkeypatch: MonkeyPatch) -> None:
    train = _dataset(rows=20, groups=4)
    valid = _dataset(rows=8, groups=2)
    cfg = StrategyMLConfig(
        calibrator_n_estimators=10,
        early_stopping_rounds=5,
        n_jobs=1,
        ensemble_seeds=[42],
    )
    seen: list[np.ndarray] = []

    def _fake_fit_one(
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_weight: np.ndarray,
        valid_x: np.ndarray,
        valid_y: np.ndarray,
        valid_weight: np.ndarray,
        cfg: StrategyMLConfig,
        alpha: float,
        seed: int | None = None,
    ) -> object:
        del train_x, train_weight, valid_x, valid_y, valid_weight, cfg, alpha, seed
        seen.append(np.asarray(train_y, dtype=np.float32).copy())
        return object()

    monkeypatch.setattr("src.domain.futures.strategy.calibrator._fit_one", _fake_fit_one)

    fit_quantile_calibrators(train=train, valid=valid, cfg=cfg)
    assert len(seen) == 3
    for y in seen:
        np.testing.assert_allclose(y, train.y_ev, rtol=0.0, atol=0.0)


def test_fit_quantile_calibrators_ensemble_seeds(monkeypatch: MonkeyPatch) -> None:
    train = _dataset(rows=20, groups=4)
    valid = _dataset(rows=8, groups=2)
    cfg = StrategyMLConfig(
        calibrator_n_estimators=10,
        early_stopping_rounds=5,
        n_jobs=1,
        ensemble_seeds=[101, 102],
    )
    seen_seeds: list[int | None] = []

    def _fake_fit_one(
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_weight: np.ndarray,
        valid_x: np.ndarray,
        valid_y: np.ndarray,
        valid_weight: np.ndarray,
        cfg: StrategyMLConfig,
        alpha: float,
        seed: int | None = None,
    ) -> object:
        del train_x, train_y, train_weight, valid_x, valid_y, valid_weight, cfg, alpha
        seen_seeds.append(seed)
        return object()

    monkeypatch.setattr("src.domain.futures.strategy.calibrator._fit_one", _fake_fit_one)

    fit_quantile_calibrators(train=train, valid=valid, cfg=cfg)
    # 2 seeds * 3 quantiles = 6 fits
    assert len(seen_seeds) == 6
    assert seen_seeds == [101, 101, 101, 102, 102, 102]


def test_calibrator_fit_uses_config_hyperparams(monkeypatch: MonkeyPatch) -> None:
    train = _dataset(rows=20, groups=4)
    valid = _dataset(rows=8, groups=2)
    captured: dict[str, typing.Any] = {}

    class _FakeRegressor:
        def __init__(self, **kwargs: float) -> None:
            captured.update(kwargs)

        def fit(self, *args: object, **kwargs: object) -> _FakeRegressor:
            del args, kwargs
            return self

        def predict(self, x: typing.Sized) -> np.ndarray:
            return np.zeros((len(x),), dtype=np.float64)

    monkeypatch.setattr("src.domain.futures.strategy.calibrator.lgb.LGBMRegressor", _FakeRegressor)
    cfg = StrategyMLConfig(
        calibrator_learning_rate=0.019,
        calibrator_feature_fraction=0.63,
        calibrator_bagging_fraction=0.62,
        calibrator_bagging_freq=4,
        calibrator_lambda_l2=3.0,
        calibrator_reg_alpha=0.8,
        max_depth=6,
        calibrator_max_depth_cap=4,
        ensemble_seeds=[42],
    )
    fit_quantile_calibrators(train=train, valid=valid, cfg=cfg)
    assert captured["learning_rate"] == 0.019
    assert captured["feature_fraction"] == 0.63
    assert captured["bagging_fraction"] == 0.62
    assert captured["bagging_freq"] == 4
    assert captured["lambda_l2"] == 3.0
    assert captured["reg_alpha"] == 0.8
    assert captured["max_depth"] == 4


def test_calibrator_fit_propagates_sample_weights(monkeypatch: MonkeyPatch) -> None:
    train = _dataset(rows=20, groups=4)
    valid = _dataset(rows=8, groups=2)
    train.sample_weight[:] = np.linspace(1.0, 2.0, train.sample_weight.shape[0], dtype=np.float32)
    valid.sample_weight[:] = np.linspace(2.0, 3.0, valid.sample_weight.shape[0], dtype=np.float32)
    seen: dict[str, typing.Any] = {}

    class _FakeRegressor:
        def __init__(self, **kwargs: float) -> None:
            del kwargs

        def fit(self, *args: object, **kwargs: object) -> _FakeRegressor:
            del args
            seen["train"] = np.asarray(kwargs.get("sample_weight"), dtype=np.float32).copy()
            ev_sw = kwargs.get("eval_sample_weight")
            assert isinstance(ev_sw, list)
            seen["valid"] = np.asarray(ev_sw[0], dtype=np.float32).copy()
            return self

        def predict(self, x: typing.Sized) -> np.ndarray:
            return np.zeros((len(x),), dtype=np.float64)

    monkeypatch.setattr("src.domain.futures.strategy.calibrator.lgb.LGBMRegressor", _FakeRegressor)
    cfg = StrategyMLConfig(
        calibrator_n_estimators=10,
        early_stopping_rounds=5,
        n_jobs=1,
        ensemble_seeds=[42],
    )
    fit_quantile_calibrators(train=train, valid=valid, cfg=cfg)
    np.testing.assert_allclose(seen["train"], train.sample_weight, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(seen["valid"], valid.sample_weight, rtol=0.0, atol=0.0)


def test_compute_conservative_ev_prob_x_magnitude_mode() -> None:
    cfg = StrategyMLConfig(ev_mode="prob_x_magnitude", alpha_clip_bps=100.0)
    q10 = np.array([-0.03, -0.02, -0.01], dtype=np.float32)
    q50 = np.array([-0.01, 0.0, 0.02], dtype=np.float32)
    q90 = np.array([0.01, 0.02, 0.03], dtype=np.float32)

    ev = compute_conservative_ev(q10=q10, q50=q50, q90=q90, cfg=cfg)

    assert ev.shape == (3,)
    assert ev.dtype == np.float32
    assert np.all(np.isfinite(ev))


def test_compute_conservative_ev_quantile_tail_blend_zero_matches_baseline() -> None:
    cfg = StrategyMLConfig(ev_mode="quantile", ev_tail_blend_weight=0.0, alpha_clip_bps=1_000.0)
    q10 = np.array([-0.03, -0.02, 0.00], dtype=np.float32)
    q50 = np.array([-0.01, 0.005, 0.01], dtype=np.float32)
    q90 = np.array([0.01, 0.03, 0.04], dtype=np.float32)
    ev = compute_conservative_ev(q10=q10, q50=q50, q90=q90, cfg=cfg)
    uncertainty = np.maximum(q90 - q10, np.float32(1e-8))
    downside = np.maximum(q50 - q10, np.float32(0.0))
    upside = np.maximum(q90 - q50, np.float32(0.0))
    med_unc = np.median(uncertainty)
    lam = np.float32(cfg.lambda_tail)
    lam_dynamic = np.clip(
        lam * (uncertainty / np.maximum(med_unc, np.float32(1e-8))),
        np.float32(0.0),
        lam * np.float32(2.0),
    )
    is_long = q50 >= np.float32(0.0)
    penalty_ratio = np.where(is_long, downside / uncertainty, upside / uncertainty)
    penalty_term = np.clip(lam_dynamic * penalty_ratio, np.float32(0.0), np.float32(0.99))
    expected = (q50 * (np.float32(1.0) - penalty_term)).astype(np.float32)
    np.testing.assert_allclose(ev, expected, rtol=0.0, atol=0.0)


def test_compute_conservative_ev_quantile_tail_blend_is_bounded() -> None:
    cfg = StrategyMLConfig(ev_mode="quantile", ev_tail_blend_weight=0.5, alpha_clip_bps=200.0)
    q10 = np.array([0.00, 0.001, 0.002], dtype=np.float32)
    q50 = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    q90 = np.array([0.01, 0.012, 0.014], dtype=np.float32)
    ev = compute_conservative_ev(q10=q10, q50=q50, q90=q90, cfg=cfg)
    assert np.all(ev <= q90 + np.float32(1e-8))
    assert np.all(ev >= np.float32(0.0))
    assert np.max(ev) <= np.float32(cfg.alpha_clip_bps / 10000.0) + np.float32(1e-8)


def test_calibrator_reproducibility() -> None:
    train = _dataset(rows=180, groups=20)
    valid = _dataset(rows=45, groups=5)
    cfg = StrategyMLConfig(
        calibrator_n_estimators=30,
        early_stopping_rounds=10,
        n_jobs=2,  # test with multiple threads
    )

    score_train = np.linspace(-1.0, 1.0, train.X.shape[0], dtype=np.float32)
    score_valid = np.linspace(-0.8, 0.8, valid.X.shape[0], dtype=np.float32)

    fit1 = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=score_train,
        rank_score_valid=score_valid,
        cfg=cfg,
    )
    ev1 = predict_conservative_ev(models=fit1, dataset=valid, rank_score=score_valid, cfg=cfg)

    fit2 = fit_quantile_calibrators(
        train=train,
        valid=valid,
        rank_score_train=score_train,
        rank_score_valid=score_valid,
        cfg=cfg,
    )
    ev2 = predict_conservative_ev(models=fit2, dataset=valid, rank_score=score_valid, cfg=cfg)

    assert np.array_equal(ev1, ev2)

