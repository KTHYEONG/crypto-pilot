from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.strategy.common.alignment import AlignedMarketData


def _minimal_aligned(n_bars: int = 200, n_sym: int = 5) -> AlignedMarketData:
    import pandas as pd

    base_dt = pd.date_range("2026-01-01", periods=n_bars, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    ones = np.ones((n_bars, n_sym), dtype=np.float64)
    mask = np.ones((n_bars, n_sym), dtype=bool)
    return AlignedMarketData(
        datetimes=base_dt,
        symbols=("BTCUSDT", *(f"SYM{i}" for i in range(n_sym - 1))),
        open_2d=ones * 100.0,
        high_2d=ones * 105.0,
        low_2d=ones * 95.0,
        close_2d=ones * 100.0,
        volume_2d=ones * 1000.0,
        funding_2d=np.zeros((n_bars, n_sym), dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=mask,
        kill_mask=~mask,
    )


def _minimal_labeled_events() -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    return pd.DataFrame({
        "datetime": dt,
        "symbol": "SYM0",
        "event_id": list(range(10)),
        "entry_idx": range(10),
        "exit_idx": range(1, 11),
        "side": [1] * 10,
        "expected_gross_bps": [50.0] * 10,
        "expected_net_bps": [40.0] * 10,
        "expected_holding_bars": [3] * 10,
        "quality_weight": [1.0] * 10,
        "strategy_id": ["test_strat"] * 10,
        "native_tf": ["1h"] * 10,
    })


def test_l1_result_cache_hit_on_second_call(tmp_path, mocker):
    cache_dir = str(tmp_path / "l1_cache")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"L1_RESULT_CACHE_ENABLED": True, "L1_RESULT_CACHE_DIR": cache_dir},
        clear=False,
    )
    lbl = _minimal_labeled_events()
    aligned = _minimal_aligned(n_bars=200, n_sym=5)

    from src.domain.futures.strategy.walk_forward import WFFold

    outer_folds = (
        WFFold(fit_start=0, fit_end=50, cal_start=40, cal_end=50, oos_start=50, oos_end=80),
    )

    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    cfg = CandidateStrategyConfig()

    mocker.patch(
        "src.domain.futures.strategy.event_grid_contracts.normalize_native_l1_events",
        return_value=mocker.MagicMock(
            eligible_events=lbl,
            audit=None,
        ),
    )
    _fake_l1 = Layer1Result(
        signals_per_fold=(),
        oos_stacked={"SYM0": [0.1]},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=True,
        n_valid=1,
        n_total=1,
    )
    l1_nested_spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf",
        return_value=_fake_l1,
    )

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_per_tf_l1

    result1 = run_per_tf_l1(
        tf="1h",
        labeled_events=lbl,
        aligned=aligned,
        outer_folds=outer_folds,
        cfg=cfg,
        seed=42,
    )
    result2 = run_per_tf_l1(
        tf="1h",
        labeled_events=lbl,
        aligned=aligned,
        outer_folds=outer_folds,
        cfg=cfg,
        seed=42,
    )
    assert l1_nested_spy.call_count == 1
    assert result1.tf == "1h"
    assert result2.tf == "1h"


def test_l1_result_cache_miss_on_cfg_change(tmp_path, mocker):
    cache_dir = str(tmp_path / "l1_cache")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"L1_RESULT_CACHE_ENABLED": True, "L1_RESULT_CACHE_DIR": cache_dir},
        clear=False,
    )
    lbl = _minimal_labeled_events()
    aligned = _minimal_aligned(n_bars=200, n_sym=5)

    from src.domain.futures.strategy.walk_forward import WFFold

    outer_folds = (
        WFFold(fit_start=0, fit_end=50, cal_start=40, cal_end=50, oos_start=50, oos_end=80),
    )

    from src.domain.futures.strategy.config import CandidateStrategyConfig

    mocker.patch(
        "src.domain.futures.strategy.event_grid_contracts.normalize_native_l1_events",
        return_value=mocker.MagicMock(
            eligible_events=lbl,
            audit=None,
        ),
    )
    l1_nested_spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf",
        return_value=mocker.MagicMock(
            signals_per_fold=(),
            oos_stacked={"SYM0": [0.1]},
            pooled_ic=0.0,
            pooled_tstat=0.0,
            breadth=0.0,
            valid_coverage=0.0,
            fold_pass_ratio=0.0,
            gate_passed=True,
            n_valid=1,
            n_total=1,
        ),
    )

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_per_tf_l1

    cfg_a = CandidateStrategyConfig()
    cfg_b = CandidateStrategyConfig()
    object.__setattr__(cfg_b, "label_horizon_bars", 24)

    run_per_tf_l1(
        tf="1h", labeled_events=lbl, aligned=aligned, outer_folds=outer_folds, cfg=cfg_a, seed=42
    )
    run_per_tf_l1(
        tf="1h", labeled_events=lbl, aligned=aligned, outer_folds=outer_folds, cfg=cfg_b, seed=42
    )
    assert l1_nested_spy.call_count == 2
