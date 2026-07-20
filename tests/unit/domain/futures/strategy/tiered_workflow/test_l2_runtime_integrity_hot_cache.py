from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.crisis_policy import (
    CrisisWindowMetrics,
    evaluate_crisis_survival,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import _should_load_cache
from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    _recompute_deployed_cagr,
    assert_selection_replay_parity,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import apply_deployment


def _evaluation(*, deployed: tuple[float, ...], cagr: float) -> SimpleNamespace:
    return SimpleNamespace(
        returns_hybrid=tuple(v / 2.0 for v in deployed),
        boosted_returns_hybrid=tuple(v / 2.0 for v in deployed),
        deployed_returns_hybrid=deployed,
        cagr_hybrid=cagr,
        mdd_hybrid=0.01,
        fold_pass_ratio=0.75,
        trade_count=100,
        deploy_leverage=2.0,
        master_tf="8h",
    )


def test_should_load_cache_uses_current_rss_instead_of_hwm() -> None:
    result = _should_load_cache(
        0.25,
        threshold_mb=11_500,
        expansion_ratio=15.0,
        current_rss_mb=5_800.0,
    )

    assert result is True


@pytest.mark.parametrize(
    ("cache_mb", "expected"),
    [(64.0, True), (64.01, False)],
)
def test_should_load_cache_unknown_rss_allows_only_small_cache(
    cache_mb: float,
    expected: bool,
) -> None:
    result = _should_load_cache(
        cache_mb,
        current_rss_mb=-1.0,
        unknown_rss_cache_limit_mb=64.0,
    )

    assert result is expected


def test_parity_selfcheck_blocks_deployed_return_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    deployed = (0.01, 0.01, -0.005, 0.002)
    replay = _evaluation(deployed=deployed, cagr=0.99)
    final = _evaluation(deployed=deployed, cagr=0.99)

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        gate=True,
    )

    assert result is False
    assert "DECOUPLED" in caplog.text


def test_parity_deployed_returns_do_not_apply_leverage_twice() -> None:
    unit = np.asarray([0.001, 0.002, -0.001, 0.001], dtype=np.float64)
    deployed_result = apply_deployment(rets=unit, leverage=2.0, bars_per_year=1095.0)
    deployed = tuple(float(v) for v in deployed_result.scaled_rets)
    replay = _evaluation(deployed=deployed, cagr=deployed_result.cagr)
    final = _evaluation(deployed=deployed, cagr=deployed_result.cagr)

    result = assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        gate=True,
    )

    assert result is True


def test_crisis_window_status_matches_aggregate_verdict() -> None:
    window = CrisisWindowMetrics(
        label="luna_ftx_2022_collapse",
        status="stress_tested_fail",
        detail="placeholder",
        symbol_count=45,
        observation_days=365,
        bar_count=1095,
        event_count=500,
        trade_count=113,
        mdd=0.1969,
        cagr=-0.0446,
        cvar_95=0.0192,
    )

    assessment = evaluate_crisis_survival(
        (window,),
        max_mdd_abs=0.21,
        min_cagr=-0.05,
        max_cvar_95=0.06,
        min_symbols=10,
        min_observation_days=30,
        min_trades=10,
        min_usable_windows=1,
    )

    assert assessment.status == "stress_tested_pass"
    assert assessment.window_results[0].status == "stress_tested_pass"


def test_recompute_deployed_cagr_uses_deployed_field() -> None:
    deployed = (0.001, 0.002, -0.001, 0.001)
    cagr_val = 0.15
    obj = SimpleNamespace(
        deployed_returns_hybrid=deployed,
        returns_hybrid=tuple(v / 2.0 for v in deployed),
        cagr_hybrid=cagr_val,
        deploy_leverage=2.0,
        master_tf="8h",
    )
    result = _recompute_deployed_cagr(obj)
    assert result is not None
    assert result > 0.0


def test_recompute_deployed_cagr_fallback_legacy() -> None:
    unit_rets = (0.001, 0.002, -0.001, 0.001)
    obj = SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=unit_rets,
        cagr_hybrid=0.08,
        deploy_leverage=2.0,
        master_tf="8h",
    )
    result = _recompute_deployed_cagr(obj)
    assert result is not None
    assert result > 0.0


def test_crisis_window_replaces_placeholder_status() -> None:
    window = CrisisWindowMetrics(
        label="sample_crisis",
        status="pending",
        detail="placeholder",
        symbol_count=50,
        observation_days=400,
        bar_count=1200,
        event_count=600,
        trade_count=200,
        mdd=0.25,
        cagr=-0.03,
        cvar_95=0.04,
    )
    assessment = evaluate_crisis_survival(
        (window,),
        max_mdd_abs=0.30,
        min_cagr=-0.10,
        max_cvar_95=0.05,
        min_symbols=10,
        min_observation_days=30,
        min_trades=10,
        min_usable_windows=1,
    )
    assert assessment.window_results[0].status in (
        "stress_tested_pass", "stress_tested_fail", "stress_data_invalid",
    )
    assert assessment.window_results[0].status != "pending"


def test_crisis_window_aggregate_detail_consistency() -> None:
    windows = (
        CrisisWindowMetrics(
            label="w1_pass",
            status="pending",
            detail="",
            symbol_count=50,
            observation_days=400,
            bar_count=1200,
            event_count=600,
            trade_count=200,
            mdd=0.15,
            cagr=0.02,
            cvar_95=0.03,
        ),
        CrisisWindowMetrics(
            label="w2_fail",
            status="pending",
            detail="",
            symbol_count=50,
            observation_days=400,
            bar_count=1200,
            event_count=600,
            trade_count=200,
            mdd=0.35,
            cagr=-0.08,
            cvar_95=0.07,
        ),
    )
    assessment = evaluate_crisis_survival(
        windows,
        max_mdd_abs=0.30,
        min_cagr=-0.10,
        max_cvar_95=0.06,
        min_symbols=10,
        min_observation_days=30,
        min_trades=10,
        min_usable_windows=1,
    )
    assert assessment.status == "stress_tested_fail"
    w1 = assessment.window_results[0]
    w2 = assessment.window_results[1]
    assert w1.status == "stress_tested_pass"
    assert w2.status == "stress_tested_fail"


def test_recompute_deployed_cagr_fallback_no_leverage() -> None:
    obj = SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=(0.001, 0.002, -0.001, 0.001),
        cagr_hybrid=0.08,
        deploy_leverage=None,
        master_tf="8h",
    )
    result = _recompute_deployed_cagr(obj)
    assert result is not None
    assert result > 0.0


def test_recompute_deployed_cagr_insufficient_rets() -> None:
    obj = SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=(0.001,),
        cagr_hybrid=0.01,
        deploy_leverage=2.0,
        master_tf="8h",
    )
    assert _recompute_deployed_cagr(obj) is None


def test_recompute_deployed_cagr_no_bars_per_year() -> None:
    obj = SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=(0.001, 0.002, -0.001, 0.001),
        cagr_hybrid=0.08,
        deploy_leverage=2.0,
        master_tf="",
    )
    assert _recompute_deployed_cagr(obj) is None


def test_recompute_deployed_cagr_edge_empty_array() -> None:
    assert _recompute_deployed_cagr(SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=(),
        cagr_hybrid=0.0,
        deploy_leverage=None,
        master_tf="8h",
    )) is None


def test_recompute_deployed_cagr_edge_single_element() -> None:
    from src.domain.futures.strategy.tiered_workflow.replay_parity import _cagr as _rp_cagr
    assert _rp_cagr([0.001], bars_per_year=1095.0) == 0.0


def test_recompute_deployed_cagr_edge_all_nan() -> None:
    result = _recompute_deployed_cagr(SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=(float("nan"), float("nan")),
        cagr_hybrid=0.0,
        deploy_leverage=None,
        master_tf="8h",
    ))
    assert result == 0.0


def test_recompute_deployed_cagr_exception_path(mocker: pytest.MockFixture) -> None:
    obj = SimpleNamespace(
        deployed_returns_hybrid=(),
        returns_hybrid=(0.001, 0.002, -0.001, 0.001),
        cagr_hybrid=0.08,
        deploy_leverage=2.0,
        master_tf="8h",
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.risk_deployment.apply_deployment",
        side_effect=ValueError("simulated failure"),
    )
    assert _recompute_deployed_cagr(obj) is None
