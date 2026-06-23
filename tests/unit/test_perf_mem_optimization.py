# tests/unit/test_perf_mem_optimization.py

import os
from unittest.mock import MagicMock

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _build_tradeable_mask_vectorized,
    _scatter_signals_jit,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    resolve_safe_nested_workers,
)


def test_scenario_5_adaptive_worker_cap() -> None:
    """Scenario 5: Adaptive worker cap stage-dependent scaling verification."""
    from unittest.mock import MagicMock, patch

    # Mock CPU cores count to 8, mock available memory to 16GB
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 8

    mock_mem = MagicMock()
    mock_mem.available = 16 * 1024**3

    try:
        frame_bytes = 1024 * 1024  # 1MB (negligible)

        with patch("psutil.virtual_memory", return_value=mock_mem):
            workers_ev = resolve_safe_nested_workers(16, frame_bytes, stage="evidence")
            workers_out = resolve_safe_nested_workers(16, frame_bytes, stage="outer")
            workers_opt = resolve_safe_nested_workers(16, frame_bytes, stage="l2_optuna")

        # CPU cap = max(1, 8*0.75) = 6.
        # evidence: compact_result=False -> stage_cap=3. max_workers=min(6,3)=3.
        # outer:    stage_cap=3. max_workers=min(6,3)=3.
        # l2_optuna: stage_cap=4. max_workers=min(6,4)=4.
        # Memory: safe_mem_gb=11.2, estimated_proc_gb=0.9005 -> mem_limit=12 (no restriction)
        assert workers_ev == 3
        assert workers_out == 3
        assert workers_opt == 4
    finally:
        os.cpu_count = original_cpu_count


def test_scenario_4_tradeable_mask_vectorized() -> None:
    """Scenario 4: Verify vectorized tradeable mask matches loop-based implementation."""
    t_max = 100
    n_sym = 5
    
    aligned = MagicMock(spec=AlignedMarketData)
    aligned.active_mask = np.random.choice([True, False], size=(t_max, n_sym))
    aligned.warm_mask = np.random.choice([True, False], size=(t_max, n_sym))
    aligned.execution_eligibility_mask = np.random.choice([True, False], size=(t_max, n_sym))
    aligned.strategy_readiness_mask = np.random.choice([True, False], size=(t_max, n_sym))
    aligned.promotion_active_mask = np.random.choice([True, False], size=(t_max, n_sym))
    aligned.entry_block_mask = np.random.choice([True, False], size=(t_max, n_sym))
    aligned.kill_mask = np.random.choice([True, False], size=(t_max, n_sym))
    
    # Loop-based equivalent logic for bit-exact comparison
    loop_result = np.zeros((t_max, n_sym), dtype=np.bool_)
    for t in range(t_max):
        active = aligned.active_mask[t]
        warm = aligned.warm_mask[t]
        elig = aligned.execution_eligibility_mask[t]
        ready = aligned.strategy_readiness_mask[t]
        prom = aligned.promotion_active_mask[t]
        eb = aligned.entry_block_mask[t]
        kill = aligned.kill_mask[t]
        loop_result[t] = active & warm & elig & ready & prom & (~eb) & (~kill)
        
    vec_result = _build_tradeable_mask_vectorized(aligned, t_max, n_sym)
    assert np.array_equal(vec_result, loop_result)


def test_scenario_6_capacity_clip_vectorized() -> None:
    """Scenario 6: Verify capacity clip numpy vectorization behaves identically to the loop."""
    w = np.array([0.1, -0.05, 0.3, -0.4], dtype=np.float64)
    cap_row = np.array([1000.0, 50.0, 0.0, 500.0], dtype=np.float64)
    portfolio_nav = 1000.0
    min_order_usdt = 100.0
    
    # 1. Loop-based reference
    w_loop = w.copy()
    for i in range(len(w_loop)):
        intended = abs(w_loop[i]) * portfolio_nav
        if intended < min_order_usdt:
            w_loop[i] = 0.0
            continue
        cap = cap_row[i]
        if cap > 0.0:
            max_w = cap / max(portfolio_nav, 1.0)
            if abs(w_loop[i]) > max_w:
                w_loop[i] = np.sign(w_loop[i]) * max_w
                
    # 2. Vectorized logic under test
    w_vec = w.copy()
    intended_vec = np.abs(w_vec) * portfolio_nav
    w_vec[intended_vec < min_order_usdt] = 0.0
    
    cap_positive = cap_row > 0.0
    if np.any(cap_positive):
        max_w_vec = np.where(cap_positive, cap_row / max(portfolio_nav, 1.0), np.inf)
        over = np.abs(w_vec) > max_w_vec
        w_vec[over] = np.sign(w_vec[over]) * max_w_vec[over]
        
    assert np.array_equal(w_vec, w_loop)


def test_scenario_3_signal_scatter_jit() -> None:
    """Scenario 3: Verify Numba JIT scatter signal output matches old loop-based logic."""
    t_max = 500
    n_sleeve = 3
    
    expected_gross_bps_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    expected_net_bps_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    holding_bars_2d = np.ones((t_max, n_sleeve), dtype=np.float64)
    side_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    quality_weight_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    event_strength_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    signal_mask_2d = np.zeros((t_max, n_sleeve), dtype=np.bool_)
    
    # 3 mock events
    decision_idxs = np.array([10, 20, 10], dtype=np.int64)
    holding_bars_arr = np.array([5, 10, 8], dtype=np.int64)
    sleeve_js = np.array([0, 1, 0], dtype=np.int64)
    gross_vals = np.array([100.0, 150.0, 120.0], dtype=np.float64)
    net_vals = np.array([80.0, 120.0, 100.0], dtype=np.float64)
    side_vals = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    qw_vals = np.array([0.8, 0.9, 0.85], dtype=np.float64)
    strengths = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    
    # Run loop reference
    expected_gross_loop = expected_gross_bps_2d.copy()
    expected_net_loop = expected_net_bps_2d.copy()
    holding_bars_loop = holding_bars_2d.copy()
    side_loop = side_2d.copy()
    quality_weight_loop = quality_weight_2d.copy()
    event_strength_loop = event_strength_2d.copy()
    signal_mask_loop = signal_mask_2d.copy()
    
    for e in range(len(decision_idxs)):
        sleeve_j = sleeve_js[e]
        start = decision_idxs[e] + 1
        end = min(t_max, start + holding_bars_arr[e])
        if start >= end:
            continue
        g_val = gross_vals[e]
        n_val = net_vals[e]
        h_bars = float(holding_bars_arr[e])
        s_val = side_vals[e]
        q_val = qw_vals[e]
        str_val = strengths[e]
        
        for t in range(start, end):
            if not signal_mask_loop[t, sleeve_j]:
                signal_mask_loop[t, sleeve_j] = True
                expected_gross_loop[t, sleeve_j] = g_val
                expected_net_loop[t, sleeve_j] = n_val
                holding_bars_loop[t, sleeve_j] = h_bars
                side_loop[t, sleeve_j] = s_val
                quality_weight_loop[t, sleeve_j] = q_val
                event_strength_loop[t, sleeve_j] = str_val
            elif str_val > event_strength_loop[t, sleeve_j]:
                expected_gross_loop[t, sleeve_j] = g_val
                expected_net_loop[t, sleeve_j] = n_val
                holding_bars_loop[t, sleeve_j] = h_bars
                side_loop[t, sleeve_j] = s_val
                quality_weight_loop[t, sleeve_j] = q_val
                event_strength_loop[t, sleeve_j] = str_val
                
    # Run JIT JIT JIT JIT JIT
    _scatter_signals_jit(
        decision_idxs,
        holding_bars_arr,
        sleeve_js,
        gross_vals,
        net_vals,
        side_vals,
        qw_vals,
        strengths,
        expected_gross_bps_2d,
        expected_net_bps_2d,
        holding_bars_2d,
        side_2d,
        quality_weight_2d,
        event_strength_2d,
        signal_mask_2d,
        t_max,
    )
    
    assert np.array_equal(expected_gross_bps_2d, expected_gross_loop)
    assert np.array_equal(expected_net_bps_2d, expected_net_loop)
    assert np.array_equal(holding_bars_2d, holding_bars_loop)
    assert np.array_equal(side_2d, side_loop)
    assert np.array_equal(quality_weight_2d, quality_weight_loop)
    assert np.array_equal(event_strength_2d, event_strength_loop)
    assert np.array_equal(signal_mask_2d, signal_mask_loop)


def test_l2_optuna_low_memory_fallback() -> None:
    """Verify that under low memory conditions, the batch size falls back to 1 (sequential)."""
    from unittest.mock import MagicMock, patch
    
    # Mock psutil.virtual_memory().available to return 1.5 GB
    mock_mem = MagicMock()
    mock_mem.available = 1.5 * 1024 * 1024 * 1024  # 1.5 GB
    
    with patch("psutil.virtual_memory", return_value=mock_mem), \
         patch("src.execution.opt_main_futures.get_or_create_study") as mock_get_study, \
         patch("src.execution.opt_main_futures.setup_optuna_storage", return_value=(None, None)), \
         patch("src.domain.futures.strategy.tiered_workflow.selection._signal_batch_fingerprint", return_value="mocked_fingerprint"), \
         patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache"):
        
        mock_study = MagicMock()
        mock_get_study.return_value = mock_study
        
        from src.execution.opt_main_futures import _run_tiered_l2_study
        
        # Mock inputs
        signal_batch = MagicMock()
        signal_batch.events = []
        aligned = MagicMock()
        cfg = MagicMock()
        window = MagicMock()
        caps = MagicMock()
        
        with patch("src.execution.opt_main_futures.OPT_FUTURES_CONFIG", {"L2_OPTUNA_BATCH_SIZE": "4"}):
            _run_tiered_l2_study(
                signal_batch=signal_batch,
                aligned=aligned,
                cfg=cfg,
                window=window,
                caps=caps,
                tf="1h",
                n_trials=10,
                seed=42,
                l2_sim_cache=MagicMock(),
            )
            
            # Since available memory is 1.5 GB (< 3.0 GB), batch_size must fall back to 1.
            # Thus, study.optimize should have been called with n_jobs=1.
            mock_study.optimize.assert_called_once()
            _, kwargs = mock_study.optimize.call_args
            assert kwargs.get("n_jobs") == 1

