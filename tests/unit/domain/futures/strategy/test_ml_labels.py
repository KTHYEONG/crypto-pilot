from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.labels import build_label_panel


def _aligned_for_labels() -> AlignedMarketData:
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(4)],
        dtype="datetime64[ns]",
    )
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((4, 2), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((4, 2), dtype=np.float64),
        active_mask=np.ones((4, 2), dtype=bool),
        warm_mask=np.ones((4, 2), dtype=bool),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )


def test_build_label_panel_uses_t_plus_1_open_close_alignment() -> None:
    """Verify B2 beta-residualized labels use t+1 execution alignment.

    With only 4 bars, trailing beta defaults to 1.0 (insufficient history).
    Expected values: gross_long - beta * market_fwd_ret (equal-weighted).
    t=0: gross=log(121/110), mfr=mean(log(121/110), log(221/210)), beta=1.0
    t=2: gross=log(143/130), mfr=mean(log(143/130), log(243/230)), beta=1.0
    """
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    # market forward return at t=0: equal-weighted log return from open[1] to close[1]
    mfr0 = float(np.nanmean(np.log(close_2d[1] / open_2d[1])))
    mfr2 = float(np.nanmean(np.log(close_2d[3] / open_2d[3])))
    expected_t0 = np.log(121.0 / 110.0) - 1.0 * mfr0  # beta=1.0 (insufficient history)
    expected_t2 = np.log(143.0 / 130.0) - 1.0 * mfr2

    panel = build_label_panel(
        _aligned_for_labels(),
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    np.testing.assert_allclose(panel.long_net_ret[0, 0], expected_t0, rtol=1e-5)
    np.testing.assert_allclose(panel.long_net_ret[2, 0], expected_t2, rtol=1e-5)
    assert np.isnan(panel.long_net_ret[3, 0])


def test_build_label_panel_enforces_eligibility_mask_on_outputs() -> None:
    aligned = _aligned_for_labels()
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=np.ones((4, 1), dtype=bool),
        warm_mask=np.array([[True, True], [False, False], [True, True], [True, True]]),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )
    panel = build_label_panel(aligned, StrategyMLConfig(min_group_size=2))
    assert not panel.eligible_mask[1, 0]
    assert panel.sample_weight[1, 0] == 0.0


def test_build_label_panel_cs_demean_zeros_cross_sectional_mean() -> None:
    """CS-demean enforces per-timestep CS mean ≈ 0 when funding differs across symbols.

    Asymmetric funding (sym1=0.05, sym0=0.0) creates a non-zero CS mean before demean.
    Without demean: mean(long_net[t]) ≈ -funding[1]/2 = -0.025 ≠ 0.
    After demean: mean(long_net[t, eligible]) must be ≈ 0 for every valid timestep.
    """
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(4)],
        dtype="datetime64[ns]",
    )
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    funding_2d = np.array([[0.0, 0.05], [0.0, 0.05], [0.0, 0.05], [0.0, 0.05]])

    aligned = AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((4, 2), 1000.0, dtype=np.float64),
        funding_2d=funding_2d,
        active_mask=np.ones((4, 2), dtype=bool),
        warm_mask=np.ones((4, 2), dtype=bool),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )

    panel = build_label_panel(
        aligned,
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )

    valid_t_found = False
    for t in range(panel.long_net_ret.shape[0]):
        mask_t = np.isfinite(panel.long_net_ret[t]) & panel.eligible_mask[t]
        if int(mask_t.sum()) < 2:
            continue
        valid_t_found = True
        cs_mean = float(np.mean(panel.long_net_ret[t, mask_t]))
        np.testing.assert_allclose(cs_mean, 0.0, atol=1e-6)
    assert valid_t_found, "no valid timestep with >=2 eligible symbols found"


def test_build_label_panel_exec_net_ret_is_pre_cs_demean() -> None:
    """Track 2: exec_net_ret must preserve pre-CS-demean absolute values.

    When funding is asymmetric across symbols, signed_net_ret (CS-demeaned) has
    per-timestep mean ≈ 0. exec_net_ret must retain the non-zero pre-demean mean,
    confirming it is captured before CS-demean is applied.
    """
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(4)],
        dtype="datetime64[ns]",
    )
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    # Asymmetric funding: sym1 gets 0.05 per bar, creating non-zero CS mean before demean
    funding_2d = np.array([[0.0, 0.05], [0.0, 0.05], [0.0, 0.05], [0.0, 0.05]])

    aligned = AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((4, 2), 1000.0, dtype=np.float64),
        funding_2d=funding_2d,
        active_mask=np.ones((4, 2), dtype=bool),
        warm_mask=np.ones((4, 2), dtype=bool),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )

    panel = build_label_panel(
        aligned,
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )

    # exec_net_ret: at least one valid timestep must have |CS-mean| > tolerance
    # (pre-CS-demean, so asymmetric funding creates non-zero mean)
    exec_non_zero_mean_found = False
    for t in range(panel.exec_net_ret.shape[0]):
        mask_t = np.isfinite(panel.exec_net_ret[t]) & panel.eligible_mask[t]
        if int(mask_t.sum()) < 2:
            continue
        cs_mean_exec = float(np.mean(panel.exec_net_ret[t, mask_t]))
        if abs(cs_mean_exec) > 1e-6:
            exec_non_zero_mean_found = True
            break
    assert exec_non_zero_mean_found, (
        "exec_net_ret should retain pre-CS-demean absolute values: "
        "no timestep found with |CS-mean| > 1e-6"
    )

    # signed_net_ret: CS-mean must be ≈ 0 for every valid timestep (CS-demeaned)
    for t in range(panel.signed_net_ret.shape[0]):
        mask_t = np.isfinite(panel.signed_net_ret[t]) & panel.eligible_mask[t]
        if int(mask_t.sum()) < 2:
            continue
        cs_mean_signed = float(np.mean(panel.signed_net_ret[t, mask_t]))
        np.testing.assert_allclose(
            cs_mean_signed,
            0.0,
            atol=1e-6,
            err_msg=f"signed_net_ret CS-mean at t={t} should be ≈ 0 after CS-demean",
        )


def test_build_label_panel_exec_net_ret_differs_from_signed_net_ret() -> None:
    """Track 2: exec_net_ret and signed_net_ret must differ when funding is asymmetric."""
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(4)],
        dtype="datetime64[ns]",
    )
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    funding_2d = np.array([[0.0, 0.05], [0.0, 0.05], [0.0, 0.05], [0.0, 0.05]])

    aligned = AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((4, 2), 1000.0, dtype=np.float64),
        funding_2d=funding_2d,
        active_mask=np.ones((4, 2), dtype=bool),
        warm_mask=np.ones((4, 2), dtype=bool),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )

    panel = build_label_panel(
        aligned,
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )

    # With asymmetric funding, CS-demean shifts values — exec_net_ret != signed_net_ret
    valid_mask = np.isfinite(panel.exec_net_ret) & panel.eligible_mask
    assert np.any(valid_mask), "no valid cells found"
    assert not np.allclose(
        panel.exec_net_ret[valid_mask],
        panel.signed_net_ret[valid_mask],
        atol=1e-7,
    ), "exec_net_ret must differ from signed_net_ret when CS-demean has non-zero effect"


def test_build_label_panel_applies_ev_scaled_sample_weight() -> None:
    aligned = _aligned_for_labels()
    panel = build_label_panel(
        aligned,
        StrategyMLConfig(
            label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2,
            sample_weight_time_decay_halflife_bars=None,  # isolate EV-scaling, no time-decay
        ),
    )
    liq_weight = np.clip(np.log1p(np.maximum(aligned.volume_2d, 0.0)), 0.25, 2.0).astype(np.float32)
    expected = np.where(
        panel.eligible_mask,
        liq_weight * (1.0 + 2.0 * np.abs(panel.signed_net_ret)),
        0.0,
    ).astype(np.float32)
    np.testing.assert_allclose(panel.sample_weight, expected, rtol=1e-6, atol=1e-8)


def test_build_label_panel_exposes_explicit_target_contract_metadata() -> None:
    panel = build_label_panel(
        _aligned_for_labels(),
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    assert panel.rank_target is not None
    assert panel.magnitude_target is not None
    # cost_clearance_target 3필드는 LabelDiagnostics로 격리됨 — LabelPanel에 없음
    assert not hasattr(panel, "cost_clearance_target")
    assert panel.metadata["rank_target_key"] == "signed_net_ret"
    assert panel.metadata["magnitude_target_key"] == "exec_net_ret"
    # LabelDiagnostics 요약은 metadata에 보존됨
    assert "_label_diagnostics_summary" in panel.metadata


def test_build_label_panel_exposes_forward_gross_rank_contracts() -> None:
    panel = build_label_panel(
        _aligned_for_labels(),
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    assert panel.forward_gross_ret is not None
    assert panel.forward_gross_rank_target is not None
    assert panel.forward_gross_relevance is not None
    assert panel.forward_gross_ret.shape == panel.signed_net_ret.shape
    assert panel.forward_gross_rank_target.shape == panel.signed_net_ret.shape
    assert panel.forward_gross_relevance.shape == panel.relevance.shape


# ---------------------------------------------------------------------------
# calibrator_target toggle tests
# ---------------------------------------------------------------------------


def _aligned_multi_symbol(n_symbols: int = 8, t_len: int = 20) -> AlignedMarketData:
    """Build multi-symbol AlignedMarketData with realistic cross-sectional dispersion."""
    rng = np.random.default_rng(42)
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(t_len)],
        dtype="datetime64[ns]",
    )
    # Prices spread across different levels with independent noise per symbol
    base_prices = np.linspace(100.0, 800.0, n_symbols)
    open_2d = np.outer(np.ones(t_len), base_prices) * (
        1.0 + rng.normal(scale=0.005, size=(t_len, n_symbols))
    )
    close_2d = open_2d * (1.0 + rng.normal(scale=0.01, size=(t_len, n_symbols)))
    # Ensure positive prices
    open_2d = np.abs(open_2d) + 0.01
    close_2d = np.abs(close_2d) + 0.01
    # Asymmetric funding across symbols to create non-trivial residualization
    funding_row = rng.uniform(-0.001, 0.001, size=n_symbols)
    funding_2d = np.tile(funding_row, (t_len, 1))

    return AlignedMarketData(
        datetimes=dt,
        symbols=tuple(f"SYM{i}USDT" for i in range(n_symbols)),
        open_2d=open_2d,
        high_2d=close_2d + 0.5,
        low_2d=close_2d - 0.5,
        close_2d=close_2d,
        volume_2d=np.full((t_len, n_symbols), 5000.0, dtype=np.float64),
        funding_2d=funding_2d,
        active_mask=np.ones((t_len, n_symbols), dtype=bool),
        warm_mask=np.ones((t_len, n_symbols), dtype=bool),
        entry_block_mask=np.zeros((t_len, n_symbols), dtype=bool),
        kill_mask=np.zeros((t_len, n_symbols), dtype=bool),
    )


def test_build_label_panel_gross_target_differs_from_beta_residualized() -> None:
    """calibrator_target='gross' must produce exec_net_ret with larger variance.

    The gross target retains the market-beta component, so its cross-sectional
    dispersion should be >= the beta-residualized target which removes that component.
    """
    # Arrange
    aligned = _aligned_multi_symbol(n_symbols=8, t_len=20)
    cfg_resid = StrategyMLConfig(
        label_horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_group_size=2,
        calibrator_target="beta_residualized",
    )
    cfg_gross = StrategyMLConfig(
        label_horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_group_size=2,
        calibrator_target="gross",
    )

    # Act
    panel_resid = build_label_panel(aligned, cfg_resid)
    panel_gross = build_label_panel(aligned, cfg_gross)

    # Assert — exec_net_ret arrays must differ
    valid_resid = np.isfinite(panel_resid.exec_net_ret) & panel_resid.eligible_mask
    valid_gross = np.isfinite(panel_gross.exec_net_ret) & panel_gross.eligible_mask
    assert np.any(valid_resid), "no valid cells in beta_residualized exec_net_ret"
    assert np.any(valid_gross), "no valid cells in gross exec_net_ret"
    # The two targets must not be identical
    common_mask = valid_resid & valid_gross
    assert np.any(common_mask), "no common valid cells to compare"
    assert not np.allclose(
        panel_resid.exec_net_ret[common_mask],
        panel_gross.exec_net_ret[common_mask],
        atol=1e-7,
    ), "gross and beta_residualized exec_net_ret must differ"
    # Gross target variance >= residualized (market component retained)
    var_gross = float(np.nanvar(panel_gross.exec_net_ret[common_mask]))
    var_resid = float(np.nanvar(panel_resid.exec_net_ret[common_mask]))
    assert var_gross >= var_resid, (
        f"gross var ({var_gross:.6f}) should be >= resid var ({var_resid:.6f})"
    )


def test_build_label_panel_residual_target_default_unchanged() -> None:
    """Default calibrator_target='beta_residualized' equals explicit config."""
    # Arrange — default config (no explicit calibrator_target); default is beta_residualized
    aligned = _aligned_multi_symbol(n_symbols=6, t_len=15)
    cfg_default = StrategyMLConfig(
        label_horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_group_size=2,
    )
    cfg_explicit = StrategyMLConfig(
        label_horizon_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_group_size=2,
        calibrator_target="beta_residualized",
    )

    # Act
    panel_default = build_label_panel(aligned, cfg_default)
    panel_explicit = build_label_panel(aligned, cfg_explicit)

    # Assert — both produce identical exec_net_ret
    np.testing.assert_array_equal(
        panel_default.exec_net_ret,
        panel_explicit.exec_net_ret,
    )


def test_strategy_ml_config_rejects_invalid_calibrator_target() -> None:
    """StrategyMLConfig must raise ValueError when calibrator_target is not a valid literal."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="calibrator_target"):
        StrategyMLConfig(calibrator_target="invalid")  # type: ignore[arg-type]


def test_build_label_panel_exposes_dual_side_targets() -> None:
    panel = build_label_panel(
        _aligned_for_labels(),
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    assert panel.rank_target_long is not None
    assert panel.rank_target_short is not None
    assert panel.magnitude_target_long is not None
    assert panel.magnitude_target_short is not None
    # cost_clearance_target_long/short는 LabelDiagnostics로 격리됨 — LabelPanel에 없음
    assert not hasattr(panel, "cost_clearance_target_long")
    assert not hasattr(panel, "cost_clearance_target_short")
    assert panel.relevance_long is not None
    assert panel.relevance_short is not None


def test_build_label_panel_keeps_b1_no_fee_slippage_deduction() -> None:
    aligned = _aligned_for_labels()
    cfg = StrategyMLConfig(label_horizon_bars=1, fee_bps=50.0, slippage_bps=50.0, min_group_size=2)
    panel = build_label_panel(aligned, cfg)
    assert panel.metadata["round_trip_cost_bps"] > 0.0
    valid = panel.eligible_mask & np.isfinite(panel.exec_net_ret)
    assert np.any(valid)


def test_build_label_panel_magnitude_target_long_is_signed_exec_net_ret() -> None:
    """magnitude_target_long must equal exec_net_ret (signed, not censored).

    The long calibrator receives the full signed distribution to learn direction
    + magnitude jointly. Censored max(x,0) targets collapse fold OOS IC to
    consistently negative values, causing quality gate failure.
    """
    # Arrange
    aligned = _aligned_for_labels()
    cfg = StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2)

    # Act
    panel = build_label_panel(aligned, cfg)

    # Assert — signed: magnitude_target_long[valid] == exec_net_ret[valid]
    assert panel.magnitude_target_long is not None
    valid_mask = panel.eligible_mask & np.isfinite(panel.exec_net_ret)
    long_valid = panel.magnitude_target_long[valid_mask]
    assert long_valid.size > 0, "no valid cells to assert on"
    np.testing.assert_array_almost_equal(
        long_valid,
        panel.exec_net_ret[valid_mask].astype(np.float32),
        decimal=6,
        err_msg="magnitude_target_long must equal exec_net_ret (signed), not max(exec_net_ret,0)",
    )
    # Metadata key must reflect signed contract
    assert panel.metadata["magnitude_target_long_key"] == "exec_net_ret"


def test_build_label_panel_magnitude_target_short_is_signed_neg_exec_net_ret() -> None:
    """magnitude_target_short must equal -exec_net_ret (signed, not censored).

    The short calibrator receives the full signed distribution to learn direction
    + magnitude jointly. Censored max(-x,0) targets collapse fold OOS IC.
    """
    # Arrange
    aligned = _aligned_for_labels()
    cfg = StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2)

    # Act
    panel = build_label_panel(aligned, cfg)

    # Assert — signed: magnitude_target_short[valid] == -exec_net_ret[valid]
    assert panel.magnitude_target_short is not None
    valid_mask = panel.eligible_mask & np.isfinite(panel.exec_net_ret)
    short_valid = panel.magnitude_target_short[valid_mask]
    assert short_valid.size > 0, "no valid cells to assert on"
    np.testing.assert_array_almost_equal(
        short_valid,
        (-panel.exec_net_ret[valid_mask]).astype(np.float32),
        decimal=6,
        err_msg=(
            "magnitude_target_short must equal -exec_net_ret (signed), "
            "not max(-exec_net_ret,0)"
        ),
    )
    # Metadata key must reflect signed contract
    assert panel.metadata["magnitude_target_short_key"] == "-exec_net_ret"


# ---------------------------------------------------------------------------
# Phase 1C — time-decay sample weighting (spec §A)
# ---------------------------------------------------------------------------


def _aligned_for_timedecay_flat(n_bars: int = 20) -> AlignedMarketData:
    """Flat-price panel: all log returns = 0 → y_ev_abs = 0 → EV-scaling is neutralized.

    With constant prices, sample_weight = liq_weight * time_decay only,
    allowing pure isolation of the time-decay multiplier.
    """
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    close_2d = np.full((n_bars, 2), 100.0, dtype=np.float64)  # flat → log-return = 0
    open_2d = close_2d.copy()
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((n_bars, 2), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((n_bars, 2), dtype=np.float64),
        active_mask=np.ones((n_bars, 2), dtype=bool),
        warm_mask=np.ones((n_bars, 2), dtype=bool),
        entry_block_mask=np.zeros((n_bars, 2), dtype=bool),
        kill_mask=np.zeros((n_bars, 2), dtype=bool),
    )


def test_time_decay_halflife_recent_weight_higher() -> None:
    """halflife=1080 적용 시 모든 valid row의 가중치가 단조증가(최신 bar가 더 높음)해야 한다.

    격리 전략: flat price(log-ret=0) → y_ev_abs=0 → EV-scaling 제거.
    결과로 sample_weight = liq_weight(상수) * time_decay(단조감소 in t) → 순수 decay 검증.
    """
    # Arrange — flat prices neutralize EV-scaling variance
    n_bars = 20
    aligned = _aligned_for_timedecay_flat(n_bars)
    cfg = StrategyMLConfig(
        label_horizon_bars=1,
        purge_bars=1,
        embargo_bars=1,
        sample_weight_time_decay_halflife_bars=1080,
    )

    # Act
    panel = build_label_panel(aligned, cfg)

    # eligible rows with non-zero weight
    valid_rows = np.any(panel.sample_weight > 0, axis=1)
    assert valid_rows.any(), "No valid rows with positive sample_weight"
    valid_indices = np.where(valid_rows)[0]

    # Assert — with flat prices, weights per row are all identical across symbols.
    # time_decay[t] = exp(-lam * (T-1 - t)), so valid_indices[-1] > valid_indices[0].
    assert len(valid_indices) >= 2, "Need ≥ 2 valid rows to compare"
    first_idx = int(valid_indices[0])
    last_idx = int(valid_indices[-1])
    w_first = float(panel.sample_weight[first_idx, 0])
    w_last = float(panel.sample_weight[last_idx, 0])
    assert w_last > w_first, (
        f"time-decay: expected w[last]({w_last:.8f}) > w[first]({w_first:.8f})"
    )
    # Verify monotonicity across all consecutive valid rows
    weights_col0 = panel.sample_weight[valid_indices, 0]
    assert np.all(np.diff(weights_col0) > 0), (
        "sample_weight must be strictly increasing along time axis when prices are flat"
    )


def test_time_decay_disabled_when_halflife_none() -> None:
    """halflife=None 이면 time-decay 없이 uniform 가중치를 유지한다."""
    # Arrange — flat prices: EV-scaling은 0으로 중립화. decay/no-decay 비교만 격리.
    n_bars = 20
    aligned = _aligned_for_timedecay_flat(n_bars)
    cfg_decay = StrategyMLConfig(
        label_horizon_bars=1, purge_bars=1, embargo_bars=1,
        sample_weight_time_decay_halflife_bars=1080,
    )
    cfg_nodecay = StrategyMLConfig(
        label_horizon_bars=1, purge_bars=1, embargo_bars=1,
        sample_weight_time_decay_halflife_bars=None,
    )

    # Act
    panel_decay = build_label_panel(aligned, cfg_decay)
    panel_nodecay = build_label_panel(aligned, cfg_nodecay)

    # Assert — decay variant must have strictly different (non-uniform) weights
    # across time relative to no-decay variant
    w_decay = panel_decay.sample_weight
    w_nodecay = panel_nodecay.sample_weight
    # At least one row should differ (time-decay shrinks early rows)
    assert not np.allclose(w_decay, w_nodecay, atol=1e-6), (
        "time-decay weights must differ from no-decay weights"
    )
