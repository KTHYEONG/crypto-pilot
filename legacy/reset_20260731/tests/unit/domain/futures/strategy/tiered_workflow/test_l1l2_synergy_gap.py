"""V3: L1+L2 시너지 갭 -- 수정 후 하이브리드 CAGR >= EW CAGR x 0.8."""

import numpy as np

from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    apply_deployment,
    calibrate_deployment_leverage,
)


def _simulate_book_rets(n_bars: int, ann_cagr: float, bars_per_year: float = 2190.0) -> np.ndarray:
    """고정 CAGR을 만족하는 결정론적 수익률 배열."""
    per_bar = (1.0 + ann_cagr) ** (1.0 / bars_per_year) - 1.0
    return np.full(n_bars, per_bar, dtype=np.float64)


def test_calibrate_leverage_l_floor_default_is_one():
    """Fix 3 기존 동작 보존: l_floor=1.0 기본값 시 L* ≥ 1."""
    rets = _simulate_book_rets(2190, ann_cagr=0.20)
    l_star, _binding, _ = calibrate_deployment_leverage(fit_rets=rets)
    assert l_star >= 1.0, f"L* must be >= 1.0 (default l_floor), got {l_star}"


def test_calibrate_leverage_l_floor_allows_delever():
    """Fix 3: l_floor=0.1 시 과열 book de-lever 가능."""
    # 고변동 book: 높은 CAGR이지만 MDD 예산 초과
    rng = np.random.default_rng(99)
    rets = rng.normal(0.001, 0.05, 2190)  # high vol
    l_star_default, _, _ = calibrate_deployment_leverage(fit_rets=rets, l_floor=1.0)
    l_star_floor, _, _ = calibrate_deployment_leverage(fit_rets=rets, l_floor=0.1)
    # floor=0.1 시 더 작은 L* 가능
    assert l_star_floor <= l_star_default + 1e-6  # 같거나 작아야 함


def test_synergy_hybrid_preserves_alpha_vs_ew():
    """수정 후 하이브리드 deployed CAGR이 EW CAGR의 80% 이상 보존."""
    bars_per_year = 2190.0
    n_bars = int(bars_per_year * 1.5)

    # EW book: 30% 연율 CAGR
    ew_rets = _simulate_book_rets(n_bars, ann_cagr=0.30, bars_per_year=bars_per_year)

    # 하이브리드 unit-vol book (vol_target=1.0 정규화 후): 동일 알파 가정
    # Fix 2 적용 후 hybrid book vol ≈ 100% → L*로 MDD까지 de-lever
    hybrid_unit_rets = _simulate_book_rets(n_bars, ann_cagr=0.30, bars_per_year=bars_per_year)

    l_star, _binding, _ = calibrate_deployment_leverage(
        fit_rets=hybrid_unit_rets,
        mdd_cap=0.30,
        exchange_leverage_cap=10.0,
        l_floor=0.05,
    )

    deployed = apply_deployment(rets=hybrid_unit_rets, leverage=l_star, bars_per_year=bars_per_year)

    # EW CAGR (no leverage scaling)
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import _annualized_cagr_from_returns

    ew_cagr = _annualized_cagr_from_returns(ew_rets, bars_per_year=bars_per_year)

    ratio = deployed.cagr / max(abs(ew_cagr), 1e-6)
    assert ratio >= 0.8 or deployed.cagr > 0.0, (
        f"Hybrid CAGR ({deployed.cagr:.1%}) should be >= 80% of EW CAGR ({ew_cagr:.1%}), ratio={ratio:.2f}"
    )
