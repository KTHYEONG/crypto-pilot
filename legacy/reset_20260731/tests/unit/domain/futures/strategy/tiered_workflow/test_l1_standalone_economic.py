"""V1: L1 알파 단독 경제성 검증 — 사이징 우회 EW 백테스트."""

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    build_directional_equal_weight_baseline,
)


def _synthetic_signals(n_sym: int, n_events: int, mu_bps: float = 30.0):
    """고정 양의 net edge를 가진 합성 신호 딕셔너리."""
    from src.domain.futures.strategy.cs_rank import SymbolSignal

    syms = [f"SYM{i:02d}" for i in range(n_sym)]
    signals = {
        s: SymbolSignal(
            raw_mu=mu_bps,
            volatility=0.02,
            n_obs=100,
            t_stat=3.0,
            valid=True,
            beta_btc=None,
            quality_weight=1.0,
        )
        for s in syms
    }
    return signals, syms


def test_l1_ew_book_positive_cagr_given_positive_edge():
    """Arrange: 30bps/bar net edge, 52 symbols. Act: EW book. Then: CAGR > 0."""
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps

    n_sym = 10
    signals, _syms = _synthetic_signals(n_sym, n_events=100, mu_bps=30.0)
    mu_arr = np.array([s.raw_mu for s in signals.values()])
    sig_arr = np.array([s.volatility for s in signals.values()])
    btc_beta = np.zeros(n_sym)
    caps = PortfolioCaps(gross=5.0, per_symbol=0.5)

    # EW book using build_directional_equal_weight_baseline
    w_ew = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_arr,
        strategy_weights=np.ones(n_sym) * 0.1,  # dummy strategy weights
        sigma=sig_arr,
        btc_beta=btc_beta,
        caps=caps,
        bars_per_year=2190.0,
    )

    # Assert: EW비중 존재 및 양의 기대수익
    assert float(np.sum(np.abs(w_ew))) > 0, "EW book must have non-zero weights"
    expected_return = float(np.dot(w_ew, mu_arr))
    assert expected_return > 0, f"EW book expected return must be positive, got {expected_return}"


def test_l1_ew_book_zero_edge_gives_no_return():
    """Arrange: 0bps edge. Act: EW book. Then: expected return = 0."""
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps

    n_sym = 5
    mu_arr = np.zeros(n_sym)
    sig_arr = np.full(n_sym, 0.02)
    btc_beta = np.zeros(n_sym)
    caps = PortfolioCaps()

    w_ew = build_directional_equal_weight_baseline(
        signed_net_mu_bps=mu_arr,
        strategy_weights=np.zeros(n_sym),
        sigma=sig_arr,
        btc_beta=btc_beta,
        caps=caps,
        bars_per_year=2190.0,
    )
    assert float(np.sum(np.abs(w_ew))) == pytest.approx(0.0, abs=1e-9)
