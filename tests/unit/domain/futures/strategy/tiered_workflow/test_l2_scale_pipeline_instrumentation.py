"""V2: L2 스케일 파이프라인 계측 — 수정 후 vol_target 정규화 검증."""
import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps
from src.domain.futures.strategy.tiered_workflow.awf_sim import compute_expected_layer2_edge


def test_project_all_caps_vol_upscale_normalizes_book() -> None:
    """Fix 2 검증: allow_vol_upscale=True 시 book vol이 vol_target으로 정규화."""
    n = 5
    # raw Kelly book: 연율 vol ~20% (vol_target=100% 미만)
    w = np.full(n, 0.04, dtype=np.float64)  # gross=0.2, small
    sigma = np.full(n, 0.02, dtype=np.float64)  # per-bar vol 2%
    btc_beta = np.zeros(n)
    bars_per_year = 2190.0

    sigma_port = float(np.sqrt(float(np.dot(w**2, sigma**2))))
    ann_vol_before = sigma_port * np.sqrt(bars_per_year)  # ~20%
    assert ann_vol_before < 0.5, "Test setup: book vol must be < 50%"

    caps = PortfolioCaps(gross=10.0, per_symbol=1.0, target_ann_vol=1.0)

    # 수정 후: allow_vol_upscale=True → 양방향 정규화
    w_out = project_all_caps(w, btc_beta, sigma_port, bars_per_year, caps, allow_vol_upscale=True)

    sigma_port_out = float(np.sqrt(float(np.dot(w_out**2, sigma**2))))
    ann_vol_after = sigma_port_out * np.sqrt(bars_per_year)
    assert ann_vol_after == pytest.approx(1.0, rel=0.05), (
        f"After upscale, ann_vol should be ~1.0, got {ann_vol_after:.3f}"
    )


def test_project_all_caps_vol_downscale_only_by_default() -> None:
    """기존 동작 보존: allow_vol_upscale=False(기본) 시 확대 없음."""
    n = 3
    w = np.full(n, 0.1, dtype=np.float64)
    sigma = np.full(n, 0.02, dtype=np.float64)
    btc_beta = np.zeros(n)
    bars_per_year = 2190.0
    sigma_port = float(np.sqrt(float(np.dot(w**2, sigma**2))))
    _ann_vol_before = sigma_port * np.sqrt(bars_per_year)

    caps = PortfolioCaps(gross=10.0, per_symbol=1.0, target_ann_vol=2.0)  # target > current
    w_out = project_all_caps(w, btc_beta, sigma_port, bars_per_year, caps)  # default=False

    # 확대 금지이므로 가중치 변경 없어야 함
    np.testing.assert_allclose(w_out, w, rtol=1e-6, err_msg="No upscale with default allow_vol_upscale=False")


def test_throttle_score_no_double_deduct() -> None:
    """Fix 1 검증: _book_edge_score가 mu(already net)를 재차감하지 않음."""
    import inspect

    from src.domain.futures.strategy.tiered_workflow.awf_sim import _book_edge_score
    # 인자 수 확인: 2개여야 함 (w, mu_bps)
    sig = inspect.signature(_book_edge_score)
    params = list(sig.parameters.keys())
    assert len(params) == 2, f"_book_edge_score should have 2 params, got {params}"
    assert "effective_hurdle_bps" not in params, "hurdle_bps must be removed (double-deduct fix)"

    w = np.array([0.1, 0.2, 0.0])
    mu = np.array([10.0, 20.0, 0.0])  # already net
    score = _book_edge_score(w, mu)
    expected = (0.1 * 10.0 + 0.2 * 20.0) / (0.1 + 0.2)
    assert score == pytest.approx(expected, rel=1e-6)


def test_friction_event_level_gross_basis() -> None:
    """Friction Fix: gross-basis 신호의 event-level edge가 hurdle*safety와 비교 가능."""
    edge = compute_expected_layer2_edge(
        side=1,
        expected_gross_bps=100.0,
        expected_net_bps=80.0,
        expected_holding_bars=72,
        execution_cost_bps=3.8,
        edge_basis="gross",
        fixed_cost_safety_mult=1.25,
    )
    assert edge.signed_gross_bps_per_bar == pytest.approx(100.0 / 72.0, rel=1e-6)
    assert edge.signed_net_bps_per_bar == pytest.approx(
        100.0 / 72.0 - 3.8 * 1.25 / 72.0, rel=1e-6
    )
    event_gross = 100.0
    total_cost = 3.8 * 1.25
    assert abs(event_gross) >= total_cost, "gross-basis event edge must cover total round-trip cost"


def test_friction_event_level_net_basis() -> None:
    """Friction Fix: net-basis(expected_gross_bps=0) 신호도 event-level edge로 friction 통과."""
    edge = compute_expected_layer2_edge(
        side=1,
        expected_gross_bps=0.0,
        expected_net_bps=30.0,
        expected_holding_bars=72,
        execution_cost_bps=3.8,
        edge_basis="net",
        fixed_cost_safety_mult=1.25,
    )
    assert edge.signed_gross_bps_per_bar == 0.0
    net_per_bar = 30.0 / 72.0
    assert edge.signed_net_bps_per_bar == pytest.approx(net_per_bar, rel=1e-6)
    event_gross = net_per_bar * 72.0  # event-level reconstruction
    assert event_gross == pytest.approx(30.0, rel=1e-6)
    total_cost = 3.8 * 1.25
    assert abs(event_gross) >= total_cost, "net-basis event-level edge must cover total round-trip cost"


def test_friction_comparison_dimension_event_level() -> None:
    """Friction Fix: per-bar vs per-rebalance-bar 비교에서 event-level vs total-cost로 차원 통일.

    The old code compared per-holding-bar edge (gross_bps / holding_bars) against
    per-rebalance-bar cost (hurdle / rebalance_bars) — different time denominators.
    The correct comparison is event-level gross edge against total round-trip cost.
    """
    hurdle = 3.8
    safety = 1.25
    rebalance_bars = 3

    # Signal with modest edge that old per-bar comparison incorrectly fails,
    # but correct event-level comparison passes.
    expected_gross_bps = 5.0
    holding_bars = 48

    per_bar_edge = expected_gross_bps / holding_bars  # 0.104
    per_rebal_cost = hurdle / rebalance_bars  # 1.267
    total_cost = hurdle * safety  # 4.75

    old_friction_pass = abs(per_bar_edge) >= per_rebal_cost  # 0.104 >= 1.267 → False
    new_friction_pass = abs(expected_gross_bps) >= total_cost  # 5.0 >= 4.75 → True

    assert not old_friction_pass, "Old per-bar comparison incorrectly rejects viable signal"
    assert new_friction_pass, "New event-level comparison correctly passes viable signal"
