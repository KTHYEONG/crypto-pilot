"""V2: L2 스케일 파이프라인 계측 — 수정 후 vol_target 정규화 검증."""
import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps


def test_project_all_caps_vol_upscale_normalizes_book():
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


def test_project_all_caps_vol_downscale_only_by_default():
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


def test_throttle_score_no_double_deduct():
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
