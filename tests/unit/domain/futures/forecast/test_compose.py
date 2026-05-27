"""Tests for forecast/compose.py — compose_mu SSOT."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.forecast.compose import compose_mu
from src.domain.futures.forecast.contracts import (
    AlphaArtifactHash,
    AlphaForecast,
    CostForecast,
)

_SHAPE = (20, 5)
_DUMMY_HASH = AlphaArtifactHash("", "", "", "", "", "test", 0)


def _make_alpha(long_val: float = 0.005, short_val: float = 0.003) -> AlphaForecast:
    al = np.full(_SHAPE, long_val, dtype=np.float32)
    as_ = np.full(_SHAPE, short_val, dtype=np.float32)
    return AlphaForecast(
        datetimes=np.array([]),
        symbols=(),
        alpha_long_2d=al,
        alpha_short_2d=as_,
        q10_long_2d=None, q50_long_2d=None, q90_long_2d=None,
        q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
        confidence_long_2d=None, confidence_short_2d=None,
        eligible_mask=np.ones(_SHAPE, dtype=bool),
        source="test",
        artifact_hash=_DUMMY_HASH,
    )


def _make_cost(bps: float = 14.0) -> CostForecast:
    bps_2d = np.full(_SHAPE, bps, dtype=np.float64)
    frac_2d = bps_2d / 10000.0
    return CostForecast(
        execution_cost_bps_2d=bps_2d,
        execution_cost_fraction_2d=frac_2d,
        uncertainty_bps_2d=np.zeros(_SHAPE),
        capacity_notional_2d=None,
        source="test",
    )


_BASE_PARAMS: dict = {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 10.0}


class TestComposeMuHurdle:
    def test_above_hurdle_passes(self) -> None:
        # alpha=50bps, cost=14bps, hurdle=10bps → net=36bps > 10bps → xs > 0
        af = _make_alpha(long_val=0.005, short_val=0.004)
        cf = _make_cost(bps=14.0)
        xs_l, xs_s, mu_l, mu_s = compose_mu(af, cf, _BASE_PARAMS)

        assert np.all(xs_l > 0.0)
        assert np.all(xs_s > 0.0)

    def test_below_hurdle_zeroed(self) -> None:
        # alpha=5bps, cost=14bps → net=-9bps < 10bps → xs=0
        af = _make_alpha(long_val=0.0005, short_val=0.0004)
        cf = _make_cost(bps=14.0)
        xs_l, xs_s, mu_l, mu_s = compose_mu(af, cf, _BASE_PARAMS)

        assert np.all(xs_l == 0.0)
        assert np.all(xs_s == 0.0)

    def test_hurdle_boundary_exact(self) -> None:
        # alpha=24bps, cost=14bps → net=10bps == hurdle → passes (>=)
        hurdle_bps = 10.0
        cost_bps = 14.0
        alpha_frac = (hurdle_bps + cost_bps) / 10000.0
        af = _make_alpha(long_val=alpha_frac, short_val=alpha_frac)
        cf = _make_cost(bps=cost_bps)
        xs_l, xs_s, _, _ = compose_mu(af, cf, _BASE_PARAMS)

        assert np.all(xs_l >= 0.0)
        assert np.all(xs_l > 0.0)  # 정확히 hurdle이면 통과

    def test_mu_equals_alpha_minus_cost(self) -> None:
        af = _make_alpha(long_val=0.01, short_val=0.008)
        cf = _make_cost(bps=14.0)
        _, _, mu_l, mu_s = compose_mu(af, cf, _BASE_PARAMS)

        expected_mu_l = 1.0 * 0.01 - 14.0 / 10000.0
        expected_mu_s = 1.0 * 0.008 - 14.0 / 10000.0
        # float32 alpha → float64 변환 시 정밀도 손실 허용
        np.testing.assert_allclose(mu_l, expected_mu_l, rtol=1e-5)
        np.testing.assert_allclose(mu_s, expected_mu_s, rtol=1e-5)

    def test_beta_alpha_scales_signal(self) -> None:
        af = _make_alpha(long_val=0.01, short_val=0.01)
        cf = _make_cost(bps=0.0)
        params_half = {**_BASE_PARAMS, "BETA_ALPHA": 0.5}
        _, _, mu_l_half, _ = compose_mu(af, cf, params_half)
        _, _, mu_l_full, _ = compose_mu(af, cf, _BASE_PARAMS)

        np.testing.assert_allclose(mu_l_half, mu_l_full * 0.5, rtol=1e-9)


class TestComposeMuAmortize:
    def test_amortize_off_by_default(self) -> None:
        # COST_GATE_AMORTIZE 미설정 → full cost 차감
        af = _make_alpha(long_val=0.002, short_val=0.002)
        cf = _make_cost(bps=14.0)
        _, _, mu_no_flag, _ = compose_mu(af, cf, _BASE_PARAMS, holding_bars=6)
        _, _, mu_flag_false, _ = compose_mu(
            af, cf, {**_BASE_PARAMS, "COST_GATE_AMORTIZE": False}, holding_bars=6
        )
        np.testing.assert_array_equal(mu_no_flag, mu_flag_false)

    def test_amortize_on_divides_cost(self) -> None:
        # COST_GATE_AMORTIZE=True, holding_bars=5 → cost / 5
        cost_bps = 20.0
        holding = 5
        af = _make_alpha(long_val=0.01, short_val=0.01)
        cf = _make_cost(bps=cost_bps)

        _, _, mu_full, _ = compose_mu(af, cf, _BASE_PARAMS)
        _, _, mu_amort, _ = compose_mu(
            af, cf, {**_BASE_PARAMS, "COST_GATE_AMORTIZE": True}, holding_bars=holding
        )

        expected_extra = (cost_bps / 10000.0) * (1.0 - 1.0 / holding)
        np.testing.assert_allclose(mu_amort - mu_full, expected_extra, rtol=1e-9)

    def test_amortize_holding_1_no_change(self) -> None:
        # holding_bars=1 → cost / 1 = 동일
        af = _make_alpha(long_val=0.01)
        cf = _make_cost(bps=14.0)
        _, _, mu_base, _ = compose_mu(af, cf, _BASE_PARAMS)
        _, _, mu_amort, _ = compose_mu(
            af, cf, {**_BASE_PARAMS, "COST_GATE_AMORTIZE": True}, holding_bars=1
        )
        np.testing.assert_array_equal(mu_base, mu_amort)

    def test_no_double_cost_composition(self) -> None:
        # compose_mu는 cost를 1회만 차감
        af = _make_alpha(long_val=0.01)
        cf_zero = _make_cost(bps=0.0)
        cf_14 = _make_cost(bps=14.0)

        _, _, mu_zero, _ = compose_mu(af, cf_zero, _BASE_PARAMS)
        _, _, mu_14, _ = compose_mu(af, cf_14, _BASE_PARAMS)

        diff = mu_zero - mu_14
        np.testing.assert_allclose(diff, 14.0 / 10000.0, rtol=1e-9)


class TestComposeMuOutputShapes:
    def test_output_shapes_match_input(self) -> None:
        af = _make_alpha()
        cf = _make_cost()
        xs_l, xs_s, mu_l, mu_s = compose_mu(af, cf, _BASE_PARAMS)

        for arr in (xs_l, xs_s, mu_l, mu_s):
            assert arr.shape == _SHAPE

    def test_xs_is_nonnegative(self) -> None:
        af = _make_alpha()
        cf = _make_cost()
        xs_l, xs_s, _, _ = compose_mu(af, cf, _BASE_PARAMS)

        assert np.all(xs_l >= 0.0)
        assert np.all(xs_s >= 0.0)

    def test_xs_leq_mu_when_positive(self) -> None:
        af = _make_alpha(long_val=0.01)
        cf = _make_cost(bps=0.0)
        xs_l, _, mu_l, _ = compose_mu(af, cf, _BASE_PARAMS)
        pos_mask = mu_l > 0.0
        np.testing.assert_allclose(xs_l[pos_mask], mu_l[pos_mask], rtol=1e-9)
