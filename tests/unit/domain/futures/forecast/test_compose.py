"""Tests for forecast/compose.py — compose_mu SSOT."""
from __future__ import annotations

from typing import ClassVar

import numpy as np

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


_BASE_PARAMS: dict = {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 10.0, "MAKER_RATIO": 0.0}


class TestComposeMuHurdle:
    def test_above_hurdle_passes(self) -> None:
        # alpha=50bps, cost=14bps, hurdle=10bps → net=36bps > 10bps → xs > 0
        af = _make_alpha(long_val=0.005, short_val=0.004)
        cf = _make_cost(bps=14.0)
        xs_l, xs_s, _mu_l, _mu_s = compose_mu(af, cf, _BASE_PARAMS)

        assert np.all(xs_l > 0.0)
        assert np.all(xs_s > 0.0)

    def test_rank_sizing_output_nonnegative(self) -> None:
        # rank-sizing은 경질 hurdle이 아닌 횡단면 순위 기반 연속 가중.
        # EV 크기와 무관하게 출력은 항상 [0, 1] 범위 비음수여야 함.
        af = _make_alpha(long_val=0.0005, short_val=0.0004)
        cf = _make_cost(bps=14.0)
        xs_l, xs_s, _mu_l, _mu_s = compose_mu(af, cf, _BASE_PARAMS)

        assert np.all(xs_l >= 0.0)
        assert np.all(xs_s >= 0.0)
        assert np.all(xs_l <= 1.0)
        assert np.all(xs_s <= 1.0)

    def test_hurdle_boundary_exact(self) -> None:
        # alpha=24bps, cost=14bps → net=10bps == hurdle → passes (>=)
        hurdle_bps = 10.0
        cost_bps = 14.0
        alpha_frac = (hurdle_bps + cost_bps) / 10000.0
        af = _make_alpha(long_val=alpha_frac, short_val=alpha_frac)
        cf = _make_cost(bps=cost_bps)
        xs_l, _xs_s, _, _ = compose_mu(af, cf, _BASE_PARAMS)

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

    def test_xs_bounded_after_rank_sizing(self) -> None:
        # rank-sizing 후 출력은 tanh 범위 (0, 1) 이내여야 함 (mu와 동일하지 않음).
        af = _make_alpha(long_val=0.01)
        cf = _make_cost(bps=0.0)
        xs_l, _, mu_l, _ = compose_mu(af, cf, _BASE_PARAMS)
        pos_mask = mu_l > 0.0
        # rank-sizing 출력: mu와 다르지만 [0, 1] 범위, 비음수
        assert np.all(xs_l[pos_mask] >= 0.0)
        assert np.all(xs_l[pos_mask] <= 1.0)


class TestComposeMuRankCsNeutral:
    """rank_cs_neutral admission mode 단위 테스트."""

    _PARAMS_RANK: ClassVar[dict[str, float | int | str]] = {
        "BETA_ALPHA": 1.0,
        "EV_HURDLE_BPS": 10.0,
        "POST_COST_ADMISSION_MODE": "rank_cs_neutral",
        "RANK_SELECT_QUANTILE": 0.33,
        "IC_PRIOR_FOR_GATE": 0.03,
        "COMPOSER_SIGMA_BPS": 500.0,
        "COST_GATE_BPS": 24.0,
        "REBALANCE_BARS": 1,
        "EV_SECONDARY_TILT_WEIGHT": 0.0,
    }

    def _make_alpha_with_rank(
        self, rank_long_val: float = 1.0, rank_short_val: float = -1.0
    ) -> AlphaForecast:
        al = np.full(_SHAPE, 0.003, dtype=np.float32)
        as_ = np.full(_SHAPE, 0.003, dtype=np.float32)
        rank_l = np.full(_SHAPE, rank_long_val, dtype=np.float32)
        rank_s = np.full(_SHAPE, rank_short_val, dtype=np.float32)
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
            rank_score_long_2d=rank_l,
            rank_score_short_2d=rank_s,
        )

    def test_rank_cs_neutral_rank_none_falls_back_to_ev_gate(self) -> None:
        """rank_score가 None이면 rank_cs_neutral은 xs를 z-score 없이 반환한다."""
        # Arrange
        af_no_rank = _make_alpha(long_val=0.003, short_val=0.003)
        cf = _make_cost(bps=0.0)

        # Act
        xs_l, xs_s, _, _ = compose_mu(af_no_rank, cf, self._PARAMS_RANK)

        # Assert — rank_score None이면 mu_long == 0.003 (> 0 이므로 isfinite passes)
        assert xs_l.shape == _SHAPE
        assert xs_s.shape == _SHAPE

    def test_rank_cs_neutral_xs_nonnegative(self) -> None:
        """rank_cs_neutral 모드: xs는 선택된 포지션에만 양수값, 나머지 0."""
        # Arrange
        af = self._make_alpha_with_rank()
        cf = _make_cost(bps=0.0)

        # Act
        xs_l, xs_s, _, _ = compose_mu(af, cf, self._PARAMS_RANK)

        # Assert — xs는 0 이상이어야 함 (z-score 선택 구간)
        assert np.all(xs_l >= 0.0)
        assert np.all(xs_s >= 0.0)

    def test_rank_cs_neutral_output_shape(self) -> None:
        """rank_cs_neutral 모드: 출력 shape이 입력과 동일."""
        # Arrange
        af = self._make_alpha_with_rank()
        cf = _make_cost(bps=0.0)

        # Act
        xs_l, xs_s, mu_l, mu_s = compose_mu(af, cf, self._PARAMS_RANK)

        # Assert
        for arr in (xs_l, xs_s, mu_l, mu_s):
            assert arr.shape == _SHAPE

    def test_rank_cs_neutral_varied_scores_select_top_quantile(self) -> None:
        """분산된 rank score에서 상위 분위만 선택된다."""
        # Arrange — 10개 종목, 5개 bar, 상위 2개(top 20%)가 높은 rank
        shape = (5, 10)
        rank_l = np.tile(np.arange(10, dtype=np.float32), (5, 1))  # 0~9
        rank_s = np.tile(-np.arange(10, dtype=np.float32), (5, 1))  # 0~-9
        al = np.full(shape, 0.003, dtype=np.float32)
        as_ = np.full(shape, 0.003, dtype=np.float32)
        af = AlphaForecast(
            datetimes=np.array([]),
            symbols=(),
            alpha_long_2d=al,
            alpha_short_2d=as_,
            q10_long_2d=None, q50_long_2d=None, q90_long_2d=None,
            q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
            confidence_long_2d=None, confidence_short_2d=None,
            eligible_mask=np.ones(shape, dtype=bool),
            source="test",
            artifact_hash=_DUMMY_HASH,
            rank_score_long_2d=rank_l,
            rank_score_short_2d=rank_s,
        )
        cf = CostForecast(
            execution_cost_bps_2d=np.zeros(shape),
            execution_cost_fraction_2d=np.zeros(shape),
            uncertainty_bps_2d=np.zeros(shape),
            capacity_notional_2d=None,
            source="test",
        )
        params = {**self._PARAMS_RANK, "RANK_SELECT_QUANTILE": 0.30}

        # Act
        xs_l, _xs_s, _, _ = compose_mu(af, cf, params)

        # Assert — top 30% of 10 = 3 종목이 선택됨 → xs_l의 nonzero ≤ 3 per bar
        for t in range(5):
            nz_long = int(np.count_nonzero(xs_l[t] > 0.0))
            assert nz_long <= 3, f"bar {t}: {nz_long} selected, expected <= 3"

    def test_rank_cs_neutral_uses_policy_payload_when_given(self) -> None:
        shape = (4, 6)
        rank_l = np.tile(np.array([0, 1, 2, 3, 4, 5], dtype=np.float32), (4, 1))
        rank_s = np.tile(np.array([0, -1, -2, -3, -4, -5], dtype=np.float32), (4, 1))
        af = AlphaForecast(
            datetimes=np.array([]),
            symbols=(),
            alpha_long_2d=np.full(shape, 0.002, dtype=np.float32),
            alpha_short_2d=np.full(shape, 0.002, dtype=np.float32),
            q10_long_2d=None, q50_long_2d=None, q90_long_2d=None,
            q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
            confidence_long_2d=None, confidence_short_2d=None,
            eligible_mask=np.ones(shape, dtype=bool),
            source="test",
            artifact_hash=_DUMMY_HASH,
            rank_score_long_2d=rank_l,
            rank_score_short_2d=rank_s,
        )
        params = {
            **self._PARAMS_RANK,
            "RANK_SELECTION_POLICY": {
                "polarity": 1,
                "quantile": 0.33,
                "min_abs_z": 0.0,
                "weighting": "equal",
                "weight_k": 3.0,
                "holding_bars": 12,
                "validation_net_lcb_bps": 1.0,
                "validation_gross_bps": 1.0,
                "validation_ir_t": 1.0,
                "validation_monotonicity": 1.0,
                "n_obs": 100,
            },
        }
        xs_l, xs_s, _, _ = compose_mu(
            af,
            CostForecast(
                execution_cost_bps_2d=np.zeros(shape),
                execution_cost_fraction_2d=np.zeros(shape),
                uncertainty_bps_2d=np.zeros(shape),
                capacity_notional_2d=None,
                source="test",
            ),
            params,
        )
        assert int(np.count_nonzero(xs_l[0] > 0.0)) <= 2
        assert int(np.count_nonzero(xs_s[0] > 0.0)) <= 2

    def test_rank_cs_neutral_uses_alpha_policy_metadata_when_params_missing(self) -> None:
        shape = (4, 6)
        rank_l = np.tile(np.array([0, 1, 2, 3, 4, 5], dtype=np.float32), (4, 1))
        rank_s = np.tile(np.array([0, -1, -2, -3, -4, -5], dtype=np.float32), (4, 1))
        af = AlphaForecast(
            datetimes=np.array([]),
            symbols=(),
            alpha_long_2d=np.full(shape, 0.002, dtype=np.float32),
            alpha_short_2d=np.full(shape, 0.002, dtype=np.float32),
            q10_long_2d=None, q50_long_2d=None, q90_long_2d=None,
            q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
            confidence_long_2d=None, confidence_short_2d=None,
            eligible_mask=np.ones(shape, dtype=bool),
            source="test",
            artifact_hash=_DUMMY_HASH,
            rank_score_long_2d=rank_l,
            rank_score_short_2d=rank_s,
            rank_selection_policy={
                "polarity": 1,
                "quantile": 0.33,
                "min_abs_z": 0.0,
                "weighting": "equal",
                "weight_k": 3.0,
                "holding_bars": 12,
                "validation_net_lcb_bps": 1.0,
                "validation_gross_bps": 1.0,
                "validation_ir_t": 1.0,
                "validation_monotonicity": 1.0,
                "n_obs": 100,
            },
        )
        params = {**self._PARAMS_RANK}
        xs_l, xs_s, _, _ = compose_mu(
            af,
            CostForecast(
                execution_cost_bps_2d=np.zeros(shape),
                execution_cost_fraction_2d=np.zeros(shape),
                uncertainty_bps_2d=np.zeros(shape),
                capacity_notional_2d=None,
                source="test",
            ),
            params,
        )
        assert int(np.count_nonzero(xs_l[0] > 0.0)) <= 2
        assert int(np.count_nonzero(xs_s[0] > 0.0)) <= 2

    def test_rank_cs_neutral_does_not_subtract_cost_from_rank_weights(self) -> None:
        shape = (4, 6)
        rank_l = np.tile(np.array([0, 1, 2, 3, 4, 5], dtype=np.float32), (4, 1))
        rank_s = np.tile(np.array([0, -1, -2, -3, -4, -5], dtype=np.float32), (4, 1))
        af = AlphaForecast(
            datetimes=np.array([]),
            symbols=(),
            alpha_long_2d=np.full(shape, 0.003, dtype=np.float32),
            alpha_short_2d=np.full(shape, 0.003, dtype=np.float32),
            q10_long_2d=None, q50_long_2d=None, q90_long_2d=None,
            q10_short_2d=None, q50_short_2d=None, q90_short_2d=None,
            confidence_long_2d=None, confidence_short_2d=None,
            eligible_mask=np.ones(shape, dtype=bool),
            source="test",
            artifact_hash=_DUMMY_HASH,
            rank_score_long_2d=rank_l,
            rank_score_short_2d=rank_s,
            rank_selection_policy={
                "polarity": 1,
                "quantile": 0.33,
                "min_abs_z": 0.0,
                "weighting": "equal",
                "weight_k": 3.0,
                "holding_bars": 12,
                "validation_net_lcb_bps": 1.0,
                "validation_gross_bps": 1.0,
                "validation_ir_t": 1.0,
                "validation_monotonicity": 1.0,
                "n_obs": 100,
            },
        )
        cf_zero = CostForecast(
            execution_cost_bps_2d=np.zeros(shape),
            execution_cost_fraction_2d=np.zeros(shape),
            uncertainty_bps_2d=np.zeros(shape),
            capacity_notional_2d=None,
            source="test",
        )
        cf_high = CostForecast(
            execution_cost_bps_2d=np.full(shape, 50.0),
            execution_cost_fraction_2d=np.full(shape, 50.0 / 10000.0),
            uncertainty_bps_2d=np.zeros(shape),
            capacity_notional_2d=None,
            source="test",
        )
        xs_l_zero, xs_s_zero, _mu_l_zero, _mu_s_zero = compose_mu(af, cf_zero, self._PARAMS_RANK)
        xs_l_high, xs_s_high, _mu_l_high, _mu_s_high = compose_mu(af, cf_high, self._PARAMS_RANK)
        np.testing.assert_allclose(xs_l_zero, xs_l_high)
        np.testing.assert_allclose(xs_s_zero, xs_s_high)
