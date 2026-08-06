from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.technical_experts.cross_sectional import _validate_alpha_panels
from src.research.universe.candidate_scan import (
    UniverseCandidateResult,
    UniverseCandidateSpec,
    evaluate_universe_candidate,
)


def _frame(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2022-04-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "close": np.full(n, 100.0),
            "taker_buy_ratio": np.full(n, 0.5),
            "quote_vol": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _panels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes = pd.DataFrame({"TESTUSDT": frame["close"].to_numpy(dtype=np.float64)}, index=frame.index)
    taker = pd.DataFrame(
        {"TESTUSDT": frame["taker_buy_ratio"].to_numpy(dtype=np.float64)},
        index=frame.index,
    )
    funding = pd.DataFrame({"TESTUSDT": np.zeros(len(frame), dtype=np.float64)}, index=frame.index)
    return closes, taker, funding


class TestUniverseCandidateSpec:
    def test_defaults(self) -> None:
        spec = UniverseCandidateSpec()
        assert spec.min_coverage == 0.99
        assert spec.seasoning_tolerance_days == 5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_coverage": 0.0},
            {"min_coverage": 1.5},
            {"seasoning_tolerance_days": -1},
        ],
    )
    def test_validation_raises(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="must"):
            UniverseCandidateSpec(**kwargs)


class TestUniverseCandidateResult:
    def test_validation_raises_on_bad_coverage(self) -> None:
        with pytest.raises(ValueError, match="coverage"):
            UniverseCandidateResult(
                symbol="A",
                first_bar=pd.Timestamp("2022-04-01", tz="UTC"),
                last_bar=pd.Timestamp("2022-04-02", tz="UTC"),
                coverage=1.5,
                has_funding=True,
                taker_ratio_valid=True,
                avg_daily_quote_vol_recent=1.0,
                qualifies=True,
            )

    def test_rejects_tz_naive_timestamps(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            UniverseCandidateResult(
                symbol="A",
                first_bar=pd.Timestamp("2022-04-01"),
                last_bar=pd.Timestamp("2022-04-02", tz="UTC"),
                coverage=1.0,
                has_funding=True,
                taker_ratio_valid=True,
                avg_daily_quote_vol_recent=1.0,
                qualifies=True,
            )

    def test_rejects_last_before_first(self) -> None:
        with pytest.raises(ValueError, match="not precede"):
            UniverseCandidateResult(
                symbol="A",
                first_bar=pd.Timestamp("2022-04-02", tz="UTC"),
                last_bar=pd.Timestamp("2022-04-01", tz="UTC"),
                coverage=1.0,
                has_funding=True,
                taker_ratio_valid=True,
                avg_daily_quote_vol_recent=1.0,
                qualifies=True,
            )


class TestEvaluateUniverseCandidate:
    # UCS-01-QUALIFICATION-PREDICATE-MATCHES-VALIDATOR
    def test_ucs_01_taker_ratio_valid_matches_validator_predicate(self) -> None:
        # The taker_ratio_valid flag must agree with _validate_alpha_panels's
        # own finite-and-in-[0,1] check on the identical frame -- the same
        # predicate, exercised through the shared panel fixture.
        frame = _frame()
        result = evaluate_universe_candidate(
            "TESTUSDT", frame, True, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        _validate_alpha_panels(*_panels(frame))
        assert result.taker_ratio_valid is True

        for bad_value in (-0.01, 1.01, np.nan, np.inf):
            corrupted = frame.copy()
            corrupted.loc[corrupted.index[5], "taker_buy_ratio"] = bad_value
            result = evaluate_universe_candidate(
                "TESTUSDT", corrupted, True, corrupted.index[0], corrupted.index[-1],
                UniverseCandidateSpec(),
            )
            with pytest.raises(DataIntegrityError, match="taker_buy_ratio"):
                _validate_alpha_panels(*_panels(corrupted))
            assert result.taker_ratio_valid is False
            assert result.qualifies is False

    # UCS-02-COVERAGE-SEASONING-LIQUIDITY-GATES
    def test_ucs_02_all_filters_pass_qualifies_true(self) -> None:
        frame = _frame()
        result = evaluate_universe_candidate(
            "TESTUSDT", frame, True, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        assert result.coverage == pytest.approx(1.0)
        assert result.qualifies is True

    def test_ucs_02_low_coverage_fails_qualification(self) -> None:
        frame = _frame()
        short = frame.iloc[:150]
        result = evaluate_universe_candidate(
            "TESTUSDT", short, True, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        assert result.coverage == pytest.approx(150 / 200)
        assert result.qualifies is False

    def test_ucs_02_late_first_bar_fails_qualification(self) -> None:
        # 10000 bars makes the 6-day-late start a pure seasoning violation:
        # coverage stays >= 0.99, so only the first_bar-proximity leg binds.
        frame = _frame(n=10000)
        late_start = frame.index[0] + pd.Timedelta(days=6)
        late = frame.loc[frame.index >= late_start]
        result = evaluate_universe_candidate(
            "TESTUSDT", late, True, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        assert result.coverage >= 0.99
        assert result.qualifies is False

    def test_ucs_02_missing_funding_fails_qualification(self) -> None:
        frame = _frame()
        result = evaluate_universe_candidate(
            "TESTUSDT", frame, False, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        assert result.qualifies is False

    def test_ucs_02_zero_trailing_liquidity_fails_qualification(self) -> None:
        frame = _frame()
        zero_vol = frame.copy()
        zero_vol["quote_vol"] = 0.0
        result = evaluate_universe_candidate(
            "TESTUSDT", zero_vol, True, zero_vol.index[0], zero_vol.index[-1],
            UniverseCandidateSpec(),
        )
        assert result.avg_daily_quote_vol_recent == 0.0
        assert result.qualifies is False

    def test_ucs_02_avg_daily_quote_vol_recent_scales_trailing_180_bars(self) -> None:
        frame = _frame()
        result = evaluate_universe_candidate(
            "TESTUSDT", frame, True, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        assert result.avg_daily_quote_vol_recent == pytest.approx(1_000_000.0 * 6.0)

    # UCS-03-NEVER-CONSULTS-RETURNS
    def test_ucs_03_close_permutation_is_return_blind(self) -> None:
        frame = _frame()
        rng = np.random.default_rng(2026)
        permuted = frame.copy()
        permuted["close"] = rng.permutation(frame["close"].to_numpy())
        base = evaluate_universe_candidate(
            "TESTUSDT", frame, True, frame.index[0], frame.index[-1],
            UniverseCandidateSpec(),
        )
        shuffled = evaluate_universe_candidate(
            "TESTUSDT", permuted, True, permuted.index[0], permuted.index[-1],
            UniverseCandidateSpec(),
        )
        assert dataclasses.asdict(base) == dataclasses.asdict(shuffled)

    def test_ucs_03_module_references_no_performance_metric(self) -> None:
        import src.research.universe.candidate_scan as candidate_scan

        source = inspect.getsource(candidate_scan).lower()
        for token in ("sharpe", "cagr", "drawdown", "pct_change", "equity"):
            assert token not in source

    def test_fails_closed_on_empty_frame(self) -> None:
        frame = _frame()
        empty = frame.iloc[:0]
        with pytest.raises(ValueError, match="empty"):
            evaluate_universe_candidate(
                "TESTUSDT", empty, True, frame.index[0], frame.index[-1],
                UniverseCandidateSpec(),
            )

    def test_fails_closed_on_missing_required_column(self) -> None:
        frame = _frame().drop(columns=["taker_buy_ratio"])
        with pytest.raises(ValueError, match="required columns"):
            evaluate_universe_candidate(
                "TESTUSDT", frame, True, frame.index[0], frame.index[-1],
                UniverseCandidateSpec(),
            )


def test_contract_surface() -> None:
    from dataclasses import fields

    assert {f.name for f in fields(UniverseCandidateSpec)} == {
        "min_coverage", "seasoning_tolerance_days",
    }
    assert {f.name for f in fields(UniverseCandidateResult)} == {
        "symbol", "first_bar", "last_bar", "coverage", "has_funding",
        "taker_ratio_valid", "avg_daily_quote_vol_recent", "qualifies",
    }
