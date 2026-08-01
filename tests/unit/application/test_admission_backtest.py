from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application import admission_backtest as app
from src.application.admission_backtest import (
    run_technical_library_admission_backtest,
)
from src.research.expert_portfolio.admission_types import (
    TechnicalLibraryAdmissionBacktestRequest,
    admission_proposal_id,
    expert_ids_from_admission_proposal_id,
)
from src.research.expert_portfolio.models import ContextualRouterSpec


def _request(*expert_ids: str) -> TechnicalLibraryAdmissionBacktestRequest:
    return TechnicalLibraryAdmissionBacktestRequest(
        expert_ids=expert_ids,
        router=ContextualRouterSpec("BTCUSDT", 1, 1, 30),
        start="2024-01-01",
        max_workers=1,
    )


def _fake_evidence(
    symbol: str,
    sources: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs,
    signal_delay_bars: int,
) -> dict[str, app._SelectedEvidence]:
    index = pd.date_range("2024-01-01", periods=1000, freq="D", tz="UTC")
    values = np.full(len(index), 0.0002, dtype=float)
    values[0] = np.nan
    trade_index = np.arange(40)
    frames = pd.DataFrame({
        "pnl": np.ones(40),
        "return_pct": np.full(40, 0.01),
        "entry_bar": trade_index,
        "exit_bar": trade_index + 1,
    })
    return {
        source: {
            "returns": pd.Series(values, index=index),
            "trades": frames.assign(expert_id=f"{source}:{symbol}"),
        }
        for source in sources
    }


def test_proposal_backtest_runs_base_and_stress_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LAE-10-PROPOSAL-BACKTEST-NO-MUTATION: execution remains in memory and
    # stress reuses the base target weights.
    monkeypatch.setattr(app, "_selected_symbol_worker", _fake_evidence)
    index = pd.date_range("2024-01-01", periods=1000, freq="D", tz="UTC")
    monkeypatch.setattr(
        app,
        "_build_admission_context",
        lambda router, panel_index, start, end: pd.Series(
            ["up_low_vol"] * len(panel_index), index=panel_index,
        ),
    )
    monkeypatch.setattr(app, "compute_code_hash", lambda _: "c" * 64)
    monkeypatch.setattr(
        app,
        "technical_data_hashes",
        lambda symbol: {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    )
    original_run = app.run_expert_portfolio
    fixed_weight_calls: list[bool] = []

    def spy_run(*args, **kwargs):
        fixed_weight_calls.append(kwargs.get("fixed_weights") is not None)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(app, "run_expert_portfolio", spy_run)
    report = run_technical_library_admission_backtest(_request(
        "technical_macd_histogram_regime_long_v1:BTCUSDT",
        "technical_rsi_trend_pullback_long_v1:ETHUSDT",
    ))

    assert report.status == "COMPLETE"
    assert report.proposal_id.startswith("lae-v1:")
    assert report.execution_workers == 1
    assert report.window_start == str(index[0])
    assert report.window_end == str(index[-1])
    assert report.code_hash == "c" * 64
    assert report.data_hashes == {
        "BTCUSDT": {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
        "ETHUSDT": {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    }
    assert fixed_weight_calls == [False, True]
    assert report.to_report_dict()["promotion"]["status"] in {
        "OBSERVATION_PASS", "HOLDOUT_PASS", "REJECTED",
    }


def test_proposal_backtest_rejects_duplicate_family_before_worker() -> None:
    # LAE-11-PROPOSAL-INTEGRITY: malformed structural proposals fail closed.
    with pytest.raises(ValueError, match="duplicate family"):
        run_technical_library_admission_backtest(_request(
            "technical_macd_histogram_regime_long_v1:BTCUSDT",
            "technical_macd_histogram_regime_short_v1:ETHUSDT",
        ))


def test_proposal_backtest_rejects_duplicate_expert_ids() -> None:
    with pytest.raises(ValueError, match="expert_ids must be unique"):
        TechnicalLibraryAdmissionBacktestRequest(
            expert_ids=(
                "technical_macd_histogram_regime_long_v1:BTCUSDT",
                "technical_macd_histogram_regime_long_v1:BTCUSDT",
            ),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
        )


def test_proposal_backtest_rejects_empty_expert_ids() -> None:
    with pytest.raises(ValueError, match="expert_ids must not be empty"):
        TechnicalLibraryAdmissionBacktestRequest(
            expert_ids=(),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
        )


def test_proposal_id_rejects_non_lexical_ids() -> None:
    with pytest.raises(ValueError, match="lexical"):
        expert_ids_from_admission_proposal_id("lae-v1:b|a")


def test_proposal_id_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="at least one expert"):
        expert_ids_from_admission_proposal_id("lae-v1:")


def test_proposal_id_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="start with"):
        expert_ids_from_admission_proposal_id("proposal:a")


def test_proposal_id_rejects_empty_expert_name() -> None:
    with pytest.raises(ValueError, match="empty ids"):
        admission_proposal_id(("",))


def test_proposal_id_rejects_duplicate_experts() -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        admission_proposal_id(("a", "a"))


def test_proposal_backtest_rejects_holdout_end() -> None:
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        TechnicalLibraryAdmissionBacktestRequest(
            expert_ids=("technical_macd_histogram_regime_long_v1:BTCUSDT",),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
            end="2026-01-01",
        )


def test_proposal_backtest_rejects_non_positive_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        TechnicalLibraryAdmissionBacktestRequest(
            expert_ids=("technical_macd_histogram_regime_long_v1:BTCUSDT",),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
            max_workers=0,
        )


def test_proposal_backtest_rejects_non_positive_initial_equity() -> None:
    with pytest.raises(ValueError, match="initial_equity"):
        TechnicalLibraryAdmissionBacktestRequest(
            expert_ids=("technical_macd_histogram_regime_long_v1:BTCUSDT",),
            router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
            initial_equity=0.0,
        )
