from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.expert_portfolio.evaluation import run_expert_portfolio_evaluation
from src.research.baseline.backtest import BacktestResult
from src.research.expert_portfolio.backtest import ExpertPortfolioBacktestResult
from src.research.expert_portfolio.models import (
    ExpertDefinition,
    ExpertPortfolioEvaluationRequest,
    ExpertPortfolioSpec,
)

_APPLICATION_MODULE = "src.application.research.expert_portfolio.evaluation"


def _spec() -> ExpertPortfolioSpec:
    return ExpertPortfolioSpec(experts=(
        ExpertDefinition("e1", "return_source", "f1", ("S1",), "run_backtest", "hash1"),
        ExpertDefinition("e2", "return_source", "f1", ("S2",), "run_backtest", "hash2"),
    ))


def _concentrated_equity() -> pd.Series:
    """A strictly-positive 3-year daily ledger whose entire growth lands in 2024."""
    idx = pd.date_range("2023-01-01", "2025-12-31", freq="D", tz="UTC")
    values = np.full(len(idx), 10_000.0)
    values[idx.year == 2024] = 20_000.0
    values[idx.year >= 2025] = 20_000.0
    return pd.Series(values, index=idx, name="equity", dtype=np.float64)


def _component_trades(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = 40
    return pd.DataFrame({
        "expert_id": ["e1"] * n,
        "symbol": ["S1"] * n,
        "entry_bar": np.arange(n),
        "exit_bar": np.arange(n) + 1,
        "entry_time": idx[:n],
        "exit_time": idx[1 : n + 1],
        "entry_price": [100.0] * n,
        "exit_price": [101.0] * n,
        "qty": [1.0] * n,
        "reason": ["channel"] * n,
        "pnl": [10.0] * n,
        "return_pct": [0.01] * n,
        "funding_pnl": [0.0] * n,
    })


def _synthetic_result(
    equity: pd.Series,
    panel: pd.DataFrame,
) -> ExpertPortfolioBacktestResult:
    return ExpertPortfolioBacktestResult(
        backtest_result=BacktestResult(
            equity=equity,
            trades=pd.DataFrame(columns=["entry_bar"]),
            signals=pd.DataFrame(),
        ),
        target_weights=pd.DataFrame(
            {"e1": 0.0, "e2": 0.0, "CASH": 1.0}, index=equity.index,
        ),
        allocation_cost=pd.Series(0.0, index=equity.index, name="allocation_cost"),
        component_returns=panel,
    )


def _patch_backend(
    monkeypatch: pytest.MonkeyPatch,
    spec: ExpertPortfolioSpec,
):
    equity = _concentrated_equity()
    panel = pd.DataFrame({"e1": [0.001] * len(equity), "e2": [0.001] * len(equity)}, index=equity.index)
    trades = _component_trades(equity.index)
    result = _synthetic_result(equity, panel)
    stress_fixed_weights: list[object] = []
    stress_delay: list[int] = []
    base_decision_contexts: list[object] = []

    def fake_run(
        component_returns, library, costs, *, initial_equity=10_000.0,
        fixed_weights=None, signal_delay_bars=0, decision_context=None,
    ):
        if fixed_weights is not None:
            stress_fixed_weights.append(fixed_weights)
            stress_delay.append(signal_delay_bars)
        else:
            base_decision_contexts.append(decision_context)
        return result

    from src.research.expert_portfolio.registry import RegisteredExpertLibrary
    from src.research.provenance.registration import RegistrationRecord

    def fake_resolve(_library_id, *, catalog=None, ledger_path=None):
        return RegisteredExpertLibrary(
            library_id="lib-a",
            registration_id="reg-1",
            spec=spec,
            registration=RegistrationRecord(
                registration_id="reg-1",
                library_id="lib-a",
                status="ACTIVE",
                fingerprint={"experts": []},
                registered_at="2026-01-01T00:00:00+00:00",
                record={"registration_id": "reg-1"},
            ),
        )

    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.resolve_registered_library", fake_resolve,
    )
    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.build_component_panel",
        lambda library, start, end, costs, *, signal_delay_bars=0: (panel, trades),
    )
    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_expert_portfolio", fake_run)
    return stress_fixed_weights, stress_delay, base_decision_contexts


def test_fold_failure_remains_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # EP-07: no allocation result can bypass a failing fold gate; promotion is
    # REJECTED without changing any threshold.
    _patch_backend(monkeypatch, _spec())
    report = run_expert_portfolio_evaluation(
        ExpertPortfolioEvaluationRequest(library_id="lib-a", log_run=False),
    )
    assert report.status == "PASS"
    assert report.promotion.status == "REJECTED"
    assert report.fold_distribution.gate_pass is False
    assert report.fold_distribution.max_period_contribution > 0.40


def test_unregistered_library_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(_library_id: str, *, catalog=None, ledger_path=None) -> ExpertPortfolioSpec:
        raise ValueError("library 'nope' is not in the catalog")

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.resolve_registered_library", reject)
    with pytest.raises(ValueError, match="not in the catalog"):
        run_expert_portfolio_evaluation(
            ExpertPortfolioEvaluationRequest(library_id="nope", log_run=False),
        )


def test_stress_reuses_base_target_weights_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    # EP-06: the stress run must receive the base target series and the existing
    # one-bar delay; it must never recompute targets around stressed costs.
    stress_fixed_weights, stress_delay, _ = _patch_backend(monkeypatch, _spec())
    report = run_expert_portfolio_evaluation(
        ExpertPortfolioEvaluationRequest(library_id="lib-a", log_run=False),
    )
    assert len(stress_fixed_weights) == 1
    assert stress_delay == [1]
    base = _synthetic_result(_concentrated_equity(), pd.DataFrame())
    pd.testing.assert_frame_equal(
        stress_fixed_weights[0],  # type: ignore[arg-type]
        base.target_weights,
    )
    assert report.promotion.status == "REJECTED"


def test_log_run_appends_one_provenance_record(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch, _spec())
    records: list[dict[str, object]] = []

    def fake_record(*, library_fingerprint, allocation_cost_total, result, metrics,
                    observation_gate, fold_distribution, stress_gate,
                    holdout_gate=None, promotion=None, **_: object):
        records.append({
            "library_fingerprint": library_fingerprint,
            "allocation_cost_total": allocation_cost_total,
            "metrics": metrics,
            "promotion": promotion,
        })
        return {"git_sha": "abc", "git_dirty": False, "kind": "expert_portfolio"}

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.record_expert_portfolio_run", fake_record)
    report = run_expert_portfolio_evaluation(
        ExpertPortfolioEvaluationRequest(library_id="lib-a", log_run=True),
    )
    assert len(records) == 1
    assert records[0]["promotion"].status == "REJECTED"
    assert records[0]["allocation_cost_total"] == 0.0
    assert report.record == {"git_sha": "abc", "git_dirty": False, "kind": "expert_portfolio"}


def _component_result(idx: pd.DatetimeIndex, expert_id: str) -> BacktestResult:
    equity = pd.Series(10_000.0 * (1.0 + 0.001 * np.arange(len(idx))), index=idx, name="equity")
    trades = pd.DataFrame({
        "entry_bar": [0],
        "exit_bar": [1],
        "entry_price": [100.0],
        "exit_price": [101.0],
        "qty": [1.0],
        "reason": ["channel"],
        "pnl": [1.0],
        "return_pct": [0.01],
        "funding_pnl": [0.0],
    })
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


def test_run_component_dispatch_single_symbol_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.application.research.expert_portfolio.evaluation import _run_component
    from src.research.baseline.backtest import BacktestResult
    from src.research.contracts import CostModel
    from src.research.expert_portfolio.models import ExpertDefinition

    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    expected = BacktestResult(
        equity=pd.Series(10_000.0, index=idx), trades=pd.DataFrame(), signals=pd.DataFrame(),
    )
    calls: list[object] = []

    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.load_ohlcv_4h",
        lambda path, start=None, end=None: frame,
    )
    monkeypatch.setattr(
        "src.research.expert_portfolio.runners.run_backtest",
        lambda df, spec, costs, signal_delay_bars=0: calls.append((df, spec.symbol, costs, signal_delay_bars)) or expected,
    )
    definition = ExpertDefinition("e1", "src", "f1", ("BTCUSDT",), "run_backtest", "hash")
    result = _run_component(definition, None, "2025-12-31", CostModel(), 1)
    assert result is expected
    assert calls == [(frame, "BTCUSDT", CostModel(), 1)]


def test_run_component_dispatch_directional_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.application.research.expert_portfolio.evaluation import _run_component
    from src.research.baseline.backtest import BacktestResult
    from src.research.contracts import CostModel
    from src.research.expert_portfolio.models import ExpertDefinition

    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    funding = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    expected = BacktestResult(
        equity=pd.Series(10_000.0, index=idx), trades=pd.DataFrame(), signals=pd.DataFrame(),
    )
    calls: list[object] = []

    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.load_ohlcv_4h",
        lambda path, start=None, end=None: frame,
    )
    monkeypatch.setattr(f"{_APPLICATION_MODULE}.load_funding_rates", lambda path: funding)
    monkeypatch.setattr(
        "src.research.expert_portfolio.runners.run_directional_backtest",
        lambda df, spec, costs, rates, signal_delay_bars=0: calls.append((df, spec.symbol, rates)) or expected,
    )
    definition = ExpertDefinition(
        "e1", "src", "f1", ("BTCUSDT",), "run_directional_backtest", "hash",
    )
    result = _run_component(definition, None, "2025-12-31", CostModel(), 0)
    assert result is expected
    assert calls[0][0] is frame
    assert calls[0][1] == "BTCUSDT"
    assert calls[0][2] is funding


def test_run_component_rejects_unsupported_runner() -> None:
    from src.application.research.expert_portfolio.evaluation import _run_component
    from src.research.contracts import CostModel
    from src.research.expert_portfolio.models import ExpertDefinition

    definition = ExpertDefinition(
        "e1", "src", "f1", ("A",), "run_pair_residual", "hash",
    )
    with pytest.raises(ValueError, match="not registered"):
        _run_component(definition, None, None, CostModel(), 0)


def test_build_component_panel_builds_common_index_and_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.application.research.expert_portfolio.evaluation import build_component_panel
    from src.research.contracts import CostModel

    idx = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    results = {
        "e1": _component_result(idx, "e1"),
        "e2": _component_result(idx, "e2"),
    }

    def fake_run_component(definition, start, end, costs, signal_delay_bars):
        return results[definition.expert_id]

    monkeypatch.setattr(f"{_APPLICATION_MODULE}._run_component", fake_run_component)
    spec = _spec()
    panel, trades = build_component_panel(spec, None, "2025-12-31", CostModel())
    assert list(panel.columns) == ["e1", "e2"]
    assert len(panel) == 40
    assert panel.index.equals(idx)
    assert len(trades) == 2
    assert "exit_time" in trades.columns
    assert trades["expert_id"].tolist() == ["e1", "e2"]


def test_build_component_panel_fails_closed_on_short_common_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.application.research.expert_portfolio.evaluation import build_component_panel
    from src.common.errors import DataIntegrityError
    from src.research.contracts import CostModel

    idx_a = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    idx_b = pd.date_range("2025-01-01", periods=5, freq="4h", tz="UTC")

    def fake_run_component(definition, start, end, costs, signal_delay_bars):
        return BacktestResult(
            equity=pd.Series(10_000.0, index=idx_a if definition.expert_id == "e1" else idx_b),
            trades=pd.DataFrame(),
            signals=pd.DataFrame(),
        )

    monkeypatch.setattr(f"{_APPLICATION_MODULE}._run_component", fake_run_component)
    with pytest.raises(DataIntegrityError, match="common bars"):
        build_component_panel(_spec(), None, "2025-12-31", CostModel())


def _routed_spec() -> ExpertPortfolioSpec:
    from src.research.expert_portfolio.models import ContextualRouterSpec

    return ExpertPortfolioSpec(
        experts=(
            ExpertDefinition("e1", "return_source", "f1", ("S1",), "run_backtest", "hash1"),
            ExpertDefinition("e2", "return_source", "f1", ("S2",), "run_backtest", "hash2"),
        ),
        router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
    )


def test_build_library_decision_context_is_none_without_router() -> None:
    from src.application.research.expert_portfolio.evaluation import build_library_decision_context

    assert build_library_decision_context(_spec(), _concentrated_equity().index, None, None) is None


def test_build_library_decision_context_requires_exact_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ECR-05: context labels must align exactly to the component panel index;
    # a missing panel timestamp fails closed instead of forward-filling.
    from src.application.research.expert_portfolio.evaluation import build_library_decision_context
    from src.common.errors import DataIntegrityError

    panel_idx = _concentrated_equity().index
    close_idx = panel_idx[: len(panel_idx) - 2]  # two panel timestamps missing
    ohlcv = pd.DataFrame({"close": 100.0}, index=close_idx)
    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.load_ohlcv_4h",
        lambda path, start=None, end=None: ohlcv,
    )
    with pytest.raises(DataIntegrityError, match="align exactly"):
        build_library_decision_context(_routed_spec(), panel_idx, None, None)


def test_build_library_decision_context_rejects_extra_context_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.application.research.expert_portfolio.evaluation import build_library_decision_context
    from src.common.errors import DataIntegrityError

    panel_idx = _concentrated_equity().index
    ohlcv_idx = panel_idx.append(pd.date_range(panel_idx[-1] + pd.Timedelta("4h"), periods=1, freq="4h", tz="UTC"))
    ohlcv = pd.DataFrame({"close": 100.0}, index=ohlcv_idx)
    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.load_ohlcv_4h",
        lambda path, start=None, end=None: ohlcv,
    )
    with pytest.raises(DataIntegrityError, match="align exactly"):
        build_library_decision_context(_routed_spec(), panel_idx, None, None)


def test_router_context_used_once_and_stress_stays_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ECR-05: the base run receives the aligned decision context exactly once;
    # the stress run receives the exact base target frame and performs no fresh
    # contextual selection under stressed costs.
    stress_fixed_weights, stress_delay, base_decision_contexts = _patch_backend(
        monkeypatch, _routed_spec(),
    )
    equity = _concentrated_equity()
    context = pd.Series(["up_low_vol"] * len(equity), index=equity.index)
    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.build_library_decision_context",
        lambda spec, index, start, end: context,
    )
    report = run_expert_portfolio_evaluation(
        ExpertPortfolioEvaluationRequest(library_id="lib-a", log_run=False),
    )
    assert base_decision_contexts == [context]
    assert len(stress_fixed_weights) == 1
    assert stress_delay == [1]
    base = _synthetic_result(equity, pd.DataFrame())
    pd.testing.assert_frame_equal(
        stress_fixed_weights[0],  # type: ignore[arg-type]
        base.target_weights,
    )
    assert report.promotion.status == "REJECTED"
