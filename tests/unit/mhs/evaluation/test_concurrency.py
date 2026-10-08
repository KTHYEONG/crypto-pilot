"""Post-book diagnostic gating and deployment-readiness invariants."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

import src.mhs.evaluation.concurrency as concurrency
from src.common.errors import DataIntegrityError
from src.mhs import statistics
from src.mhs.evaluation.windows import _book_outcome
from src.strategy.horizons import horizon_log_return
from tests.fixtures.mhs_requests import research_baseline
from tests.unit.mhs.test_evaluation_appresearch import _build_book_outcome_args


def test_concurrency_module_present() -> None:
    assert concurrency.__name__ == "src.mhs.evaluation.concurrency"
    assert callable(concurrency._run_books_concurrent)


def test_concurrency_post_book_base_panel_injection() -> None:
    assert "base_panel" in inspect.signature(concurrency._run_post_book_concurrently).parameters


def test_post_diag_deploy_readiness_independent_of_report_only_flags(mhs_market, monkeypatch) -> None:
    args = _build_book_outcome_args(mhs_market)
    blend, _ = _book_outcome(**args)
    monkeypatch.setattr(statistics, "_BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(statistics, "_BOOTSTRAP_MEAN_BLOCK", 24)
    opens = args["opens"]
    signal = horizon_log_return(np.log(opens), 48)
    kwargs = {
        "blend_report": blend, "root": args["root"],
        "execution_symbols": list(opens.columns),
        "minute_grid": pd.date_range(args["start"], args["end"], freq="3min"),
        "eligible": opens.notna(), "opens": opens,
        "bar_funding": args["bar_funding"], "grid_1h": args["grid_1h"],
        "fast": args["spec"],
    }
    off = concurrency._run_post_diag_deploy(
        **kwargs, signal_48h=pd.DataFrame(), request=research_baseline(),
    )
    on = concurrency._run_post_diag_deploy(
        **kwargs, signal_48h=signal,
        request=research_baseline(placebo_diagnostic=True, bootstrap_ci_diagnostic=True),
    )
    assert off[0] is None
    assert off[1] is None
    net = blend.primary.ledger.equity.resample("1h").last().dropna().pct_change().dropna()
    assert on[0] == statistics._bootstrap_ci(net, 20, 24, statistics._BOOTSTRAP_SEED)
    assert on[1] == statistics._placebo_sharpe_percentile(
        signal, opens.notna(), opens, args["bar_funding"], args["grid_1h"],
        args["spec"], blend.primary_naive_sharpe, 500, statistics._BOOTSTRAP_SEED,
    )
    assert off[2:] == on[2:]
    assert off[4] is not None

    kwargs["blend_report"] = dataclasses.replace(blend, primary_naive_sharpe=None)
    assert concurrency._run_post_diag_deploy(
        **kwargs, signal_48h=pd.DataFrame(), request=research_baseline(),
    )[1] is None
    with pytest.raises(DataIntegrityError, match="naive Sharpe for the placebo"):
        concurrency._run_post_diag_deploy(
            **kwargs, signal_48h=signal, request=research_baseline(placebo_diagnostic=True),
        )


class _SynchronousFuture:
    def __init__(self, result: object) -> None:
        self._result = result

    def result(self, timeout: object = None) -> object:
        return self._result


def _micro_books_args(**overrides: object) -> dict[str, object]:
    from src.core.types import BOOK_SPECS

    grid_1h = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    frame = pd.DataFrame(0.01, index=grid_1h, columns=["AAA", "BBB"])
    return {
        "root": "/nonexistent",
        "request": research_baseline(**overrides),
        "n_symbols": 2,
        "grid_1h": grid_1h,
        "fast": BOOK_SPECS["fast_reversal"],
        "slow": BOOK_SPECS["slow_momentum"],
        "fast_grid": grid_1h[::6],
        "slow_grid": grid_1h[::24],
        "w_fast": frame,
        "w_slow": frame,
        "w_fast_execution": frame,
        "w_slow_execution": frame,
        "opens": frame,
        "bar_funding": frame,
        "phase_fast": None,
        "phase_slow": None,
        "phase_blend": None,
        "start": grid_1h[0],
        "end": grid_1h[-1],
        "funding_by_symbol": {},
        "blend_1h": frame,
        "execution_mask": pd.DataFrame(True, index=grid_1h, columns=["AAA", "BBB"]),
        "initial_equity": 1.0,
    }


def test_run_books_concurrent_gates_reference_books(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference-book replay set follows ``reference_books_diagnostic``: the
    default run submits only the blend and returns ``None`` for the standalone
    books; the opted-in run submits all three in declared order."""
    from types import SimpleNamespace

    import src.mhs.evaluation.concurrency as concurrency_mod
    from src.mhs.contracts import MhsResourceMeasurement
    from src.core.resources import _StageRecorder

    submitted: list[str] = []

    def _stub_worker(name: str, *args: object, **kwargs: object) -> tuple:
        measurement = MhsResourceMeasurement(stage=f"replay_{name}", elapsed_ms=0, rss_bytes=1)
        return SimpleNamespace(name=name, failure=None), (measurement,), {}

    class _SynchronousExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _SynchronousExecutor:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def submit(self, fn: object, *args: object, **kwargs: object):
            submitted.append(args[0])
            return _SynchronousFuture(fn(*args, **kwargs))

    monkeypatch.setattr(concurrency_mod, "ProcessPoolExecutor", _SynchronousExecutor, raising=False)
    monkeypatch.setattr(concurrency_mod.windows, "_book_outcome_worker", _stub_worker)

    off_recorder = _StageRecorder(log_run=False)
    fast, slow, blend, traces, members = concurrency_mod._run_books_concurrent(
        **_micro_books_args(), telemetry=off_recorder,
    )
    assert submitted == ["blend"]
    assert fast is None
    assert slow is None
    assert blend is not None
    assert blend.name == "blend"
    assert traces == {}
    assert members is None
    assert [m.stage for m in off_recorder.records] == ["replay_blend"]

    submitted.clear()
    on_recorder = _StageRecorder(log_run=False)
    fast, slow, blend, traces, members = concurrency_mod._run_books_concurrent(
        **_micro_books_args(reference_books_diagnostic=True), telemetry=on_recorder,
    )
    assert submitted == ["fast_reversal", "slow_momentum", "blend"]
    assert fast is not None
    assert fast.name == "fast_reversal"
    assert slow is not None
    assert slow.name == "slow_momentum"
    assert blend is not None
    assert blend.name == "blend"
    assert [m.stage for m in on_recorder.records] == [
        "replay_fast_reversal", "replay_slow_momentum", "replay_blend",
    ]
