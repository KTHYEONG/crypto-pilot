"""Wiring smoke test for src.mhs.pipeline.stages.committee.build_committee.

Verifies the S4 stage reaches ``phase_diagnostics``/``active_blend_book_and_grid``
through the ``stage_services`` seam after the P4 refactor (previously private
``evaluation.`` attribute lookups). Every collaborator besides the seam is
monkeypatched so this exercises the non-committee-capital branch as a fast,
deterministic unit test; the full numeric behaviour is covered by the
evaluation/golden-identity suites.
"""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import dataclasses

import pandas as pd
import pytest

import src.mhs.pipeline.stages.committee as committee_stage
import src.mhs.evaluation.books as books_mod
import src.mhs.evaluation.committee as committee_mod
import src.mhs.evaluation.diagnostics as diagnostics_mod
from src.strategy.features import FeatureAdmission
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


class _FakeBand:
    def __init__(self, sign: int) -> None:
        self.sign = sign


class _FakeSpec:
    def __init__(self, horizon_hours: int) -> None:
        self.band = _FakeBand(1)
        self.horizon_hours = horizon_hours
        self.min_symbols = 1


def _bare_context(**overrides) -> PipelineContext:
    frame = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    params = {
        "discovery_gate": False,
        "trend_sleeve": False,
        "trend_efficiency_overlay": False,
        "phase_diagnostic": True,
        "signal_48h_diagnostic": True,
    }
    params.update(overrides)
    ctx = PipelineContext(
        config=research_baseline(**params),
        resolved_end=None,
        start=_GRID[0],
        end=_GRID[-1],
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=_GRID,
        close=frame,
        opens=frame,
        quote_vol=frame,
        taker_buy_quote=None,
        symbols=_SYMS,
    )
    ctx.log_close = frame
    ctx.eligible = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.bar_funding = pd.DataFrame(0.0001, index=_GRID, columns=_SYMS)
    ctx.fast_grid = _GRID
    ctx.slow_grid = _GRID
    ctx.fast = _FakeSpec(48)
    ctx.slow = _FakeSpec(168)
    ctx.w_fast_1h = frame
    ctx.w_slow_1h = frame
    ctx.execution_mask = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    return ctx


def test_build_committee_reaches_seam_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_phase_diagnostics(*_a: object, **_k: object) -> str:
        calls.append("_phase_diagnostics")
        return "phase-result"

    def _fake_active_blend_book_and_grid(fast, slow, fast_grid, slow_grid):
        calls.append("_active_blend_book_and_grid")
        return slow, slow_grid

    monkeypatch.setattr(committee_stage, "_phase_diagnostics", _fake_phase_diagnostics, raising=False)
    monkeypatch.setattr(diagnostics_mod, "_phase_diagnostics", _fake_phase_diagnostics, raising=False)
    monkeypatch.setattr(
        committee_stage, "_active_blend_book_and_grid", _fake_active_blend_book_and_grid, raising=False
    )
    monkeypatch.setattr(books_mod, "_active_blend_book_and_grid", _fake_active_blend_book_and_grid, raising=False)
    monkeypatch.setattr(
        committee_stage, "realized_vol",
        lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "horizon_log_return",
        lambda log_close, horizon: pd.DataFrame(0.0, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "efficiency_ratio",
        lambda log_close, horizon: pd.DataFrame(0.5, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage._statistics, "_xs_rank_ic", lambda *_a, **_k: {"mean_ic": 0.0},
    )
    monkeypatch.setattr(
        committee_stage._statistics, "_date_clustered_ols", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        committee_stage._scaling, "_regime_cash_scale",
        lambda vol_mean: pd.Series(1.0, index=vol_mean.index),
    )

    ctx = _bare_context()
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))

    assert calls == [
        "_active_blend_book_and_grid", "_phase_diagnostics",
    ]
    assert ctx.phase_fast is None
    assert ctx.phase_slow is None
    assert ctx.phase_blend == "phase-result"
    assert ctx.committee_execution_book is None


def test_build_committee_broadcasts_top_level_evidence_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO_MHS_COMMITTEE_FOLD_BOUNDARY_WEIGHTS (INV-WALK-FORWARD-INDEPENDENCE):
    # under committee_capital=True with evidence weighting on, every fold index
    # receives its OWN boundary-fitted member-weight mix -- the top-level mix
    # is never replicated across folds.
    captured: dict[str, object] = {}

    def _fake_by_boundary(
        *_a: object, **_k: object
    ) -> tuple[dict[str, FeatureAdmission], dict[str, dict[str, float]]]:
        weights = {
            "top_level": {"member_a": 0.7, "member_b": 0.3},
            "fold_0": {"member_a": 0.9, "member_b": 0.1},
            "fold_1": {"member_a": 0.5, "member_b": 0.5},
        }
        train_ends = _k.get("train_ends")
        if train_ends is None and len(_a) >= 7:
            train_ends = _a[6]
        assert isinstance(train_ends, dict)
        admission = {
            label: FeatureAdmission(
                cutoff=train_ends[label], admitted=tuple(w.keys()),
            )
            for label, w in weights.items()
        }
        return admission, weights

    def _fake_committee_execution_book(*_a: object, **_k: object) -> pd.DataFrame:
        captured["member_weights"] = _k.get("member_weights")
        captured["coverage_cutoff"] = _k.get("coverage_cutoff")
        captured["admission"] = _k.get("admission")
        return pd.DataFrame(0.0, index=_GRID, columns=_SYMS)

    monkeypatch.setattr(
        committee_stage, "_committee_boundary_admission_and_weights", _fake_by_boundary, raising=False
    )
    monkeypatch.setattr(committee_mod, "_committee_boundary_admission_and_weights", _fake_by_boundary, raising=False)
    class _FoldStub:
        def __init__(self, train_end: pd.Timestamp) -> None:
            self.train_end = train_end

    monkeypatch.setattr(
        committee_stage, "resolved_anchored_folds",
        lambda _cfg: (
            _FoldStub(pd.Timestamp("2022-01-01", tz="UTC")),
            _FoldStub(pd.Timestamp("2023-01-01", tz="UTC")),
        ),
    )
    monkeypatch.setattr(
        committee_stage, "_committee_execution_book", _fake_committee_execution_book, raising=False
    )
    monkeypatch.setattr(committee_mod, "_committee_execution_book", _fake_committee_execution_book, raising=False)
    monkeypatch.setattr(committee_stage, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(diagnostics_mod, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(
        committee_stage, "_active_blend_book_and_grid",
        lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False
    )
    monkeypatch.setattr(books_mod, "_active_blend_book_and_grid", lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False)
    monkeypatch.setattr(
        committee_stage, "realized_vol",
        lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "horizon_log_return",
        lambda log_close, horizon: pd.DataFrame(0.0, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "efficiency_ratio",
        lambda log_close, horizon: pd.DataFrame(0.5, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(committee_stage._statistics, "_xs_rank_ic", lambda *_a, **_k: {"mean_ic": 0.0})
    monkeypatch.setattr(committee_stage._statistics, "_date_clustered_ols", lambda *_a, **_k: {})
    monkeypatch.setattr(
        committee_stage._scaling, "_regime_cash_scale",
        lambda vol_mean: pd.Series(1.0, index=vol_mean.index),
    )
    monkeypatch.setattr(
        committee_stage, "funding_carry_execution_book", lambda *_a, **_k: None,
    )

    ctx = _bare_context()
    ctx.config = dataclasses.replace(
        ctx.config, committee_capital=True, committee_evidence_weighting=True,
    )
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))

    assert ctx._fold_committee_weights == {
        0: {"member_a": 0.9, "member_b": 0.1},
        1: {"member_a": 0.5, "member_b": 0.5},
    }
    assert captured["member_weights"] == {"member_a": 0.7, "member_b": 0.3}
    # SCENARIO_MHS_COMMITTEE_STAGE_THREADS_COVERAGE_CUTOFF: the deployed book
    # must be admitted under the SAME boundary that fit its member_weights.
    assert captured["admission"].cutoff == committee_stage.COMMITTEE_OOS_START  # type: ignore[union-attr]
    assert captured["coverage_cutoff"] is None
    assert list(captured["admission"].admitted) == ["member_a", "member_b"]  # type: ignore[union-attr]
    # Fold admissions mirror the per-boundary admission (I-FOLD-ADMISSION-PIT).
    assert ctx._fold_committee_admission is not None
    assert ctx._fold_committee_admission[0].cutoff == pd.Timestamp("2022-01-01", tz="UTC")
    assert ctx._fold_committee_admission[1].cutoff == pd.Timestamp("2023-01-01", tz="UTC")
    assert tuple(ctx._fold_committee_admission[0].admitted) == ("member_a", "member_b")


def test_build_committee_threads_beta_neutralize(monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO_BUILD_COMMITTEE_STAGE_THREADS_BETA_NEUTRALIZE: under
    # committee_capital=True the stage computes a causal beta DataFrame and
    # threads it into _committee_execution_book only when beta_neutralize is
    # on; the default (False) passes beta=None (byte-identical default path).
    captured: dict[str, object] = {}

    def _fake_by_boundary(
        *_a: object, **_k: object
    ) -> tuple[dict[str, FeatureAdmission], dict[str, dict[str, float]]]:
        weights = {
            "top_level": {"member_a": 0.7, "member_b": 0.3},
            "fold_0": {"member_a": 0.7, "member_b": 0.3},
        }
        train_ends = _k.get("train_ends")
        if train_ends is None and len(_a) >= 7:
            train_ends = _a[6]
        assert isinstance(train_ends, dict)
        admission = {
            label: FeatureAdmission(
                cutoff=train_ends[label], admitted=tuple(w.keys()),
            )
            for label, w in weights.items()
        }
        return admission, weights

    def _fake_committee_execution_book(*_a: object, **_k: object) -> pd.DataFrame:
        captured["beta"] = _k.get("beta")
        captured["admission"] = _k.get("admission")
        captured["coverage_cutoff"] = _k.get("coverage_cutoff")
        return pd.DataFrame(0.0, index=_GRID, columns=_SYMS)

    class _FoldStub:
        def __init__(self, train_end: pd.Timestamp) -> None:
            self.train_end = train_end

    monkeypatch.setattr(
        committee_stage, "_committee_boundary_admission_and_weights", _fake_by_boundary, raising=False
    )
    monkeypatch.setattr(committee_mod, "_committee_boundary_admission_and_weights", _fake_by_boundary, raising=False)
    monkeypatch.setattr(
        committee_stage, "resolved_anchored_folds",
        lambda _cfg: (_FoldStub(pd.Timestamp("2022-01-01", tz="UTC")),),
    )
    monkeypatch.setattr(
        committee_stage, "_committee_execution_book", _fake_committee_execution_book, raising=False
    )
    monkeypatch.setattr(committee_mod, "_committee_execution_book", _fake_committee_execution_book, raising=False)
    monkeypatch.setattr(committee_stage, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(diagnostics_mod, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(
        committee_stage, "_active_blend_book_and_grid",
        lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False
    )
    monkeypatch.setattr(books_mod, "_active_blend_book_and_grid", lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False)
    monkeypatch.setattr(
        committee_stage, "realized_vol",
        lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "horizon_log_return",
        lambda log_close, horizon: pd.DataFrame(0.0, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "efficiency_ratio",
        lambda log_close, horizon: pd.DataFrame(0.5, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(committee_stage._statistics, "_xs_rank_ic", lambda *_a, **_k: {"mean_ic": 0.0})
    monkeypatch.setattr(committee_stage._statistics, "_date_clustered_ols", lambda *_a, **_k: {})
    monkeypatch.setattr(
        committee_stage._scaling, "_regime_cash_scale",
        lambda vol_mean: pd.Series(1.0, index=vol_mean.index),
    )
    monkeypatch.setattr(
        committee_stage, "funding_carry_execution_book", lambda *_a, **_k: None,
    )

    ctx = _bare_context()
    ctx.config = dataclasses.replace(ctx.config, committee_capital=True)
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))
    assert captured["beta"] is None

    ctx_on = _bare_context()
    ctx_on.config = dataclasses.replace(
        ctx_on.config, committee_capital=True, beta_neutralize=True,
    )
    committee_stage.build_committee(ctx_on, StageTelemetry(log_run=False))
    assert isinstance(captured["beta"], pd.DataFrame)
    # Admission is always threaded, even without evidence weighting.
    assert captured["admission"].cutoff == committee_stage.COMMITTEE_OOS_START  # type: ignore[union-attr]
    assert captured["coverage_cutoff"] is None
    assert ctx_on._fold_committee_admission is not None
    assert ctx_on._fold_committee_admission[0].cutoff == pd.Timestamp("2022-01-01", tz="UTC")


def test_fold_committee_uses_its_own_boundary_weights(monkeypatch) -> None:
    import src.mhs.pipeline.stages.committee as stage
    weights = {'top_level': {'a': 0.9}, 'fold_0': {'a': 0.1}, 'fold_1': {'a': 0.2}}
    folds = [type('F', (), {'train_end': i})() for i in (1, 2)]
    mapped = stage._fold_weights_from_boundaries(weights, folds)
    assert mapped == {0: {'a': 0.1}, 1: {'a': 0.2}}
    assert mapped[0] is not weights['top_level']


def _numeric_context(**overrides) -> PipelineContext:
    import numpy as np

    from src.core.types import BOOK_SPECS

    ctx = _bare_context(**overrides)
    grid = pd.date_range("2021-01-01", periods=400, freq="1h", tz="UTC")
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0001, 0.01, (400, 10)), axis=0)),
        index=grid, columns=[f"S{i}" for i in range(10)],
    )
    ctx.grid_1h = grid
    ctx.log_close = np.log(frame)
    ctx.opens = frame
    ctx.eligible = frame.notna()
    ctx.execution_mask = frame.notna()
    ctx.bar_funding = frame * 0.0
    ctx.fast = BOOK_SPECS["fast_reversal"]
    ctx.slow = BOOK_SPECS["slow_momentum"]
    ctx.fast_grid = grid[::ctx.fast.step_hours]
    ctx.slow_grid = grid[::ctx.slow.step_hours]
    ctx.w_fast_1h = frame * 0.01
    ctx.w_slow_1h = frame * 0.02
    return ctx


def test_default_run_computes_no_report_only_panel_diagnostics() -> None:
    ctx_off = _numeric_context(
        phase_diagnostic=False, signal_48h_diagnostic=False, placebo_diagnostic=False,
    )
    committee_stage.build_committee(ctx_off, StageTelemetry(log_run=False))
    assert ctx_off.phase_fast is None
    assert ctx_off.phase_slow is None
    assert ctx_off.phase_blend is None
    assert ctx_off.signal_48h.empty
    assert ctx_off.xs_ic == {}
    assert ctx_off.regression == {}
    assert ctx_off.horizon_diagnostics == {}
    ctx_on = _numeric_context(phase_diagnostic=True, signal_48h_diagnostic=True)
    committee_stage.build_committee(ctx_on, StageTelemetry(log_run=False))
    assert ctx_off.blend_1h.equals(ctx_on.blend_1h)
    assert ctx_off.committee_execution_book == ctx_on.committee_execution_book
    assert (ctx_off.regime_scale == ctx_on.regime_scale).all()


def test_opt_in_reproduces_panel_diagnostics() -> None:
    from src.mhs.evidence import PhaseDiagnosticResult

    ctx = _numeric_context(phase_diagnostic=True, signal_48h_diagnostic=True, reference_books_diagnostic=True)
    log_close = ctx.log_close
    signal = committee_stage.horizon_log_return(log_close, 48)
    expected_phases = [
        diagnostics_mod._phase_diagnostics(
            log_close, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h, spec,
        ) for spec in (ctx.fast, ctx.slow, ctx.slow)
    ]
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))
    for phase, expected in zip((ctx.phase_fast, ctx.phase_slow, ctx.phase_blend), expected_phases, strict=True):
        assert isinstance(phase, PhaseDiagnosticResult)
        assert phase == expected
    pd.testing.assert_frame_equal(ctx.signal_48h, signal)
    assert ctx.xs_ic == committee_stage._statistics._xs_rank_ic(signal, ctx.opens, forward_bars=48)
    assert ctx.regression == committee_stage._statistics._date_clustered_ols(ctx.opens, signal, forward_bars=48)
    assert set(ctx.horizon_diagnostics) == {"realized_vol_48h_mean", "efficiency_ratio_48h_mean"}


def test_placebo_alone_materializes_48h_signal_only() -> None:
    ctx = _numeric_context(phase_diagnostic=False, signal_48h_diagnostic=False, placebo_diagnostic=True)
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))
    assert not ctx.signal_48h.empty
    assert ctx.xs_ic == {}


def test_reference_book_phases_follow_the_replay_set() -> None:
    ctx = _numeric_context(
        phase_diagnostic=True, reference_books_diagnostic=False, committee_member_attribution=False,
    )
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))
    assert ctx.phase_blend is not None
    assert ctx.phase_fast is None
    assert ctx.phase_slow is None

    ctx_attr = _numeric_context(
        phase_diagnostic=True, reference_books_diagnostic=False, committee_member_attribution=True,
    )
    committee_stage.build_committee(ctx_attr, StageTelemetry(log_run=False))
    assert ctx_attr.phase_fast is None
    assert ctx_attr.phase_slow is not None

    ctx_ref = _numeric_context(phase_diagnostic=True, reference_books_diagnostic=True)
    ref_frames = (
        ctx_ref.log_close, ctx_ref.eligible, ctx_ref.opens, ctx_ref.bar_funding, ctx_ref.grid_1h,
        ctx_ref.fast, ctx_ref.slow, ctx_ref.fast_grid, ctx_ref.slow_grid,
    )
    committee_stage.build_committee(ctx_ref, StageTelemetry(log_run=False))
    assert ctx_ref.phase_blend is not None
    assert ctx_ref.phase_fast is not None
    assert ctx_ref.phase_slow is not None
    (log_close, eligible, opens, bar_funding, grid_1h, fast, slow, fast_grid, slow_grid) = ref_frames
    assert ctx_ref.phase_blend == diagnostics_mod._phase_diagnostics(
        log_close, eligible, opens, bar_funding, grid_1h,
        books_mod._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)[0],
    )


def _committee_stage_harness(
    monkeypatch: pytest.MonkeyPatch,
    by_boundary: object,
    train_ends: tuple[pd.Timestamp, ...],
    *,
    evidence_weighting: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    """Install the committee-stage seams for admission wiring tests.

    Returns ``(captured_book_kwargs, captured_boundary_kwargs)``; ``by_boundary``
    is the stubbed ``_committee_boundary_admission_and_weights`` callable.
    """
    captured_book: dict[str, object] = {}
    captured_boundary: dict[str, object] = {}

    def _fake_committee_execution_book(*_a: object, **_k: object) -> pd.DataFrame:
        captured_book.update(_k)
        return pd.DataFrame(0.0, index=_GRID, columns=_SYMS)

    class _FoldStub:
        def __init__(self, train_end: pd.Timestamp) -> None:
            self.train_end = train_end

    monkeypatch.setattr(
        committee_stage, "_committee_boundary_admission_and_weights", by_boundary, raising=False
    )
    monkeypatch.setattr(
        committee_mod, "_committee_boundary_admission_and_weights", by_boundary, raising=False
    )
    monkeypatch.setattr(
        committee_stage, "resolved_anchored_folds",
        lambda _cfg: tuple(_FoldStub(end) for end in train_ends),
    )
    monkeypatch.setattr(
        committee_stage, "_committee_execution_book", _fake_committee_execution_book, raising=False
    )
    monkeypatch.setattr(
        committee_mod, "_committee_execution_book", _fake_committee_execution_book, raising=False
    )
    monkeypatch.setattr(committee_stage, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(diagnostics_mod, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(
        committee_stage, "_active_blend_book_and_grid",
        lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False
    )
    monkeypatch.setattr(books_mod, "_active_blend_book_and_grid", lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False)
    monkeypatch.setattr(
        committee_stage, "realized_vol",
        lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "horizon_log_return",
        lambda log_close, horizon: pd.DataFrame(0.0, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "efficiency_ratio",
        lambda log_close, horizon: pd.DataFrame(0.5, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(committee_stage._statistics, "_xs_rank_ic", lambda *_a, **_k: {"mean_ic": 0.0})
    monkeypatch.setattr(committee_stage._statistics, "_date_clustered_ols", lambda *_a, **_k: {})
    monkeypatch.setattr(
        committee_stage._scaling, "_regime_cash_scale",
        lambda vol_mean: pd.Series(1.0, index=vol_mean.index),
    )
    monkeypatch.setattr(
        committee_stage, "funding_carry_execution_book", lambda *_a, **_k: None,
    )

    ctx = _bare_context()
    ctx.config = dataclasses.replace(
        ctx.config, committee_capital=True, committee_evidence_weighting=evidence_weighting,
    )
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))
    captured_boundary["ctx"] = ctx
    return captured_book, captured_boundary


def test_fold_admission_populated_without_evidence_weighting(monkeypatch: pytest.MonkeyPatch) -> None:
    # I-FOLD-ADMISSION-PIT: admission is computed whenever committee_capital is
    # on, with or without evidence weighting.
    train_ends = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"))

    def _fake_by_boundary(*_a: object, **_k: object) -> tuple[dict[str, FeatureAdmission], dict[str, dict[str, float]]]:
        received = _k.get("train_ends", _a[6] if len(_a) >= 7 else None)
        assert _k.get("evidence_weighting") is False
        assert set(received) == {"top_level", "fold_0", "fold_1"}
        admission = {
            label: FeatureAdmission(cutoff=received[label], admitted=("member_a", "member_b"))
            for label in received
        }
        return admission, {}

    captured_book, captured = _committee_stage_harness(
        monkeypatch, _fake_by_boundary, train_ends, evidence_weighting=False,
    )
    ctx = captured["ctx"]
    assert ctx._fold_committee_admission is not None
    assert set(ctx._fold_committee_admission) == {0, 1}
    for i, end in enumerate(train_ends):
        assert ctx._fold_committee_admission[i].cutoff == end
        assert tuple(ctx._fold_committee_admission[i].admitted) == ("member_a", "member_b")
    assert ctx._fold_committee_weights is None
    assert captured_book["member_weights"] is None
    assert captured_book["admission"].cutoff == committee_stage.COMMITTEE_OOS_START  # type: ignore[union-attr]
    assert captured_book.get("coverage_cutoff") is None


def test_fold_weights_and_admission_come_from_same_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    # I-COVERAGE-PIT parity: with evidence weighting the admission is read from
    # the same build that fits the weights, so the two can never disagree.
    train_ends = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"))
    per_label = {
        "top_level": ("member_a", "member_b"),
        "fold_0": ("member_a",),
        "fold_1": ("member_a", "member_b"),
    }

    def _fake_by_boundary(*_a: object, **_k: object) -> tuple[dict[str, FeatureAdmission], dict[str, dict[str, float]]]:
        received = _k.get("train_ends", _a[6] if len(_a) >= 7 else None)
        assert _k.get("evidence_weighting") is True
        admission = {
            label: FeatureAdmission(cutoff=received[label], admitted=per_label[label])
            for label in received
        }
        weights = {label: {name: 1.0 / len(members) for name in members} for label, members in per_label.items()}
        return admission, weights

    captured_book, captured = _committee_stage_harness(
        monkeypatch, _fake_by_boundary, train_ends, evidence_weighting=True,
    )
    ctx = captured["ctx"]
    assert ctx._fold_committee_admission is not None
    assert ctx._fold_committee_weights is not None
    for i, end in enumerate(train_ends):
        assert ctx._fold_committee_admission[i].cutoff == end
        assert set(ctx._fold_committee_weights[i]) == set(ctx._fold_committee_admission[i].admitted)
    assert set(ctx._fold_committee_weights[0]) == {"member_a"}
    assert set(ctx._fold_committee_weights[1]) == {"member_a", "member_b"}
    assert captured_book["admission"].cutoff == committee_stage.COMMITTEE_OOS_START  # type: ignore[union-attr]
    assert set(captured_book["member_weights"]) == set(captured_book["admission"].admitted)  # type: ignore[union-attr]
    assert captured_book.get("coverage_cutoff") is None


def test_non_committee_run_has_no_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without committee capital no boundary admission is computed at all.
    def _forbidden(*_a: object, **_k: object):
        raise AssertionError("_committee_boundary_admission_and_weights must not run")

    monkeypatch.setattr(
        committee_stage, "_committee_boundary_admission_and_weights", _forbidden, raising=False
    )
    monkeypatch.setattr(
        committee_mod, "_committee_boundary_admission_and_weights", _forbidden, raising=False
    )
    monkeypatch.setattr(committee_stage, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(diagnostics_mod, "_phase_diagnostics", lambda *_a, **_k: "phase-result", raising=False)
    monkeypatch.setattr(
        committee_stage, "_active_blend_book_and_grid",
        lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False
    )
    monkeypatch.setattr(books_mod, "_active_blend_book_and_grid", lambda fast, slow, fast_grid, slow_grid: (slow, slow_grid), raising=False)
    monkeypatch.setattr(
        committee_stage, "realized_vol",
        lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "horizon_log_return",
        lambda log_close, horizon: pd.DataFrame(0.0, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "efficiency_ratio",
        lambda log_close, horizon: pd.DataFrame(0.5, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(committee_stage._statistics, "_xs_rank_ic", lambda *_a, **_k: {"mean_ic": 0.0})
    monkeypatch.setattr(committee_stage._statistics, "_date_clustered_ols", lambda *_a, **_k: {})
    monkeypatch.setattr(
        committee_stage._scaling, "_regime_cash_scale",
        lambda vol_mean: pd.Series(1.0, index=vol_mean.index),
    )

    ctx = _bare_context()
    assert ctx.config.committee_capital is False
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))
    assert ctx._fold_committee_admission is None
