"""Wiring smoke test for src.mhs.pipeline.stages.fold.run_folds.

Verifies the S7 stage reaches ``run_post_book_concurrently``/
``guard_stage_or_breach``/``fold_blend_parity``/``fold_growth_concentration``/
``load_feature_panels``/``committee_diagnostic`` through the
``stage_services`` seam after the P4 refactor (previously private
``evaluation.`` attribute lookups).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import src.mhs.pipeline.stages.fold as fold_stage
from src.application.research.mhs.evaluation import DataIntegrityError
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


class _ExposureFoldStub:
    def __init__(self, vs: pd.Timestamp, ve: pd.Timestamp) -> None:
        self.train_end = vs
        self.validation_start = vs
        self.validation_end = ve


class _RecordingRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, stage: str, **_kwargs: object) -> None:
        self.calls.append(stage)


def _bare_context(*, committee_book: bool = False) -> PipelineContext:
    frame = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    ctx = PipelineContext(
        config=dataclasses.replace(
            MhsRunConfig(), multi_feature_book=False, committee_book=committee_book,
        ),
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
    ctx.recorder = _RecordingRecorder()
    ctx.blend_report = None
    ctx.execution_symbols = _SYMS
    ctx.minute_grid = _GRID
    ctx.signal_48h = frame
    ctx.eligible = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.bar_funding = frame
    ctx.fast = "fast-spec"
    ctx.fold_funding = {}
    ctx.initial_equity = 1.0
    ctx.fold_slow_horizons = {}
    ctx.fold_fast_horizons = {}
    ctx.fold_funding_carry = {}
    ctx._fold_committee_weights = None
    ctx.blend_traces = {}
    ctx.book_reasons = ()
    ctx.aligned_symbols = _SYMS
    ctx.execution_mask = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.funding_window = {}
    return ctx


def test_run_folds_reaches_seam_functions_default_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_run_post_book_concurrently(*_a: object, **_k: object):
        calls.append("_run_post_book_concurrently")
        return (None, None, {}, {}, [], "deployment-stub")

    def _fake_guard_stage_or_breach(*_a: object, **_k: object) -> None:
        calls.append("_guard_stage_or_breach")
        return None

    def _fake_fold_blend_parity(*_a: object, **_k: object) -> tuple[object, tuple]:
        calls.append("_fold_blend_parity")
        return ("parity-stub", ())

    def _fake_fold_growth_concentration(*_a: object, **_k: object) -> tuple[object, tuple]:
        calls.append("_fold_growth_concentration")
        return ("concentration-stub", ())

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("committee/multi-feature diagnostics must not run when both flags are off")

    monkeypatch.setattr(fold_stage, "_run_post_book_concurrently", _fake_run_post_book_concurrently)
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", _fake_guard_stage_or_breach)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", _fake_fold_blend_parity)
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", _fake_fold_growth_concentration)
    monkeypatch.setattr(fold_stage, "_load_feature_panels", _boom)
    monkeypatch.setattr(fold_stage, "_committee_diagnostic", _boom)
    monkeypatch.setattr(fold_stage, "_multi_feature_diagnostic", _boom)
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: (None, None, None),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=False)
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))

    assert calls == [
        "_run_post_book_concurrently", "_guard_stage_or_breach",
        "_fold_blend_parity",
        # 관측치 선계산(보정 전)과 보정 임계값 최종 판정으로 2회 호출된다.
        "_fold_growth_concentration", "_fold_growth_concentration",
    ]
    assert ctx.folds == ()
    assert ctx.fold_blend_parity == "parity-stub"
    assert ctx.fold_growth_concentration == "concentration-stub"
    assert ctx.deployment is not None


def test_run_folds_resolves_boundary_growth_budget_vols(monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO_MHS_FOLD_GROWTH_BUDGET_PROPAGATION_06 (stage wiring): under
    # growth_budget mode with a pre-vol-target reference ledger, run_folds
    # builds the boundary map from the top-level reference and forwards only
    # the fold-indexed float mapping to the fold pool.
    # SCENARIO_MHS_GROWTH_BUDGET_STILL_BYTE_IDENTICAL: this pre-existing test
    # must keep passing unmodified after the constant_risk blend-slice change
    # -- the new fold_blend_exposure_scale forwarding is keyword-only trailing.
    captured: dict[str, object] = {}

    def _fake_by_boundary(_ref, _envelope, train_ends):
        captured["train_ends"] = dict(train_ends)
        return {f"fold_{i}": 0.30 + i / 100 for i in range(len(train_ends) - 1)} | {
            "top_level": 0.29,
        }

    def _fake_run_post_book_concurrently(*args: object, **kwargs: object):
        # 위치 인자 순서: ..., fold_committee_weights, fold_growth_budget_target_vol,
        # exposure_warmup_returns (I-WARM 워밍업 Series가 마지막).
        captured["forwarded"] = kwargs.get("fold_growth_budget_target_vol", args[-2])
        captured["warmup"] = kwargs.get("exposure_warmup_returns", args[-1])
        return (None, None, {}, {}, [], None)

    class _FoldStub:
        train_end = pd.Timestamp("2022-01-01", tz="UTC")

    monkeypatch.setattr(fold_stage._scaling, "_growth_budget_target_vol_by_boundary", _fake_by_boundary)
    monkeypatch.setattr(fold_stage, "phase_1_anchored_purged_folds", lambda: (_FoldStub(),))
    monkeypatch.setattr(fold_stage, "_run_post_book_concurrently", _fake_run_post_book_concurrently)
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", lambda *_a, **_k: None)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: (None, None, None),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=False)
    ctx.config = dataclasses.replace(ctx.config, pnl_vol_target_mode="growth_budget")
    ctx.blend_report = type(
        "_BlendStub", (),
        {
            "pre_vol_target_reference": type(
                "_RefStub", (), {"ledger": type("_LedgerStub", (), {
                    "equity": pd.Series([1.0, 1.1, 1.2], index=_GRID),
                })()},
            ),
            "primary_max_drawdown": -0.1,
            "primary": None,
        },
    )()
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))

    assert set(captured["train_ends"]) == {"top_level", "fold_0"}
    assert ctx._fold_growth_budget_target_vol is not None
    assert captured["forwarded"] == {0: 0.30}
    assert ctx._fold_growth_budget_target_vol == {0: 0.30}

def test_run_folds_slices_blend_exposure_scale_for_constant_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO_MHS_FOLD_SLICES_BLEND_EXPOSURE_SCALE: under constant_risk the
    # stage no longer resolves or broadcasts any target vol -- it slices the
    # blend book's already-deployed exposure_scale series onto each fold's
    # [validation_start, validation_end] window and forwards that mapping
    # verbatim to the fold pool (I-SCALE-IS-DEPLOYED-OVERLAY).
    captured: dict[str, object] = {}

    def _must_not_resolve(*_a: object, **_k: object) -> float:
        raise AssertionError("constant_risk folds must reuse the blend exposure scale, not re-solve any target vol")

    def _fake_run_post_book_concurrently(*args: object, **kwargs: object):
        captured["forwarded"] = kwargs.get("fold_blend_exposure_scale")
        return (None, None, {}, {}, [], None)

    folds = (
        _ExposureFoldStub(pd.Timestamp("2021-06-01", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC")),
        _ExposureFoldStub(pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-06-30", tz="UTC")),
    )
    idx = pd.date_range("2021-01-01", "2025-12-31", freq="D", tz="UTC")
    exposure = pd.Series(idx.dayofyear.astype(float), index=idx)

    monkeypatch.setattr(fold_stage._scaling, "_constant_risk_target_vol", _must_not_resolve)
    monkeypatch.setattr(
        fold_stage._scaling, "_constant_risk_target_vol_by_boundary", _must_not_resolve,
    )
    monkeypatch.setattr(fold_stage, "phase_1_anchored_purged_folds", lambda: folds)
    monkeypatch.setattr(fold_stage, "_run_post_book_concurrently", _fake_run_post_book_concurrently)
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", lambda *_a, **_k: None)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: (None, None, None),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=False)
    ctx.config = dataclasses.replace(ctx.config, pnl_vol_target_mode="constant_risk")
    ctx.blend_report = type(
        "_BlendStub", (),
        {
            "pre_vol_target_reference": type(
                "_RefStub", (), {"ledger": type("_LedgerStub", (), {
                    "equity": pd.Series([1.0, 1.1, 1.2], index=_GRID),
                })()},
            ),
            "exposure_scale": exposure,
            "primary_max_drawdown": -0.1,
            "primary": None,
        },
    )()
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))

    assert ctx._fold_growth_budget_target_vol is None
    assert set(ctx._fold_blend_exposure_scale) == {0, 1}
    for i, fold in enumerate(folds):
        mask = (exposure.index >= fold.validation_start) & (exposure.index <= fold.validation_end)
        pd.testing.assert_series_equal(ctx._fold_blend_exposure_scale[i], exposure.loc[mask])
    assert captured["forwarded"] is ctx._fold_blend_exposure_scale


def test_run_folds_constant_risk_requires_blend_exposure_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO_MHS_FOLD_MISSING_BLEND_SCALE_FAILS_CLOSED: a missing blend
    # exposure_scale under constant_risk raises DataIntegrityError -- never a
    # silent empty-dict fallback that would let the run proceed unscaled.
    def _must_not_run(*_a: object, **_k: object):
        raise AssertionError("the fold pool must never start without the blend exposure scale")

    monkeypatch.setattr(
        fold_stage, "phase_1_anchored_purged_folds",
        lambda: (_ExposureFoldStub(pd.Timestamp("2021-06-01", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC")),),
    )
    monkeypatch.setattr(fold_stage, "_run_post_book_concurrently", _must_not_run)

    ctx = _bare_context(committee_book=False)
    ctx.config = dataclasses.replace(ctx.config, pnl_vol_target_mode="constant_risk")
    ctx.blend_report = type(
        "_BlendStub", (),
        {
            "pre_vol_target_reference": type(
                "_RefStub", (), {"ledger": type("_LedgerStub", (), {
                    "equity": pd.Series([1.0, 1.1, 1.2], index=_GRID),
                })()},
            ),
            "exposure_scale": None,
            "primary_max_drawdown": -0.1,
            "primary": None,
        },
    )()
    with pytest.raises(DataIntegrityError, match="exposure_scale"):
        fold_stage.run_folds(ctx, StageTelemetry(log_run=False))


def test_run_folds_skips_boundary_vols_outside_growth_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default (exante_target/conservative) run must never touch the
    # boundary resolver -- byte-identical by construction.
    def _must_not_be_called(*_args: object, **_kwargs: object):
        raise AssertionError("boundary resolver must not run outside growth_budget mode")

    monkeypatch.setattr(fold_stage._scaling, "_growth_budget_target_vol_by_boundary", _must_not_be_called)
    monkeypatch.setattr(fold_stage, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, [], None))
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", lambda *_a, **_k: None)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: (None, None, None),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=False)
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))
    assert ctx._fold_growth_budget_target_vol is None


def test_run_folds_reaches_committee_diagnostic_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        fold_stage, "_run_post_book_concurrently",
        lambda *_a, **_k: (None, None, {}, {}, [], "deployment-stub"),
    )
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", lambda *_a, **_k: None)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(
        fold_stage, "_load_feature_panels",
        lambda *_a, **_k: calls.append("_load_feature_panels") or "panels-stub",
    )
    monkeypatch.setattr(
        fold_stage, "_committee_diagnostic",
        lambda *_a, **_k: calls.append("_committee_diagnostic") or "committee-diag-stub",
    )
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: (None, None, None),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=True)
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))

    assert calls == ["_load_feature_panels", "_committee_diagnostic"]
    assert ctx.committee_diagnostic == "committee-diag-stub"
