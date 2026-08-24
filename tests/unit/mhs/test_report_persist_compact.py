"""SCENARIO_MHS_PERF_P1_01/P1_02: compact-tier payload expansion removal.

- The compact JSON is byte-identical to the pre-change implementation for the
  standard replay shape, and no ledger timestamp string appears anywhere.
- touch/ladder replays are row-count stubs (five ARTIFACT_CATEGORIES keys),
  never expanded into the 'compact' JSON.
- ``_collect_replay_entries`` enumerates replays generically (touch and ladder
  included) while preserving the frozen legacy ordering.
"""

from __future__ import annotations

import dataclasses
import json
import tracemalloc

import pandas as pd
import pytest

from src.application.research.mhs.contracts import (
    MhsBookReport,
    MhsFoldReport,
    MhsHorizonDiagnosticReport,
    MhsResearchGoResult,
)
from src.mhs.evidence import (
    DeploymentReadinessResult,
    PhaseDiagnosticResult,
    TailSensitivityResult,
)
from src.mhs.execution import StrategyExecutionReplayResult
from src.mhs.params import ARTIFACT_CATEGORIES
from src.mhs.report.persist import (
    _collect_replay_entries,
    _persist_mhs_report_compact,
    _replay_category_row_counts,
    _stubbed_report_for_payload,
    _write_json_report,
)
from src.mhs.report.schema import MhsHorizonDiagnosticReport as ReportSchema
from tests.unit.mhs.test_golden_digest import _synthetic_replay

_BOOK_ORDER = ("fast_reversal", "slow_momentum")


def _small_book(name: str, replay: StrategyExecutionReplayResult, *, with_touch_ladder: bool) -> MhsBookReport:
    base = {
        "name": name, "band": "FAST", "horizon_hours": 24, "step_hours": 6,
        "tranche_count": 1, "n_symbols": 1,
        "phase": PhaseDiagnosticResult(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),
        "prescreen": {},
        "tail": TailSensitivityResult(0.0, 0.0, {}, 1, 0, 0.0, 0.0, 0.0, 0.0),
        "primary": replay, "stress": replay,
        "primary_autocorr_sharpe": 0.1, "primary_naive_sharpe": 0.1,
        "primary_net_ann": 0.01,
        "primary_geometric_cagr": 0.01, "primary_max_drawdown": -0.01,
        "primary_annualized_turnover": 1.0, "stress_naive_sharpe": 0.1,
        "patient_reference": replay, "patient_reference_naive_sharpe": -0.7,
        "pre_vol_target_reference": replay,
        "pre_vol_target_reference_naive_sharpe": 0.2,
    }
    if with_touch_ladder:
        base.update(touch=replay, touch_naive_sharpe=0.05, ladder=replay, ladder_naive_sharpe=0.06)
    return MhsBookReport(**base)


def _small_fold(index: int, replay: StrategyExecutionReplayResult) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=index,
        validation_start="2021-01-01",
        validation_end="2021-03-28",
        strict=replay,
        stress=replay,
        primary_valid=True,
        primary_autocorr_sharpe=0.5,
        primary_naive_sharpe=0.6,
        primary_net_ann=0.02,
        primary_geometric_cagr=0.02,
        primary_max_drawdown=-0.05,
        stress_naive_sharpe=0.4,
        decision_intents=10,
        termination_counts={"MISSING_DATA": 0},
        failures=(),
        strict_elapsed_seconds=1.0,
        stress_elapsed_seconds=1.0,
    )


def _build_report(with_touch_ladder: bool) -> tuple[MhsHorizonDiagnosticReport, pd.DatetimeIndex]:
    replay = dataclasses.replace(_synthetic_replay(n=2000, seed=11))
    ledger_index = replay.ledger.equity.index
    books = {
        name: _small_book(name, replay, with_touch_ladder=with_touch_ladder)
        for name in _BOOK_ORDER
    }
    blend = _small_book("blend", replay, with_touch_ladder=with_touch_ladder)
    folds = tuple(_small_fold(i, replay) for i in range(2))
    report = MhsHorizonDiagnosticReport(
        feature="multi_horizon_market_state", status="COMPLETE", start="2021-01-01",
        end="2021-04-30", resolved_end="2021-04-30", partition="dev",
        execution_tiers_bps=(2.5, 5.0), books=books,
        blend=blend,
        blend_target_gross=0.0, blend_cash_fraction=0.0, eligible_symbols=1,
        trials_attempted=1, deflated_sharpe_ratio=None, xs_rank_ic={},
        date_clustered_regression={}, horizon_diagnostics={}, bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=DeploymentReadinessResult(
            0.01, -0.01, 1.0, -0.01, -0.01, -0.01, -0.01, 0, None, 0.5, 0.0, 0.0, {}, {},
            {}, False, False, False, False,
        ),
        synthetic_stress={}, participation_warnings={}, termination_counts={},
        unsupported_assumptions=(), anchored_folds=(), folds=folds,
        research_go=MhsResearchGoResult(False, (), 0, 0),
        fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="MARK_PRICE",
        execution_timeframe="1m", execution_universe_size=1,
        execution_symbols=("A",), run_elapsed_seconds=0.1,
    )
    assert isinstance(report, ReportSchema)
    return report, ledger_index


class TestCompactByteIdentity:
    """SCENARIO_MHS_PERF_P1_01_COMPACT_BYTE_IDENTICAL."""

    def test_compact_output_matches_pre_change_implementation(self, tmp_path) -> None:
        report, ledger_index = _build_report(with_touch_ladder=False)

        # Reference construction replicating the PRE-CHANGE implementation:
        # expand everything via to_payload(), then overwrite the four core
        # categories per book/blend plus strict/stress per fold with row-count
        # stubs, then append artifacts/replay_ids -- exactly what the old code
        # serialized (expansions were discarded microseconds later).
        legacy_payload = report.to_payload()
        entries = _collect_replay_entries(report)
        row_counts = {rid: _replay_category_row_counts(r) for rid, r in entries}
        for book_name, book_report in report.books.items():
            book_payload = legacy_payload["books"][book_name]
            for field in ("primary", "stress", "patient_reference", "pre_vol_target_reference"):
                if getattr(book_report, field) is not None:
                    book_payload[field] = {
                        category: {"row_count": row_counts[f"{book_name}_{field}"][category]}
                        for category in ARTIFACT_CATEGORIES
                    }
        blend = legacy_payload["blend"]
        for field in ("primary", "stress", "patient_reference", "pre_vol_target_reference"):
            blend[field] = {
                category: {"row_count": row_counts[f"blend_{field}"][category]}
                for category in ARTIFACT_CATEGORIES
            }
        for fold_report in report.folds:
            fold_payload = legacy_payload["folds"][fold_report.fold_index]
            for field in ("strict", "stress"):
                fold_payload[field] = {
                    category: {"row_count": row_counts[f"fold{fold_report.fold_index}_{field}"][category]}
                    for category in ARTIFACT_CATEGORIES
                }
        legacy_payload["artifacts"] = {
            category: {"file": f"{category}.parquet", "row_count": sum(
                rc[category] for rc in row_counts.values()
            )}
            for category in ARTIFACT_CATEGORIES
        }
        daily_rows = len(report.books["fast_reversal"].primary.ledger.equity.groupby(
            ledger_index.normalize(),
        ))
        legacy_payload["artifacts"]["daily_ledger"] = {
            "file": "daily_ledger.parquet",
            "row_count": daily_rows * len(entries),
        }
        legacy_payload["replay_ids"] = [replay_id for replay_id, _ in entries]

        reference_path = tmp_path / "reference.json"
        _write_json_report(reference_path, legacy_payload)

        target = tmp_path / "compact.json"
        tracemalloc.start()
        persisted = _persist_mhs_report_compact(report, target)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert persisted == target
        assert target.read_bytes() == reference_path.read_bytes()
        assert peak < 200 * 2**20
        # No ledger timestamp string may appear anywhere in the file.
        text = target.read_text(encoding="utf-8")
        assert ledger_index[100].isoformat() not in text


class TestTouchLadderNotLeaked:
    """SCENARIO_MHS_PERF_P1_02_TOUCH_LADDER_NOT_LEAKED."""

    def test_touch_and_ladder_are_row_count_stubs(self, tmp_path) -> None:
        report, _ledger_index = _build_report(with_touch_ladder=True)

        target = tmp_path / "compact.json"
        persisted = _persist_mhs_report_compact(report, target)

        assert persisted == target
        assert target.stat().st_size < 200_000
        payload = json.loads(target.read_text(encoding="utf-8"))
        for name, book in report.books.items():
            book_payload = payload["books"][name]
            for field in ("touch", "ladder"):
                stub = book_payload[field]
                assert set(stub) == set(ARTIFACT_CATEGORIES)
                assert all(set(v) == {"row_count"} for v in stub.values())
                assert stub["ledger"]["row_count"] == len(
                    getattr(book, field).ledger.equity
                )
        text = target.read_text(encoding="utf-8")
        assert "2021-01-01T12:" not in text  # a ledger bar timestamp

    def test_collect_replay_entries_ordering_with_touch_ladder(self) -> None:
        report, _ = _build_report(with_touch_ladder=True)
        ids = [replay_id for replay_id, _ in _collect_replay_entries(report)]

        expected_head = [
            f"{name}_{field}"
            for name in _BOOK_ORDER
            for field in (
                "primary", "stress", "patient_reference", "pre_vol_target_reference",
                "touch", "ladder",
            )
        ]
        assert ids[: len(expected_head)] == expected_head
        # Blend follows the books with the same core ordering, folds last.
        assert ids[len(expected_head): len(expected_head) + 4] == [
            "blend_primary", "blend_stress",
            "blend_patient_reference", "blend_pre_vol_target_reference",
            "blend_touch", "blend_ladder",
        ][0:4]
        assert ids[-4:] == [
            "fold0_strict", "fold0_stress", "fold1_strict", "fold1_stress",
        ]

    def test_stubbed_report_never_expands_a_series(self) -> None:
        report, _ = _build_report(with_touch_ladder=True)
        entries = _collect_replay_entries(report)
        row_counts = {rid: _replay_category_row_counts(r) for rid, r in entries}

        stubbed = _stubbed_report_for_payload(report, row_counts)
        for _replay_id, _replay in entries:
            pass
        for book in stubbed.books.values():
            assert all(
                getattr(book, f.name) is None
                for f in dataclasses.fields(book)
                if isinstance(getattr(report.books[book.name], f.name), StrategyExecutionReplayResult)
            )
        assert stubbed.blend is not None
        for fold in stubbed.folds:
            assert fold.strict is None
            assert fold.stress is None
        # The original report object is untouched (dataclasses.replace copy).
        assert report.books["fast_reversal"].primary is not None

    def test_blend_target_weights_never_reach_compact_json(self, tmp_path) -> None:
        """SCENARIO_MHS_PERF_P1_03: blend.target_weights/exposure_scale are the
        research-live seam (I-SIGNAL-FIDELITY) consumed only by
        emit_deployed_target_weights on the *unstubbed* report; the compact
        JSON must never carry the raw decision-grid weight matrix.
        """
        report, ledger_index = _build_report(with_touch_ladder=False)
        weights = pd.DataFrame(
            {"AAAUSDT": [0.01, 0.02, 0.03]},
            index=pd.date_range("2021-01-01", periods=3, freq="1D", tz="UTC"),
        )
        scale = pd.Series([1.0, 0.9, 0.8], index=weights.index)
        report = dataclasses.replace(
            report,
            blend=dataclasses.replace(report.blend, target_weights=weights, exposure_scale=scale),
        )

        entries = _collect_replay_entries(report)
        row_counts = {rid: _replay_category_row_counts(r) for rid, r in entries}
        stubbed = _stubbed_report_for_payload(report, row_counts)
        assert stubbed.blend.target_weights is None
        assert stubbed.blend.exposure_scale is None
        # The original report object is untouched (dataclasses.replace copy):
        # emit_deployed_target_weights consumes report.blend.target_weights
        # separately, after persistence, and must still see the real matrix.
        assert report.blend.target_weights is weights

        target = tmp_path / "report.json"
        persisted = _persist_mhs_report_compact(report, target)
        assert persisted is not None
        raw = persisted.read_text(encoding="utf-8")
        assert "AAAUSDT" not in raw
        assert "0.03" not in raw
        payload = json.loads(raw)
        assert payload["blend"]["target_weights"] is None
        assert payload["blend"]["exposure_scale"] is None


@pytest.mark.parametrize("with_touch_ladder", [False, True])
def test_collect_replay_entries_covers_every_replay_field(with_touch_ladder: bool) -> None:
    report, _ = _build_report(with_touch_ladder=with_touch_ladder)
    entries = _collect_replay_entries(report)
    book_entries = len(_collect_replay_entries(
        dataclasses.replace(report, blend=None, folds=())
    ))
    expected_count = (
        book_entries
        + (6 if with_touch_ladder else 4)  # blend
        + 2 * len(report.folds)
    )
    assert len(entries) == expected_count
