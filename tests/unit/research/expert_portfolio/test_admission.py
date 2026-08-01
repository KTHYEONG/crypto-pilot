from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.expert_portfolio.admission import (
    evaluate_library_admission,
    pairwise_joint_negative_rates,
    pairwise_log_return_correlation,
)
from src.research.expert_portfolio.admission_types import LibraryAdmissionConfig
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition


def _expert(expert_id: str, family: str, symbol: str) -> ExpertDefinition:
    return ExpertDefinition(
        expert_id=expert_id,
        return_source=f"source_{expert_id}",
        family=family,
        symbols=(symbol,),
        runner="run_technical_expert",
        code_hash="h",
    )


def _panel(completed: dict[str, list[float]]) -> pd.DataFrame:
    columns = sorted(completed)
    n = len(next(iter(completed.values()))) + 1
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    data = {col: [np.nan, *completed[col]] for col in columns}
    return pd.DataFrame(data, index=index)[columns]


def _context(n: int, labels: list[str] | None = None) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    if labels is None:
        labels = ["up_low_vol"] * n
    return pd.Series(labels, index=index)


def _config(**overrides: object) -> LibraryAdmissionConfig:
    base: dict[str, object] = {
        "min_experts": 2,
        "max_experts": 4,
        "min_closed_trades": 1,
        "min_active_return_bars": 1,
        "max_abs_pairwise_log_return_correlation": 0.8,
        "max_joint_negative_return_rate": 0.5,
        "min_context_covered_states": 1,
        "max_combinations": 100,
    }
    base.update(overrides)
    return LibraryAdmissionConfig(**base)  # type: ignore[arg-type]


def _router(min_context_history_bars: int = 1) -> ContextualRouterSpec:
    return ContextualRouterSpec("S1", 1, 1, min_context_history_bars)


class TestCandidateDataIntegrity:
    def test_valid_common_index_panel_is_accepted(self) -> None:
        panel = _panel({"a": [0.02, -0.01, 0.03], "b": [0.01, 0.02, -0.02]})
        report = evaluate_library_admission(
            panel,
            {"a": 2, "b": 2},
            (_expert("a", "f1", "s1"), _expert("b", "f2", "s2")),
            _context(len(panel)),
            _router(),
            _config(min_experts=1, max_experts=2),
        )
        assert report.status == "COMPLETE"

    def test_only_initial_all_nan_row_is_permitted(self) -> None:
        # LAE-02: a missing completed return is never zero-filled.
        panel = _panel({"a": [0.02, np.nan, 0.03], "b": [0.01, 0.02, -0.02]})
        with pytest.raises(DataIntegrityError, match="finite"):
            evaluate_library_admission(
                panel,
                {"a": 2, "b": 2},
                (_expert("a", "f1", "s1"), _expert("b", "f2", "s2")),
                _context(len(panel)),
                _router(),
                _config(min_experts=1, max_experts=2),
            )

    def test_non_finite_completed_return_is_rejected(self) -> None:
        panel = _panel({"a": [0.02, np.inf, 0.03], "b": [0.01, 0.02, -0.02]})
        with pytest.raises(DataIntegrityError, match="finite"):
            evaluate_library_admission(
                panel,
                {"a": 2, "b": 2},
                (_expert("a", "f1", "s1"), _expert("b", "f2", "s2")),
                _context(len(panel)),
                _router(),
                _config(min_experts=1, max_experts=2),
            )

    def test_total_loss_return_is_rejected(self) -> None:
        panel = _panel({"a": [0.02, -1.0, 0.03], "b": [0.01, 0.02, -0.02]})
        with pytest.raises(DataIntegrityError, match="strictly greater"):
            evaluate_library_admission(
                panel,
                {"a": 2, "b": 2},
                (_expert("a", "f1", "s1"), _expert("b", "f2", "s2")),
                _context(len(panel)),
                _router(),
                _config(min_experts=1, max_experts=2),
            )

    def test_non_all_nan_first_row_is_rejected(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        panel = pd.DataFrame(
            {"a": [0.02, 0.02, 0.03], "b": [0.01, 0.02, -0.02]}, index=index,
        )
        with pytest.raises(DataIntegrityError, match="all-NaN"):
            evaluate_library_admission(
                panel,
                {"a": 2, "b": 2},
                (_expert("a", "f1", "s1"), _expert("b", "f2", "s2")),
                _context(len(panel)),
                _router(),
                _config(min_experts=1, max_experts=2),
            )

    def test_misaligned_decision_context_is_rejected(self) -> None:
        panel = _panel({"a": [0.02, -0.01], "b": [0.01, 0.02]})
        context = _context(len(panel) - 1)
        with pytest.raises(DataIntegrityError, match="exactly aligned"):
            evaluate_library_admission(
                panel,
                {"a": 2, "b": 2},
                (_expert("a", "f1", "s1"), _expert("b", "f2", "s2")),
                context,
                _router(),
                _config(min_experts=1, max_experts=2),
            )


def _permissive() -> dict[str, object]:
    return {
        "max_abs_pairwise_log_return_correlation": 1.0,
        "max_joint_negative_return_rate": 1.0,
    }


class TestStructuralCombinations:
    def test_enumeration_is_lexical_and_deterministic(self) -> None:
        # LAE-03: size-2 proposals over four distinct experts are exactly the six
        # lexically ordered pairs and nothing else.
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f3", "s3"),
            _expert("d", "f4", "s4"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03, 0.01],
                "b": [0.01, 0.02, -0.02, 0.03],
                "c": [-0.01, 0.02, 0.01, -0.02],
                "d": [0.03, 0.01, -0.01, 0.02],
            }
        )
        config = _config(min_experts=2, max_experts=2, **_permissive())
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(), config,
        )
        assert report.status == "COMPLETE"
        assert [p.expert_ids for p in report.proposals] == [
            ("a", "b"), ("a", "c"), ("a", "d"),
            ("b", "c"), ("b", "d"), ("c", "d"),
        ]
        assert report.proposals == evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(), config,
        ).proposals

    def test_enumeration_respects_requested_subset_sizes(self) -> None:
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f3", "s3"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03],
                "b": [0.01, 0.02, -0.02],
                "c": [-0.01, 0.02, 0.01],
            }
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=1, max_experts=2, **_permissive()),
        )
        sizes = {len(p.expert_ids) for p in report.proposals}
        assert sizes == {1, 2}
        assert {p.expert_ids for p in report.proposals} == {
            ("a",), ("b",), ("c",),
            ("a", "b"), ("a", "c"), ("b", "c"),
        }

    def test_duplicate_family_pair_is_rejected(self) -> None:
        # LAE-03: a candidate pair sharing a family never appears.
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f1", "s3"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03],
                "b": [0.01, 0.02, -0.02],
                "c": [-0.01, 0.02, 0.01],
            }
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=2, max_experts=2, **_permissive()),
        )
        assert [p.expert_ids for p in report.proposals] == [("a", "b"), ("b", "c")]

    def test_duplicate_symbol_pair_is_rejected(self) -> None:
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s1"),
            _expert("c", "f3", "s2"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03],
                "b": [0.01, 0.02, -0.02],
                "c": [-0.01, 0.02, 0.01],
            }
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=2, max_experts=2, **_permissive()),
        )
        assert [p.expert_ids for p in report.proposals] == [("a", "c"), ("b", "c")]


class TestDiversificationAndTail:
    def test_excess_pairwise_correlation_keeps_proposal_ineligible(self) -> None:
        # LAE-04: a perfectly correlated pair exceeds the pre-registered cap and
        # no size-2 proposal is emitted.
        experts = (_expert("a", "f1", "s1"), _expert("b", "f2", "s2"))
        panel = _panel({"a": [0.01] * 8, "b": [0.01] * 8})
        report = evaluate_library_admission(
            panel, {e.expert_id: 3 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=2, max_experts=2,
                    max_abs_pairwise_log_return_correlation=0.8),
        )
        assert report.status == "COMPLETE"
        assert report.proposals == ()

    def test_excess_joint_negative_rate_keeps_proposal_ineligible(self) -> None:
        experts = (_expert("a", "f1", "s1"), _expert("b", "f2", "s2"))
        panel = _panel({"a": [-0.01, 0.01] * 5, "b": [-0.01, 0.02] * 5})
        report = evaluate_library_admission(
            panel, {e.expert_id: 3 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=2, max_experts=2,
                    max_abs_pairwise_log_return_correlation=1.0,
                    max_joint_negative_return_rate=0.2),
        )
        assert report.status == "COMPLETE"
        assert report.proposals == ()

    def test_undefined_correlation_keeps_proposal_ineligible(self) -> None:
        # A constant-return candidate leaves correlation undefined, which is
        # never treated as compatible.
        experts = (_expert("a", "f1", "s1"), _expert("b", "f2", "s2"))
        panel = _panel({"a": [0.01] * 8, "b": [0.02] * 8})
        report = evaluate_library_admission(
            panel, {e.expert_id: 3 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=2, max_experts=2,
                    max_abs_pairwise_log_return_correlation=1.0),
        )
        assert report.proposals == ()

    def test_loss_making_candidate_is_still_admitted_without_ranking(self) -> None:
        # Admission is not a profitability gate: a valid but loss-making
        # candidate remains a proposal and no return-based ranking occurs.
        experts = (_expert("a", "f1", "s1"), _expert("b", "f2", "s2"))
        panel = _panel({"a": [0.02, 0.01] * 4, "b": [-0.01, -0.005] * 4})
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=1, max_experts=1, **_permissive()),
        )
        assert {p.expert_ids for p in report.proposals} == {("a",), ("b",)}


class TestContextCoverage:
    def test_terminal_row_and_unavailable_never_count_toward_coverage(self) -> None:
        # LAE-05: only known causal states before the terminal row count; the
        # terminal "down_high_vol" row and any "unavailable" row are excluded.
        experts = (_expert("a", "f1", "s1"), _expert("b", "f2", "s2"))
        panel = _panel({"a": [0.02, -0.01, 0.03, 0.01, 0.02], "b": [0.01, 0.02, -0.02, 0.03, -0.01]})
        labels = [
            "up_low_vol", "unavailable", "up_low_vol", "up_low_vol",
            "up_low_vol", "down_high_vol",
        ]
        config = _config(
            min_experts=1, max_experts=2, min_context_covered_states=2,
            **_permissive(),
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel), labels), _router(min_context_history_bars=1), config,
        )
        assert report.context_coverage["up_low_vol"] == 4
        assert report.context_coverage["down_high_vol"] == 0
        assert report.covered_states == 1
        assert report.coverage_sufficient is False
        assert all(not p.eligible for p in report.proposals)

    def test_sufficient_coverage_marks_proposals_eligible(self) -> None:
        experts = (_expert("a", "f1", "s1"), _expert("b", "f2", "s2"))
        panel = _panel({"a": [0.02, -0.01, 0.03, 0.01, 0.02], "b": [0.01, 0.02, -0.02, 0.03, -0.01]})
        labels = [
            "up_low_vol", "down_high_vol", "up_low_vol", "up_low_vol",
            "down_high_vol", "down_high_vol",
        ]
        config = _config(
            min_experts=1, max_experts=2, min_context_covered_states=2,
            **_permissive(),
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel), labels), _router(min_context_history_bars=1), config,
        )
        assert report.context_coverage["down_high_vol"] == 2
        assert report.covered_states == 2
        assert report.coverage_sufficient is True
        assert report.proposals
        assert all(p.eligible for p in report.proposals)


class TestBoundedSearch:
    def test_exact_feasible_count_over_budget_fails_closed(self) -> None:
        # LAE-06: C(5, 2) = 10 structural combinations exceed the exact budget of
        # 5, so the evaluator fails closed and returns no subset at all.
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f3", "s3"),
            _expert("d", "f4", "s4"),
            _expert("e", "f5", "s5"),
        )
        panel = _panel({e.expert_id: [0.01, -0.01, 0.02, 0.01] for e in experts})
        config = _config(
            min_experts=2, max_experts=2, max_combinations=5, **_permissive(),
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(), config,
        )
        assert report.status == "FAIL_CLOSED"
        assert report.proposals == ()
        assert report.candidates

    def test_exact_budget_pass_still_enumerates_every_combo(self) -> None:
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f3", "s3"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03],
                "b": [0.01, 0.02, -0.02],
                "c": [-0.01, 0.02, 0.01],
            }
        )
        report = evaluate_library_admission(
            panel, {e.expert_id: 2 for e in experts}, experts,
            _context(len(panel)), _router(),
            _config(min_experts=2, max_experts=2, max_combinations=3,
                    **_permissive()),
        )
        assert report.status == "COMPLETE"
        assert len(report.proposals) == 3


class TestVectorizedPairwise:
    def test_joint_negative_rates_match_nested_loop_reference(self) -> None:
        # LAE-07B: the matrix-product computation equals the reference nested
        # loop and never allocates a bar-by-candidate-by-candidate array.
        rng = np.random.default_rng(42)
        completed = rng.normal(0.0, 0.02, size=(50, 4))
        got = pairwise_joint_negative_rates(completed)
        n = completed.shape[0]
        expected = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                expected[i, j] = np.mean(
                    (completed[:, i] < 0.0) & (completed[:, j] < 0.0)
                )
        np.testing.assert_allclose(got, expected, atol=1e-12)
        assert got.ndim == 2
        assert got.shape == (4, 4)
        assert pairwise_log_return_correlation(completed).shape == (4, 4)
