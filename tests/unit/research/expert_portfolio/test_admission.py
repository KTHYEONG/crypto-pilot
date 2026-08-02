from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.expert_portfolio.admission import (
    _enumerate_proposals,
    _proposal_pair_diagnostics,
    evaluate_library_admission,
    pairwise_joint_negative_rates,
    pairwise_log_return_correlation,
    prefilter_admitted_by_family_symbol,
    priority_shortlist_family_unique_proposals,
    shortlist_admission_proposals,
)
from src.research.expert_portfolio.admission_types import (
    AdmissionProposal,
    CandidateAdmissionResult,
    LibraryAdmissionConfig,
)
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

    def test_rejected_candidate_is_excluded_from_compatibility_matrix(self) -> None:
        # A non-admitted candidate (closed-trade shortfall) must not break the
        # compatibility matrix shape when admitted candidates are a strict
        # subset of the panel columns.
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f3", "s3"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03, 0.01],
                "b": [0.01, 0.02, -0.02, 0.03],
                "c": [-0.01, 0.02, 0.01, -0.02],
            }
        )
        config = _config(min_experts=2, max_experts=2, **_permissive())
        report = evaluate_library_admission(
            panel,
            {"a": 2, "b": 2, "c": 0},
            experts,
            _context(len(panel)),
            _router(),
            config,
        )
        assert report.status == "COMPLETE"
        admitted = [c.expert_id for c in report.candidates if c.admitted]
        assert admitted == ["a", "b"]
        assert [p.expert_ids for p in report.proposals] == [("a", "b")]


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


class TestProposalShortlist:
    def test_empty_shortlist_is_empty(self) -> None:
        # LAP-04: a universe with no pair-compatible proposals shortlists to nothing.
        assert shortlist_admission_proposals((), 24) == ()

    def test_shortlist_rejects_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="max_backtest_proposals"):
            shortlist_admission_proposals((), 0)

    def test_shortlist_is_size_stratified_and_ranked_by_diversification(
        self,
    ) -> None:
        # LAP-04: proposals are ordered by ascending size and, within a size, by
        # the rank_key (max then mean abs correlation, max then mean joint-negative
        # rate), with no performance metric participating.
        proposals = (
            _proposal(("a", "b"), 0.5, 0.3, 0.5, 0.3),
            _proposal(("a", "c"), 0.4, 0.2, 0.4, 0.2),
            _proposal(("a", "b", "c"), 0.3, 0.1, 0.2, 0.1),
            _proposal(("a", "b", "d"), 0.6, 0.4, 0.4, 0.3),
            _proposal(("a", "c", "d"), 0.2, 0.05, 0.15, 0.05),
        )
        shortlist = shortlist_admission_proposals(proposals, 24)
        assert [p.expert_ids for p in shortlist] == [
            ("a", "c"), ("a", "b"),
            ("a", "c", "d"), ("a", "b", "c"), ("a", "b", "d"),
        ]

    def test_shortlist_selects_six_per_size_then_round_robin_fill(self) -> None:
        # LAP-04: six proposals per available size are selected first and any
        # unused budget is assigned one at a time to the remaining sizes in
        # ascending rank, still capped by the budget.
        from collections import Counter

        size2 = tuple(
            _proposal((f"s2_{i}", f"s2_{i + 1}"), float(i), float(i), float(i), float(i))
            for i in range(8)
        )
        size3 = tuple(
            _proposal((f"s3_{i}", f"s3_{i + 1}", f"s3_{i + 2}"), float(i), float(i), float(i), float(i))
            for i in range(8)
        )
        shortlist = shortlist_admission_proposals((*size2, *size3), 14)
        assert len(shortlist) == 14
        # size-stratified: the size-2 block (six phase-1 picks plus one
        # round-robin fill slot) always precedes the size-3 block
        assert [p.expert_ids for p in shortlist[:7]] == [p.expert_ids for p in size2[:7]]
        assert [p.expert_ids for p in shortlist[7:]] == [p.expert_ids for p in size3[:7]]
        assert Counter(len(p.expert_ids) for p in shortlist) == {2: 7, 3: 7}

    def test_shortlist_size_with_fewer_than_six_contributes_all(self) -> None:
        # LAP-04: a size with fewer than six proposals contributes all of them
        # and the unused budget is consumed by the remaining sizes.
        size2 = tuple(
            _proposal((f"s2_{i}", f"s2_{i + 1}"), float(i), float(i), float(i), float(i))
            for i in range(5)
        )
        size3 = tuple(
            _proposal((f"s3_{i}", f"s3_{i + 1}", f"s3_{i + 2}"), float(i), float(i), float(i), float(i))
            for i in range(8)
        )
        shortlist = shortlist_admission_proposals((*size2, *size3), 11)
        assert len(shortlist) == 11
        assert [p.expert_ids for p in shortlist[:5]] == [p.expert_ids for p in size2]
        assert [p.expert_ids for p in shortlist[5:]] == [p.expert_ids for p in size3[:6]]

    def test_shortlist_exhausts_budget_mid_pass_across_sizes(self) -> None:
        # LAP-04: when the round-robin fill lands exactly on the budget mid-pass,
        # the later sizes simply receive nothing further.
        from collections import Counter

        proposals: list[AdmissionProposal] = []
        for size, tag in ((2, "a"), (3, "b"), (4, "c")):
            proposals.extend(
                _proposal(
                    tuple(f"{tag}_{i + j}" for j in range(size)),
                    float(i), float(i), float(i), float(i),
                )
                for i in range(8)
            )
        shortlist = shortlist_admission_proposals(tuple(proposals), 19)
        assert len(shortlist) == 19
        assert Counter(len(p.expert_ids) for p in shortlist) == {2: 7, 3: 6, 4: 6}

    def test_shortlist_small_budget_caps_before_six_per_size(self) -> None:
        # LAP-04: when the budget cannot fund six per size, the round-robin
        # caps each size fairly and never exceeds the budget.
        proposals: list[AdmissionProposal] = []
        for size, tag in ((2, "a"), (3, "b"), (4, "c")):
            proposals.extend(
                _proposal(
                    tuple(f"{tag}_{i + j}" for j in range(size)),
                    float(i), float(i), float(i), float(i),
                )
                for i in range(8)
            )
        shortlist = shortlist_admission_proposals(tuple(proposals), 4)
        assert len(shortlist) == 4
        assert [p.expert_ids for p in shortlist] == [
            ("a_0", "a_1"), ("a_1", "a_2"),
            ("b_0", "b_1", "b_2"),
            ("c_0", "c_1", "c_2", "c_3"),
        ]

    def test_shortlist_breaks_exact_score_ties_lexically(self) -> None:
        # LAP-04: identical rank_keys are ordered by the lexical proposal_id.
        proposals = (
            _proposal(("b", "c"), 0.5, 0.5, 0.5, 0.5),
            _proposal(("a", "d"), 0.5, 0.5, 0.5, 0.5),
        )
        shortlist = shortlist_admission_proposals(proposals, 24)
        assert [p.expert_ids for p in shortlist] == [("a", "d"), ("b", "c")]

    def test_shortlist_never_invents_missing_rank_data(self) -> None:
        # LAP-04: a proposal with zero pair diagnostics ranks first (no pairs)
        # and never fabricates scores from portfolio or OOS metrics.
        singleton = _proposal(("a",), 0.0, 0.0, 0.0, 0.0)
        pair = _proposal(("a", "b"), 0.9, 0.9, 0.9, 0.9)
        shortlist = shortlist_admission_proposals((pair, singleton), 24)
        assert [p.expert_ids for p in shortlist] == [("a",), ("a", "b")]


def _proposal(
    expert_ids: tuple[str, ...],
    max_abs_corr: float,
    max_joint_negative: float,
    mean_abs_corr: float,
    mean_joint_negative: float,
) -> AdmissionProposal:
    return AdmissionProposal(
        expert_ids=expert_ids,
        eligible=True,
        max_abs_pair_log_return_correlation=max_abs_corr,
        max_pair_joint_negative_rate=max_joint_negative,
        mean_abs_pair_log_return_correlation=mean_abs_corr,
        mean_pair_joint_negative_rate=mean_joint_negative,
    )


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


def _synthetic_universe(
    n_families: int,
    per_family: int,
    seed: int,
    *,
    max_abs_corr: float = 0.5,
    max_joint: float = 0.15,
) -> tuple[tuple[ExpertDefinition, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Random family/symbol universe with gates matching the rolling-v3 pair screen."""
    rng = np.random.default_rng(seed)
    n = n_families * per_family
    experts = tuple(
        _expert(f"e{i}", f"f{i // per_family}", f"SYM{i}") for i in range(n)
    )
    correlation = np.zeros((n, n))
    joint_negative = np.zeros((n, n))
    compatibility = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            c = rng.uniform(-0.6, 0.6)
            jn = rng.uniform(0.0, 0.3)
            correlation[i, j] = correlation[j, i] = c
            joint_negative[i, j] = joint_negative[j, i] = jn
            compatibility[i, j] = compatibility[j, i] = (
                abs(c) <= max_abs_corr and jn <= max_joint
            )
    np.fill_diagonal(compatibility, False)
    return experts, correlation, joint_negative, compatibility


class TestPriorityShortlist:
    def test_priority_shortlist_is_deterministic_and_complete(self) -> None:
        # RAP-V3: the best-first search is the sole rolling generator; it is
        # deterministic and completes within budget on the synthetic universe.
        experts, corr, jn, compat = _synthetic_universe(4, 1, 0)
        compat[:] = True
        np.fill_diagonal(compat, False)
        config = LibraryAdmissionConfig(2, 3, 1, 1, 1.0, 1.0, 1, 10_000)
        priority = priority_shortlist_family_unique_proposals(
            experts, corr, jn, compat, config, 24,
        )
        assert priority.generation_status == "COMPLETE"
        assert priority.proposals
        assert priority.generated_nodes > 0
        assert priority.generation_limit == config.max_combinations
        repeat = priority_shortlist_family_unique_proposals(
            experts, corr, jn, compat, config, 24,
        )
        assert {p.expert_ids for p in repeat.proposals} == {
            p.expert_ids for p in priority.proposals
        }

    def test_priority_shortlist_provable_sufficiency_on_dense_universe(self) -> None:
        # RAP-V3: top_n=shortlist_budget per size is provably sufficient —
        # passing only the per-size top-shortlist_budget reproduces the exact
        # allocation of the full universe when both are ranked by rank_key.
        experts, corr, jn, compat = _synthetic_universe(9, 5, 2)
        config = LibraryAdmissionConfig(2, 4, 1, 1, 0.5, 0.15, 1, 1_000_000)
        enumerated = _enumerate_proposals(experts, compat, 2, 4)
        full_proposals = tuple(
            AdmissionProposal(
                expert_ids=tuple(experts[i].expert_id for i in subset),
                eligible=True,
                **_proposal_pair_diagnostics(corr, jn, subset),
            )
            for subset in enumerated
        )
        full_shortlist = shortlist_admission_proposals(full_proposals, 24)
        priority = priority_shortlist_family_unique_proposals(
            experts, corr, jn, compat, config, 24,
        )
        assert priority.generation_status == "COMPLETE"
        assert {p.expert_ids for p in full_shortlist} == {
            p.expert_ids for p in priority.proposals
        }

    def test_priority_shortlist_completes_on_dense_90_candidate_universe(self) -> None:
        # RAP-V3: on a dense 90-candidate universe the best-first search
        # completes within the node budget and produces a shortlist.
        experts, corr, jn, compat = _synthetic_universe(9, 10, 7)
        compat[:] = True
        np.fill_diagonal(compat, False)
        config = LibraryAdmissionConfig(2, 5, 1, 1, 0.5, 0.15, 1, 1_000_000)
        result = priority_shortlist_family_unique_proposals(
            experts, corr, jn, compat, config, 24,
        )
        assert result.generation_status == "COMPLETE"
        assert result.proposals
        assert result.generated_nodes < config.max_combinations

    def test_priority_shortlist_family_unique_proposals_rejects_shortlist_budget_below_one(
        self,
    ) -> None:
        # RAP-V3: a non-positive shortlist budget fails closed like the v2 path.
        experts, corr, jn, compat = _synthetic_universe(4, 1, 0)
        config = LibraryAdmissionConfig(2, 3, 1, 1, 1.0, 1.0, 1, 100)
        with pytest.raises(ValueError, match="shortlist_budget"):
            priority_shortlist_family_unique_proposals(
                experts, corr, jn, compat, config, 0,
            )

    def test_priority_shortlist_family_unique_proposals_fails_closed_on_node_budget_exhaustion(
        self,
    ) -> None:
        # RAP-V3: exhausting the cumulative node budget fails closed with empty
        # proposals instead of raising or silently truncating the shortlist.
        experts, corr, jn, compat = _synthetic_universe(9, 10, 7)
        config = LibraryAdmissionConfig(2, 5, 1, 1, 0.5, 0.15, 1, 5)
        result = priority_shortlist_family_unique_proposals(
            experts, corr, jn, compat, config, 24,
        )
        assert result.generation_status == "FAIL_CLOSED_NODE_BUDGET"
        assert result.proposals == ()
        assert result.generated_nodes == config.max_combinations


class TestPrefilterByFamilySymbol:
    def test_prefilter_top_k_exceeding_every_group_is_a_noop(self) -> None:
        experts = tuple(_expert(f"e{i}", f"f{i % 3}", f"S{i % 2}") for i in range(6))
        candidates = tuple(
            CandidateAdmissionResult(
                experts[i].expert_id, i + 10, i + 20, True, None,
            )
            for i in range(6)
        )
        kept, dropped = prefilter_admitted_by_family_symbol(
            experts, tuple(range(6)), candidates, top_k=5,
        )
        assert kept == tuple(range(6))
        assert dropped == ()

    def test_prefilter_keeps_top_k_per_symbol_family_by_activity_then_trades(self) -> None:
        experts = (
            _expert("a_long", "f1", "S1"),
            _expert("a_short", "f1", "S1"),
            _expert("b", "f2", "S1"),
        )
        candidates = (
            CandidateAdmissionResult("a_long", 10, 50, True, None),
            CandidateAdmissionResult("a_short", 30, 20, True, None),
            CandidateAdmissionResult("b", 15, 40, True, None),
        )
        kept, dropped = prefilter_admitted_by_family_symbol(
            experts, (0, 1, 2), candidates, top_k=1,
        )
        assert kept == (0, 2)
        assert len(dropped) == 1
        assert dropped[0]["expert_id"] == "a_short"

    def test_prefilter_rejects_non_positive_top_k(self) -> None:
        experts = (_expert("a", "f1", "S1"),)
        candidates = (CandidateAdmissionResult("a", 1, 1, True, None),)
        with pytest.raises(ValueError, match="top_k"):
            prefilter_admitted_by_family_symbol(experts, (0,), candidates, 0)


class TestEvaluatePriorityFamilyUniqueV3:
    def test_evaluate_library_admission_priority_family_unique_v3_matches_v2_screening_output(
        self,
    ) -> None:
        # RAP-V3: the v3 request takes the identical early-return screening path
        # as v2 — candidates, coverage, and proposals are byte-identical.
        experts = (
            _expert("a", "f1", "s1"),
            _expert("b", "f2", "s2"),
            _expert("c", "f3", "s3"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03, 0.01],
                "b": [0.01, 0.02, -0.02, 0.03],
                "c": [-0.01, 0.02, 0.01, -0.02],
            }
        )
        kwargs = {
            "component_returns": panel,
            "component_trade_counts": {e.expert_id: 2 for e in experts},
            "experts": experts,
            "decision_context": _context(len(panel)),
            "router": _router(),
            "config": _config(min_experts=2, max_experts=2, **_permissive()),
        }
        v2 = evaluate_library_admission(**kwargs, proposal_search="bounded_family_unique_v2")
        v3 = evaluate_library_admission(**kwargs, proposal_search="priority_family_unique_v3")
        assert v2 == v3
        assert v3.status == "COMPLETE"
        assert v3.generation_status == "BOUNDED_PENDING"
        assert v3.proposals == ()

    def test_static_admission_pipeline_exact_legacy_v1_unaffected_by_rolling_cleanup(self) -> None:
        # The rolling-v1/v2 cleanup deleted the bounded same-symbol generators
        # only; the static pipeline's exact_legacy_v1 branch and its distinct-
        # symbol enumeration must still enumerate every feasible combination.
        experts = (
            _expert("a", "f1", "S1"),
            _expert("b", "f2", "S2"),
            _expert("c", "f3", "S3"),
        )
        panel = _panel(
            {
                "a": [0.02, -0.01, 0.03, 0.01],
                "b": [0.01, 0.02, -0.02, 0.03],
                "c": [-0.01, 0.02, 0.01, -0.02],
            }
        )
        report = evaluate_library_admission(
            panel,
            {e.expert_id: 2 for e in experts},
            experts,
            _context(len(panel)),
            _router(),
            _config(min_experts=1, max_experts=2, **_permissive()),
        )
        assert report.status == "COMPLETE"
        assert {p.expert_ids for p in report.proposals} == {
            ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"),
        }
