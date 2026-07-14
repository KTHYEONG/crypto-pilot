from __future__ import annotations

from unittest.mock import mock_open

import src.domain.futures.alpha_foundry.memory as memory_module
from src.domain.futures.alpha_foundry.memory import (
    L0MemoryBudget,
    _detect_physical_limit_mb,
    _read_vm_rss_mb,
    admit_memory_stage,
    current_process_rss_mb,
    resolve_effective_memory_budget,
    resolve_ltf_exec_1m_plan,
)


class TestMemoryBudget:

    def test_resolve_budget_defaults_returns_positive(self) -> None:
        budget = resolve_effective_memory_budget()
        assert budget.limit_mb > 0
        assert budget.safety_margin_mb > 0

    def test_admit_memory_stage_under_budget(self) -> None:
        budget = L0MemoryBudget(limit_mb=1024, safety_margin_mb=64)
        assert admit_memory_stage(
            budget=budget, stage="test",
            estimated_increment_mb=100, current_rss_mb=500,
        )

    def test_admit_memory_stage_over_budget(self) -> None:
        budget = L0MemoryBudget(limit_mb=1024, safety_margin_mb=64)
        assert not admit_memory_stage(
            budget=budget, stage="test",
            estimated_increment_mb=500, current_rss_mb=500,
        )

    def test_rss_returns_non_negative(self) -> None:
        rss = current_process_rss_mb()
        assert rss >= 0

    def test_read_vm_rss(self) -> None:
        rss = _read_vm_rss_mb()
        # May return 0 in test env without /proc/self/status
        assert rss >= 0

    def test_physical_limit_detected(self) -> None:
        limit = _detect_physical_limit_mb()
        if limit is not None:
            assert limit > 0

    def test_rss_read_failure_returns_zero(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("x")))
        assert memory_module._read_vm_rss_mb() == 0

    def test_cgroup_read_failure_returns_none(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(memory_module.Path, "exists", lambda _self: True)
        monkeypatch.setattr(memory_module.Path, "read_text", lambda _self: (_ for _ in ()).throw(OSError("x")))
        assert memory_module._detect_cgroup_limit_mb() is None

    def test_physical_fallback_reads_memtotal(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(memory_module.ctypes, "CDLL", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("x")))
        monkeypatch.setattr("builtins.open", mock_open(read_data="MemTotal: 2048 kB\n"))
        assert memory_module._detect_physical_limit_mb() == 2


class TestLtfExec1mPlan:

    def test_plan_with_skip_reason_returns_empty(self) -> None:
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset({"BTCUSDT"}),
            valid_symbols=frozenset({"BTCUSDT"}),
            skip_reason="test_skip",
        )
        assert plan.symbols == ()
        assert plan.skip_reason == "test_skip"

    def test_plan_selects_eligible_symbols(self) -> None:
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset({"BTCUSDT", "ETHUSDT"}),
            valid_symbols=frozenset({"BTCUSDT"}),
            max_symbols=64,
        )
        assert "BTCUSDT" in plan.symbols
        assert "ETHUSDT" not in plan.symbols

    def test_plan_no_covered_returns_skip_reason(self) -> None:
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset(),
            valid_symbols=frozenset({"BTCUSDT"}),
        )
        assert plan.symbols == ()
        assert plan.skip_reason == "no_covered_symbols"

    def test_plan_budget_rejected_returns_skip(self) -> None:
        from src.domain.futures.alpha_foundry.memory import L0MemoryBudget

        budget = L0MemoryBudget(limit_mb=1, safety_margin_mb=0)
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset({"BTCUSDT"}),
            valid_symbols=frozenset({"BTCUSDT"}),
            budget=budget,
        )
        assert plan.symbols == ()

    def test_plan_respects_max_workers_2(self) -> None:
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset({"BTCUSDT", "ETHUSDT"}),
            valid_symbols=frozenset({"BTCUSDT", "ETHUSDT"}),
            max_workers=2,
        )
        assert plan.max_workers == 2

    def test_plan_clamps_max_workers_above_2(self) -> None:
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset({"BTCUSDT"}),
            valid_symbols=frozenset({"BTCUSDT"}),
            max_workers=5,
        )
        assert plan.max_workers <= 2

    def test_plan_default_max_workers_1(self) -> None:
        plan = resolve_ltf_exec_1m_plan(
            covered_symbols=frozenset({"BTCUSDT"}),
            valid_symbols=frozenset({"BTCUSDT"}),
        )
        assert plan.max_workers == 1
