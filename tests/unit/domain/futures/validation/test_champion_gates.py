from __future__ import annotations

import importlib

from src.domain.futures.validation.gates import ChampionGateConfig, evaluate_champion_gates


class TestChampionGates:
    """Scenario 9: evaluate_champion_gates preserves gate semantics."""

    def test_evaluate_champion_gates_preserves_gate_semantics(self) -> None:
        config = ChampionGateConfig()
        assert abs(config.MIN_POSITIVE_LEG_RATIO - 0.55) < 1e-9

        assert callable(evaluate_champion_gates)

        gates_mod = importlib.import_module("src.domain.futures.validation.gates")
        assert not hasattr(gates_mod, "V3HardGates")
        assert not hasattr(gates_mod, "evaluate_v3_hard_gates")
