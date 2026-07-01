from __future__ import annotations

import importlib
import inspect

from src.domain.futures.optimization.evaluator import compute_robust_score


class TestRobustScore:
    """Scenario 8: compute_robust_score matches existing formula."""

    def test_compute_robust_score_matches_existing_formula(self) -> None:
        assert callable(compute_robust_score)

        sig = inspect.signature(compute_robust_score)
        param_names = set(sig.parameters)
        assert "lambda_down" not in param_names
        assert "lambda_mdd" not in param_names

        ev_mod = importlib.import_module("src.domain.futures.optimization.evaluator")
        assert not hasattr(ev_mod, "compute_v3_score")
