from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from src.domain.futures.optimization.workflow import evaluate_l2_trial_cached


@dataclass
class _MockConfig:
    dummy: int = 1
    extra: str = ""


class TestEvaluateL2TrialCached:
    def test_reuses_on_same_key(self) -> None:
        sentinel = object()
        with patch(
            "src.domain.futures.optimization.workflow.evaluate_l2_trial",
            return_value=sentinel,
        ) as mock_real:
            cache = object()
            cfg = _MockConfig(dummy=1)
            signal_batch = object()
            caps = object()
            memo: dict[Any, Any] = {}

            result1 = evaluate_l2_trial_cached(
                cache=cache,
                signal_batch=signal_batch,
                aligned=object(),
                awf_folds=(),
                config=cfg,
                caps=caps,
                tf="4h",
                _memo=memo,
            )
            result2 = evaluate_l2_trial_cached(
                cache=cache,
                signal_batch=signal_batch,
                aligned=object(),
                awf_folds=(),
                config=cfg,
                caps=caps,
                tf="4h",
                _memo=memo,
            )

        assert mock_real.call_count == 1
        assert result2 is result1

    def test_recomputes_on_config_change(self) -> None:
        with patch(
            "src.domain.futures.optimization.workflow.evaluate_l2_trial",
            return_value=object(),
        ) as mock_real:
            cache = object()
            cfg1 = _MockConfig(dummy=1)
            cfg2 = _MockConfig(dummy=2)
            signal_batch = object()
            caps = object()
            memo: dict[Any, Any] = {}

            evaluate_l2_trial_cached(
                cache=cache,
                signal_batch=signal_batch,
                aligned=object(),
                awf_folds=(),
                config=cfg1,
                caps=caps,
                tf="4h",
                _memo=memo,
            )
            evaluate_l2_trial_cached(
                cache=cache,
                signal_batch=signal_batch,
                aligned=object(),
                awf_folds=(),
                config=cfg2,
                caps=caps,
                tf="4h",
                _memo=memo,
            )

        assert mock_real.call_count == 2
