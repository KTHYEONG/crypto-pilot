from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.application.futures.runner.active_pipeline import (
    _has_l1_delivery_candidates,
    _resolve_effective_evidence_start,
    _run_strategy_stage,
)


@dataclass
class _FakeDatetime:
    _d: date

    def date(self) -> date:
        return self._d


@dataclass
class _FakeWindow:
    effective_from: _FakeDatetime
    active_symbols: frozenset[str]


def _win(d: date, syms: frozenset[str]) -> _FakeWindow:
    return _FakeWindow(effective_from=_FakeDatetime(d), active_symbols=syms)


def test_effective_evidence_start_requires_two_consecutive_stable_quarters() -> None:
    windows = [
        _win(date(2023, 4, 1), frozenset()),
        _win(date(2023, 7, 1), frozenset(f"SYM{i}" for i in range(60))),
        _win(date(2023, 10, 1), frozenset()),
        _win(date(2024, 1, 1), frozenset(f"SYM{i}" for i in range(55))),
        _win(date(2024, 4, 1), frozenset(f"SYM{i}" for i in range(58))),
    ]

    result = _resolve_effective_evidence_start(
        tf="4h", timeline_windows=windows, data_start=date(2023, 4, 29),
        regime_floor=date(2023, 1, 1), min_universe_size=50, membership_warmup_days=10,
    )

    assert result == date(2024, 1, 1)


def test_effective_evidence_start_respects_membership_warmup_days() -> None:
    windows = [
        _win(date(2023, 4, 1), frozenset(f"SYM{i}" for i in range(60))),
        _win(date(2023, 7, 1), frozenset(f"SYM{i}" for i in range(60))),
    ]

    result = _resolve_effective_evidence_start(
        tf="4h", timeline_windows=windows, data_start=date(2023, 4, 29),
        regime_floor=date(2023, 1, 1), min_universe_size=50, membership_warmup_days=90,
    )

    assert result > date(2023, 7, 1)


def test_effective_evidence_start_raises_when_never_stable() -> None:
    windows = [_win(date(2023, 4, 1), frozenset()), _win(date(2023, 7, 1), frozenset({"BTCUSDT"}))]

    with pytest.raises(ValueError, match="never reaches"):
        _resolve_effective_evidence_start(
            tf="4h", timeline_windows=windows, data_start=date(2023, 4, 29),
            regime_floor=date(2023, 1, 1), min_universe_size=50, membership_warmup_days=10,
        )


def test_strategy_stage_wires_causal_cutoff_and_delivery_manifest() -> None:
    """The runner must forward both sides of the L0→L1 delivery contract."""
    source = inspect.getsource(_run_strategy_stage)

    assert "l0_evidence_end=l0_evidence_end" in source
    assert "l0_delivery_manifest=l0_delivery_manifest" in source
    assert "consume_candidate_output_for_tiered(" in source


def test_has_l1_delivery_candidates_uses_multi_tf_manifest_not_base_report() -> None:
    """HTF L0 candidates must not be discarded when the base-TF report is empty."""
    output = SimpleNamespace(
        l0_delivery_manifest=SimpleNamespace(final_selected_recipe_ids=("recipe:12h",)),
    )

    assert _has_l1_delivery_candidates(output)
    assert not _has_l1_delivery_candidates(SimpleNamespace(l0_delivery_manifest=None))


def test_tiered_labeled_events_marks_unrouted_events_with_empty_l0_recipe_id() -> None:
    """Unrouted events must be filtered by the L0 manifest, not crash L1."""
    from src.application.futures.runner.active_pipeline import _tiered_labeled_events

    source = SimpleNamespace(labeled_unfiltered=pd.DataFrame({"native_tf": ["4h"]}))

    labeled = _tiered_labeled_events(cast(Any, source))

    assert labeled["l0_recipe_id"].tolist() == [""]
