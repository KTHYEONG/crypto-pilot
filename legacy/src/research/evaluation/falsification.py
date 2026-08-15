from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np
from scipy.stats import norm

from src.common.logging import setup_logger

_logger = setup_logger("Falsification")


@dataclass(frozen=True, slots=True)
class FalsificationConfig:
    plateau_ratio: float = 0.70
    min_neighbors: int = 2
    base_t_floor: float = 2.0
    holdout_retention: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.plateau_ratio <= 1:
            raise ValueError(
                f"plateau_ratio must be in (0, 1], got {self.plateau_ratio}"
            )
        if self.min_neighbors < 1:
            raise ValueError(f"min_neighbors must be >= 1, got {self.min_neighbors}")
        if self.base_t_floor < 0:
            raise ValueError(f"base_t_floor must be >= 0, got {self.base_t_floor}")
        if not 0 < self.holdout_retention <= 1:
            raise ValueError(
                f"holdout_retention must be in (0, 1], got {self.holdout_retention}"
            )


def multiplicity_adjusted_t_floor(family_size: int, base_t_floor: float = 2.0) -> float:
    """Bonferroni correction in t space: ``alpha = 2 * sf(base_t_floor)``.

    A family of ``family_size`` independent hypotheses requires
    ``isf(alpha / (2 * family_size))``; ``family_size == 1`` reduces to
    ``base_t_floor`` exactly, so the floor is never inflated for a single screen.
    """
    if family_size < 1:
        raise ValueError(f"family_size must be >= 1, got {family_size}")
    if base_t_floor < 0:
        raise ValueError(f"base_t_floor must be >= 0, got {base_t_floor}")
    alpha = 2 * float(norm.sf(base_t_floor))
    return float(norm.isf(alpha / (2 * family_size)))


@dataclass(frozen=True, slots=True)
class PlateauResult:
    chosen: float
    chosen_score: float
    neighbor_scores: tuple[float, ...]
    neighbor_ratio: float
    passed: bool


def evaluate_parameter_plateau(
    scores: Mapping[float, float],
    chosen: float,
    config: FalsificationConfig,
) -> PlateauResult:
    """Plateau test: a chosen peak must not be an isolated spike.

    The ``config.min_neighbors`` keys nearest to ``chosen`` by absolute key
    distance (ties broken by the smaller key) define the neighbourhood;
    ``neighbor_ratio`` is the median neighbour score over the chosen score.
    ``passed`` requires at least ``min_neighbors`` neighbours, a strictly
    positive chosen score, and a ratio at least ``plateau_ratio`` -- it fails
    closed in every other case.
    """
    if chosen not in scores:
        raise KeyError(f"chosen parameter {chosen} absent from scores")
    chosen_score = float(scores[chosen])
    others = sorted(
        (key for key in scores if key != chosen),
        key=lambda key: (abs(key - chosen), key),
    )
    if len(others) < config.min_neighbors:
        return PlateauResult(chosen, chosen_score, (), 0.0, False)
    neighbors = others[: config.min_neighbors]
    neighbor_scores = tuple(float(scores[key]) for key in neighbors)
    if chosen_score <= 0:
        return PlateauResult(chosen, chosen_score, neighbor_scores, 0.0, False)
    neighbor_ratio = float(np.median(neighbor_scores)) / chosen_score
    passed = neighbor_ratio >= config.plateau_ratio
    return PlateauResult(chosen, chosen_score, neighbor_scores, neighbor_ratio, passed)


@dataclass(frozen=True, slots=True)
class FalsificationVerdict:
    passed: bool
    binding_constraint: str
    plateau: PlateauResult
    oos_t_stat: float
    required_t_floor: float
    holdout_retention: float

    def __post_init__(self) -> None:
        if self.binding_constraint not in (
            "none", "plateau", "multiplicity", "fold_concentration", "symbol_holdout",
        ):
            raise ValueError(
                f"binding_constraint must be one of 'none'/'plateau'/'multiplicity'/"
                f"'fold_concentration'/'symbol_holdout', got {self.binding_constraint}"
            )


def evaluate_falsification(
    *,
    parameter_scores: Mapping[float, float],
    chosen_parameter: float,
    oos_t_stat: float,
    family_size: int,
    dev_score: float,
    holdout_score: float,
    fold_gate_pass: bool,
    config: FalsificationConfig = FalsificationConfig(),  # noqa: B008
) -> FalsificationVerdict:
    """Compose the plateau, multiplicity, fold-concentration, and symbol-holdout
    gates into one verdict.

    The first failing gate is named in ``binding_constraint`` in the fixed order
    ``'plateau'``, ``'multiplicity'``, ``'fold_concentration'``,
    ``'symbol_holdout'``; ``'none'`` is returned only when every gate passes.
    ``fold_gate_pass`` is the caller's pre-computed
    :func:`src.research.evaluation.reliability.compute_equal_duration_fold_distribution`
    ``gate_pass`` on the qualification-period equity, guarding against an
    ``oos_t_stat`` that clears the multiplicity floor only because it is
    disproportionately concentrated in one favourable sub-period rather than
    being distributed across the qualification window (see
    ``docs/specs/growth_engine_fold_concentration_gate.md``).  The function only
    composes existing evidence: it never mutates thresholds and never
    re-selects a parameter or recomputes the fold distribution itself.
    """
    plateau = evaluate_parameter_plateau(parameter_scores, chosen_parameter, config)
    required_t_floor = multiplicity_adjusted_t_floor(family_size, config.base_t_floor)

    binding_constraint = "none"
    if not plateau.passed:
        binding_constraint = "plateau"
    elif oos_t_stat < required_t_floor:
        binding_constraint = "multiplicity"
    elif not fold_gate_pass:
        binding_constraint = "fold_concentration"
    elif dev_score <= 0 or holdout_score < config.holdout_retention * dev_score:
        binding_constraint = "symbol_holdout"

    verdict = FalsificationVerdict(
        passed=binding_constraint == "none",
        binding_constraint=binding_constraint,
        plateau=plateau,
        oos_t_stat=oos_t_stat,
        required_t_floor=required_t_floor,
        holdout_retention=config.holdout_retention,
    )
    _logger.info(
        "falsification passed=%s binding=%s plateau=%.3f oos_t=%.3f floor=%.3f "
        "fold_gate_pass=%s dev=%.3f holdout=%.3f",
        verdict.passed, verdict.binding_constraint, verdict.plateau.neighbor_ratio,
        verdict.oos_t_stat, verdict.required_t_floor, fold_gate_pass, dev_score, holdout_score,
        extra={"tag": "EVAL"},
    )
    return verdict


def _check_contract() -> None:
    """Executable assertions locking the frozen falsification contract surface."""
    config = FalsificationConfig()
    assert (config.plateau_ratio, config.min_neighbors, config.base_t_floor) == (0.70, 2, 2.0)
    assert abs(multiplicity_adjusted_t_floor(1) - 2.0) < 1e-9
    assert abs(multiplicity_adjusted_t_floor(9) - 2.8035) < 1e-3
    assert multiplicity_adjusted_t_floor(20) > multiplicity_adjusted_t_floor(5)
    assert {f.name for f in fields(PlateauResult)} == {
        "chosen", "chosen_score", "neighbor_scores", "neighbor_ratio", "passed",
    }
    assert {f.name for f in fields(FalsificationVerdict)} == {
        "passed", "binding_constraint", "plateau", "oos_t_stat",
        "required_t_floor", "holdout_retention",
    }
    fold_verdict = evaluate_falsification(
        parameter_scores={1.0: 0.95, 2.0: 1.00, 3.0: 0.98},
        chosen_parameter=2.0, oos_t_stat=3.5, family_size=1,
        dev_score=1.0, holdout_score=0.9, fold_gate_pass=False,
    )
    assert fold_verdict.binding_constraint == "fold_concentration"
    assert fold_verdict.passed is False


_check_contract()
