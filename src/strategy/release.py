"""Immutable strategy release record: the ACCEPT verdict that gates real money.

Data only: live loads releases through this module without importing
``src.evaluation`` (layer contract). Evaluation digests are opaque strings here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.errors import DataIntegrityError

if TYPE_CHECKING:
    from src.strategy.targets import StrategySpec

RELEASES_DIRNAME = "releases"


@dataclass(frozen=True, slots=True)
class EvaluationCriteria:
    min_discovery_days: int = 730
    dsr_min: float = 0.95
    growth_lcb_alpha: float = 0.05
    plateau_min_fraction: float = 0.5
    max_top5_funding_dependence: bool = True
    max_participation_p95: float = 0.01
    participation_basis: str = "adv30_median_prior_day"
    max_withdrawn_seat_fraction: float = 0.001
    risk_envelope: str = "growth_extreme_budgeted"
    min_holdout_days: int = 60
    holdout_growth_min_quantile: float = 0.05
    holdout_drawdown_max_quantile: float = 0.95

    def __post_init__(self) -> None:
        if type(self.max_top5_funding_dependence) is not bool:
            raise DataIntegrityError("funding dependence policy must be boolean")
        if self.participation_basis != "adv30_median_prior_day":
            raise DataIntegrityError("participation basis must be adv30_median_prior_day")
        for name in ("min_discovery_days", "min_holdout_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise DataIntegrityError(f"{name} must be an integer >= 2")
        for name in ("dsr_min", "growth_lcb_alpha", "plateau_min_fraction", "max_participation_p95",
                     "holdout_growth_min_quantile", "holdout_drawdown_max_quantile"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise DataIntegrityError(f"{name} must be finite and in (0, 1]")
        fraction = self.max_withdrawn_seat_fraction
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction < 0.05:
            raise DataIntegrityError("max_withdrawn_seat_fraction must be finite and in [0, 0.05)")


@dataclass(frozen=True, slots=True)
class StrategyRelease:
    strategy_id: str
    legacy_ids: tuple[str, ...]
    spec_digest: str
    sizing: Mapping[str, str | float | None]
    design_data_cutoff: pd.Timestamp
    target_capital_usdt: float
    risk_envelope: str
    criteria: EvaluationCriteria
    criteria_digest: str
    evaluation_digest: str | None
    verdict: str | None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def criteria_digest(criteria: EvaluationCriteria) -> str:
    payload = {
        "min_discovery_days": criteria.min_discovery_days,
        "dsr_min": criteria.dsr_min,
        "growth_lcb_alpha": criteria.growth_lcb_alpha,
        "plateau_min_fraction": criteria.plateau_min_fraction,
        "max_top5_funding_dependence": criteria.max_top5_funding_dependence,
        "max_participation_p95": criteria.max_participation_p95,
        "participation_basis": criteria.participation_basis,
        "max_withdrawn_seat_fraction": criteria.max_withdrawn_seat_fraction,
        "risk_envelope": criteria.risk_envelope,
        "min_holdout_days": criteria.min_holdout_days,
        "holdout_growth_min_quantile": criteria.holdout_growth_min_quantile,
        "holdout_drawdown_max_quantile": criteria.holdout_drawdown_max_quantile,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def strategy_spec_digest(spec: StrategySpec, sizing: Mapping[str, str | float | None]) -> str:
    """Content digest of the strategy definition plus the sizing that trades it."""
    from src.core.params import ACCOUNT_EXPOSURE_CAP

    members = [(str(m.name), int(m.sign)) for m in spec.members]
    sized = {str(k): v for k, v in dict(sizing).items()}
    sized.setdefault("exposure_cap", ACCOUNT_EXPOSURE_CAP)
    payload = {
        "strategy_id": spec.strategy_id,
        "breadth": spec.breadth,
        "members": members,
        "min_rank_symbols": spec.min_rank_symbols,
        "snapshot_hour_utc": spec.snapshot_hour_utc,
        "release_hour_utc": spec.release_hour_utc,
        "entry_hour_utc": spec.entry_hour_utc,
        "design_data_cutoff": pd.Timestamp(spec.design_data_cutoff).tz_convert("UTC").isoformat(),
        "exposure_multiplier": float(spec.exposure_multiplier),
        "name_clip": spec.name_clip,
        "sizing": sized,
        "implementation": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("features.py", "targets.py", "books.py", "universe.py", "sizing.py")
        },
        "shared_parameters": hashlib.sha256((Path(__file__).parent.parent / "core" / "params.py").read_bytes()).hexdigest(),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def releases_dir(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if root is not None:
        return Path(root) / "src" / "strategy" / RELEASES_DIRNAME
    return base / "strategy" / RELEASES_DIRNAME


def release_path(strategy_id: str, *, root: Path | None = None) -> Path:
    if not isinstance(strategy_id, str) or not strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    return releases_dir(root) / f"{strategy_id}.json"


def _parse_criteria(raw: Any) -> EvaluationCriteria:
    if not isinstance(raw, dict):
        raise DataIntegrityError("release criteria must be a mapping")
    if "max_withdrawn_seat_fraction" not in raw:
        raise DataIntegrityError("release criteria missing max_withdrawn_seat_fraction")
    try:
        return EvaluationCriteria(**raw)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"release criteria invalid: {exc}") from exc


def load_release(strategy_id: str, *, root: Path | None = None) -> StrategyRelease:
    """Load the committed release record; fails closed on any integrity error."""
    from src.strategy.targets import resolve_strategy_id  # noqa: PLC0415

    canonical = resolve_strategy_id(strategy_id)
    path = release_path(canonical, root=root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataIntegrityError(f"release record missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"release record corrupt: {path}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"release record corrupt: {path}")
    try:
        cutoff = pd.Timestamp(str(raw["design_data_cutoff"])).tz_convert("UTC")
        criteria = _parse_criteria(raw.get("criteria"))
        expected = criteria_digest(criteria)
        stored_digest = str(raw.get("criteria_digest", expected))
        evaluation_digest = raw.get("evaluation_digest")
        verdict = raw.get("verdict")
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"release record invalid: {path}: {exc}") from exc
    if stored_digest != expected:
        raise DataIntegrityError(f"release criteria digest mismatch: {path}")
    if verdict is not None and verdict not in ("accept", "reject", "inconclusive", "invalid"):
        raise DataIntegrityError(f"release verdict invalid: {path}")
    sizing = dict(raw.get("sizing", {}))
    from src.core.params import STRATEGY_RISK_ENVELOPE

    envelope = str(raw.get("risk_envelope", criteria.risk_envelope))
    if envelope != STRATEGY_RISK_ENVELOPE or criteria.risk_envelope != envelope:
        raise DataIntegrityError(f"release risk envelope differs from the registered sizing cap: {path}")
    return StrategyRelease(
        strategy_id=str(raw.get("strategy_id", canonical)),
        legacy_ids=tuple(raw.get("legacy_ids", ())),
        spec_digest=str(raw.get("spec_digest", "")),
        sizing=sizing,
        design_data_cutoff=cutoff,
        target_capital_usdt=float(raw.get("target_capital_usdt", 0.0)),
        risk_envelope=envelope,
        criteria=criteria,
        criteria_digest=stored_digest,
        evaluation_digest=None if evaluation_digest is None else str(evaluation_digest),
        verdict=None if verdict is None else str(verdict),
    )


def record_acceptance(
    strategy_id: str,
    *,
    spec_digest: str,
    evaluation_digest: str,
    root: Path | None = None,
    expected_criteria_digest: str | None = None,
) -> StrategyRelease:
    """Write ``evaluation_digest``/``verdict`` for an ACCEPT whose digests match the file."""
    from src.strategy.targets import resolve_strategy_id  # noqa: PLC0415

    canonical = resolve_strategy_id(strategy_id)
    path = release_path(canonical, root=root)
    release = load_release(canonical, root=root)
    if not spec_digest or release.spec_digest != spec_digest:
        raise DataIntegrityError("accept requires a matching spec digest")
    if not evaluation_digest:
        raise DataIntegrityError("evaluation_digest must be non-empty")
    if expected_criteria_digest is not None and expected_criteria_digest != release.criteria_digest:
        raise DataIntegrityError("accept requires a matching criteria digest")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"release record unreadable: {path}") from exc
    raw["evaluation_digest"] = evaluation_digest
    raw["verdict"] = "accept"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_release(canonical, root=root)
