"""Chosen strategy values and release ledgers, read from the private strategy store.

The public tree carries the code and methodology only. The chosen configuration and the
research ledgers live in a store root selected by ``CRYPTO_STRATEGY_STORE``; without it the
public example store is used and flagged, and every certifying or live path refuses it.
"""

from __future__ import annotations

import functools
import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

import pandas as pd

from src.common.errors import DataIntegrityError

_STORE_ENV: Final = "CRYPTO_STRATEGY_STORE"
_STRATEGY_VALUES_SCHEMA: Final = "crypto.strategy_values.v1"
_PROJECT_DIRNAME: Final = "crypto-pilot"
_RELEASES_DIRNAME: Final = "releases"
_ML_TILT_SECTION: Final = "ml_tilt"
EXAMPLE_STORE_ROOT: Final[Path] = Path(__file__).resolve().parents[2] / "config" / "strategy_store.example"

Sign = Literal[-1, 1]


@dataclass(frozen=True, slots=True)
class StrategyValues:
    members: tuple[tuple[str, Sign], ...]
    breadth: int
    control_breadth: int
    min_rank_symbols: int
    name_clip: float
    growth_exposure_multiplier: float
    committee_sets: Mapping[str, tuple[str, ...]] = field(repr=False)
    committee_default: str
    example: bool
    root: Path
    revision: str | None

    @property
    def committee_members(self) -> tuple[str, ...]:
        return self.committee_sets[self.committee_default]


@dataclass(frozen=True, slots=True)
class MlTiltValues:
    tilt: float
    horizon_days: int
    label_clip: float
    vol_floor_quantile: float
    smooth_halflife_days: float
    retrain_days: int
    purge_days: int
    schedule_anchor: pd.Timestamp
    feature_breadth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_trees: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    seed: int
    min_train_rows: int


def strategy_store_root() -> Path:
    raw = os.environ.get(_STORE_ENV, "").strip()
    if not raw:
        return EXAMPLE_STORE_ROOT
    root = Path(raw)
    if not root.is_absolute():
        raise DataIntegrityError(f"{_STORE_ENV} must be an absolute path")
    return root


def store_revision(root: Path) -> str | None:
    try:
        text = (root / "STORE_REVISION").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _required(table: Mapping[str, object], key: str, where: str) -> object:
    if key not in table:
        raise DataIntegrityError(f"strategy store missing key: {where}.{key}")
    return table[key]


def _text(table: Mapping[str, object], key: str, where: str) -> str:
    value = _required(table, key, where)
    if not isinstance(value, str) or not value:
        raise DataIntegrityError(f"strategy store key must be a non-empty string: {where}.{key}")
    return value


def _integer(table: Mapping[str, object], key: str, where: str, *, minimum: int) -> int:
    value = _required(table, key, where)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataIntegrityError(f"strategy store key must be an integer >= {minimum}: {where}.{key}")
    return value


def _real(table: Mapping[str, object], key: str, where: str) -> float:
    value = _required(table, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataIntegrityError(f"strategy store key must be a finite number: {where}.{key}")
    if float(value) <= 0.0:
        raise DataIntegrityError(f"strategy store key must be positive: {where}.{key}")
    return float(value)


def _members(flow: Mapping[str, object]) -> tuple[tuple[str, Sign], ...]:
    raw = _required(flow, "members", "flow_mom")
    if not isinstance(raw, list) or not raw:
        raise DataIntegrityError("strategy store key must be a non-empty array: flow_mom.members")
    members: list[tuple[str, Sign]] = []
    for index, entry in enumerate(raw):
        where = f"flow_mom.members[{index}]"
        if not isinstance(entry, dict):
            raise DataIntegrityError(f"strategy store entry must be a table: {where}")
        name = _text(entry, "name", where)
        sign = _required(entry, "sign", where)
        if isinstance(sign, bool) or not isinstance(sign, int) or sign not in (-1, 1):
            raise DataIntegrityError(f"strategy store key must be -1 or 1: {where}.sign")
        members.append((name, cast(Sign, sign)))
    if len({name for name, _ in members}) != len(members):
        raise DataIntegrityError("strategy store flow_mom.members contains duplicate names")
    return tuple(members)


def _committee(committee: Mapping[str, object]) -> tuple[Mapping[str, tuple[str, ...]], str]:
    sets_raw = _required(committee, "sets", "committee")
    if not isinstance(sets_raw, dict) or not sets_raw:
        raise DataIntegrityError("strategy store key must be a non-empty table: committee.sets")
    sets: dict[str, tuple[str, ...]] = {}
    for set_name, names in sets_raw.items():
        where = f"committee.sets.{set_name}"
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
            raise DataIntegrityError(f"strategy store key must be a non-empty array of names: {where}")
        if len(set(names)) != len(names):
            raise DataIntegrityError(f"strategy store key contains duplicate names: {where}")
        sets[set_name] = tuple(names)
    default = _text(committee, "default", "committee")
    if default not in sets:
        raise DataIntegrityError("strategy store committee.default is not a committee set")
    return MappingProxyType(sets), default


def load_strategy_values(root: Path | None = None) -> StrategyValues:
    store_root = strategy_store_root() if root is None else root
    path = store_root / _PROJECT_DIRNAME / "params.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise DataIntegrityError(f"strategy store params unreadable: {path}") from exc
    meta = _required(raw, "meta", "root")
    flow = _required(raw, "flow_mom", "root")
    committee = _required(raw, "committee", "root")
    if not isinstance(meta, dict) or not isinstance(flow, dict) or not isinstance(committee, dict):
        raise DataIntegrityError(f"strategy store params malformed: {path}")
    if _text(meta, "schema", "meta") != _STRATEGY_VALUES_SCHEMA:
        raise DataIntegrityError("strategy store meta.schema is not supported")
    for key in ("version", "created_at", "source", "validated_by"):
        _text(meta, key, "meta")
    example = meta.get("example", False)
    if not isinstance(example, bool):
        raise DataIntegrityError("strategy store key must be a boolean: meta.example")
    committee_sets, committee_default = _committee(committee)
    return StrategyValues(
        members=_members(flow),
        breadth=_integer(flow, "breadth", "flow_mom", minimum=1),
        control_breadth=_integer(flow, "control_breadth", "flow_mom", minimum=1),
        min_rank_symbols=_integer(flow, "min_rank_symbols", "flow_mom", minimum=2),
        name_clip=_real(flow, "name_clip", "flow_mom"),
        growth_exposure_multiplier=_real(flow, "growth_exposure_multiplier", "flow_mom"),
        committee_sets=committee_sets,
        committee_default=committee_default,
        example=example,
        root=store_root,
        revision=store_revision(store_root),
    )


def _ml_number(table: Mapping[str, object], key: str) -> float:
    value = _required(table, key, _ML_TILT_SECTION)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataIntegrityError(f"strategy store key must be a finite number: {_ML_TILT_SECTION}.{key}")
    return float(value)


def _ml_range_error(key: str) -> DataIntegrityError:
    return DataIntegrityError(f"strategy store key is out of range: {_ML_TILT_SECTION}.{key}")


def _ml_schedule_anchor(table: Mapping[str, object]) -> pd.Timestamp:
    where = f"{_ML_TILT_SECTION}.schedule_anchor"
    text = _text(table, "schedule_anchor", _ML_TILT_SECTION)
    try:
        anchor = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"strategy store key must be a UTC timestamp: {where}") from exc
    if anchor.tzinfo is None or anchor.utcoffset() != timedelta(0):
        raise DataIntegrityError(f"strategy store key must be a UTC timestamp: {where}")
    return anchor


def ml_tilt_values(root: Path | None = None) -> MlTiltValues | None:
    """Parse the ``[ml_tilt]`` section of the store params; None when the section is absent."""
    store_root = strategy_store_root() if root is None else root
    path = store_root / _PROJECT_DIRNAME / "params.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise DataIntegrityError(f"strategy store params unreadable: {path}") from exc
    if _ML_TILT_SECTION not in raw:
        return None
    section = raw[_ML_TILT_SECTION]
    if not isinstance(section, dict):
        raise DataIntegrityError(f"strategy store params malformed: {path}")
    values = MlTiltValues(
        tilt=_ml_number(section, "tilt"),
        horizon_days=_integer(section, "horizon_days", _ML_TILT_SECTION, minimum=1),
        label_clip=_real(section, "label_clip", _ML_TILT_SECTION),
        vol_floor_quantile=_ml_number(section, "vol_floor_quantile"),
        smooth_halflife_days=_real(section, "smooth_halflife_days", _ML_TILT_SECTION),
        retrain_days=_integer(section, "retrain_days", _ML_TILT_SECTION, minimum=1),
        purge_days=_integer(section, "purge_days", _ML_TILT_SECTION, minimum=1),
        schedule_anchor=_ml_schedule_anchor(section),
        feature_breadth=_integer(section, "feature_breadth", _ML_TILT_SECTION, minimum=1),
        num_leaves=_integer(section, "num_leaves", _ML_TILT_SECTION, minimum=2),
        min_child_samples=_integer(section, "min_child_samples", _ML_TILT_SECTION, minimum=1),
        learning_rate=_real(section, "learning_rate", _ML_TILT_SECTION),
        n_trees=_integer(section, "n_trees", _ML_TILT_SECTION, minimum=1),
        subsample=_real(section, "subsample", _ML_TILT_SECTION),
        colsample_bytree=_real(section, "colsample_bytree", _ML_TILT_SECTION),
        reg_lambda=_real(section, "reg_lambda", _ML_TILT_SECTION),
        seed=_integer(section, "seed", _ML_TILT_SECTION, minimum=0),
        min_train_rows=_integer(section, "min_train_rows", _ML_TILT_SECTION, minimum=1),
    )
    if not 0.0 < values.tilt < 1.0:
        raise _ml_range_error("tilt")
    if values.purge_days < values.horizon_days + 1:
        raise _ml_range_error("purge_days")
    if not 0.0 <= values.vol_floor_quantile < 0.5:
        raise _ml_range_error("vol_floor_quantile")
    if values.feature_breadth < load_strategy_values(store_root).breadth:
        raise _ml_range_error("feature_breadth")
    if not 0.0 < values.subsample <= 1.0:
        raise _ml_range_error("subsample")
    if not 0.0 < values.colsample_bytree <= 1.0:
        raise _ml_range_error("colsample_bytree")
    return values

@functools.cache
def strategy_values() -> StrategyValues:
    return load_strategy_values(strategy_store_root())


def releases_dir_of(root: Path) -> Path:
    return root / _PROJECT_DIRNAME / _RELEASES_DIRNAME


def releases_root() -> Path:
    return releases_dir_of(strategy_store_root())


def refuse_example_store(action: str) -> None:
    if strategy_values().example:
        raise DataIntegrityError(f"{action} requires the private strategy store; the public example store is active")
