from __future__ import annotations

import inspect
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings

import src.live.settings as settings_module
from src.live.settings import LiveSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs" / "flags.toml"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
COMPOSE_SERVICE = "mhs-live"
BUDGET_TOOL = REPO_ROOT / "tools" / "instruction_budget.py"
RULE_FILE = ".agents/rules/config-flags.md"
TOGGLE_NAME_RE = re.compile(r"(?i)(enable|disable|resend|_on_|allow|auto_adopt|force_)")
KINDS = frozenset({"env_binding", "safety_interlock", "component", "experiment"})
COMPOSE_OVERRIDABLE_KINDS = frozenset({"env_binding", "safety_interlock"})
EXPECTED_COMPONENTS = frozenset(
    {
        "LiveSettings.exec_depth_capture_enabled",
        "LiveSettings.recorder_watch_enabled",
        "LiveSettings.funding_prefetch_enabled",
        "LiveSettings.derisk_mode_enabled",
    }
)
REVIEW_GRACE_DAYS = 14
TODAY: date = date.today()
FIXED_TODAY = date(2026, 11, 11)

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_SERVICE_HEADER_RE = re.compile(r"^  (?P<name>[A-Za-z0-9_-]+):\s*$")
_ENVIRONMENT_HEADER_RE = re.compile(r"^    environment:\s*$")
_SERVICE_KEY_RE = re.compile(r"^    \S")
_ENV_ENTRY_RE = re.compile(r"^      - (?P<key>[A-Z][A-Z0-9_]*)=(?P<value>\S*)\s*$")


def discover_toggles(classes: Iterable[type[BaseModel]] | None = None) -> dict[str, tuple[bool, str]]:
    """Map qualified toggle name to (code default, env name)."""
    if classes is None:
        classes = [
            cls
            for _, cls in inspect.getmembers(settings_module, inspect.isclass)
            if issubclass(cls, BaseSettings) and cls.__module__ == settings_module.__name__
        ]
    toggles: dict[str, tuple[bool, str]] = {}
    for cls in classes:
        prefix = str(cls.model_config.get("env_prefix", ""))
        for field_name, field in cls.model_fields.items():
            if field.annotation is not bool or not TOGGLE_NAME_RE.search(field_name):
                continue
            if field.validation_alias is not None and not isinstance(field.validation_alias, str):
                raise ValueError(f"{cls.__name__}.{field_name}: toggle with a choice of aliases has no single env name")
            env_name = field.validation_alias if isinstance(field.validation_alias, str) else prefix + field_name.upper()
            toggles[f"{cls.__name__}.{field_name}"] = (bool(field.default), env_name)
    return toggles


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        rows = tomllib.load(handle).get("flag", [])
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["name"]
        if name in registry:
            raise ValueError(f"duplicate flag registry row: {name}")
        registry[name] = row
    return registry


def parse_compose_env(path: Path = COMPOSE_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    in_service = False
    in_environment = False
    for line in path.read_text(encoding="utf-8").splitlines():
        header = _SERVICE_HEADER_RE.match(line)
        if header:
            in_service = header["name"] == COMPOSE_SERVICE
            in_environment = False
            continue
        if not in_service:
            continue
        if _ENVIRONMENT_HEADER_RE.match(line):
            in_environment = True
            continue
        if _SERVICE_KEY_RE.match(line):
            in_environment = False
            continue
        entry = _ENV_ENTRY_RE.match(line) if in_environment else None
        if entry is None:
            continue
        if entry["key"] in env:
            raise ValueError(f"{path}: duplicate environment key {entry['key']}")
        env[entry["key"]] = entry["value"]
    if not env:
        raise ValueError(f"{path}: no environment entries for service {COMPOSE_SERVICE}")
    return env


def _as_bool(value: str) -> bool | None:
    token = value.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def coverage_violations(
    toggles: Mapping[str, tuple[bool, str]], registry: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    missing = [
        f"{name}: toggle has no row in docs/flags.toml ({RULE_FILE})" for name in toggles if name not in registry
    ]
    stale = [f"{name}: registry row names no existing toggle ({RULE_FILE})" for name in registry if name not in toggles]
    return [*missing, *stale]


def kind_violations(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: kind {row.get('kind')!r} is not one of {sorted(KINDS)} ({RULE_FILE})"
        for name, row in registry.items()
        if row.get("kind") not in KINDS
    ]


def default_violations(toggles: Mapping[str, tuple[bool, str]], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: registry default {row.get('default')!r} differs from code default {toggles[name][0]!r} ({RULE_FILE})"
        for name, row in registry.items()
        if name in toggles and row.get("default") != toggles[name][0]
    ]


def component_violations(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: component must default to True, found {row.get('default')!r} ({RULE_FILE})"
        for name, row in registry.items()
        if row.get("kind") == "component" and row.get("default") is not True
    ]


def compose_override_violations(
    toggles: Mapping[str, tuple[bool, str]], registry: Mapping[str, Mapping[str, Any]], compose_env: Mapping[str, str]
) -> list[str]:
    violations: list[str] = []
    for name, (default, env_name) in toggles.items():
        if env_name not in compose_env:
            continue
        kind = registry.get(name, {}).get("kind")
        if kind not in COMPOSE_OVERRIDABLE_KINDS:
            violations.append(f"{env_name}: {kind} toggle must not be set in compose ({RULE_FILE})")
            continue
        parsed = _as_bool(compose_env[env_name])
        if parsed is None:
            violations.append(f"{env_name}: compose value {compose_env[env_name]!r} is not a boolean ({RULE_FILE})")
        elif kind != "env_binding" and parsed == default:
            violations.append(f"{env_name}: compose sets the code default {default!r} ({RULE_FILE})")
    return violations


def _deadline_violations(name: str, row: Mapping[str, Any], today: date) -> list[str]:
    violations: list[str] = []
    if not str(row.get("exit_condition", "")).strip():
        violations.append(f"{name}: experiment needs a non-empty exit_condition ({RULE_FILE})")
    try:
        review_by = date.fromisoformat(str(row.get("review_by", "")))
    except ValueError:
        violations.append(f"{name}: experiment review_by must be YYYY-MM-DD ({RULE_FILE})")
        return violations
    if today > review_by + timedelta(days=REVIEW_GRACE_DAYS):
        violations.append(
            f"{name}: experiment review_by {review_by.isoformat()} is more than {REVIEW_GRACE_DAYS} days past; "
            f"delete the flag and its OFF branch ({RULE_FILE})"
        )
    return violations


def experiment_violations(registry: Mapping[str, Mapping[str, Any]], today: date) -> list[str]:
    return [
        violation
        for name, row in registry.items()
        if row.get("kind") == "experiment"
        for violation in _deadline_violations(name, row, today)
    ]


def test_registry_covers_every_toggle() -> None:
    toggles = discover_toggles()

    assert set(EXPECTED_COMPONENTS) <= set(toggles)
    assert coverage_violations(toggles, load_registry()) == []


def test_registry_kind_is_valid() -> None:
    assert kind_violations(load_registry()) == []
    assert kind_violations({"X_ENABLED": {"kind": "rollout", "default": True}}) == [
        f"X_ENABLED: kind 'rollout' is not one of {sorted(KINDS)} ({RULE_FILE})"
    ]


def test_registry_default_matches_code() -> None:
    assert default_violations(discover_toggles(), load_registry()) == []


def test_component_defaults_on_and_not_in_compose() -> None:
    registry = load_registry()
    components = {name for name, row in registry.items() if row.get("kind") == "component"}

    assert components == EXPECTED_COMPONENTS
    assert component_violations(registry) == []
    assert component_violations({"X_ENABLED": {"kind": "component", "default": False}}) != []
    assert compose_override_violations(discover_toggles(), registry, parse_compose_env()) == []


def test_compose_overrides_only_binding_or_interlock() -> None:
    toggles = discover_toggles()
    registry = load_registry()
    compose_env = parse_compose_env()
    overridden = {name for name, (_, env_name) in toggles.items() if env_name in compose_env}

    assert overridden == {"LiveSettings.tax_collection_enabled"}
    assert registry["LiveSettings.tax_collection_enabled"]["kind"] == "env_binding"
    assert compose_override_violations(toggles, registry, compose_env) == []
    assert compose_override_violations(toggles, registry, {"LIVE_RECORDER_WATCH_ENABLED": "false"}) == [
        f"LIVE_RECORDER_WATCH_ENABLED: component toggle must not be set in compose ({RULE_FILE})"
    ]
    assert compose_override_violations(toggles, registry, {"LIVE_VENUE_FORCE_CLOSE_AUTO_ADOPT": "true"}) == []
    assert compose_override_violations(toggles, registry, {"LIVE_TAX_COLLECTION_ENABLED": "true"}) == []


def test_experiment_requires_deadline_and_exit() -> None:
    base = {"kind": "experiment", "default": False, "reason": "r"}
    missing_exit = {**base, "exit_condition": "", "review_by": FIXED_TODAY.isoformat()}
    missing_date = {**base, "exit_condition": "metric holds", "review_by": ""}
    overdue_13 = {**base, "exit_condition": "metric holds", "review_by": (FIXED_TODAY - timedelta(days=13)).isoformat()}
    overdue_15 = {**base, "exit_condition": "metric holds", "review_by": (FIXED_TODAY - timedelta(days=15)).isoformat()}

    assert experiment_violations({"X_ENABLED": missing_exit}, FIXED_TODAY) != []
    assert experiment_violations({"X_ENABLED": missing_date}, FIXED_TODAY) != []
    assert experiment_violations({"X_ENABLED": overdue_13}, FIXED_TODAY) == []
    assert experiment_violations({"X_ENABLED": overdue_15}, FIXED_TODAY) != []
    assert experiment_violations(load_registry(), TODAY) == []


def test_policy_detects_new_default_off_flag() -> None:
    class _Synthetic(BaseModel):
        NEW_FEATURE_ENABLED: bool = False

    toggles = discover_toggles([LiveSettings, _Synthetic])

    assert toggles["_Synthetic.NEW_FEATURE_ENABLED"] == (False, "NEW_FEATURE_ENABLED")
    assert coverage_violations(toggles, load_registry()) == [
        f"_Synthetic.NEW_FEATURE_ENABLED: toggle has no row in docs/flags.toml ({RULE_FILE})"
    ]


def test_config_flags_rule_within_budget() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(BUDGET_TOOL)], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "violations=0" in result.stdout
