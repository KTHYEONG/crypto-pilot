"""Scenario registry aggregating groups A, B and C."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass, replace
from types import MappingProxyType
from typing import Any

from src.live.errors import LiveTradingError
from src.live.testnet_drill.faults import redact
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import ScenarioResult, Verdict
from src.live.testnet_drill.scenarios import ScenarioSpec, acceptance, account_config, lifecycle
from src.live.testnet_drill.session import DirtySessionError


def _collect() -> dict[str, ScenarioSpec]:
    aggregated: dict[str, ScenarioSpec] = {}
    for module in (account_config, acceptance, lifecycle):
        for spec in module.scenario_specs():
            if spec.scenario_id in aggregated:
                raise ValueError(f"duplicate scenario {spec.scenario_id}")
            aggregated[spec.scenario_id] = spec
    return aggregated


SCENARIOS: Mapping[str, ScenarioSpec] = MappingProxyType(_collect())


def _sort_key(scenario_id: str) -> tuple[str, int]:
    return scenario_id[0], int(scenario_id[1:])


def run_selected(
    session_factory: Callable[[], Any],
    selection: str | list[str],
    config: Any,
) -> tuple[ScenarioResult, ...]:
    """Run selected scenarios against fresh sessions from the factory."""
    if isinstance(selection, str):
        wanted = [selection] if selection != "all" else sorted(SCENARIOS, key=_sort_key)
    else:
        wanted = list(selection)
    expanded: list[str] = []
    for item in wanted:
        if len(item) == 1 and item in ("A", "B", "C"):
            expanded.extend(sorted((s for s in SCENARIOS if s.startswith(item)), key=_sort_key))
        else:
            expanded.append(item)
    unknown = set(expanded) - SCENARIOS.keys()
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    results: list[ScenarioResult] = []
    for scenario_id in sorted(set(expanded), key=_sort_key):
        spec = SCENARIOS[scenario_id]
        session = session_factory()
        try:
            evidence = spec.run(session, config)
        except DirtySessionError:
            raise
        except DrillRefused as exc:
            # A refused precondition (e.g. notional cap) sends nothing and teardown already ran; later scenarios stay valid.
            results.append(
                ScenarioResult(
                    scenario_id,
                    Verdict.SKIPPED,
                    expectations=(("scenario_executed", False),),
                    proposal_rationale=redact(str(exc)),
                )
            )
            continue
        except (LiveTradingError, ValueError) as exc:
            results.append(
                ScenarioResult(
                    scenario_id,
                    Verdict.FAIL,
                    expectations=(("scenario_executed", False),),
                    proposal_rationale=redact(str(exc)),
                )
            )
            break
        evidence_path = session.workspace / "scenario_evidence.json"
        if not evidence_path.resolve().is_relative_to(session.workspace.resolve()):
            raise ValueError("scenario evidence must remain inside workspace")
        evidence_path.write_text(
            redact(
                json.dumps(
                    evidence.payload,
                    default=lambda value: (
                        asdict(value) if is_dataclass(value) and not isinstance(value, type) else str(value)
                    ),
                )
            ),
            encoding="utf-8",
        )
        results.append(replace(spec.evaluate(evidence), evidence_path=str(evidence_path)))
        if spec.group == "A" and results[-1].verdict.value == "fail":
            break
    return tuple(results)
