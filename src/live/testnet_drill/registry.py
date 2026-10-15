"""Scenario registry aggregating groups A through G."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass, replace
from types import MappingProxyType
from typing import Any

from src.common.errors import DataIntegrityError
from src.live.errors import LiveTradingError
from src.live.testnet_drill.faults import redact
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import ScenarioResult, Verdict
from src.live.testnet_drill.scenarios import (
    ScenarioSpec,
    acceptance,
    account_config,
    account_state,
    cycle_e2e,
    lifecycle,
    restart_recovery,
    transport_faults,
)
from src.live.testnet_drill.session import DirtySessionError


def _collect() -> dict[str, ScenarioSpec]:
    aggregated: dict[str, ScenarioSpec] = {}
    for module in (account_config, acceptance, lifecycle, transport_faults, restart_recovery, account_state, cycle_e2e):
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
    *,
    continue_on_fail: bool = False,
) -> tuple[ScenarioResult, ...]:
    """Run selected scenarios in A..G order against fresh sessions from the factory.

    A failed group-A scenario or an aborted scenario (clean-slate failure, venue error) stops the run unless
    ``continue_on_fail``. ``DirtySessionError`` always aborts, carrying the results so far in ``partial_results``.
    """
    if isinstance(selection, str):
        wanted = [selection] if selection != "all" else sorted(SCENARIOS, key=_sort_key)
    else:
        wanted = list(selection)
    expanded: list[str] = []
    for item in wanted:
        if len(item) == 1 and item in ("A", "B", "C", "D", "E", "F", "G"):
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
        except DirtySessionError as exc:
            results.append(
                ScenarioResult(
                    scenario_id,
                    Verdict.FAIL,
                    expectations=(("session_restored", False),),
                    proposal_rationale=redact(f"DIRTY residuals={exc.residuals}"),
                )
            )
            exc.partial_results = tuple(results)
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
        except (LiveTradingError, DataIntegrityError, ValueError) as exc:
            results.append(
                ScenarioResult(
                    scenario_id,
                    Verdict.FAIL,
                    expectations=(("scenario_executed", False),),
                    proposal_rationale=redact(f"{type(exc).__name__}: {exc}"),
                )
            )
            if continue_on_fail:
                continue
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
        results.append(
            replace(
                spec.evaluate(evidence),
                evidence_path=str(evidence_path),
                testnet_reset_suspected=evidence.payload.get("testnet_reset_suspected") is True,
            )
        )
        if spec.group == "A" and results[-1].verdict is Verdict.FAIL and not continue_on_fail:
            break
    return tuple(results)
