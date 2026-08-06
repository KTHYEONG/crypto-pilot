"""Consolidated pass/fail ledger for XS alpha reliability-gate reports.

Every ``xs_alpha_*`` profile report is upserted into exactly one of two
files under the caller's report directory, keyed by its own report name --
never a new per-profile file, never an appended history.
``<dir>/xs_alpha_reliability_pass.json`` holds every report whose
``reliability.lcb.verdict`` (or legacy ``reliability.verdict``) is
``"PASS"``; ``<dir>/xs_alpha_reliability_fail.json`` holds every other
outcome (``FAIL``, ``PENDING``, or no reliability field measured at all,
e.g. pre-reliability-gate profiles). A profile whose verdict flips between
runs is removed from whichever ledger it no longer belongs in, so each
profile lives in exactly one file at all times. The directory is the
caller's own report path's parent (``docs/results`` in production, an
isolated ``tmp_path`` in tests) rather than a hardcoded constant, so tests
never touch the real repo's ledger files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PASS_FILENAME = "xs_alpha_reliability_pass.json"  # noqa: S105 -- ledger filename, not a credential
_FAIL_FILENAME = "xs_alpha_reliability_fail.json"


def _reliability_verdict(payload: dict[str, Any]) -> str | None:
    reliability = payload.get("reliability")
    if not isinstance(reliability, dict):
        return None
    lcb = reliability.get("lcb")
    if isinstance(lcb, dict) and "verdict" in lcb:
        return str(lcb["verdict"])
    verdict = reliability.get("verdict")
    return str(verdict) if verdict is not None else None


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        loaded: dict[str, Any] = json.load(f)
    return loaded


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def persist_reliability_ledger_entry(
    report_name: str, payload: dict[str, Any], ledger_dir: Path,
) -> Path:
    """Upsert one profile's report payload into the pass/fail ledger under ``ledger_dir``.

    Reads ``payload["reliability"]`` to route to the pass or fail ledger,
    removes ``report_name`` from the other ledger first (so a flipped verdict
    never leaves a stale duplicate behind), then writes the target ledger
    with the entry inserted/overwritten. Returns the path the entry now
    lives in.
    """
    verdict = _reliability_verdict(payload)
    pass_path = ledger_dir / _PASS_FILENAME
    fail_path = ledger_dir / _FAIL_FILENAME
    target, other = (pass_path, fail_path) if verdict == "PASS" else (fail_path, pass_path)

    other_ledger = _load_ledger(other)
    if report_name in other_ledger:
        del other_ledger[report_name]
        _write_ledger(other, other_ledger)

    target_ledger = _load_ledger(target)
    target_ledger[report_name] = payload
    _write_ledger(target, target_ledger)
    return target


def _check_contract() -> None:
    assert _PASS_FILENAME == "xs_alpha_reliability_pass.json"  # noqa: S105
    assert _FAIL_FILENAME == "xs_alpha_reliability_fail.json"
    assert _reliability_verdict({}) is None
    assert _reliability_verdict({"reliability": {"lcb": {"verdict": "PASS"}}}) == "PASS"
    assert _reliability_verdict({"reliability": {"verdict": "FAIL"}}) == "FAIL"


_check_contract()
