from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ANTI_PATTERNS_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "decisions" / "anti_patterns.json"


def _load_anti_patterns(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"anti-patterns store is not a JSON list: {path}")
    return records


def _save_anti_patterns(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.json")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def record_rejected_candidate(
    *,
    candidate_id: str,
    data_hash: str,
    code_hash: str,
    hypothesis_id: str,
    failed_gates: list[str],
    reason: str,
    metrics: dict[str, object],
    run_log_reference: str,
    anti_patterns_path: Path = ANTI_PATTERNS_PATH,
) -> bool:
    """Append one idempotent anti-pattern record keyed by candidate fingerprint.

    The entry is keyed by ``candidate_id + data_hash + code_hash``: recording
    the same rejection twice is a no-op, and a failure is never used to alter
    inputs or relax a gate threshold. Returns ``True`` when a new record was
    appended and ``False`` for an idempotent skip.
    """
    if not candidate_id or not data_hash or not code_hash:
        raise ValueError("candidate_id, data_hash and code_hash are required")

    records = _load_anti_patterns(anti_patterns_path)
    for record in records:
        if (
            record.get("candidate_id") == candidate_id
            and record.get("data_hash") == data_hash
            and record.get("code_hash") == code_hash
        ):
            return False

    records.append({
        "domain": "cash_carry",
        "candidate_id": candidate_id,
        "data_hash": data_hash,
        "code_hash": code_hash,
        "failed_hypothesis": hypothesis_id,
        "failed_gates": failed_gates,
        "failure_reason": reason,
        "metrics": metrics,
        "falsified_date": datetime.now(UTC).date().isoformat(),
        "run_log_reference": run_log_reference,
    })
    _save_anti_patterns(records, anti_patterns_path)
    return True
