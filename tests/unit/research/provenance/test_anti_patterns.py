from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.provenance.anti_patterns import record_rejected_candidate


def _kwargs(**overrides):
    base = {
        "candidate_id": "cand-1",
        "data_hash": "data-1",
        "code_hash": "code-1",
        "hypothesis_id": "cash_and_carry_basis",
        "failed_gates": ["observation", "stress"],
        "reason": "promotion status=REJECTED",
        "metrics": {"cagr": 0.05, "trade_count": 12},
        "run_log_reference": "2026-07-31T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestResearchMemory:
    def test_records_rejection_once(self, tmp_path: Path) -> None:
        path = tmp_path / "anti_patterns.json"
        assert record_rejected_candidate(anti_patterns_path=path, **_kwargs()) is True
        records = json.loads(path.read_text(encoding="utf-8"))
        assert len(records) == 1
        assert records[0]["candidate_id"] == "cand-1"
        assert records[0]["failed_gates"] == ["observation", "stress"]

    def test_idempotent_same_fingerprint_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "anti_patterns.json"
        record_rejected_candidate(anti_patterns_path=path, **_kwargs())
        assert record_rejected_candidate(anti_patterns_path=path, **_kwargs()) is False
        records = json.loads(path.read_text(encoding="utf-8"))
        assert len(records) == 1

    def test_distinct_fingerprint_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "anti_patterns.json"
        record_rejected_candidate(anti_patterns_path=path, **_kwargs())
        record_rejected_candidate(
            anti_patterns_path=path, **_kwargs(candidate_id="cand-2", data_hash="data-2"),
        )
        records = json.loads(path.read_text(encoding="utf-8"))
        assert len(records) == 2

    def test_missing_fingerprint_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="required"):
            record_rejected_candidate(
                anti_patterns_path=tmp_path / "x.json",
                candidate_id="",
                data_hash="",
                code_hash="",
                hypothesis_id="h",
                failed_gates=["observation"],
                reason="r",
                metrics={},
                run_log_reference="ref",
            )

    def test_existing_entries_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "anti_patterns.json"
        path.write_text(json.dumps([{"domain": "risk", "failed_hypothesis": "old"}]), encoding="utf-8")
        record_rejected_candidate(anti_patterns_path=path, **_kwargs())
        records = json.loads(path.read_text(encoding="utf-8"))
        assert len(records) == 2
