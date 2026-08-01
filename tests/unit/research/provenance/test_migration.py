from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.provenance.ledger import load_events, load_evaluation_runs
from src.research.provenance.registration import migrate_legacy_candidate_registry

_DIRECTIONAL_ROW = {
    "domain": "sleeve_blend",
    "candidate_id": "funding_signed_directional_v1",
    "hypothesis_id": "funding_signed_directional",
    "symbols": ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"],
    "rules": {
        "long": "baseline long breakout AND last settled funding <= 0",
        "short": "mirror Donchian breakdown AND last settled funding >= 0",
        "execution": "signal at completed bar close, order at next bar open",
    },
    "parameters": {
        "history_days": 30,
        "max_symbol_weight": 0.25,
        "max_period_contribution": 0.40,
        "leverage": 1.0,
    },
    "data_hashes": {
        "BTCUSDT": {"ohlcv_1h": "a" * 64, "funding": "b" * 64},
        "ETHUSDT": {"ohlcv_1h": "c" * 64, "funding": "d" * 64},
    },
    "code_hash": "e" * 64,
    "observation_end": "2025-12-31 23:59:59",
    "return_source": "funding_gated_long_short_directional",
    "registration_ts": "2026-08-01T02:55:02.536039+00:00",
    "status": "REGISTERED",
}


def _write_source(path: Path) -> None:
    path.write_text(json.dumps([_DIRECTIONAL_ROW], indent=2), encoding="utf-8")


class TestMigration:
    def test_migrate_directional_candidate_as_retired_once(self, tmp_path: Path) -> None:
        # PL-MIGRATE-001: one RETIRED event preserves the legacy id and nested
        # data hashes; a second run adds nothing.
        source = tmp_path / "candidate_registry.json"
        ledger = tmp_path / "runs.jsonl"
        _write_source(source)

        report = migrate_legacy_candidate_registry(source, ledger)
        assert report.appended_count == 1
        assert report.existing_count == 0

        second = migrate_legacy_candidate_registry(source, ledger)
        assert second.appended_count == 0
        assert second.existing_count == 1

        events = load_events(ledger)
        assert len(events) == 1
        event = events[0]
        assert event.record_type == "retirement"
        payload = event.payload
        assert payload["status"] == "RETIRED"
        assert payload["library_id"] == "funding_signed_directional_v1"
        assert payload["registered_at"] == "2026-08-01T02:55:02.536039+00:00"
        legacy = payload["legacy"]
        assert legacy["candidate_id"] == "funding_signed_directional_v1"
        assert legacy["data_hashes"]["BTCUSDT"]["ohlcv_1h"] == "a" * 64
        assert legacy["rules"]["long"].startswith("baseline")

    def test_migrated_record_is_excluded_from_evaluation_runs(self, tmp_path: Path) -> None:
        source = tmp_path / "candidate_registry.json"
        ledger = tmp_path / "runs.jsonl"
        _write_source(source)
        migrate_legacy_candidate_registry(source, ledger)
        assert load_evaluation_runs(ledger).empty

    def test_missing_source_is_idempotent_noop(self, tmp_path: Path) -> None:
        report = migrate_legacy_candidate_registry(tmp_path / "missing.json", tmp_path / "runs.jsonl")
        assert report.appended_count == 0
        assert report.existing_count == 0

    def test_unrecognized_row_raises_without_writing(self, tmp_path: Path) -> None:
        source = tmp_path / "candidate_registry.json"
        ledger = tmp_path / "runs.jsonl"
        source.write_text(json.dumps([{"hypothesis_id": "no candidate id"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="candidate_id"):
            migrate_legacy_candidate_registry(source, ledger)
        assert load_events(ledger) == []
