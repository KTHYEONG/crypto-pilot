from __future__ import annotations

from pathlib import Path

import pytest

from src.research.provenance.ledger import (
    LedgerEvent,
    append_event,
    build_evaluation_event,
    load_events,
    load_evaluation_runs,
)


def _registration_event() -> LedgerEvent:
    return LedgerEvent(
        record_type="registration",
        payload={
            "registration_id": "reg-1",
            "library_id": "lib-a",
            "status": "ACTIVE",
            "registered_at": "2026-01-01T00:00:00+00:00",
            "fingerprint": {"experts": [{"expert_id": "e1"}]},
        },
    )


def _evaluation_event() -> LedgerEvent:
    return build_evaluation_event(
        workflow="expert_portfolio",
        ts="2026-01-02T00:00:00+00:00",
        git_sha="abc123",
        git_dirty=False,
        metrics={"cagr": 0.2, "mdd": -0.1},
        reliability={"observation": {"verdict": "PASS"}},
        promotion={"status": "OBSERVATION_PASS"},
        parent_registration_id="reg-1",
        kind="expert_portfolio",
        allocation_cost_total=0.008,
    )


class TestLedgerAppend:
    def test_append_event_is_immutable_and_readable(self, tmp_path: Path) -> None:
        # PL-LEDGER-001: one valid append yields one readable event without
        # rewriting history.
        ledger = tmp_path / "runs.jsonl"
        appended = append_event(_registration_event(), ledger_path=ledger)
        assert appended.recorded_at != ""
        events = load_events(ledger)
        assert len(events) == 1
        assert events[0].record_type == "registration"
        assert events[0].payload["registration_id"] == "reg-1"
        assert events[0].recorded_at == appended.recorded_at
        lines = ledger.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_append_is_append_only_across_calls(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        append_event(_registration_event(), ledger_path=ledger)
        append_event(_evaluation_event(), ledger_path=ledger)
        append_event(_registration_event(), ledger_path=ledger)
        assert len(ledger.read_text(encoding="utf-8").splitlines()) == 3
        assert len(load_events(ledger)) == 3

    def test_append_rejects_unknown_record_type(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        with pytest.raises(ValueError, match="record_type"):
            append_event(
                LedgerEvent(record_type="unknown", payload={"k": "v"}),
                ledger_path=ledger,
            )
        assert not ledger.exists()

    def test_append_rejects_unsupported_schema_version(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        with pytest.raises(ValueError, match="schema_version"):
            append_event(
                LedgerEvent(
                    record_type="evaluation", payload={"k": "v"}, schema_version=2,
                ),
                ledger_path=ledger,
            )
        assert not ledger.exists()

    def test_malformed_line_raises_line_numbered_error(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        ledger.write_text(
            '{"record_type": "evaluation", "schema_version": 1, "payload": {}}\nnot-json\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="line 2"):
            load_events(ledger)

    def test_unsupported_schema_version_raises_line_numbered_error(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        ledger.write_text(
            '{"record_type": "evaluation", "schema_version": 2, "payload": {}}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="line 1"):
            load_events(ledger)

    def test_missing_ledger_returns_empty(self, tmp_path: Path) -> None:
        assert load_events(tmp_path / "missing.jsonl") == []


class TestLoadEvaluationRuns:
    def test_load_evaluation_runs_filters_events_and_normalizes_legacy(self, tmp_path: Path) -> None:
        # PL-LEDGER-002: a mixed ledger exposes exactly the evaluation rows.
        ledger = tmp_path / "runs.jsonl"
        ledger.write_text(
            '\n'.join([
                # legacy no-version row -> schema_version 0 evaluation
                '{"ts": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT", "metrics": {"cagr": 0.05}}',
                # v1 registration (excluded)
                '{"record_type": "registration", "schema_version": 1, "recorded_at": "2026-01-01T00:00:00+00:00",'
                ' "payload": {"registration_id": "reg-1", "library_id": "lib-a", "status": "ACTIVE"}}',
                # v1 evaluation (included)
                '{"record_type": "evaluation", "schema_version": 1, "recorded_at": "2026-01-02T00:00:00+00:00",'
                ' "payload": {"ts": "2026-01-02T00:00:00+00:00", "kind": "expert_portfolio",'
                ' "metrics": {"cagr": 0.2}, "parent_registration_id": "reg-1"}}',
            ]) + '\n',
            encoding="utf-8",
        )
        df = load_evaluation_runs(ledger)
        assert len(df) == 2
        assert set(df["record_type"]) == {"evaluation"}
        assert set(df["schema_version"]) == {0, 1}
        assert df.loc[0, "metrics.cagr"] == 0.05
        assert df.loc[1, "kind"] == "expert_portfolio"
        assert df.loc[1, "parent_registration_id"] == "reg-1"

    def test_empty_and_missing_ledger_return_empty_frame(self, tmp_path: Path) -> None:
        assert load_evaluation_runs(tmp_path / "missing.jsonl").empty
        empty = tmp_path / "runs.jsonl"
        empty.write_text("", encoding="utf-8")
        assert load_evaluation_runs(empty).empty

    def test_legacy_comparison_columns_survive_normalization(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        ledger.write_text(
            '{"ts": "2026-01-01T00:00:00+00:00", "symbol": "BTCUSDT",'
            ' "reliability": {"observation": {"verdict": "PASS", "lcb90_cagr": 0.16}}}\n',
            encoding="utf-8",
        )
        df = load_evaluation_runs(ledger)
        assert df.loc[0, "symbol"] == "BTCUSDT"
        assert df.loc[0, "reliability.observation.verdict"] == "PASS"
        assert df.loc[0, "reliability.observation.lcb90_cagr"] == 0.16
