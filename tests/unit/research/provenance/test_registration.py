from __future__ import annotations

from pathlib import Path

import pytest

from src.research.expert_portfolio.catalog import ExpertLibraryCatalog
from src.research.expert_portfolio.models import ExpertDefinition
from src.research.provenance.ledger import load_events
from src.research.provenance.registration import register_expert_library, register_registration


class TestRegisterExpertLibrary:
    def test_register_expert_library_is_idempotent_and_fingerprinted(
        self,
        tmp_path: Path,
        expert_library_catalog: ExpertLibraryCatalog,
    ) -> None:
        # PL-REG-001: repeated registration adds no differing ACTIVE record.
        ledger = tmp_path / "runs.jsonl"
        first = register_expert_library(
            "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
        )
        second = register_expert_library(
            "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
        )
        assert first.registration_id == second.registration_id
        assert first.status == "ACTIVE"
        assert len(load_events(ledger)) == 1
        assert load_events(ledger)[0].record_type == "registration"

    def test_fingerprint_derives_from_code_and_data_files(
        self,
        tmp_path: Path,
        expert_library_catalog: ExpertLibraryCatalog,
        expert_library_blueprint,
    ) -> None:
        ledger = tmp_path / "runs.jsonl"
        registration = register_expert_library(
            "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
        )
        fingerprint = registration.fingerprint
        assert fingerprint["code_hash"]
        assert set(fingerprint["data_hashes"]) == {"ohlcv_AUSDT", "ohlcv_BUSDT"}  # type: ignore[arg-type]

        engine = next(iter(expert_library_blueprint.code_units.values()))
        engine.write_text("# changed\nVALUE = 2\n", encoding="utf-8")
        # a changed implementation is a distinct fingerprint, so the already-
        # ACTIVE library cannot be re-registered under the old id: it fails
        # closed instead of silently creating a second ACTIVE record.
        with pytest.raises(ValueError, match="already ACTIVE"):
            register_expert_library(
                "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
            )
        assert len(load_events(ledger)) == 1

    def test_conflicting_active_fingerprint_fails_closed(
        self,
        tmp_path: Path,
        expert_library_catalog: ExpertLibraryCatalog,
        expert_library_blueprint,
    ) -> None:
        ledger = tmp_path / "runs.jsonl"
        register_expert_library("valid_library", catalog=expert_library_catalog, ledger_path=ledger)
        engine = next(iter(expert_library_blueprint.code_units.values()))
        engine.write_text("# changed\nVALUE = 5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="already ACTIVE"):
            register_expert_library("valid_library", catalog=expert_library_catalog, ledger_path=ledger)

    def test_unknown_library_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not in the catalog"):
            register_expert_library(
                "nope", catalog=ExpertLibraryCatalog(blueprints={}), ledger_path=tmp_path / "runs.jsonl",
            )

    def test_forbidden_return_source_cannot_become_active(self, tmp_path: Path) -> None:
        from tests.fixtures.catalog import write_blueprint_files

        from src.research.expert_portfolio.catalog import ExpertLibraryBlueprint

        code_units, data_files = write_blueprint_files(tmp_path)
        forbidden = ExpertLibraryBlueprint(
            library_id="forbidden_library",
            experts=(
                ExpertDefinition(
                    "e1", "funding_signed_directional", "directional", ("AUSDT",), "run_backtest", "abc",
                ),
            ),
            supported_runners=frozenset({"run_backtest"}),
            code_units=code_units,
            data_files=data_files,
            observation_end="2025-12-31",
        )
        catalog = ExpertLibraryCatalog(blueprints={forbidden.library_id: forbidden})
        with pytest.raises(ValueError, match="forbidden"):
            register_expert_library(
                "forbidden_library", catalog=catalog, ledger_path=tmp_path / "runs.jsonl",
            )

    def test_observation_end_beyond_data_fails_closed(
        self,
        tmp_path: Path,
        expert_library_catalog: ExpertLibraryCatalog,
        expert_library_blueprint,
    ) -> None:
        from src.research.expert_portfolio.catalog import ExpertLibraryBlueprint

        beyond = ExpertLibraryBlueprint(
            library_id="beyond_library",
            experts=expert_library_blueprint.experts,
            supported_runners=expert_library_blueprint.supported_runners,
            code_units=expert_library_blueprint.code_units,
            data_files=expert_library_blueprint.data_files,
            observation_end="2030-12-31",
        )
        catalog = ExpertLibraryCatalog(blueprints={beyond.library_id: beyond})
        with pytest.raises(ValueError, match="beyond available data"):
            register_expert_library(
                "beyond_library", catalog=catalog, ledger_path=tmp_path / "runs.jsonl",
            )

    def test_unsupported_runner_key_fails_closed(self, tmp_path: Path, expert_library_blueprint) -> None:
        from src.research.expert_portfolio.catalog import ExpertLibraryBlueprint
        from src.research.expert_portfolio.models import ExpertDefinition

        # A blueprint whose expert uses a runner outside the supported set is
        # rejected at construction; no registration can ever proceed.
        unsupported = ExpertDefinition(
            "e3", "cointegration_residual", "pair_residual", ("CUSDT",), "run_pair_residual", "xyz",
        )
        with pytest.raises(ValueError, match="not supported"):
            ExpertLibraryBlueprint(
                library_id="bad_runner_library",
                experts=(*expert_library_blueprint.experts, unsupported),
                supported_runners=frozenset({"run_backtest"}),
                code_units=expert_library_blueprint.code_units,
                data_files=expert_library_blueprint.data_files,
                observation_end="2025-12-31",
            )

    def test_missing_data_file_fails_closed(self, tmp_path: Path, expert_library_blueprint) -> None:
        from src.research.expert_portfolio.catalog import ExpertLibraryBlueprint

        missing = ExpertLibraryBlueprint(
            library_id="missing_data_library",
            experts=expert_library_blueprint.experts,
            supported_runners=expert_library_blueprint.supported_runners,
            code_units=expert_library_blueprint.code_units,
            data_files={**expert_library_blueprint.data_files, "ohlcv_CUSDT": tmp_path / "CUSDT.parquet"},
            observation_end="2025-12-31",
        )
        catalog = ExpertLibraryCatalog(blueprints={missing.library_id: missing})
        with pytest.raises(ValueError, match="missing"):
            register_expert_library(
                "missing_data_library", catalog=catalog, ledger_path=tmp_path / "runs.jsonl",
            )


class TestGenericRegistration:
    def test_register_registration_is_idempotent_by_fingerprint(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        fingerprint = {"experts": [], "code_hash": "a" * 64}
        first = register_registration(
            library_id="lib-a", fingerprint=fingerprint, ledger_path=ledger,
        )
        second = register_registration(
            library_id="lib-a", fingerprint=fingerprint, ledger_path=ledger,
        )
        assert first.registration_id == second.registration_id
        assert len(load_events(ledger)) == 1

    def test_retired_registration_uses_retirement_record_type(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        register_registration(
            library_id="old-lib",
            fingerprint={"candidate_id": "old-lib", "code_hash": "b" * 64},
            status="RETIRED",
            ledger_path=ledger,
        )
        events = load_events(ledger)
        assert len(events) == 1
        assert events[0].record_type == "retirement"
        assert events[0].payload["status"] == "RETIRED"

    def test_invalid_status_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ACTIVE or RETIRED"):
            register_registration(
                library_id="lib-a",
                fingerprint={"code_hash": "c" * 64},
                status="PENDING",
                ledger_path=tmp_path / "runs.jsonl",
            )
