from __future__ import annotations

from pathlib import Path

import pytest

from src.research.expert_portfolio.catalog import (
    ExpertLibraryCatalog,
    ExpertLibraryBlueprint,
    compute_blueprint_fingerprint,
    default_catalog,
    registration_id_from_fingerprint,
)
from src.research.expert_portfolio.contracts import ExpertDefinition
from src.research.expert_portfolio.runners import resolve_component_runner


class TestComponentRunnerRegistry:
    def test_component_runner_registry_is_fail_closed(self) -> None:
        # PL-EXPERT-002: a known runner resolves; an unknown key fails closed
        # before any data execution.
        assert callable(resolve_component_runner("run_backtest"))
        with pytest.raises(ValueError, match="not registered"):
            resolve_component_runner("run_pair_residual")


class TestCatalog:
    def test_blueprint_rejects_unsupported_runner(
        self, tmp_path: Path, expert_library_blueprint: ExpertLibraryBlueprint,
    ) -> None:
        bad = ExpertDefinition(
            "e3", "cointegration_residual", "pair_residual", ("CUSDT",), "run_pair_residual", "xyz",
        )
        with pytest.raises(ValueError, match="not supported"):
            ExpertLibraryBlueprint(
                library_id="bad_library",
                experts=(*expert_library_blueprint.experts, bad),
                supported_runners=frozenset({"run_backtest"}),
                code_units=expert_library_blueprint.code_units,
                data_files=expert_library_blueprint.data_files,
                observation_end="2025-12-31",
            )

    def test_catalog_unknown_library_fails_closed(self) -> None:
        catalog = ExpertLibraryCatalog(blueprints={})
        with pytest.raises(ValueError, match="not in the catalog"):
            catalog["nope"]

    def test_fingerprint_is_deterministic_and_binds_code_bytes(
        self, expert_library_blueprint: ExpertLibraryBlueprint,
    ) -> None:
        first = compute_blueprint_fingerprint(expert_library_blueprint)
        second = compute_blueprint_fingerprint(expert_library_blueprint)
        assert first == second
        assert "code_hash" in first
        assert "data_hashes" in first
        assert set(first["data_hashes"]) == {"ohlcv_AUSDT", "ohlcv_BUSDT"}  # type: ignore[arg-type]

        engine = next(iter(expert_library_blueprint.code_units.values()))
        engine.write_text("# changed\nVALUE = 2\n", encoding="utf-8")
        assert compute_blueprint_fingerprint(expert_library_blueprint) != first

    def test_registration_id_derives_only_from_fingerprint(self) -> None:
        base = {"experts": [], "observation_end": "2025-12-31", "code_hash": "a" * 64}
        assert registration_id_from_fingerprint(base) == registration_id_from_fingerprint(base)
        assert registration_id_from_fingerprint(base) != registration_id_from_fingerprint(
            {**base, "data_hashes": {"ohlcv": "b" * 64}}
        )


class TestDefaultCatalog:
    def test_default_catalog_excludes_unmeasured_oi_candidate(self) -> None:
        # FD-07: the unmeasured OI candidate and the retired carry return
        # sources are absent; data intake and a rejected screen leave the
        # deployed expert set unchanged.
        catalog = default_catalog()
        assert "open_interest_deleveraging_v1" not in catalog.blueprints
        retired = {
            "cash_and_carry_basis",
            "altcoin_spot_perp_funding_carry",
            "funding_dispersion_carry",
            "funding_dispersion_reverse",
        }
        assert not set(catalog.blueprints).intersection(retired)
