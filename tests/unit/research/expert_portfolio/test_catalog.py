from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.expert_portfolio.catalog import (
    ExpertLibraryCatalog,
    ExpertLibraryBlueprint,
    build_technical_price_v1_blueprint,
    compute_blueprint_fingerprint,
    default_catalog,
    registration_id_from_fingerprint,
)
from src.research.expert_portfolio.contracts import ContextualRouterSpec, ExpertDefinition
from src.research.expert_portfolio.runners import (
    ComponentRunRequest,
    _run_technical_expert,
    component_data_requirements,
    resolve_component_runner,
)
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import CostModel


class TestComponentRunnerRegistry:
    def test_component_runner_registry_is_fail_closed(self) -> None:
        # PL-EXPERT-002: a known runner resolves; an unknown key fails closed
        # before any data execution.
        assert callable(resolve_component_runner("run_backtest"))
        with pytest.raises(ValueError, match="not registered"):
            resolve_component_runner("run_pair_residual")

    def test_technical_runner_is_registered(self) -> None:
        # TE-05: the frozen technical runner resolves from the central mapping
        # and declares exactly the causal ohlcv and funding slots.
        assert callable(resolve_component_runner("run_technical_expert"))
        assert component_data_requirements("run_technical_expert") == ("ohlcv", "funding")

    def test_technical_runner_executes_registered_candidate(self, monkeypatch) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)
        funding = pd.Series(0.0, index=index)
        expected = BacktestResult(
            equity=pd.Series(10_000.0, index=index),
            trades=pd.DataFrame(),
            signals=pd.DataFrame(),
        )
        calls: list[object] = []
        monkeypatch.setattr(
            "src.research.expert_portfolio.runners.run_technical_expert_backtest",
            lambda bars, candidate, costs, rates, signal_delay_bars=0: calls.append(
                (bars, candidate.return_source, costs, rates, signal_delay_bars)
            ) or expected,
        )
        definition = ExpertDefinition(
            "technical_macd", "technical_macd_histogram_regime_long_v1",
            "macd_histogram_regime", ("BTCUSDT",), "run_technical_expert", "hash",
        )
        result = _run_technical_expert(
            definition,
            {"ohlcv": frame, "funding": funding},
            ComponentRunRequest(CostModel(), signal_delay_bars=1),
        )
        assert result is expected
        assert calls[0][1] == "technical_macd_histogram_regime_long_v1"

    def test_technical_runner_requires_one_symbol(self) -> None:
        definition = ExpertDefinition(
            "technical_pair", "technical_macd_histogram_regime_long_v1",
            "macd_histogram_regime", ("BTCUSDT", "ETHUSDT"), "run_technical_expert", "hash",
        )
        with pytest.raises(ValueError, match="exactly one symbol"):
            _run_technical_expert(definition, {}, ComponentRunRequest(CostModel()))


class TestCatalog:
    def test_blueprint_preserves_router_in_spec_and_fingerprint(
        self, expert_library_blueprint: ExpertLibraryBlueprint,
    ) -> None:
        router = ContextualRouterSpec("BTCUSDT", 60, 20, 30)
        routed = ExpertLibraryBlueprint(
            library_id=expert_library_blueprint.library_id,
            experts=expert_library_blueprint.experts,
            supported_runners=expert_library_blueprint.supported_runners,
            code_units=expert_library_blueprint.code_units,
            data_files=expert_library_blueprint.data_files,
            observation_end=expert_library_blueprint.observation_end,
            router=router,
        )
        assert routed.to_spec().router == router
        assert compute_blueprint_fingerprint(routed)["router"] == {
            "context_symbol": "BTCUSDT",
            "trend_lookback_bars": 60,
            "volatility_lookback_bars": 20,
            "min_context_history_bars": 30,
            "confidence": 0.90,
        }

    def test_technical_blueprint_requires_approved_expert(self, expert_library_blueprint: ExpertLibraryBlueprint) -> None:
        with pytest.raises(ValueError, match="at least one approved expert"):
            build_technical_price_v1_blueprint(
                (), code_units=expert_library_blueprint.code_units,
                data_files=expert_library_blueprint.data_files, observation_end="2025-12-31",
            )

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

    def test_technical_blueprint_requires_single_symbol(
        self, expert_library_blueprint: ExpertLibraryBlueprint,
    ) -> None:
        expert = ExpertDefinition(
            "technical_pair", "technical_macd_histogram_regime_long_v1",
            "macd_histogram_regime", ("BTCUSDT", "ETHUSDT"), "run_technical_expert", "hash",
        )
        with pytest.raises(ValueError, match="exactly one symbol"):
            build_technical_price_v1_blueprint(
                (expert,), code_units=expert_library_blueprint.code_units,
                data_files=expert_library_blueprint.data_files, observation_end="2025-12-31",
            )

    def test_technical_blueprint_requires_technical_runner(
        self, expert_library_blueprint: ExpertLibraryBlueprint,
    ) -> None:
        expert = ExpertDefinition(
            "technical_wrong_runner", "technical_macd_histogram_regime_long_v1",
            "macd_histogram_regime", ("BTCUSDT",), "run_backtest", "hash",
        )
        with pytest.raises(ValueError, match="must use runner"):
            build_technical_price_v1_blueprint(
                (expert,), code_units=expert_library_blueprint.code_units,
                data_files=expert_library_blueprint.data_files, observation_end="2025-12-31",
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
