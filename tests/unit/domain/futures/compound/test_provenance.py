from __future__ import annotations

import hashlib
import json

import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import CausalFold
from src.domain.futures.compound.provenance import (
    canonical_config_payload,
    compute_candidate_hash,
    compute_fold_manifest_hash,
    compute_risk_policy_hash,
    compute_strategy_spec_hash,
)


class TestComputeStrategySpecHash:
    def test_compute_strategy_spec_hash_is_deterministic_and_config_sensitive(self) -> None:
        c1 = CompoundEngineConfig()
        c2 = CompoundEngineConfig()
        assert compute_strategy_spec_hash(config=c1) == compute_strategy_spec_hash(config=c2)

    def test_config_change_yields_different_hash(self) -> None:
        from dataclasses import replace
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        c1 = CompoundEngineConfig()
        c2 = replace(c1, dynamic_compounding=replace(c1.dynamic_compounding, target_ann_vol=0.25))
        h1 = compute_strategy_spec_hash(config=c1)
        h2 = compute_strategy_spec_hash(config=c2)
        assert h1 != h2

    def test_hash_length_is_32_hex_chars(self) -> None:
        h = compute_strategy_spec_hash(config=CompoundEngineConfig())
        assert len(h) == 32
        int(h, 16)


class TestHashStability:
    def test_compute_strategy_spec_hash_float_normalization_is_drift_stable(self) -> None:
        from dataclasses import replace
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        base = CompoundEngineConfig()
        c1 = replace(base, dynamic_compounding=replace(base.dynamic_compounding, target_ann_vol=0.15))
        c2 = replace(base, dynamic_compounding=replace(base.dynamic_compounding, target_ann_vol=0.1500000000000001))
        assert compute_strategy_spec_hash(config=c1) == compute_strategy_spec_hash(config=c2)

    def test_code_version_change(self) -> None:
        c1 = CompoundEngineConfig(strategy_code_version="v1")
        c2 = CompoundEngineConfig(strategy_code_version="v2")
        assert compute_strategy_spec_hash(config=c1) != compute_strategy_spec_hash(config=c2)


class TestComputeFoldManifestHash:
    def test_compute_fold_manifest_hash_differs_when_only_boundaries_shift(self) -> None:
        folds1 = (
            CausalFold(0, 0, 100, 80, 90, 100, 200, 5, 2),
            CausalFold(1, 100, 200, 180, 190, 200, 300, 5, 2),
        )
        folds2 = (
            CausalFold(0, 0, 100, 80, 90, 120, 200, 5, 2),
            CausalFold(1, 100, 200, 180, 190, 200, 300, 5, 2),
        )
        h1 = compute_fold_manifest_hash(folds1, max_target_horizon_bars=24)
        h2 = compute_fold_manifest_hash(folds2, max_target_horizon_bars=24)
        assert h1 != h2

    def test_empty_folds_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="folds must be non-empty"):
            compute_fold_manifest_hash((), max_target_horizon_bars=24)

    def test_format_matches_expected_pattern(self) -> None:
        folds = (
            CausalFold(0, 0, 100, 80, 90, 100, 200, 5, 2),
        )
        result = compute_fold_manifest_hash(folds, max_target_horizon_bars=24)
        assert result.startswith("folds_1_")
        assert len(result) == len("folds_1_") + 24


class TestComputeRiskPolicyHash:
    def test_risk_policy_changes_hash(self) -> None:
        from dataclasses import replace
        from src.domain.futures.compound.config import RiskOverlayConfig
        c1 = CompoundEngineConfig()
        c2 = replace(c1, risk=replace(c1.risk, soft_drawdown_start=0.05))
        h1 = compute_risk_policy_hash(config=c1)
        h2 = compute_risk_policy_hash(config=c2)
        assert h1 != h2


class TestComputeCandidateHash:
    def test_same_inputs_same_hash(self) -> None:
        h1 = compute_candidate_hash(
            strategy_spec_hash="a" * 32, fold_manifest_hash="b" * 32,
            descriptor_ids=["sig1", "sig2"], risk_policy_hash="c" * 32,
        )
        h2 = compute_candidate_hash(
            strategy_spec_hash="a" * 32, fold_manifest_hash="b" * 32,
            descriptor_ids=["sig2", "sig1"], risk_policy_hash="c" * 32,
        )
        assert h1 == h2

    def test_descriptor_ordering_normalized(self) -> None:
        h1 = compute_candidate_hash(
            strategy_spec_hash="a" * 32, fold_manifest_hash="b" * 32,
            descriptor_ids=["sig1", "sig2"], risk_policy_hash="c" * 32,
        )
        h2 = compute_candidate_hash(
            strategy_spec_hash="a" * 32, fold_manifest_hash="b" * 32,
            descriptor_ids=["sig2", "sig1"], risk_policy_hash="c" * 32,
        )
        assert h1 == h2


class TestCanonicalConfigPayload:
    def test_includes_code_version(self) -> None:
        payload = canonical_config_payload(CompoundEngineConfig())
        assert "strategy_code_version" in payload
        assert payload["strategy_code_version"] == "compound-2026-07-26"
