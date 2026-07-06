from __future__ import annotations

import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.contracts import PosteriorGateConfig
from src.domain.futures.alpha_foundry.posterior import shrink_l1_evidence_hierarchical
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


class TestShrinkL1EvidenceHierarchical:
    def test_uses_family_prior_for_sparse_symbol(self) -> None:
        rows = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
                "recipe_id": ["r1", "r1", "r1", "r1"],
                "family": ["trend_ma", "trend_ma", "trend_ma", "trend_ma"],
                "timeframe": ["4h", "4h", "4h", "4h"],
                "activation_context": ["pooled", "pooled", "pooled", "pooled"],
                "net_bps": [5.0, 6.0, 100.0, 120.0],
                "fold_id": [0, 1, 0, 1],
                "effective_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        config = PosteriorGateConfig(prior_effective_n=30.0)
        cost = ExecutionCostModel()
        posterior = shrink_l1_evidence_hierarchical(raw_rows=rows, cost_model=cost, config=config)
        btc = [p for p in posterior if p.symbol == "BTCUSDT"]
        eth = [p for p in posterior if p.symbol == "ETHUSDT"]
        assert len(btc) > 0
        assert len(eth) > 0
        # sparse symbol (BTC, low net_bps) should be closer to family mean than raw outlier (ETH)
        assert btc[0].posterior_mu_bps < eth[0].posterior_mu_bps

    def test_raises_on_missing_required_columns(self) -> None:
        rows = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "symbol": ["BTC"],
                "family": ["f"],
                "timeframe": ["4h"],
                "activation_context": ["pooled"],
                "fold_id": [0],
                "effective_weight": [1.0],
            }
        )
        with pytest.raises(ValueError, match="net_bps"):
            shrink_l1_evidence_hierarchical(
                raw_rows=rows,
                cost_model=ExecutionCostModel(),
                config=PosteriorGateConfig(),
            )

    def test_non_positive_lcb_creates_observe(self) -> None:
        rows = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "recipe_id": ["r1"],
                "family": ["trend_ma"],
                "timeframe": ["4h"],
                "activation_context": ["pooled"],
                "net_bps": [-5.0],
                "fold_id": [0],
                "effective_weight": [1.0],
            }
        )
        config = PosteriorGateConfig(prior_effective_n=30.0)
        cost = ExecutionCostModel()
        posterior = shrink_l1_evidence_hierarchical(raw_rows=rows, cost_model=cost, config=config)
        assert posterior[0].activation_contract == "observe"
        assert posterior[0].quality_weight == 0.0
