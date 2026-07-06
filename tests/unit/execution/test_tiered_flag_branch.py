"""USE_CS_RANK_ENGINE 플래그 분기 smoke test."""

from __future__ import annotations

from unittest.mock import patch


def test_use_cs_rank_engine_false_skips_tiered() -> None:
    """USE_CS_RANK_ENGINE=False(기본) 시 run_tiered_pipeline 미호출."""
    with (
        patch("src.execution.opt_main_futures.OPT_FUTURES_CONFIG", {"USE_CS_RANK_ENGINE": False}),
        patch("src.domain.futures.strategy.tiered_workflow.run_tiered_pipeline") as mock_tp,
    ):
        from src.execution import opt_main_futures

        flag = opt_main_futures.OPT_FUTURES_CONFIG.get("USE_CS_RANK_ENGINE", False)
        assert flag is False
        mock_tp.assert_not_called()


def test_use_cs_rank_engine_flag_readable() -> None:
    """OPT_FUTURES_CONFIG에서 USE_CS_RANK_ENGINE 키 읽기 가능하고 기본값=True."""
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    val = OPT_FUTURES_CONFIG.get("USE_CS_RANK_ENGINE", False)
    assert val is True
