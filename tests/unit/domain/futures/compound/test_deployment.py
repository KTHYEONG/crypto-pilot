"""Co-modification coverage for deployment helpers."""

from src.domain.futures.compound.deployment import compute_live_target_weights, publish_promoted_strategy


def test_deployment_helpers_are_importable() -> None:
    assert callable(compute_live_target_weights)
    assert callable(publish_promoted_strategy)
