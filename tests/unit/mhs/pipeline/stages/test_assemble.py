

def test_report_wires_reliability_and_blocks_deployment() -> None:
    import dataclasses
    from src.mhs.evidence import DeploymentReadinessResult
    from src.mhs.reliability import BacktestCertificationLevel, BacktestReliabilityResult, gate_deployment_readiness
    fields = {f.name: 0.0 for f in dataclasses.fields(DeploymentReadinessResult)}
    fields.update(time_under_water_bars=0, recovery_bars=None, leverage_ruin_probabilities={}, concentration={}, participation_warnings={}, research_go_eligible=True, execution_go_eligible=False, pilot_go_eligible=False, scale_go_eligible=False)
    deployment = DeploymentReadinessResult(**fields)
    reliability = BacktestReliabilityResult(False, BacktestCertificationLevel.UNCERTIFIED, ('PRIMARY_EXECUTION_INVALID',), (), None, 0)
    gated = gate_deployment_readiness(deployment, reliability)
    assert not gated.research_go_eligible
    assert deployment.research_go_eligible
