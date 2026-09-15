

def test_all_mhs_entrypoints_share_zombie_mask_default() -> None:
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
    from src.mhs.live_strategy import LIVE_RUNTIME_DATA_POLICY, LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig
    assert MHS_DATA_POLICY_DEFAULT == 'zombie_mask_v1'
    assert MhsDiagnosticRequest().data_policy == MHS_DATA_POLICY_DEFAULT
    assert MhsRunConfig().data_policy == MHS_DATA_POLICY_DEFAULT
    assert LiveStrategyParams.__dataclass_fields__['data_policy'].default == MHS_DATA_POLICY_DEFAULT
    assert LIVE_RUNTIME_DATA_POLICY == MHS_DATA_POLICY_DEFAULT
