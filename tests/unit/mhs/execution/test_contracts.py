

def test_funding_alignment_distinguishes_known_zero_from_unknown() -> None:
    import pandas as pd
    from src.mhs.execution.contracts import align_funding_with_knowledge
    grid = pd.date_range('2025-01-01', periods=4, freq='1h', tz='UTC')
    source = pd.Series([0.0, 0.001], index=pd.DatetimeIndex([grid[0], grid[-1]]))
    result = align_funding_with_knowledge({'BTCUSDT': source}, grid, symbols=['BTCUSDT', 'MISSUSDT'], source_failures={'MISSUSDT': 'missing'})
    assert result.known['BTCUSDT'].all()
    assert not result.known['MISSUSDT'].any()
    assert (result.rates['MISSUSDT'] == 0.0).all()


def test_funding_alignment_marks_wide_observation_gaps_unknown() -> None:
    import pandas as pd
    from src.mhs.execution.contracts import align_funding_with_knowledge
    grid = pd.date_range('2025-01-01', periods=12, freq='1h', tz='UTC')
    source = pd.Series([0.001, 0.002], index=pd.DatetimeIndex([grid[0], grid[-1]]))
    result = align_funding_with_knowledge({'BTCUSDT': source}, grid, symbols=['BTCUSDT'])
    assert bool(result.known['BTCUSDT'].iloc[0])
    assert bool(result.known['BTCUSDT'].iloc[-1])
    assert not bool(result.known['BTCUSDT'].iloc[1:11].any())
