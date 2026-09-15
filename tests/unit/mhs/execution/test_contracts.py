

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


def test_utc_epoch_ns_matches_legacy_to_datetime_conversion() -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.execution.contracts import _utc_epoch_ns

    def legacy(index):
        return np.asarray(pd.DatetimeIndex(pd.to_datetime(index, utc=True)), dtype='datetime64[ns]').astype('int64')

    ms_utc = pd.DatetimeIndex(pd.to_datetime([1704067200000, 1704096000000], unit='ms', utc=True)).as_unit('ms')
    naive = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 08:00'])
    other_tz = pd.DatetimeIndex(['2024-01-01 09:00', '2024-01-01 17:00'], tz='Asia/Seoul')
    strings = pd.Index(['2024-01-01T00:00:00Z', '2024-01-01T08:00:00Z'])
    for index in (ms_utc, naive, other_tz, strings):
        out = _utc_epoch_ns(index)
        assert out.dtype == np.dtype('int64')
        np.testing.assert_array_equal(out, legacy(index))


def test_align_funding_with_knowledge_accepts_ms_unit_utc_index() -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.execution import align_funding_with_knowledge
    # Given: a loader-shaped series (datetime64[ms, UTC] index), 8h settlements
    idx = pd.DatetimeIndex(pd.date_range('2024-01-01', periods=4, freq='8h', tz='UTC')).as_unit('ms')
    series = pd.Series([0.0001, 0.0002, 0.0003, 0.0004], index=idx)
    grid = pd.date_range('2024-01-01', periods=12, freq='2h', tz='UTC')
    # When
    out = align_funding_with_knowledge({'A': series}, grid, symbols=['A'])
    # Then: every grid bar lies inside the observed span (last settlement 2024-01-02 00:00),
    # and in-window settlements (00:00, 08:00, 16:00) bucket onto bars 0, 4, 8
    known = out.known['A'].to_numpy()
    assert known.all()
    rates = out.rates['A'].to_numpy()
    assert rates[0] == 0.0001
    assert rates[4] == 0.0002
    assert rates[8] == 0.0003
    assert np.count_nonzero(rates) == 3
    # a grid starting before the first settlement is unknown until it
    early = pd.date_range('2023-12-31 20:00', periods=4, freq='2h', tz='UTC')
    early_out = align_funding_with_knowledge({'A': series}, early, symbols=['A'])
    assert early_out.known['A'].tolist() == [False, False, True, True]


