

def test_causal_state_marks_unrealized_pnl_once() -> None:
    import numpy as np
    from src.mhs.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=1000.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=None)
    state.apply_fill(symbol_index=0, quantity_delta=10.0, fill_price=100.0, fee_bps=0.0)
    state.advance_to(event_ns=1, marks=np.array([110.0]), funding_rates=np.zeros(1), funding_known=np.ones(1, dtype=bool))
    assert state.cash == 0.0
    assert state.equity() == 1100.0


def test_causal_state_does_not_charge_pre_entry_funding() -> None:
    import numpy as np
    from src.mhs.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=1000.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=None)
    charged = state.advance_to(event_ns=1, marks=np.array([100.0]), funding_rates=np.array([0.01]), funding_known=np.ones(1, dtype=bool))
    state.apply_fill(symbol_index=0, quantity_delta=10.0, fill_price=100.0, fee_bps=0.0)
    assert charged == 0.0
    assert state.equity() == 1000.0


def test_causal_state_rejects_unknown_funding_for_held_position() -> None:
    import numpy as np
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=0.0, units=np.array([10.0]), last_marks=np.array([100.0]), last_event_ns=0)
    with pytest.raises(DataIntegrityError, match='funding'):
        state.advance_to(event_ns=1, marks=np.array([100.0]), funding_rates=np.array([0.0]), funding_known=np.zeros(1, dtype=bool))


def test_causal_state_rejects_decreasing_event_time() -> None:
    import numpy as np
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=1000.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=5)
    with pytest.raises(DataIntegrityError, match='monotonic'):
        state.advance_to(event_ns=4, marks=np.array([100.0]), funding_rates=np.zeros(1), funding_known=np.ones(1, dtype=bool))


def test_reconcile_causal_state_rejects_divergence() -> None:
    import numpy as np
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.execution.accounting import CausalPortfolioState, reconcile_causal_state
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult
    index = pd.date_range('2025-01-01', periods=2, freq='3min', tz='UTC')
    ledger = SimulatedInventoryLedgerResult(
        equity=pd.Series([1000.0, 1000.0], index=index, dtype='float64'),
        net_returns=pd.Series([0.0], index=index[1:], dtype='float64'),
        simulated_units=None,
        mark_to_market_pnl=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        funding_charge=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        fee_charge=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        fill_turnover=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        fill_source='OHLCV_IMMEDIATE_TAKER',
        mark_source='OHLCV_CLOSE_FALLBACK',
        primary_valid=True,
        invalid_reasons=(),
    )
    state = CausalPortfolioState(cash=0.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=None)
    with pytest.raises(DataIntegrityError, match='diverged'):
        reconcile_causal_state(state, ledger)
