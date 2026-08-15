from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.signals.dataset import _resolve_gross_target_bps


class TestDataset:
    """Scenario 10: build_candidate_dataset requires gross target column."""

    def test_build_candidate_dataset_requires_gross_target_column(self) -> None:
        df = pd.DataFrame({"net_event_bps": [1.0, 2.0], "timestamp": [1, 2]})
        with pytest.raises(ValueError, match="gross target column required"):
            _resolve_gross_target_bps(df, allow_label_free=False)

        df2 = pd.DataFrame({"gross_event_bps": [1.0, 2.0]})
        result = _resolve_gross_target_bps(df2, allow_label_free=True)
        assert result is not None
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0], dtype=np.float32))
