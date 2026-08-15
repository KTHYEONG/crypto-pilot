from __future__ import annotations

import numpy as np

from src.domain.futures.forecast.contracts import ExitPathRequest


def test_exit_path_request_contract_preserves_array_shapes() -> None:
    request = ExitPathRequest(
        np.array([0]), np.array([1]), np.array([1]), np.array([1]),
        np.array([1.0]), np.array([2.0]), np.array([1]), np.array([0]),
        np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)),
        np.ones((2, 1)), np.ones((2, 1)), np.zeros((2, 1)),
        np.array([np.nan]), np.array([0.0]), 0.0,
    )
    assert request.open_2d.shape == (2, 1)
