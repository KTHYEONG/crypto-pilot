from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_entry_cooldown(
    *,
    tradeable: NDArray[np.bool_],
    active_mask_2d: NDArray[np.bool_] | None,
    t: int,
    cooldown_bars: int,
) -> NDArray[np.bool_]:
    result = np.asarray(tradeable, dtype=np.bool_).copy()
    if cooldown_bars <= 0 or t < 0:
        return result
    if active_mask_2d is None or active_mask_2d.ndim != 2 or active_mask_2d.shape[1] != result.size:
        return result

    for sym_idx in range(result.size):
        if not bool(result[sym_idx]):
            continue
        streak = 0
        back = t
        while back >= 0 and bool(active_mask_2d[back, sym_idx]):
            streak += 1
            back -= 1
        if streak <= cooldown_bars:
            result[sym_idx] = False
    return result
