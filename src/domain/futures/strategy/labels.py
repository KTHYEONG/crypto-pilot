from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.settings import FILLS_PER_ROUND_TRIP
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LabelPanel


def _build_relevance(
    signed_ret: NDArray[np.float32],
    eligible: NDArray[np.bool_],
    min_group_size: int,
) -> NDArray[np.int32]:
    rel = np.zeros(signed_ret.shape, dtype=np.int32)
    for t in range(signed_ret.shape[0]):
        idx = np.flatnonzero(eligible[t] & np.isfinite(signed_ret[t]))
        if idx.size < min_group_size:
            continue
        vals = signed_ret[t, idx]
        q15 = float(np.nanpercentile(vals, 15))
        q35 = float(np.nanpercentile(vals, 35))
        q65 = float(np.nanpercentile(vals, 65))
        q85 = float(np.nanpercentile(vals, 85))
        rel[t, idx] = np.where(
            vals >= q85,
            4,
            np.where(vals >= q65, 3, np.where(vals >= q35, 2, np.where(vals >= q15, 1, 0))),
        )
    return np.asarray(rel, dtype=np.int32)


def build_label_panel(aligned: AlignedMarketData, cfg: StrategyMLConfig) -> LabelPanel:
    """Build t+1 execution aligned label tensors."""
    t_len, n_len = aligned.close_2d.shape
    horizon = cfg.label_horizon_bars
    long_net = np.full((t_len, n_len), np.nan, dtype=np.float32)
    short_net = np.full((t_len, n_len), np.nan, dtype=np.float32)
    eligible = (
        aligned.active_mask
        & aligned.warm_mask
        & ~aligned.entry_block_mask
        & ~aligned.kill_mask
    )
    # Round-trip = 진입 fill + 청산 fill (execution sim은 양 leg 모두 Taker).
    # FILLS_PER_ROUND_TRIP=2 이므로: 2*(fee_bps + slippage_bps) = 14bps (기본값 5+2 per side).
    # [ML-UPGRADE] Gross Alpha 학습을 위해 모델 단에서는 비용 차감을 0.0으로 설정합니다.
    cost = np.float64(0.0)

    for t in range(t_len - horizon):
        entry = aligned.open_2d[t + 1]
        exit_ = aligned.close_2d[t + horizon]
        valid_px = (entry > 0.0) & (exit_ > 0.0) & np.isfinite(entry) & np.isfinite(exit_)
        row_ok = eligible[t] & valid_px
        if not np.any(row_ok):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gross_long = np.log(exit_ / entry)
            gross_short = np.log(entry / exit_)
        funding = aligned.funding_2d[t]
        long_net[t, row_ok] = (gross_long[row_ok] - cost - funding[row_ok]).astype(np.float32)
        short_net[t, row_ok] = (gross_short[row_ok] - cost + funding[row_ok]).astype(np.float32)

    signed = long_net.copy()
    finite_long = np.isfinite(long_net)
    rel = _build_relevance(
        signed_ret=signed,
        eligible=eligible & finite_long,
        min_group_size=cfg.min_group_size,
    )

    liq_weight = np.clip(np.log1p(np.maximum(aligned.volume_2d, 0.0)), 0.25, 2.0)
    sample_weight = np.where(eligible & finite_long, liq_weight, 0.0).astype(np.float32)
    sample_weight = np.clip(sample_weight, 0.0, 2.0)
    return LabelPanel(
        long_net_ret=long_net,
        short_net_ret=short_net,
        signed_net_ret=signed,
        relevance=rel,
        sample_weight=sample_weight,
        eligible_mask=eligible & finite_long,
    )
