from __future__ import annotations

import gc
import logging
from collections.abc import MutableMapping
from dataclasses import dataclass

from src.domain.futures.strategy.candidate_dataset import release_aligned_feature_cache
from src.domain.futures.strategy.common.alignment import AlignedMarketData, clear_aligned_data_maps_cache

logger = logging.getLogger(__name__)

PERF = 15


@dataclass(frozen=True, slots=True)
class TfResourceReleaseReport:
    tf: str
    feature_cache_bytes: int
    removed_from_per_tf_map: bool
    primary_retained: bool


def release_completed_tf_resources(
    *,
    tf: str,
    aligned_tf: AlignedMarketData,
    primary_aligned: AlignedMarketData,
    per_tf_aligned: MutableMapping[str, AlignedMarketData] | None,
) -> TfResourceReleaseReport:
    feature_cache_bytes = 0
    removed_from_per_tf_map = False
    primary_retained = aligned_tf is primary_aligned

    try:
        feature_cache_bytes = release_aligned_feature_cache(aligned_tf)
    except Exception:
        logger.warning("[TF-LIFECYCLE] feature cache release failed tf=%s", tf)

    try:
        clear_aligned_data_maps_cache()
    except Exception:
        logger.warning("[TF-LIFECYCLE] data maps cache clear failed tf=%s", tf)

    if per_tf_aligned is not None and tf in per_tf_aligned:
        if aligned_tf is primary_aligned:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[TF-LIFECYCLE] retaining primary tf=%s in per_tf map for L2", tf
                )
        else:
            per_tf_aligned.pop(tf, None)
            removed_from_per_tf_map = True

    gc.collect()

    logger.log(
        PERF,
        "[PERF] stage=tf_resource_release tf=%s feature_cache_kb=%d removed_from_map=%s primary_retained=%s",
        tf,
        feature_cache_bytes // 1024,
        removed_from_per_tf_map,
        primary_retained,
    )
    logger.debug(
        "[MEM] stage=tf_resource_release tf=%s feature_cache_bytes=%d",
        tf,
        feature_cache_bytes,
    )

    return TfResourceReleaseReport(
        tf=tf,
        feature_cache_bytes=feature_cache_bytes,
        removed_from_per_tf_map=removed_from_per_tf_map,
        primary_retained=primary_retained,
    )
