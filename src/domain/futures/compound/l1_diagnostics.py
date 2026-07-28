from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


class L1AdmissionRecorder:
    def __init__(self, path: Path | None = None) -> None:
        self._enabled = os.environ.get("L1_DEBUG") == "1"
        if path is not None:
            self._path = path
        else:
            self._path = Path("logs/l1_admission.jsonl")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            _LOGGER.warning("L1AdmissionRecorder: failed to write %s, disabling", self._path)
            self._enabled = False

    def record_sleeve(
        self, *,
        signal_id: str, fold: int, cluster: int,
        beta: float, se_hac: float, se_ols_ratio: float,
        prob: float, n_obs: int, n_blocks: int, admitted: bool,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.debug(
            "[ALGO] signal_id=%s fold=%d cluster=%d beta=%.3f se_hac=%.3f "
            "se_ols_ratio=%.3f prob=%.3f n_obs=%d n_blocks=%d admitted=%s",
            signal_id, fold, cluster, beta, se_hac, se_ols_ratio,
            prob, n_obs, n_blocks, admitted,
        )
        self._append_jsonl({
            "tag": "ALGO", "signal_id": signal_id, "fold": fold,
            "cluster": cluster, "beta": round(beta, 3),
            "se_hac": round(se_hac, 3), "se_ols_ratio": round(se_ols_ratio, 3),
            "prob": round(prob, 3), "n_obs": n_obs, "n_blocks": n_blocks,
            "admitted": admitted,
        })

    def record_gate(
        self, *,
        admitted_sleeves: int, distinct_series: int, oos_bars: int,
        ann_growth: float, ann_lcb90: float, pw_block: float,
        turnover: float, cost_drag: float,
        positive_folds: int = 0, fold_growths: tuple[float, ...] = (),
        mean_abs_net: float = 0.0, admitted: bool,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.info(
            "[EVAL] admitted_sleeves=%d distinct_series=%d oos_bars=%d "
            "ann_growth=%.4f ann_lcb90=%.4f pw_block=%.2f "
            "turnover=%.4f cost_drag=%.6f positive_folds=%d mean_abs_net=%.4f admitted=%s",
            admitted_sleeves, distinct_series, oos_bars,
            ann_growth, ann_lcb90, pw_block,
            turnover, cost_drag, positive_folds, mean_abs_net, admitted,
        )
        self._append_jsonl({
            "tag": "EVAL", "admitted_sleeves": admitted_sleeves,
            "distinct_series": distinct_series, "oos_bars": oos_bars,
            "ann_growth": round(ann_growth, 4),
            "ann_lcb90": round(ann_lcb90, 4),
            "pw_block": round(pw_block, 2),
            "turnover": round(turnover, 4),
            "cost_drag": round(cost_drag, 6),
            "positive_folds": positive_folds,
            "fold_growths": [round(g, 6) for g in fold_growths],
            "mean_abs_net": round(mean_abs_net, 4),
            "admitted": admitted,
        })
