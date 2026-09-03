"""Diagnostic-only leverage-frontier scan: artifact loading + read-only run."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs.leverage_scan import (
    _load_pre_oos_reference_returns,
    run_leverage_frontier_scan,
)
from src.mhs.params import COMMITTEE_OOS_START, GROWTH_RISK_ENVELOPES, PNL_VOL_TARGET_BURN_IN_DAYS

_REFERENCE_REPLAY_ID = "blend_pre_vol_target_reference"


def _daily_ledger_frame(
    replay_ids: tuple[str, ...],
    start: str,
    periods: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Mirror the compact-tier daily_ledger.parquet schema (persist.py rollup)."""
    dates = pd.date_range(start, periods=periods, freq="1D", tz="UTC")
    frames: list[pd.DataFrame] = []
    for offset, replay_id in enumerate(replay_ids):
        rng = np.random.default_rng(seed + offset)
        frame = pd.DataFrame(
            {
                "replay_id": replay_id,
                "date": dates,
                "equity_open": 1.0,
                "equity_high": 1.01,
                "equity_low": 0.99,
                "equity_close": 1.005,
                "daily_return": rng.normal(0.001, 0.01, periods),
                "daily_turnover": 0.0,
                "daily_fill_count": 0,
            },
        )
        frame.loc[frame.index[0], "daily_return"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestLoadPreOosReferenceReturns:
    # SCENARIO_MHS_LEVERAGE_SCAN_04
    def test_missing_artifact_fails_closed_scenario_mhs_leverage_scan_04(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _load_pre_oos_reference_returns(tmp_path / "absent" / "daily_ledger.parquet")

    def test_missing_reference_replay_id_fails_closed_scenario_mhs_leverage_scan_04(self, tmp_path: Path) -> None:
        path = tmp_path / "daily_ledger.parquet"
        _daily_ledger_frame(("blend_primary",), "2022-06-01", 395).to_parquet(path)
        with pytest.raises(ValueError, match=_REFERENCE_REPLAY_ID):
            _load_pre_oos_reference_returns(path)

    def test_short_pre_oos_slice_fails_closed_scenario_mhs_leverage_scan_04(self, tmp_path: Path) -> None:
        path = tmp_path / "daily_ledger.parquet"
        # Only December 2022 precedes the OOS start: far below the burn-in.
        _daily_ledger_frame((_REFERENCE_REPLAY_ID,), "2022-12-01", 91).to_parquet(path)
        with pytest.raises(ValueError, match=f"{PNL_VOL_TARGET_BURN_IN_DAYS}"):
            _load_pre_oos_reference_returns(path)

    def test_valid_fixture_returns_strictly_pre_oos_slice_scenario_mhs_leverage_scan_04(self, tmp_path: Path) -> None:
        path = tmp_path / "daily_ledger.parquet"
        frame = _daily_ledger_frame(
            ("blend_primary", _REFERENCE_REPLAY_ID), "2022-06-01", 395,
        )
        frame.to_parquet(path)
        series = _load_pre_oos_reference_returns(path)
        assert isinstance(series, pd.Series)
        assert (series.index < COMMITTEE_OOS_START).all()
        assert np.isfinite(series.to_numpy()).all()
        expected = (
            frame.loc[frame["replay_id"] == _REFERENCE_REPLAY_ID]
            .set_index("date")["daily_return"]
            .dropna()
        )
        expected = expected.loc[expected.index < COMMITTEE_OOS_START]
        assert len(series) == len(expected)


class TestRunLeverageFrontierScan:
    @pytest.fixture(autouse=True)
    def _valid_artifact(self, tmp_path: Path) -> None:
        self.artifact_dir = tmp_path
        self.artifact_path = tmp_path / "daily_ledger.parquet"
        _daily_ledger_frame(
            ("blend_primary", _REFERENCE_REPLAY_ID), "2022-06-01", 395,
        ).to_parquet(self.artifact_path)

    @staticmethod
    def _snapshot(root: Path) -> dict[str, int]:
        return {
            p.name: p.stat().st_mtime_ns for p in sorted(root.rglob("*")) if p.is_file()
        }

    # SCENARIO_MHS_LEVERAGE_SCAN_05
    def test_points_in_input_order_without_writes_scenario_mhs_leverage_scan_05(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        before = self._snapshot(self.artifact_dir)
        with caplog.at_level(logging.INFO, logger="MhsLeverageFrontierScan"):
            points = run_leverage_frontier_scan(
                "growth", (1.0, 2.0), artifact_path=str(self.artifact_path),
            )
        after = self._snapshot(self.artifact_dir)
        assert before == after
        assert [p.multiple for p in points] == [1.0, 2.0]
        eval_lines = [
            record.message for record in caplog.records
            if record.message.startswith("[EVAL] leverage_frontier_scan")
        ]
        assert len(eval_lines) == 2
        assert all(
            re.search(r"envelope=growth multiple=\d+\.\d{2} ", line) for line in eval_lines
        )

    def test_unknown_envelope_names_registered_keys_scenario_mhs_leverage_scan_05(self) -> None:
        before = self._snapshot(self.artifact_dir)
        with pytest.raises(ValueError, match="not_a_real_envelope") as excinfo:
            run_leverage_frontier_scan(
                "not_a_real_envelope", (1.0,), artifact_path=str(self.artifact_path),
            )
        for key in sorted(GROWTH_RISK_ENVELOPES):
            assert key in str(excinfo.value)
        assert self._snapshot(self.artifact_dir) == before
