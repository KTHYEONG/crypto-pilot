"""Unit tests for Phase 1 PIT Universe — registry.py.

Covers build_instrument_registry: OBSERVED rows, RECONSTRUCTED fallback,
validation, deduplication, sorting.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.domain.futures.universe.contracts import DataConfidence
from src.domain.futures.universe.registry import _REGISTRY_COLUMNS, build_instrument_registry

_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_T1 = datetime(2024, 3, 1, tzinfo=UTC)


def _make_snapshot(
    symbols: list[str],
    captured_at: datetime,
    status: str = "TRADING",
    *,
    extra_cols: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows = [
        {
            "symbol": sym,
            "status": status,
            "captured_at": captured_at,
            **(extra_cols or {}),
        }
        for sym in symbols
    ]
    return pd.DataFrame(rows)


def _make_first_obs(
    instrument_ids: list[str],
    symbols: list[str],
    first_observed_at: datetime,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": instrument_ids,
            "symbol": symbols,
            "first_observed_at": [first_observed_at] * len(instrument_ids),
            "last_observed_at": [first_observed_at] * len(instrument_ids),
        }
    )


class TestEmptyInputs:
    def test_empty_raw_snapshots_and_empty_first_obs_returns_empty_df(self) -> None:
        # Arrange
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([], first_observations=empty_first_obs)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        for col in _REGISTRY_COLUMNS:
            assert col in result.columns

    def test_empty_snapshot_skipped(self) -> None:
        # Arrange
        empty_snap = pd.DataFrame(columns=["symbol", "status", "captured_at"])
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([empty_snap], first_observations=empty_first_obs)

        # Assert
        assert result.empty


class TestObservedRows:
    def test_observed_rows_from_raw_snapshot(self) -> None:
        # Arrange
        snap = _make_snapshot(["BTCUSDT", "ETHUSDT"], _T0)
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([snap], first_observations=empty_first_obs)

        # Assert
        assert len(result) == 2
        assert set(result["symbol"]) == {"BTCUSDT", "ETHUSDT"}
        assert (result["confidence"] == DataConfidence.OBSERVED.value).all()
        assert (result["status"] == "TRADING").all()
        for col in _REGISTRY_COLUMNS:
            assert col in result.columns

    def test_observed_rows_instrument_id_format(self) -> None:
        # Arrange
        snap = _make_snapshot(["SOLUSDT"], _T0)
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([snap], first_observations=empty_first_obs)

        # Assert
        assert result.iloc[0]["instrument_id"] == "binance_usdt_perpetual:SOLUSDT"

    def test_instrument_id_includes_onboard_ts_when_provided(self) -> None:
        # Arrange
        snap = _make_snapshot(["DOTUSDT"], _T0, extra_cols={"onboard_at": _T0})
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([snap], first_observations=empty_first_obs)

        # Assert
        expected_ts = int(_T0.timestamp())
        assert result.iloc[0]["instrument_id"] == f"binance_usdt_perpetual:DOTUSDT:{expected_ts}"


class TestReconstructedRows:
    def test_reconstructed_rows_from_first_observations(self) -> None:
        # Arrange — no raw snapshots, one first_obs entry
        empty_first_obs = _make_first_obs(
            ["binance_usdt_perpetual:AVAXUSDT"],
            ["AVAXUSDT"],
            _T0,
        )

        # Act
        result = build_instrument_registry([], first_observations=empty_first_obs)

        # Assert
        assert len(result) == 1
        row = result.iloc[0]
        assert row["instrument_id"] == "binance_usdt_perpetual:AVAXUSDT"
        assert row["confidence"] == DataConfidence.RECONSTRUCTED.value
        assert row["symbol"] == "AVAXUSDT"

    def test_observed_instrument_not_duplicated_from_first_obs(self) -> None:
        # Arrange — BTCUSDT in raw snapshot AND first_obs
        snap = _make_snapshot(["BTCUSDT"], _T0)
        first_obs = _make_first_obs(
            ["binance_usdt_perpetual:BTCUSDT", "binance_usdt_perpetual:LINKUSDT"],
            ["BTCUSDT", "LINKUSDT"],
            _T0,
        )

        # Act
        result = build_instrument_registry([snap], first_observations=first_obs)

        # Assert — BTCUSDT appears only once (OBSERVED), LINKUSDT RECONSTRUCTED
        btc_rows = result[result["symbol"] == "BTCUSDT"]
        assert len(btc_rows) == 1
        assert btc_rows.iloc[0]["confidence"] == DataConfidence.OBSERVED.value

        link_rows = result[result["symbol"] == "LINKUSDT"]
        assert len(link_rows) == 1
        assert link_rows.iloc[0]["confidence"] == DataConfidence.RECONSTRUCTED.value


class TestValidation:
    def test_raw_snapshot_missing_required_columns_raises_value_error(self) -> None:
        # Arrange — missing 'status' column
        bad_snap = pd.DataFrame({"symbol": ["BTCUSDT"], "captured_at": [_T0]})
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act / Assert
        with pytest.raises(ValueError, match="missing required columns"):
            build_instrument_registry([bad_snap], first_observations=empty_first_obs)

    def test_raw_snapshot_missing_symbol_raises_value_error(self) -> None:
        # Arrange — missing 'symbol' column
        bad_snap = pd.DataFrame({"status": ["TRADING"], "captured_at": [_T0]})
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act / Assert
        with pytest.raises(ValueError, match="missing required columns"):
            build_instrument_registry([bad_snap], first_observations=empty_first_obs)


class TestMultipleSnapshotsAndDeduplication:
    def test_no_future_status_applied_to_past_both_rows_present(self) -> None:
        # Arrange — two snapshots at different timestamps with different status
        snap1 = _make_snapshot(["BNBUSDT"], _T0, status="TRADING")
        snap2 = _make_snapshot(["BNBUSDT"], _T1, status="SETTLING")
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([snap1, snap2], first_observations=empty_first_obs)

        # Assert — both rows present because state_valid_from differs
        bnb_rows = result[result["symbol"] == "BNBUSDT"]
        assert len(bnb_rows) == 2
        statuses = set(bnb_rows["status"].tolist())
        assert statuses == {"TRADING", "SETTLING"}

    def test_deduplication_by_instrument_id_and_state_valid_from(self) -> None:
        # Arrange — same symbol, same captured_at → should deduplicate to 1 row
        snap1 = _make_snapshot(["XRPUSDT"], _T0)
        snap2 = _make_snapshot(["XRPUSDT"], _T0)  # identical key
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([snap1, snap2], first_observations=empty_first_obs)

        # Assert — deduplicated to single row
        xrp_rows = result[result["symbol"] == "XRPUSDT"]
        assert len(xrp_rows) == 1

    def test_result_sorted_by_instrument_id_and_state_valid_from(self) -> None:
        # Arrange
        snap = _make_snapshot(["ZRXUSDT", "AAVEUSDT"], _T0)
        empty_first_obs = pd.DataFrame(columns=["instrument_id", "symbol", "first_observed_at", "last_observed_at"])

        # Act
        result = build_instrument_registry([snap], first_observations=empty_first_obs)

        # Assert — sorted by instrument_id ascending
        ids = result["instrument_id"].tolist()
        assert ids == sorted(ids)
