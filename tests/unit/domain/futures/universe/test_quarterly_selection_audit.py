from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.optimization.opt_config import get_quarterly_window
from src.domain.futures.universe import (
    Stage6Config,
    UniverseConfig,
    build_universe,
    hash_config,
    load_universe_snapshot,
)
from src.domain.futures.universe.models import RejectCode

_COMMON_ROW = {
    "tf": "4h",
    "contract_type": "PERPETUAL",
    "quote_asset": "USDT",
    "margin_asset": "USDT",
    "status": "TRADING",
    "contract_multiplier": 1.0,
    "has_kline": True,
    "has_funding": True,
    "is_coverage": 1.0,
    "n_is_bars": 4_320,
    "expected_is_bars": 4_320,
    "n_bar_gaps": 0,
    "last_60d_coverage": 0.99,
    "n_zero_volume_bars_60d": 0,
    "frozen_bars": 0,
    "has_nan": False,
    "has_inf": False,
    "has_timestamp_issues": False,
    "taker_fee_bps": 5.0,
    "tick_size": 0.01,
    "mark_price": 100.0,
    "tick_cost_bps": 0.5,
    "listing_age_days": 365,
    "funding_rate_8h": 0.001,
    "funding_zscore": 0.2,
    "basis_z_score": 0.1,
    "oi_usdt_median": 15_000_000.0,
    "risk_event_override": "",
    "vol_30d": 0.05,
    "amihud_30d": 5e-10,
    "screening_clip_usdt": 10_000.0,
    "impact_bps": 1.0,
    "half_spread_bps": 1.0,
}


class QuarterlyExpectation(TypedDict):
    """Expected universe selection state for one backtest quarter."""

    as_of: str
    selected: tuple[str, ...]
    ranked_out: str


def _quarter_row(
    *,
    symbol: str,
    as_of: str,
    adv_usdt_median: float,
    execution_bias_bps: float,
) -> dict[str, object]:
    row = dict(_COMMON_ROW)
    row.update(
        {
            "symbol": symbol,
            "date": as_of,
            "knowledge_date": as_of,
            "is_listed": True,
            "is_trading": True,
            "adv_usdt_median": adv_usdt_median,
            "amihud_30d": 5e-10,
            # Keep the execution-cost ordering stable by varying spread/impact inputs.
            "half_spread_bps": execution_bias_bps,
            "impact_bps": execution_bias_bps,
        }
    )
    return row


def _write_ledger(path: Path) -> None:
    rows = [
        # 2025-01-01 snapshot: BTC + ETH selected, XRP ranked out.
        _quarter_row(
            symbol="BTC/USDT",
            as_of="2025-01-01",
            adv_usdt_median=120_000_000.0,
            execution_bias_bps=0.8,
        ),
        _quarter_row(
            symbol="ETH/USDT",
            as_of="2025-01-01",
            adv_usdt_median=80_000_000.0,
            execution_bias_bps=1.0,
        ),
        _quarter_row(
            symbol="XRP/USDT",
            as_of="2025-01-01",
            adv_usdt_median=32_000_000.0,
            execution_bias_bps=3.5,
        ),
        # 2025-04-01 snapshot: BTC + XRP selected, ETH ranked out.
        _quarter_row(
            symbol="BTC/USDT",
            as_of="2025-04-01",
            adv_usdt_median=120_000_000.0,
            execution_bias_bps=0.8,
        ),
        _quarter_row(
            symbol="ETH/USDT",
            as_of="2025-04-01",
            adv_usdt_median=30_000_000.0,
            execution_bias_bps=4.0,
        ),
        _quarter_row(
            symbol="XRP/USDT",
            as_of="2025-04-01",
            adv_usdt_median=140_000_000.0,
            execution_bias_bps=0.7,
        ),
        # 2025-07-01 snapshot: BTC + ETH selected again.
        _quarter_row(
            symbol="BTC/USDT",
            as_of="2025-07-01",
            adv_usdt_median=120_000_000.0,
            execution_bias_bps=0.8,
        ),
        _quarter_row(
            symbol="ETH/USDT",
            as_of="2025-07-01",
            adv_usdt_median=150_000_000.0,
            execution_bias_bps=0.6,
        ),
        _quarter_row(
            symbol="XRP/USDT",
            as_of="2025-07-01",
            adv_usdt_median=28_000_000.0,
            execution_bias_bps=3.8,
        ),
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_quarterly_universe_selection_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.domain.futures.universe import pipeline as universe_pipeline
    from src.domain.futures.universe import store as universe_store

    ledger_path = tmp_path / "universe_ledger.parquet"
    snapshot_root = tmp_path / "results" / "universe"
    store_root = tmp_path / "results" / "store"
    monkeypatch.setattr(universe_pipeline, "DEFAULT_UNIVERSE_STORE_ROOT", store_root)
    monkeypatch.setattr(universe_store, "DEFAULT_UNIVERSE_STORE_ROOT", store_root)
    _write_ledger(ledger_path)

    cfg = UniverseConfig(
        stage6=Stage6Config(
            k_in=2,
            k_out=2,
            anchor_symbols=("BTC/USDT",),
            basket_ref=("BTC/USDT", "ETH/USDT", "XRP/USDT"),
            basket_weights=(0.5, 0.3, 0.2),
        )
    )

    quarterly_expectations: dict[str, QuarterlyExpectation] = {
        "2025-07-01": {
            "as_of": "2025-01-01",
            "selected": ("BTC/USDT", "ETH/USDT"),
            "ranked_out": "XRP/USDT",
        },
        "2025-10-01": {
            "as_of": "2025-04-01",
            "selected": ("BTC/USDT", "XRP/USDT"),
            "ranked_out": "ETH/USDT",
        },
        "2026-01-01": {
            "as_of": "2025-07-01",
            "selected": ("BTC/USDT", "ETH/USDT"),
            "ranked_out": "XRP/USDT",
        },
    }

    observed: dict[str, tuple[str, ...]] = {}
    for reference_date, expectation in quarterly_expectations.items():
        _fetch_start, _start, is_end, _end = get_quarterly_window(reference_date)
        assert is_end == expectation["as_of"]

        snapshot, selected_frame, report = build_universe(
            as_of=is_end,
            tf="4h",
            cfg=cfg,
            ledger_path=ledger_path,
            snapshot_root=snapshot_root,
        )

        selected_symbols = tuple(selected_frame["symbol"].astype(str).tolist())
        observed[is_end] = selected_symbols
        assert selected_symbols == expectation["selected"]
        assert tuple(meta.symbol for meta in snapshot.selected) == expectation["selected"]
        assert all(meta.vol_30d >= 0.0 for meta in snapshot.selected)
        assert all(
            np.isfinite(meta.tradeable_score) or meta.tradeable_score == float("-inf")
            for meta in snapshot.selected
        )

        ranked_out = expectation["ranked_out"]
        rejected = snapshot.rejected[ranked_out]
        assert rejected.stage3_reason == RejectCode.LOW_LIQUIDITY
        assert rejected.stage6_reason is None
        assert rejected.audit_trail[-1].startswith("stage3_liquidity:FAIL:adv_too_low")
        rejected_symbols = set(report.loc[~report["passed"].astype(bool), "symbol"].astype(str))
        assert ranked_out in rejected_symbols
        assert (
            load_universe_snapshot(as_of=is_end, tf="4h", snapshot_root=snapshot_root) is not None
        )

    assert observed == {
        "2025-01-01": ("BTC/USDT", "ETH/USDT"),
        "2025-04-01": ("BTC/USDT", "XRP/USDT"),
        "2025-07-01": ("BTC/USDT", "ETH/USDT"),
    }  # is_end keys: oos_start from OOS=6M window


def test_load_or_build_universe_snapshot_rebuilds_on_config_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.domain.futures.universe import pipeline as universe_pipeline
    from src.domain.futures.universe import store as universe_store

    ledger_path = tmp_path / "universe_ledger.parquet"
    snapshot_root = tmp_path / "results" / "universe"
    store_root = tmp_path / "results" / "store"
    monkeypatch.setattr(universe_pipeline, "DEFAULT_UNIVERSE_STORE_ROOT", store_root)
    monkeypatch.setattr(universe_store, "DEFAULT_UNIVERSE_STORE_ROOT", store_root)
    _write_ledger(ledger_path)

    base_cfg = UniverseConfig(
        stage6=Stage6Config(
            k_in=2,
            k_out=2,
            anchor_symbols=("BTC/USDT",),
            basket_ref=("BTC/USDT", "ETH/USDT", "XRP/USDT"),
            basket_weights=(0.5, 0.3, 0.2),
        )
    )
    build_universe(
        as_of="2025-01-01",
        tf="4h",
        cfg=base_cfg,
        ledger_path=ledger_path,
        snapshot_root=snapshot_root,
    )

    rebuild_cfg = UniverseConfig(
        stage6=Stage6Config(
            k_in=3,
            k_out=2,
            anchor_symbols=("BTC/USDT",),
            basket_ref=("BTC/USDT", "ETH/USDT", "XRP/USDT"),
            basket_weights=(0.5, 0.3, 0.2),
        )
    )

    called: list[bool] = []
    original_build_universe = universe_pipeline.build_universe

    def wrapped_build_universe(
        *,
        as_of: str | date,
        tf: str,
        cfg: dict[str, Any] | UniverseConfig | None = None,
        ledger_path: Path = ledger_path,
        snapshot_root: Path = snapshot_root,
        previous_selection: tuple[str, ...] | None = None,
    ) -> tuple[object, object, object]:
        called.append(True)
        return original_build_universe(
            as_of=as_of,
            tf=tf,
            cfg=cfg,
            ledger_path=ledger_path,
            snapshot_root=snapshot_root,
            previous_selection=previous_selection,
        )

    monkeypatch.setattr(universe_pipeline, "build_universe", wrapped_build_universe)

    snapshot, selected_frame, report = universe_pipeline.load_or_build_universe_snapshot(
        as_of="2025-01-01",
        tf="4h",
        cfg=rebuild_cfg,
        ledger_path=ledger_path,
        snapshot_root=snapshot_root,
    )

    assert called == [True]
    assert snapshot.config_hash == hash_config(rebuild_cfg)
    assert not selected_frame.empty
    assert not report.empty
