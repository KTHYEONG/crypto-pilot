"""Preflight gate: GET-only, zero side-effects checks before live execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _load_artifact_frame(artifact_path: Path, artifact_key: Any | None) -> pd.DataFrame:
    if artifact_path.suffix == ".enc":
        if artifact_key is None:
            from src.live.errors import ArtifactSealError

            raise ArtifactSealError(f"sealed artifact requires a key: {artifact_path}")
        from src.live.crypto import derive_key, read_sealed_parquet

        key = derive_key(artifact_key)
        return read_sealed_parquet(artifact_path, key)
    try:
        frame = pd.read_parquet(artifact_path)
    except (FileNotFoundError, OSError) as exc:
        raise DataIntegrityError(f"target weights artifact missing: {artifact_path}") from exc
    return frame


def run_preflight(
    settings: Any,
    artifact_path: Path,
    *,
    now: pd.Timestamp | None = None,
    market_client: Any | None = None,
    order_client: Any | None = None,
) -> PreflightReport:
    """GET-only, zero side effects. Never aborts on first failure."""
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    try:
        now_ts_utc = pd.Timestamp(now_ts)
        now_ts_utc = now_ts_utc.tz_localize("UTC") if now_ts_utc.tzinfo is None else now_ts_utc.tz_convert("UTC")
    except Exception:
        now_ts_utc = pd.Timestamp.now(tz="UTC")
    expected = now_ts_utc.normalize()

    checks: list[PreflightCheck] = []

    # --- artifact_readable ---
    frame: pd.DataFrame | None = None
    try:
        frame = _load_artifact_frame(Path(artifact_path), settings.artifact_key)
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is None:
            raise DataIntegrityError("target weights index must be tz-aware UTC")
        checks.append(PreflightCheck(name="artifact_readable", passed=True, detail=f"rows={len(frame)}"))
    except Exception as exc:
        checks.append(PreflightCheck(name="artifact_readable", passed=False, detail=str(exc)))
        frame = None

    # --- artifact_covers_decision_time ---
    try:
        idx_frame = frame
        if idx_frame is None:
            idx_frame = _load_artifact_frame(Path(artifact_path), settings.artifact_key)
        idx2 = pd.DatetimeIndex(idx_frame.index)
        if idx2.tz is None:
            raise DataIntegrityError("target weights index must be tz-aware UTC")
        if len(idx2) == 0:
            raise DataIntegrityError("target weights artifact is empty")
        latest = idx2.max()
        if expected in idx2:
            passed = True
            staleness = 0.0
        else:
            passed = False
            try:
                staleness = (expected - latest).total_seconds() / 3600.0
            except Exception:
                staleness = float("nan")
        detail = f"expected={expected} latest={latest} staleness_hours={float(staleness):.1f}"
        checks.append(PreflightCheck(name="artifact_covers_decision_time", passed=passed, detail=detail))
    except Exception as exc:
        # If artifact_covers fails due to read error, still ensure detail contains staleness? But we can't.
        # Provide exception text; tests for missing artifact expect this check to be False.
        # Ensure detail at least contains exception.
        checks.append(PreflightCheck(name="artifact_covers_decision_time", passed=False, detail=str(exc)))

    # Resolve market/order clients lazily (default to runner clients) ---
    # We need to avoid creating clients if already injected, to keep GET count bounded.
    # For venue checks we reuse a single exchange payload.

    exchange_payload: Any | None = None
    exchange_detail: str = ""
    # --- venue_exchange_info ---
    try:
        m_client = market_client
        if m_client is None:
            from src.live.runner import _market_client as _runner_market_client

            m_client = _runner_market_client(settings, expected)
        # keep reference for later reuse
        market_client = m_client
        exchange_payload = m_client.exchange_info()
        if not isinstance(exchange_payload, dict):
            raise DataIntegrityError("exchangeInfo returned unexpected schema")
        checks.append(PreflightCheck(name="venue_exchange_info", passed=True, detail="ok"))
        exchange_detail = "ok"
    except Exception as exc:
        checks.append(PreflightCheck(name="venue_exchange_info", passed=False, detail=str(exc)))
        exchange_detail = str(exc)
        exchange_payload = None

    # --- venue_rate_limits ---
    try:
        if exchange_payload is None:
            # No extra GET to keep at most 3 requests; reuse failure reason
            raise DataIntegrityError(f"missing exchange payload: {exchange_detail}")
        from src.live.rest import parse_rate_limits

        parse_rate_limits(exchange_payload)
        checks.append(PreflightCheck(name="venue_rate_limits", passed=True, detail="ok"))
    except Exception as exc:
        checks.append(PreflightCheck(name="venue_rate_limits", passed=False, detail=str(exc)))

    # --- account_configuration ---
    snapshot: Any | None = None
    try:
        from src.live.account import assert_venue_configuration, synthetic_flat_snapshot
        from src.live.runner import NullOrderClient

        o_client = order_client
        if o_client is None:
            from src.live.runner import _order_client as _runner_order_client

            o_client = _runner_order_client(settings, expected)
        order_client = o_client
        if isinstance(o_client, NullOrderClient):
            snapshot = synthetic_flat_snapshot(now_ts_utc)
            assert_venue_configuration(snapshot)
            checks.append(PreflightCheck(name="account_configuration", passed=True, detail="synthetic (paper, no credentials)"))
        else:
            from src.live.account import fetch_account_snapshot

            snapshot = fetch_account_snapshot(o_client, now=now_ts_utc)
            assert_venue_configuration(snapshot)
            checks.append(PreflightCheck(name="account_configuration", passed=True, detail="ok"))
    except Exception as exc:
        checks.append(PreflightCheck(name="account_configuration", passed=False, detail=str(exc)))
        snapshot = None

    # --- position_reconciliation ---
    try:
        if snapshot is None:
            raise DataIntegrityError("missing snapshot: cannot reconcile")
        from src.live.account import assert_suppressed_venue_flat, reconcile_or_halt
        from src.live.ledger import default_ledger_path, load_ledger
        from src.live.runner import _RECONCILE_TOLERANCE_FRACTION

        ledger_path = Path(settings.ledger_path) if settings.ledger_path else default_ledger_path()
        ledger_state = load_ledger(ledger_path)
        ledger_positions = ledger_state.positions
        if settings.mode.suppresses_mutations:
            assert_suppressed_venue_flat(snapshot)
        else:
            reconcile_or_halt(snapshot, ledger_positions, qty_tolerance_fraction=_RECONCILE_TOLERANCE_FRACTION)
        checks.append(PreflightCheck(name="position_reconciliation", passed=True, detail="ok"))
    except Exception as exc:
        checks.append(PreflightCheck(name="position_reconciliation", passed=False, detail=str(exc)))

    return PreflightReport(checks=tuple(checks))
