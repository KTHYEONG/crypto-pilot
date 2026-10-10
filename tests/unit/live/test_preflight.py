"""SCENARIO_LIVE_38/39: preflight 게이트는 GET-only이며 8개 점검을 모두 수행하고
어떤 개별 실패에도 예외를 던지지 않는다(I-PREFLIGHT-TOTAL)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.live.preflight import run_preflight
from src.live.settings import LiveSettings

_EXPECTED_CHECK_NAMES = (
    "artifact_readable",
    "artifact_covers_decision_time",
    "venue_exchange_info",
    "venue_rate_limits",
    "account_configuration",
    "position_reconciliation",
    "venue_leverage_plan",
    "tax_collection_ready",
    "release_gate",
)


class StubMarketClient:
    def exchange_info(self) -> dict[str, Any]:
        return {
            "symbols": [],
            "rateLimits": [
                {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
            ],
        }


class StubOrderClient:
    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        if path == "/fapi/v2/account":
            # Real Binance omits dualSidePosition from this payload.
            return {
                "totalWalletBalance": "2000",
                "availableBalance": "1900",
                "totalInitialMargin": "10",
                "totalUnrealizedProfit": "0",
                "multiAssetsMargin": "false",
            }
        if path == "/fapi/v2/positionRisk":
            return []
        if path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": False}
        raise AssertionError(f"unexpected path {path}")


class StubAllFailClient:
    """모든 호출이 OSError로 실패하며, 변이 메서드 호출 여부를 기록한다."""

    def __init__(self, mutation_calls: list[str]) -> None:
        self._mutation_calls = mutation_calls

    def exchange_info(self) -> Any:
        raise OSError("network unreachable")

    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        raise OSError("network unreachable")

    def sync_server_time(self) -> None:
        raise OSError("network unreachable")

    def open_orders(self) -> list[Any]:
        raise OSError("network unreachable")

    def new_order(self, params: Any) -> Any:
        self._mutation_calls.append("new_order")
        raise AssertionError("preflight must never place an order")

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        self._mutation_calls.append("cancel_order")
        raise AssertionError("preflight must never cancel an order")


def _write_artifact(path, index: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame({"AAAUSDT": [0.02] * len(index)}, index=index)
    frame.to_parquet(path, index=True)


def test_SCENARIO_LIVE_38_PREFLIGHT_FLAGS_STALE_ARTIFACT(tmp_path) -> None:
    now = pd.Timestamp("2026-08-27 00:00Z")
    stale_ts = now.normalize() - pd.Timedelta(days=30)

    stale_artifact = tmp_path / "stale.parquet"
    _write_artifact(stale_artifact, pd.DatetimeIndex([stale_ts]))

    settings = LiveSettings(mode="shadow", ledger_path=str(tmp_path / "ledger.json"))
    report = run_preflight(
        settings,
        stale_artifact,
        now=now,
        market_client=StubMarketClient(),
        order_client=StubOrderClient(),
    )
    by_name = {c.name: c for c in report.checks}
    assert by_name["artifact_readable"].passed is True
    coverage = by_name["artifact_covers_decision_time"]
    assert coverage.passed is False
    assert "staleness_hours=720.0" in coverage.detail

    fresh_artifact = tmp_path / "fresh.parquet"
    _write_artifact(fresh_artifact, pd.DatetimeIndex([now.normalize()]))
    report2 = run_preflight(
        settings,
        fresh_artifact,
        now=now,
        market_client=StubMarketClient(),
        order_client=StubOrderClient(),
    )
    coverage2 = {c.name: c for c in report2.checks}["artifact_covers_decision_time"]
    assert coverage2.passed is True
    assert "staleness_hours=0.0" in coverage2.detail


def test_SCENARIO_LIVE_39_PREFLIGHT_AGGREGATES_ALL_CHECKS_WITHOUT_RAISING(tmp_path) -> None:
    missing_artifact = tmp_path / "does_not_exist.parquet"
    ledger_path = tmp_path / "state" / "ledger.json"
    settings = LiveSettings(mode="shadow", ledger_path=str(ledger_path))

    mutation_calls: list[str] = []
    market_client = StubAllFailClient(mutation_calls)
    order_client = StubAllFailClient(mutation_calls)

    report = run_preflight(
        settings,
        missing_artifact,
        market_client=market_client,
        order_client=order_client,
    )

    assert report.passed is False
    assert len(report.checks) == 9
    assert tuple(c.name for c in report.checks) == (
        "artifact_readable",
        "artifact_covers_decision_time",
        "venue_exchange_info",
        "venue_rate_limits",
        "account_configuration",
        "position_reconciliation",
        "venue_leverage_plan",
        "tax_collection_ready",
        "release_gate",
    )
    assert all(c.passed is False for c in report.checks if c.name not in ("tax_collection_ready", "release_gate"))
    assert {c.name: c for c in report.checks}["tax_collection_ready"].detail == (
        "suppressed mode: simulated tax records"
    )
    assert mutation_calls == []
    assert not ledger_path.exists()


def test_preflight_venue_leverage_plan_reports_pending_changes_in_live_mode(tmp_path) -> None:
    class LeverageAwareClient(StubOrderClient):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path))
            if path == "/fapi/v1/leverageBracket":
                return [{"symbol": "AAAUSDT", "brackets": [
                    {"bracket": 1, "initialLeverage": 20, "notionalCap": 50000, "notionalFloor": 0, "maintMarginRatio": 0.01, "cum": 0},
                ]}]
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0", "marginType": "isolated", "leverage": "20"}]
            return super().request(method, path, params, signed=signed)
    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    settings = LiveSettings(mode="live_testnet", ledger_path=str(tmp_path / "ledger.json"))
    client = LeverageAwareClient()

    report = run_preflight(settings, artifact, now=now, market_client=StubMarketClient(), order_client=client)

    by_name = {c.name: c for c in report.checks}
    assert by_name["venue_leverage_plan"].passed is True
    assert by_name["venue_leverage_plan"].detail == "symbols=1 margin_type_changes=1 leverage_changes=1"
    assert {method for method, _ in client.calls} == {"GET"}
    assert ("GET", "/fapi/v1/leverageBracket") in client.calls


def test_preflight_venue_leverage_plan_skips_venue_in_suppressed_mode(tmp_path) -> None:
    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    settings = LiveSettings(mode="shadow", ledger_path=str(tmp_path / "ledger.json"))

    report = run_preflight(settings, artifact, now=now, market_client=StubMarketClient(), order_client=StubOrderClient())

    by_name = {c.name: c for c in report.checks}
    assert by_name["venue_leverage_plan"].passed is True
    assert by_name["venue_leverage_plan"].detail == "suppressed mode: venue leverage untouched"


def test_preflight_venue_leverage_plan_fails_without_artifact_in_live_mode(tmp_path) -> None:
    class LeverageAwareClient(StubOrderClient):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path))
            if path == "/fapi/v1/leverageBracket":
                return [{"symbol": "AAAUSDT", "brackets": [
                    {"bracket": 1, "initialLeverage": 20, "notionalCap": 50000, "notionalFloor": 0, "maintMarginRatio": 0.01, "cum": 0},
                ]}]
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0", "marginType": "isolated", "leverage": "20"}]
            return super().request(method, path, params, signed=signed)
    now = pd.Timestamp("2026-08-27 00:00Z")
    settings = LiveSettings(mode="live_testnet", ledger_path=str(tmp_path / "ledger.json"))

    report = run_preflight(
        settings, tmp_path / "missing.parquet", now=now,
        market_client=StubMarketClient(), order_client=LeverageAwareClient(),
    )

    by_name = {c.name: c for c in report.checks}
    assert by_name["account_configuration"].passed is True
    assert by_name["venue_leverage_plan"].passed is False
    assert "missing artifact" in by_name["venue_leverage_plan"].detail


def _live_settings_for_tax_gate(tmp_path, mode, *, enabled=True):
    kwargs = {"mode": mode, "ledger_path": str(tmp_path / "ledger.json"), "tax_collection_enabled": enabled}
    if mode in ("live_testnet", "live_mainnet"):
        kwargs.update(order_api_key="k", order_api_secret="s")  # noqa: S106 - hermetic test credential
    if mode == "live_mainnet":
        from src.live.settings import MAINNET_TRADING_ACK

        kwargs["mainnet_trading_ack"] = MAINNET_TRADING_ACK
    return LiveSettings(**kwargs)


def test_tax_collection_check_mainnet_fails_without_collection(tmp_path) -> None:
    from src.live.preflight import run_preflight, tax_collection_check

    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=False)
    check = tax_collection_check(settings, tmp_path / "tax")
    assert check.name == "tax_collection_ready"
    assert check.passed is False
    assert check.detail.startswith("[RISK] live_mainnet requires LIVE_TAX_COLLECTION_ENABLED=true")

    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    report = run_preflight(
        settings, artifact, now=now,
        market_client=StubMarketClient(), order_client=StubOrderClient(),
    )
    assert report.passed is False
    assert any(c.name == "tax_collection_ready" and not c.passed for c in report.checks)


def test_tax_collection_check_testnet_warns_only(tmp_path, caplog) -> None:
    import logging
    from src.live.preflight import run_preflight, tax_collection_check

    settings = _live_settings_for_tax_gate(tmp_path, "live_testnet", enabled=False)
    check = tax_collection_check(settings, tmp_path / "tax")
    assert check.passed is True
    assert check.detail.startswith("WARNING:")

    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    with caplog.at_level(logging.WARNING, logger="LivePreflight"):
        report = run_preflight(
            settings, artifact, now=now,
            market_client=StubMarketClient(), order_client=StubOrderClient(),
        )
    by_name = {c.name: c for c in report.checks}
    assert by_name["tax_collection_ready"].passed is True
    assert by_name["tax_collection_ready"].detail.startswith("WARNING:")
    assert "[RISK]" in caplog.text


def test_tax_collection_check_enabled_and_suppressed_pass(tmp_path) -> None:
    from src.live.preflight import tax_collection_check

    live = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    assert tax_collection_check(live, tmp_path / "tax").passed is True
    paper = _live_settings_for_tax_gate(tmp_path, "paper", enabled=False)
    check = tax_collection_check(paper, tmp_path / "tax")
    assert check.passed is True
    assert check.detail == "suppressed mode: simulated tax records"


def test_preflight_reports_account_scoped_dir(tmp_path, monkeypatch) -> None:
    import src.live.tax_ledger as tax_mod
    from src.live.preflight import run_preflight

    venue_root = tmp_path / "venue"
    monkeypatch.setattr(tax_mod, "default_venue_tax_ledger_root", lambda: venue_root)
    settings = _live_settings_for_tax_gate(tmp_path, "live_testnet", enabled=True)
    settings.record_run_id = "run_a_12345678"
    now = pd.Timestamp("2026-08-27 00:00Z")
    artifact = tmp_path / "weights.parquet"
    _write_artifact(artifact, pd.DatetimeIndex([now.normalize()]))
    report = run_preflight(
        settings, artifact, now=now,
        market_client=StubMarketClient(), order_client=StubOrderClient(),
    )
    check = {c.name: c for c in report.checks}["tax_collection_ready"]
    assert check.detail.endswith(f"tax_dir={venue_root / 'testnet'}")
    assert "runs" not in check.detail



def test_release_gate_mainnet_refused_without_accepted_release(tmp_path) -> None:
    """LIVE_MAINNET fails closed before any venue call when no ACCEPT exists."""
    from src.live.errors import LiveTradingError
    from src.live.preflight import release_gate_check

    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    settings.execution_policy = "strict_passive_repeg"
    check = release_gate_check(settings)
    assert check.name == "release_gate"
    assert check.passed is False
    assert check.detail.startswith("[RISK]")

    venue_calls: list[str] = []

    class NoVenueClient:
        def exchange_info(self) -> None:
            venue_calls.append("exchange_info")
            raise AssertionError("venue must not be touched")

        def request(self, *args: object, **kwargs: object) -> None:
            venue_calls.append("request")
            raise AssertionError("venue must not be touched")

    import pytest

    import src.live.scheduler as sched

    with pytest.raises(LiveTradingError, match="refused to start"):
        sched.run_daemon(
            settings, tmp_path / "w.parquet", tmp_path / "state.json",
            refresh_fn=lambda target: None,
            signal_step_fn=lambda target: None,
            venue_fn=lambda target: None,
            prefetch_fn=lambda target: None,
            prune_fn=lambda: None,
            sleep_fn=lambda seconds: None,
            max_iterations=1,
        )
    assert venue_calls == []


def test_release_gate_paper_never_gated(tmp_path) -> None:
    """Paper mode is an operational check: it passes with verdict=null."""
    from src.live.preflight import release_gate_check

    from src.strategy.release import load_release

    assert load_release("flow_mom_top20").verdict is None
    for mode in ("shadow", "paper", "live_testnet"):
        settings = _live_settings_for_tax_gate(tmp_path, mode, enabled=False)
        check = release_gate_check(settings)
        assert check.passed is True


def _release_root_with_verdict(tmp_path, verdict: str, evaluation_digest=None, execution_policy="strict_passive_repeg"):
    import json
    import shutil

    from src.strategy.release import release_path

    root = tmp_path / "root"
    target = root / "src" / "strategy" / "releases"
    target.mkdir(parents=True)
    shutil.copy(release_path("flow_mom_top20"), target / "flow_mom_top20.json")
    raw = json.loads((target / "flow_mom_top20.json").read_text(encoding="utf-8"))
    import hashlib
    from src.strategy.release import strategy_spec_digest
    from src.strategy.targets import FLOW_MOM_TOP20

    raw["sizing"]["unit_bootstrap_sha256"] = hashlib.sha256(b"bootstrap fixture").hexdigest()
    raw["spec_digest"] = strategy_spec_digest(FLOW_MOM_TOP20, raw["sizing"])
    (target / "flow_mom_top20.json").write_text(json.dumps(raw), encoding="utf-8")
    if verdict != "none":
        raw = json.loads((target / "flow_mom_top20.json").read_text(encoding="utf-8"))
        raw["verdict"] = verdict
        raw["evaluation_digest"] = evaluation_digest
        if execution_policy != "strict_passive_repeg":
            raw["sizing"]["execution_policy"] = execution_policy
        (target / "flow_mom_top20.json").write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    return root


def test_release_gate_accepts_matching_accepted_release(tmp_path) -> None:
    """An ACCEPT whose digests match the daemon book passes the mainnet gate."""
    from src.live.preflight import release_gate_check
    from src.strategy.release import load_release, record_acceptance

    root = _release_root_with_verdict(tmp_path, "none")
    release = load_release("flow_mom_top20", root=root)
    record_acceptance(
        "flow_mom_top20", spec_digest=release.spec_digest,
        evaluation_digest="eval-abc", root=root,
    )
    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    settings.execution_policy = "strict_passive_repeg"
    bootstrap = tmp_path / "unit.parquet"
    bootstrap.write_bytes(b"bootstrap fixture")
    settings.unit_bootstrap_path = str(bootstrap)
    settings.notional_equity_usdt = 1000
    check = release_gate_check(settings, root=root)
    assert check.passed is True
    assert "verdict=accept" in check.detail
    settings.notional_equity_usdt = 4200
    check = release_gate_check(settings, root=root)
    assert check.passed is False
    assert "capital" in check.detail
    settings.notional_equity_usdt = 1000
    bootstrap.write_bytes(b"changed bootstrap")
    check = release_gate_check(settings, root=root)
    assert check.passed is False
    assert "digest" in check.detail


def test_release_gate_rejects_digest_and_policy_mismatch(tmp_path) -> None:
    """A tampered spec digest or a foreign execution policy fails the gate."""
    import json

    from src.live.preflight import release_gate_check

    root = _release_root_with_verdict(tmp_path, "accept", evaluation_digest="eval-abc")
    raw = json.loads((root / "src" / "strategy" / "releases" / "flow_mom_top20.json").read_text(encoding="utf-8"))
    raw["spec_digest"] = "0" * 64
    (root / "src" / "strategy" / "releases" / "flow_mom_top20.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8"
    )
    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    settings.execution_policy = "strict_passive_repeg"
    assert release_gate_check(settings, root=root).passed is False

    root2 = _release_root_with_verdict(tmp_path / "other", "accept", evaluation_digest="eval-abc")
    accepted = root2 / "src" / "strategy" / "releases" / "flow_mom_top20.json"
    raw = json.loads(accepted.read_text(encoding="utf-8"))
    raw["sizing"]["execution_policy"] = "taker_parity"
    accepted.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    from src.strategy.release import load_release, strategy_spec_digest
    from src.strategy.targets import FLOW_MOM_TOP20

    reloaded = load_release("flow_mom_top20", root=root2)
    assert reloaded.spec_digest != strategy_spec_digest(FLOW_MOM_TOP20, dict(reloaded.sizing))
    assert release_gate_check(settings, root=root2).passed is False


def test_release_gate_policy_mismatch_with_matching_digest(tmp_path) -> None:
    """Same accepted digest but a different daemon policy still refuses mainnet."""
    from src.live.preflight import release_gate_check
    from src.strategy.release import load_release, record_acceptance

    root = _release_root_with_verdict(tmp_path, "none")
    release = load_release("flow_mom_top20", root=root)
    record_acceptance(
        "flow_mom_top20", spec_digest=release.spec_digest,
        evaluation_digest="eval-abc", root=root,
    )
    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    settings.execution_policy = "taker_parity"
    bootstrap = tmp_path / "unit.parquet"
    bootstrap.write_bytes(b"bootstrap fixture")
    settings.unit_bootstrap_path = str(bootstrap)
    settings.notional_equity_usdt = 1000
    check = release_gate_check(settings, root=root)
    assert check.passed is False
    assert "execution policy" in check.detail


def test_release_gate_fails_closed_without_record(tmp_path) -> None:
    """A missing release record fails closed, never open."""
    from src.live.preflight import release_gate_check

    settings = _live_settings_for_tax_gate(tmp_path, "live_mainnet", enabled=True)
    check = release_gate_check(settings, root=tmp_path / "nope")
    assert check.passed is False
    assert check.detail.startswith("[RISK] release gate failed closed")
