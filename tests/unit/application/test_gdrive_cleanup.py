"""Evidence-gated Drive cleanup invariant guards (in-memory listings, no network)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.application.ops import gdrive_cleanup as cleanup_module
from src.application.ops.gdrive_cleanup import (
    KRX_DATA,
    LEGACY_ARCHIVE,
    LIVE_DATA,
    OCI_SNAPSHOT,
    RESEARCH_FUTURES,
    CleanupCandidate,
    CleanupPlan,
    RemoteObject,
    apply_plan,
    build_cleanup_plan,
    empty_trash,
    list_remote,
    main,
    vision_symbol_probe,
)
from src.application.ops.gdrive_cleanup import _scope_rule_of


def _obj(path: str, size: int = 100) -> RemoteObject:
    return RemoteObject(path=path, size=size)


def _listings(
    live: tuple[RemoteObject, ...] = (),
    research: tuple[RemoteObject, ...] = (),
    oci: tuple[RemoteObject, ...] = (),
    krx: tuple[RemoteObject, ...] = (),
) -> dict[str, list[RemoteObject]]:
    return {LIVE_DATA: list(live), RESEARCH_FUTURES: list(research), OCI_SNAPSHOT: list(oci), KRX_DATA: list(krx)}


def _no_vision(dataset: str, symbol: str) -> bool:
    return False


def _candidates_by_rule(plan: CleanupPlan, rule: str) -> list[RemoteObject]:
    return [c.obj for c in plan.candidates if c.rule == rule]


def test_build_plan_research_copy_justifies_futures_deletion() -> None:
    live = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    research = _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 500)
    plan = build_cleanup_plan(_listings((live,), (research,)), _no_vision)
    assert len(plan.candidates) == 1
    candidate = plan.candidates[0]
    assert candidate.rule == "futures_legacy"
    assert f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet" in candidate.evidence


def test_build_plan_vision_availability_justifies_live_only_symbol() -> None:
    live = _obj(f"{LIVE_DATA}/futures/funding/DOSUSDT.parquet", 40)
    vision = lambda dataset, symbol: (dataset, symbol) == ("fundingRate", "DOSUSDT")  # noqa: E731
    plan = build_cleanup_plan(_listings((live,)), vision)
    assert len(plan.candidates) == 1
    assert plan.candidates[0].rule == "futures_legacy"
    assert plan.candidates[0].evidence == "vision:DOSUSDT"


def test_build_plan_unverifiable_futures_object_is_kept() -> None:
    live = _obj(f"{LIVE_DATA}/futures/funding/DOSUSDT.parquet", 40)
    manifest = _obj(f"{LIVE_DATA}/futures/funding/manifest.json", 5)
    stray_research = _obj("stray/elsewhere.parquet", 5)
    plan = build_cleanup_plan(_listings((live, manifest), (stray_research,)), _no_vision)
    assert plan.candidates == ()
    assert plan.kept == ((live, "no_equivalent_copy"), (manifest, "no_equivalent_copy"))


def test_build_plan_duplicate_same_name_objects_all_removed() -> None:
    first = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    second = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    research = _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 500)
    plan = build_cleanup_plan(_listings((first, second), (research,)), _no_vision)
    assert len(plan.candidates) == 2
    assert {c.obj.path for c in plan.candidates} == {first.path}


def test_build_plan_regenerable_coverage_sidecar_is_candidate() -> None:
    live = _obj(f"{LIVE_DATA}/futures/markPriceKlines/1h/BTCUSDT.coverage.json", 10)
    plan = build_cleanup_plan(_listings((live,)), _no_vision)
    assert len(plan.candidates) == 1
    assert plan.candidates[0].rule == "futures_legacy"


def test_build_plan_legacy_orderbook_needs_identical_size() -> None:
    live = _obj(f"{LIVE_DATA}/state/live_orderbook/ob_20260902.parquet", 100)
    archive = _obj(f"{LEGACY_ARCHIVE}/state/live_orderbook/ob_20260902.parquet", 101)
    plan = build_cleanup_plan(_listings((live, archive)), _no_vision)
    assert plan.candidates == ()
    assert plan.kept == ((live, "archive_size_mismatch"),)
    exact = _obj(f"{LEGACY_ARCHIVE}/state/live_orderbook/ob_20260902.parquet", 100)
    matched = build_cleanup_plan(_listings((live, exact)), _no_vision)
    assert len(matched.candidates) == 1
    assert matched.candidates[0].rule == "legacy_state"


def test_build_plan_appended_ledger_accepts_larger_archive_copy() -> None:
    live = _obj(f"{LIVE_DATA}/state/live_fills/fills_202609.parquet", 19653)
    grown = _obj(f"{LEGACY_ARCHIVE}/state/live_fills/fills_202609.parquet", 22243)
    plan = build_cleanup_plan(_listings((live, grown)), _no_vision)
    assert len(plan.candidates) == 1
    shrunk = _obj(f"{LEGACY_ARCHIVE}/state/live_fills/fills_202609.parquet", 100)
    kept_plan = build_cleanup_plan(_listings((live, shrunk)), _no_vision)
    assert kept_plan.candidates == ()
    assert kept_plan.kept == ((live, "archive_size_mismatch"),)


def test_build_plan_current_live_state_is_never_touched() -> None:
    heartbeat = _obj(f"{LIVE_DATA}/state/live_daemon_last_run.json", 50)
    archive_twin = _obj(f"{LEGACY_ARCHIVE}/state/live_daemon_last_run.json", 50)
    fresh_orderbook = _obj(f"{LIVE_DATA}/state/live_orderbook_20260922.parquet", 70)
    plan = build_cleanup_plan(_listings((heartbeat, archive_twin, fresh_orderbook)), _no_vision)
    touched = [c for c in plan.candidates if c.obj.path in {heartbeat.path, fresh_orderbook.path}]
    assert touched == []
    assert heartbeat not in [obj for obj, _reason in plan.kept]
    assert (fresh_orderbook, "no_archive_counterpart") in plan.kept


def test_build_plan_plaintext_run_artifact_requires_sealed_sibling() -> None:
    plain = _obj(f"{LIVE_DATA}/state/runs/r1/target_weights.parquet", 200)
    sealed = _obj(f"{LIVE_DATA}/state/runs/r1/target_weights.parquet.enc", 236)
    lonely = _obj(f"{LIVE_DATA}/state/runs/r2/decision_ohlcv_close.parquet", 200)
    frozen = _obj(f"{LIVE_DATA}/state/runs/r2/frozen_unit_forward.parquet", 200)
    plan = build_cleanup_plan(_listings((plain, sealed, lonely, frozen)), _no_vision)
    assert _candidates_by_rule(plan, "plaintext_run_artifact") == [plain]
    assert (lonely, "no_sealed_sibling") in plan.kept
    assert (frozen, "no_sealed_sibling") in plan.kept
    assert sealed not in [c.obj for c in plan.candidates]
    assert sealed not in [obj for obj, _reason in plan.kept]


def test_build_plan_manual_debug_backup_requires_dated_run_counterpart() -> None:
    matched = _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_20260921/target_weights.parquet", 200)
    counterpart = _obj(f"{LIVE_DATA}/state/runs/frozen_20260921/target_weights.parquet.enc", 236)
    orphan = _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_20260921/stray.parquet", 10)
    plan = build_cleanup_plan(_listings((matched, counterpart, orphan)), _no_vision)
    assert _candidates_by_rule(plan, "manual_debug_backup") == [matched]
    assert "run-counterpart:" in plan.candidates[0].evidence
    assert (orphan, "no_run_counterpart") in plan.kept


def test_build_plan_oci_snapshot_requires_live_basename_counterpart() -> None:
    live_copy = _obj(f"{LIVE_DATA}/live_capture/btc_top.json", 300)
    snap = _obj(f"{OCI_SNAPSHOT}/crypto-pilot/data/live_capture/btc_top.json", 300)
    stray = _obj(f"{OCI_SNAPSHOT}/crypto-pilot/data/live_capture/gone.json", 10)
    krx_copy = _obj(f"{KRX_DATA}/state/universe.json", 60)
    krx_snap = _obj(f"{OCI_SNAPSHOT}/krx-alpha/data/state/universe.json", 60)
    krx_orphan = _obj(f"{OCI_SNAPSHOT}/krx-alpha/data/state/missing.json", 10)
    krx_stray = _obj("stray/k.json", 5)
    unknown = _obj(f"{OCI_SNAPSHOT}/unknown/tree/file.json", 10)
    off_prefix = _obj("elsewhere/file.json", 10)
    plan = build_cleanup_plan(
        _listings((live_copy,), (), (snap, stray, krx_snap, krx_orphan, unknown, off_prefix), (krx_copy, krx_stray)),
        _no_vision,
    )
    by_path = {c.obj.path: c for c in plan.candidates}
    assert set(by_path) == {snap.path, krx_snap.path}
    assert "size=300 snapshot_size=300" in by_path[snap.path].evidence
    assert by_path[krx_snap.path].evidence.startswith("live-krx:")
    assert (stray, "no_live_counterpart") in plan.kept
    assert (krx_orphan, "no_live_counterpart") in plan.kept
    assert (unknown, "no_live_counterpart") in plan.kept
    assert off_prefix not in [obj for obj, _reason in plan.kept]


def test_build_plan_out_of_scope_roots_are_immune() -> None:
    scoped = (
        _obj(f"{LIVE_DATA}/futures/liquidations/stream.parquet", 10),
        _obj(f"{LIVE_DATA}/futures/venue_rules/rules.json", 10),
        _obj(f"{LIVE_DATA}/live_capture/btc.json", 10),
        _obj(f"{LIVE_DATA}/archive/sealed.parquet", 10),
        _obj("live/crypto-pilot/_versions/2026-09-01/data", 10),
        _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 10),
    )
    plan = build_cleanup_plan(_listings(scoped, (scoped[-1],)), _no_vision)
    assert plan.candidates == ()
    assert plan.kept == ()


def test_build_plan_missing_root_listing_fails_closed() -> None:
    listings = _listings((_obj(f"{LIVE_DATA}/state/a.json"),), ())
    del listings[RESEARCH_FUTURES]
    with pytest.raises(ValueError, match="missing required listing"):
        build_cleanup_plan(listings, _no_vision)


def _lsjson_completed(argv: list[str], stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout=stdout, stderr="")


def test_list_remote_parses_absent_root_and_rejects_errors() -> None:
    calls: list[list[str]] = []

    def ok_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        body = json.dumps([{"Path": "futures/funding/A.parquet", "Size": 7, "IsDir": False}])
        return _lsjson_completed(list(argv), body)

    objects = list_remote("rclone", LIVE_DATA, runner=ok_runner)
    assert objects == [RemoteObject(path=f"{LIVE_DATA}/futures/funding/A.parquet", size=7)]
    assert calls[0][:3] == ["rclone", "lsjson", "-R"]

    def absent_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _lsjson_completed(list(argv), "", returncode=3)

    assert list_remote("rclone", OCI_SNAPSHOT, runner=absent_runner) == []

    def failing_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _lsjson_completed(list(argv), "", returncode=1)

    with pytest.raises(RuntimeError, match="lsjson failed"):
        list_remote("rclone", LIVE_DATA, runner=failing_runner)

    def corrupt_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _lsjson_completed(list(argv), "not-json")

    with pytest.raises(RuntimeError, match="unparseable"):
        list_remote("rclone", LIVE_DATA, runner=corrupt_runner)

    def shape_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _lsjson_completed(list(argv), json.dumps({"Path": "x"}))

    with pytest.raises(RuntimeError, match="unexpected shape"):
        list_remote("rclone", LIVE_DATA, runner=shape_runner)

    def entry_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        body = json.dumps([{"Path": "a.parquet", "Size": 1, "IsDir": False}, {"Path": "d", "Size": 0, "IsDir": True}])
        return _lsjson_completed(list(argv), body)

    assert list_remote("rclone", LIVE_DATA, runner=entry_runner) == [
        RemoteObject(path=f"{LIVE_DATA}/a.parquet", size=1)
    ]

    def bad_entry_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _lsjson_completed(list(argv), json.dumps([{"Size": 1}]))

    with pytest.raises(RuntimeError, match="unexpected entry"):
        list_remote("rclone", LIVE_DATA, runner=bad_entry_runner)


def test_vision_symbol_probe_checks_scopes_memoizes_and_fails_closed() -> None:
    seen: list[str] = []

    def opener(url: str) -> bytes:
        seen.append(url)
        if "monthly/klines/BTCUSDT" in url:
            return b"<ListBucketResult><Key>data/x</Key></ListBucketResult>"
        if "daily/markPriceKlines/TRADFIUSDT" in url:
            return b"<ListBucketResult><Key>data/x</Key></ListBucketResult>"
        return b"<ListBucketResult></ListBucketResult>"

    probe = vision_symbol_probe(opener=opener)
    assert probe("klines", "BTCUSDT") is True
    assert probe("markPriceKlines", "TRADFIUSDT") is True
    assert probe("fundingRate", "NOPEUSDT") is False
    before = len(seen)
    assert probe("klines", "BTCUSDT") is True
    assert len(seen) == before

    def broken_opener(url: str) -> bytes:
        raise ConnectionError("down")

    with pytest.raises(RuntimeError, match="vision probe failed"):
        vision_symbol_probe(opener=broken_opener)("klines", "BTCUSDT")


def test_vision_symbol_probe_percent_encodes_non_ascii_symbol() -> None:
    seen: list[str] = []

    def opener(url: str) -> bytes:
        seen.append(url)
        assert url.isascii()
        return b"<ListBucketResult></ListBucketResult>"

    probe = vision_symbol_probe(opener=opener)
    assert probe("fundingRate", "牛来USDT") is False
    assert any("%E7%89%9B%E6%9D%A5USDT" in url for url in seen)


def test_empty_trash_reports_transport_failure() -> None:
    ok_calls: list[list[str]] = []

    def ok_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        ok_calls.append(list(argv))
        return _lsjson_completed(list(argv), "")

    empty_trash("rclone", runner=ok_runner)
    assert ok_calls == [["rclone", "cleanup", "gdrive:"]]

    def failing_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _lsjson_completed(list(argv), "", returncode=1)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        empty_trash("rclone", runner=failing_runner)


def _responder(
    by_root: dict[str, list[RemoteObject]],
    calls: list[list[str]],
    *,
    delete_rc: dict[str, int] | None = None,
) -> object:
    def fake(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if "lsjson" in argv:
            root = argv[-1].split("gdrive:quant-lake/")[1]
            entries = []
            for o in by_root.get(root, []):
                rel = o.path.split(root + "/", 1)[1] if root + "/" in o.path else o.path
                entries.append({"Path": rel, "Size": o.size, "IsDir": False})
            body = json.dumps(entries)
            return _lsjson_completed(list(argv), body)
        if "deletefile" in argv:
            rc = (delete_rc or {}).get(argv[-1], 0)
            return _lsjson_completed(list(argv), "", returncode=rc)
        return _lsjson_completed(list(argv), "")

    return fake


def test_main_listing_error_aborts_before_any_deletion(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def failing(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return _lsjson_completed(list(argv), "", returncode=1)

    monkeypatch.setattr(subprocess, "run", failing)
    assert main(["--apply"]) == 1
    assert calls
    assert all("lsjson" in call for call in calls)
    assert not any("deletefile" in call for call in calls)


def test_main_dry_run_is_non_mutating_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    research = _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 500)
    extra_live = (
        _obj(f"{LIVE_DATA}/futures/funding/NOPEUSDT.parquet", 40),
        _obj(f"{LIVE_DATA}/state/ledger.json", 30),
        _obj(f"{LIVE_DATA}/state/runs/r9/lonely.parquet", 200),
        _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_2026/w.parquet", 50),
        _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_20260921/orphan.parquet", 10),
        _obj(f"{LIVE_DATA}/state/runs/", 0),
        _obj(f"{LIVE_DATA}/state/runs/r1/sub/f.parquet", 10),
    )
    oci_kept = _obj(f"{OCI_SNAPSHOT}/crypto-pilot/z/gone2.json", 10)
    krx_stray = _obj("stray/k.json", 5)
    calls: list[list[str]] = []
    by_root = {
        LIVE_DATA: [live, *extra_live],
        RESEARCH_FUTURES: [research],
        OCI_SNAPSHOT: [oci_kept],
        KRX_DATA: [krx_stray],
    }
    monkeypatch.setattr(subprocess, "run", _responder(by_root, calls))
    monkeypatch.setattr(cleanup_module, "vision_symbol_probe", lambda: _no_vision)
    monkeypatch.setattr(cleanup_module, "REPORT_DIR", tmp_path)
    assert main([]) == 0
    assert calls
    assert all("lsjson" in call for call in calls)
    assert not any(cmd in call for call in calls for cmd in ("deletefile", "rmdirs", "cleanup"))
    out = capsys.readouterr().out
    assert "[SYS] stage=gdrive_cleanup rule=futures_legacy candidates=1 bytes=500 kept=1" in out
    assert "[SYS] stage=gdrive_cleanup rule=legacy_state candidates=0 bytes=0 kept=1" in out
    assert "[SYS] stage=gdrive_cleanup rule=plaintext_run_artifact candidates=0 bytes=0 kept=2" in out
    assert "[SYS] stage=gdrive_cleanup rule=manual_debug_backup candidates=0 bytes=0 kept=1" in out
    assert "[SYS] stage=gdrive_cleanup rule=oci_server_snapshot candidates=0 bytes=0 kept=1" in out
    reports = list(tmp_path.glob("gdrive_cleanup_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["mode"] == "dry-run"
    assert payload["rules"]["futures_legacy"]["candidates"] == 1
    assert payload["rules"]["plaintext_run_artifact"]["kept"] == 2
    immune = {f"{LIVE_DATA}/state/runs/", f"{LIVE_DATA}/state/runs/r1/sub/f.parquet"}
    assert not immune & {c["path"] for c in payload["candidates"]}
    assert not immune & {k["path"] for k in payload["kept"]}


def test_main_trash_purge_needs_explicit_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _responder({}, calls))
    monkeypatch.setattr(cleanup_module, "REPORT_DIR", tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["--empty-trash"])
    assert exc.value.code == 2
    assert calls == []
    with pytest.raises(SystemExit) as exc2:
        main(["--apply", "--empty-trash"])
    assert exc2.value.code == 2
    assert main(["--empty-trash", "--i-understand-irreversible"]) == 0
    assert calls == [["rclone", "cleanup", "gdrive:"]]


def test_main_trash_failure_returns_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_trash(rclone: str, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cleanup_module, "empty_trash", failing_trash)
    monkeypatch.setattr(cleanup_module, "REPORT_DIR", tmp_path)
    assert main(["--empty-trash", "--i-understand-irreversible"]) == 1


def test_main_apply_deletes_fresh_plan_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    research = _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 500)
    calls: list[list[str]] = []
    by_root = {LIVE_DATA: [live], RESEARCH_FUTURES: [research]}
    monkeypatch.setattr(subprocess, "run", _responder(by_root, calls))
    monkeypatch.setattr(cleanup_module, "vision_symbol_probe", lambda: _no_vision)
    monkeypatch.setattr(cleanup_module, "REPORT_DIR", tmp_path)
    assert main(["--apply"]) == 0
    assert any("deletefile" in call for call in calls)
    assert any("rmdirs" in call for call in calls)
    assert "status=ok mode=apply" in capsys.readouterr().out
    payload = json.loads(next(tmp_path.glob("gdrive_cleanup_*.json")).read_text(encoding="utf-8"))
    assert payload["mode"] == "apply"
    assert payload["counts"] == {"bytes": 500, "deleted": 1, "failed": 0}


def test_main_apply_reports_partial_failure_after_trying_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    second = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/ETHUSDT.parquet", 400)
    research = (
        _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 500),
        _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/ETHUSDT.parquet", 400),
    )
    calls: list[list[str]] = []
    by_root = {LIVE_DATA: [first, second], RESEARCH_FUTURES: list(research)}
    monkeypatch.setattr(
        subprocess,
        "run",
        _responder(by_root, calls, delete_rc={"gdrive:quant-lake/" + first.path: 1}),
    )
    monkeypatch.setattr(cleanup_module, "vision_symbol_probe", lambda: _no_vision)
    monkeypatch.setattr(cleanup_module, "REPORT_DIR", tmp_path)
    assert main(["--apply"]) == 1
    deletes = [call for call in calls if "deletefile" in call]
    assert len(deletes) == 2
    payload = json.loads(next(tmp_path.glob("gdrive_cleanup_*.json")).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_apply_plan_reports_partial_failure_after_trying_all() -> None:
    first = CleanupCandidate("futures_legacy", _obj("live/a.parquet", 10), "research:x size=10")
    second = CleanupCandidate("futures_legacy", _obj("live/b.parquet", 20), "research:y size=20")

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "deletefile" in argv and argv[-1].endswith("a.parquet"):
            return _lsjson_completed(list(argv), "", returncode=1)
        return _lsjson_completed(list(argv), "")

    with pytest.raises(RuntimeError, match="partial failure"):
        apply_plan(CleanupPlan(candidates=(first, second), kept=()), "rclone", runner=runner)


def test_scope_rule_of_leaves_out_of_scope_paths_unattributed() -> None:
    assert _scope_rule_of(f"{LIVE_DATA}/futures/liquidations/s.parquet") is None
    assert _scope_rule_of(f"{LIVE_DATA}/futures/ohlcv/1h/A.parquet") == "futures_legacy"
    assert _scope_rule_of(f"{OCI_SNAPSHOT}/crypto-pilot/f.json") == "oci_server_snapshot"


def test_ops_cli_forwards_gdrive_cleanup_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.main import build_root_parser

    seen: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        seen["argv"] = list(argv or [])
        return int(seen.get("rc", 0))

    monkeypatch.setattr(cleanup_module, "main", fake_main)
    parser = build_root_parser()
    args = parser.parse_args(["ops", "gdrive-cleanup", "--apply"])
    assert args.handler(args) is None
    assert seen["argv"] == ["--apply"]
    trash_args = parser.parse_args(["ops", "gdrive-cleanup", "--empty-trash", "--i-understand-irreversible"])
    assert trash_args.handler(trash_args) is None
    assert seen["argv"] == ["--empty-trash", "--i-understand-irreversible"]
    seen["rc"] = 1
    with pytest.raises(SystemExit) as exc:
        args.handler(args)
    assert exc.value.code == 1
