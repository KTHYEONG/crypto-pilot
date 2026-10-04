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
from src.application.ops.gdrive_cleanup import _evidence_size_admissible, _scope_rule_of
from src.live.crypto import SEALED_OVERHEAD_BYTES


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


S = f"{OCI_SNAPSHOT}/crypto-pilot"
K = f"{OCI_SNAPSHOT}/krx-alpha"


def _oci_candidates(plan: CleanupPlan) -> list[CleanupCandidate]:
    return [c for c in plan.candidates if c.rule == "oci_server_snapshot"]


def test_build_plan_oci_snapshot_requires_exact_relative_path_counterpart() -> None:
    live_copy = _obj(f"{LIVE_DATA}/live_capture/btc_top.json", 300)
    snap = _obj(f"{S}/live_capture/btc_top.json", 300)
    stray = _obj(f"{S}/live_capture/gone.json", 10)
    krx_copy = _obj(f"{KRX_DATA}/state/universe.json", 60)
    krx_snap = _obj(f"{K}/state/universe.json", 60)
    krx_orphan = _obj(f"{K}/state/missing.json", 10)
    krx_stray = _obj("stray/k.json", 5)
    unknown = _obj(f"{OCI_SNAPSHOT}/unknown/tree/file.json", 10)
    off_prefix = _obj("elsewhere/file.json", 10)
    plan = build_cleanup_plan(
        _listings((live_copy,), (), (snap, stray, krx_snap, krx_orphan, unknown, off_prefix), (krx_copy, krx_stray)),
        _no_vision,
    )
    by_path = {c.obj.path: c for c in plan.candidates}
    assert set(by_path) == {snap.path, krx_snap.path}
    assert by_path[snap.path].evidence == f"live:{live_copy.path} size=300 snapshot_size=300"
    assert by_path[snap.path].evidence_path == live_copy.path
    assert by_path[krx_snap.path].evidence.startswith("live-krx:")
    assert by_path[krx_snap.path].evidence_path == krx_copy.path
    assert (stray, "no_live_counterpart") in plan.kept
    assert (krx_orphan, "no_live_counterpart") in plan.kept
    assert (unknown, "counterpart_root_not_listed") in plan.kept
    assert off_prefix not in [obj for obj, _reason in plan.kept]


def test_build_plan_oci_snapshot_same_basename_in_different_directory_is_kept() -> None:
    live = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 1_000)
    snap = _obj(f"{S}/futures/ohlcv/3m/BTCUSDT.parquet", 90_000_000)
    plan = build_cleanup_plan(_listings((live,), (), (snap,)), _no_vision)
    assert _oci_candidates(plan) == []
    assert (snap, "no_live_counterpart") in plan.kept


def test_build_plan_oci_hourly_capture_partitions_never_match_across_days_or_datasets() -> None:
    live = tuple(_obj(f"{LIVE_DATA}/live_capture/top_of_book/20261002/{h:02d}.parquet", 10) for h in range(24))
    snaps = tuple(
        _obj(f"{S}/live_capture/{dataset}/202609{day:02d}/{h:02d}.parquet", 2_000_000)
        for dataset in ("top_of_book", "premium_index")
        for day in range(1, 31)
        for h in range(24)
    )
    plan = build_cleanup_plan(_listings(live, (), snaps), _no_vision)
    assert len(snaps) == 1440
    assert _oci_candidates(plan) == []
    kept_snaps = [obj for obj, _reason in plan.kept if obj.path.startswith(S + "/")]
    assert len(kept_snaps) == 1440


def test_build_plan_oci_generic_file_names_in_other_runs_are_kept() -> None:
    live = (
        _obj(f"{LIVE_DATA}/state/runs/r1/run_manifest.json", 5),
        _obj(f"{LIVE_DATA}/state/runs/run_new/tax_ledger/tax_ledger_202609.jsonl", 10),
    )
    snaps = (
        _obj(f"{S}/state/runs/r_old/run_manifest.json", 999),
        _obj(f"{S}/state/runs/run_old_oci_only/tax_ledger/tax_ledger_202609.jsonl", 50_000),
    )
    plan = build_cleanup_plan(_listings(live, (), snaps), _no_vision)
    assert plan.candidates == ()


def test_build_plan_oci_immutable_file_requires_identical_size() -> None:
    snap = _obj(f"{S}/state/live_position_ledger.json", 80_000)
    truncated = _obj(f"{LIVE_DATA}/state/live_position_ledger.json", 2)
    plan = build_cleanup_plan(_listings((truncated,), (), (snap,)), _no_vision)
    assert _oci_candidates(plan) == []
    assert (snap, "evidence_size_mismatch") in plan.kept
    identical = _obj(f"{LIVE_DATA}/state/live_position_ledger.json", 80_000)
    matched = build_cleanup_plan(_listings((identical,), (), (snap,)), _no_vision)
    assert [c.obj for c in _oci_candidates(matched)] == [snap]


def test_build_plan_oci_append_only_ledger_accepts_grown_live_copy() -> None:
    snap = _obj(f"{S}/state/runs/r1/audit/2026-09-01.jsonl", 100)
    grown = _obj(f"{LIVE_DATA}/state/runs/r1/audit/2026-09-01.jsonl", 150)
    plan = build_cleanup_plan(_listings((grown,), (), (snap,)), _no_vision)
    assert [c.obj for c in _oci_candidates(plan)] == [snap]
    assert _oci_candidates(plan)[0].evidence_path == grown.path
    shrunk = _obj(f"{LIVE_DATA}/state/runs/r1/audit/2026-09-01.jsonl", 99)
    kept_plan = build_cleanup_plan(_listings((shrunk,), (), (snap,)), _no_vision)
    assert _oci_candidates(kept_plan) == []
    assert (snap, "evidence_size_mismatch") in kept_plan.kept


def test_build_plan_oci_zero_byte_live_copy_is_never_evidence() -> None:
    live = (
        _obj(f"{LIVE_DATA}/futures/liquidations/stream.parquet", 0),
        _obj(f"{LIVE_DATA}/state/runs/r1/audit/2026-09-02.jsonl", 0),
    )
    parquet_snap = _obj(f"{S}/futures/liquidations/stream.parquet", 5_000_000)
    ledger_snap = _obj(f"{S}/state/runs/r1/audit/2026-09-02.jsonl", 0)
    plan = build_cleanup_plan(_listings(live, (), (parquet_snap, ledger_snap)), _no_vision)
    assert plan.candidates == ()
    assert (parquet_snap, "zero_byte_evidence") in plan.kept
    assert (ledger_snap, "zero_byte_evidence") in plan.kept


def test_build_plan_oci_live_duplicate_failing_the_gate_keeps_the_snapshot() -> None:
    live = (
        _obj(f"{LIVE_DATA}/live_capture/btc_top.json", 300),
        _obj(f"{LIVE_DATA}/live_capture/btc_top.json", 0),
    )
    snap = _obj(f"{S}/live_capture/btc_top.json", 300)
    plan = build_cleanup_plan(_listings(live, (), (snap,)), _no_vision)
    assert plan.candidates == ()
    assert (snap, "zero_byte_evidence") in plan.kept


def test_build_plan_oci_live_duplicate_with_other_size_is_size_mismatch() -> None:
    live = (
        _obj(f"{LIVE_DATA}/live_capture/btc_top.json", 300),
        _obj(f"{LIVE_DATA}/live_capture/btc_top.json", 299),
    )
    snap = _obj(f"{S}/live_capture/btc_top.json", 300)
    plan = build_cleanup_plan(_listings(live, (), (snap,)), _no_vision)
    assert plan.candidates == ()
    assert (snap, "evidence_size_mismatch") in plan.kept


def test_build_plan_oci_non_data_subtree_counterpart_is_not_listed() -> None:
    audit = _obj(f"{LIVE_DATA}/state/runs/r1/audit/2026-09-01.jsonl", 5)
    snap = _obj(f"{OCI_SNAPSHOT}/crypto-pilot/logs/live/orders/2026-09-01.jsonl", 5)
    plan = build_cleanup_plan(_listings((audit,), (), (snap,)), _no_vision)
    assert plan.candidates == ()
    assert (snap, "no_live_counterpart") in plan.kept


def test_build_plan_oci_krx_snapshot_without_krx_listing_is_kept() -> None:
    snap = _obj(f"{K}/state/universe.json", 60)
    listings = _listings((), (), (snap,))
    del listings[KRX_DATA]
    plan = build_cleanup_plan(listings, _no_vision)
    assert plan.candidates == ()
    assert plan.kept == ((snap, "counterpart_root_not_listed"),)


def test_build_plan_oci_snapshot_without_project_segment_is_kept() -> None:
    loose = _obj(f"{OCI_SNAPSHOT}/crypto-pilot", 10)
    plan = build_cleanup_plan(_listings((_obj(f"{LIVE_DATA}/x.json", 10),), (), (loose,)), _no_vision)
    assert plan.candidates == ()
    assert plan.kept == ((loose, "counterpart_root_not_listed"),)


def test_build_plan_oci_krx_different_directory_is_kept() -> None:
    krx_live = _obj(f"{KRX_DATA}/state/2024.parquet", 1)
    snap = _obj(f"{K}/prices/2024.parquet", 9_999)
    plan = build_cleanup_plan(_listings((), (), (snap,), (krx_live,)), _no_vision)
    assert plan.candidates == ()
    assert (snap, "no_live_counterpart") in plan.kept


def test_build_plan_oci_unnormalized_snapshot_path_is_kept() -> None:
    target = _obj(f"{LIVE_DATA}/state/x.json", 10)
    snap = _obj(f"{S}/live_capture/../state/x.json", 10)
    plan = build_cleanup_plan(_listings((target,), (), (snap,)), _no_vision)
    assert _oci_candidates(plan) == []
    assert (snap, "unnormalized_path") in plan.kept


def _chained_evidence_listings(reverse: bool = False) -> dict[str, list[RemoteObject]]:
    listings = _listings(
        (
            _obj(f"{LIVE_DATA}/state/runs/r1/fills.parquet", 100),
            _obj(f"{LIVE_DATA}/state/runs/r1/fills.parquet.enc", 100 + SEALED_OVERHEAD_BYTES),
        ),
        (),
        (_obj(f"{S}/state/runs/r1/fills.parquet", 100),),
    )
    if reverse:
        return {root: list(reversed(objs)) for root, objs in reversed(listings.items())}
    return listings


def test_build_plan_evidence_deleted_by_another_rule_demotes_dependent_candidate() -> None:
    plan = build_cleanup_plan(_chained_evidence_listings(), _no_vision)
    plain = _obj(f"{LIVE_DATA}/state/runs/r1/fills.parquet", 100)
    snap = _obj(f"{S}/state/runs/r1/fills.parquet", 100)
    assert _candidates_by_rule(plan, "plaintext_run_artifact") == [plain]
    assert _oci_candidates(plan) == []
    assert (snap, "evidence_is_candidate") in plan.kept
    assert not {c.evidence_path for c in plan.candidates} & {c.obj.path for c in plan.candidates}
    assert cleanup_module._rule_summary(plan)["oci_server_snapshot"]["kept"] == 1


def test_build_plan_evidence_closure_is_order_independent() -> None:
    forward = build_cleanup_plan(_chained_evidence_listings(), _no_vision)
    backward = build_cleanup_plan(_chained_evidence_listings(reverse=True), _no_vision)
    assert forward == backward


def test_build_plan_futures_research_copy_smaller_than_live_is_rejected() -> None:
    live = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 50_000_000)
    research = _obj(f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet", 1)
    plan = build_cleanup_plan(_listings((live,), (research,)), _no_vision)
    assert plan.candidates == ()
    assert plan.kept == ((live, "research_size_mismatch"),)
    vision_plan = build_cleanup_plan(_listings((live,), (research,)), lambda dataset, symbol: True)
    assert len(vision_plan.candidates) == 1
    assert vision_plan.candidates[0].evidence == "vision:BTCUSDT"
    assert vision_plan.candidates[0].evidence_path is None


def test_build_plan_futures_research_copy_equal_or_larger_justifies_deletion() -> None:
    live = _obj(f"{LIVE_DATA}/futures/ohlcv/1h/BTCUSDT.parquet", 500)
    research_path = f"{RESEARCH_FUTURES}/ohlcv/1h/BTCUSDT.parquet"
    for research_size in (500, 600):
        plan = build_cleanup_plan(_listings((live,), (_obj(research_path, research_size),)), _no_vision)
        assert len(plan.candidates) == 1
        assert plan.candidates[0].evidence == f"research:{research_path} size={research_size}"
        assert plan.candidates[0].evidence_path == research_path


def test_build_plan_manual_debug_backup_rejects_much_smaller_counterpart() -> None:
    backup = _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_20260915/order_journal.jsonl", 900_000)
    tiny = _obj(f"{LIVE_DATA}/state/runs/someRun_20260915/order_journal.jsonl", 10)
    plan = build_cleanup_plan(_listings((backup, tiny)), _no_vision)
    assert plan.candidates == ()
    assert (backup, "run_counterpart_size_mismatch") in plan.kept
    grown = _obj(tiny.path, 900_500)
    matched = build_cleanup_plan(_listings((backup, grown)), _no_vision)
    assert _candidates_by_rule(matched, "manual_debug_backup") == [backup]
    assert matched.candidates[0].evidence == f"run-counterpart:{grown.path} size=900500"
    assert matched.candidates[0].evidence_path == grown.path


def test_build_plan_manual_debug_backup_accepts_sealed_counterpart_with_exact_envelope_overhead() -> None:
    backup = _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_20260921/target_weights.parquet", 200)
    sealed_path = f"{LIVE_DATA}/state/runs/frozen_20260921/target_weights.parquet.enc"
    plan = build_cleanup_plan(_listings((backup, _obj(sealed_path, 200 + SEALED_OVERHEAD_BYTES))), _no_vision)
    assert _candidates_by_rule(plan, "manual_debug_backup") == [backup]
    assert plan.candidates[0].evidence_path == sealed_path
    off_by_one = build_cleanup_plan(_listings((backup, _obj(sealed_path, 200 + SEALED_OVERHEAD_BYTES + 1))), _no_vision)
    assert off_by_one.candidates == ()
    assert (backup, "run_counterpart_size_mismatch") in off_by_one.kept


def test_build_plan_manual_debug_backup_prefers_counterpart_that_survives_the_plan() -> None:
    backup = _obj(f"{LIVE_DATA}/state/runs/_manual_debug_backup_20260921/target_weights.parquet", 200)
    plain = _obj(f"{LIVE_DATA}/state/runs/r_20260921/target_weights.parquet", 200)
    sealed = _obj(f"{LIVE_DATA}/state/runs/r_20260921/target_weights.parquet.enc", 200 + SEALED_OVERHEAD_BYTES)
    plan = build_cleanup_plan(_listings((backup, plain, sealed)), _no_vision)
    assert _candidates_by_rule(plan, "plaintext_run_artifact") == [plain]
    manual = [c for c in plan.candidates if c.rule == "manual_debug_backup"]
    assert [c.obj for c in manual] == [backup]
    assert manual[0].evidence_path == sealed.path


def test_evidence_size_admissible_gate_boundaries() -> None:
    parquet = _obj(f"{LIVE_DATA}/a.parquet", 100)
    ledger = _obj(f"{LIVE_DATA}/a.jsonl", 100)
    assert _evidence_size_admissible(parquet, 100)
    assert not _evidence_size_admissible(parquet, 101)
    assert not _evidence_size_admissible(parquet, 99)
    assert _evidence_size_admissible(ledger, 101)
    assert not _evidence_size_admissible(ledger, 99)
    assert not _evidence_size_admissible(_obj(f"{LIVE_DATA}/z.parquet", 0), 0)
    assert not _evidence_size_admissible(_obj(f"{LIVE_DATA}/z.jsonl", 0), 0)
    assert _evidence_size_admissible(parquet, 100 + SEALED_OVERHEAD_BYTES, sealed_evidence=True)
    assert not _evidence_size_admissible(parquet, 100, sealed_evidence=True)


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
    keep_on_delete: frozenset[str] = frozenset(),
) -> object:
    """Fake rclone transport backed by one flat, mutable object registry (paths are absolute,
    relative to DRIVE_REMOTE). ``lsjson <remote>/<root>`` filters by path prefix -- so it answers
    both the canonical listing roots (LIVE_DATA, ...) and the narrower ``_RULE_RMDIR_ROOTS``
    sub-paths apply_plan re-lists for post-delete verification. A batch ``delete --files-from``
    actually removes listed paths from the registry, except paths in ``keep_on_delete``
    (simulates a per-file failure surviving the batch)."""
    registry: dict[str, RemoteObject] = {o.path: o for objs in by_root.values() for o in objs}

    def fake(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if "lsjson" in argv:
            root = argv[-1].split("gdrive:quant-lake/")[1]
            prefix = root + "/"
            entries = [
                {"Path": path[len(prefix):], "Size": obj.size, "IsDir": False}
                for path, obj in registry.items()
                if path.startswith(prefix)
            ]
            return _lsjson_completed(list(argv), json.dumps(entries))
        if "delete" in argv and "--files-from" in argv:
            listed = Path(argv[argv.index("--files-from") + 1]).read_text(encoding="utf-8").splitlines()
            for path in listed:
                if path not in keep_on_delete:
                    registry.pop(path, None)
            return _lsjson_completed(list(argv), "")
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
    assert not any(cmd in call for call in calls for cmd in ("delete", "rmdirs", "cleanup"))
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
    assert all("evidence_path" in c for c in payload["candidates"])
    futures = [c for c in payload["candidates"] if c["rule"] == "futures_legacy"]
    assert [c["evidence_path"] for c in futures] == [research.path]
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
    assert any(call[:2] == ["rclone", "delete"] for call in calls)
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
        _responder(by_root, calls, keep_on_delete=frozenset({first.path})),
    )
    monkeypatch.setattr(cleanup_module, "vision_symbol_probe", lambda: _no_vision)
    monkeypatch.setattr(cleanup_module, "REPORT_DIR", tmp_path)
    assert main(["--apply"]) == 1
    deletes = [call for call in calls if call[:2] == ["rclone", "delete"]]
    assert len(deletes) == 1
    payload = json.loads(next(tmp_path.glob("gdrive_cleanup_*.json")).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_apply_plan_returns_zero_counts_for_empty_plan() -> None:
    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"no rclone call expected for an empty plan: {argv}")

    assert apply_plan(CleanupPlan(candidates=(), kept=()), "rclone", runner=runner) == {
        "deleted": 0,
        "failed": 0,
        "bytes": 0,
    }


def test_apply_plan_reports_partial_failure_after_trying_all() -> None:
    first = CleanupCandidate("futures_legacy", _obj(f"{LIVE_DATA}/futures/a.parquet", 10), "research:x size=10")
    second = CleanupCandidate("futures_legacy", _obj(f"{LIVE_DATA}/futures/b.parquet", 20), "research:y size=20")

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # The batch delete "succeeds" (rc=0) but a is left behind; the post-delete
        # re-listing (scoped to _RULE_RMDIR_ROOTS["futures_legacy"] = LIVE_DATA + "/futures")
        # is what must surface the partial failure.
        if "lsjson" in argv:
            return _lsjson_completed(list(argv), json.dumps([{"Path": "a.parquet", "Size": 10, "IsDir": False}]))
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
