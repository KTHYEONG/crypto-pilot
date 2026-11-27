from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.cli.commands import archive_mirror
from src.cli.main import build_root_parser
from src.market_data.mirror.build import BuildReport
from tests.unit.market_data.mirror.conftest import FakeVision, zip_of

HEADER = b"timestamp,percentage,depth,notional\n"


def _report(**overrides: object) -> BuildReport:
    values: dict[str, object] = {
        "built_objects": 0,
        "built_sources": 0,
        "verbatim_sources": 0,
        "failed_sources": 0,
        "pending_sources": 0,
        "outbox_bytes": 0,
        "more": False,
        "capacity_blocked": None,
        "anomalies": (),
        "listing_failed": (),
    }
    values.update(overrides)
    return BuildReport(**values)  # type: ignore[arg-type]


def _parse(argv: list[str]):  # type: ignore[no-untyped-def]
    return build_root_parser(argv).parse_args(argv)


def test_mirror_subcommand_registered() -> None:
    args = _parse(["archive", "mirror", "build", "--root", "mirror-root"])
    assert args.handler is archive_mirror._run_build
    assert _parse(["archive", "mirror", "format-doc"]).handler is archive_mirror._run_format_doc


def test_dry_run_writes_no_outbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.market_data.binance.vision as vision

    fake_vision = FakeVision()
    monkeypatch.setattr(vision, "BinanceVisionDownloader", lambda: fake_vision)
    fake_vision.put(
        "data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-01-01.zip",
        zip_of(HEADER + b"2024-01-01 00:00:00,-5.00,1.0,2.0"),
    )
    root = tmp_path / "mirror"
    args = _parse(["archive", "mirror", "build", "--root", str(root), "--datasets", "bookDepth"])

    args.handler(args)

    assert not (root / "outbox").exists()
    assert not (root / "index").exists()


def test_execute_requires_byte_inputs(tmp_path: Path) -> None:
    args = _parse(["archive", "mirror", "build", "--root", str(tmp_path), "--execute"])
    with pytest.raises(SystemExit):
        args.handler(args)


@pytest.mark.parametrize(
    ("report", "code"),
    [
        (_report(capacity_blocked="BUDGET", failed_sources=1, anomalies=("format_drift:x",)), 4),
        (_report(anomalies=("format_drift:x",), failed_sources=1), 5),
        (_report(failed_sources=2), 3),
        (_report(listing_failed=("metrics",)), 3),
        (_report(), 0),
    ],
)
def test_exit_code_precedence(report: BuildReport, code: int) -> None:
    assert archive_mirror._build_exit_code(report) == code


def test_status_stale_exit_6(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status_dir = tmp_path / "_status"
    status_dir.mkdir()
    old = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
    (status_dir / "mirror_status.json").write_text(
        json.dumps({"first_build_at": old, "pending_sources": {"bookDepth": 3}}), encoding="utf-8"
    )
    args = _parse(["archive", "mirror", "status", "--root", str(tmp_path), "--max-confirm-age-hours", "12"])

    with pytest.raises(SystemExit) as stale:
        args.handler(args)

    assert stale.value.code == 6
    assert "pending_sources_total" in capsys.readouterr().out


def test_status_fresh_confirm_exit_zero(tmp_path: Path) -> None:
    status_dir = tmp_path / "_status"
    status_dir.mkdir()
    fresh = datetime.now(UTC).isoformat()
    (status_dir / "mirror_status.json").write_text(
        json.dumps({"last_confirmed_at": fresh, "pending_sources": {"bookDepth": 3}}), encoding="utf-8"
    )
    args = _parse(["archive", "mirror", "status", "--root", str(tmp_path), "--max-confirm-age-hours", "12"])
    args.handler(args)


def test_root_defaults_to_mirror_dir_and_config_overrides(tmp_path: Path) -> None:
    from src.common.paths import ARCHIVE_MIRROR_DIR

    default = _parse(["archive", "mirror", "build"])
    assert archive_mirror._root(default) == ARCHIVE_MIRROR_DIR

    tuned = _parse(
        [
            "archive",
            "mirror",
            "build",
            "--root",
            str(tmp_path),
            "--datasets",
            "bookDepth",
            "--drive-budget-gib",
            "7",
            "--outbox-cap-gib",
            "3",
            "--download-workers",
            "2",
        ]
    )
    config = archive_mirror._config(tuned)
    assert config.root == tmp_path
    assert config.datasets == ("bookDepth",)
    assert (config.drive_budget_gib, config.outbox_cap_gib, config.download_workers) == (7, 3, 2)


def test_deadline_parsing_is_today_utc() -> None:
    now = datetime(2024, 3, 2, 9, 30, tzinfo=UTC)
    parsed = archive_mirror._parse_deadline("10:15", now)
    assert parsed == datetime(2024, 3, 2, 10, 15, tzinfo=UTC)


def test_passed_deadline_builds_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _parse(["archive", "mirror", "build", "--root", str(tmp_path), "--execute", "--deadline", "00:00"])
    args.handler(args)
    out = capsys.readouterr().out
    assert '"more":true' in out
    assert not (tmp_path / "outbox").exists()


def test_busy_lock_exits_75(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.market_data.public_archive import store

    monkeypatch.setattr(archive_mirror, "_LOCK_WAIT_S", 0.0)
    args = _parse(
        ["archive", "mirror", "confirm", "--root", str(tmp_path), "--remote-listing", str(tmp_path / "r.json")]
    )
    (tmp_path / "r.json").write_text("[]", encoding="utf-8")
    with store.archive_lock(tmp_path, wait_s=0.0, holder="test"), pytest.raises(SystemExit) as busy:
        args.handler(args)
    assert busy.value.code == 75
    assert "BUSY" in capsys.readouterr().out


def test_execute_builds_and_reports_stalled_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.market_data.binance.vision as vision

    fake_vision = FakeVision()
    monkeypatch.setattr(vision, "BinanceVisionDownloader", lambda: fake_vision)
    fake_vision.put(
        "data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-01-01.zip",
        zip_of(HEADER + b"2024-01-01 00:00:00,-5.00,1.0,2.0"),
    )
    args = _parse(
        [
            "archive",
            "mirror",
            "build",
            "--root",
            str(tmp_path),
            "--datasets",
            "bookDepth",
            "--execute",
            "--remote-bytes",
            "0",
            "--drive-free-bytes",
            str(10**15),
        ]
    )

    with pytest.raises(SystemExit) as result:
        args.handler(args)

    assert result.value.code == 5  # upstream_stalled: the fixture period is long past
    assert '"built_sources":1' in capsys.readouterr().out


def test_confirm_exit_3_when_unconfirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.market_data.binance.vision as vision
    from tests.unit.market_data.mirror.test_confirm import _outbox_pairs, _remote_json

    fake_vision = FakeVision()
    monkeypatch.setattr(vision, "BinanceVisionDownloader", lambda: fake_vision)
    _outbox_pairs(fake_vision, tmp_path)
    listing = tmp_path / "remote.json"
    listing.write_text("[]", encoding="utf-8")
    args = _parse(["archive", "mirror", "confirm", "--root", str(tmp_path), "--remote-listing", str(listing)])

    with pytest.raises(SystemExit) as result:
        args.handler(args)
    assert result.value.code == 3
    _ = _remote_json


def test_restore_cli_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.market_data.binance.vision as vision
    from src.market_data.mirror.index import load_index, sidecar_path
    from tests.unit.market_data.mirror.test_confirm import _outbox_pairs

    fake_vision = FakeVision()
    monkeypatch.setattr(vision, "BinanceVisionDownloader", lambda: fake_vision)
    config = _outbox_pairs(fake_vision, tmp_path)
    sc = next(s for s in load_index(config.outbox_index_dir, "bookDepth") if s.sources[0].storage == "columnar")
    out = tmp_path / "out.csv"
    args = _parse(
        [
            "archive",
            "mirror",
            "restore",
            "--root",
            str(tmp_path),
            "--sidecar",
            str(sidecar_path(config.outbox_index_dir, sc.object_path)),
            "--object",
            str(config.outbox_objects_dir / sc.object_path[len("objects/") :]),
            "--key",
            sc.sources[0].key,
            "--out",
            str(out),
        ]
    )
    args.handler(args)
    assert out.read_bytes().startswith(HEADER)
    assert '"sha256"' in capsys.readouterr().out


def test_status_without_age_limit_prints_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _parse(["archive", "mirror", "status", "--root", str(tmp_path)])
    args.handler(args)
    assert '"outbox_nonempty":false' in capsys.readouterr().out


def test_build_busy_lock_exits_75(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.market_data.public_archive import store

    monkeypatch.setattr(archive_mirror, "_LOCK_WAIT_S", 0.0)
    args = _parse(["archive", "mirror", "build", "--root", str(tmp_path), "--datasets", "bookDepth"])
    with store.archive_lock(tmp_path, wait_s=0.0, holder="test"), pytest.raises(SystemExit) as busy:
        args.handler(args)
    assert busy.value.code == 75
    assert "build status=BUSY" in capsys.readouterr().out


def test_format_doc_prints_format_description(capsys: pytest.CaptureFixture[str]) -> None:
    args = _parse(["archive", "mirror", "format-doc"])
    args.handler(args)
    assert "Mirror object format" in capsys.readouterr().out
