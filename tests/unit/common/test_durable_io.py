"""Invariant guard tests for crash-durable whole-file replacement."""

from __future__ import annotations

import errno
import os
import stat
import threading
from pathlib import Path

import pytest

from src.common.durable_io import durable_replace, durable_write_bytes, durable_write_text


def _tmp_siblings(path: Path) -> list[Path]:
    return [p for p in path.parent.iterdir() if p.suffix == ".tmp"]


def test_durable_replace_writes_complete_content_atomically(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    durable_write_bytes(path, b"new")
    assert path.read_bytes() == b"new"
    assert _tmp_siblings(path) == []


def test_durable_replace_fsyncs_file_before_replace_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    real_fsync = os.fsync
    real_replace = os.replace
    events: list[tuple[str, str]] = []

    def _fsync(fd: int) -> None:
        try:
            events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        except OSError:
            events.append(("fsync", str(fd)))
        real_fsync(fd)

    def _replace(src: object, dst: object) -> None:
        assert isinstance(src, (str, Path))
        assert isinstance(dst, (str, Path))
        events.append(("replace", f"{src}->{dst}"))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", _fsync)
    monkeypatch.setattr(os, "replace", _replace)
    durable_write_text(path, "x")

    assert path.read_text(encoding="utf-8") == "x"
    assert _tmp_siblings(path) == []
    kinds = [kind for kind, _ in events]
    assert kinds.count("replace") == 1
    replace_at = kinds.index("replace")
    _, replace_detail = events[replace_at]
    tmp_name = replace_detail.split("->")[0]
    tmp = Path(tmp_name)
    assert tmp.parent == path.parent
    assert tmp.name == f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    file_fsyncs = [target for kind, target in events[:replace_at] if kind == "fsync"]
    assert tmp_name in file_fsyncs
    dir_fsyncs = [target for kind, target in events[replace_at + 1 :] if kind == "fsync"]
    assert str(path.parent) in dir_fsyncs


def test_durable_replace_failure_keeps_old_content_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")

    def _boom(src: Path, dst: Path) -> None:
        raise RuntimeError("power loss during replace")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(RuntimeError, match="power loss"):
        durable_write_text(path, "new")
    assert path.read_bytes() == b"old"
    assert _tmp_siblings(path) == []


def test_writer_failure_keeps_old_content_and_removes_temp(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")

    def _half_write(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise ValueError("writer exploded")

    with pytest.raises(ValueError, match="writer exploded"):
        durable_replace(path, _half_write)
    assert path.read_bytes() == b"old"
    assert _tmp_siblings(path) == []


def test_directory_open_failure_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    durable_write_text(path, "new")
    assert path.read_bytes() == b"new"
    assert _tmp_siblings(path) == []


@pytest.mark.parametrize("error", sorted({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}))
def test_directory_fsync_unsupported_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: int
) -> None:
    path = tmp_path / "state.json"
    real_open = os.open
    real_fsync = os.fsync
    dir_fds: list[int] = []

    def _open(p: object, flags: int, *args: object, **kwargs: object) -> int:
        fd = real_open(p, flags, *args, **kwargs)  # type: ignore[arg-type]
        if str(p) == str(path.parent):
            dir_fds.append(fd)
        return fd

    def _fsync(fd: int) -> None:
        if fd in dir_fds:
            raise OSError(error, "fsync not supported")
        real_fsync(fd)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "fsync", _fsync)
    durable_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "new"
    assert _tmp_siblings(path) == []
    assert len(dir_fds) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(dir_fds[0])


def test_directory_fsync_io_error_propagates_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    real_open = os.open
    real_fsync = os.fsync
    dir_fds: list[int] = []

    def _open(p: object, flags: int, *args: object, **kwargs: object) -> int:
        fd = real_open(p, flags, *args, **kwargs)  # type: ignore[arg-type]
        if str(p) == str(path.parent):
            dir_fds.append(fd)
        return fd

    def _fsync(fd: int) -> None:
        if fd in dir_fds:
            raise OSError(errno.EIO, "disk error")
        real_fsync(fd)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "fsync", _fsync)
    with pytest.raises(OSError, match="disk error") as excinfo:
        durable_write_text(path, "new")
    assert excinfo.value.errno == errno.EIO
    assert path.read_bytes() == b"new"
    assert _tmp_siblings(path) == []
    assert len(dir_fds) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(dir_fds[0])


@pytest.mark.parametrize("failure_stage", ["writer", "file_fsync"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_pre_replace_failure_preserves_old_content_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str, error_type: type[BaseException]
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    failure = error_type("interrupted write")
    real_fsync = os.fsync

    def _write(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        if failure_stage == "writer":
            raise failure

    def _fsync(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise failure
        real_fsync(fd)

    if failure_stage == "file_fsync":
        monkeypatch.setattr(os, "fsync", _fsync)
    with pytest.raises(error_type, match="interrupted write") as excinfo:
        durable_replace(path, _write)
    assert excinfo.value is failure
    assert path.read_bytes() == b"old"
    assert _tmp_siblings(path) == []


def test_missing_parent_directory_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "no_such_dir" / "state.json"
    with pytest.raises(FileNotFoundError):
        durable_write_text(path, "new")
    assert not path.parent.exists()


def test_concurrent_writers_do_not_share_a_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "shared.json"
    path.write_text("seed", encoding="utf-8")
    errors: list[BaseException] = []
    barrier = threading.Barrier(2, timeout=5)

    def _worker(payload: str) -> None:
        def _write(tmp: Path) -> None:
            tmp.write_text(payload, encoding="utf-8")
            barrier.wait()

        try:
            for _ in range(50):
                durable_replace(path, _write)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(c * 100,)) for c in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert path.read_text(encoding="utf-8") in {"A" * 100, "B" * 100}
    assert _tmp_siblings(path) == []
