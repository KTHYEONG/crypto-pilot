"""Pipeline stage order contract."""

from __future__ import annotations

from types import SimpleNamespace

import src.lab.mhs.pipeline.runner as runner
import src.lab.mhs.pipeline.stages.assemble as assemble_mod
import src.lab.mhs.pipeline.stages.book as book_mod
import src.lab.mhs.pipeline.stages.committee as committee_mod
import src.lab.mhs.pipeline.stages.fold as fold_mod
import src.lab.mhs.pipeline.stages.panel as panel_mod
import src.lab.mhs.pipeline.stages.replay as replay_mod
import src.lab.mhs.pipeline.stages.selection as selection_mod


def _patch_stages(monkeypatch, behaviours: dict[str, object]):
    modules = {
        "load_panel": (panel_mod, "load_panel"),
        "select_horizons": (selection_mod, "select_horizons"),
        "build_books": (book_mod, "build_books"),
        "build_committee": (committee_mod, "build_committee"),
        "run_replays": (replay_mod, "run_replays"),
        "run_folds": (fold_mod, "run_folds"),
        "assemble_report": (assemble_mod, "assemble_report"),
    }
    for name, (module, attr) in modules.items():
        monkeypatch.setattr(module, attr, behaviours[name])


def test_run_stages_call_order(monkeypatch) -> None:
    """Run stages call order."""
    order: list[str] = []
    sentinel = object()

    def _recorder(name):
        def _call(ctx, telemetry):
            order.append(name)
            if name == "assemble_report":
                return sentinel
            return None

        return _call

    _patch_stages(
        monkeypatch,
        {
            "load_panel": _recorder("load_panel"),
            "select_horizons": _recorder("select_horizons"),
            "build_books": _recorder("build_books"),
            "build_committee": _recorder("build_committee"),
            "run_replays": _recorder("run_replays"),
            "run_folds": _recorder("run_folds"),
            "assemble_report": _recorder("assemble_report"),
        },
    )
    ctx = SimpleNamespace(_terminal_report=None)
    result = runner.run_stages(ctx, SimpleNamespace())
    assert order == [
        "load_panel",
        "select_horizons",
        "build_books",
        "build_committee",
        "run_replays",
        "run_folds",
        "assemble_report",
    ]
    assert result is sentinel


def test_terminal_report_short_circuits_replays(monkeypatch) -> None:
    """Terminal report short-circuits replays."""
    sentinel = object()
    calls: list[str] = []

    def _noop(name):
        def _call(ctx, telemetry):
            calls.append(name)
            return None

        return _call

    def _terminal(ctx, telemetry):
        calls.append("build_committee")
        ctx._terminal_report = sentinel

    def _boom(ctx, telemetry):
        raise AssertionError("must not run after a terminal report")

    _patch_stages(
        monkeypatch,
        {
            "load_panel": _noop("load_panel"),
            "select_horizons": _noop("select_horizons"),
            "build_books": _noop("build_books"),
            "build_committee": _terminal,
            "run_replays": _boom,
            "run_folds": _boom,
            "assemble_report": _boom,
        },
    )
    ctx = SimpleNamespace(_terminal_report=None)
    assert runner.run_stages(ctx, SimpleNamespace()) is sentinel
    assert calls == ["load_panel", "select_horizons", "build_books", "build_committee"]
