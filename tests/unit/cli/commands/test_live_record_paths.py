"""Run-namespaced weights resolution: explicit --artifact wins, else settings, else default."""

from __future__ import annotations

from types import SimpleNamespace

import src.cli.commands.live as live_mod


def test_resolve_weights_path_prefers_explicit_over_settings(tmp_path) -> None:
    explicit = tmp_path / "custom.parquet"
    settings = SimpleNamespace(weights_path=str(tmp_path / "run" / "target_weights.parquet"))
    assert live_mod._resolve_weights_path(str(explicit), settings) == explicit
    assert live_mod._resolve_weights_path(None, settings) == tmp_path / "run" / "target_weights.parquet"
    assert live_mod._resolve_weights_path(None, SimpleNamespace(weights_path=None)) == live_mod.default_weights_path()
