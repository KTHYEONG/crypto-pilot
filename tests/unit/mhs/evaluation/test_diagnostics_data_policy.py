"""data_policy threading for the feature-panel loader (zombie_mask rerun)."""


def test_load_feature_panels_threads_data_policy_to_loader(monkeypatch) -> None:
    import pandas as pd

    import src.mhs.evaluation.diagnostics as diagnostics

    grid = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    policies: list[object] = []

    def _fake_loader(*_a, **kwargs):
        policies.append(kwargs["data_policy"])
        return {"close": pd.DataFrame(1.0, index=grid, columns=["AAAUSDT"])}

    monkeypatch.setattr(diagnostics, "_available_panel_columns", lambda root, columns: ("close",))
    monkeypatch.setattr(diagnostics, "load_base_panel", _fake_loader)

    panels = diagnostics._load_feature_panels(
        "root", grid[0], grid[-1], grid, ["AAAUSDT"], columns=("close",), data_policy="zombie_mask_v1",
    )
    default_panels = diagnostics._load_feature_panels("root", grid[0], grid[-1], grid, ["AAAUSDT"], columns=("close",))

    assert policies == ["zombie_mask_v1", "legacy"]
    assert list(panels["close"].columns) == ["AAAUSDT"]
    assert list(default_panels["close"].columns) == ["AAAUSDT"]
