# ruff: noqa
def test_v1_forward_scorer_stack_deleted() -> None:
    import importlib
    import pathlib

    import pytest

    for mod in ("src.mhs.signal_refresh", "src.mhs.signal_runtime", "src.mhs.signal_state", "src.mhs.deployment_bundle", "src.mhs.deployed_weights_ledger"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)

    src_root = pathlib.Path(__file__).resolve().parents[3] / "src"
    banned = ("signal_refresh", "signal_runtime", "signal_state", "deployment_bundle", "deployed_weights_ledger")
    hits = [str(p) for p in src_root.rglob("*.py") for tok in banned if tok in p.read_text(encoding="utf-8")]
    assert hits == []

    step_src = (src_root / "mhs" / "live_signal_step.py").read_text(encoding="utf-8")
    for tok in ("replay_execution_windows", "_iter_mhs_execution_windows", "execution_timeframe"):
        assert tok not in step_src
