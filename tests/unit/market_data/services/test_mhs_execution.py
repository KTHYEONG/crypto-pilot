"""Unit isolation for src.market_data.services.mhs_execution manifest sealing."""

from __future__ import annotations

import json


def test_refresh_mhs_execution_manifest_seals_required_inputs(tmp_path) -> None:
    import src.market_data.services.mhs_execution as module

    manifest = tmp_path / "plan.json"
    manifest.write_text(
        json.dumps({"timeframe": "3m", "start": "2025-01-01", "end": "2025-01-02", "symbols": []}),
        encoding="utf-8",
    )
    result = module.refresh_mhs_execution_manifest(manifest)
    attestation = tmp_path / "plan.inputs.json"
    assert attestation.exists()
    assert result["input_manifest_path"] == str(attestation)
    assert isinstance(result["input_manifest_digest"], str)
    assert len(result["input_manifest_digest"]) == 64
    sealed = json.loads(attestation.read_text(encoding="utf-8"))
    assert sealed["digest"] == result["input_manifest_digest"]
    assert sealed["files"] == []
