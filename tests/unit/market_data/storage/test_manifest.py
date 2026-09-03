"""P4 path-presence pin for the spot manifest store.

Behavioral coverage lives in
``tests/unit/market_data/services/test_spot_collection.py``.
"""

from __future__ import annotations

import src.market_data.storage.manifest as manifest


def test_manifest_module_present() -> None:
    assert manifest.__name__ == "src.market_data.storage.manifest"
    assert callable(manifest.load_spot_manifest)
