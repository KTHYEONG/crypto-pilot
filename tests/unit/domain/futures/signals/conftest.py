from __future__ import annotations

import pytest

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG


@pytest.fixture(autouse=True)
def disable_caches():
    """Disable LTF/L1 disk caches for all signal unit tests to avoid
    cross-test cache pollution from different test data producing
    the same fingerprint."""
    _old_ltf = OPT_FUTURES_CONFIG.get("LTF_PANEL_CACHE_ENABLED")
    _old_l1 = OPT_FUTURES_CONFIG.get("L1_RESULT_CACHE_ENABLED")
    OPT_FUTURES_CONFIG["LTF_PANEL_CACHE_ENABLED"] = False
    OPT_FUTURES_CONFIG["L1_RESULT_CACHE_ENABLED"] = False
    yield
    if _old_ltf is not None:
        OPT_FUTURES_CONFIG["LTF_PANEL_CACHE_ENABLED"] = _old_ltf
    else:
        OPT_FUTURES_CONFIG.pop("LTF_PANEL_CACHE_ENABLED", None)
    if _old_l1 is not None:
        OPT_FUTURES_CONFIG["L1_RESULT_CACHE_ENABLED"] = _old_l1
    else:
        OPT_FUTURES_CONFIG.pop("L1_RESULT_CACHE_ENABLED", None)
