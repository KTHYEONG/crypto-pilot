"""MHS input data policy: single source of truth (INV-POLICY-SINGLE-SOURCE).

Every MHS entrypoint (``MhsDiagnosticRequest``, ``MhsRunConfig``,
``LiveStrategyParams``, the live runtime) shares ``MHS_DATA_POLICY_DEFAULT``.
The new default is ``zombie_mask_v1``; artifacts that predate the policy field
keep their legacy interpretation and are never auto-upgraded.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal


class MhsDataPolicy(StrEnum):
    """Registered MHS input-data contracts."""

    LEGACY = "legacy"
    ZOMBIE_MASK_V1 = "zombie_mask_v1"


MHS_DATA_POLICY_DEFAULT: Final[Literal["zombie_mask_v1"]] = "zombie_mask_v1"
