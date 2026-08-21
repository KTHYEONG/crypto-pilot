"""S5: Phase diagnostics + cross-sectional IC + regression + multi-feature.

Stage boundary: ``telemetry.record("multi_feature_diagnostic")`` (L4116).
Consumes: PanelState, BookState.
Produces: xs_ic, regression, horizon_diagnostics, trend/multi-feature diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class DiagnosticState:
    """Output of S5: statistical diagnostics."""

    xs_ic: float | None = None
    regression: Any = None
    horizon_diagnostics: dict[str, Any] | None = None
    trend_sleeve_diagnostic: Any = None
    multi_feature_diagnostic: Any = None
    signal_48h: pd.DataFrame | None = None
