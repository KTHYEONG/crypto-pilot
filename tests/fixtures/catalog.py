from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.expert_portfolio.catalog import ExpertLibraryBlueprint, ExpertLibraryCatalog
from src.research.expert_portfolio.contracts import ExpertDefinition


def write_blueprint_files(base: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Deterministic code and data files for a two-expert valid catalog fixture."""
    code_units = {
        "expert.engine": base / "engine.py",
        "expert.signal": base / "signal.py",
    }
    for unit_id, path in code_units.items():
        path.write_text(f"# {unit_id}\nVALUE = 1\n", encoding="utf-8")
    idx = pd.date_range("2024-01-01", "2026-01-01", freq="D", tz="UTC")
    data_files = {
        "ohlcv_AUSDT": base / "AUSDT.parquet",
        "ohlcv_BUSDT": base / "BUSDT.parquet",
    }
    for path in data_files.values():
        pd.DataFrame({"close": np.full(len(idx), 100.0)}, index=idx).to_parquet(path)
    return code_units, data_files


@pytest.fixture
def expert_library_blueprint(tmp_path: Path) -> ExpertLibraryBlueprint:
    code_units, data_files = write_blueprint_files(tmp_path)
    return ExpertLibraryBlueprint(
        library_id="valid_library",
        experts=(
            ExpertDefinition(
                "e1", "cointegration_residual", "pair_residual", ("AUSDT",), "run_backtest", "abc",
            ),
            ExpertDefinition(
                "e2", "cointegration_residual", "pair_residual", ("BUSDT",), "run_backtest", "def",
            ),
        ),
        supported_runners=frozenset({"run_backtest"}),
        code_units=code_units,
        data_files=data_files,
        observation_end="2025-12-31",
    )


@pytest.fixture
def expert_library_catalog(
    expert_library_blueprint: ExpertLibraryBlueprint,
) -> ExpertLibraryCatalog:
    return ExpertLibraryCatalog(blueprints={expert_library_blueprint.library_id: expert_library_blueprint})
