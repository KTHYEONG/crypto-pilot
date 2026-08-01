from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

# Canonical cash-and-carry semantic units hashed for candidate provenance.
# Façade modules are deliberately excluded: only the real implementation files
# participate in the code fingerprint.
CANONICAL_CARRY_CODE_UNITS: Mapping[str, Path] = {
    "application.cash_carry_evaluation": Path("src/application/cash_carry_evaluation.py"),
    "cash_carry.backtest": Path("src/research/cash_carry/backtest.py"),
    "cash_carry.market_data": Path("src/research/cash_carry/market_data.py"),
    "cash_carry.signal": Path("src/research/cash_carry/signal.py"),
    "market_data.storage.loaders": Path("src/market_data/storage/loaders.py"),
}

# Canonical technical-expert semantic units hashed for candidate provenance.
# Only the real implementation files participate in the code fingerprint; the
# application dispatch layer never does.
TECHNICAL_CODE_UNITS: Mapping[str, Path] = {
    "application.expert_evaluation": Path("src/application/expert_evaluation.py"),
    "technical_experts.backtest": Path("src/research/technical_experts/backtest.py"),
    "technical_experts.catalog": Path("src/research/technical_experts/catalog.py"),
    "technical_experts.contracts": Path("src/research/technical_experts/contracts.py"),
    "technical_experts.indicators": Path("src/research/technical_experts/indicators.py"),
    "technical_experts.signals": Path("src/research/technical_experts/signals.py"),
    "market_data.storage.loaders": Path("src/market_data/storage/loaders.py"),
}


def compute_code_hash(
    logical_units: Mapping[str, Path] = CANONICAL_CARRY_CODE_UNITS,  # noqa: B008
) -> str:
    """Hash the ordered logical-unit IDs and their canonical source bytes.

    Sorting by logical-unit ID makes the digest independent of input-mapping
    order; any canonical semantic source change produces a different digest.
    A declared canonical source that does not exist raises ``FileNotFoundError``
    rather than silently hashing an empty file.
    """
    digest = hashlib.sha256()
    for unit_id in sorted(logical_units):
        digest.update(unit_id.encode("utf-8"))
        digest.update(Path(logical_units[unit_id]).read_bytes())
    return digest.hexdigest()
