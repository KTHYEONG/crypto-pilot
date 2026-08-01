"""Pre-register the sealed funding-gated directional sleeve candidate.

Append-only provenance record for ``funding_signed_directional_v1``: the fixed
5-symbol set, the mutually exclusive long/short funding rules, the frozen risk
budget, the data fingerprints, and the code fingerprint. Idempotent: an
identical record is a no-op and a conflicting one is an error.

Run with ``uv run python src/cli/register_directional_candidate.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.common.config import BASE_DIR, funding_path, ohlcv_path
from src.research.sleeve_blend.contracts import FIXED_DIRECTIONAL_SYMBOLS

_logger = logging.getLogger("RegisterDirectionalCandidate")

REGISTRY_PATH = BASE_DIR / "docs" / "results" / "candidate_registry.json"

CANDIDATE_ID = "funding_signed_directional_v1"
HYPOTHESIS_ID = "funding_signed_directional"
RETURN_SOURCE = "funding_gated_long_short_directional"
OBSERVATION_END = "2025-12-31 23:59:59"

CODE_UNITS: dict[str, Path] = {
    "application.sleeve_blend_evaluation": Path("src/application/sleeve_blend_evaluation.py"),
    "baseline.backtest": Path("src/research/baseline/backtest.py"),
    "baseline.signal": Path("src/research/baseline/signal.py"),
    "cli.run_sleeve_blend_backtest": Path("src/cli/run_sleeve_blend_backtest.py"),
    "research.contracts": Path("src/research/contracts.py"),
    "sleeve_blend.backtest": Path("src/research/sleeve_blend/backtest.py"),
    "sleeve_blend.contracts": Path("src/research/sleeve_blend/contracts.py"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for unit_id in sorted(CODE_UNITS):
        digest.update(unit_id.encode("utf-8"))
        digest.update(Path(CODE_UNITS[unit_id]).read_bytes())
    return digest.hexdigest()


def _data_hashes() -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    for symbol in FIXED_DIRECTIONAL_SYMBOLS:
        ohlcv = ohlcv_path(symbol, "1h")
        funding = funding_path(symbol)
        if not ohlcv.exists() or not funding.exists():
            raise FileNotFoundError(f"data missing for {symbol}: {ohlcv}, {funding}")
        hashes[symbol] = {
            "ohlcv_1h": _sha256(ohlcv),
            "funding": _sha256(funding),
        }
    return hashes


def _load(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"candidate registry is not a JSON list: {path}")
    return records


def _save(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temp.replace(path)


def main() -> None:
    record: dict[str, object] = {
        "domain": "sleeve_blend",
        "candidate_id": CANDIDATE_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "symbols": list(FIXED_DIRECTIONAL_SYMBOLS),
        "rules": {
            "long": "baseline long breakout AND last settled funding <= 0",
            "short": "mirror Donchian breakdown AND last settled funding >= 0",
            "execution": "signal at completed bar close, order at next bar open",
        },
        "parameters": {
            "history_days": 30,
            "max_symbol_weight": 0.25,
            "max_period_contribution": 0.40,
            "leverage": 1.0,
        },
        "data_hashes": _data_hashes(),
        "code_hash": _code_hash(),
        "observation_end": OBSERVATION_END,
        "return_source": RETURN_SOURCE,
        "registration_ts": datetime.now(UTC).isoformat(),
        "status": "REGISTERED",
    }
    records = _load(REGISTRY_PATH)
    for existing in records:
        if existing.get("candidate_id") == CANDIDATE_ID:
            comparable = {k: v for k, v in record.items() if k != "registration_ts"}
            existing_cmp = {
                k: v for k, v in existing.items() if k != "registration_ts"
            }
            if existing_cmp == comparable:
                _logger.info("already registered: %s", CANDIDATE_ID)
                return
            raise ValueError(
                f"duplicate candidate_id {CANDIDATE_ID} registered with a different payload"
            )
    records.append(record)
    _save(records, REGISTRY_PATH)
    _logger.info("registered %s -> %s", CANDIDATE_ID, REGISTRY_PATH)
    _logger.info("code_hash=%s", record["code_hash"])
    _logger.info("symbols=%s", list(FIXED_DIRECTIONAL_SYMBOLS))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
