"""Live runtime: cloud-owned rolling state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import SecretStr

from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR
from src.live.errors import ArtifactSealError

SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class LiveRuntime:
    schema_version: int
    params_digest: str
    last_decision_date: pd.Timestamp
    held_target_row: dict[str, float]
    reference_daily_returns: pd.Series


def default_runtime_path() -> Path:
    return DATA_DIR / "state" / "live_runtime.json"


def bootstrap_runtime(params: Any, bootstrap_reference: pd.Series) -> LiveRuntime:
    last = pd.Timestamp(params.backtest_window[1]).tz_convert("UTC").normalize()
    ref = bootstrap_reference.copy() if not bootstrap_reference.empty else pd.Series(dtype="float64")
    if not ref.empty:
        ref = ref.sort_index()
        if ref.index.tz is None:
            ref.index = ref.index.tz_localize("UTC")
        else:
            ref.index = ref.index.tz_convert("UTC")
    return LiveRuntime(
        schema_version=SCHEMA_VERSION,
        params_digest=str(params.strategy_digest),
        last_decision_date=last,
        held_target_row=dict(params.bootstrap_held_row),
        reference_daily_returns=ref,
    )


def _serialize_runtime(rt: LiveRuntime) -> dict[str, Any]:
    ref = rt.reference_daily_returns
    if not ref.empty:
        ref = ref.sort_index()
        ref_dict = {pd.Timestamp(k).isoformat(): float(v) for k, v in ref.items()}
    else:
        ref_dict = {}
    return {
        "schema_version": int(rt.schema_version),
        "params_digest": str(rt.params_digest),
        "last_decision_date": pd.Timestamp(rt.last_decision_date).isoformat(),
        "held_target_row": {str(k): float(v) for k, v in rt.held_target_row.items()},
        "reference_daily_returns": ref_dict,
    }


def _deserialize_runtime(raw: dict[str, Any]) -> LiveRuntime:
    if not isinstance(raw, dict):
        raise DataIntegrityError("live runtime must be a JSON object")
    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise DataIntegrityError(f"unknown schema_version {schema!r}")
    params_digest = raw.get("params_digest")
    if not isinstance(params_digest, str) or not params_digest:
        raise DataIntegrityError("live runtime missing params_digest")
    last_raw = raw.get("last_decision_date")
    if not isinstance(last_raw, str):
        raise DataIntegrityError("live runtime missing last_decision_date")
    try:
        last_ts = pd.Timestamp(last_raw)
    except Exception as exc:
        raise DataIntegrityError(f"last_decision_date invalid: {exc}") from exc
    if last_ts.tzinfo is None:
        raise DataIntegrityError("last_decision_date must be tz-aware")
    last_ts = last_ts.tz_convert("UTC")
    held_raw = raw.get("held_target_row")
    if not isinstance(held_raw, dict):
        raise DataIntegrityError("held_target_row must be object")
    held: dict[str, float] = {}
    for k, v in held_raw.items():
        if isinstance(v, bool):
            raise DataIntegrityError(f"held_target_row non-numeric {k!r}")
        try:
            held[str(k)] = float(v)
        except Exception as exc:
            raise DataIntegrityError(f"held_target_row non-numeric {k!r}") from exc
    ref_raw = raw.get("reference_daily_returns")
    if not isinstance(ref_raw, dict):
        raise DataIntegrityError("reference_daily_returns must be object")
    if ref_raw:
        idx: list[pd.Timestamp] = []
        vals: list[float] = []
        for iso, val in ref_raw.items():
            try:
                ts = pd.Timestamp(iso)
            except Exception as exc:
                raise DataIntegrityError(f"return timestamp invalid {iso!r}") from exc
            if ts.tzinfo is None:
                raise DataIntegrityError("reference index must be tz-aware")
            ts = ts.tz_convert("UTC")
            if isinstance(val, bool):
                raise DataIntegrityError(f"return entry non-numeric {iso!r}")
            try:
                fval = float(val)
            except Exception as exc:
                raise DataIntegrityError(f"return entry non-numeric {iso!r}") from exc
            idx.append(ts)
            vals.append(fval)
        series = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype="float64").sort_index()
    else:
        series = pd.Series(dtype="float64")
    return LiveRuntime(
        schema_version=int(schema),
        params_digest=str(params_digest),
        last_decision_date=last_ts,
        held_target_row=held,
        reference_daily_returns=series,
    )


def save_runtime(path: Path, runtime: LiveRuntime, *, artifact_key: SecretStr | None = None) -> Path:
    path = Path(path)
    payload = _serialize_runtime(runtime)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if artifact_key is not None:
        from src.live.crypto import derive_key, seal_bytes

        dest = path if str(path).endswith(".enc") else Path(str(path) + ".enc")
        dest.parent.mkdir(parents=True, exist_ok=True)
        sealed = seal_bytes(data, derive_key(artifact_key))
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(sealed)
        os.replace(tmp, dest)
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def load_or_bootstrap_runtime(path: Path, params: Any, bootstrap_reference: pd.Series, *, artifact_key: SecretStr | None = None) -> LiveRuntime:
    path = Path(path)
    is_enc = str(path).endswith(".enc")
    candidate: Path | None = None
    if path.exists():
        candidate = path
    elif artifact_key is not None:
        enc = path if is_enc else Path(str(path) + ".enc")
        if enc.exists():
            candidate = enc
        enc2 = Path(str(path) + ".enc")
        if enc2.exists() and candidate is None:
            candidate = enc2
    if candidate is None:
        rt = bootstrap_runtime(params, bootstrap_reference)
        save_runtime(path, rt, artifact_key=artifact_key)
        return rt
    if str(candidate).endswith(".enc"):
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {candidate}")
        try:
            from src.live.crypto import derive_key, open_bytes

            blob = candidate.read_bytes()
            plain = open_bytes(blob, derive_key(artifact_key))
            raw = json.loads(plain.decode("utf-8"))
        except ArtifactSealError:
            raise
        except Exception as exc:
            raise DataIntegrityError(f"live runtime file corrupt: {candidate}: {exc}") from exc
    else:
        try:
            raw_text = candidate.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"live runtime file corrupt: {candidate}") from exc
        except Exception as exc:
            raise DataIntegrityError(f"live runtime file corrupt: {candidate}: {exc}") from exc
    if isinstance(raw, bytes):
        raise DataIntegrityError(f"live runtime file corrupt: {candidate}")
    return _deserialize_runtime(raw)


def adopt_params(runtime: LiveRuntime, params: Any, bootstrap_reference: pd.Series) -> tuple[LiveRuntime, str]:
    ref = bootstrap_reference.copy() if not bootstrap_reference.empty else pd.Series(dtype="float64")
    if not ref.empty:
        ref = ref.sort_index()
        if ref.index.tz is None:
            ref.index = ref.index.tz_localize("UTC")
        else:
            ref.index = ref.index.tz_convert("UTC")
    reason = "bootstrap" if not runtime.held_target_row else "soft_swap"
    new_rt = LiveRuntime(
        schema_version=runtime.schema_version,
        params_digest=str(params.strategy_digest),
        last_decision_date=runtime.last_decision_date,
        held_target_row=dict(runtime.held_target_row),
        reference_daily_returns=ref,
    )
    return new_rt, reason


def reconcile_runtime_params(
    runtime: LiveRuntime, params: Any, bootstrap_reference: pd.Series
) -> tuple[LiveRuntime, str | None]:
    if runtime.params_digest == str(params.strategy_digest):
        return runtime, None
    return adopt_params(runtime, params, bootstrap_reference)
