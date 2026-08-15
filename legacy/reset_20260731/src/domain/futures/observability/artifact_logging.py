from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _path_or_dash(path: str | Path | None) -> str:
    return str(path) if path is not None else "-"


def _truncate_json_payload(payload: str, max_chars: int) -> str:
    if len(payload) <= max_chars:
        return payload
    preview = payload[: max_chars - 50]
    return json.dumps({"truncated": True, "preview": preview})


def emit_json_artifact_debug(
    *,
    logger: logging.Logger,
    artifact_name: str,
    run_id: str,
    payload: Mapping[str, Any],
    path: str | Path | None = None,
    max_chars: int = 50_000,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        payload_str = json.dumps(payload, default=str, ensure_ascii=False)
    except Exception:
        logger.warning("ARTIFACT_JSON_FAIL name=%s run_id=%s serialization failed", artifact_name, run_id)
        return
    payload_str = _truncate_json_payload(payload_str, max_chars)
    pdash = _path_or_dash(path)
    logger.debug("ARTIFACT_JSON_BEGIN name=%s run_id=%s path=%s", artifact_name, run_id, pdash)
    logger.debug(payload_str)
    logger.debug("ARTIFACT_JSON_END name=%s run_id=%s", artifact_name, run_id)


def _dicts_to_csv_text(rows: Sequence[Mapping[str, Any]], row_limit: int) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for i, row in enumerate(rows):
        if i >= row_limit:
            break
        writer.writerow({k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in row.items()})
    return output.getvalue().rstrip("\r\n")


def emit_csv_artifact_debug(
    *,
    logger: logging.Logger,
    artifact_name: str,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    path: str | Path | None = None,
    row_limit: int = 200,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    total = len(rows)
    emitted = min(total, row_limit)
    pdash = _path_or_dash(path)
    csv_text = _dicts_to_csv_text(rows, row_limit)
    logger.debug(
        "ARTIFACT_CSV_BEGIN name=%s run_id=%s rows=%s/%s path=%s",
        artifact_name,
        run_id,
        emitted,
        total,
        pdash,
    )
    if csv_text:
        logger.debug(csv_text)
    if total > row_limit:
        logger.debug("__truncated__,true")
    logger.debug("ARTIFACT_CSV_END name=%s run_id=%s", artifact_name, run_id)


def emit_dataframe_artifact_debug(
    *,
    logger: logging.Logger,
    artifact_name: str,
    run_id: str,
    frame: pd.DataFrame,
    path: str | Path | None = None,
    row_limit: int = 200,
) -> None:
    rows = list(frame.to_dict(orient="records"))
    emit_csv_artifact_debug(
        logger=logger,
        artifact_name=artifact_name,
        run_id=run_id,
        rows=rows,
        path=path,
        row_limit=row_limit,
    )
