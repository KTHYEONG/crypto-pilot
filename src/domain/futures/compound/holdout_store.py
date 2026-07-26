from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from src.domain.futures.compound.contracts import (
    DeploymentVerdict,
    L3ValidationResult,
    SealedHoldoutManifest,
)

_logger = logging.getLogger(__name__)


class HoldoutConflictError(RuntimeError):
    ...


class HoldoutReuseError(RuntimeError):
    ...


class HoldoutNotFoundError(RuntimeError):
    ...


class SealedHoldoutStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        _logger.debug("SealedHoldoutStore initialized: %s", self._db_path)

    def _ensure_db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=EXTRA")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sealed_holdouts (
                    holdout_id TEXT NOT NULL,
                    start_time_ns INTEGER NOT NULL,
                    end_time_ns INTEGER NOT NULL,
                    holdout_days INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    data_manifest_hash TEXT NOT NULL,
                    strategy_spec_hash TEXT NOT NULL DEFAULT '',
                    universe_state_hash TEXT NOT NULL DEFAULT '',
                    first_consumed_at_ns INTEGER,
                    result_json TEXT,
                    created_at_ns INTEGER NOT NULL,
                    consumed_spec_hash TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (holdout_id)
                )
            """)
            columns = {
                str(row[1]) for row in self._conn.execute("PRAGMA table_info(sealed_holdouts)")
            }
            if "universe_state_hash" not in columns:
                self._conn.execute(
                    "ALTER TABLE sealed_holdouts ADD COLUMN universe_state_hash TEXT NOT NULL DEFAULT ''"
                )
            if "consumed_spec_hash" not in columns:
                self._conn.execute(
                    "ALTER TABLE sealed_holdouts ADD COLUMN consumed_spec_hash TEXT NOT NULL DEFAULT ''"
                )
            self._conn.commit()
        return self._conn

    def create(self, manifest: SealedHoldoutManifest) -> SealedHoldoutManifest:
        conn = self._ensure_db()
        now_ns = int(time.time_ns())
        try:
            conn.execute(
                """
                INSERT INTO sealed_holdouts
                    (holdout_id, start_time_ns, end_time_ns, holdout_days,
                     model_version, data_manifest_hash, strategy_spec_hash, universe_state_hash,
                     first_consumed_at_ns, result_json, created_at_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (
                    manifest.holdout_id,
                    manifest.start_time_ns,
                    manifest.end_time_ns,
                    manifest.holdout_days,
                    manifest.model_version,
                    manifest.data_manifest_hash,
                    manifest.strategy_spec_hash,
                    manifest.universe_state_hash,
                    now_ns,
                ),
            )
            conn.commit()
            _logger.info("holdout %s created", manifest.holdout_id)
        except sqlite3.IntegrityError:
            msg = f"holdout {manifest.holdout_id} already exists"
            raise HoldoutConflictError(msg) from None
        return manifest

    def get_manifest(self, holdout_id: str) -> SealedHoldoutManifest:
        conn = self._ensure_db()
        row = conn.execute(
            """SELECT holdout_id, start_time_ns, end_time_ns, holdout_days,
                      model_version, data_manifest_hash, strategy_spec_hash, universe_state_hash,
                      first_consumed_at_ns, consumed_spec_hash
               FROM sealed_holdouts WHERE holdout_id = ?""",
            (holdout_id,),
        ).fetchone()
        if row is None:
            raise HoldoutNotFoundError(f"holdout {holdout_id} not found")
        return SealedHoldoutManifest(
            holdout_id=row[0],
            start_time_ns=row[1],
            end_time_ns=row[2],
            holdout_days=row[3],
            model_version=row[4],
            data_manifest_hash=row[5],
            strategy_spec_hash=row[6] or "",
            universe_state_hash=row[7] or "",
            first_consumed_at_ns=row[8],
            consumed_spec_hash=row[9] or "",
        )

    def ensure_sealed(self, manifest: SealedHoldoutManifest) -> SealedHoldoutManifest:
        conn = self._ensure_db()
        row = conn.execute(
            """SELECT holdout_id, data_manifest_hash, model_version,
                      strategy_spec_hash, first_consumed_at_ns, consumed_spec_hash
               FROM sealed_holdouts WHERE holdout_id = ?""",
            (manifest.holdout_id,),
        ).fetchone()
        if row is None:
            return self.create(manifest)
        (_hid, stored_data_hash, _stored_model,
         stored_spec_hash, consumed_at_ns, _consumed_spec) = row

        if stored_data_hash != manifest.data_manifest_hash:
            raise HoldoutReuseError(
                f"holdout {manifest.holdout_id} data_manifest_hash mismatch: "
                f"{stored_data_hash} != {manifest.data_manifest_hash}"
            )
        if stored_spec_hash == "" and consumed_at_ns is None:
            _logger.warning("holdout %s: backfilling empty spec_hash", manifest.holdout_id)
            conn.execute(
                "UPDATE sealed_holdouts SET strategy_spec_hash = ? WHERE holdout_id = ?",
                (manifest.strategy_spec_hash, manifest.holdout_id),
            )
            conn.commit()
            return replace(manifest, strategy_spec_hash=manifest.strategy_spec_hash)
        if stored_spec_hash == "" and consumed_at_ns is not None:
            raise HoldoutReuseError(
                f"holdout {manifest.holdout_id} empty spec_hash already consumed, permanently invalid"
            )
        return self.get_manifest(manifest.holdout_id)

    def consume(
        self,
        *,
        holdout_id: str,
        model_version: str,
        data_manifest_hash: str,
        strategy_spec_hash: str,
        evaluate: Callable[[SealedHoldoutManifest], L3ValidationResult],
        universe_state_hash: str = "",
    ) -> L3ValidationResult:
        conn = self._ensure_db()
        cursor = conn.execute(
            """
            SELECT start_time_ns, end_time_ns, holdout_days, model_version,
                   data_manifest_hash, strategy_spec_hash, universe_state_hash, first_consumed_at_ns,
                   result_json, consumed_spec_hash
            FROM sealed_holdouts WHERE holdout_id = ?
            """,
            (holdout_id,),
        )
        row = cursor.fetchone()
        if row is None:
            msg = f"holdout {holdout_id} not found"
            raise HoldoutNotFoundError(msg)

        (stored_start, stored_end, stored_days, stored_model,
         stored_data_hash, _stored_spec_hash, stored_universe_hash, consumed_at_ns,
         result_json, stored_consumed_spec) = row

        if (
            stored_data_hash != data_manifest_hash
            or stored_model != model_version
            or (stored_universe_hash != "" and stored_universe_hash != universe_state_hash)
        ):
            msg = (
                f"holdout {holdout_id} hash mismatch: "
                f"model={stored_model}!={model_version} "
                f"data_hash={stored_data_hash}!={data_manifest_hash} "
                f"universe_hash={stored_universe_hash}!={universe_state_hash}"
            )
            raise HoldoutReuseError(msg)

        if consumed_at_ns is not None and result_json is not None:
            if stored_consumed_spec == strategy_spec_hash:
                _logger.info("holdout %s already consumed with same spec, returning cached result", holdout_id)
                return _deserialize_result(result_json)
            msg = (
                f"holdout {holdout_id} already consumed with spec_hash={stored_consumed_spec}, "
                f"cannot reuse with {strategy_spec_hash}"
            )
            raise HoldoutReuseError(msg)

        now_ns = int(time.time_ns())
        manifest = SealedHoldoutManifest(
            holdout_id=holdout_id,
            start_time_ns=stored_start,
            end_time_ns=stored_end,
            holdout_days=stored_days,
            model_version=stored_model,
            data_manifest_hash=stored_data_hash,
            strategy_spec_hash=strategy_spec_hash,
            universe_state_hash=stored_universe_hash,
            first_consumed_at_ns=now_ns,
            consumed_spec_hash=strategy_spec_hash,
        )

        result = evaluate(manifest)

        conn.execute(
            """
            UPDATE sealed_holdouts
            SET first_consumed_at_ns = ?, result_json = ?, strategy_spec_hash = ?,
                consumed_spec_hash = ?
            WHERE holdout_id = ? AND first_consumed_at_ns IS NULL
            """,
            (now_ns, _serialize_result(result), strategy_spec_hash, strategy_spec_hash, holdout_id),
        )
        conn.commit()

        _update_first_consumed(manifest, now_ns)

        if consumed_at_ns is None and result_json is None:
            _logger.info("holdout %s consumed and persisted", holdout_id)

        return result


def _serialize_result(result: L3ValidationResult) -> str:
    return json.dumps({
        "verdict": result.verdict.value,
        "posterior_growth_probability": result.posterior_growth_probability,
        "holdout_days": result.holdout_days,
        "max_drawdown": result.max_drawdown,
        "daily_cvar95": result.daily_cvar95,
        "reasons": list(result.reasons),
    })


def _deserialize_result(raw: str) -> L3ValidationResult:
    data = json.loads(raw)
    return L3ValidationResult(
        verdict=DeploymentVerdict(data["verdict"]),
        posterior_growth_probability=data["posterior_growth_probability"],
        holdout_days=data["holdout_days"],
        max_drawdown=data["max_drawdown"],
        daily_cvar95=data["daily_cvar95"],
        reasons=tuple(data["reasons"]),
    )


def _update_first_consumed(
    manifest: SealedHoldoutManifest, now_ns: int,
) -> SealedHoldoutManifest:
    object.__setattr__(manifest, "first_consumed_at_ns", now_ns)
    return manifest
