"""Chronological nested policy selection over registered estimator windows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.process import RefitPoint
from src.quant.evaluation.reliability import derive_block_size

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("selection is a library module")

INNER_EVIDENCE_INSUFFICIENT = "INNER_EVIDENCE_INSUFFICIENT"
CONTROL_PREREQUISITES_INVALID = "CONTROL_PREREQUISITES_INVALID"
NO_ESTIMATED_IMPROVEMENT = "NO_ESTIMATED_IMPROVEMENT"
PAIRED_LCB_IMPROVEMENT = "PAIRED_LCB_IMPROVEMENT"


@dataclass(frozen=True, slots=True)
class TrainingWindowSpec:
    """Identify a registered estimator window independently of observed performance.

    Equal-member control does not estimate profitable member weights; expanding
    and rolling policies use the same registered member estimator and execution rules.
    """

    policy_id: str
    kind: Literal["expanding", "rolling", "equal_member"]
    months: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise DataIntegrityError("policy_id must be a nonempty identity")
        if self.kind == "rolling":
            if isinstance(self.months, bool) or not isinstance(self.months, int) or self.months <= 0:
                raise DataIntegrityError("rolling policies require positive integral months")
        elif self.months is not None:
            raise DataIntegrityError("expanding and equal-member policies must not declare months")


@dataclass(frozen=True, slots=True)
class NestedSelectionSpec:
    """Freeze chronological policy comparison, common evidence and uncertainty rules.

    No outer evaluation statistic may rewrite the candidate set, control or scoring
    rule of the procedure that generated it.
    """

    policies: tuple[TrainingWindowSpec, ...]
    control_policy_id: str
    minimum_inner_labels: int
    alpha: float
    bootstrap_paths: int
    seed: int
    fit_latency: pd.Timedelta

    def __post_init__(self) -> None:
        ids = [p.policy_id for p in self.policies]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or ids.count(self.control_policy_id) != 1
            or isinstance(self.minimum_inner_labels, bool)
            or not isinstance(self.minimum_inner_labels, int)
            or self.minimum_inner_labels < 1
            or isinstance(self.bootstrap_paths, bool)
            or not isinstance(self.bootstrap_paths, int)
            or self.bootstrap_paths < 1
            or not isinstance(self.alpha, float | int)
            or not np.isfinite(float(self.alpha))
            or not 0.0 < float(self.alpha) < 1.0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not isinstance(self.fit_latency, pd.Timedelta)
            or self.fit_latency < pd.Timedelta(0)
        ):
            raise DataIntegrityError("nested selection controls are invalid")


@dataclass(frozen=True, slots=True)
class InnerFitAudit:
    """Record the actual matured fit dependencies preceding each inner application."""

    point: RefitPoint
    train_start: pd.Timestamp | None
    n_train_labels: int
    latest_label_end: pd.Timestamp | None
    latest_input_available_at: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class InnerPolicyEvidence:
    """Carry a policy's causal fit-then-apply trajectory for earlier inner evaluation.

    Combined policy returns are evaluated directly; nonlinear costs and netted
    holdings cannot be inferred by averaging stand-alone member return streams.
    """

    policy_id: str
    daily_returns: pd.Series
    available_at: pd.DatetimeIndex
    turnover: pd.Series
    valid: pd.Series
    procedure_digest: str
    input_manifest_digest: str | None
    economic_source: Literal["hourly_proxy", "inventory_3m"]
    prerequisites_ready_at: pd.Timestamp
    fit_audits: tuple[InnerFitAudit, ...]

    def __post_init__(self) -> None:
        """Validate one policy trajectory before chronological comparison so positional
        alignment, maturity and provenance cannot be substituted by matching lengths.
        """
        if (
            not isinstance(self.policy_id, str)
            or not self.policy_id
            or not isinstance(self.procedure_digest, str)
            or not self.procedure_digest
            or self.economic_source not in ("hourly_proxy", "inventory_3m")
            or not isinstance(self.daily_returns, pd.Series)
            or not isinstance(self.turnover, pd.Series)
            or not isinstance(self.valid, pd.Series)
            or not isinstance(self.available_at, pd.DatetimeIndex)
            or len(self.daily_returns) != len(self.available_at)
            or len(self.daily_returns) != len(self.turnover)
            or len(self.daily_returns) != len(self.valid)
        ):
            raise DataIntegrityError("inner policy evidence is invalid")
        for series in (self.daily_returns, self.turnover, self.valid):
            index = series.index
            if (
                not isinstance(index, pd.DatetimeIndex)
                or index.tz is None
                or str(index.tz) != "UTC"
                or len(set(index)) != len(index)
                or not bool(index.is_monotonic_increasing)
                or not index.equals(self.daily_returns.index)
            ):
                raise DataIntegrityError("inner evidence series require unique monotonic UTC labels")
        if (
            self.available_at.tz is None
            or str(self.available_at.tz) != "UTC"
            or not bool(self.available_at.is_monotonic_increasing)
        ):
            raise DataIntegrityError("inner evidence availability must be ordered UTC labels")
        if not isinstance(self.prerequisites_ready_at, pd.Timestamp):
            raise DataIntegrityError("inner evidence requires a valid prerequisites_ready_at")
        ready = self.prerequisites_ready_at
        if ready.tz is None or str(ready.tz) != "UTC":
            raise DataIntegrityError("inner evidence requires a valid prerequisites_ready_at")
        values = self.daily_returns.to_numpy(dtype="float64")
        flags = self.valid.to_numpy(dtype=bool)
        if bool(((~np.isfinite(values)) & flags).any()):
            raise DataIntegrityError("inner evidence retains missing returns only where valid is false")
        if bool((flags & ~(np.isfinite(values) & (values > -1.0))).any()):
            raise DataIntegrityError("known inner returns must be finite and exceed minus one")
        if len(self.daily_returns):
            if not self.fit_audits:
                raise DataIntegrityError("inner evidence requires fit audits for every application date")
            for stamp in self.daily_returns.index:
                covering = sum(
                    1 for audit in self.fit_audits
                    if audit.point.effective_from <= stamp < audit.point.effective_to
                )
                if covering != 1:
                    raise DataIntegrityError("inner evidence audits must cover every date exactly once")


@dataclass(frozen=True, slots=True)
class RefitPolicyChoice:
    """Audit a chronological policy choice using only previously completed inner labels.

    Missing comparison evidence is an explicit inactive choice, not permission to
    use a historically profitable default.
    """

    point: RefitPoint
    policy_id: str | None
    train_start: pd.Timestamp | None
    inner_start: pd.Timestamp | None
    inner_end: pd.Timestamp | None
    n_inner_labels: int
    paired_growth_lcb: float | None
    reason_codes: tuple[str, ...]


def _policy_train_start(policy: TrainingWindowSpec, point: RefitPoint) -> pd.Timestamp | None:
    if policy.kind == "rolling" and policy.months is not None:
        return pd.Timestamp(point.train_end - pd.DateOffset(months=int(policy.months)))
    return None


def _paired_growth_lcb(diff: np.ndarray, *, alpha: float, n_paths: int, seed: int) -> float:
    block = int(derive_block_size(diff))
    n = len(diff)
    n_blocks = (n + block - 1) // block
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    paths = np.asarray(diff[idx.reshape(n_paths, n_blocks * block)[:, :n]], dtype="float64")
    means = paths.mean(axis=1)
    return float(np.quantile(means, alpha))


def choose_refit_policy(
    evidence: Mapping[str, InnerPolicyEvidence],
    point: RefitPoint,
    *,
    spec: NestedSelectionSpec,
) -> RefitPolicyChoice:
    """Select a registered policy from common, earlier fit-then-apply evidence.

    Args:
        evidence: Complete fixed-pool policy trajectories and their provenance.
        point: Outer application interval and its fitting information cutoff.
        spec: Frozen comparison pool, control, uncertainty and tie rules.
    Returns:
        An auditable choice, or an inactive choice if common evidence is insufficient.
    Raises:
        DataIntegrityError: Pool identities, chronology or financial evidence disagree.
    """
    policy_ids = [p.policy_id for p in spec.policies]
    by_id = {p.policy_id: p for p in spec.policies}
    if set(evidence.keys()) != set(policy_ids):
        raise DataIntegrityError("evidence must cover the declared policy pool exactly")
    digests = {(e.procedure_digest, e.input_manifest_digest, e.economic_source) for e in evidence.values()}
    if len(digests) != 1:
        raise DataIntegrityError("policy identities and provenance must agree")
    for ev in evidence.values():
        vals = ev.daily_returns.to_numpy(dtype="float64")
        flags = ev.valid.to_numpy(dtype=bool)
        if len(ev.daily_returns) and not ev.fit_audits:
            raise DataIntegrityError("inner evidence requires fit audits for every application trajectory")
        if bool((flags & ~(np.isfinite(vals) & (vals > -1.0))).any()):
            raise DataIntegrityError("known inner returns must be finite and exceed minus one")
        if not isinstance(ev.daily_returns.index, pd.DatetimeIndex):
            raise DataIntegrityError("inner evidence requires a DatetimeIndex")
        for audit in ev.fit_audits:
            if (
                (audit.latest_label_end is not None and audit.latest_label_end > audit.point.train_end)
                or (
                    audit.latest_input_available_at is not None
                    and audit.latest_input_available_at > audit.point.train_end
                )
                or audit.point.train_end > audit.point.effective_from - spec.fit_latency
            ):
                raise DataIntegrityError("inner fits must precede their own application")
    common = evidence[policy_ids[0]].daily_returns.index
    for pid in policy_ids[1:]:
        common = common.intersection(evidence[pid].daily_returns.index)
    common = common.sort_values()
    cutoff = point.train_end
    ready = max(e.prerequisites_ready_at for e in evidence.values())
    kept: list[pd.Timestamp] = []
    for stamp in common:
        if stamp < ready:
            continue
        ok = True
        for pid in policy_ids:
            ev = evidence[pid]
            pos = ev.daily_returns.index.get_loc(stamp)
            if not bool(ev.valid.iloc[pos]):
                ok = False
                break
            if ev.available_at[pos] > cutoff:
                ok = False
                break
        if ok:
            kept.append(pd.Timestamp(stamp))
    common_dates = pd.DatetimeIndex(kept)
    control = by_id[spec.control_policy_id]
    control_ev = evidence[spec.control_policy_id]
    if len(common_dates) < spec.minimum_inner_labels:
        if control_ev.prerequisites_ready_at <= cutoff:
            return RefitPolicyChoice(point, spec.control_policy_id, _policy_train_start(control, point), None, None, len(common_dates), None, (INNER_EVIDENCE_INSUFFICIENT,))
        return RefitPolicyChoice(point, None, None, None, None, len(common_dates), None, (INNER_EVIDENCE_INSUFFICIENT, CONTROL_PREREQUISITES_INVALID))
    control_vals = np.log1p(control_ev.daily_returns.loc[common_dates].to_numpy(dtype="float64"))
    non_control = [p for p in spec.policies if p.policy_id != spec.control_policy_id]
    adj_alpha = float(spec.alpha) / float(len(non_control)) if non_control else float(spec.alpha)
    best_lcb: float | None = None
    best_id: str | None = None
    best_turnover = float("inf")
    for policy in non_control:
        ev = evidence[policy.policy_id]
        cand_vals = np.log1p(ev.daily_returns.loc[common_dates].to_numpy(dtype="float64"))
        diff = cand_vals - control_vals
        lcb = _paired_growth_lcb(diff, alpha=adj_alpha, n_paths=spec.bootstrap_paths, seed=spec.seed)
        turnover = float(ev.turnover.loc[common_dates].to_numpy(dtype="float64").mean())
        if best_lcb is None or lcb > best_lcb + 1e-12 or (abs(lcb - best_lcb) <= 1e-12 and turnover < best_turnover):
            best_lcb = lcb
            best_id = policy.policy_id
            best_turnover = turnover
    inner_start: pd.Timestamp | None = pd.Timestamp(common_dates[0])
    inner_end: pd.Timestamp | None = pd.Timestamp(common_dates[-1])
    if best_lcb is None or best_lcb <= 0.0:
        return RefitPolicyChoice(point, spec.control_policy_id, _policy_train_start(control, point), inner_start, inner_end, len(common_dates), best_lcb, (NO_ESTIMATED_IMPROVEMENT,))
    winner = by_id[best_id] if best_id is not None else control
    return RefitPolicyChoice(point, best_id, _policy_train_start(winner, point), inner_start, inner_end, len(common_dates), best_lcb, (PAIRED_LCB_IMPROVEMENT,))
