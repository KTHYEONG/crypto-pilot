"""MHS deploy gate: single-verdict deployment decision (INV-SINGLE-VERDICT).

The deploy verdict is owned solely by :class:`DeployGateResult`, derived from
the already-materialized daily fold returns (no replay). Axis 0 (integrity)
short-circuits before any alpha/survival statistic is computed
(INV-INTEGRITY-FIRST); axis 1 tests edge against the absolute floor 0
(INV-NO-CIRCULAR); axis 2 compares bootstrap-distribution probabilities
against the registered envelope budgets (INV-DISTRIBUTION-NOT-PATH).

Registered constants are the error rate and the risk envelope only; the E3
binomial threshold and the S3 shuffle null quantile are derived per run
(INV-DERIVED-THRESHOLD). All randomness flows from ``np.random.default_rng``
seeded explicitly (INV-DETERMINISTIC).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.common.errors import DataIntegrityError
from src.mhs.deployment_policy import live_parity_blockers
from src.mhs.params import GrowthRiskEnvelope
from src.quant.evaluation.reliability import derive_block_size

DEPLOY_GATE_ALPHA: float = 0.05
DEPLOY_GATE_BOOTSTRAP_PATHS: int = 2000
DEPLOY_GATE_NULL_DRAWS: int = 5000
DEPLOY_GATE_SEED: int = 20260917
DEPLOY_GATE_BARS_PER_YEAR: float = 365.0
TAIL_FOLD_FRACTION: float = 0.25

GATE_REPORT_NOT_COMPLETE: str = "I0_REPORT_NOT_COMPLETE"
GATE_FOLD_INTEGRITY: str = "I1_FOLD_INTEGRITY_FAILURE"
GATE_BLEND_LEDGER_INVALID: str = "I2_BLEND_LEDGER_INVALID"
GATE_INPUT_UNSEALED: str = "I3_INPUT_UNSEALED"
GATE_LIVE_PARITY_BLOCKED: str = "I4_LIVE_PARITY_BLOCKED"
GATE_BACKTEST_RELIABILITY_NOT_ELIGIBLE: str = "I5_BACKTEST_RELIABILITY_NOT_ELIGIBLE"
GATE_OOS_GROWTH: str = "E1_OOS_GROWTH_LCB_NOT_POSITIVE"
GATE_STRESS_GROWTH: str = "E2_STRESS_GROWTH_LCB_NOT_POSITIVE"
GATE_EDGE_BREADTH: str = "E3_EDGE_BREADTH_BELOW_BINOMIAL_CRITICAL"
GATE_MDD_BUDGET: str = "S1_MDD_BUDGET_EXCEEDED"
GATE_RUIN_BUDGET: str = "S2_RUIN_BUDGET_EXCEEDED"
GATE_TIME_CONCENTRATION: str = "S3_GROWTH_TIME_CONCENTRATED"
INTEGRITY_GATE_CODES: frozenset[str] = frozenset(
    {
        GATE_REPORT_NOT_COMPLETE,
        GATE_FOLD_INTEGRITY,
        GATE_BLEND_LEDGER_INVALID,
        GATE_INPUT_UNSEALED,
        GATE_LIVE_PARITY_BLOCKED,
    }
)


@dataclass(frozen=True, slots=True)
class DeployGateResult:
    """Single deployment verdict (INV-SINGLE-VERDICT).

    ``reason_codes`` is a sorted tuple; ``metrics`` carries only the values
    actually computed up to the blocking axis.
    """

    go: bool
    reason_codes: tuple[str, ...]
    metrics: dict[str, float]


def block_bootstrap_paths(
    returns: np.ndarray, *, n_paths: int, path_len: int, seed: int
) -> np.ndarray:
    """Fixed-length circular block bootstrap with shape ``(n_paths, path_len)``."""
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if path_len < 1:
        raise ValueError(f"path_len must be >= 1, got {path_len}")
    x = np.asarray(returns, dtype="float64")
    if x.size == 0:
        raise ValueError("returns must be non-empty")
    if not bool(np.isfinite(x).all()):
        raise ValueError("returns must be finite")
    block = int(derive_block_size(x))
    n_blocks = (path_len + block - 1) // block
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(x)
    return np.asarray(
        x[idx.reshape(n_paths, n_blocks * block)[:, :path_len]], dtype="float64"
    )


def annualized_log_growth_lcb(
    returns: pd.Series,
    *,
    alpha: float = DEPLOY_GATE_ALPHA,
    n_paths: int = DEPLOY_GATE_BOOTSTRAP_PATHS,
    seed: int = DEPLOY_GATE_SEED,
) -> tuple[float, float]:
    """Return ``(lcb, point)`` of the annualized log growth."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(returns) < 2:
        raise ValueError("returns must contain at least 2 rows")
    x = returns.to_numpy(dtype="float64")
    if not bool(np.isfinite(x).all()):
        raise ValueError("returns must be finite")
    if bool((x <= -1.0).any()):
        raise ValueError("returns must exceed -1.0 for log1p")
    point = float(np.log1p(x).mean() * DEPLOY_GATE_BARS_PER_YEAR)
    paths = block_bootstrap_paths(x, n_paths=n_paths, path_len=len(x), seed=seed)
    g = np.log1p(paths).mean(axis=1) * DEPLOY_GATE_BARS_PER_YEAR
    lcb = float(np.quantile(g, alpha))
    return (lcb, point)


def profitable_fold_critical_count(
    n_folds: int, *, alpha: float = DEPLOY_GATE_ALPHA
) -> int:
    """Minimum profitable-fold count beating the exact binomial null (p=0.5)."""
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for k in range(1, n_folds + 1):
        if float(binom.sf(k - 1, n_folds, 0.5)) <= alpha:
            return k
    raise DataIntegrityError(
        f"no binomial critical count exists for n_folds={n_folds} alpha={alpha}"
    )


def survival_probabilities(
    returns: pd.Series,
    *,
    max_drawdown: float,
    ruin_fraction: float,
    horizon_years: float,
    n_paths: int = DEPLOY_GATE_BOOTSTRAP_PATHS,
    seed: int = DEPLOY_GATE_SEED,
) -> tuple[float, float]:
    """Bootstrap-distribution breach probabilities ``(p_mdd, p_ruin)``."""
    if not 0.0 < max_drawdown <= 1.0:
        raise ValueError(f"max_drawdown must be in (0, 1], got {max_drawdown}")
    if not 0.0 < ruin_fraction < 1.0:
        raise ValueError(f"ruin_fraction must be in (0, 1), got {ruin_fraction}")
    if not horizon_years > 0:
        raise ValueError(f"horizon_years must be > 0, got {horizon_years}")
    x = returns.to_numpy(dtype="float64")
    path_len = round(horizon_years * DEPLOY_GATE_BARS_PER_YEAR)
    paths = block_bootstrap_paths(x, n_paths=n_paths, path_len=path_len, seed=seed)
    cum = np.cumprod(1.0 + paths, axis=1)
    augmented = np.concatenate(
        [np.ones((cum.shape[0], 1), dtype="float64"), cum.astype("float64")], axis=1
    )
    mdd = (1.0 - augmented / np.maximum.accumulate(augmented, axis=1)).max(axis=1)
    return (
        float(np.mean(mdd > max_drawdown)),
        float(np.mean(cum[:, -1] < ruin_fraction)),
    )


def tail_concentration(
    fold_returns: Sequence[pd.Series],
    *,
    alpha: float = DEPLOY_GATE_ALPHA,
    n_draws: int = DEPLOY_GATE_NULL_DRAWS,
    seed: int = DEPLOY_GATE_SEED,
) -> tuple[float, float]:
    """Return ``(observed_share, null_quantile)`` of tail-fold log-growth share."""
    folds = list(fold_returns)
    if len(folds) < 2:
        raise ValueError("fold_returns must contain at least 2 folds")
    arrays = [s.to_numpy(dtype="float64") for s in folds]
    lens = [len(a) for a in arrays]
    pooled = np.concatenate(arrays)
    n = len(folds)
    tail_start = n - max(1, round(n * TAIL_FOLD_FRACTION))
    g = [float(np.log1p(a).sum()) for a in arrays]
    total = float(sum(g))
    if total <= 0.0:
        raise DataIntegrityError("total log growth must be positive for concentration")
    observed = float(sum(g[tail_start:]) / total)
    rng = np.random.default_rng(seed)
    splits = np.cumsum(lens)[:-1]
    # 순열은 pooled의 multiset을 보존하므로 draw의 총로그성장은 위에서 검증한
    # total과 같은 수다(합산 순서 차이뿐). draw별 양수 검사는 도달 불가 분기라 두지 않는다.
    shares: list[float] = []
    for _ in range(n_draws):
        perm = rng.permutation(pooled)
        parts = np.split(perm, splits)
        gi = [float(np.log1p(p).sum()) for p in parts]
        shares.append(float(sum(gi[tail_start:]) / float(sum(gi))))
    null_quantile = float(
        np.quantile(np.asarray(shares, dtype="float64"), 1.0 - alpha)
    )
    return (observed, null_quantile)


def fold_daily_returns(replay: Any) -> pd.Series:
    """Daily returns of a fold replay's equity ledger.

    ``replay`` is ``Any`` because callers pass both typed replay results and
    lightweight ``SimpleNamespace`` doubles.
    """
    if replay is None:
        raise DataIntegrityError("fold replay is None")
    ledger = getattr(replay, "ledger", None)
    if ledger is None:
        raise DataIntegrityError("fold replay has no ledger")
    equity = getattr(ledger, "equity", None)
    if equity is None:
        raise DataIntegrityError("fold replay ledger has no equity")
    return equity.resample("1D").last().pct_change().dropna().astype("float64")


def integrity_reasons_from_report(report: Any, request: Any) -> tuple[str, ...]:
    """Axis-0 fact collection; every absence fires fail-closed.

    ``report``/``request`` are ``Any`` because callers pass both the typed
    diagnostic report/request and lightweight ``SimpleNamespace`` doubles.
    Only ``getattr(obj, name, None)`` defaults are used; no broad
    ``try/except`` fallback exists.
    """
    # 지연 임포트: mhs.evaluation 패키지 초기화가 report.persist -> deploy_gate로
    # 되돌아 들어오는 순환을 모듈 최상단에서 피한다(INV-SINGLE-CERTIFICATION).
    from src.mhs.evaluation.integrity import replay_ledger_certified

    codes: list[str] = []
    if getattr(report, "status", None) != "COMPLETE":
        codes.append(GATE_REPORT_NOT_COMPLETE)
    folds = getattr(report, "folds", ())
    if folds is None or len(folds) == 0:
        codes.append(GATE_FOLD_INTEGRITY)
    else:
        for fold in folds:
            strict = getattr(fold, "strict", None)
            failures = getattr(fold, "failures", None)
            if strict is None or failures is None or len(failures) != 0:
                codes.append(GATE_FOLD_INTEGRITY)
                break
    blend = getattr(report, "blend", None)
    primary = getattr(blend, "primary", None) if blend is not None else None
    # 단일 인증 헬퍼가 말기 재고 공개 여부를 판정한다(인라인 추출 금지).
    if blend is None or primary is None or not replay_ledger_certified(primary):
        codes.append(GATE_BLEND_LEDGER_INVALID)
    reliability = getattr(report, "backtest_reliability", None)
    digest = (
        getattr(reliability, "input_manifest_digest", None)
        if reliability is not None
        else None
    )
    if reliability is None or digest is None:
        codes.append(GATE_INPUT_UNSEALED)
    if reliability is None or getattr(reliability, "eligible", None) is not True:
        codes.append(GATE_BACKTEST_RELIABILITY_NOT_ELIGIBLE)
    if live_parity_blockers(request):
        codes.append(GATE_LIVE_PARITY_BLOCKED)
    return tuple(sorted(set(codes)))


def evaluate_deploy_gate(
    *,
    fold_returns: Sequence[pd.Series],
    fold_stress_returns: Sequence[pd.Series],
    integrity_reasons: tuple[str, ...],
    envelope: GrowthRiskEnvelope,
    alpha: float = DEPLOY_GATE_ALPHA,
    n_paths: int = DEPLOY_GATE_BOOTSTRAP_PATHS,
    n_draws: int = DEPLOY_GATE_NULL_DRAWS,
    seed: int = DEPLOY_GATE_SEED,
) -> DeployGateResult:
    """Three-axis gate with short-circuit at each axis boundary."""
    if len(tuple(integrity_reasons)):
        return DeployGateResult(
            go=False,
            reason_codes=tuple(sorted(set(integrity_reasons))),
            metrics={},
        )
    fr = list(fold_returns)
    sr = list(fold_stress_returns)
    if len(fr) != len(sr):
        raise ValueError("fold_returns and fold_stress_returns must match in length")
    oos = pd.concat(fr).sort_index()
    stress = pd.concat(sr).sort_index()
    e1_lcb, e1_pt = annualized_log_growth_lcb(oos, alpha=alpha, n_paths=n_paths, seed=seed)
    e2_lcb, e2_pt = annualized_log_growth_lcb(
        stress, alpha=alpha, n_paths=n_paths, seed=seed
    )
    critical = profitable_fold_critical_count(len(fr), alpha=alpha)
    n_prof = sum(
        1 for s in fr if float(np.log1p(s.to_numpy(dtype="float64")).sum()) > 0.0
    )
    n_prof_stress = sum(
        1 for s in sr if float(np.log1p(s.to_numpy(dtype="float64")).sum()) > 0.0
    )
    axis1_codes: list[str] = []
    if e1_lcb <= 0.0:
        axis1_codes.append(GATE_OOS_GROWTH)
    if e2_lcb <= 0.0:
        axis1_codes.append(GATE_STRESS_GROWTH)
    if n_prof < critical or n_prof_stress < critical:
        axis1_codes.append(GATE_EDGE_BREADTH)
    metrics: dict[str, float] = {
        "n_folds": float(len(fr)),
        "breadth_critical_count": float(critical),
        "profitable_folds": float(n_prof),
        "profitable_folds_stress": float(n_prof_stress),
        "oos_ann_log_growth": float(e1_pt),
        "oos_ann_log_growth_lcb": float(e1_lcb),
        "stress_ann_log_growth": float(e2_pt),
        "stress_ann_log_growth_lcb": float(e2_lcb),
    }
    if axis1_codes:
        return DeployGateResult(
            go=False,
            reason_codes=tuple(sorted(set(axis1_codes))),
            metrics=metrics,
        )
    p_mdd, p_ruin = survival_probabilities(
        oos,
        max_drawdown=envelope.max_drawdown,
        ruin_fraction=envelope.ruin_fraction,
        horizon_years=envelope.horizon_years,
        n_paths=n_paths,
        seed=seed,
    )
    observed, null_q = tail_concentration(fr, alpha=alpha, n_draws=n_draws, seed=seed)
    axis2_codes: list[str] = []
    if p_mdd > envelope.max_drawdown_prob:
        axis2_codes.append(GATE_MDD_BUDGET)
    if p_ruin > envelope.max_ruin_prob:
        axis2_codes.append(GATE_RUIN_BUDGET)
    if observed > null_q:
        axis2_codes.append(GATE_TIME_CONCENTRATION)
    metrics.update(
        {
            "p_mdd_breach": float(p_mdd),
            "mdd_budget": float(envelope.max_drawdown),
            "mdd_budget_prob": float(envelope.max_drawdown_prob),
            "p_ruin": float(p_ruin),
            "ruin_fraction": float(envelope.ruin_fraction),
            "ruin_max_prob": float(envelope.max_ruin_prob),
            "tail_share_observed": float(observed),
            "tail_share_null_quantile": float(null_q),
        }
    )
    return DeployGateResult(
        go=not axis2_codes,
        reason_codes=tuple(sorted(set(axis2_codes))),
        metrics=metrics,
    )


def deploy_gate_from_report(
    report: Any,
    request: Any,
    *,
    alpha: float = DEPLOY_GATE_ALPHA,
    n_paths: int = DEPLOY_GATE_BOOTSTRAP_PATHS,
    n_draws: int = DEPLOY_GATE_NULL_DRAWS,
    seed: int = DEPLOY_GATE_SEED,
) -> DeployGateResult:
    """Thin extraction seam: strict fold ledgers plus the resolved envelope.

    ``report``/``request`` are ``Any`` because callers pass both the typed
    diagnostic report/request and lightweight ``SimpleNamespace`` doubles.
    """
    import src.mhs.research_go as research_go_mod

    envelope = research_go_mod._resolved_growth_envelope(request)
    reasons = integrity_reasons_from_report(report, request)
    if reasons:
        return evaluate_deploy_gate(
            fold_returns=(),
            fold_stress_returns=(),
            integrity_reasons=reasons,
            envelope=envelope,
            alpha=alpha,
            n_paths=n_paths,
            n_draws=n_draws,
            seed=seed,
        )
    folds = sorted(
        getattr(report, "folds", ()),
        key=lambda f: getattr(f, "fold_index", 0),
    )
    fr = tuple(fold_daily_returns(getattr(f, "strict", None)) for f in folds)
    sr = tuple(fold_daily_returns(getattr(f, "stress", None)) for f in folds)
    return evaluate_deploy_gate(
        fold_returns=fr,
        fold_stress_returns=sr,
        integrity_reasons=(),
        envelope=envelope,
        alpha=alpha,
        n_paths=n_paths,
        n_draws=n_draws,
        seed=seed,
    )
