from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import optuna.storages.journal
from numpy.typing import NDArray
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from src.domain.futures.optimization.ml_context import MLPhaseDContext
from src.domain.futures.optimization.objectives import objective_ml_phase_d
from src.domain.futures.optimization.observability.dashboard import safe_float
from src.domain.futures.optimization.observability.trial_observability import (
    build_compact_trial_summary,
    increment_failure_reason_count,
    init_failure_reason_counts,
    trial_elapsed_seconds,
)
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

_logger: logging.Logger = logging.getLogger("run_tracker")


# Removed Redis connection helper functions


def _normalize_phase_workers(phase_workers: dict[str, int]) -> dict[str, int]:
    """Normalize phase workers to canonical keys."""
    return {
        "phase_a1": int(phase_workers.get("phase_a1", 1)),
        "phase_a2": int(phase_workers.get("phase_a2", 1)),
        "phase_b": int(phase_workers.get("phase_b", 1)),
    }


def log_optuna_contract(
    *,
    project_root: str | Path,
    requested_trials_per_phase: int,
    phase_workers: dict[str, int],
    seed: int,
    storage_url: str,
) -> dict[str, Any]:
    """Persist and log Optuna phase contract."""
    from src.domain.futures.optimization.workflow import (
        build_phase_a1_pruner,
        build_phase_a1_sampler,
        build_phase_a2_sampler,
        build_phase_b_sampler,
    )

    normalized_workers = _normalize_phase_workers(phase_workers)
    phase_trials = {
        "phase_a1": max(1, int(requested_trials_per_phase * 0.5)),
        "phase_a2": max(1, int(requested_trials_per_phase * 0.2)),
        "phase_b": max(1, int(requested_trials_per_phase * 0.3)),
    }
    sampler_by_phase = {
        "phase_a1": build_phase_a1_sampler(seed).__class__.__name__,
        "phase_a2": build_phase_a2_sampler(seed).__class__.__name__,
        "phase_b": build_phase_b_sampler(seed).__class__.__name__,
    }
    pruner_name = build_phase_a1_pruner().__class__.__name__
    planned_total = int(sum(phase_trials.values()))
    payload: dict[str, Any] = {
        "requested_trials_per_phase": int(requested_trials_per_phase),
        "planned_total_trials": planned_total,
        "trials_per_phase": phase_trials,
        "worker_by_phase": normalized_workers,
        "sampler_by_phase": sampler_by_phase,
        "pruner_by_phase": {
            "phase_a1": pruner_name,
            "phase_a2": pruner_name,
            "phase_b": pruner_name,
        },
        "storage_url": storage_url,
    }

    # Layer2 스타일 헤더의 연장선으로 심플하게 출력
    _logger.info(f"   [OPTUNA] Storage: {storage_url}")
    _logger.info(
        f"   [PHASES] Total: {planned_total} "
        f"(A1:{phase_trials['phase_a1']}, A2:{phase_trials['phase_a2']}, B:{phase_trials['phase_b']})"
    )

    for phase_key in ("phase_a1", "phase_a2", "phase_b"):
        rationale = ""
        if phase_key == "phase_b" and int(normalized_workers.get(phase_key, 1)) == 1:
            rationale = " rationale=sqlite_complete_trial_race_prevention"
        _logger.debug(
            "[OPTUNA-CONTRACT] phase=%s sampler=%s pruner=%s workers=%d trials=%d%s",
            phase_key,
            sampler_by_phase[phase_key],
            pruner_name,
            int(normalized_workers.get(phase_key, 1)),
            int(phase_trials[phase_key]),
            rationale,
        )
    out_dir = Path(project_root) / "logs" / "futures" / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optuna_contract.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def setup_optuna_storage(project_root: str | Path) -> tuple[str, optuna.storages.BaseStorage]:
    """SQLite WAL 기반 RDBStorage를 생성하고 반환한다.

    Returns:
        (storage_url, storage) 튜플.
        storage_url: "sqlite:///absolute/path/to/optuna.db" 형식.
    """
    import sqlite3

    db_dir = Path(project_root) / "logs" / "futures" / "optimization"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "optuna.db"

    # Enable SQLite WAL mode before initializing optuna RDBStorage to prevent DB lock contention
    with contextlib.suppress(Exception), sqlite3.connect(str(db_path), timeout=10.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

    storage_url = f"sqlite:///{db_path.resolve()}"
    storage = optuna.storages.RDBStorage(
        storage_url,
        engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}},
    )

    return storage_url, storage


def get_or_create_study(
    study_name: str,
    storage: optuna.storages.BaseStorage,
    sampler: optuna.samplers.BaseSampler,
    resume: bool = False,
    pruner: optuna.pruners.BasePruner | None = None,
    directions: list[str] | tuple[str, ...] | None = None,
) -> optuna.Study:
    """Get existing or create new Optuna study, optionally deleting existing one."""
    if not resume:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            _logger.info(" [OPT] Deleted existing study '%s' for a fresh start.", study_name)
        except KeyError:
            pass  # Study doesn't exist yet

    create_kwargs: dict[str, Any] = {
        "study_name": study_name,
        "storage": storage,
        "sampler": sampler,
        "pruner": pruner,
        "load_if_exists": True,
    }
    if directions is not None and len(directions) > 1:
        create_kwargs["directions"] = list(directions)
    else:
        direction = str(directions[0]) if directions else "maximize"
        create_kwargs["direction"] = direction

    study = optuna.create_study(**create_kwargs)
    return study


def champion_store_study_name(tag: str) -> str:
    """전역 챔피언 레저(영구 보존, 매 실행 초기화 대상에서 제외) study 이름."""
    return f"l2_champion_store_{tag}"


@contextmanager
def isolated_optuna_storage() -> Iterator[optuna.storages.BaseStorage]:
    """champion_store 리저와 격리된 storage. [ADR_20260705_L1L2_REGIME_CONDITIONAL_WEIGHT]

    호출부 배선(economic replay harness 교체)은 범위 밖 — 유틸리티만 제공.
    """
    storage = optuna.storages.InMemoryStorage()
    yield storage


def _distribution_for_spec(spec: Mapping[str, Any]) -> optuna.distributions.BaseDistribution:
    spec_type = spec["type"]
    if spec_type == "categorical":
        return optuna.distributions.CategoricalDistribution(spec["choices"])
    if spec_type == "int":
        return optuna.distributions.IntDistribution(int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    return optuna.distributions.FloatDistribution(
        float(spec["low"]),
        float(spec["high"]),
        step=float(spec["step"]) if "step" in spec else None,
        log=bool(spec.get("log", False)),
    )


def load_champion_params(tag: str, storage: optuna.storages.BaseStorage) -> dict[str, Any] | None:
    """영구 챔피언 레저에서 현재까지의 최고 파라미터를 조회. 레저가 비어있으면 None."""
    try:
        study = optuna.load_study(study_name=champion_store_study_name(tag), storage=storage)
    except KeyError:
        return None
    if not study.trials:
        return None
    try:
        return dict(study.best_trial.params)
    except ValueError:
        return None


def update_champion_store(
    tag: str,
    storage: optuna.storages.BaseStorage,
    params: Mapping[str, Any],
    value: float,
    space: Mapping[str, Mapping[str, Any]],
) -> bool:
    """현 run의 챔피언이 영구 레저의 기존 최고값보다 우수하면 갱신.

    레저 study는 `get_or_create_study(resume=False)` 초기화 대상이 아니며,
    `optimize()`/`ask()`로 샘플링되지 않는 순수 기록용이므로 search space가
    실행마다 바뀌어도 dynamic-search-space 경고와 무관하다.

    Returns:
        True면 신규 챔피언으로 갱신됨, False면 기존 챔피언이 더 우수해 유지됨.
    """
    study_name = champion_store_study_name(tag)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    prior_best = study.best_value if study.trials else float("-inf")
    if value <= prior_best:
        return False

    distributions = {key: _distribution_for_spec(space[key]) for key in params if key in space}
    trial = optuna.trial.create_trial(
        params={k: v for k, v in params.items() if k in distributions},
        distributions=distributions,
        value=value,
        state=TrialState.COMPLETE,
    )
    study.add_trial(trial)
    return True


def adr_sharpe_pool_study_name(tag: str) -> str:
    """ADR-레벨 전량 기록(승패 무관) study 이름. champion_store_study_name과 별개 study.
    [ADR_20260705_L3_ROLLING_HOLDOUT_PANEL]
    """
    return f"adr_sharpe_pool_{tag}"


def record_adr_evaluation(
    tag: str,
    storage: optuna.storages.BaseStorage,
    *,
    sharpe: float,
    adr_id: str,
) -> None:
    """ADR 시도의 realized Sharpe를 승패 무관 전량 기록(update_champion_store와 달리
    조건부 return 없음). params/distributions는 빈 dict — 이 study는 튜닝용이 아니라
    순수 이력 기록용이므로 dynamic-search-space 경고와 무관(champion_store와 동일 근거).
    """
    study_name = adr_sharpe_pool_study_name(tag)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    trial = optuna.trial.create_trial(
        params={},
        distributions={},
        value=sharpe,
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={"adr_id": adr_id},
    )
    study.add_trial(trial)


def get_adr_sharpe_pool(
    tag: str,
    storage: optuna.storages.BaseStorage,
) -> NDArray[np.float64]:
    """해당 tag의 과거 ADR Sharpe 이력 전체를 배열로 반환. study 없으면 빈 배열."""
    study_name = adr_sharpe_pool_study_name(tag)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return np.array([], dtype=np.float64)

    values = [t.value for t in study.trials if t.value is not None]
    return np.array(values, dtype=np.float64)


def compute_adr_level_deflated_sharpe(
    candidate_returns: NDArray[np.float64],
    *,
    tag: str,
    storage: optuna.storages.BaseStorage,
    tf: str,
) -> float:
    """기존 _deflated_sharpe_probability를 ADR-레벨 pool로 호출하는 얇은 래퍼.
    신규 통계 공식 없음 — allocation.metrics._deflated_sharpe_probability 재사용.
    """
    from src.domain.futures.optimization.metrics import (
        _bars_per_year_for_tf,
        _deflated_sharpe_probability,
    )

    pool = get_adr_sharpe_pool(tag, storage)
    return _deflated_sharpe_probability(
        selected_rets=candidate_returns,
        completed_trial_sharpes=pool,
        effective_trial_count=float(len(pool)),
        bars_per_year=_bars_per_year_for_tf(tf),
    )


def _nsga2_constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
    """NSGA-II feasibility constraints. Return ≤0 = feasible, >0 = violation magnitude."""
    ua = trial.user_attrs
    violations = []
    # C1: CHOP loss share must be ≤ 0.60
    chop_loss = float(ua.get("awf_chop_loss_share", 0.0))
    violations.append(chop_loss - 0.60)
    # C2: CHOP trade share must be ≤ 0.70
    chop_trade = float(ua.get("awf_chop_trade_share", 0.0))
    violations.append(chop_trade - 0.70)
    # C3: worst AWF leg log-TW must be ≥ -0.10
    worst_leg = float(ua.get("awf_worst_leg_log_tw", -999.0))
    violations.append(-0.10 - worst_leg)  # violation if worst_leg < -0.10
    # C4: minimum positive leg fraction must be ≥ 0.40
    pos_frac = float(ua.get("awf_pos_frac", 0.0))
    violations.append(0.40 - pos_frac)  # violation if pos_frac < 0.40
    return violations


def ml_phase_d_sampler(
    seed: int, n_trials: int = 200, constraints_func: Any | None = None
) -> optuna.samplers.BaseSampler:
    """Multivariate TPE sampler for single-objective J-score maximization."""
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
        pop = int(OPT_FUTURES_CONFIG.get("FUTURES_NSGA2_POPULATION_SIZE", 30))
        effective_constraints = constraints_func if constraints_func is not None else _nsga2_constraints
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=pop,
            crossover_prob=0.9,
            mutation_prob=0.1,
            constraints_func=effective_constraints,
        )
    cfg_startup = int(OPT_FUTURES_CONFIG.get("tpe_n_startup_trials", 50))
    frac = float(OPT_FUTURES_CONFIG.get("FUTURES_ML_PHASE_D_TPE_STARTUP_FRAC", 1.0))
    frac = max(0.01, min(1.0, frac))
    from_frac = max(1, int(float(n_trials) * frac))
    n_startup = max(1, min(cfg_startup, from_frac, max(1, n_trials - 1)))
    return TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_ei_candidates=48,
    )


def ml_phase_d_sampler_coordinate(seed: int, n_trials: int, phase: str) -> optuna.samplers.BaseSampler:
    """Phase B: more startup trials + multivariate TPE; phases A/C use 20 startup."""
    if OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False):
        return ml_phase_d_sampler(seed, n_trials)
    nt = max(2, int(n_trials))
    if phase == "B":
        n_startup = min(40, max(1, nt - 1))
        return TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
            group=True,
            constant_liar=True,
            n_ei_candidates=48,
        )
    n_startup = min(20, max(1, nt - 1))
    return TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        multivariate=False,
        group=False,
        constant_liar=True,
        n_ei_candidates=24,
    )


def ml_trial_passes_hard_gates(
    trial: optuna.trial.FrozenTrial,
    pbo_obs: float = 0.0,
    check_pbo: bool = True,
    *,
    pbo_max: float | None = None,
    dsr_min: float | None = None,
) -> bool:
    """Check if a trial passes the hard quality gates defined in config."""
    cfg = OPT_FUTURES_CONFIG

    # Use IS metrics if AWF metrics are missing (optimization phase)
    is_mdd = trial.user_attrs.get("IS_MDD")
    is_dsr = trial.user_attrs.get("IS_DSR")

    if is_mdd is not None and is_dsr is not None:
        # Optimization phase checks
        is_mdd_val = float(is_mdd)
        is_dsr_val = float(is_dsr)
        mdd_limit = float(cfg.get("FUTURES_MAX_MDD", 25.0))
        dsr_floor = float(dsr_min if dsr_min is not None else cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.30))
        if is_mdd_val >= mdd_limit:
            return False
        return is_dsr_val >= dsr_floor

    # Historical AWF-based checks
    pbo_lim = float(pbo_max if pbo_max is not None else cfg.get("FUTURES_PBO_MAX", 0.40))
    if check_pbo and float(pbo_obs) >= pbo_lim:
        return False
    dsr = float(trial.user_attrs.get("gate1_dsr", -9.0))
    dsr_floor = float(dsr_min if dsr_min is not None else cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.80))
    if dsr < dsr_floor:
        return False
    p10_floor = float(cfg.get("FUTURES_AWF_P10_LOG_TW_MIN", -0.10))
    p10_cpcv = float(
        trial.user_attrs.get("awf_worst_leg_log_tw", trial.user_attrs.get("ml_p10_log_growth_cpcv", -999.0))
    )
    if p10_cpcv <= p10_floor:
        return False
    mdd_limit = float(cfg.get("FUTURES_MAX_MDD", 22.0))
    if float(trial.user_attrs.get("awf_worst_mdd_pct", trial.user_attrs.get("ml_worst_mdd_cpcv", 999.0))) >= mdd_limit:
        return False
    # Dynamic trade density: scale minimum trades with IS span to avoid regime-size bias.
    span_bars = int(trial.user_attrs.get("gate1_eff_ref_len", 0))
    bars_per_trade_est = float(cfg.get("FUTURES_BARS_PER_TRADE_EST", 200))
    min_trades_dynamic = max(12.0, float(span_bars) / bars_per_trade_est) if span_bars > 0 else 12.0
    return float(trial.user_attrs.get("avg_trades", 0.0)) >= min_trades_dynamic


# Global context for worker processes to leverage Linux Copy-on-Write (CoW)
# This prevents expensive IPC pickling of large context objects.
_GLOBAL_BASE_CTX: MLPhaseDContext | None = None
_GLOBAL_OBJECTIVE_FN: Callable[[optuna.Trial, MLPhaseDContext], Any] = objective_ml_phase_d


def resolve_futures_parallel_policy(symbol_count: int) -> int:
    """Determine the optimal number of workers for parallel optimization."""
    logical_cpus = max(1, os.cpu_count() or 1)
    return max(1, min(8, logical_cpus))


def optimize_worker(s_name: str, s_url: str, chunk_size: int) -> None:
    """Worker function for parallel Optuna optimization using global context."""
    import os

    with contextlib.suppress(Exception):
        # Lower CPU priority so that optimization does not lag host gaming or chrome activities
        os.nice(10)

    global _GLOBAL_BASE_CTX
    if _GLOBAL_BASE_CTX is None:
        raise RuntimeError("Worker started without _GLOBAL_BASE_CTX")
    objective_fn = _GLOBAL_OBJECTIVE_FN

    # Each process loads the study and runs its portion of trials
    inner_storage = optuna.storages.RDBStorage(
        s_url, engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}}
    )
    study = optuna.load_study(study_name=s_name, storage=inner_storage)
    trial_timeout_sec = int(OPT_FUTURES_CONFIG.get("FUTURES_OPT_TRIAL_TIMEOUT_SEC", 180))

    def _objective_with_timeout(tr: optuna.Trial) -> Any:
        if trial_timeout_sec <= 0:
            return objective_fn(tr, _GLOBAL_BASE_CTX)

        def _timeout_handler(_signum: int, _frame: Any) -> None:
            raise RuntimeError(f"trial_timeout>{trial_timeout_sec}s")

        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(trial_timeout_sec)
        try:
            return objective_fn(tr, _GLOBAL_BASE_CTX)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)

    def _trial_finish_callback(_study: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        elapsed = trial_elapsed_seconds(tr)
        if tr.state == optuna.trial.TrialState.COMPLETE:
            _logger.info("%s", build_compact_trial_summary(tr, elapsed_sec=elapsed))
        elif tr.state == optuna.trial.TrialState.PRUNED:
            summary = build_compact_trial_summary(tr, elapsed_sec=elapsed)
            # Replace [TRIAL] prefix with [PRUNE] for easier grep
            prune_msg = summary.replace("[TRIAL]", "[PRUNE]", 1)
            _logger.debug("%s", prune_msg)

    study.optimize(
        _objective_with_timeout,
        n_trials=chunk_size,
        n_jobs=1,
        catch=(ValueError, RuntimeError),
        callbacks=[_trial_finish_callback],
    )


def cfg_hash_for_run(cfg: dict[str, Any]) -> str:
    """Generate a stable hash for the relevant futures config keys."""
    relevant = {k: v for k, v in cfg.items() if str(k).startswith("FUTURES_")}
    raw = json.dumps(relevant, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def build_joint_study_name(
    tf: str,
    fetch_start_date: str,
    end_date: str,
    symbols: list[str],
    cfg: dict[str, Any],
) -> str:
    """Build a unique study name for Optuna based on parameters."""
    symbols_sorted = sorted(str(s) for s in symbols)
    sym_raw = json.dumps(symbols_sorted, ensure_ascii=True, separators=(",", ":"))
    sym_fp = hashlib.sha256(sym_raw.encode()).hexdigest()[:10]
    cfg_fp = cfg_hash_for_run(cfg)
    return f"futures_joint_tpe_tf{tf}_win{fetch_start_date}_{end_date}_n{len(symbols_sorted)}_s{sym_fp}_c{cfg_fp}"


def short_git_rev(project_root: str | Path) -> str:
    """Best-effort short git rev; 'nogit' when unavailable."""
    try:
        out = subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def build_run_id(
    tf: str,
    fetch_start_date: str,
    end_date: str,
    symbols: list[str],
    cfg: dict[str, Any],
    project_root: str | Path,
) -> str:
    """Generate a unique Run ID for a specific optimization run."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sym_raw = json.dumps(sorted(str(s) for s in symbols), ensure_ascii=True, separators=(",", ":"))
    sym_fp = hashlib.sha256(sym_raw.encode()).hexdigest()[:8]
    cfg_fp = cfg_hash_for_run(cfg)[:8]
    return f"{ts}_tf{tf}_s{sym_fp}_c{cfg_fp}_{short_git_rev(project_root)}"


def apply_ops_profile_overrides(
    cfg: dict[str, Any],
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply profile config overrides to runtime config and return flattened applied map."""
    applied: dict[str, Any] = {}
    if not profile:
        return applied
    overrides = profile.get("config_overrides")
    if not isinstance(overrides, dict):
        return applied
    for key, value in overrides.items():
        if isinstance(value, dict):
            base_val = cfg.get(key)
            if isinstance(base_val, dict):
                merged = dict(base_val)
                for sub_key, sub_value in value.items():
                    merged[sub_key] = sub_value
                    applied[f"{key}.{sub_key}"] = sub_value
                cfg[key] = merged
            else:
                cfg[key] = dict(value)
                for sub_key, sub_value in value.items():
                    applied[f"{key}.{sub_key}"] = sub_value
        else:
            cfg[key] = value
            applied[key] = value
    return applied


def collect_run_summary_from_study(
    study_ml: optuna.Study,
    run_id: str,
    *,
    study_name: str,
    requested_trials: int,
    optuna_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect statistics and best trials for a specific Run ID from an Optuna study."""
    trials = study_ml.get_trials(deepcopy=False)
    scoped = [t for t in trials if str(t.user_attrs.get("run_id", "")) == str(run_id)]
    complete = [t for t in scoped if t.state == TrialState.COMPLETE]
    pareto = list(study_ml.best_trials or [])
    pareto_scoped = [t for t in pareto if str(t.user_attrs.get("run_id", "")) == str(run_id)]

    def _tmetric(t: optuna.trial.FrozenTrial, key: str, default: float) -> float:
        return safe_float(t.user_attrs.get(key, default), default=default)

    best_robust = None
    best_mu = None
    if complete:
        best_robust = max(complete, key=lambda t: _tmetric(t, "awf_robust_score", -1e9))
        best_mu = max(complete, key=lambda t: _tmetric(t, "awf_mu_log", -1e9))

    pass_raw = 0
    for t in complete:
        mu = _tmetric(t, "awf_mu_log", -9.0)
        pos = _tmetric(t, "awf_pos_frac", 0.0)
        if (1.0 - pos) < 0.45 and mu >= 0.0:
            pass_raw += 1

    completed_trials_per_phase = {"phase_a1": 0, "phase_a2": 0, "phase_b": 0}
    for t in complete:
        phase_key = str(t.user_attrs.get("phase", "")).strip().lower()
        if phase_key in completed_trials_per_phase:
            completed_trials_per_phase[phase_key] += 1
    contract = dict(optuna_contract or {})
    trials_per_phase = dict(contract.get("trials_per_phase", {}))
    worker_by_phase = _normalize_phase_workers(dict(contract.get("worker_by_phase", {})))
    sampler_by_phase = dict(contract.get("sampler_by_phase", {}))
    storage_url = str(contract.get("storage_url", ""))
    requested_trials_per_phase = int(contract.get("requested_trials_per_phase", requested_trials))
    planned_total_trials = int(
        contract.get(
            "planned_total_trials",
            sum(int(trials_per_phase.get(k, requested_trials_per_phase)) for k in completed_trials_per_phase),
        )
    )

    return {
        "run_id": run_id,
        "study_name": study_name,
        "requested_trials": int(requested_trials),
        "requested_trials_per_phase": requested_trials_per_phase,
        "planned_total_trials": planned_total_trials,
        "trials_per_phase": {
            "phase_a1": int(trials_per_phase.get("phase_a1", requested_trials_per_phase)),
            "phase_a2": int(trials_per_phase.get("phase_a2", requested_trials_per_phase)),
            "phase_b": int(trials_per_phase.get("phase_b", requested_trials_per_phase)),
        },
        "completed_trials_per_phase": completed_trials_per_phase,
        "sampler_by_phase": {
            "phase_a1": str(sampler_by_phase.get("phase_a1", "")),
            "phase_a2": str(sampler_by_phase.get("phase_a2", "")),
            "phase_b": str(sampler_by_phase.get("phase_b", "")),
        },
        "worker_by_phase": worker_by_phase,
        "storage_url": storage_url,
        "scoped_trials": len(scoped),
        "scoped_complete": len(complete),
        "scoped_pareto": len(pareto_scoped),
        "awf_pass_raw_count": int(pass_raw),
        "best_robust": {
            "trial_number": int(best_robust.number),
            "awf_robust_score": _tmetric(best_robust, "awf_robust_score", -1e9),
            "awf_mu_log": _tmetric(best_robust, "awf_mu_log", -9.0),
            "awf_worst_leg_log_tw": _tmetric(best_robust, "awf_worst_leg_log_tw", -9.0),
            "awf_worst_mdd_pct": _tmetric(best_robust, "awf_worst_mdd_pct", 999.0),
            "awf_pos_frac": _tmetric(best_robust, "awf_pos_frac", 0.0),
        }
        if best_robust is not None
        else None,
        "best_mu": {
            "trial_number": int(best_mu.number),
            "awf_robust_score": _tmetric(best_mu, "awf_robust_score", -1e9),
            "awf_mu_log": _tmetric(best_mu, "awf_mu_log", -9.0),
            "awf_worst_leg_log_tw": _tmetric(best_mu, "awf_worst_leg_log_tw", -9.0),
            "awf_worst_mdd_pct": _tmetric(best_mu, "awf_worst_mdd_pct", 999.0),
            "awf_pos_frac": _tmetric(best_mu, "awf_pos_frac", 0.0),
        }
        if best_mu is not None
        else None,
    }


def build_p7_ops_summary(
    *,
    mode: str,
    ml_integrity_report: dict[str, Any] | None,
    alpha_filter_meta: dict[str, Any] | None,
    alpha_goal_meta: dict[str, Any] | None,
    alpha_cache_meta: dict[str, Any] | None,
    study_user_attrs: dict[str, Any] | None,
    selection_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a structured P7 operations summary for post-run diagnosis."""
    integrity = dict(ml_integrity_report or {})
    alpha_filter = dict(alpha_filter_meta or {})
    alpha_goal = dict(alpha_goal_meta or {})
    alpha_cache = dict(alpha_cache_meta or {})
    study_attrs = dict(study_user_attrs or {})
    selection = dict(selection_summary or {})

    panel = integrity.get("panel", {}) if isinstance(integrity.get("panel"), dict) else {}
    panel_nan_pct = float(panel.get("nan_pct", 0.0) or 0.0)
    panel_prefill_nan_pct = float(integrity.get("panel_pre_fillna_nan_pct", 0.0) or 0.0)
    n_stage_rows = len(integrity.get("stages", []) or [])
    n_feature_group_rows = len(integrity.get("feature_group_coverage", []) or [])

    n_surviving = int(alpha_filter.get("n_surviving", 0) or 0)
    n_components = int(alpha_filter.get("n_components", 0) or 0)
    elite_zero_after_survival = bool(alpha_filter.get("elite_zero_after_survival", False))

    no_candidate_reason = str(study_attrs.get("obs_no_valid_candidates_reason", "") or "")
    failure_reason_counts = (
        dict(study_attrs.get("obs_failure_reason_counts", {}))
        if isinstance(study_attrs.get("obs_failure_reason_counts", {}), dict)
        else {}
    )
    reject_reason_count = (
        dict(selection.get("selection_reject_reason_count", {}))
        if isinstance(selection.get("selection_reject_reason_count", {}), dict)
        else {}
    )

    reason_codes: list[str] = []
    codes = alpha_goal.get("reason_codes", [])
    if isinstance(codes, list):
        for code in codes:
            txt = str(code).strip()
            if txt:
                reason_codes.append(txt)
    if no_candidate_reason:
        reason_codes.append(f"no_candidate:{no_candidate_reason}")
    if elite_zero_after_survival:
        reason_codes.append("elite_zero_after_survival")
    dedup_reason_codes = list(dict.fromkeys(reason_codes))

    cache_state = str(alpha_cache.get("cache_state", "n/a") or "n/a")
    cache_enabled = cache_state not in {"disabled", "n/a"}

    health_status = "pass"
    if (
        no_candidate_reason
        or n_surviving <= 0
        or elite_zero_after_survival
        or "no_elite_components" in dedup_reason_codes
    ):
        health_status = "fail"
    elif panel_nan_pct > 0.05 or panel_prefill_nan_pct > 0.15 or dedup_reason_codes:
        health_status = "warn"

    if not math.isfinite(panel_nan_pct):
        panel_nan_pct = 1.0
    if not math.isfinite(panel_prefill_nan_pct):
        panel_prefill_nan_pct = 1.0

    return {
        "framework": "p7-ops-summary-v1",
        "mode": str(mode),
        "health_status": health_status,
        "integrity": {
            "panel_nan_pct": float(panel_nan_pct),
            "panel_prefill_nan_pct": float(panel_prefill_nan_pct),
            "stage_rows": int(n_stage_rows),
            "feature_group_rows": int(n_feature_group_rows),
        },
        "alpha": {
            "n_surviving": int(n_surviving),
            "n_components": int(n_components),
            "elite_zero_after_survival": bool(elite_zero_after_survival),
            "goal_verdict": str(alpha_goal.get("verdict", "unknown")),
        },
        "optuna_observability": {
            "no_candidate_reason": no_candidate_reason,
            "failure_reason_counts": failure_reason_counts,
            "selection_reject_reason_count": reject_reason_count,
        },
        "alpha_cache": {
            "enabled": bool(cache_enabled),
            "state": cache_state,
            "schema": str(alpha_cache.get("cache_schema", "")),
        },
        "reason_codes": dedup_reason_codes,
    }


def _cleanup_old_runs(out_dir: Path, max_files: int = 50) -> None:
    """Keep only the most recent N summary files to prevent directory bloating."""
    try:
        # Sort by modification time, newest first
        files = sorted(
            out_dir.glob("*.summary.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if len(files) > max_files:
            for f in files[max_files:]:
                try:
                    f.unlink(missing_ok=True)
                except Exception as e:
                    _logger.debug("Failed to delete old summary file %s: %s", f, e)
    except Exception as e:
        _logger.debug("Failed to cleanup old runs: %s", e)


def write_run_summary_snapshot(summary: dict[str, Any], project_root: str | Path) -> Path:
    """Save run summary to JSON, append to index.jsonl, and rotate old files."""
    out_dir = Path(project_root) / "logs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(summary.get("run_id", "unknown"))
    out_path = out_dir / f"{run_id}.summary.json"

    # 1. Write or update the detailed JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 2. Append to the cumulative index (journal)
    with open(out_dir / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # 3. Cleanup old run summaries (keep latest 50)
    _cleanup_old_runs(out_dir, max_files=50)

    return out_path


def run_optimization_loop(
    base_ctx: MLPhaseDContext,
    study_name: str,
    storage_url: str,
    storage: optuna.storages.BaseStorage,
    n_trials: int,
    seed: int,
    resume: bool = False,
    n_workers: int = 6,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    enqueue_params: list[dict[str, Any]] | None = None,
    objective_fn: Callable[[optuna.Trial, MLPhaseDContext], Any] | None = None,
    directions: list[str] | tuple[str, ...] | None = None,
    phase_label: str = "Optimization",
) -> optuna.Study:
    """Orchestrate Step 4: Parallel Optuna optimization loop with Micro-Batching.

    Uses ProcessPoolExecutor (fork) with Global State to eliminate IPC pickling overhead.
    Dispatches tasks in small chunks to balance between DB lock contention and
    dynamic load distribution.
    """
    import gc
    import multiprocessing
    import os
    import threading
    from concurrent.futures import ProcessPoolExecutor, TimeoutError

    import optuna
    from tqdm import tqdm

    t_loop_start = time.perf_counter()
    global _GLOBAL_BASE_CTX, _GLOBAL_OBJECTIVE_FN

    # 1. Prevent CPU Thrashing: Disable nested multi-threading in sub-processes
    for env_var in [
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[env_var] = "1"

    from optuna.pruners import MedianPruner

    _pruner = (
        pruner
        if pruner is not None
        else MedianPruner(
            n_startup_trials=int(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_STARTUP_TRIALS", 40)),
            n_warmup_steps=int(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_WARMUP_STEPS", 2)),
        )
    )

    study_ml = get_or_create_study(
        study_name=study_name,
        storage=storage,
        sampler=(sampler if sampler is not None else ml_phase_d_sampler(seed=seed, n_trials=n_trials)),
        resume=resume,
        pruner=_pruner,
        directions=directions,
    )
    study_ml.set_user_attr("latest_run_id", base_ctx.run_id)
    failure_reason_counts = init_failure_reason_counts()
    if enqueue_params:
        for params in enqueue_params:
            try:
                study_ml.enqueue_trial(dict(params))
            except Exception as e:
                _logger.warning("Failed to enqueue seed params: %s", e)

    # 2. Zero-IPC Setup: Assign global context and freeze memory for fork()
    _GLOBAL_BASE_CTX = base_ctx
    _GLOBAL_OBJECTIVE_FN = objective_fn if objective_fn is not None else objective_ml_phase_d
    gc.collect()
    with contextlib.suppress(AttributeError, RuntimeError):
        gc.freeze()

    # 3. Micro-Batching: Calculate chunk size to balance throughput and latency
    # Typically 2-4 trials per chunk is a good balance for SQLite WAL.
    chunk_cap = int(OPT_FUTURES_CONFIG.get("FUTURES_OPT_CHUNK_SIZE_CAP", 4))
    chunk_cap = max(1, chunk_cap)
    chunk_size = max(1, min(chunk_cap, n_trials // max(1, (n_workers * 2))))
    n_chunks = n_trials // chunk_size
    remainder = n_trials % chunk_size
    chunks = [chunk_size] * n_chunks
    if remainder > 0:
        chunks.append(remainder)

    # 4. Progress Tracking: Background poller for smooth tqdm updates
    stop_event = threading.Event()

    def progress_poller(s_name: str, s_url: str, target: int, r_id: str | None) -> None:
        poller_storage = optuna.storages.RDBStorage(
            s_url, engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}}
        )
        try:
            study = optuna.load_study(study_name=s_name, storage=poller_storage)
            with tqdm(total=target, desc=f"  {phase_label:<32}", unit="trial", leave=True) as pbar:
                while not stop_event.is_set():
                    try:
                        trials = study.get_trials(
                            deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]
                        )
                        if r_id:
                            trials = [t for t in trials if t.user_attrs.get("run_id") == r_id]

                        count = len(trials)
                        ok_count = sum(1 for t in trials if t.state == TrialState.COMPLETE)
                        pruned_count = sum(1 for t in trials if t.state == TrialState.PRUNED)

                        pbar.set_postfix_str(f"OK:{ok_count} | PRUNED:{pruned_count}")
                        pbar.n = min(count, target)
                        pbar.refresh()
                        if count >= target:
                            break
                    except Exception as e:
                        _logger.debug("Progress poller failed to fetch trials: %s", e)
                    time.sleep(5)  # Reduce polling frequency to prevent SQLite DB lock contention
                pbar.n = target
                pbar.refresh()
        except Exception as e:
            _logger.debug("Progress poller terminated: %s", e)

    poller_enabled = bool(OPT_FUTURES_CONFIG.get("FUTURES_OPT_ENABLE_PROGRESS_POLLER", True))
    poller_thread = None
    if poller_enabled:
        poller_thread = threading.Thread(
            target=progress_poller, args=(study_name, storage_url, n_trials, base_ctx.run_id), daemon=True
        )
        poller_thread.start()
    chunk_timeout_sec = int(OPT_FUTURES_CONFIG.get("FUTURES_OPT_CHUNK_TIMEOUT_SEC", 1200))

    # 5. Parallel Execution: Dynamic dispatch of chunked tasks
    mp_ctx = multiprocessing.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            t_submit = time.perf_counter()
            futures = [executor.submit(optimize_worker, study_name, storage_url, c_size) for c_size in chunks]
            _logger.debug(
                "[PROF] opt_loop futures_submit elapsed_s=%.4f",
                time.perf_counter() - t_submit,
            )
            # Wait chunk-by-chunk with timeout to avoid indefinite tail stalls.
            for i, future in enumerate(futures, start=1):
                t_chunk = time.perf_counter()
                try:
                    future.result(timeout=None if chunk_timeout_sec <= 0 else chunk_timeout_sec)
                    _logger.debug(
                        "[PROF] opt_loop chunk %d/%d completed_s=%.4f",
                        i,
                        len(futures),
                        time.perf_counter() - t_chunk,
                    )
                except TimeoutError:
                    _logger.error(
                        "Worker batch timeout (chunk %d/%d, timeout=%ss).", i, len(futures), chunk_timeout_sec
                    )
                    # Cancel queued work and break out; completed trials remain in DB.
                    for f in futures[i:]:
                        f.cancel()
                    break
                except Exception as e:
                    _logger.error("Worker batch failed: %s", e)
    finally:
        try:
            trials = study_ml.get_trials(
                deepcopy=False,
                states=[TrialState.FAIL, TrialState.PRUNED],
            )
            for tr in trials:
                reason = str((tr.user_attrs or {}).get("obs_reason", "unknown"))
                increment_failure_reason_count(failure_reason_counts, reason)
        except Exception as e:
            _logger.debug("Failed to collect failure reason counts: %s", e)
        study_ml.set_user_attr("obs_failure_reason_counts", dict(failure_reason_counts))
        stop_event.set()
        if poller_thread is not None:
            poller_thread.join(timeout=5)
        _GLOBAL_BASE_CTX = None  # Clear global reference
        _GLOBAL_OBJECTIVE_FN = objective_ml_phase_d
        _logger.debug(
            "[PROF] run_optimization_loop total_elapsed_s=%.4f",
            time.perf_counter() - t_loop_start,
        )

    return study_ml
