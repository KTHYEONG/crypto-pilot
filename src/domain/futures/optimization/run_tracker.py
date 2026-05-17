from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.optimization.dashboard import safe_float
from src.domain.futures.optimization.optimizer import (
    MLPhaseDContext,
    objective_ml_phase_d,
)

_logger: logging.Logger = logging.getLogger("run_tracker")


def setup_optuna_storage(project_root: str | Path) -> tuple[str, optuna.storages.RDBStorage]:
    """Set up high-performance Optuna storage with SQLite WAL mode."""
    storage_path = Path(project_root) / "logs" / "optuna_futures.db"
    storage_url = f"sqlite:///{storage_path}"

    # 1. SQLAlchemy Engine with WAL mode & Connection Pooling
    engine = create_engine(
        storage_url,
        connect_args={"check_same_thread": False, "timeout": 60},
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
    )

    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA synchronous=NORMAL;"))
        conn.execute(text("PRAGMA cache_size=-64000;"))  # 64MB Cache

    storage = optuna.storages.RDBStorage(
        storage_url,
        engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}}
    )
    return storage_url, storage


def get_or_create_study(
    study_name: str,
    storage: optuna.storages.RDBStorage,
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


def ml_phase_d_sampler_coordinate(
    seed: int, n_trials: int, phase: str
) -> optuna.samplers.BaseSampler:
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
        mdd_limit = float(cfg.get("FUTURES_MAX_MDD", 25.0))
        dsr_floor = float(
            dsr_min if dsr_min is not None else cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.30)
        )
        if is_mdd >= mdd_limit:
            return False
        if is_dsr < dsr_floor:
            return False
        return True

    # Historical AWF-based checks
    pbo_lim = float(pbo_max if pbo_max is not None else cfg.get("FUTURES_PBO_MAX", 0.40))
    if check_pbo and float(pbo_obs) >= pbo_lim:
        return False
    dsr = float(trial.user_attrs.get("gate1_dsr", -9.0))
    dsr_floor = float(
        dsr_min if dsr_min is not None else cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.80)
    )
    if dsr < dsr_floor:
        return False
    p10_floor = float(cfg.get("FUTURES_AWF_P10_LOG_TW_MIN", -0.10))
    p10_cpcv = float(
        trial.user_attrs.get(
            "awf_worst_leg_log_tw", trial.user_attrs.get("ml_p10_log_growth_cpcv", -999.0)
        )
    )
    if p10_cpcv <= p10_floor:
        return False
    mdd_limit = float(cfg.get("FUTURES_MAX_MDD", 22.0))
    if (
        float(
            trial.user_attrs.get(
                "awf_worst_mdd_pct", trial.user_attrs.get("ml_worst_mdd_cpcv", 999.0)
            )
        )
        >= mdd_limit
    ):
        return False
    # Dynamic trade density: scale minimum trades with IS span to avoid regime-size bias.
    span_bars = int(trial.user_attrs.get("gate1_eff_ref_len", 0))
    bars_per_trade_est = float(cfg.get("FUTURES_BARS_PER_TRADE_EST", 200))
    min_trades_dynamic = (
        max(12.0, float(span_bars) / bars_per_trade_est) if span_bars > 0 else 12.0
    )
    if float(trial.user_attrs.get("avg_trades", 0.0)) < min_trades_dynamic:
        return False
    return True


# Global context for worker processes to leverage Linux Copy-on-Write (CoW)
# This prevents expensive IPC pickling of large context objects.
_GLOBAL_BASE_CTX: MLPhaseDContext | None = None
_GLOBAL_OBJECTIVE_FN: Callable[[optuna.Trial, MLPhaseDContext], Any] = objective_ml_phase_d


def resolve_futures_parallel_policy(symbol_count: int) -> int:
    """Determine the optimal number of workers for parallel optimization."""
    logical_cpus = max(1, os.cpu_count() or 1)
    return max(1, min(8, logical_cpus))


def optimize_worker(s_name: str, s_url: str, chunk_size: int):
    """Worker function for parallel Optuna optimization using global context."""
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

    def _objective_with_timeout(tr: optuna.Trial):
        if trial_timeout_sec <= 0:
            return objective_fn(tr, _GLOBAL_BASE_CTX)

        def _timeout_handler(_signum, _frame):
            raise RuntimeError(f"trial_timeout>{trial_timeout_sec}s")

        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(trial_timeout_sec)
        try:
            return objective_fn(tr, _GLOBAL_BASE_CTX)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)

    study.optimize(
        _objective_with_timeout,
        n_trials=chunk_size,
        n_jobs=1,
        catch=(ValueError, RuntimeError),
    )


def cfg_hash_for_run(cfg: dict[str, Any]) -> str:
    """Generate a stable hash for the relevant futures config keys."""
    relevant = {k: v for k, v in cfg.items() if str(k).startswith("FUTURES_")}
    raw = json.dumps(relevant, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:10]  # nosec: S324


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
    sym_fp = hashlib.md5(sym_raw.encode()).hexdigest()[:10]  # nosec: S324
    cfg_fp = cfg_hash_for_run(cfg)
    return (
        f"futures_joint_tpe_tf{tf}_win{fetch_start_date}_{end_date}_"
        f"n{len(symbols_sorted)}_s{sym_fp}_c{cfg_fp}"
    )


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
    sym_raw = json.dumps(
        sorted(str(s) for s in symbols),
        ensure_ascii=True,
        separators=(",", ":")
    )
    sym_fp = hashlib.md5(sym_raw.encode()).hexdigest()[:8]  # nosec: S324
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

    return {
        "run_id": run_id,
        "study_name": study_name,
        "requested_trials": int(requested_trials),
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
        } if best_robust is not None else None,
        "best_mu": {
            "trial_number": int(best_mu.number),
            "awf_robust_score": _tmetric(best_mu, "awf_robust_score", -1e9),
            "awf_mu_log": _tmetric(best_mu, "awf_mu_log", -9.0),
            "awf_worst_leg_log_tw": _tmetric(best_mu, "awf_worst_leg_log_tw", -9.0),
            "awf_worst_mdd_pct": _tmetric(best_mu, "awf_worst_mdd_pct", 999.0),
            "awf_pos_frac": _tmetric(best_mu, "awf_pos_frac", 0.0),
        } if best_mu is not None else None,
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
                except Exception:
                    pass
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
    storage: optuna.storages.RDBStorage,
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
    import time
    from concurrent.futures import ProcessPoolExecutor, TimeoutError

    import optuna
    from tqdm import tqdm

    global _GLOBAL_BASE_CTX, _GLOBAL_OBJECTIVE_FN

    # 1. Prevent CPU Thrashing: Disable nested multi-threading in sub-processes
    for env_var in [
        "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]:
        os.environ[env_var] = "1"

    from optuna.pruners import MedianPruner
    _pruner = pruner if pruner is not None else MedianPruner(
        n_startup_trials=int(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_STARTUP_TRIALS", 40)),
        n_warmup_steps=int(OPT_FUTURES_CONFIG.get("FUTURES_PRUNER_WARMUP_STEPS", 2)),
    )

    study_ml = get_or_create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler if sampler is not None else ml_phase_d_sampler(seed=seed, n_trials=n_trials),
        resume=resume,
        pruner=_pruner,
        directions=directions,
    )
    study_ml.set_user_attr("latest_run_id", base_ctx.run_id)
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
    try:
        gc.freeze()
    except (AttributeError, RuntimeError):
        pass

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
            with tqdm(total=target, desc=f"  {phase_label}", unit="trial", leave=True) as pbar:
                while not stop_event.is_set():
                    try:
                        trials = study.get_trials(
                            deepcopy=False, 
                            states=[TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]
                        )
                        if r_id:
                            count = sum(1 for t in trials if t.user_attrs.get("run_id") == r_id)
                        else:
                            count = len(trials)
                        
                        pbar.n = min(count, target)
                        pbar.refresh()
                        if count >= target:
                            break
                    except Exception:
                        pass
                    time.sleep(3)  # Relaxed polling interval to reduce DB load
                pbar.n = target
                pbar.refresh()
        except Exception:
            pass

    poller_enabled = bool(OPT_FUTURES_CONFIG.get("FUTURES_OPT_ENABLE_PROGRESS_POLLER", True))
    poller_thread = None
    if poller_enabled:
        poller_thread = threading.Thread(
            target=progress_poller,
            args=(study_name, storage_url, n_trials, base_ctx.run_id),
            daemon=True
        )
        poller_thread.start()
    chunk_timeout_sec = int(OPT_FUTURES_CONFIG.get("FUTURES_OPT_CHUNK_TIMEOUT_SEC", 1200))

    # 5. Parallel Execution: Dynamic dispatch of chunked tasks
    mp_ctx = multiprocessing.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(optimize_worker, study_name, storage_url, c_size)
                for c_size in chunks
            ]
            # Wait chunk-by-chunk with timeout to avoid indefinite tail stalls.
            for i, future in enumerate(futures, start=1):
                try:
                    future.result(timeout=None if chunk_timeout_sec <= 0 else chunk_timeout_sec)
                except TimeoutError:
                    _logger.error(
                        "Worker batch timeout (chunk %d/%d, timeout=%ss).",
                        i, len(futures), chunk_timeout_sec
                    )
                    # Cancel queued work and break out; completed trials remain in DB.
                    for f in futures[i:]:
                        f.cancel()
                    break
                except Exception as e:
                    _logger.error("Worker batch failed: %s", e)
    finally:
        stop_event.set()
        if poller_thread is not None:
            poller_thread.join(timeout=5)
        _GLOBAL_BASE_CTX = None  # Clear global reference
        _GLOBAL_OBJECTIVE_FN = objective_ml_phase_d

    return study_ml
